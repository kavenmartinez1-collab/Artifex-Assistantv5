"""
Artifex Assistant V5 — Claude CLI engine.

Spawns the `claude` CLI as a subprocess to get inference from an
Anthropic Claude Code subscription.  The OAuth state lives in
%USERPROFILE%\\.claude\\ on the user account that runs Artifex; the
ANTHROPIC_API_KEY env var is stripped from the subprocess env so the
CLI uses the OAuth subscription rather than billing against an API key.

Subprocess invocation is purely list-arg with stdin redirection — no
shell pipe, no f-string interpolation into shell commands.  This is a
deliberate departure from the prior downstream-service implementation, which
had a shell-injection RCE in its long-prompt path (`type "..." | claude
--print` via subprocess_shell).

Concurrency: the api server's model_queue serializes all chat
completions across backends, so this engine doesn't need its own lock.
ToS posture (single-user OAuth subscription) inherits that
serialization automatically.
"""

import logging
import os
import subprocess
import tempfile
import time

from core.engine_base import BaseEngine

_log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = float(os.environ.get("CLAUDE_CLI_TIMEOUT", "1800"))
# Bare "claude" works on every platform — Windows installs claude.exe to
# %USERPROFILE%\.local\bin and Python's subprocess on Windows resolves
# unsuffixed names via PATHEXT.  An earlier draft defaulted to
# "claude.cmd" on Windows, which was wrong (no such file exists in the
# npm-installed layout).
DEFAULT_BIN = "claude"
AUTH_PROBE_CACHE_S = 60.0


def _claude_home() -> str:
    """Path where the claude CLI keeps its OAuth state."""
    home = os.environ.get("USERPROFILE") or os.path.expanduser("~")
    return os.path.join(home, ".claude")


def probe_auth_state(cli_path: str | None = None) -> tuple[bool, str]:
    """Module-level probe used by /v1/models and engine startup.

    Returns (ok, reason).  ok=True only when the CLI runs and the OAuth
    state directory exists.  reason is empty on success or a short
    diagnostic for the false case.
    """
    bin_ = cli_path or DEFAULT_BIN
    try:
        r = subprocess.run(
            [bin_, "--version"], capture_output=True, timeout=10,
        )
    except FileNotFoundError:
        return False, f"`{bin_}` not on PATH"
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"could not run `{bin_} --version`: {e}"
    if r.returncode != 0:
        stderr = (r.stderr or b"").decode("utf-8", errors="replace").strip()
        return False, f"`{bin_} --version` exit {r.returncode}: {stderr[:200]}"
    home = _claude_home()
    if not os.path.isdir(home):
        return False, (
            f"{home} missing — run `claude /login` as the user that "
            f"owns the Artifex process"
        )
    return True, ""


class ClaudeCliEngine(BaseEngine):
    """Engine that shells out to the `claude` CLI for inference.

    One instance per model name; the model name selects the `--model`
    flag on the CLI invocation.  Sentinel `claude-code` means "let the
    CLI pick" (no `--model` arg passed).
    """

    # The CLI emits plain assistant text — no `<think>...</think>`
    # markers — so the stream begins outside any thinking block.
    # The default True (legacy transformers behavior) would otherwise
    # route the entire response into on_thinking and emerge as
    # delta.x_thinking on the SSE wire, which upstream-client's stream parser
    # silently drops.
    stream_starts_in_think = False

    def __init__(self, model_name: str, model_config: dict | None = None):
        self.model_name = model_name
        cfg = model_config or {}
        self._cli_path = cfg.get("cli_path") or DEFAULT_BIN
        self._context_length = int(cfg.get("context_length") or 0)
        self._configured_timeout = cfg.get("timeout")
        self._loaded = False
        # Cached auth probe.  /v1/models hits this every refresh, so
        # caching matters; the actual subprocess only runs every
        # AUTH_PROBE_CACHE_S seconds.
        self._auth_ok = False
        self._auth_reason = "not yet probed"
        self._auth_checked_at = 0.0
        self._last_gen_stats: dict = {}

    # ── Lifecycle ───────────────────────────────────────────────────

    def load(self, status_callback=None):
        if self._loaded:
            return
        if not self._is_auth_fresh():
            self._refresh_auth()
        if not self._auth_ok:
            raise RuntimeError(
                f"claude CLI not available: {self._auth_reason}. "
                f"Install the CLI and run `claude /login` on the host "
                f"running Artifex."
            )
        self._loaded = True
        if status_callback:
            status_callback(f"claude CLI ready — {self.model_name}")

    def unload(self, status_callback=None):
        self._loaded = False
        if status_callback:
            status_callback("claude CLI engine released")

    def is_loaded(self) -> bool:
        # Re-probe lazily so /v1/models reflects auth state after the
        # user runs `claude /login` mid-session without a server
        # restart.
        if self._loaded and not self._is_auth_fresh():
            self._refresh_auth()
            if not self._auth_ok:
                self._loaded = False
        return self._loaded

    def needs_reload(self) -> bool:
        """Always False — subprocess invocation has no persistent state."""
        return False

    def periodic_cleanup(self):
        pass

    def get_context_size(self) -> int:
        return self._context_length

    # ── Auth probe ──────────────────────────────────────────────────

    def _is_auth_fresh(self) -> bool:
        return time.time() - self._auth_checked_at < AUTH_PROBE_CACHE_S

    def _refresh_auth(self):
        ok, reason = probe_auth_state(self._cli_path)
        self._auth_ok = ok
        self._auth_reason = reason
        self._auth_checked_at = time.time()
        if not ok:
            _log.warning("claude CLI auth probe failed: %s", reason)

    # ── Token counting ──────────────────────────────────────────────

    def count_tokens(self, text: str) -> int:
        # The CLI doesn't expose its tokenizer; 4-char heuristic matches
        # the rest of the codebase.
        return len(text) // 4

    # ── Generation ──────────────────────────────────────────────────

    def generate_streaming(self, messages, max_tokens, temperature,
                           on_token=None, on_complete=None,
                           enable_thinking=True,
                           grammar=None, response_format=None,
                           raw_output=False,
                           web_tools=False) -> str:
        """Run inference via the claude CLI.

        The CLI doesn't expose sampling controls (temperature,
        max_tokens), grammar, or response_format, so those args are
        accepted-and-ignored to satisfy the BaseEngine contract.

        Multi-turn history is folded into a single user prompt with
        role markers — `claude --print` is single-turn.  Image parts
        in multimodal content arrays are dropped (vision goes through
        Ollama / transformers in this codebase).

        Streaming: this engine is non-streaming at the CLI level (no
        `--output-format stream-json` mapping yet).  When `on_token`
        is provided, the full response is emitted as a single chunk so
        the streaming-interface caller still gets it.

        web_tools: when True, passes `--allowedTools WebSearch,WebFetch`
        so the CLI uses Anthropic-native tool execution.  Unlike the
        other backends — which get web tools via the @search/@web_read
        post-processor wrapper in api/server.py — claude executes the
        tools internally and returns the final synthesized response.
        """
        self.load()
        self._last_gen_stats = {}

        system_prompt, user_prompt = _messages_to_cli_inputs(messages)

        cmd: list[str] = [self._cli_path, "--print"]
        if self.model_name and self.model_name != "claude-code":
            cmd.extend(["--model", self.model_name])
        if web_tools:
            cmd.extend(["--allowedTools", "WebSearch,WebFetch"])

        sys_tempfile: str | None = None
        try:
            if system_prompt:
                sys_tempfile = _write_tempfile(system_prompt)
                cmd.extend(["--system-prompt-file", sys_tempfile])

            # Force the OAuth subscription path — strip any API key
            # that would otherwise switch the CLI into billed mode.
            env = os.environ.copy()
            env.pop("ANTHROPIC_API_KEY", None)

            timeout = float(self._configured_timeout or DEFAULT_TIMEOUT)
            _log.info(
                "claude CLI: model=%s sys_chars=%d user_chars=%d timeout=%ds",
                self.model_name, len(system_prompt), len(user_prompt),
                int(timeout),
            )
            t0 = time.perf_counter()

            try:
                proc = subprocess.run(
                    cmd,
                    input=user_prompt.encode("utf-8"),
                    capture_output=True,
                    timeout=timeout,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                raise TimeoutError(
                    f"claude CLI did not return within {timeout:.0f}s"
                )

            if proc.returncode != 0:
                stderr = proc.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    f"claude CLI exited with code {proc.returncode}: "
                    f"{stderr[:500]}"
                )

            response = proc.stdout.decode("utf-8", errors="replace").strip()
            elapsed = time.perf_counter() - t0
            self._last_gen_stats = {
                "prompt_tokens": self.count_tokens(
                    (system_prompt or "") + user_prompt
                ),
                "completion_tokens": self.count_tokens(response),
                "finish_reason": "stop",
                "duration_s": elapsed,
            }
            _log.info(
                "claude CLI done: %d chars in %.1fs (model=%s)",
                len(response), elapsed, self.model_name,
            )

            if on_token and response:
                on_token(response)
            if on_complete:
                on_complete(response)
            return response

        finally:
            if sys_tempfile:
                try:
                    os.unlink(sys_tempfile)
                except OSError:
                    pass


def _messages_to_cli_inputs(messages: list[dict]) -> tuple[str, str]:
    """Convert OpenAI-format messages to (system_prompt, user_prompt).

    System messages concat into the system prompt; user/assistant
    messages concat into a single prompt with role markers since
    `claude --print` is single-turn.  Multimodal content arrays are
    flattened to their text parts; image parts are dropped (vision
    goes through other backends).
    """
    sys_parts: list[str] = []
    user_parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_pieces: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_pieces.append(part.get("text", ""))
            content = "\n".join(text_pieces)
        if not isinstance(content, str):
            content = str(content)
        if role == "system":
            sys_parts.append(content)
        elif role == "assistant":
            user_parts.append(f"[Previous assistant response]\n{content}")
        else:
            user_parts.append(content)
    return "\n\n".join(sys_parts), "\n\n".join(user_parts)


def _write_tempfile(content: str) -> str:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8",
    )
    try:
        f.write(content)
    finally:
        f.close()
    return f.name
