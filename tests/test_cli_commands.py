"""The CLI loop (ui/cli_assistant.run_assistant) driven with scripted input:
no model, no network, no real workspace."""
import builtins
import types

import pytest

import ui.cli_assistant as cli


class _Engine:
    def __init__(self, ctx=0, fail_load=False):
        self.ctx = ctx
        self.fail_load = fail_load
        self.loads = 0
        self.unloads = 0

    def load(self, status_callback=None):
        self.loads += 1
        if self.fail_load:
            raise RuntimeError("server not reachable")

    def unload(self):
        self.unloads += 1

    def is_loaded(self):
        return True

    def periodic_cleanup(self):
        pass

    def get_context_size(self):
        return self.ctx


class _Store:
    count = 0


class _KM:
    session_store = None
    reference_store = _Store()

    def set_workspace(self, path):
        return True

    def bind_workspace_store(self, path):
        pass

    def get_workspace(self):
        return None

    def get_workspace_summary(self, max_tokens=150):
        return "(test workspace)"

    def render_for_prompt(self, token_budget=0):
        return ""


@pytest.fixture
def cli_env(monkeypatch):
    """Everything run_assistant touches outside itself, faked. Returns a
    namespace: run(lines) -> printed text, plus what the fakes recorded."""
    rec = types.SimpleNamespace(engines=[], backends=[], runs=[], services=0,
                                next_engine_fails=False, ctx=0)

    def create_engine():
        e = _Engine(ctx=rec.ctx, fail_load=rec.next_engine_fails and bool(rec.engines))
        rec.engines.append(e)
        return e

    class _Runner:
        def __init__(self, engine, **kw):
            self.config = kw.get("config")

        def run(self, goal, history):
            rec.runs.append((goal, self.config, len(history)))
            history.append({"role": "user", "content": goal})
            history.append({"role": "assistant", "content": "ok"})
            return types.SimpleNamespace(status="done", rounds=1, actions_run=0,
                                         summary="ok")

    def get_service():
        rec.services += 1
        raise AssertionError("a pipeline must not run in this test")

    import core.proc_tree
    import core.services
    monkeypatch.setattr(core.proc_tree, "tie_children_to_this_process", lambda: None)
    monkeypatch.setattr(core.services, "get_service", get_service)
    monkeypatch.setattr(cli, "install_all_hooks", lambda: None)
    monkeypatch.setattr(cli, "KnowledgeManager", _KM)
    monkeypatch.setattr(cli, "harness", types.SimpleNamespace(
        detect=lambda path: types.SimpleNamespace(is_empty=True)))
    monkeypatch.setattr(cli, "create_engine", create_engine)
    monkeypatch.setattr(cli, "AgentRunner", _Runner)
    monkeypatch.setattr(cli, "set_active_backend", lambda name: rec.backends.append(name))
    monkeypatch.setattr(cli, "get_active_model_name", lambda: "test-model")
    monkeypatch.setattr(cli, "get_assistant_tools_prompt", lambda: "tools")
    monkeypatch.setattr(cli, "auto_save", lambda *a, **k: None)

    def run(lines, capsys):
        script = iter([""] + list(lines))      # first prompt: the workspace

        def fake_input(prompt=""):
            try:
                return next(script)
            except StopIteration:
                raise EOFError

        monkeypatch.setattr(builtins, "input", fake_input)
        cli.run_assistant()
        return capsys.readouterr().out

    rec.run = run
    return rec


def test_backend_llama_cpp_is_accepted(cli_env, capsys):
    # advertised in the banner and the usage text, but used to be rejected
    out = cli_env.run(["/backend llama_cpp", "exit"], capsys)
    assert cli_env.backends == ["llama_cpp"]
    assert "Backend switched to: llama_cpp" in out
    assert "Unknown backend" not in out


def test_backend_unknown_is_still_rejected(cli_env, capsys):
    out = cli_env.run(["/backend nope", "exit"], capsys)
    assert cli_env.backends == []
    assert "Unknown backend: nope" in out


def test_help_lists_every_implemented_command(cli_env, capsys):
    out = cli_env.run(["/help", "exit"], capsys)
    # once in the banner, once for /help
    assert out.count("/models add <#>") == 2
    for cmd in ("/context", "/cleanup", "/index", "/open", "/kb search",
                "/backend transformers|ollama|llama_cpp", "/help"):
        assert cmd in out, cmd
    assert cli_env.runs == []          # /help is not sent to the model


def test_a_failing_command_does_not_end_the_session(cli_env, capsys):
    cli_env.next_engine_fails = True   # the engine /backend creates won't load
    out = cli_env.run(["/backend ollama", "/help", "exit"], capsys)
    assert "RuntimeError: server not reachable" in out
    assert "the session is still open" in out
    # the loop went on: /help after the failure printed the list again
    assert out.count("/models add <#>") == 2


def test_kb_is_a_command_in_pipeline_modes_too(cli_env, capsys):
    # it used to fall into the pipeline block and become an image prompt
    out = cli_env.run(["/mode image_gen", "/kb", "exit"], capsys)
    assert "Knowledge Base:" in out
    assert cli_env.services == 0
    assert "Running image_gen" not in out


def test_run_uses_medium_reasoning_effort_like_the_api(cli_env, capsys):
    cli_env.run(["/run fix the bug", "exit"], capsys)
    (goal, config, _), = cli_env.runs
    assert goal == "fix the bug"
    assert config.reasoning_effort == "medium"


def test_history_is_left_to_the_runner_when_ctx_is_known():
    # AgentRunner compacts by tokens; folding by message count after every
    # turn threw context away and changed the prompt prefix each turn.
    history = [{"role": "system", "content": "S"}] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
        for i in range(40)]
    assert cli._settle_history(history, _Engine(ctx=32768), 6) is history
    assert len(history) == 41


def test_history_still_folds_when_ctx_is_unknown():
    history = [{"role": "system", "content": "S"}] + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
        for i in range(40)]

    class _NoCtx:
        def get_context_size(self):
            raise RuntimeError("not loaded")

    for engine in (_Engine(ctx=0), _NoCtx()):
        out = cli._settle_history(list(history), engine, 6)
        assert len(out) < len(history)
        assert out[0] == history[0]


def test_turns_do_not_fold_history_on_a_known_context(cli_env, capsys):
    cli_env.ctx = 32768
    cli_env.run([f"question {i}" for i in range(8)] + ["exit"], capsys)
    # history length seen by each turn grows by 2 per turn, never shrinks
    seen = [n for _, _, n in cli_env.runs]
    assert seen == [1 + 2 * i for i in range(8)]
