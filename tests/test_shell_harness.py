"""
Shell harness fixes from agent run 6024eea3124b (2026-10-05).

That run asked the agent to open a terminal, start `claude` and type `/rc`.
It exposed four harness problems, each covered here:
  1. a shell block was split into one process per line, so PowerShell
     variables and here-strings never survived past their own line;
  2. actions ran grouped by type (shell before edit) instead of in the order
     the model wrote them;
  3. every PowerShell read (`Get-Command claude`) classified HIGH, so Guided
     asked for approval on each one and the risk-budget gate tripped;
  4. there was no "don't ask again" — every command asked anew.
"""

import sys

import pytest

from core.agent_loop import AgentRunner, AutonomyLevel, Decision, RunConfig
from core.sandbox import RiskLevel, clear_policy_hooks
from core.sandbox.approvals import ApprovalMemory
from core.sandbox.human_gate import GateState
from core.sandbox.policy import PolicyDecision, classify_shell_risk, split_shell_segments
from tools.agent_tools import AgentAction, _looks_powershell, extract_agent_actions

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")


@pytest.fixture(autouse=True)
def _clean_hooks():
    clear_policy_hooks()
    yield
    clear_policy_hooks()


# ── 1. a shell block is one script ──────────────────────────────────────────

class TestWholeBlock:
    def test_multiline_powershell_is_one_action(self):
        resp = ("```powershell\n"
                "$wsh = New-Object -ComObject WScript.Shell\n"
                "for ($i = 0; $i -lt 3; $i++) { $wsh.SendKeys([char]175) }\n"
                "```")
        shells = [a for a in extract_agent_actions(resp) if a.type == "shell"]
        assert len(shells) == 1
        assert shells[0].content.count("\n") == 1
        assert "(+1 more lines)" in shells[0].display

    def test_here_string_block_stays_whole(self):
        resp = ('```powershell\nAdd-Type -TypeDefinition @"\n'
                "public class V { }\n"
                '"@\n[V]::new()\n```')
        shells = [a for a in extract_agent_actions(resp) if a.type == "shell"]
        assert len(shells) == 1
        assert '@"' in shells[0].content and '"@' in shells[0].content

    def test_prompt_prefixes_stripped_comments_kept(self):
        resp = "```bash\n$ ls -la\n# just a comment\n```"
        [a] = extract_agent_actions(resp)
        assert a.content == "ls -la\n# just a comment"

    def test_prose_only_fence_is_not_a_command(self):
        assert extract_agent_actions("```console\nOutput: 42\n```") == []

    @windows_only
    def test_variables_survive_across_lines(self, tmp_path):
        from tools.agent_tools import run_shell_command
        ok, out = run_shell_command('$x = 41\n$x += 1\nWrite-Output "x=$x"',
                                    cwd=str(tmp_path))
        assert ok, out
        assert out.strip() == "x=42"

    @windows_only
    def test_echo_first_line_still_runs_as_powershell(self, tmp_path):
        from tools.agent_tools import run_shell_command
        ok, out = run_shell_command('echo start\n$n = 2 * 3\nWrite-Output "n=$n"',
                                    cwd=str(tmp_path))
        assert ok, out
        assert "n=6" in out

    @windows_only
    def test_stdin_is_closed(self, tmp_path):
        # Nothing can type into an agent command; a reader must see EOF at
        # once instead of blocking until the timeout.
        from tools.agent_tools import run_shell_command
        ok, out = run_shell_command(
            'python -c "import sys; print(repr(sys.stdin.read()))"',
            cwd=str(tmp_path), timeout=30)
        assert ok, out
        assert out.strip() == "''"

    def test_dialect_detection(self):
        assert _looks_powershell("echo hi\n$x = 1")
        assert _looks_powershell("Get-ChildItem | Select-Object Name")
        assert _looks_powershell("[Console]::Beep()")
        assert not _looks_powershell("ls -la && cat README.md")
        assert not _looks_powershell("cat > f.py << 'EOF'\nprint(1)\nEOF")


# ── 2. actions run in written order ─────────────────────────────────────────

class TestWrittenOrder:
    def test_edit_then_run(self):
        resp = ("```edit\nFILE: go.ps1\nOLD:\nNEW:\nWrite-Output hi\n```\n"
                "```powershell\n.\\go.ps1\n```")
        assert [a.type for a in extract_agent_actions(resp)] == ["edit_file", "shell"]

    def test_markers_and_blocks_interleave(self):
        resp = ('@read_file("a.txt")\n'
                "```powershell\nGet-ChildItem\n```\n"
                '@glob("*.py")\n'
                "```python\nprint(1)\n```\n"
                '@search("x")')
        assert [a.type for a in extract_agent_actions(resp)] == [
            "read_file", "shell", "glob", "python", "search"]

    def test_inline_code_still_inert_and_offsets_hold(self):
        resp = ('Use `@read_file("doc.md")` later.\n'
                "```powershell\nGet-Date\n```\n"
                '@read_file("real.md")')
        acts = extract_agent_actions(resp)
        assert [(a.type, a.content) for a in acts] == [
            ("shell", "Get-Date"), ("read_file", "real.md|1")]

    def test_native_calls_ordered_with_fences(self):
        resp = ("```powershell\nGet-Date\n```\n"
                "<tool_call>\n<function=glob>\n*.md\n</function>\n</tool_call>")
        assert [a.type for a in extract_agent_actions(resp)] == ["shell", "glob"]


# ── 3. PowerShell reads are SAFE, everything else is not ─────────────────────

class TestPowerShellRisk:
    @pytest.mark.parametrize("cmd", [
        "Get-Command claude",
        "Test-Path .\\scripts",
        "Get-ChildItem -Recurse *.py | Select-Object FullName",
        "Get-Process | Where-Object { $_.CPU -gt 10 } | Sort-Object CPU",
        "gci .\\scripts | ft Name, Length",
        "Get-Content README.md | Select-String TODO",
        "Get-ChildItem -Path (Join-Path $env:USERPROFILE 'Desktop')",
        "ls; pwd",
        "Get-Date\nGet-Location",
        "$files = Get-ChildItem .\\scripts\n$n = 3\nWrite-Output $files.Count $n",
        "$ErrorActionPreference = 'Stop'\nGet-Item x",
    ])
    def test_safe(self, cmd):
        assert classify_shell_risk(cmd) == RiskLevel.SAFE, cmd

    @pytest.mark.parametrize("cmd", [
        "Get-ChildItem | Remove-Item",
        "gci | % Delete",
        "gci | ForEach-Object { $_.Delete() }",
        "gci | ? { rm $_ }",
        "Get-Content (Remove-Item x)",
        "[IO.File]::Delete('x')",
        "(Get-Item x).Delete()",
        "Get-Credential",
        "Get-Process; Stop-Process -Id 1",
        "Get-Date > out.txt",
        "Get-Date\nSet-Content x.txt hi",
        "Start-Process notepad",
        "Get-Help Remove-Item | Out-File help.txt",
        "$x = Remove-Item y",
        "$x = python evil.py",
        "$x = \"$(rm y)\"",
    ])
    def test_not_safe(self, cmd):
        assert classify_shell_risk(cmd) != RiskLevel.SAFE, cmd

    def test_segment_split_is_quote_and_block_aware(self):
        assert split_shell_segments('git commit -m "a; b" && git log') == [
            'git commit -m "a; b"', "git log"]
        assert split_shell_segments("gci | ? { $_.x; $_.y }") == [
            "gci", "? { $_.x; $_.y }"]
        assert split_shell_segments("# note\nls\n\npwd") == ["ls", "pwd"]


# ── 4. remembered approvals ─────────────────────────────────────────────────

def _dec(risk=RiskLevel.HIGH):
    return PolicyDecision(allowed=True, requires_confirmation=True,
                          risk_level=risk, reason="t")


def _sh(cmd):
    return AgentAction("shell", cmd, cmd)


class TestApprovalMemory:
    def test_prefix_covers_same_command_family(self):
        m = ApprovalMemory()
        assert m.remember(_sh("git commit -m first")) == ["commands starting `git commit`"]
        assert m.covers(_sh("git commit -m second"), _dec())
        assert m.covers(_sh("git status; git commit -am x"), _dec())
        assert not m.covers(_sh("git push origin main"), _dec())

    def test_chained_unapproved_command_not_covered(self):
        m = ApprovalMemory()
        m.remember(_sh("Set-Volume 50"))
        assert m.covers(_sh("Set-Volume 20"), _dec())
        assert not m.covers(_sh("Set-Volume 20; Remove-Item x"), _dec())
        assert not m.covers(_sh("Set-Volume (Remove-Item x)"), _dec())
        assert not m.covers(_sh("Set-Volume $(rm x)"), _dec())

    def test_wrappers_and_destructive_heads_are_exact_only(self):
        m = ApprovalMemory()
        for cmd in ("powershell -File a.ps1", "Remove-Item .\\out.txt",
                    "Start-Process notepad"):
            m.remember(_sh(cmd))
            assert m.covers(_sh(cmd), _dec())
        assert not m.covers(_sh("powershell -Command evil"), _dec())
        assert not m.covers(_sh("Remove-Item .\\other.txt"), _dec())
        assert not m.covers(_sh("Start-Process calc"), _dec())

    def test_multiline_script_is_exact_only(self):
        m = ApprovalMemory()
        script = "$w = New-Object -ComObject WScript.Shell\n$w.SendKeys('a')"
        m.remember(_sh(script))
        assert m.covers(_sh(script), _dec())
        assert not m.covers(_sh(script + "\nRemove-Item x"), _dec())

    def test_critical_never_covered(self):
        m = ApprovalMemory()
        cmd = "rm -rf build"
        m.remember(_sh(cmd))
        assert not m.covers(_sh(cmd), _dec(RiskLevel.CRITICAL))

    def test_edit_rule_is_per_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        m = ApprovalMemory()
        m.remember(AgentAction("edit_file", "a.py\x00x\x00y", "edit a.py"))
        assert m.covers(AgentAction("edit_file", "a.py\x00p\x00q", "e"), _dec(RiskLevel.MEDIUM))
        assert not m.covers(AgentAction("edit_file", "b.py\x00p\x00q", "e"), _dec(RiskLevel.MEDIUM))


# ── loop integration ────────────────────────────────────────────────────────

class _Engine:
    def __init__(self, responses):
        self.responses, self.i = list(responses), 0

    def get_context_size(self):
        return 8192

    def generate_streaming(self, messages, max_tokens=0, temperature=0.0, on_token=None):
        r = self.responses[self.i] if self.i < len(self.responses) else '@done("x")'
        self.i += 1
        if on_token:
            on_token(r)
        return r


def _runner(responses, monkeypatch, approval, gate=None):
    ran = []
    monkeypatch.setattr("core.agent_loop.run_agent_action",
                        lambda a, **k: (ran.append(a.content) or True, "ok"))
    cfg = RunConfig.default(AutonomyLevel.GUIDED)
    cfg.max_rounds = 6
    events = []
    runner = AgentRunner(_Engine(responses), build_system_prompt=lambda: "S",
                         emit=events.append, request_approval=approval,
                         config=cfg, gate=gate or GateState(interval=0))
    return runner, events, ran


def test_always_skips_later_prompts(monkeypatch):
    asked = []
    def approval(a, d, r):
        asked.append(a.content if a else r)
        return Decision.APPROVE_ALWAYS
    runner, events, ran = _runner(
        ["```powershell\nSet-Volume 50\n```",
         "```powershell\nSet-Volume 30\n```",
         "```powershell\nSet-Volume 10\n```", '@done("ok")'],
        monkeypatch, approval)
    res = runner.run("vol", [{"role": "system", "content": "x"}])
    assert res.status == "done"
    assert asked == ["Set-Volume 50"]
    assert ran == ["Set-Volume 50", "Set-Volume 30", "Set-Volume 10"]
    kinds = [e.kind for e in events]
    assert kinds.count("approval_remembered") == 1
    assert kinds.count("auto_approved") == 2


def test_powershell_reads_never_prompt_in_guided(monkeypatch):
    asked = []
    runner, _, ran = _runner(
        ["```powershell\nGet-Command claude\n```", '@done("ok")'], monkeypatch,
        lambda a, d, r: asked.append(a) or Decision.APPROVE)
    runner.run("x", [{"role": "system", "content": "x"}])
    assert asked == [] and ran == ["Get-Command claude"]


def test_human_approved_actions_do_not_drain_risk_budget(monkeypatch):
    # Six HIGH commands, each approved by a person: the budget gate (15
    # points, HIGH = 4) used to pause after the fourth.
    gate = GateState(interval=0, max_actions=0, risk_budget=15)
    pauses = []
    def approval(a, d, r):
        if a is None:
            pauses.append(r)
        return Decision.APPROVE
    responses = [f"```powershell\nInvoke-Thing {i}\n```" for i in range(6)]
    runner, _, ran = _runner(responses + ['@done("ok")'], monkeypatch, approval, gate)
    runner.run("x", [{"role": "system", "content": "x"}])
    assert len(ran) == 6
    assert pauses == []
