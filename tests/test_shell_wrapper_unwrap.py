"""
A model that wraps its script in `powershell -Command "..."` (agent bench
2026-11-11, Qwen3.8-Flash-Next on native tools: about half of its run_shell
calls). The harness already runs shell text in PowerShell, so the wrapper was
parsed twice: $variables were expanded to nothing by the outer shell and `\\"`
split the argument. The wrapper is now removed and the script runs once.
"""
import sys

import pytest

from tools.agent_tools import run_shell_command, unwrap_powershell_wrapper

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows-only")


class TestUnwrap:
    def test_double_quoted_script(self):
        cmd = ('powershell -NoProfile -Command "Get-ChildItem -Recurse | '
               "ForEach-Object { '{0} [{1}]' -f $_.FullName, $_.Length }\"")
        script, wrapped = unwrap_powershell_wrapper(cmd)
        assert wrapped
        assert script == "Get-ChildItem -Recurse | ForEach-Object { '{0} [{1}]' -f $_.FullName, $_.Length }"

    def test_escaped_inner_quotes(self):
        cmd = ("powershell -NoProfile -Command \"Set-Content -Path 'app.js' "
               "-Value 'console.log(\\\"ready\\\");' -NoNewline\"")
        script, wrapped = unwrap_powershell_wrapper(cmd)
        assert wrapped
        assert script == "Set-Content -Path 'app.js' -Value 'console.log(\"ready\");' -NoNewline"

    def test_multiline_and_more_flags(self):
        cmd = ('pwsh.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "\n'
               "$root='My Documents'\nRemove-Item -LiteralPath \\\"$root\Old Stuff\\\" -Recurse\n\"")
        script, wrapped = unwrap_powershell_wrapper(cmd)
        assert wrapped
        assert script == "$root='My Documents'\nRemove-Item -LiteralPath \"$root\Old Stuff\" -Recurse"

    def test_single_quoted_script(self):
        script, wrapped = unwrap_powershell_wrapper("powershell -c 'Write-Output ''hi'' '")
        assert wrapped and script == "Write-Output 'hi' "

    @pytest.mark.parametrize("cmd", [
        "Get-ChildItem | Select-Object Name",                       # no wrapper
        'powershell -NoProfile -Command "Get-Date"; Get-Location',  # something after it
        'powershell -NoProfile -Command "Write-Host "a b""',        # unescaped inner quote
        "powershell -NoProfile -Command Get-Date",                  # not quoted
        "powershell -NoProfile -File run.ps1",                      # a file, not a script
        'powershell -EncodedCommand "AAAA"',
        'python -c "print(1)"',
        'powershell -NoProfile -Command ""',
    ])
    def test_left_alone(self, cmd):
        assert unwrap_powershell_wrapper(cmd) == (cmd, False)


@windows_only
class TestRunsOnce:
    def test_variables_survive(self, tmp_path):
        (tmp_path / "a.txt").write_text("x")
        ok, out = run_shell_command(
            'powershell -NoProfile -Command "$n = (Get-ChildItem -File).Count; '
            "Get-ChildItem -File | ForEach-Object { '{0}:{1}' -f $_.Name, $n }\"", cwd=str(tmp_path))
        assert ok, out
        assert out.strip() == "a.txt:1"

    def test_escaped_quotes_reach_the_file(self, tmp_path):
        ok, out = run_shell_command(
            "powershell -NoProfile -Command \"Set-Content -Path 'app.js' "
            "-Value 'console.log(\\\"ready\\\");' -Encoding ascii -NoNewline\"", cwd=str(tmp_path))
        assert ok, out
        assert (tmp_path / "app.js").read_text() == 'console.log("ready");'

    def test_wrapped_script_with_a_unix_word_stays_in_powershell(self, tmp_path):
        # `| sort` alone would have sent it to Git Bash
        (tmp_path / "b.txt").write_text("x")
        (tmp_path / "a.txt").write_text("x")
        ok, out = run_shell_command('powershell -NoProfile -Command "(dir -Name) | sort"', cwd=str(tmp_path))
        assert ok, out
        assert out.split() == ["a.txt", "b.txt"]
