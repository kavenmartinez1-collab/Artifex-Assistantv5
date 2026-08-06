"""Tests for tools/agent_tools.py — action extraction, tool parsing."""

import os
import tempfile
import pytest

from tools.agent_tools import extract_agent_actions


class TestExtractAgentActions:
    """Test the action extraction regex parser."""

    def test_extract_shell_action(self):
        """Detects ```bash``` code blocks as shell actions."""
        response = 'Let me check:\n```bash\nls -la\n```\n'
        actions = extract_agent_actions(response)
        assert len(actions) >= 1
        assert any(a.type == "shell" for a in actions)

    def test_extract_python_action(self):
        """Detects ```python``` code blocks as python actions."""
        response = 'Here:\n```python\nprint("hello")\n```\n'
        actions = extract_agent_actions(response)
        assert len(actions) >= 1
        assert any(a.type == "python" for a in actions)

    def test_extract_read_file_tool(self):
        """Detects @read_file("path") tool markers."""
        response = 'Let me read it:\n@read_file("config.py")\n'
        actions = extract_agent_actions(response)
        assert len(actions) >= 1
        assert any(a.type == "read_file" for a in actions)

    def test_extract_architecture_tool(self):
        """Detects @architecture() tool marker."""
        response = 'Let me map the project:\n@architecture()\n'
        actions = extract_agent_actions(response)
        assert len(actions) >= 1
        assert any(a.type == "architecture" for a in actions)

    def test_extract_search_tool(self):
        """Detects @search("query") tool marker."""
        response = '@search("python tutorial")\n'
        actions = extract_agent_actions(response)
        assert len(actions) >= 1
        assert any(a.type in ("websearch", "search") for a in actions)

    def test_extract_grep_tool(self):
        """Detects @grep("pattern", "path") tool marker."""
        response = '@grep("def main", ".")\n'
        actions = extract_agent_actions(response)
        assert len(actions) >= 1
        assert any(a.type == "grep" for a in actions)

    def test_extract_glob_tool(self):
        """Detects @glob("pattern") tool marker."""
        response = '@glob("**/*.py")\n'
        actions = extract_agent_actions(response)
        assert len(actions) >= 1
        assert any(a.type == "glob" for a in actions)

    def test_extract_edit_block(self):
        """Detects ```edit``` blocks."""
        response = (
            'Fix:\n```edit\n'
            'FILE: test.py\n'
            'OLD:\nfoo\n'
            'NEW:\nbar\n'
            '```\n'
        )
        actions = extract_agent_actions(response)
        assert len(actions) >= 1
        assert any(a.type == "edit_file" for a in actions)

    def test_no_actions(self):
        """Plain text response has no actions."""
        response = "Sure, Python is a great language for beginners."
        actions = extract_agent_actions(response)
        assert len(actions) == 0

    def test_multiple_actions(self):
        """Multiple actions in one response are all extracted."""
        response = (
            '@read_file("a.py")\n'
            '@read_file("b.py")\n'
            '```bash\necho hello\n```\n'
        )
        actions = extract_agent_actions(response)
        assert len(actions) >= 3

    def test_tool_marker_in_code_block_not_shell(self):
        """Tool markers inside bash blocks should NOT become shell actions."""
        response = '```bash\n@search("Artemis 2 launch")\n```\n'
        actions = extract_agent_actions(response)
        # Should have a search action but NOT a shell action
        assert any(a.type == "search" for a in actions)
        assert not any(a.type == "shell" for a in actions)

    def test_tool_marker_in_code_block_mixed(self):
        """Bash block with both real commands and tool markers: only commands become shell."""
        response = '```bash\nls -la\n@read_file("test.py")\n```\n'
        actions = extract_agent_actions(response)
        shell_actions = [a for a in actions if a.type == "shell"]
        read_actions = [a for a in actions if a.type == "read_file"]
        assert len(shell_actions) == 1
        assert shell_actions[0].content == "ls -la"
        assert len(read_actions) == 1


class TestEditBlockParsing:
    """Regression: multi-line NEW content must survive extraction intact."""

    def test_multiline_new_not_truncated(self):
        # A lazy (.*?)$ with MULTILINE used to stop NEW at its first line,
        # silently truncating every multi-line replacement (agent_bench find).
        response = (
            "```edit\n"
            "FILE: mathx.py\n"
            "OLD:\n"
            "def add(a, b):\n"
            "    return a - b\n"
            "NEW:\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "```"
        )
        actions = extract_agent_actions(response)
        edits = [a for a in actions if a.type == "edit_file"]
        assert len(edits) == 1
        path, old, new = edits[0].content.split("\x00")
        assert path == "mathx.py"
        assert old == "def add(a, b):\n    return a - b"
        assert new == "def add(a, b):\n    return a + b"

    def test_single_line_new_still_works(self):
        response = (
            "```edit\nFILE: x.py\nOLD:\na = 1\nNEW:\na = 2\n```"
        )
        actions = extract_agent_actions(response)
        edits = [a for a in actions if a.type == "edit_file"]
        assert len(edits) == 1
        _, old, new = edits[0].content.split("\x00")
        assert (old, new) == ("a = 1", "a = 2")


class TestMarkerNormalization:
    """Inline-code inertness + native <tool_call> tolerance (agent_bench finds)."""

    def test_backticked_markers_are_inert(self):
        # The system prompt promises backticked markers don't execute —
        # a model listing its tools in a table must fire nothing.
        response = (
            "Here are my tools:\n\n"
            "| Tool | Purpose |\n"
            "|------|---------|\n"
            '| `@read_file("path")` | Read a file |\n'
            '| `@glob("**/*.py")` | Find files |\n'
            '| `@search("query")` | Web search |\n'
            "| `@architecture()` | Project map |\n"
        )
        assert extract_agent_actions(response) == []

    def test_unbackticked_marker_still_fires(self):
        response = 'Reading it now.\n@read_file("config.ini")\n'
        actions = extract_agent_actions(response)
        assert [a.type for a in actions] == ["read_file"]

    def test_hybrid_tool_call_marker(self):
        # Qwen3.x sometimes wraps a marker in its native tag:
        response = '<tool_call>:glob("config.ini")\n'
        actions = extract_agent_actions(response)
        assert [a.type for a in actions] == ["glob"]
        assert actions[0].content == "config.ini"

    def test_json_tool_call(self):
        response = ('<tool_call>{"name": "read_file", '
                    '"arguments": {"path": "app/main.py"}}</tool_call>')
        actions = extract_agent_actions(response)
        assert [a.type for a in actions] == ["read_file"]
        assert actions[0].content == "app/main.py|1"

    def test_json_tool_call_shell(self):
        response = ('<tool_call>{"name": "shell", '
                    '"arguments": {"command": "pytest -q"}}</tool_call>')
        actions = extract_agent_actions(response)
        assert [a.type for a in actions] == ["shell"]
        assert actions[0].content == "pytest -q"

    def test_json_tool_call_unknown_name_ignored(self):
        response = ('<tool_call>{"name": "launch_missiles", '
                    '"arguments": {"target": "moon"}}</tool_call>')
        assert extract_agent_actions(response) == []


class TestEmptyOldEdit:
    def test_empty_old_gets_clear_error(self, tmp_path):
        import os
        target = tmp_path / "f.py"
        target.write_text("x = 1\n", encoding="utf-8")
        from tools.agent_tools import run_edit_file
        ok, out = run_edit_file(f"{target}\x00\x00new text")
        assert not ok
        assert "OLD is empty" in out
        assert target.read_text(encoding="utf-8") == "x = 1\n"

    def test_bold_wrapped_markers_are_inert(self):
        response = (
            "My tools:\n"
            '1. **@architecture()** - project map\n'
            '2. **@read_file("path")** - read a file\n'
            '3. **@search("query")** - web search\n'
        )
        assert extract_agent_actions(response) == []


class TestNestedFencePython:
    """Python blocks writing markdown-with-code-fences must not truncate."""

    def test_python_block_containing_markdown_fences(self):
        response = (
            "```python\n"
            'content = """# Report\n'
            "Usage example:\n"
            "```python\n"
            "parse_header(line)\n"
            "```\n"
            '"""\n'
            'open("report.md", "w").write(content)\n'
            "```\n"
        )
        actions = extract_agent_actions(response)
        py = [a for a in actions if a.type == "python"]
        assert len(py) == 1
        import ast
        ast.parse(py[0].content)  # must be complete, valid code
        assert 'open("report.md", "w")' in py[0].content

    def test_broken_python_still_first_fence(self):
        # No candidate parses -> legacy first-fence behavior, so the
        # model's own syntax error surfaces unchanged.
        response = "```python\ndef broken(:\n```\nprose after\n"
        actions = extract_agent_actions(response)
        py = [a for a in actions if a.type == "python"]
        assert len(py) == 1
        assert py[0].content.strip() == "def broken(:"


class TestShellRedirectEncoding:
    def test_powershell_redirect_writes_utf8(self, tmp_path):
        # PS 5.1 default `>` is UTF-16 — the executor must force UTF-8 so
        # redirected files are readable by python/read_file downstream.
        import sys
        if sys.platform != "win32":
            import pytest
            pytest.skip("Windows-only")
        from tools.agent_tools import run_shell_command
        out_file = tmp_path / "r.txt"
        ok, _ = run_shell_command(f'echo 57 > "{out_file}"', cwd=str(tmp_path))
        assert ok
        raw = out_file.read_bytes()
        assert b"\x00" not in raw, f"UTF-16 leak: {raw[:20]!r}"
        assert "57" in raw.decode("utf-8-sig")

    def test_shell_python_is_venv_python(self, tmp_path):
        import sys
        if sys.platform != "win32":
            import pytest
            pytest.skip("Windows-only")
        from tools.agent_tools import run_shell_command, _PYTHON_BIN
        ok, out = run_shell_command(
            'python -c "import sys; print(sys.executable)"', cwd=str(tmp_path))
        assert ok
        assert out.strip().lower() == _PYTHON_BIN.lower(), out
