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
