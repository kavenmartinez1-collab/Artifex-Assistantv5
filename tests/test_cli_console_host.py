"""The CLI's event sink must show every reply exactly once."""
from core.agent_loop import AgentEvent
from ui.cli_assistant import _ConsoleHost


def _run(events, capsys):
    host = _ConsoleHost()
    for ev in events:
        host.emit(ev)
    return capsys.readouterr().out


def test_answer_given_through_finish_tool_is_shown(capsys):
    # native finish call, no streamed text: only the done summary carries it
    out = _run([AgentEvent("round_start", round=1),
                AgentEvent("done", summary="The last line is FizzBuzz.", round=1)], capsys)
    assert "The last line is FizzBuzz." in out


def test_streamed_answer_is_not_printed_twice(capsys):
    out = _run([AgentEvent("round_start", round=1),
                AgentEvent("assistant_chunk", text="Hi there.", round=1),
                AgentEvent("assistant_message", text="Hi there.", round=1),
                AgentEvent("done", summary="Hi there.", round=1)], capsys)
    assert out.count("Hi there.") == 1


def test_unstreamed_message_is_printed_once(capsys):
    out = _run([AgentEvent("round_start", round=1),
                AgentEvent("assistant_message", text="Done.", round=1),
                AgentEvent("done", summary="Done.", round=1)], capsys)
    assert out.count("Done.") == 1
