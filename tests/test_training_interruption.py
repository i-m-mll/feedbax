from __future__ import annotations

from io import StringIO
from pathlib import Path

from feedbax.training.interruption import RunInterruptionController


def test_checkpoint_stop_can_arrive_before_the_executor_starts(tmp_path: Path) -> None:
    request = tmp_path / "checkpoint-stop.json"
    request.write_text('{"action":"stop"}\n')
    controller = RunInterruptionController(stop_file=request)

    decision = controller.poll()

    assert decision is not None
    assert decision.action == "stop"
    assert decision.requested_at_unix_seconds == request.stat().st_mtime
    assert controller.poll() is None


def test_non_interactive_interrupt_stops_without_reading_stdin() -> None:
    output = StringIO()
    controller = RunInterruptionController(
        interactive=False,
        read_choice=lambda: (_ for _ in ()).throw(AssertionError("must not read stdin")),
        output=output,
        now=lambda: 12.5,
    )

    controller.request_interrupt()

    decision = controller.poll()
    assert decision is not None
    assert decision.action == "stop"
    assert decision.source == "non_interactive"
    assert decision.requested_at_unix_seconds == 12.5
    assert controller.poll() is None
    assert output.getvalue() == "Interrupted; stopping at the next durable checkpoint.\n"


def test_interactive_interrupt_reprompts_then_uses_selected_action() -> None:
    output = StringIO()
    choices = iter(["unexpected", "t"])
    controller = RunInterruptionController(
        interactive=True,
        read_choice=lambda: next(choices),
        output=output,
        now=lambda: 5.0,
    )

    controller.request_interrupt()

    decision = controller.poll()
    assert decision is not None
    assert decision.action == "terminate"
    assert decision.source == "interactive"
    assert "Please enter c, s, or t." in output.getvalue()
