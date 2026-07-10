"""Signal-safe, user-directed interruption decisions for training runners."""

from __future__ import annotations

import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TextIO


CancellationAction = Literal["continue", "stop", "terminate"]


@dataclass(frozen=True)
class CancellationDecision:
    """One operator decision made after an interruption request."""

    action: CancellationAction
    source: Literal["interactive", "non_interactive", "second_interrupt", "test"]
    requested_at_unix_seconds: float

    def as_provenance(self) -> dict[str, object]:
        """Return durable, JSON-compatible cancellation provenance."""
        return {
            "action": self.action,
            "source": self.source,
            "requested_at_unix_seconds": self.requested_at_unix_seconds,
        }


class RunInterruptionController:
    """Convert SIGINT into a deterministic run-control decision.

    The signal handler only records that an interrupt arrived. Prompting happens
    later at a safe progress or monitor boundary. Non-interactive execution
    never reads stdin and always chooses a stop at the next durable checkpoint.
    """

    def __init__(
        self,
        *,
        interactive: bool | None = None,
        read_choice: Callable[[], str] | None = None,
        output: TextIO | None = None,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.interactive = sys.stdin.isatty() if interactive is None else interactive
        self._read_choice = read_choice or (lambda: input().strip())
        self._output = output or sys.stderr
        self._now = now
        self._interrupt_count = 0
        self._requested_at: float | None = None
        self._delivered = False
        self._previous_handler: signal.Handlers | None = None

    def __enter__(self) -> "RunInterruptionController":
        self._previous_handler = signal.signal(signal.SIGINT, self._handle_sigint)
        return self

    def __exit__(self, *_exc_info: object) -> None:
        if self._previous_handler is not None:
            signal.signal(signal.SIGINT, self._previous_handler)
            self._previous_handler = None

    def request_interrupt(self) -> None:
        """Record an interrupt request; exposed for focused tests."""
        self._interrupt_count += 1
        self._requested_at = self._now()
        self._delivered = False

    def poll(self) -> CancellationDecision | None:
        """Return one pending decision at a safe execution boundary."""
        if self._interrupt_count == 0 or self._delivered:
            return None
        self._delivered = True
        requested_at = self._requested_at if self._requested_at is not None else self._now()
        if self._interrupt_count >= 2:
            self._write("Second interruption received; terminating now.\n")
            return CancellationDecision("terminate", "second_interrupt", requested_at)
        if not self.interactive:
            self._write("Interrupted; stopping at the next durable checkpoint.\n")
            return CancellationDecision("stop", "non_interactive", requested_at)

        while True:
            self._write(
                "Interrupted. Choose [c]ontinue, [s]top at next durable checkpoint, "
                "or [t]erminate: "
            )
            choice = self._read_choice().strip().lower()
            actions = {"c": "continue", "s": "stop", "t": "terminate"}
            if choice in actions:
                action = actions[choice]
                return CancellationDecision(action, "interactive", requested_at)
            self._write("Please enter c, s, or t.\n")

    def _handle_sigint(self, _signum: int, _frame: object) -> None:
        self.request_interrupt()

    def _write(self, message: str) -> None:
        self._output.write(message)
        self._output.flush()
