"""Utilities for controlling Python warnings behaviour within this project.

Currently exposes a helper to de-duplicate warnings, ensuring that each unique
warning message is shown only once for the lifetime of the interpreter.  The
implementation respects whatever `logging.captureWarnings(True)` has already
configured – it simply wraps the existing `warnings.showwarning` callable.
"""
from __future__ import annotations

import warnings
from typing import Callable, Set, Tuple

__all__ = [
    "enable_warning_dedup",
]


# Type alias for the original signature of warnings.showwarning
_ShowWarningType = Callable[
    [warnings.WarningMessage | str, type[Warning], str, int, object | None, str | None],
    None,
]


def enable_warning_dedup() -> _ShowWarningType:
    """Install a wrapper around :pyfunc:`warnings.showwarning` that ensures each
    *unique* warning is emitted only once.

    The uniqueness key is a tuple ``(str(message), category, filename)``.  The
    function returns the *previous* ``warnings.showwarning`` so that the caller
    can restore it later if desired.
    """
    shown: Set[Tuple[str, type[Warning], str]] = set()

    orig_showwarning: _ShowWarningType = warnings.showwarning  # type: ignore[arg-type]

    def _dedup_showwarning(
        message: warnings.WarningMessage | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file=None,
        line: str | None = None,
    ) -> None:  # pragma: no cover – simple wrapper
        key = (str(message), category, filename)
        if key in shown:
            return
        shown.add(key)
        orig_showwarning(message, category, filename, lineno, file, line)  # type: ignore[arg-type]

    warnings.showwarning = _dedup_showwarning  # type: ignore[assignment]
    return orig_showwarning 