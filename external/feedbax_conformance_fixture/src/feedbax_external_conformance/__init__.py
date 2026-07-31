"""External clean-wheel conformance fixture for Feedbax."""

from .result import (
    RESULT_SCHEMA_ID,
    RESULT_SCHEMA_VERSION,
    ConformanceResult,
    load_result,
)
from pathlib import Path


def run_fixture(*, source_root: Path | None = None) -> ConformanceResult:
    """Load the execution stack only when the installed fixture is run."""
    from .network import network_denied

    with network_denied():
        from .runner import run_fixture as _run_fixture

        return _run_fixture(source_root=source_root)


__all__ = [
    "RESULT_SCHEMA_ID",
    "RESULT_SCHEMA_VERSION",
    "ConformanceResult",
    "load_result",
    "run_fixture",
]
