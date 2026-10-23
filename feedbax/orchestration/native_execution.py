"""Provider-neutral native-executor identity checks."""

from collections.abc import Sequence
from pathlib import Path


def is_native_training_command(command: Sequence[str]) -> bool:
    """Return whether a command invokes Feedbax's native training executor."""
    try:
        index = command.index("execute-training-run-spec")
    except ValueError:
        return False
    return index > 0 and Path(command[index - 1]).name in {"feedbax", "feedbax.exe"}
