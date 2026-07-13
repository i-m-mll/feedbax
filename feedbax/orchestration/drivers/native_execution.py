"""Native executor context injection shared by orchestration drivers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from feedbax.orchestration.bundle import RunRowSpec
from feedbax.training.diagnostics import NativeExecutionProducerContext


class NativeExecutionContextError(ValueError):
    """Raised when a native row cannot receive one canonical producer context."""


def is_native_training_command(command: Sequence[str]) -> bool:
    """Return whether a command invokes Feedbax's native training executor."""

    try:
        index = command.index("execute-training-run-spec")
    except ValueError:
        return False
    if index == 0:
        return False
    launcher = Path(command[index - 1]).name
    return launcher in {"feedbax", "feedbax.exe"}


def inject_native_execution_context(
    command: Sequence[str],
    *,
    row: RunRowSpec,
    environment_fingerprint: str,
    collection_root: Path | str,
) -> list[str]:
    """Append one canonical inline producer context to a native row command.

    Non-native commands are returned unchanged. Native commands must use the
    row's assembly envelope and authored-to-execution provenance; pre-supplied
    context options are rejected so orchestration never launches with a caller
    wrapper that can drift from the canonical row.
    """

    normalized = [str(part) for part in command]
    if not is_native_training_command(normalized):
        return normalized
    if row.execution.row_provenance is None:
        raise NativeExecutionContextError(
            f"native execution row {row.row_id!r} lacks TrainingRowProvenance"
        )
    if not environment_fingerprint:
        raise NativeExecutionContextError(
            f"native execution row {row.row_id!r} lacks a realized environment fingerprint"
        )
    context_options = ("--execution-context", "--execution-context-json")
    conflicting = sorted(
        part
        for part in normalized
        if any(part == option or part.startswith(f"{option}=") for option in context_options)
    )
    if conflicting:
        raise NativeExecutionContextError(
            "native execution context is orchestration-owned; remove caller-supplied "
            f"options {conflicting!r}"
        )
    context = NativeExecutionProducerContext(
        execution=row.execution,
        environment_fingerprint=environment_fingerprint,
        collection_root=str(collection_root),
    )
    payload = json.dumps(
        context.model_dump(mode="json", exclude_none=True),
        sort_keys=True,
        separators=(",", ":"),
    )
    return [*normalized, "--execution-context-json", payload]
