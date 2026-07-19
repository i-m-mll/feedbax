"""Native executor context injection shared by orchestration drivers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from feedbax.orchestration.bundle import RunRowSpec
from feedbax.training.diagnostics import NativeExecutionProducerContext


NATIVE_TRAINING_COLLECTION_OUTPUTS = (
    "manifest.json",
    "training-diagnostics.json",
    "checkpoints",
)


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


def uses_registered_native_execution(row: RunRowSpec) -> bool:
    """Return whether orchestration owns native payload and output routing for a row."""

    return (
        row.launch.payload_routing.get("kind") == "registered-execution-payload"
        and is_native_training_command(row.launch.command)
    )


def missing_native_training_collection_outputs(row: RunRowSpec) -> list[str]:
    """Return required row-local native outputs absent from the collection contract."""

    if not uses_registered_native_execution(row):
        return []
    declared = set(row.launch.collect)
    return sorted(set(NATIVE_TRAINING_COLLECTION_OUTPUTS) - declared)


def bind_native_execution_command(
    command: Sequence[str],
    *,
    row: RunRowSpec,
    payload_path: Path | str,
    collection_root: Path | str,
) -> tuple[list[str], RunRowSpec]:
    """Bind one registered native row to its staged input and output roots."""

    normalized = [str(part) for part in command]
    if row.launch.payload_routing.get("kind") != "registered-execution-payload":
        return normalized, row
    if not is_native_training_command(normalized):
        return normalized, row
    command_index = normalized.index("execute-training-run-spec")
    if command_index + 1 < len(normalized) and not normalized[command_index + 1].startswith("-"):
        raise NativeExecutionContextError(
            "registered execution payload routing owns the native spec argument"
        )

    output_options = ("--manifest-root", "--checkpoint-root", "--run-id")
    conflicting = sorted(
        part
        for part in normalized
        if any(part == option or part.startswith(f"{option}=") for option in output_options)
    )
    if conflicting:
        raise NativeExecutionContextError(
            "registered native row output bindings are orchestration-owned; remove "
            f"caller-supplied options {conflicting!r}"
        )

    staged_payload = str(payload_path)
    row_root = Path(collection_root)
    normalized.insert(command_index + 1, staged_payload)
    provenance = row.execution.row_provenance
    if provenance is None:
        raise NativeExecutionContextError(
            f"native execution row {row.row_id!r} lacks TrainingRowProvenance"
        )
    normalized.extend(
        [
            "--manifest-root",
            str(row_root / "manifests"),
            "--checkpoint-root",
            str(row_root / "checkpoints"),
            "--run-id",
            provenance.planned_run_id,
        ]
    )
    bound_row = row.model_copy(
        update={
            "execution": row.execution.model_copy(
                update={"payload": row.execution.payload.model_copy(update={"uri": staged_payload})}
            )
        }
    )
    return normalized, bound_row


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
