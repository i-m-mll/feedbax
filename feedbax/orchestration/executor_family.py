"""Typed built-in executor-family applicability for the common lifecycle."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from feedbax.contracts.evaluation_lifecycle import EvaluationLifecycleEvidence
from feedbax.orchestration.bundle import ExecutionFamily, RunBundle, RunRowSpec
from feedbax.orchestration.drivers.native_execution import (
    bind_native_execution_command,
    inject_native_execution_context,
    missing_native_training_collection_outputs,
)


EVALUATION_COLLECTION_OUTPUTS = (
    "evaluation-matrix-result.json",
    "evaluation",
)
_EVALUATION_TRAINING_ONLY_CHECKS = (
    "checkpoint_cadence",
    "completed_batches",
    "environment_fingerprint",
    "execution_identity",
    "lr_trace",
    "manifest_valid",
    "seeds",
)


class ExecutorFamilyError(ValueError):
    """Raised when a row is incompatible with its declared built-in executor."""


class ExecutorFamilyAdapter(Protocol):
    """Closed typed applicability seam; this is deliberately not a registry."""

    family: ExecutionFamily

    def bind_command(
        self,
        command: Sequence[str],
        *,
        bundle: RunBundle,
        row: RunRowSpec,
        payload_path: Path | str,
        collection_root: Path | str,
        inputs_root: Path | str,
        environment_fingerprint: str,
        update_budget: int | None = None,
    ) -> tuple[list[str], RunRowSpec]: ...

    def missing_collection_outputs(
        self, row: RunRowSpec, collected: Mapping[str, str]
    ) -> list[str]: ...

    def declared_conformance_inapplicable(self) -> Mapping[str, str]: ...


@dataclass(frozen=True, slots=True)
class NativeTrainingExecutorAdapter:
    """Existing native-training behavior behind the typed family seam."""

    family: ExecutionFamily = "native-training"

    def bind_command(
        self,
        command: Sequence[str],
        *,
        bundle: RunBundle,
        row: RunRowSpec,
        payload_path: Path | str,
        collection_root: Path | str,
        inputs_root: Path | str,
        environment_fingerprint: str,
        update_budget: int | None = None,
    ) -> tuple[list[str], RunRowSpec]:
        del bundle, inputs_root
        bound, bound_row = bind_native_execution_command(
            command,
            row=row,
            payload_path=payload_path,
            collection_root=collection_root,
            update_budget=update_budget,
        )
        return (
            inject_native_execution_context(
                bound,
                row=bound_row,
                environment_fingerprint=environment_fingerprint,
                collection_root=collection_root,
            ),
            bound_row,
        )

    def missing_collection_outputs(
        self, row: RunRowSpec, collected: Mapping[str, str]
    ) -> list[str]:
        declared_missing = {
            Path(source).name for source in row.launch.collect
        } - set(collected)
        return sorted(
            declared_missing | set(missing_native_training_collection_outputs(row))
        )

    def declared_conformance_inapplicable(self) -> Mapping[str, str]:
        return {}


@dataclass(frozen=True, slots=True)
class EvaluationMatrixExecutorAdapter:
    """Bind one whole matrix to the public matrix harness and its staged roots."""

    family: ExecutionFamily = "evaluation-matrix"

    def bind_command(
        self,
        command: Sequence[str],
        *,
        bundle: RunBundle,
        row: RunRowSpec,
        payload_path: Path | str,
        collection_root: Path | str,
        inputs_root: Path | str,
        environment_fingerprint: str,
        update_budget: int | None = None,
    ) -> tuple[list[str], RunRowSpec]:
        del environment_fingerprint
        if update_budget is not None:
            raise ExecutorFamilyError(
                "evaluation-matrix executor does not accept a native update budget"
            )
        normalized = [str(part) for part in command]
        try:
            index = normalized.index("matrix-harness")
        except ValueError as exc:
            raise ExecutorFamilyError(
                "evaluation-matrix rows must invoke the public `feedbax matrix-harness`"
            ) from exc
        if row.launch.payload_routing.get("kind") != "registered-execution-payload":
            raise ExecutorFamilyError(
                "evaluation-matrix rows require registered execution payload routing"
            )
        if index + 1 < len(normalized) and not normalized[index + 1].startswith("-"):
            raise ExecutorFamilyError(
                "registered evaluation payload routing owns the matrix spec argument"
            )
        owned = (
            "--manifest-root",
            "--orchestration-bundle",
            "--orchestration-inputs-root",
            "--orchestration-row-id",
        )
        conflicts = sorted(
            part
            for part in normalized
            if any(part == option or part.startswith(f"{option}=") for option in owned)
        )
        if conflicts:
            raise ExecutorFamilyError(
                "evaluation executor output and staged-root bindings are orchestration-owned; "
                f"remove caller-supplied options {conflicts!r}"
            )
        normalized.insert(index + 1, str(payload_path))
        normalized.extend(
            (
                "--manifest-root",
                str(Path(collection_root) / "evaluation"),
                "--orchestration-bundle",
                str(Path(inputs_root) / "run-bundle.json"),
                "--orchestration-inputs-root",
                str(inputs_root),
                "--orchestration-row-id",
                row.row_id,
                "--lifecycle-result",
                str(Path(collection_root) / "evaluation-matrix-result.json"),
                "--batch",
            )
        )
        bound_row = row.model_copy(
            update={
                "execution": row.execution.model_copy(
                    update={
                        "payload": row.execution.payload.model_copy(
                            update={"uri": str(payload_path)}
                        )
                    }
                )
            }
        )
        return normalized, bound_row

    def missing_collection_outputs(
        self, row: RunRowSpec, collected: Mapping[str, str]
    ) -> list[str]:
        del row
        missing = sorted(set(EVALUATION_COLLECTION_OUTPUTS) - set(collected))
        if missing:
            return missing
        evidence = EvaluationLifecycleEvidence.model_validate_json(
            Path(collected["evaluation-matrix-result.json"]).read_text(encoding="utf-8")
        )
        if evidence.executor_family != self.family:
            raise ExecutorFamilyError("collected evaluation lifecycle family drifted")
        return []

    def declared_conformance_inapplicable(self) -> Mapping[str, str]:
        return {
            check_id: "evaluation matrices do not emit native-training evidence"
            for check_id in _EVALUATION_TRAINING_ONLY_CHECKS
        }


_NATIVE_TRAINING_ADAPTER = NativeTrainingExecutorAdapter()
_EVALUATION_MATRIX_ADAPTER = EvaluationMatrixExecutorAdapter()


def executor_family_adapter(family: ExecutionFamily) -> ExecutorFamilyAdapter:
    """Return one of the two built-in adapters without adding discovery machinery."""
    if family == "native-training":
        return _NATIVE_TRAINING_ADAPTER
    if family == "evaluation-matrix":
        return _EVALUATION_MATRIX_ADAPTER
    raise ExecutorFamilyError(f"unsupported execution family: {family!r}")


def evaluation_lifecycle_payload(path: Path | str) -> dict[str, object]:
    """Load and normalize collected evaluation terminal evidence."""
    evidence = EvaluationLifecycleEvidence.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )
    return json.loads(evidence.model_dump_json())


__all__ = [
    "EVALUATION_COLLECTION_OUTPUTS",
    "EvaluationMatrixExecutorAdapter",
    "ExecutorFamilyAdapter",
    "ExecutorFamilyError",
    "NativeTrainingExecutorAdapter",
    "evaluation_lifecycle_payload",
    "executor_family_adapter",
]
