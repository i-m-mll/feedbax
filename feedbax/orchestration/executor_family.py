"""Typed built-in executor-family applicability for the common lifecycle."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from feedbax.contracts.strict_json import strict_json_loads, strict_model_validate_json

from feedbax.contracts.evaluation_lifecycle import (
    EvaluationBatchCompactionEvidence,
    EVALUATION_COLLECTION_OUTPUTS,
    EvaluationLifecycleEvidence,
    EvaluationMatrixBatchPlan,
    EvaluationMatrixOrderedUnionEvidence,
    EvaluationWorkerTopologyEvidence,
)
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.contracts.manifest import canonical_json_bytes, sha256_bytes
from feedbax.contracts.migrations import migrate_structured_spec_payload
from feedbax.orchestration.bundle import ExecutionFamily, RunBundle, RunRowSpec
from feedbax.orchestration.drivers.native_execution import (
    bind_native_execution_command,
    inject_native_execution_context,
    missing_native_training_collection_outputs,
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
        repo_root: Path | str | None,
        environment_fingerprint: str,
        update_budget: int | None = None,
        native_context_injector: Callable[..., list[str]] | None = None,
    ) -> tuple[list[str], RunRowSpec]: ...

    def missing_collection_outputs(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        collected: Mapping[str, str],
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
        repo_root: Path | str | None,
        environment_fingerprint: str,
        update_budget: int | None = None,
        native_context_injector: Callable[..., list[str]] | None = None,
    ) -> tuple[list[str], RunRowSpec]:
        del bundle, inputs_root, repo_root
        bound, bound_row = bind_native_execution_command(
            command,
            row=row,
            payload_path=payload_path,
            collection_root=collection_root,
            update_budget=update_budget,
        )
        context_injector = native_context_injector or inject_native_execution_context
        return (
            context_injector(
                bound,
                row=bound_row,
                environment_fingerprint=environment_fingerprint,
                collection_root=collection_root,
            ),
            bound_row,
        )

    def missing_collection_outputs(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        collected: Mapping[str, str],
    ) -> list[str]:
        del bundle
        declared_missing = {Path(source).name for source in row.launch.collect} - set(collected)
        return sorted(declared_missing | set(missing_native_training_collection_outputs(row)))

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
        repo_root: Path | str | None,
        environment_fingerprint: str,
        update_budget: int | None = None,
        native_context_injector: Callable[..., list[str]] | None = None,
    ) -> tuple[list[str], RunRowSpec]:
        del environment_fingerprint
        if native_context_injector is not None:
            raise ExecutorFamilyError(
                "evaluation-matrix executor does not accept a native context injector"
            )
        if update_budget is not None:
            raise ExecutorFamilyError(
                "evaluation-matrix executor does not accept a native update budget"
            )
        if repo_root is None or not str(repo_root):
            raise ExecutorFamilyError(
                "evaluation-matrix executor requires a governed runtime repo root"
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
            "--repo-root",
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
                "--repo-root",
                str(repo_root),
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
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        collected: Mapping[str, str],
    ) -> list[str]:
        missing = sorted(set(EVALUATION_COLLECTION_OUTPUTS) - set(collected))
        if missing:
            return missing
        evidence = strict_model_validate_json(
            EvaluationLifecycleEvidence,
            Path(collected["evaluation-matrix-result.json"]).read_text(encoding="utf-8"),
        )
        topology = strict_model_validate_json(
            EvaluationWorkerTopologyEvidence,
            Path(collected["evaluation-worker-topology.json"]).read_text(encoding="utf-8"),
        )
        compaction_path = Path(collected["evaluation-batch-compaction.json"])
        compaction = EvaluationBatchCompactionEvidence.model_validate(
            migrate_structured_spec_payload(
                "EvaluationBatchCompactionEvidence",
                strict_json_loads(compaction_path.read_text(encoding="utf-8")),
                path=str(compaction_path),
            ).payload
        )
        plan = EvaluationMatrixBatchPlan.model_validate(
            migrate_structured_spec_payload(
                "EvaluationMatrixBatchPlan",
                row.launch.metadata.get("batch_plan"),
                path="launch.metadata.batch_plan",
            ).payload
        )
        expected_worker_count = min(
            bundle.launch_policy.max_parallel_rows,
            len(plan.batches),
        )
        observed_batch_ids = {
            batch_id for process in topology.processes for batch_id in process.ordered_batch_ids
        }
        timed_batch_ids = {
            timing.batch_id for process in topology.processes for timing in process.batch_timings
        }
        if topology.requested_worker_count != expected_worker_count or observed_batch_ids != {
            batch.batch_id for batch in plan.batches
        }:
            raise ExecutorFamilyError(
                "evaluation worker topology drifted from the authenticated batch plan"
            )
        if timed_batch_ids != observed_batch_ids:
            raise ExecutorFamilyError(
                "evaluation batch timings drifted from the authenticated worker topology"
            )
        if (
            compaction.matrix_intent_hash != plan.matrix_intent_hash
            or compaction.ordered_batch_ids != tuple(batch.batch_id for batch in plan.batches)
            or compaction.declared_leaf_ids != tuple(item.leaf_id for item in plan.consumers)
            or compaction.required_leaf_ids_by_batch
            != {batch.batch_id: tuple(batch.required_leaf_ids or ()) for batch in plan.batches}
        ):
            raise ExecutorFamilyError(
                "evaluation compaction evidence drifted from the authenticated batch plan"
            )
        provider = ImmutableArtifactBlobProvider(
            compaction_path.parent / "evaluation-batch-compaction"
        )
        for reclamation in compaction.reclamations:
            for acknowledgement in reclamation.leaf_acknowledgements:
                provider.get_bytes(acknowledgement.fragment)
                provider.get_bytes(acknowledgement.merge_state)
        for terminal in compaction.terminal_products:
            provider.get_bytes(terminal)
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
    evidence = strict_model_validate_json(
        EvaluationLifecycleEvidence, Path(path).read_text(encoding="utf-8")
    )
    # Trusted internal projection of the model validated immediately above.
    return json.loads(evidence.model_dump_json())


def evaluation_matrix_ordered_union(
    bundle: RunBundle,
    lifecycles: Sequence[EvaluationLifecycleEvidence],
) -> EvaluationMatrixOrderedUnionEvidence:
    """Verify ordered batch outcomes reconstruct the compiler-bound matrix."""
    if len(bundle.rows) != 1 or len(lifecycles) != 1:
        raise ExecutorFamilyError("evaluation ordered-union evidence is incomplete")
    row = bundle.rows[0]
    lifecycle = lifecycles[0]
    metadata = row.launch.metadata
    try:
        plan = EvaluationMatrixBatchPlan.model_validate(
            migrate_structured_spec_payload(
                "EvaluationMatrixBatchPlan",
                metadata.get("batch_plan"),
                path="launch.metadata.batch_plan",
            ).payload
        )
    except ValueError as exc:
        raise ExecutorFamilyError("evaluation row lacks an authenticated batch plan") from exc
    if lifecycle.orchestration_row_id != row.row_id:
        raise ExecutorFamilyError("evaluation lifecycle row identity drifted")
    ordered_row_ids = [row_id for batch in plan.batches for row_id in batch.ordered_row_ids]
    if tuple(ordered_row_ids) != lifecycle.ordered_row_ids:
        raise ExecutorFamilyError(
            "evaluation lifecycle outcomes drifted from the batch-plan ordered union"
        )
    intent_hash = metadata.get("matrix_intent_hash")
    expected_hash = metadata.get("matrix_ordered_row_ids_sha256")
    if plan.matrix_intent_hash != intent_hash:
        raise ExecutorFamilyError("evaluation batch plan matrix identity drifted")
    if not isinstance(expected_hash, str):
        raise ExecutorFamilyError("evaluation row lacks its matrix order hash")
    observed_hash = sha256_bytes(canonical_json_bytes(ordered_row_ids))
    if observed_hash != expected_hash:
        raise ExecutorFamilyError(
            "evaluation batch ordered union does not reconstruct the authored matrix"
        )
    return EvaluationMatrixOrderedUnionEvidence(
        matrix_intent_hash=plan.matrix_intent_hash,
        ordered_row_ids_sha256=expected_hash,
        ordered_batch_ids=tuple(batch.batch_id for batch in plan.batches),
        ordered_row_ids=tuple(ordered_row_ids),
    )


__all__ = [
    "EVALUATION_COLLECTION_OUTPUTS",
    "EvaluationMatrixExecutorAdapter",
    "ExecutorFamilyAdapter",
    "ExecutorFamilyError",
    "NativeTrainingExecutorAdapter",
    "evaluation_lifecycle_payload",
    "evaluation_matrix_ordered_union",
    "executor_family_adapter",
]
