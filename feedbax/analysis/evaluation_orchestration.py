"""Assembly binding from authored evaluation matrices to governed orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import Field, model_validator

from feedbax.analysis.evaluation import (
    EvaluationRunMatrixSpec,
    materialize_evaluation_run_matrix,
    resolve_evaluation_matrix_authoring,
)
from feedbax.contracts.evaluation_composition import (
    EvaluationRunMatrixDeltaSpec,
    evaluation_matrix_delta_envelope_hash,
    is_evaluation_run_matrix_delta_payload,
)
from feedbax.contracts.evaluation_lifecycle import (
    EVALUATION_COLLECTION_OUTPUTS,
    EVALUATION_MATRIX_BATCH_PLAN_SCHEMA_VERSION,
    EvaluationMatrixBatchPlan,
    EvaluationMatrixBatchUnit,
)
from feedbax.contracts.base import (
    StrictModel,
    canonical_json_bytes,
    sha256_bytes,
)
from feedbax.contracts.manifest import (
    EVALUATION_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID,
    EVALUATION_RUN_MATRIX_DELTA_SPEC_SCHEMA_VERSION,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION,
)
from feedbax.contracts.spec_storage import (
    canonicalize_immutable_input_identities,
    training_run_execution_hash,
)
from feedbax.orchestration.staged_root_custody import (
    STAGED_ROOT_CUSTODY_SCHEMA_VERSION,
    StagedRootCustody,
)


EVALUATION_RUN_MATRIX_COMPILER_ID = "feedbax.evaluation.run_matrix"
EVALUATION_RUN_MATRIX_COMPILER_VERSION = "feedbax.evaluation.run_matrix.v1"
EVALUATION_MATRIX_RESOLVED_SEMANTICS_SCHEMA_ID = "feedbax.spec.evaluation_matrix_resolved_semantics"
EVALUATION_MATRIX_RESOLVED_SEMANTICS_SCHEMA_VERSION = (
    "feedbax.spec.evaluation_matrix_resolved_semantics.v1"
)
EVALUATION_MATRIX_EXECUTION_CAPSULE_SCHEMA_ID = (
    "feedbax.manifest.evaluation_matrix_execution_capsule"
)
EVALUATION_MATRIX_EXECUTION_CAPSULE_SCHEMA_VERSION = (
    "feedbax.manifest.evaluation_matrix_execution_capsule.v1"
)


class EvaluationMatrixExecutionCapsule(StrictModel):
    """Identity capsule for one whole-matrix governed executor invocation."""

    schema_id: str = EVALUATION_MATRIX_EXECUTION_CAPSULE_SCHEMA_ID
    schema_version: str = EVALUATION_MATRIX_EXECUTION_CAPSULE_SCHEMA_VERSION
    materializer_commit: str = Field(min_length=1)
    relevant_schema_versions: dict[str, str] = Field(min_length=1)
    dependency_lock_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    input_data_identities: list[dict[str, Any]] = Field(default_factory=list)
    intent_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    resolved_root_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_identity(self) -> "EvaluationMatrixExecutionCapsule":
        if self.schema_id != EVALUATION_MATRIX_EXECUTION_CAPSULE_SCHEMA_ID:
            raise ValueError("unsupported evaluation-matrix execution capsule schema_id")
        if self.schema_version != EVALUATION_MATRIX_EXECUTION_CAPSULE_SCHEMA_VERSION:
            raise ValueError("unsupported evaluation-matrix execution capsule schema_version")
        canonical_inputs = canonicalize_immutable_input_identities(self.input_data_identities)
        if self.input_data_identities != canonical_inputs:
            raise ValueError("evaluation-matrix capsule inputs must use canonical identity order")
        expected = training_run_execution_hash(
            self.resolved_root_hash,
            canonical_inputs,
        )
        if self.execution_hash != expected:
            raise ValueError("evaluation-matrix execution hash does not match resolved semantics")
        return self


def evaluation_matrix_intent_hash(authored: Mapping[str, Any]) -> str:
    """Return the authored identity without treating a delta locator as content."""
    if is_evaluation_run_matrix_delta_payload(authored):
        return evaluation_matrix_delta_envelope_hash(
            EvaluationRunMatrixDeltaSpec.model_validate(authored)
        )
    return sha256_bytes(canonical_json_bytes(dict(authored)))


def compile_evaluation_run_matrix_for_orchestration(
    authored: Mapping[str, Any],
    *,
    run_set_id: str,
    context: Any,
    registry: Any,
) -> Any:
    """Compile one matrix row with an authenticated persistent-worker batch plan."""
    from feedbax.orchestration.assembly import CompiledExecutionRow, CompiledRunSet
    from feedbax.orchestration.bundle import RowLaunchSpec

    if context.repo_root is None:
        raise ValueError("evaluation matrix assembly requires AssemblyContext.repo_root")
    matrix, composition = resolve_evaluation_matrix_authoring(
        authored,
        repo_root=context.repo_root,
    )
    rows = materialize_evaluation_run_matrix(
        matrix,
        registry=registry,
        repo_root=context.repo_root,
    )
    _validate_staged_root_applicability(matrix, context.staged_roots)
    ordered_row_ids = tuple(row.row_id for row in rows)
    payload_hashes = {
        row.row_id: sha256_bytes(
            canonical_json_bytes(row.payload.model_dump(mode="json", exclude_none=True))
        )
        for row in rows
    }
    intent_hash = evaluation_matrix_intent_hash(authored)
    ordered_row_ids_sha256 = sha256_bytes(canonical_json_bytes(ordered_row_ids))
    plan = context.evaluation_batch_plan or EvaluationMatrixBatchPlan(
        matrix_intent_hash=intent_hash,
        batches=(
            EvaluationMatrixBatchUnit(
                batch_id="whole-matrix",
                ordered_row_ids=ordered_row_ids,
            ),
        ),
    )
    if plan.matrix_intent_hash != intent_hash:
        raise ValueError("evaluation batch plan matrix_intent_hash does not match authored matrix")
    planned_order = tuple(row_id for batch in plan.batches for row_id in batch.ordered_row_ids)
    if planned_order != ordered_row_ids:
        raise ValueError(
            "evaluation batch plan ordered union must exactly equal the authored matrix row order"
        )
    shared_semantics = {
        "schema_id": EVALUATION_MATRIX_RESOLVED_SEMANTICS_SCHEMA_ID,
        "schema_version": EVALUATION_MATRIX_RESOLVED_SEMANTICS_SCHEMA_VERSION,
        "run_set_id": run_set_id,
        "authored_schema_id": authored["schema_id"],
        "authored_schema_version": authored["schema_version"],
        "flattened_matrix_sha256": sha256_bytes(
            canonical_json_bytes(matrix.model_dump(mode="json", exclude_none=True))
        ),
        "ordered_row_ids": list(ordered_row_ids),
        "canonical_payload_sha256": payload_hashes,
        "ordered_row_ids_sha256": ordered_row_ids_sha256,
        "composition": (
            composition.model_dump(mode="json", exclude_none=True)
            if composition is not None
            else None
        ),
        "staged_roots": [
            {
                "binding_name": item.binding_name,
                "root_kind": item.root_kind,
                "custody_ref": item.custody_ref,
                "content_sha256": item.content_sha256,
            }
            for item in context.staged_roots
        ],
    }
    lifecycle_row_id = f"matrix-{intent_hash[:16]}"
    return CompiledRunSet(
        rows=[
            CompiledExecutionRow(
                row_id=lifecycle_row_id,
                execution_family="evaluation-matrix",
                payload=dict(authored),
                resolved_semantics={
                    **shared_semantics,
                    "batch_plan": plan.model_dump(mode="json", exclude_none=True),
                },
                immutable_inputs=list(context.resolved_inputs),
                launch=RowLaunchSpec(
                    command=["python", "-m", "feedbax", "matrix-harness"],
                    collect=list(EVALUATION_COLLECTION_OUTPUTS),
                    payload_routing={
                        "kind": "registered-execution-payload",
                        "spec": "execution.payload",
                    },
                    metadata={
                        "matrix_intent_hash": intent_hash,
                        "matrix_ordered_row_ids_sha256": ordered_row_ids_sha256,
                        "batch_plan": plan.model_dump(mode="json", exclude_none=True),
                    },
                ),
            )
        ]
    )


class EvaluationRunMatrixCompiler:
    """Registered compiler for direct and content-pinned evaluation authoring."""

    def __init__(self, registry: Any) -> None:
        self.registry = registry

    def compile(
        self,
        *,
        authored: Mapping[str, Any],
        run_set_id: str,
        context: Any,
    ) -> Any:
        return compile_evaluation_run_matrix_for_orchestration(
            authored,
            run_set_id=run_set_id,
            context=context,
            registry=self.registry,
        )


class EvaluationMatrixIdentityAdapter:
    """Bind evaluation intent and resolved whole-matrix semantics."""

    def intent_hash(self, authored: Mapping[str, Any]) -> str:
        return evaluation_matrix_intent_hash(authored)

    def build_capsule(
        self,
        row: Any,
        *,
        identities: Any,
        context: Any,
    ) -> Mapping[str, Any]:
        versions = {
            "evaluation_run_matrix": EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION,
            "evaluation_matrix_batch_plan": EVALUATION_MATRIX_BATCH_PLAN_SCHEMA_VERSION,
            "resolved_semantics": EVALUATION_MATRIX_RESOLVED_SEMANTICS_SCHEMA_VERSION,
            "execution_capsule": EVALUATION_MATRIX_EXECUTION_CAPSULE_SCHEMA_VERSION,
        }
        if row.payload["schema_id"] == EVALUATION_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID:
            versions["evaluation_run_matrix_delta"] = (
                EVALUATION_RUN_MATRIX_DELTA_SPEC_SCHEMA_VERSION
            )
        if context.staged_roots:
            versions["staged_root_custody"] = STAGED_ROOT_CUSTODY_SCHEMA_VERSION
        return EvaluationMatrixExecutionCapsule(
            materializer_commit=context.materializer_commit,
            relevant_schema_versions=versions,
            dependency_lock_digest=context.dependency_lock_digest,
            environment_digest=context.environment_digest,
            input_data_identities=identities.immutable_inputs,
            intent_hash=identities.intent_hash,
            resolved_root_hash=identities.resolved_root_hash,
            execution_hash=identities.execution_hash,
        ).model_dump(mode="json", exclude_none=True)

    def capsule_identities(self, capsule: Mapping[str, Any]) -> Any:
        from feedbax.orchestration.assembly import RowIdentities

        value = EvaluationMatrixExecutionCapsule.model_validate(capsule)
        return RowIdentities(
            intent_hash=value.intent_hash,
            resolved_root_hash=value.resolved_root_hash,
            immutable_inputs=value.input_data_identities,
            execution_hash=value.execution_hash,
        )


def register_evaluation_run_matrix_compiler(registry: Any, *, evaluation_registry: Any) -> None:
    """Register the two authored evaluation schemas against one closed compiler."""
    compiler = EvaluationRunMatrixCompiler(evaluation_registry)
    adapter = EvaluationMatrixIdentityAdapter()
    for schema_id in (
        EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID,
        EVALUATION_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID,
    ):
        registry.register(
            schema_id=schema_id,
            compiler_id=EVALUATION_RUN_MATRIX_COMPILER_ID,
            compiler_version=EVALUATION_RUN_MATRIX_COMPILER_VERSION,
            compiler=compiler,
            identity_adapter=adapter,
        )


def _validate_staged_root_applicability(
    matrix: EvaluationRunMatrixSpec,
    staged_roots: tuple[StagedRootCustody, ...],
) -> None:
    available = {(item.root_kind, item.binding_name) for item in staged_roots}
    required: set[tuple[str, str]] = set()
    for name, prerequisite in matrix.staged_parents.items():
        if prerequisite.artifact_provider is None:
            required.add(("manifest-store", name))
        else:
            required.add(("artifact-provider", prerequisite.artifact_provider))
        checkpoint_binding = prerequisite.parent.metadata.get("checkpoint_custody_binding")
        if isinstance(checkpoint_binding, str) and checkpoint_binding:
            required.add(("checkpoint-custody", checkpoint_binding))
    missing = sorted(required - available)
    if missing:
        raise ValueError(
            f"evaluation matrix staged parents lack governed staged-root custody: {missing!r}"
        )


__all__ = [
    "EVALUATION_MATRIX_EXECUTION_CAPSULE_SCHEMA_ID",
    "EVALUATION_MATRIX_EXECUTION_CAPSULE_SCHEMA_VERSION",
    "EVALUATION_MATRIX_RESOLVED_SEMANTICS_SCHEMA_ID",
    "EVALUATION_MATRIX_RESOLVED_SEMANTICS_SCHEMA_VERSION",
    "EVALUATION_RUN_MATRIX_COMPILER_ID",
    "EVALUATION_RUN_MATRIX_COMPILER_VERSION",
    "EvaluationMatrixExecutionCapsule",
    "EvaluationMatrixIdentityAdapter",
    "EvaluationRunMatrixCompiler",
    "compile_evaluation_run_matrix_for_orchestration",
    "evaluation_matrix_intent_hash",
    "register_evaluation_run_matrix_compiler",
]
