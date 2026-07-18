"""Public emission API for three-layer training-run specification storage."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from feedbax.contracts.manifest import (
    ArtifactRef,
    StrictModel,
    TrainingRunManifest,
    sha256_file,
)
from feedbax.contracts.run_matrix import (
    TRAINING_ROW_LOWERING_RESULT_SCHEMA_VERSION,
    TRAINING_ROW_PLANNING_PROVENANCE_SCHEMA_VERSION,
    TRAINING_ROW_PROVENANCE_SCHEMA_VERSION,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
    InlineMatrixBaseSpec,
    TrainingRunMatrixSpec,
)
from feedbax.contracts.migrations import default_spec_registry
from feedbax.contracts.spec_storage import (
    TRAINING_RUN_EXECUTION_CAPSULE_SCHEMA_VERSION,
    TrainingRunExecutionCapsule,
    build_resolved_semantics_snapshot,
    store_canonical_json_artifact,
    training_run_execution_hash,
    training_run_authored_envelope_hash,
    training_run_composed_intent_hash,
    training_run_intent_hash,
    training_spec_canonical_bytes,
)
from feedbax.training.run_matrix import (
    MaterializedRunMatrix,
    RowPayloadValidator,
    TrainingRowLowerer,
    materialize_adapted_run_matrix,
    materialize_run_matrix,
)


TRAINING_RUN_MATRIX_COMPILER_ID = "feedbax.training.run_matrix"
TRAINING_RUN_MATRIX_COMPILER_VERSION_V1 = "feedbax.training.run_matrix.v1"
TRAINING_RUN_MATRIX_COMPILER_VERSION_V2 = "feedbax.training.run_matrix.v2"
TRAINING_RUN_MATRIX_COMPILER_VERSION = "feedbax.training.run_matrix.v3"


class TrainingSpecStorageResult(StrictModel):
    """Identity and custody pointers returned by the registered emitter."""

    intent_hash: str
    authored_envelope_hash: str
    composed_intent_hash: str
    execution_hash: str
    resolved_root_hash: str
    authored_path: str
    snapshot_artifact: ArtifactRef
    capsule_artifact: ArtifactRef
    capsule: TrainingRunExecutionCapsule


def compile_training_run_matrix(
    authored: TrainingRunMatrixSpec | Mapping[str, Any],
    *,
    run_set_id: str,
    context: Any,
    allow_inline_base: bool = False,
    row_validator: RowPayloadValidator | None = None,
    row_lowerer: TrainingRowLowerer | None = None,
) -> Any:
    """Purely lower an authored training matrix into generic compiled rows."""
    from feedbax.orchestration.assembly import CompiledExecutionRow, CompiledRunSet
    from feedbax.orchestration.bundle import RowLaunchSpec

    matrix = (
        authored
        if isinstance(authored, TrainingRunMatrixSpec)
        else TrainingRunMatrixSpec.model_validate(
            default_spec_registry.migrate("TrainingRunMatrixSpec", authored).payload
        )
    )
    if isinstance(matrix.base, InlineMatrixBaseSpec) and not allow_inline_base:
        raise ValueError(
            "inline matrix bases are tests/fixtures only; production assembly requires "
            "a content-pinned authored_intent or resolved_output reference"
        )
    repo_root = context.repo_root
    if repo_root is None:
        raise ValueError("training matrix assembly requires AssemblyContext.repo_root")
    if row_validator is None and row_lowerer is None:
        materialized = materialize_run_matrix(matrix, repo_root=repo_root)
    else:
        materialized = materialize_adapted_run_matrix(
            matrix,
            repo_root=repo_root,
            row_validator=row_validator,
            row_lowerer=row_lowerer,
        )
    rows = []
    for row in materialized.rows:
        coordinate = _row_coordinate(materialized, row)
        resolved = {
            "run_set_id": run_set_id,
            "row_id": row.row_id,
            "planned_run_id": row.planned_run_id,
            "seed": row.seed,
            "axis_coordinates": (
                coordinate.model_dump(mode="json", exclude_none=True)
                if coordinate is not None
                else None
            ),
            "overrides": [
                item.model_dump(mode="json", exclude_none=True)
                if hasattr(item, "model_dump")
                else item
                for item in row.overrides
            ],
            "row_provenance": row.provenance.model_dump(mode="json", exclude_none=True),
            "payload": row.payload,
        }
        command = [
            "python",
            "-m",
            "feedbax",
            "execute-training-run-spec",
        ]
        checkpoint_progress = row.payload.get("checkpoint_progress")
        if (
            isinstance(checkpoint_progress, Mapping)
            and checkpoint_progress.get("continuation") is not None
        ):
            command.append("--resume")
        rows.append(
            CompiledExecutionRow(
                row_id=row.row_id,
                payload=row.payload,
                resolved_semantics=resolved,
                provenance=row.provenance,
                immutable_inputs=list(context.resolved_inputs),
                launch=RowLaunchSpec(
                    command=command,
                    collect=["manifest.json", "training-diagnostics.json"],
                    payload_routing={
                        "kind": "registered-execution-payload",
                        "spec": "execution.payload",
                        "manifest_root": "row-local",
                        "checkpoint_root": "row-local",
                    },
                ),
            )
        )
    return CompiledRunSet(rows=rows)


class TrainingRunMatrixCompiler:
    """Default registered compiler for TrainingRunMatrixSpec."""

    def __init__(
        self,
        *,
        allow_inline_base: bool = False,
        row_validator: RowPayloadValidator | None = None,
        row_lowerer: TrainingRowLowerer | None = None,
    ) -> None:
        self.allow_inline_base = allow_inline_base
        self.row_validator = row_validator
        self.row_lowerer = row_lowerer

    def compile(
        self,
        request: Any,
        *,
        authored: Mapping[str, Any],
        run_set_id: str,
        context: Any,
    ) -> Any:
        del request
        return compile_training_run_matrix(
            authored,
            run_set_id=run_set_id,
            context=context,
            allow_inline_base=self.allow_inline_base,
            row_validator=self.row_validator,
            row_lowerer=self.row_lowerer,
        )


class TrainingRunIdentityAdapter:
    """Training-family semantic adapter for generic execution envelopes."""

    def intent_hash(self, authored: Mapping[str, Any]) -> str:
        return training_run_intent_hash(authored)

    def build_capsule(
        self,
        row: Any,
        *,
        identities: Any,
        context: Any,
    ) -> Mapping[str, Any]:
        versions = {
            "training_run_matrix": TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
            "resolved_semantics": "feedbax.spec.training_run_resolved_semantics.v1",
            "execution_capsule": TRAINING_RUN_EXECUTION_CAPSULE_SCHEMA_VERSION,
            "training_row_provenance": TRAINING_ROW_PROVENANCE_SCHEMA_VERSION,
            "training_row_planning_provenance": (
                TRAINING_ROW_PLANNING_PROVENANCE_SCHEMA_VERSION
            ),
        }
        _record_payload_schema_versions(row.payload, versions)
        return TrainingRunExecutionCapsule(
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

        value = TrainingRunExecutionCapsule.model_validate(capsule)
        assert value.execution_hash is not None
        return RowIdentities(
            intent_hash=value.intent_hash,
            resolved_root_hash=value.resolved_root_hash,
            immutable_inputs=value.input_data_identities,
            execution_hash=value.execution_hash,
        )


def register_training_run_matrix_compiler(
    registry: Any,
    *,
    allow_inline_base: bool = False,
    row_validator: RowPayloadValidator | None = None,
    row_lowerer: TrainingRowLowerer | None = None,
) -> None:
    """Register the default matrix compiler and identity adapter."""
    registry.register(
        schema_id=TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
        compiler_id=TRAINING_RUN_MATRIX_COMPILER_ID,
        compiler_version=TRAINING_RUN_MATRIX_COMPILER_VERSION,
        compiler=TrainingRunMatrixCompiler(
            allow_inline_base=allow_inline_base,
            row_validator=row_validator,
            row_lowerer=row_lowerer,
        ),
        identity_adapter=TrainingRunIdentityAdapter(),
    )


def stamp_training_run_manifest_identities(
    manifest: TrainingRunManifest,
    storage: TrainingSpecStorageResult,
) -> TrainingRunManifest:
    """Stamp dual identity without permitting archived identity redefinition."""
    if manifest.intent_hash is not None and manifest.intent_hash != storage.intent_hash:
        raise ValueError(
            "archived TrainingRunManifest intent_hash must never be overwritten"
        )
    expected_execution = training_run_execution_hash(
        storage.resolved_root_hash,
        storage.capsule.input_data_identities,
    )
    if manifest.execution_hash is not None and manifest.execution_hash != expected_execution:
        raise ValueError(
            "materializer drift: archived TrainingRunManifest execution_hash must never "
            "be overwritten"
        )
    payload = manifest.model_dump(mode="json", exclude_none=True)
    payload.update(
        {
            "intent_hash": storage.intent_hash,
            "execution_hash": expected_execution,
            "resolved_semantics_root_hash": storage.resolved_root_hash,
            "input_data_identities": storage.capsule.input_data_identities,
        }
    )
    return TrainingRunManifest.model_validate(payload)


def emit_training_run_spec_storage(
    authored: TrainingRunMatrixSpec | Mapping[str, Any],
    *,
    repo_root: Path,
    authored_path: Path,
    custody_root: Path,
    materializer_commit: str,
    dependency_lock_path: Path,
    input_data_identities: list[dict[str, Any]] | None = None,
    environment_digest: str | None = None,
    allow_inline_base: bool = False,
    row_validator: RowPayloadValidator | None = None,
    row_lowerer: TrainingRowLowerer | None = None,
    _test_materializer_transform: Callable[[MaterializedRunMatrix], MaterializedRunMatrix]
    | None = None,
) -> TrainingSpecStorageResult:
    """Write layer 1 canonically and layers 2/3 through local custody.

    Inline bases are fixture-only and rejected unless ``allow_inline_base`` is
    deliberately enabled by a test or fixture producer.
    """
    if isinstance(authored, TrainingRunMatrixSpec):
        matrix = authored
    else:
        migrated = default_spec_registry.migrate("TrainingRunMatrixSpec", authored)
        matrix = TrainingRunMatrixSpec.model_validate(migrated.payload)
    if isinstance(matrix.base, InlineMatrixBaseSpec) and not allow_inline_base:
        raise ValueError(
            "inline matrix bases are tests/fixtures only; production emission requires "
            "a content-pinned authored_intent or resolved_output reference"
        )
    authored_document = matrix.model_dump(mode="json", exclude_none=True)
    intent_hash = training_run_intent_hash(authored_document)
    authored_envelope_hash = training_run_authored_envelope_hash(authored_document)
    if row_validator is None and row_lowerer is None:
        materialized = materialize_run_matrix(matrix, repo_root=repo_root)
    else:
        materialized = materialize_adapted_run_matrix(
            matrix,
            repo_root=repo_root,
            row_validator=row_validator,
            row_lowerer=row_lowerer,
        )
    if _test_materializer_transform is not None:
        materialized = _test_materializer_transform(materialized)
    authored_path.parent.mkdir(parents=True, exist_ok=True)
    authored_path.write_bytes(training_spec_canonical_bytes(authored_document) + b"\n")

    resolved_tree = {
        "run_set_id": materialized.run_set_id,
        "rows": [
            {
                "row_id": row.row_id,
                "planned_run_id": row.planned_run_id,
                "seed": row.seed,
                "axis_coordinates": (
                    _row_coordinate(materialized, row).model_dump(
                        mode="json", exclude_none=True
                    )
                    if _row_coordinate(materialized, row) is not None
                    else None
                ),
                "overrides": [
                    override.model_dump(mode="json", exclude_none=True)
                    if hasattr(override, "model_dump")
                    else override
                    for override in row.overrides
                ],
                "row_provenance": row.provenance.model_dump(
                    mode="json", exclude_none=True
                ),
                "payload": row.payload,
            }
            for row in materialized.rows
        ],
    }
    composed_intent_hash = training_run_composed_intent_hash(resolved_tree)
    snapshot = build_resolved_semantics_snapshot(resolved_tree)
    snapshot_artifact = store_canonical_json_artifact(
        snapshot,
        root=custody_root,
        role="training_run_resolved_semantics",
        logical_name=f"{authored_path.stem}.resolved.json",
    )
    inputs = list(input_data_identities or [])
    relevant_versions = {
        "training_run_matrix": matrix.schema_version,
        "resolved_semantics": snapshot["schema_version"],
        "execution_capsule": TRAINING_RUN_EXECUTION_CAPSULE_SCHEMA_VERSION,
        "training_row_provenance": TRAINING_ROW_PROVENANCE_SCHEMA_VERSION,
        "training_row_planning_provenance": (
            TRAINING_ROW_PLANNING_PROVENANCE_SCHEMA_VERSION
        ),
    }
    if row_lowerer is not None:
        relevant_versions["training_row_lowering_result"] = (
            TRAINING_ROW_LOWERING_RESULT_SCHEMA_VERSION
        )
    for row in materialized.rows:
        _record_payload_schema_versions(row.payload, relevant_versions)
    capsule = TrainingRunExecutionCapsule(
        materializer_commit=materializer_commit,
        relevant_schema_versions=relevant_versions,
        dependency_lock_digest=sha256_file(dependency_lock_path),
        environment_digest=environment_digest,
        input_data_identities=inputs,
        intent_hash=intent_hash,
        resolved_root_hash=snapshot["root_hash"],
    )
    capsule_artifact = store_canonical_json_artifact(
        capsule,
        root=custody_root,
        role="training_run_execution_capsule",
        logical_name=f"{authored_path.stem}.execution.json",
    )
    assert capsule.execution_hash is not None
    return TrainingSpecStorageResult(
        intent_hash=intent_hash,
        authored_envelope_hash=authored_envelope_hash,
        composed_intent_hash=composed_intent_hash,
        execution_hash=capsule.execution_hash,
        resolved_root_hash=snapshot["root_hash"],
        authored_path=str(authored_path),
        snapshot_artifact=snapshot_artifact,
        capsule_artifact=capsule_artifact,
        capsule=capsule,
    )


def _record_payload_schema_versions(
    payload: Mapping[str, Any],
    versions: dict[str, str],
) -> None:
    schema_id = payload.get("schema_id")
    schema_version = payload.get("schema_version")
    if isinstance(schema_id, str) and isinstance(schema_version, str):
        versions[schema_id] = schema_version
    graph = payload.get("graph")
    if isinstance(graph, Mapping):
        inline = graph.get("inline")
        if isinstance(inline, Mapping):
            graph_id = inline.get("schema_id")
            graph_version = inline.get("schema_version")
            if isinstance(graph_id, str) and isinstance(graph_version, str):
                versions[graph_id] = graph_version


def _row_coordinate(materialized: MaterializedRunMatrix, row: Any) -> Any:
    if row.coordinate is not None:
        return row.coordinate
    return next(
        (
            coordinate
            for coordinate in materialized.run_set_manifest.axes.runs
            if coordinate.run_id == row.planned_run_id
        ),
        None,
    )
