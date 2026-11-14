"""Compile training rows and bind their retained execution identities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from feedbax.contracts.manifest import (
    TRAINING_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID,
    TrainingRunAxisCoordinate,
)
from feedbax.contracts.run_matrix import (
    TRAINING_ROW_LOWERING_RESULT_SCHEMA_VERSION,
    TRAINING_ROW_PLANNING_PROVENANCE_SCHEMA_VERSION,
    TRAINING_ROW_PROVENANCE_SCHEMA_VERSION,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
    InlineMatrixBaseSpec,
    TrainingRowProvenance,
    TrainingRunMatrixSpec,
)
from feedbax.contracts.migrations import default_spec_registry
from feedbax.contracts.training import TRAINING_RUN_SPEC_SCHEMA_ID, TrainingProgramRegistry
from feedbax.contracts.worker import NATIVE_TRAINING_COLLECTION_OUTPUTS
from feedbax.contracts.training_matrix_composition import (
    TrainingRunMatrixDeltaSpec,
    flatten_training_run_matrix_delta,
    is_training_run_matrix_delta_payload,
    training_matrix_delta_envelope_hash,
)
from feedbax.contracts.spec_storage import (
    TRAINING_RUN_EXECUTION_CAPSULE_SCHEMA_VERSION,
    TrainingRunExecutionCapsule,
    canonical_training_run_spec_sha256,
    training_run_intent_hash,
)
from feedbax.training.run_matrix import (
    MaterializedRunMatrix,
    RowPayloadValidator,
    TrainingRowLowerer,
    materialize_adapted_run_matrix,
    materialize_run_matrix,
)


if TYPE_CHECKING:
    from feedbax.orchestration.assembly import CompiledExecutionRow
    from feedbax.orchestration.bundle import ImmutableInputIdentity


TRAINING_RUN_MATRIX_COMPILER_ID = "feedbax.training.run_matrix"
TRAINING_RUN_MATRIX_COMPILER_VERSION_V1 = "feedbax.training.run_matrix.v1"
TRAINING_RUN_MATRIX_COMPILER_VERSION_V2 = "feedbax.training.run_matrix.v2"
TRAINING_RUN_MATRIX_COMPILER_VERSION = "feedbax.training.run_matrix.v3"


def compile_training_run_matrix(
    authored: TrainingRunMatrixSpec | Mapping[str, Any],
    *,
    run_set_id: str,
    context: Any,
    method_registry: TrainingProgramRegistry,
    allow_inline_base: bool = False,
    row_validator: RowPayloadValidator | None = None,
    row_lowerer: TrainingRowLowerer | None = None,
) -> Any:
    """Purely lower an authored training matrix into generic compiled rows."""
    from feedbax.orchestration.assembly import CompiledRunSet

    repo_root = context.repo_root
    if repo_root is None:
        raise ValueError("training matrix assembly requires AssemblyContext.repo_root")
    if isinstance(authored, TrainingRunMatrixSpec):
        matrix = authored
    elif is_training_run_matrix_delta_payload(authored):
        delta = TrainingRunMatrixDeltaSpec.model_validate(authored)
        matrix = TrainingRunMatrixSpec.model_validate(
            flatten_training_run_matrix_delta(delta, repo_root=repo_root).payload
        )
    else:
        matrix = TrainingRunMatrixSpec.model_validate(
            default_spec_registry.migrate("TrainingRunMatrixSpec", authored).payload
        )
    if isinstance(matrix.base, InlineMatrixBaseSpec) and not allow_inline_base:
        raise ValueError(
            "inline matrix bases are tests/fixtures only; production assembly requires "
            "a content-pinned authored_intent or resolved_output reference"
        )
    if row_validator is None:
        materialized = materialize_run_matrix(
            matrix,
            repo_root=repo_root,
            method_registry=method_registry,
            row_lowerer=row_lowerer,
        )
    else:
        materialized = materialize_adapted_run_matrix(
            matrix,
            repo_root=repo_root,
            row_validator=row_validator,
            row_lowerer=row_lowerer,
            row_lowering_context=getattr(
                context,
                "training_row_lowering_context",
                None,
            ),
        )
    return CompiledRunSet(
        rows=[
            compile_native_training_row(
                row.payload,
                provenance=row.provenance,
                coordinate=_row_coordinate(materialized, row),
                overrides=row.overrides,
                run_set_id=run_set_id,
                immutable_inputs=context.resolved_inputs,
            )
            for row in materialized.rows
        ]
    )


def compile_native_training_row(
    payload: dict[str, Any],
    *,
    provenance: TrainingRowProvenance,
    coordinate: TrainingRunAxisCoordinate | None,
    overrides: list[Any],
    run_set_id: str,
    immutable_inputs: tuple[ImmutableInputIdentity, ...] = (),
    plugin_modules: tuple[str, ...] = (),
) -> CompiledExecutionRow:
    """Bind one already-materialized native row to its registered executor."""
    from feedbax.orchestration.assembly import CompiledExecutionRow
    from feedbax.orchestration.bundle import RowLaunchSpec

    resolved = {
        "run_set_id": run_set_id,
        "row_id": provenance.row_id,
        "planned_run_id": provenance.planned_run_id,
        "seed": provenance.seed,
        "axis_coordinates": (
            coordinate.model_dump(mode="json", exclude_none=True)
            if coordinate is not None
            else None
        ),
        "overrides": [
            item.model_dump(mode="json", exclude_none=True)
            if hasattr(item, "model_dump")
            else item
            for item in overrides
        ],
        "row_provenance": provenance.model_dump(mode="json", exclude_none=True),
        "payload": payload,
    }
    command = ["python", "-m", "feedbax", "execute-training-run-spec"]
    checkpoint_progress = payload.get("checkpoint_progress")
    if (
        isinstance(checkpoint_progress, Mapping)
        and checkpoint_progress.get("continuation") is not None
    ):
        command.append("--resume")
    for module in plugin_modules:
        command.extend(("--plugin", module))
    return CompiledExecutionRow(
        row_id=provenance.row_id,
        execution_family="native-training",
        payload=payload,
        resolved_semantics=resolved,
        provenance=provenance,
        immutable_inputs=list(immutable_inputs),
        launch=RowLaunchSpec(
            command=command,
            collect=list(NATIVE_TRAINING_COLLECTION_OUTPUTS),
            payload_routing={
                "kind": "registered-execution-payload",
                "spec": "execution.payload",
                "manifest_root": "row-local",
                "checkpoint_root": "row-local",
            },
        ),
    )


class TrainingRunMatrixCompiler:
    """Default registered compiler for TrainingRunMatrixSpec."""

    def __init__(
        self,
        *,
        method_registry: TrainingProgramRegistry,
        allow_inline_base: bool = False,
        row_validator: RowPayloadValidator | None = None,
        row_lowerer: TrainingRowLowerer | None = None,
    ) -> None:
        self.allow_inline_base = allow_inline_base
        self.method_registry = method_registry
        if row_lowerer is None or row_validator is None:
            raise ValueError("row_lowerer and row_validator must be injected after bootstrap")
        self.row_validator = row_validator
        self.row_lowerer = row_lowerer

    def compile(
        self,
        *,
        authored: Mapping[str, Any],
        run_set_id: str,
        context: Any,
    ) -> Any:
        return compile_training_run_matrix(
            authored,
            run_set_id=run_set_id,
            context=context,
            method_registry=self.method_registry,
            allow_inline_base=self.allow_inline_base,
            row_validator=self.row_validator,
            row_lowerer=self.row_lowerer,
        )


class TrainingRunIdentityAdapter:
    """Training-family semantic adapter for generic execution envelopes."""

    def intent_hash(self, authored: Mapping[str, Any]) -> str:
        if authored.get("schema_id") == TRAINING_RUN_SPEC_SCHEMA_ID:
            return canonical_training_run_spec_sha256(authored)
        if is_training_run_matrix_delta_payload(authored):
            return training_matrix_delta_envelope_hash(
                TrainingRunMatrixDeltaSpec.model_validate(authored)
            )
        return training_run_intent_hash(authored)

    def build_capsule(
        self,
        row: Any,
        *,
        identities: Any,
        context: Any,
    ) -> Mapping[str, Any]:
        versions = {
            "resolved_semantics": "feedbax.spec.training_run_resolved_semantics.v1",
            "execution_capsule": TRAINING_RUN_EXECUTION_CAPSULE_SCHEMA_VERSION,
            "training_row_provenance": TRAINING_ROW_PROVENANCE_SCHEMA_VERSION,
            "training_row_planning_provenance": (TRAINING_ROW_PLANNING_PROVENANCE_SCHEMA_VERSION),
        }
        if (
            context.authored_ref is None
            or context.authored_ref.schema_id != TRAINING_RUN_SPEC_SCHEMA_ID
        ):
            versions["training_run_matrix"] = TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION
        if (
            context.authored_ref is not None
            and context.authored_ref.schema_id == TRAINING_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID
        ):
            versions["training_run_matrix_delta"] = context.authored_ref.schema_version
        if row.provenance is not None and row.provenance.lowerer_identities:
            versions["training_row_lowering_result"] = TRAINING_ROW_LOWERING_RESULT_SCHEMA_VERSION
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
    method_registry: TrainingProgramRegistry,
    allow_inline_base: bool = False,
    row_validator: RowPayloadValidator | None = None,
    row_lowerer: TrainingRowLowerer | None = None,
) -> None:
    """Register the default matrix compiler and identity adapter."""
    compiler = TrainingRunMatrixCompiler(
        method_registry=method_registry,
        allow_inline_base=allow_inline_base,
        row_validator=row_validator,
        row_lowerer=row_lowerer,
    )
    adapter = TrainingRunIdentityAdapter()
    for schema_id in (
        TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
        TRAINING_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID,
    ):
        registry.register(
            schema_id=schema_id,
            compiler_id=TRAINING_RUN_MATRIX_COMPILER_ID,
            compiler_version=TRAINING_RUN_MATRIX_COMPILER_VERSION,
            compiler=compiler,
            identity_adapter=adapter,
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
