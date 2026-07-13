"""Contracts for governed multi-row training run matrices."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field, field_validator, model_validator

from feedbax.contracts.expressions import Coalesce, ValueExpr, ValueQuery
from feedbax.contracts.extraction import SourceBinding
from feedbax.contracts.manifest import (
    OverridePatch,
    StrictModel,
    TrainingSweepAxis,
    TrainingSweepCombinationSpec,
)


TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID = "feedbax.spec.training_run_matrix"
TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION_V1 = "feedbax.spec.training_run_matrix.v1"
TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION_V2 = "feedbax.spec.training_run_matrix.v2"
TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION = "feedbax.spec.training_run_matrix.v3"
AUTHORED_TRAINING_ROW_SCHEMA_ID = "feedbax.spec.authored_training_row"
AUTHORED_TRAINING_ROW_SCHEMA_VERSION = f"{AUTHORED_TRAINING_ROW_SCHEMA_ID}.v1"
TRAINING_ROW_LOWERING_RESULT_SCHEMA_ID = "feedbax.spec.training_row_lowering_result"
TRAINING_ROW_LOWERING_RESULT_SCHEMA_VERSION = (
    f"{TRAINING_ROW_LOWERING_RESULT_SCHEMA_ID}.v1"
)
TRAINING_ROW_PROVENANCE_SCHEMA_ID = "feedbax.spec.training_row_provenance"
TRAINING_ROW_PROVENANCE_SCHEMA_VERSION_V1 = f"{TRAINING_ROW_PROVENANCE_SCHEMA_ID}.v1"
TRAINING_ROW_PROVENANCE_SCHEMA_VERSION = f"{TRAINING_ROW_PROVENANCE_SCHEMA_ID}.v2"
TRAINING_ROW_PLANNING_PROVENANCE_SCHEMA_ID = (
    "feedbax.spec.training_row_planning_provenance"
)
TRAINING_ROW_PLANNING_PROVENANCE_SCHEMA_VERSION_V1 = (
    f"{TRAINING_ROW_PLANNING_PROVENANCE_SCHEMA_ID}.v1"
)
TRAINING_ROW_PLANNING_PROVENANCE_SCHEMA_VERSION = (
    f"{TRAINING_ROW_PLANNING_PROVENANCE_SCHEMA_ID}.v2"
)
RUN_MATRIX_MATERIALIZATION_SCHEMA_ID = "feedbax.manifest.run_matrix_materialization"
RUN_MATRIX_MATERIALIZATION_SCHEMA_VERSION_V1 = "feedbax.run_matrix_materialization.v1"
RUN_MATRIX_MATERIALIZATION_SCHEMA_VERSION_V2 = f"{RUN_MATRIX_MATERIALIZATION_SCHEMA_ID}.v2"
RUN_MATRIX_MATERIALIZATION_SCHEMA_VERSION = f"{RUN_MATRIX_MATERIALIZATION_SCHEMA_ID}.v3"
_PATH_SAFE_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class RowLowererIdentity(StrictModel):
    """Stable identity of the implementation that lowered one authored row."""

    lowerer_id: str = Field(min_length=1)
    lowerer_version: str = Field(min_length=1)


class AuthoredTrainingRow(StrictModel):
    """Axis-patched authored row supplied to a registered row lowerer."""

    schema_id: Literal["feedbax.spec.authored_training_row"] = (
        AUTHORED_TRAINING_ROW_SCHEMA_ID
    )
    schema_version: Literal["feedbax.spec.authored_training_row.v1"] = (
        AUTHORED_TRAINING_ROW_SCHEMA_VERSION
    )
    row_id: str = Field(min_length=1)
    row_index: int = Field(ge=0)
    payload: dict[str, Any]
    payload_hash: str
    seed: int | None = None
    axis_coordinates: dict[str, Any]
    overrides: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("payload_hash")
    @classmethod
    def _validate_payload_hash(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("payload_hash must be a lowercase sha256 digest")
        return value


class TrainingRowLoweringResult(StrictModel):
    """Authoritative execution payload returned by one declared row lowerer."""

    schema_id: Literal["feedbax.spec.training_row_lowering_result"] = (
        TRAINING_ROW_LOWERING_RESULT_SCHEMA_ID
    )
    schema_version: Literal["feedbax.spec.training_row_lowering_result.v1"] = (
        TRAINING_ROW_LOWERING_RESULT_SCHEMA_VERSION
    )
    execution_payload: dict[str, Any]
    lowerer_identities: list[RowLowererIdentity] = Field(min_length=1)


class TrainingRowPlanningProvenance(StrictModel):
    """Authored and lowerer identity bound into deterministic planned-run IDs."""

    schema_id: Literal["feedbax.spec.training_row_planning_provenance"] = (
        TRAINING_ROW_PLANNING_PROVENANCE_SCHEMA_ID
    )
    schema_version: Literal["feedbax.spec.training_row_planning_provenance.v2"] = (
        TRAINING_ROW_PLANNING_PROVENANCE_SCHEMA_VERSION
    )
    authored_payload_hash: str
    lowered_execution_payload_hash: str
    lowerer_identities: list[RowLowererIdentity] = Field(default_factory=list)

    @field_validator("authored_payload_hash", "lowered_execution_payload_hash")
    @classmethod
    def _validate_payload_hash(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("payload hashes must be lowercase sha256 digests")
        return value


class TrainingRowProvenance(StrictModel):
    """Canonical authored-to-execution provenance for one materialized row."""

    schema_id: Literal["feedbax.spec.training_row_provenance"] = (
        TRAINING_ROW_PROVENANCE_SCHEMA_ID
    )
    schema_version: Literal["feedbax.spec.training_row_provenance.v2"] = (
        TRAINING_ROW_PROVENANCE_SCHEMA_VERSION
    )
    row_id: str = Field(min_length=1)
    row_index: int = Field(ge=0)
    planned_run_id: str = Field(min_length=1)
    authored_payload_hash: str
    lowered_execution_payload_hash: str
    seed: int | None = None
    axis_coordinates: dict[str, Any]
    overrides: list[dict[str, Any]] = Field(default_factory=list)
    lowerer_identities: list[RowLowererIdentity] = Field(default_factory=list)

    @field_validator("authored_payload_hash", "lowered_execution_payload_hash")
    @classmethod
    def _validate_payload_hash(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("payload hashes must be lowercase sha256 digests")
        return value


class InlineMatrixBaseSpec(StrictModel):
    """Inline base reserved for tests, fixtures, and explicit emission opt-in."""

    kind: Literal["inline"] = "inline"
    inline: dict[str, Any]


class AuthoredIntentMatrixBaseSpec(StrictModel):
    """Canonical-content-pinned reference to an authored matrix envelope."""

    kind: Literal["authored_intent"] = "authored_intent"
    ref: str
    content_hash: str
    pin_algorithm: Literal["canonical_json_v1", "legacy_raw_sha256"] = "canonical_json_v1"
    payload_path: str | None = None
    symbolic_name: str | None = None

    @model_validator(mode="after")
    def _validate_ref(self) -> "AuthoredIntentMatrixBaseSpec":
        _validate_base_reference(self.ref, self.content_hash, self.payload_path)
        return self


class ResolvedOutputMatrixBaseSpec(StrictModel):
    """Resolved-root-pinned reference to immutable layer-2 semantics."""

    kind: Literal["resolved_output"] = "resolved_output"
    ref: str
    resolved_root_hash: str
    payload_path: str | None = None
    symbolic_name: str | None = None

    @model_validator(mode="after")
    def _validate_ref(self) -> "ResolvedOutputMatrixBaseSpec":
        _validate_base_reference(self.ref, self.resolved_root_hash, self.payload_path)
        return self


MatrixBaseSpec: TypeAlias = Annotated[
    InlineMatrixBaseSpec | AuthoredIntentMatrixBaseSpec | ResolvedOutputMatrixBaseSpec,
    Field(discriminator="kind"),
]


def _validate_base_reference(ref: str, digest: str, payload_path: str | None) -> None:
    if Path(ref).is_absolute():
        raise ValueError("/base/ref must be repo-relative")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("/base content identity must be a lowercase sha256 digest")
    if payload_path is not None:
        _validate_dotted_path(payload_path, "/base/payload_path")


class MatrixDerivation(StrictModel):
    """One grammar-derived value written into the base payload before expansion."""

    output_path: str
    query: ValueExpr

    @model_validator(mode="after")
    def _validate_derivation(self) -> "MatrixDerivation":
        _validate_dotted_path(self.output_path, "/derivations/output_path")
        return self


class MatrixRow(StrictModel):
    """One explicit named row in a run matrix."""

    row_id: str
    label: str | None = None
    overrides: list[OverridePatch] = Field(default_factory=list)
    seed: int | None = None
    notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_row(self) -> "MatrixRow":
        if not _PATH_SAFE_RE.match(self.row_id):
            raise ValueError(f"/rows/row_id is not path-safe: {self.row_id!r}")
        return self


class MatrixCompositionDelta(StrictModel):
    """One ordered authored layer with explicit ancestor-override acknowledgement."""

    layer_id: str
    patches: list[OverridePatch] = Field(default_factory=list)
    acknowledges_ancestor_paths: list[str] = Field(default_factory=list)
    schema_id: str | None = None
    schema_version: str | None = None

    @model_validator(mode="after")
    def _validate_delta(self) -> "MatrixCompositionDelta":
        if not _PATH_SAFE_RE.match(self.layer_id):
            raise ValueError(f"/deltas/layer_id is not path-safe: {self.layer_id!r}")
        if (self.schema_id is None) != (self.schema_version is None):
            raise ValueError(
                "/deltas schema_id and schema_version must be declared together at a boundary"
            )
        for path in self.acknowledges_ancestor_paths:
            _validate_dotted_path(path, "/deltas/acknowledges_ancestor_paths")
        return self


class DurableSlotTransform(StrictModel):
    transform_id: str
    version: str
    slot: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class ForkFromSelectedCheckpoint(StrictModel):
    kind: Literal["fork_from_selected_checkpoint"] = "fork_from_selected_checkpoint"
    source_execution_hash: str
    source_row_id: str
    checkpoint_transaction_id: str
    checkpoint_root_hash: str
    slot_transforms: list[DurableSlotTransform] = Field(default_factory=list)

    @model_validator(mode="after")
    def _hashes(self) -> "ForkFromSelectedCheckpoint":
        _validate_digest(self.source_execution_hash, "/dependencies/source_execution_hash")
        _validate_digest(self.checkpoint_root_hash, "/dependencies/checkpoint_root_hash")
        return self


class ContinuationReconciliation(StrictModel):
    kind: Literal["continuation_reconciliation"] = "continuation_reconciliation"
    source_completed_batches: int = Field(ge=0)
    additional_batches: int = Field(gt=0)
    expected_target_total: int = Field(gt=0)

    @model_validator(mode="after")
    def _total(self) -> "ContinuationReconciliation":
        total = self.source_completed_batches + self.additional_batches
        if total != self.expected_target_total:
            raise ValueError(f"/dependencies target total drift: computed={total}")
        return self


class LineageGraftDependency(StrictModel):
    kind: Literal["lineage_graft"] = "lineage_graft"
    lineage_event_hash: str
    interpretation: Literal["supersedes_for_interpretation", "new_execution"]

    @model_validator(mode="after")
    def _hash(self) -> "LineageGraftDependency":
        _validate_digest(self.lineage_event_hash, "/dependencies/lineage_event_hash")
        return self


class StoppedRowStatus(StrictModel):
    kind: Literal["stopped_row"] = "stopped_row"
    row_id: str
    completed_batches: int = Field(ge=0)
    reason: str
    checkpoint_root_hash: str | None = None

    @model_validator(mode="after")
    def _hash(self) -> "StoppedRowStatus":
        if self.checkpoint_root_hash is not None:
            _validate_digest(self.checkpoint_root_hash, "/dependencies/checkpoint_root_hash")
        return self


class TaskIdentityGate(StrictModel):
    kind: Literal["task_identity_gate"] = "task_identity_gate"
    identity_kind: Literal["training_run_spec", "method_payload", "task", "dataset"]
    expected_identity_hash: str

    @model_validator(mode="after")
    def _hash(self) -> "TaskIdentityGate":
        _validate_digest(self.expected_identity_hash, "/dependencies/expected_identity_hash")
        return self


ExecutionDependency: TypeAlias = Annotated[
    ForkFromSelectedCheckpoint
    | ContinuationReconciliation
    | LineageGraftDependency
    | StoppedRowStatus
    | TaskIdentityGate,
    Field(discriminator="kind"),
]


class MatrixForkSpec(StrictModel):
    """Fork-from-source-checkpoint launch semantics for a matrix."""

    source_run_id: str | None = None
    lr_continuation: Literal["continue", "restart"]
    parity: Literal["require", "skip"] = "require"
    expected_slots: list[str] = Field(default_factory=list)


class TrainingRunMatrixSpec(StrictModel):
    """Durable governed authoring contract for a matrix of training runs."""

    schema_id: str = TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID
    schema_version: str = TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION
    name: str
    issue: str | None = None
    base: MatrixBaseSpec
    deltas: list[MatrixCompositionDelta] = Field(default_factory=list)
    execution_dependencies: list[ExecutionDependency] = Field(default_factory=list)
    sources: list[SourceBinding] = Field(default_factory=list)
    derivations: list[MatrixDerivation] = Field(default_factory=list)
    rows: list[MatrixRow] = Field(default_factory=list)
    axes: list[TrainingSweepAxis] = Field(default_factory=list)
    combination: TrainingSweepCombinationSpec = Field(default_factory=TrainingSweepCombinationSpec)
    fork: MatrixForkSpec | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_spec(self) -> "TrainingRunMatrixSpec":
        if self.schema_id != TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID:
            raise ValueError(
                f"/schema_id unsupported TrainingRunMatrixSpec schema_id {self.schema_id!r}; "
                f"expected {TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID!r}"
            )
        if self.schema_version != TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION:
            raise ValueError(
                "/schema_version unsupported TrainingRunMatrixSpec schema_version "
                f"{self.schema_version!r}; expected {TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION!r}"
            )
        if not self.name.strip():
            raise ValueError("/name must not be empty")
        if bool(self.rows) == bool(self.axes):
            raise ValueError("/rows and /axes are mutually exclusive and one is required")
        row_ids = [row.row_id for row in self.rows]
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("/rows row_id values must be unique")
        axis_ids = [axis.id for axis in self.axes]
        if len(set(axis_ids)) != len(axis_ids):
            raise ValueError("/axes axis ids must be unique")
        for axis in self.axes:
            if axis.path not in {"seed", "master_prng_key", "prng_key"}:
                _validate_dotted_path(axis.path, f"/axes/{axis.id}/path")
        output_paths = [derivation.output_path for derivation in self.derivations]
        if len(set(output_paths)) != len(output_paths):
            raise ValueError("/derivations output_path values must not collide")
        aliases = [source.alias for source in self.sources]
        if len(set(aliases)) != len(aliases):
            raise ValueError("/sources aliases must be unique")
        if self.derivations and not self.sources:
            for derivation in self.derivations:
                if not _query_can_evaluate_without_sources(derivation.query):
                    raise ValueError("/derivations require /sources unless all queries have defaults")
        return self


def apply_override_patches(
    payload: dict[str, Any],
    patches: list[OverridePatch | dict[str, Any]],
) -> dict[str, Any]:
    """Apply ``OverridePatch`` records to a deep copy of ``payload``."""
    result = deepcopy(payload)
    for raw_patch in patches:
        patch = (
            raw_patch
            if isinstance(raw_patch, OverridePatch)
            else OverridePatch.model_validate(raw_patch)
        )
        _apply_patch(result, patch)
    return result


def apply_composition_deltas(
    payload: dict[str, Any],
    deltas: list[MatrixCompositionDelta],
    *,
    ancestor_written_paths: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, str], set[str]]:
    """Apply ordered layers and fail closed on unacknowledged ancestor overrides."""
    result = deepcopy(payload)
    written = set(ancestor_written_paths or ())
    attribution: dict[str, str] = {}
    current_schema = _payload_schema_identity(result)
    for delta in deltas:
        acknowledged = set(delta.acknowledges_ancestor_paths)
        declared_boundary = (delta.schema_id, delta.schema_version)
        prior_identities = _schema_identities(result)
        for patch in delta.patches:
            if patch.path in written and patch.path not in acknowledged:
                raise ValueError(
                    f"/deltas/{delta.layer_id}/{patch.path} overrides an ancestor-written "
                    "path without explicit acknowledgement"
                )
            try:
                _apply_patch(result, patch)
            except ValueError as error:
                raise ValueError(f"/deltas/{delta.layer_id}: {error}") from error
            written.add(patch.path)
            attribution[patch.path] = delta.layer_id
        resulting_schema = _payload_schema_identity(result)
        resulting_identities = _schema_identities(result)
        if declared_boundary != (None, None):
            if declared_boundary == current_schema:
                raise ValueError(
                    f"/deltas/{delta.layer_id} declares schema boundary "
                    f"{declared_boundary!r} but does not change the active identity"
                )
            if resulting_schema != declared_boundary:
                raise ValueError(
                    f"/deltas/{delta.layer_id} declares schema boundary {declared_boundary!r} "
                    f"but flattened payload has identity {resulting_schema!r}"
                )
        elif resulting_identities != prior_identities:
            raise ValueError(
                f"/deltas/{delta.layer_id} changes schema identity from "
                f"{prior_identities!r} to {resulting_identities!r} without a declared "
                "schema_id/schema_version boundary"
            )
        current_schema = resulting_schema
    return result, attribution, written


def _payload_schema_identity(payload: dict[str, Any]) -> tuple[Any, Any]:
    return payload.get("schema_id"), payload.get("schema_version")


def _schema_identities(value: Any, path: str = "") -> dict[str, tuple[Any, Any]]:
    identities: dict[str, tuple[Any, Any]] = {}
    if isinstance(value, dict):
        if "schema_id" in value or "schema_version" in value:
            identities[path or "/"] = (value.get("schema_id"), value.get("schema_version"))
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            identities.update(_schema_identities(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{path}.{index}" if path else str(index)
            identities.update(_schema_identities(child, child_path))
    return identities


def _validate_digest(value: str, path: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{path} must be a lowercase sha256 digest")


def _apply_patch(root: dict[str, Any], patch: OverridePatch) -> None:
    parts = patch.path.split(".")
    parent = _resolve_parent(root, parts, path=patch.path)
    leaf = parts[-1]
    exists = _contains_key(parent, leaf)
    if patch.op == "add":
        if exists:
            raise ValueError(f"add patch path already exists: {patch.path!r}")
        _set_child(parent, leaf, deepcopy(patch.value), path=patch.path)
        return
    if patch.op == "replace":
        if not exists:
            raise ValueError(
                f"replace patch cannot set missing field {leaf!r}: {patch.path!r}"
            )
        _set_child(parent, leaf, deepcopy(patch.value), path=patch.path)
        return
    if not exists:
        raise ValueError(f"remove patch path is missing: {patch.path!r}")
    _remove_child(parent, leaf, path=patch.path)


def _resolve_parent(root: Any, parts: list[str], *, path: str) -> Any:
    if not parts:
        raise ValueError(f"patch path must be non-empty: {path!r}")
    current = root
    for part in parts[:-1]:
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if 0 <= index < len(current):
                current = current[index]
                continue
        raise ValueError(f"patch path cannot traverse missing segment {part!r}: {path!r}")
    return current


def _contains_key(parent: Any, key: str) -> bool:
    if isinstance(parent, dict):
        return key in parent
    if isinstance(parent, list) and key.isdigit():
        index = int(key)
        return 0 <= index < len(parent)
    return False


def _set_child(parent: Any, key: str, value: Any, *, path: str) -> None:
    if isinstance(parent, dict):
        parent[key] = value
        return
    if isinstance(parent, list) and key.isdigit():
        index = int(key)
        if 0 <= index < len(parent):
            parent[index] = value
            return
    raise ValueError(f"patch path cannot set segment {key!r}: {path!r}")


def _remove_child(parent: Any, key: str, *, path: str) -> None:
    if isinstance(parent, dict):
        del parent[key]
        return
    if isinstance(parent, list) and key.isdigit():
        index = int(key)
        if 0 <= index < len(parent):
            del parent[index]
            return
    raise ValueError(f"patch path cannot remove segment {key!r}: {path!r}")


def _validate_dotted_path(path: str, field_path: str) -> None:
    if not path.strip() or any(not part for part in path.split(".")):
        raise ValueError(f"{field_path} is not dotted-path-like: {path!r}")


def _query_can_evaluate_without_sources(query: ValueExpr) -> bool:
    if isinstance(query, ValueQuery):
        return "default" in query.model_fields_set
    if isinstance(query, Coalesce):
        return query.default is not None or all(
            "default" in child.model_fields_set for child in query.queries
        )
    return False
