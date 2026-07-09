"""Contracts for governed multi-row training run matrices."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import Field, model_validator

from feedbax.contracts.expressions import Coalesce, ValueExpr, ValueQuery
from feedbax.contracts.extraction import SourceBinding
from feedbax.contracts.manifest import (
    OverridePatch,
    StrictModel,
    TrainingSweepAxis,
    TrainingSweepCombinationSpec,
)


TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID = "feedbax.spec.training_run_matrix"
TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION = "feedbax.spec.training_run_matrix.v1"
_PATH_SAFE_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class MatrixBaseSpec(StrictModel):
    """Location of the base ``TrainingRunSpec`` payload for a matrix."""

    inline: dict[str, Any] | None = None
    ref: str | None = None
    payload_path: str | None = None
    sha256: str | None = None

    @model_validator(mode="after")
    def _validate_base(self) -> "MatrixBaseSpec":
        if (self.inline is None) == (self.ref is None):
            raise ValueError("/base exactly one of inline or ref is required")
        if self.ref is not None and Path(self.ref).is_absolute():
            raise ValueError("/base/ref must be repo-relative")
        if self.payload_path is not None:
            _validate_dotted_path(self.payload_path, "/base/payload_path")
        if self.sha256 is not None and self.ref is None:
            raise ValueError("/base/sha256 is only allowed with /base/ref")
        return self


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
            raise ValueError(f"replace patch path is missing: {patch.path!r}")
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
