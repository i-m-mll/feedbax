"""Domain-independent contracts and expansion for base/row/delta matrices."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
import json
import math
from pathlib import Path
import re
from typing import Any, Generic, TypeVar

from pydantic import Field, model_validator

from feedbax.contracts.expressions import ContextItem, ExpressionContext, ValueExpr, evaluate_query
from feedbax.contracts.extraction import SourceBinding, load_expression_context, set_dotted_path
from feedbax.contracts.manifest import (
    OverridePatch,
    StrictModel,
    canonical_json_bytes,
    sha256_bytes,
)


PayloadT = TypeVar("PayloadT", bound=StrictModel)
_PATH_SAFE_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContentPinnedJsonBase(StrictModel):
    """Relative JSON document whose canonical content hash is authoritative."""

    ref: str
    sha256: str

    @model_validator(mode="after")
    def _validate_base(self) -> "ContentPinnedJsonBase":
        if not self.ref.strip() or Path(self.ref).is_absolute():
            raise ValueError("content-pinned JSON base ref must be a non-empty relative path")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("content-pinned JSON base sha256 must be lowercase hexadecimal")
        return self


class MatrixAxisValue(StrictModel):
    """One named value on an authored matrix axis."""

    id: str
    label: str | None = None
    deltas: list[OverridePatch] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_value(self) -> "MatrixAxisValue":
        _validate_path_safe_id(self.id, "axis value id")
        paths = [delta.path for delta in self.deltas]
        if len(paths) != len(set(paths)):
            raise ValueError(f"axis value {self.id!r} has duplicate delta paths")
        for delta in self.deltas:
            if delta.op != "remove":
                _require_json_value(delta.value, f"axis value {self.id!r} delta {delta.path!r}")
        return self


class MatrixAxis(StrictModel):
    """One ordered authored axis with ordered named values."""

    id: str
    label: str | None = None
    values: list[MatrixAxisValue] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_axis(self) -> "MatrixAxis":
        _validate_path_safe_id(self.id, "axis id")
        value_ids = [value.id for value in self.values]
        if len(value_ids) != len(set(value_ids)):
            raise ValueError(f"axis {self.id!r} value ids must be unique")
        return self


class MatrixAxisCoordinate(StrictModel):
    """One deterministic coordinate selected from an ordered axis product."""

    row_id: str
    value_indices: dict[str, int]
    value_ids: dict[str, str]
    deltas: list[OverridePatch] = Field(default_factory=list)


class RowDerivation(StrictModel):
    """A value derived after a row's ordered deltas have been applied."""

    output_path: str
    query: ValueExpr

    @model_validator(mode="after")
    def _validate_output_path(self) -> "RowDerivation":
        _validate_dotted_path(self.output_path, "derivations.output_path")
        return self


class MatrixRow(StrictModel):
    """One named condition row with ordered typed deltas."""

    row_id: str
    label: str | None = None
    deltas: list[OverridePatch] = Field(default_factory=list)
    derivations: list[RowDerivation] = Field(default_factory=list)
    output_path: str | None = None
    spec_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_row(self) -> "MatrixRow":
        if not _PATH_SAFE_RE.fullmatch(self.row_id):
            raise ValueError(f"row.row_id is not path-safe: {self.row_id!r}")
        paths = [derivation.output_path for derivation in self.derivations]
        if len(paths) != len(set(paths)):
            raise ValueError("row.derivations output_path values must be unique")
        for field_name in ("output_path", "spec_path"):
            value = getattr(self, field_name)
            if value is not None and Path(value).is_absolute():
                raise ValueError(f"row.{field_name} must be relative")
        return self


class RowMatrixSpec(StrictModel, Generic[PayloadT]):
    """Generic base plus condition rows and external derivation sources."""

    base: PayloadT
    rows: list[MatrixRow] = Field(min_length=1)
    sources: list[SourceBinding] = Field(default_factory=list)
    derivations: list[RowDerivation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_matrix(self) -> "RowMatrixSpec[PayloadT]":
        row_ids = [row.row_id for row in self.rows]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError("rows row_id values must be unique")
        aliases = [source.alias for source in self.sources]
        if len(aliases) != len(set(aliases)):
            raise ValueError("sources aliases must be unique")
        paths = [derivation.output_path for derivation in self.derivations]
        if len(paths) != len(set(paths)):
            raise ValueError("derivations output_path values must be unique")
        return self


class MaterializedMatrixRow(StrictModel, Generic[PayloadT]):
    """One fully resolved row and its deterministic relative paths."""

    row_id: str
    payload: PayloadT
    output_path: str
    spec_path: str


def load_content_pinned_json_base(
    base: ContentPinnedJsonBase,
    *,
    repo_root: Path | str | None,
) -> dict[str, Any]:
    """Load and verify one canonical-JSON content-pinned object."""
    if repo_root is None:
        raise ValueError("content-pinned JSON base requires repo_root")
    root = Path(repo_root).resolve()
    path = (root / base.ref).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"content-pinned JSON base escapes repo_root: {base.ref!r}") from exc
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load content-pinned JSON base {base.ref!r}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("content-pinned JSON base must contain a JSON object")
    _require_json_value(payload, "content-pinned JSON base")
    actual_sha256 = sha256_bytes(canonical_json_bytes(payload))
    if actual_sha256 != base.sha256:
        raise ValueError(
            f"content-pinned JSON base hash mismatch for {base.ref!r}: "
            f"expected {base.sha256}, got {actual_sha256}"
        )
    return payload


def ordered_index_product(
    axis_lengths: Sequence[tuple[str, int]],
) -> list[dict[str, int]]:
    """Return a stable Cartesian product in authored axis order."""
    if not axis_lengths:
        raise ValueError("ordered index product requires at least one axis")
    axis_ids = [axis_id for axis_id, _ in axis_lengths]
    if len(axis_ids) != len(set(axis_ids)):
        raise ValueError("ordered index product axis ids must be unique")
    for axis_id, length in axis_lengths:
        _validate_path_safe_id(axis_id, "axis id")
        if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
            raise ValueError(f"ordered index product axis {axis_id!r} must be non-empty")

    coordinates: list[dict[str, int]] = [{}]
    for axis_id, length in axis_lengths:
        coordinates = [
            {**coordinate, axis_id: index}
            for coordinate in coordinates
            for index in range(length)
        ]
    return coordinates


def expand_matrix_axes(axes: Sequence[MatrixAxis]) -> list[MatrixAxisCoordinate]:
    """Expand ordered authored axes into canonical row coordinates and deltas."""
    axis_ids = [axis.id for axis in axes]
    if len(axis_ids) != len(set(axis_ids)):
        raise ValueError("matrix axis ids must be unique")
    indexed = ordered_index_product([(axis.id, len(axis.values)) for axis in axes])
    coordinates: list[MatrixAxisCoordinate] = []
    row_ids: set[str] = set()
    expected = set(axis_ids)
    for indices in indexed:
        if set(indices) != expected:
            raise ValueError("ordered axis product produced an incomplete coordinate")
        selected = [axis.values[indices[axis.id]] for axis in axes]
        value_ids = {axis.id: value.id for axis, value in zip(axes, selected)}
        row_id = "--".join(
            f"{axis.id}-{value.id}" for axis, value in zip(axes, selected)
        )
        _validate_path_safe_id(row_id, "generated row id")
        if row_id in row_ids:
            raise ValueError(f"generated matrix row_id collision: {row_id!r}")
        row_ids.add(row_id)
        deltas = [delta for value in selected for delta in value.deltas]
        paths = [delta.path for delta in deltas]
        duplicates = sorted({path for path in paths if paths.count(path) > 1})
        if duplicates:
            raise ValueError(
                f"matrix coordinate {row_id!r} selects duplicate delta paths {duplicates!r}"
            )
        coordinates.append(
            MatrixAxisCoordinate(
                row_id=row_id,
                value_indices=dict(indices),
                value_ids=value_ids,
                deltas=deltas,
            )
        )
    return coordinates


def derive_row_path(
    row_id: str,
    *,
    explicit_path: str | None = None,
    suffix: str = ".json",
) -> str:
    """Derive a relative path from ``row_id``, unless an explicit path is supplied."""
    if not _PATH_SAFE_RE.fullmatch(row_id):
        raise ValueError(f"row_id is not path-safe: {row_id!r}")
    path = explicit_path if explicit_path is not None else f"{row_id}{suffix}"
    if Path(path).is_absolute():
        raise ValueError("row path must be relative")
    return path


def materialize_matrix_rows(
    spec: RowMatrixSpec[PayloadT],
    *,
    repo_root: Path | str | None = None,
) -> list[MaterializedMatrixRow[PayloadT]]:
    """Apply each row's deltas, then evaluate its derivations against that row."""
    if spec.sources and repo_root is None:
        raise ValueError("matrix sources require repo_root")
    source_context = (
        load_expression_context(spec.sources, repo_root) if spec.sources else ExpressionContext()
    )
    materialized: list[MaterializedMatrixRow[PayloadT]] = []
    for row in spec.rows:
        payload = _apply_deltas(spec.base.model_dump(mode="python"), row.deltas)
        for derivation in [*spec.derivations, *row.derivations]:
            context = ExpressionContext(
                items={
                    **source_context.items,
                    "row": ContextItem(kind="matrix_row", payload=deepcopy(payload)),
                }
            )
            value = evaluate_query(derivation.query, context)
            set_dotted_path(payload, derivation.output_path, value)
        materialized.append(
            MaterializedMatrixRow[spec.base.__class__](
                row_id=row.row_id,
                payload=spec.base.__class__.model_validate(payload),
                output_path=derive_row_path(
                    row.row_id, explicit_path=row.output_path, suffix="/output.json"
                ),
                spec_path=derive_row_path(
                    row.row_id, explicit_path=row.spec_path, suffix="/spec.json"
                ),
            )
        )
    return materialized


def _apply_deltas(payload: dict[str, Any], deltas: list[OverridePatch]) -> dict[str, Any]:
    # Keep this core independent from the training matrix module while reusing
    # the shared OverridePatch contract.
    result = deepcopy(payload)
    for delta in deltas:
        parts = delta.path.split(".")
        parent: Any = result
        for part in parts[:-1]:
            if isinstance(parent, dict) and part in parent:
                parent = parent[part]
            elif isinstance(parent, list) and part.isdigit() and int(part) < len(parent):
                parent = parent[int(part)]
            else:
                raise ValueError(
                    f"delta path cannot traverse missing segment {part!r}: {delta.path!r}"
                )
        leaf = parts[-1]
        exists = (
            leaf in parent
            if isinstance(parent, dict)
            else leaf.isdigit() and int(leaf) < len(parent)
        )
        if delta.op == "add" and exists:
            raise ValueError(f"add delta path already exists: {delta.path!r}")
        if delta.op in {"replace", "remove"} and not exists:
            raise ValueError(f"{delta.op} delta path is missing: {delta.path!r}")
        if delta.op == "remove":
            del parent[leaf if isinstance(parent, dict) else int(leaf)]
        else:
            parent[leaf if isinstance(parent, dict) else int(leaf)] = deepcopy(delta.value)
    return result


def _validate_dotted_path(path: str, field: str) -> None:
    if not path.strip() or any(not part for part in path.split(".")):
        raise ValueError(f"{field} is not dotted-path-like: {path!r}")


def _validate_path_safe_id(value: str, field: str) -> None:
    if not _PATH_SAFE_RE.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"{field} is not path-safe: {value!r}")


def _require_json_value(value: Any, field: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} must contain finite JSON numbers")
        return
    if isinstance(value, list):
        for item in value:
            _require_json_value(item, field)
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{field} must contain only string JSON object keys")
        for item in value.values():
            _require_json_value(item, field)
        return
    raise ValueError(f"{field} contains a non-JSON value of type {type(value).__name__}")


__all__ = [
    "ContentPinnedJsonBase",
    "MaterializedMatrixRow",
    "MatrixAxis",
    "MatrixAxisCoordinate",
    "MatrixAxisValue",
    "MatrixRow",
    "RowDerivation",
    "RowMatrixSpec",
    "derive_row_path",
    "expand_matrix_axes",
    "load_content_pinned_json_base",
    "materialize_matrix_rows",
    "ordered_index_product",
]
