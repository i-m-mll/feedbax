"""Domain-independent contracts and expansion for base/row/delta matrices."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any, Generic, TypeVar

from pydantic import Field, model_validator

from feedbax.contracts.expressions import ContextItem, ExpressionContext, ValueExpr, evaluate_query
from feedbax.contracts.extraction import SourceBinding, load_expression_context, set_dotted_path
from feedbax.contracts.manifest import OverridePatch, StrictModel


PayloadT = TypeVar("PayloadT", bound=StrictModel)
_PATH_SAFE_RE = re.compile(r"^[A-Za-z0-9._-]+$")


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


__all__ = [
    "MaterializedMatrixRow",
    "MatrixRow",
    "RowDerivation",
    "RowMatrixSpec",
    "derive_row_path",
    "materialize_matrix_rows",
]
