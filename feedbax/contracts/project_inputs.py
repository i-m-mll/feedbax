"""Authored input references for project steps."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Annotated, Literal

from pydantic import (
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from feedbax.contracts.base import StrictModel
from feedbax.contracts.request_refs import CheckpointBatches, CheckpointCadence, Name, UpdateRule
from feedbax.contracts.selection import ManifestPredicate

# These names serve as file stems and path segments in project-level patches.
ProjectName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")]
RowKey = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_=-]*$")]
NonemptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Best(StrictModel):
    by: NonemptyText
    k: int = Field(ge=1)
    order: Literal["min", "max"] = "max"
    per: list[ProjectName] = Field(default_factory=list)


class When(StrictModel):
    metric: NonemptyText
    op: Literal["lt", "le", "eq", "ne", "ge", "gt"]
    value: float


class SelectedRow(StrictModel):
    """A saved input-member key and its exact producer address within a step."""

    key: Name
    variant: RowKey
    row: RowKey


def producer_member_key(variant: str, row: str, *, gathers_variants: bool) -> str:
    return f"{variant}/{row}" if gathers_variants else row


class SelectionTerms(StrictModel):
    axes: dict[ProjectName, list[ProjectName]] = Field(default_factory=dict)
    best: Best | None = None
    when: When | None = None


def _require_row_list(rows: object) -> object:
    if rows is None:
        raise ValueError("rows must be a list when supplied")
    return rows


class SubsetTerms(SelectionTerms):
    rows: list[SelectedRow] | None = Field(default=None, exclude_if=lambda rows: rows is None)

    _require_rows = field_validator(
        "rows", mode="before", json_schema_input_type=list[SelectedRow]
    )(_require_row_list)

    @field_validator("rows")
    @classmethod
    def _distinct_keys(cls, rows: list[SelectedRow] | None) -> list[SelectedRow] | None:
        if rows is not None and len({row.key for row in rows}) != len(rows):
            raise ValueError("row member keys must be distinct")
        return rows


class SweepExceptionTerms(SelectionTerms):
    rows: list[RowKey] | None = Field(default=None, exclude_if=lambda rows: rows is None)

    _require_rows = field_validator(
        "rows", mode="before", json_schema_input_type=list[RowKey]
    )(_require_row_list)


class StoreSubset(SubsetTerms):
    query: ManifestPredicate


class NamedSubset(StrictModel):
    subset: ProjectName


class SubsetSelectionError(ValueError):
    """A refusal at a path relative to the subset terms."""

    def __init__(self, path: str, detail: str) -> None:
        self.path = path
        self.detail = detail
        super().__init__(f"{path}: {detail}" if path else detail)


def select_step_rows[Row](
    terms: SubsetTerms | None,
    rows: Iterable[Row],
    *,
    address: Callable[[Row], tuple[str, str]],
    member_key: Callable[[Row], str],
    coordinates: Callable[[Row], Mapping[str, str]],
) -> tuple[tuple[str, Row], ...]:
    """Select keyed addresses in saved order or select axis matches in Plan order."""
    members = tuple(rows)
    if terms is None:
        return tuple((member_key(row), row) for row in members)
    if isinstance(terms, StoreSubset):
        raise SubsetSelectionError("query", "store queries cannot select static step rows")
    matches = _axis_matches(terms.axes, members, coordinates=coordinates)
    if terms.rows is None:
        return tuple((member_key(row), row) for row, match in zip(members, matches, strict=True) if match)
    indices = {address(row): index for index, row in enumerate(members)}
    selected = []
    for row in terms.rows:
        where = row.variant, row.row
        if where not in indices:
            raise SubsetSelectionError("rows", f"unknown row address {row.variant}/{row.row}")
        index = indices[where]
        if matches[index]:
            selected.append((row.key, members[index]))
    return tuple(selected)


def select_sweep_exception_rows[Row](
    terms: SweepExceptionTerms,
    rows: Iterable[Row],
    *,
    key: Callable[[Row], str],
    coordinates: Callable[[Row], Mapping[str, str]],
) -> tuple[Row, ...]:
    """Intersect local row IDs and axis terms while retaining sweep order."""
    for measure in ("best", "when"):
        if getattr(terms, measure) is not None:
            raise SubsetSelectionError(measure, "measure terms require bound results")
    members = tuple(rows)
    keys = tuple(map(key, members))
    if terms.rows is not None:
        selected = set(terms.rows)
        missing = sorted(selected.difference(keys))
        if missing:
            raise SubsetSelectionError("rows", f"unknown row IDs {missing!r}")
    else:
        selected = set(keys)
    matches = _axis_matches(terms.axes, members, coordinates=coordinates)
    return tuple(
        row for row, row_id, match in zip(members, keys, matches, strict=True)
        if match and row_id in selected
    )


def _axis_matches[Row](
    axes: Mapping[str, list[str]],
    members: tuple[Row, ...],
    *,
    coordinates: Callable[[Row], Mapping[str, str]],
) -> tuple[bool, ...]:
    if not axes:
        return (True,) * len(members)
    member_coordinates = tuple(map(coordinates, members))
    allowed = {axis: set(values) for axis, values in axes.items()}
    known: dict[str, set[str]] = {}
    for coordinate in member_coordinates:
        for axis, value in coordinate.items():
            if axis in allowed:
                known.setdefault(axis, set()).add(value)
    for axis, values in allowed.items():
        if axis not in known:
            raise SubsetSelectionError(f"axes.{axis}", f"unknown axis ID {axis!r}")
        missing = sorted(values - known[axis])
        if missing:
            raise SubsetSelectionError(f"axes.{axis}", f"unknown value IDs {missing!r}")

    return tuple(
        all(coordinate.get(axis) in values for axis, values in allowed.items())
        for coordinate in member_coordinates
    )


def subset_to_rows(terms: SubsetTerms, bound_rows: Iterable[SelectedRow]) -> SubsetTerms:
    """Replace a measured step subset with its exact keyed rows."""
    if isinstance(terms, StoreSubset):
        raise SubsetSelectionError("query", "store subsets do not bind step row IDs")
    if terms.best is None and terms.when is None:
        raise SubsetSelectionError("", "To Rows requires a measure term")
    return SubsetTerms(rows=list(bound_rows))


class BestCheckpoint(StrictModel):
    best_by: NonemptyText


class BatchCheckpoint(StrictModel):
    at: int = Field(ge=0)


class BarrierCheckpoint(StrictModel):
    barrier: NonemptyText


type CheckpointRule = (
    Literal["last"] | BestCheckpoint | BatchCheckpoint | BarrierCheckpoint
    | CheckpointBatches | CheckpointCadence | UpdateRule
)

class StepReference(StrictModel):
    step: ProjectName
    where: SubsetTerms | NamedSubset | None = None
    row: RowKey | None = None
    variants: Literal["all"] | list[RowKey] | None = None
    checkpoint: CheckpointRule | None = None
    element: ProjectName | None = Field(default=None, exclude_if=lambda value: value is None)
    product: ProjectName | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def _check_row_selector(self) -> StepReference:
        if self.row is not None and self.where is not None:
            raise ValueError("row is the shorthand for a single-row subset")
        if (self.element is None) != (self.product is None):
            raise ValueError("element and product must be declared together")
        return self


class RunReference(StrictModel):
    run: NonemptyText


class OmittedReference(StrictModel):
    not_applicable: NonemptyText


type Reference = StepReference | RunReference | StoreSubset | NamedSubset | OmittedReference
type InputBinding = Reference | list[Reference] | dict[ProjectName, Reference]
