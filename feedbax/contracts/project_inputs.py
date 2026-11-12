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


class SubsetTerms(StrictModel):
    axes: dict[ProjectName, list[ProjectName]] = Field(default_factory=dict)
    rows: list[RowKey] | None = Field(default=None, exclude_if=lambda rows: rows is None)
    best: Best | None = None
    when: When | None = None

    @field_validator("rows", mode="before", json_schema_input_type=list[RowKey])
    @classmethod
    def _require_row_list(cls, rows: object) -> object:
        if rows is None:
            raise ValueError("rows must be a list when supplied")
        return rows


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


def select_static_rows[Row](
    terms: SubsetTerms | None,
    rows: Iterable[Row],
    *,
    key: Callable[[Row], str],
    coordinates: Callable[[Row], Mapping[str, str]],
) -> tuple[Row, ...]:
    """Intersect authored row and axis terms while retaining upstream order."""
    if terms is None:
        return tuple(rows)
    if isinstance(terms, StoreSubset):
        raise SubsetSelectionError("query", "store queries cannot select static step rows")
    for measure in ("best", "when"):
        if getattr(terms, measure) is not None:
            raise SubsetSelectionError(measure, "measure terms require bound results")

    members = tuple(rows)
    if terms.rows is not None:
        selected = set(terms.rows)
        keys = tuple(map(key, members))
        missing = sorted(selected.difference(keys))
        if missing:
            raise SubsetSelectionError("rows", f"unknown row IDs {missing!r}")
    else:
        keys = None
        selected = set()

    if not terms.axes:
        if keys is None:
            return members
        return tuple(row for row, row_id in zip(members, keys, strict=True) if row_id in selected)

    member_coordinates = tuple(map(coordinates, members))
    allowed = {axis: set(values) for axis, values in terms.axes.items()}
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
        row
        for index, (row, coordinate) in enumerate(zip(members, member_coordinates, strict=True))
        if (keys is None or keys[index] in selected)
        and all(coordinate.get(axis) in values for axis, values in allowed.items())
    )


def subset_to_rows(terms: SubsetTerms, bound_rows: Iterable[str]) -> SubsetTerms:
    """Replace a measured step subset with the exact row IDs already bound."""
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


type CheckpointRule = Literal["last"] | BestCheckpoint | BatchCheckpoint | BarrierCheckpoint


class StepReference(StrictModel):
    step: ProjectName
    where: SubsetTerms | NamedSubset | None = None
    row: RowKey | None = None
    variants: Literal["all"] | list[RowKey] | None = None
    checkpoint: CheckpointRule | None = None

    @model_validator(mode="after")
    def _check_row_selector(self) -> StepReference:
        if self.row is not None and self.where is not None:
            raise ValueError("row is the shorthand for a single-row subset")
        return self


class RunReference(StrictModel):
    run: NonemptyText


class OmittedReference(StrictModel):
    not_applicable: NonemptyText


type Reference = StepReference | RunReference | StoreSubset | NamedSubset | OmittedReference
type InputBinding = Reference | list[Reference] | dict[ProjectName, Reference]
