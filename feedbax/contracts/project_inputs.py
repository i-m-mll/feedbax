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
from feedbax.contracts.run_rule import RuleRefusal, SweepCoordinate, choose, parse
from feedbax.expression_text import RuleBest
from feedbax.contracts.selection import ManifestPredicate

# These names serve as file stems and path segments in project-level patches.
ProjectName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")]
RowKey = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_=-]*$")]
AxisValueId = Name
"""An axis value's ID: a value's name, or an input axis's member key (``seed=0``, ``variant/row``)."""
NonemptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
type Pointer = Annotated[str, StringConstraints(pattern=r"^(?:/(?:[^~/]|~[01])*)*$")]


class DocumentReference(StrictModel):
    document: ProjectName
    pointer: Pointer | None = None


class InitialModelReference[DocumentT](StrictModel):
    kind: Literal["initial_model"]
    source: DocumentT
    seed: int = Field(default=0, ge=0, lt=2**32, strict=True)


class TaskDocumentReference[DocumentT](StrictModel):
    kind: Literal["task"]
    source: DocumentT


class ObjectiveDocumentReference[DocumentT](StrictModel):
    kind: Literal["objective"]
    source: DocumentT


type DocumentInputReference = Annotated[
    InitialModelReference[DocumentReference] | TaskDocumentReference[DocumentReference]
    | ObjectiveDocumentReference[DocumentReference], Field(discriminator="kind"),
]


class SelectedRow(StrictModel):
    """A saved input-member key and its exact producer address within a step."""

    key: Name
    variant: RowKey
    row: RowKey


def producer_member_key(variant: str, row: str, *, gathers_variants: bool) -> str:
    return f"{variant}/{row}" if gathers_variants else row


def _require_row_list(rows: object) -> object:
    if rows is None:
        raise ValueError("rows must be a list when supplied")
    return rows


class SubsetTerms(StrictModel):
    """Which runs of a step a selection reads: its run rule as written, and the runs To Rows pinned."""

    rule: str = ""
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


class SweepExceptionTerms(StrictModel):
    """Which rows of a sweep an exception applies to: a run rule on the step's axes, within ``rows`` when given."""

    rule: str = ""
    rows: list[RowKey] | None = Field(default=None, exclude_if=lambda rows: rows is None)

    _require_rows = field_validator(
        "rows", mode="before", json_schema_input_type=list[RowKey]
    )(_require_row_list)


class StoreSubset(SubsetTerms):
    query: ManifestPredicate


class NamedSubset(StrictModel):
    subset: ProjectName


def select_step_rows[Row](
    terms: SubsetTerms | None,
    rows: Iterable[Row],
    *,
    address: Callable[[Row], tuple[str, str]],
    member_key: Callable[[Row], str],
    coordinates: Callable[[Row], Mapping[str, SweepCoordinate]],
    producer: str,
) -> tuple[tuple[tuple[str, Row], ...], RuleBest | None]:
    """The keyed rows the rule chooses, in Plan order or in pinned order, and its Best term while it waits."""
    members = tuple(rows)
    if terms is None:
        return tuple((member_key(row), row) for row in members), None
    if isinstance(terms, StoreSubset):
        raise RuleRefusal("query", "store queries cannot select static step rows")
    chosen = choose(terms.rule, members, coordinates=coordinates, producer=producer)
    if terms.rows is None:
        return tuple((member_key(row), row) for row in chosen.rows), chosen.best
    picked = set(map(id, chosen.rows))
    indices = {address(row): index for index, row in enumerate(members)}
    selected = []
    for row in terms.rows:
        where = row.variant, row.row
        if where not in indices:
            raise RuleRefusal("rows", f"unknown row address {row.variant}/{row.row}")
        member = members[indices[where]]
        if id(member) in picked:
            selected.append((row.key, member))
    return tuple(selected), None


def select_sweep_exception_rows[Row](
    terms: SweepExceptionTerms,
    rows: Iterable[Row],
    *,
    key: Callable[[Row], str],
    coordinates: Callable[[Row], Mapping[str, SweepCoordinate]],
    producer: str,
) -> tuple[Row, ...]:
    """The rows among ``rows`` (when given) that the rule's axis terms choose, in sweep order."""
    members = tuple(rows)
    keys = tuple(map(key, members))
    if terms.rows is not None:
        selected = set(terms.rows)
        missing = sorted(selected.difference(keys))
        if missing:
            raise RuleRefusal("rows", f"unknown row IDs {missing!r}")
    else:
        selected = set(keys)
    chosen = set(map(id, choose(terms.rule, members, coordinates=coordinates, producer=producer, ranked=False).rows))
    return tuple(row for row, row_id in zip(members, keys, strict=True) if id(row) in chosen and row_id in selected)


def subset_to_rows(terms: SubsetTerms, bound_rows: Iterable[SelectedRow]) -> SubsetTerms:
    """Pin a measured step subset to its exact keyed rows beside its rule."""
    if isinstance(terms, StoreSubset):
        raise RuleRefusal("query", "store subsets do not bind step row IDs")
    if not any(isinstance(term, RuleBest) for term in parse(terms.rule)):
        raise RuleRefusal("rule", "To Rows requires a lowest or highest term")
    return SubsetTerms(rule=terms.rule, rows=list(bound_rows))


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


type Reference = StepReference | RunReference | StoreSubset | NamedSubset | OmittedReference | DocumentInputReference
type InputBinding = Reference | list[StepReference]
