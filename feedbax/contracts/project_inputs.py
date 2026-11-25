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
from feedbax.contracts.expression_values import UnresolvedValue
from feedbax.contracts.request_refs import CheckpointBatches, CheckpointCadence, Name, UpdateRule
from feedbax.contracts.run_rule import RuleRefusal, SweepCoordinate, choose, parse
from feedbax.contracts.expression_text import RuleBest, RuleCondition
from feedbax.contracts.selection import ManifestPredicate

# These names serve as file stems and path segments in project-level patches.
ProjectName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")]
RowKey = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_=-]*$")]
AxisValueId = Literal[""] | Name
"""An axis value's ID: a value's name, or an input axis's member key (``seed=0``, ``variant/row``)."""
NonemptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
type Pointer = Annotated[str, StringConstraints(pattern=r"^(?:/(?:[^~/]|~[01])*)*$")]


class DocumentReference(StrictModel):
    document: ProjectName
    pointer: Pointer | None = None
    unresolved: Literal[True] | None = Field(default=None, exclude_if=lambda value: value is None)
    """The original spelling of a missing target whose key was subsequently taken."""


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


class LoadedModelReference(StrictModel):
    kind: Literal["loaded_model"]
    source: DocumentReference


type DocumentInputReference = Annotated[
    InitialModelReference[DocumentReference] | TaskDocumentReference[DocumentReference]
    | ObjectiveDocumentReference[DocumentReference] | LoadedModelReference, Field(discriminator="kind"),
]


class BoundRank(StrictModel):
    """A measured rank settled on an exact producer run by To Rows."""

    key: Name
    variant: RowKey
    row: RowKey


def producer_member_key(variant: str, row: str, *, gathers_variants: bool) -> str:
    return f"{variant}/{row}" if gathers_variants else row


def _require_row_list(rows: object) -> object:
    if rows is None:
        raise ValueError("rows must be a list when supplied")
    return rows


class SelectedAxisValue(StrictModel):
    id: AxisValueId
    label: str
    """The value's display name at selection time, used if the ID subsequently disappears."""


type AxisSelectionClause = Annotated[
    dict[ProjectName, Annotated[list[SelectedAxisValue], Field(min_length=1)]], Field(min_length=1),
]


class SelectedAxisValueProblem(StrictModel):
    clause_index: int = Field(ge=0)
    axis: ProjectName
    missing: SelectedAxisValue
    remaining: list[SelectedAxisValue]


class SelectedAxisValueRefusal(RuleRefusal):
    def __init__(self, index: int, problem: SelectedAxisValueProblem) -> None:
        self.problem = problem
        super().__init__(
            f"clauses.{problem.clause_index}.{problem.axis}.{index}",
            f"{problem.axis} no longer has {problem.missing.label!r}.",
        )


class SubsetTerms(StrictModel):
    """The written rule and selected axis values, with settled rank provenance for To Rows."""

    rule: str = ""
    bound_ranks: list[BoundRank] | None = Field(default=None, exclude_if=lambda ranks: ranks is None)
    clauses: Annotated[list[AxisSelectionClause], Field(min_length=1)] = Field(
        default_factory=list, exclude_if=lambda clauses: not clauses,
    )
    """The union of these clauses. Each clause constrains only its named axes. Omit clauses to leave all axes free."""

    @field_validator("bound_ranks")
    @classmethod
    def _distinct_ranks(cls, ranks: list[BoundRank] | None) -> list[BoundRank] | None:
        if ranks is not None and len({rank.key for rank in ranks}) != len(ranks):
            raise ValueError("bound rank keys must be distinct")
        return ranks

    @field_validator("clauses")
    @classmethod
    def _canonical_clauses(cls, clauses: list[AxisSelectionClause]) -> list[AxisSelectionClause]:
        canonical = {}
        for clause in clauses:
            ordered = {}
            for axis, values in sorted(clause.items()):
                by_id = {}
                for value in sorted(values, key=lambda value: (value.id, value.label)):
                    by_id.setdefault(value.id, value)
                ordered[axis] = list(by_id.values())
            identity = tuple((axis, tuple(value.id for value in values)) for axis, values in ordered.items())
            labels = tuple(value.label for values in ordered.values() for value in values)
            if identity not in canonical or labels < canonical[identity][0]:
                canonical[identity] = (labels, ordered)
        return [canonical[identity][1] for identity in sorted(canonical)]


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
    unresolved: Literal[True] | None = Field(default=None, exclude_if=lambda value: value is None)


def select_step_rows[Row, Coordinate: SweepCoordinate](
    terms: SubsetTerms | None,
    rows: Iterable[Row],
    *,
    address: Callable[[Row], tuple[str, str]],
    member_key: Callable[[Row], str],
    coordinates: Callable[[Row], Mapping[str, Coordinate]],
    coordinate_label: Callable[[str, Coordinate], str],
    missing: Callable[[SelectedAxisValueRefusal], None],
    producer: str,
    final_metrics: Callable[[tuple[Row, ...]], frozenset[str]] | None = None,
) -> tuple[tuple[tuple[str, Row], ...], RuleBest | None, tuple[RuleCondition, ...]]:
    """Select each matching producer row once and report the values of invalid clauses."""
    members = tuple(rows)
    if terms is None:
        return tuple((member_key(row), row) for row in members), None, ()
    if isinstance(terms, StoreSubset):
        raise RuleRefusal("query", "store queries cannot select static step rows")
    chosen_members = members
    if terms.clauses:
        indexed: dict[str, dict[str, tuple[Coordinate, set[int]]]] = {}
        for position, row in enumerate(members):
            for axis, value in coordinates(row).items():
                _, positions = indexed.setdefault(axis, {}).setdefault(value.value_id, (value, set()))
                positions.add(position)
        matched: set[int] = set()
        for clause_index, clause in enumerate(terms.clauses):
            broken = False
            constraint_matches = []
            for axis, wanted in clause.items():
                options = indexed.get(axis, {})
                absent = [(index, value) for index, value in enumerate(wanted) if value.id not in options]
                if absent:
                    broken = True
                    remaining = [
                        SelectedAxisValue(id=option.value_id, label=coordinate_label(axis, option))
                        for option, _ in options.values()
                    ]
                    for index, value in absent:
                        missing(SelectedAxisValueRefusal(index, SelectedAxisValueProblem(
                            clause_index=clause_index, axis=axis, missing=value, remaining=remaining,
                        )))
                else:
                    constraint_matches.append(set.union(*(options[value.id][1] for value in wanted)))
            if not broken:
                matched.update(set.intersection(*constraint_matches))
        chosen_members = tuple(members[position] for position in sorted(matched))
    chosen = choose(terms.rule, chosen_members, coordinates=coordinates, producer=producer, final_metrics=final_metrics)
    if terms.bound_ranks is None:
        return tuple((member_key(row), row) for row in chosen.rows), chosen.best, chosen.conditions
    picked = set(map(id, chosen.rows))
    indices = {address(row): index for index, row in enumerate(members)}
    selected: list[tuple[str, Row]] = []
    for rank in terms.bound_ranks:
        where = rank.variant, rank.row
        if where not in indices:
            raise RuleRefusal("bound_ranks", f"unknown measured member {rank.variant}/{rank.row}")
        member = members[indices[where]]
        if id(member) in picked:
            selected.append((rank.key, member))
    return tuple(selected), None, chosen.conditions


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


def subset_to_rows(terms: SubsetTerms, bound_ranks: Iterable[BoundRank]) -> SubsetTerms:
    """Settle a measured step subset on the producer run chosen for each rank."""
    if isinstance(terms, StoreSubset):
        raise RuleRefusal("query", "store subsets do not bind step row IDs")
    if not any(isinstance(term, RuleBest) for term in parse(terms.rule)):
        raise RuleRefusal("rule", "To Rows requires a lowest or highest term")
    return terms.model_copy(update={"bound_ranks": list(bound_ranks)})


class BestCheckpoint(StrictModel):
    best_by: NonemptyText


class BatchCheckpoint(StrictModel):
    at: int = Field(ge=0)


class BarrierCheckpoint(StrictModel):
    barrier: NonemptyText


type CheckpointRule = (
    Literal["last"] | BestCheckpoint | BatchCheckpoint | BarrierCheckpoint
    | CheckpointBatches | CheckpointCadence | UpdateRule | UnresolvedValue
)

class StepReference(StrictModel):
    step: ProjectName
    where: SubsetTerms | NamedSubset | None = None
    variants: Literal["all"] | list[RowKey] | None = None
    checkpoint: CheckpointRule | None = None
    element: ProjectName | None = Field(default=None, exclude_if=lambda value: value is None)
    product: ProjectName | None = Field(default=None, exclude_if=lambda value: value is None)
    unresolved: Literal[True] | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def _check_product(self) -> StepReference:
        if (self.element is None) != (self.product is None):
            raise ValueError("element and product must be declared together")
        return self


class RunReference(StrictModel):
    run: NonemptyText


class OmittedReference(StrictModel):
    not_applicable: NonemptyText


type Reference = StepReference | RunReference | StoreSubset | NamedSubset | OmittedReference | DocumentInputReference
type InputBinding = Reference | list[StepReference]
