"""Authored input references for project steps."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

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
    rows: list[RowKey] = Field(default_factory=list)
    best: Best | None = None
    when: When | None = None


class StoreSubset(SubsetTerms):
    query: ManifestPredicate


class NamedSubset(StrictModel):
    subset: ProjectName


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
    variants: Literal["all"] | list[ProjectName] | None = None
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
