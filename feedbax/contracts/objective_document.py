"""Authored Objective documents and their lowering for execution."""

from enum import StrEnum

from pydantic import Field

from feedbax.contracts.expression_values import (
    Expression,
    LiteralValue,
    ValueBase,
    ValueRecord,
    ValueType,
)
from feedbax.contracts.stored_members import StoredMembers, keyed_members, project_member
from feedbax.contracts.window import Window


OBJECTIVE_DOCUMENT_SCHEMA_ID = "feedbax.spec.objective_document"
STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID = "feedbax.spec.objective_document.v2"


class Reduction(StrEnum):
    MEAN = "mean"
    SUM = "sum"
    LAST = "last"
    MAX = "max"


class _TermFields(ValueRecord):
    measure: Expression
    measure_type: ValueType = ValueType(base=ValueBase.REAL)
    weight: Expression = LiteralValue(value=1)
    reduction: Reduction = Reduction.MEAN
    window: Window | None = None
    enabled: bool = True


class _TermIdentity(ValueRecord):
    id: str


class ObjectiveTerm(_TermFields, _TermIdentity):
    name: str | None = Field(default=None, min_length=1, exclude_if=lambda value: value is None)


class StoredObjectiveTerm(_TermFields):
    label: str | None = Field(default=None, min_length=1, exclude_if=lambda value: value is None)


class ObjectiveDocument(ValueRecord):
    id: str
    terms: tuple[ObjectiveTerm, ...] = ()


class StoredObjectiveDocument(ValueRecord):
    """An ordered map of Objective terms addressed by stable key."""

    id: str
    terms: StoredMembers[StoredObjectiveTerm] = Field(default_factory=dict)

    def lower(self) -> ObjectiveDocument:
        return project_member(self, ObjectiveDocument, {
            "terms": lambda terms: tuple(
                project_member(term, ObjectiveTerm, id=key, name=term.label) for key, term in terms.items()
            ),
        })

    @classmethod
    def from_lowered(cls, objective: ObjectiveDocument) -> "StoredObjectiveDocument":
        return project_member(objective, cls, {
            "terms": lambda terms: keyed_members(
                (term.id, project_member(term, StoredObjectiveTerm, label=term.name)) for term in terms
            ),
        })
