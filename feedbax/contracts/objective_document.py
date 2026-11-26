"""Authored Objective documents and their lowering for execution."""

from enum import StrEnum

from pydantic import Field

from feedbax.contracts.expression_storage import StoredExpression, lower_expression, store_expression
from feedbax.contracts.expression_values import (
    Expression,
    LiteralValue,
    ValueBase,
    ValueRecord,
    ValueType,
)
from feedbax.contracts.stored_members import StoredMembers, keyed_members, project_member
from feedbax.contracts.window import StoredWindow, Window


OBJECTIVE_DOCUMENT_SCHEMA_ID = "feedbax.spec.objective_document"
STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID = "feedbax.spec.objective_document.v2"


class Reduction(StrEnum):
    MEAN = "mean"
    SUM = "sum"
    LAST = "last"
    MAX = "max"


class _TermFields[ExpressionT, WindowT](ValueRecord):
    measure: ExpressionT
    measure_type: ValueType = ValueType(base=ValueBase.REAL)
    weight: ExpressionT
    reduction: Reduction = Reduction.MEAN
    window: WindowT | None = None
    enabled: bool = True


class _TermIdentity(ValueRecord):
    id: str


class ObjectiveTermValue(_TermFields[Expression, Window]):
    weight: Expression = LiteralValue(value=1)


class ObjectiveTerm(ObjectiveTermValue, _TermIdentity):
    name: str | None = Field(default=None, min_length=1, exclude_if=lambda value: value is None)


class StoredObjectiveTerm(_TermFields[StoredExpression, StoredWindow]):
    weight: StoredExpression = Field(default=1, validate_default=True)
    label: str | None = Field(default=None, min_length=1, exclude_if=lambda value: value is None)

    def lower(self, key: str) -> ObjectiveTerm:
        term = project_member(self, ObjectiveTerm, {
            "measure": lower_expression,
            "window": lambda window: window.lower() if window is not None else None,
        }, id=key, name=self.label, weight=lower_expression(self.weight))
        if "weight" not in self.model_fields_set:
            term.model_fields_set.discard("weight")
        return term

    @classmethod
    def from_lowered(cls, term: ObjectiveTerm) -> "StoredObjectiveTerm":
        return project_member(term, cls, {
            "measure": store_expression,
            "weight": store_expression,
            "window": lambda window: StoredWindow.from_lowered(window) if window is not None else None,
        }, label=term.name)


class ObjectiveDocument(ValueRecord):
    id: str
    terms: tuple[ObjectiveTerm, ...] = ()


class StoredObjectiveDocument(ValueRecord):
    """An ordered map of Objective terms addressed by stable key."""

    id: str
    terms: StoredMembers[StoredObjectiveTerm] = Field(default_factory=dict)

    def lower(self) -> ObjectiveDocument:
        return project_member(self, ObjectiveDocument, {
            "terms": lambda terms: tuple(term.lower(key) for key, term in terms.items()),
        })

    @classmethod
    def from_lowered(cls, objective: ObjectiveDocument) -> "StoredObjectiveDocument":
        return project_member(objective, cls, {
            "terms": lambda terms: keyed_members((term.id, StoredObjectiveTerm.from_lowered(term)) for term in terms),
        })
