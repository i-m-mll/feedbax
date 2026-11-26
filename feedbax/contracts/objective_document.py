"""Authored Objective documents and their lowering for execution."""

from collections.abc import Mapping
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
from feedbax.contracts.member_family import StoredFamilyMember, expand_members
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
    family: str | None = Field(default=None, exclude_if=lambda value: value is None)

    @property
    def report_key(self) -> str:
        return self.family if self.family is not None else self.id


class StoredObjectiveTerm(_TermFields[StoredExpression, StoredWindow], StoredFamilyMember):
    weight: StoredExpression = Field(default=1, validate_default=True)
    label: str | None = Field(default=None, min_length=1, exclude_if=lambda value: value is None)

    def lower(self, key: str, *, family: str | None = None) -> ObjectiveTerm:
        term = project_member(self, ObjectiveTerm, {
            "measure": lower_expression,
            "window": lambda window: window.lower() if window is not None else None,
        }, id=key, name=self.label, family=family, weight=lower_expression(self.weight))
        if "weight" not in self.model_fields_set:
            term.model_fields_set.discard("weight")
        return term

    @classmethod
    def from_lowered(cls, term: ObjectiveTerm) -> "StoredObjectiveTerm":
        return cls.from_value(term, label=term.name)

    @classmethod
    def from_value(
        cls, term: ObjectiveTermValue, *, label: str | None, scopes: Mapping[str, str] | None = None,
    ) -> "StoredObjectiveTerm":
        return project_member(term, cls, {
            "measure": lambda expression: store_expression(expression, scopes=scopes),
            "weight": lambda expression: store_expression(expression, scopes=scopes),
            "window": lambda window: StoredWindow.from_lowered(window, scopes=scopes) if window is not None else None,
        }, label=label)


class ObjectiveDocument(ValueRecord):
    id: str
    terms: tuple[ObjectiveTerm, ...] = ()

    def for_execution(self) -> "ObjectiveDocument":
        if all(term.name is None for term in self.terms):
            return self
        return self.model_copy(update={"terms": tuple(
            term if term.name is None else term.model_copy(update={"name": None}) for term in self.terms
        )})


class StoredObjectiveDocument(ValueRecord):
    """An ordered map of Objective terms addressed by stable key."""

    id: str
    terms: StoredMembers[StoredObjectiveTerm] = Field(default_factory=dict)

    def lower(self, *, first: bool = False) -> ObjectiveDocument:
        return project_member(self, ObjectiveDocument, {
            "terms": lambda terms: tuple(
                term.value.lower(
                    term.source_key if first else term.key,
                    family=term.source_key if term.coordinates and not first else None,
                )
                for term in expand_members(terms, first=first, label_placeholders=False)
            ),
        })

    @classmethod
    def from_lowered(cls, objective: ObjectiveDocument) -> "StoredObjectiveDocument":
        return project_member(objective, cls, {
            "terms": lambda terms: keyed_members((term.id, StoredObjectiveTerm.from_lowered(term)) for term in terms),
        })


def objective_weight_title(payload: object, path: tuple[str, ...]) -> str | None:
    """Name a scalar Weight axis by the current label of its Objective term."""
    if len(path) not in (3, 4) or path[0] != "terms" or path[2] != "weight" or path[3:] not in ((), ("value",)):
        return None
    terms = payload.get("terms") if isinstance(payload, dict) else None
    term = terms.get(path[1]) if isinstance(terms, dict) else None
    if not isinstance(term, dict):
        return None
    label = term.get("label") or path[1].replace("-", " ").replace("_", " ").capitalize()
    return f"{label} weight"
