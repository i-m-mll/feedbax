"""Stored objective document and its terms."""

from enum import StrEnum

from pydantic import Field, JsonValue
from feedbax.contracts.expression_values import (
    Expression,
    LiteralValue,
    ValueBase,
    ValueRecord,
    ValueType,
)
from feedbax.contracts.window import Window


OBJECTIVE_DOCUMENT_SCHEMA_ID = "feedbax.spec.objective_document"


class Reduction(StrEnum):
    MEAN = "mean"
    SUM = "sum"
    LAST = "last"
    MAX = "max"


class ObjectiveTerm(ValueRecord):
    id: str
    measure: Expression
    measure_type: ValueType = ValueType(base=ValueBase.REAL)
    weight: Expression = LiteralValue(value=1)
    reduction: Reduction = Reduction.MEAN
    window: Window | None = None
    enabled: bool = True
    name: str | None = Field(default=None, min_length=1, exclude_if=lambda value: value is None)


class ObjectiveDocument(ValueRecord):
    id: str
    terms: tuple[ObjectiveTerm, ...] = ()

def objective_weight_title(payload: JsonValue, path: tuple[str, ...]) -> str | None:
    """Name a scalar Weight axis by the current name of its Objective term."""
    if len(path) not in (3, 4) or path[0] != "terms" or path[2] != "weight" or path[3:] not in ((), ("value",)):
        return None
    terms = payload.get("terms") if isinstance(payload, dict) else None
    if not isinstance(terms, list):
        return None
    for index, term in enumerate(terms, 1):
        if isinstance(term, dict) and term.get("id") == path[1]:
            return f"{term.get('name') or f'Term {index}'} weight"
    return None
