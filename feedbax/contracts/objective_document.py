"""Stored objective document and its terms."""

from enum import StrEnum

from feedbax.contracts.expression_values import (
    Expression,
    LiteralValue,
    ValueBase,
    ValueRecord,
    ValueType,
)


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
    epochs: tuple[str, ...] = ()
    enabled: bool = True


class ObjectiveDocument(ValueRecord):
    id: str
    terms: tuple[ObjectiveTerm, ...] = ()
