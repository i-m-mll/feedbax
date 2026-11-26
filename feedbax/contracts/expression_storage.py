"""The scalar and text values stored in authored expression fields."""

from collections.abc import Callable, Mapping
import re
from typing import Annotated, Any

from pydantic import BeforeValidator, PlainSerializer, StringConstraints

from feedbax.contracts.expression_text import parse_stored_expression, print_stored_expression
from feedbax.contracts.expression_units import UNITS
from feedbax.contracts.expression_values import (
    RESERVED_NAMES,
    TIME,
    CallValue,
    DrawValue,
    Expression,
    LiteralValue,
    NameValue,
    Scalar,
    SetVariableOrigin,
    UnresolvedValue,
)
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.stored_members import scoped_set_member_id
from feedbax.contracts.time_units import TimeValue, literal_time


MAX_AUTHORED_EXPRESSION_TEXT = 8192
type AuthoredExpressionText = Annotated[str, StringConstraints(max_length=MAX_AUTHORED_EXPRESSION_TEXT)]
type AuthoredExpressionValue = Scalar | AuthoredExpressionText

_KEY = r"[A-Za-z_][A-Za-z0-9_-]*"
_STORED_EPOCH = re.compile(rf"epoch\.({_KEY})\.(condition|limit)")
_LOWERED_EPOCH = re.compile(rf"epoch:({_KEY}):(condition|limit)")
_STORED_SET = re.compile(rf"set\.({_KEY})\.({_KEY})\.({_KEY})")
EXPRESSION_MEMBER_RESERVED_KEYS = RESERVED_NAMES | frozenset({
    "true", "false", "and", "or", "not", "epoch", "set",
    "graph", "inputs", "outputs", "boundary", "in", "out", "physical", "component", "wire", "connection",
})


def _map_refs(value: Expression, convert: Callable[[str], str]) -> Expression:
    match value:
        case NameValue(reference=reference):
            spelling = convert(reference)
            return value if spelling == reference else value.model_copy(update={"reference": spelling})
        case CallValue(args=args, kwargs=kwargs):
            return value.model_copy(update={
                "args": tuple(_map_refs(child, convert) for child in args),
                "kwargs": {key: _map_refs(child, convert) for key, child in kwargs.items()},
            })
        case DrawValue(args=args):
            return value.model_copy(update={"args": tuple(_map_refs(child, convert) for child in args)})
        case _:
            return value


def stored_reference(reference: str) -> str:
    """Spell a lowered Model or epoch reference by its authored key."""
    if found := _LOWERED_EPOCH.fullmatch(reference):
        return f"epoch.{found[1]}.{found[2]}"
    if reference.startswith("{"):
        try:
            return ModelAddress.from_text(reference).to_authored_text()
        except ValueError:
            pass
    return reference


def stored_set_reference(origin: SetVariableOrigin) -> str:
    return f"set.{origin.axis}.{origin.entry}.{origin.variable}"


def _lowered_reference(spelling: str) -> str:
    if found := _STORED_EPOCH.fullmatch(spelling):
        return f"epoch:{found[1]}:{found[2]}"
    if found := _STORED_SET.fullmatch(spelling):
        return scoped_set_member_id(found[1], found[2], found[3])
    if "." in spelling or "/" in spelling or spelling in ("graph", "inputs", "outputs"):
        try:
            return ModelAddress.from_authored_text(spelling).to_text()
        except ValueError:
            pass
    return spelling


def store_expression(value: Expression, *, scopes: Mapping[str, str] | None = None) -> Expression:
    """Convert runtime references to authored spellings, including known Set origins."""
    if scopes is None:
        return _map_refs(value, stored_reference)
    return _map_refs(value, lambda reference: scopes.get(reference) or stored_reference(reference))


def stored_expression_value(value: Expression, *, scopes: Mapping[str, str] | None = None) -> AuthoredExpressionValue:
    return print_stored_expression(store_expression(value, scopes=scopes))


def rename_expression_references(value: Expression, references: Mapping[str, str]) -> Expression:
    """Replace the named references of an expression while preserving its representation."""
    return _map_refs(value, lambda reference: references.get(reference, reference))


def lower_expression(value: Expression) -> Expression:
    """Resolve key spellings without consulting presentation labels or the editor's scope."""
    return _map_refs(value, _lowered_reference)


def _read_expression(raw: Any) -> Expression:
    if isinstance(raw, (LiteralValue, NameValue, CallValue, DrawValue, UnresolvedValue)):
        return store_expression(raw)
    if type(raw) not in (bool, int, float, str):
        raise ValueError("An authored expression must be a scalar or text.")
    if isinstance(raw, str) and len(raw) > MAX_AUTHORED_EXPRESSION_TEXT:
        raise ValueError("An expression can have at most 8,192 characters.")
    parsed = parse_stored_expression(raw)
    if isinstance(parsed, tuple):
        return UnresolvedValue(text=raw)
    return parsed.value


type StoredExpression = Annotated[
    Expression,
    BeforeValidator(_read_expression, json_schema_input_type=AuthoredExpressionValue),
    PlainSerializer(print_stored_expression, return_type=AuthoredExpressionValue),
]


def _read_time(raw: Any) -> TimeValue | UnresolvedValue:
    if isinstance(raw, (TimeValue, UnresolvedValue)):
        return raw
    if not isinstance(raw, str):
        raise ValueError("An authored time must be literal text, such as '50 ms' or '30 st'.")
    if len(raw) > MAX_AUTHORED_EXPRESSION_TEXT:
        raise ValueError("A time can have at most 8,192 characters.")
    parsed = parse_stored_expression(raw)
    if not isinstance(parsed, tuple) and isinstance(parsed.value, LiteralValue):
        literal = parsed.value
        unit = UNITS.get(literal.unit)
        if type(literal.value) is not bool and unit is not None and unit.dimensions == TIME:
            return literal_time(literal)
    return UnresolvedValue(text=raw)


def _write_time(value: TimeValue | UnresolvedValue) -> str:
    if isinstance(value, UnresolvedValue):
        return value.text
    text = print_stored_expression(LiteralValue(value=value.value, unit=value.unit))
    assert isinstance(text, str)
    return text


type StoredTime = Annotated[
    TimeValue | UnresolvedValue,
    BeforeValidator(_read_time, json_schema_input_type=AuthoredExpressionText),
    PlainSerializer(_write_time, return_type=AuthoredExpressionText),
]
