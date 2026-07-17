"""Shared JSON value types for method-authored numeric diagnostics."""

from __future__ import annotations

import math
from typing import Annotated, Any, TypeAliasType

from pydantic import BeforeValidator


def _validate_numeric_boolean_json(value: Any) -> Any:
    if type(value) is bool or type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("numeric diagnostic values must be finite")
        return value
    if isinstance(value, list):
        return [_validate_numeric_boolean_json(item) for item in value]
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise ValueError("numeric diagnostic mappings require string keys")
        return {key: _validate_numeric_boolean_json(item) for key, item in value.items()}
    raise ValueError(
        "numeric diagnostic values must contain only booleans, integers, finite reals, "
        "lists, and string-keyed mappings"
    )


NumericBooleanJsonValue = TypeAliasType(
    "NumericBooleanJsonValue",
    Annotated[
        bool | int | float | list["NumericBooleanJsonValue"] | dict[str, "NumericBooleanJsonValue"],
        BeforeValidator(_validate_numeric_boolean_json),
    ],
)
