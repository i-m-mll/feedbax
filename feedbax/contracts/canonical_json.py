"""Versioned JSON byte contracts for durable identity.

``canonical_json_v1`` is the historical Python ``json`` spelling. Its quirks
are permanent because existing digests were computed from those exact bytes.
``canonical_json_v2`` is strict RFC 8785 / JCS JSON with the additional rule
that Python integers must fit the JavaScript safe-integer range.

Callers own any semantic projection into JSON. These encoders accept a JSON
value; they do not convert models, arrays, paths, or other application types.
"""

from __future__ import annotations

from enum import StrEnum
import json
import math
from typing import Any, Literal, TypeAlias, cast

import rfc8785


CANONICAL_JSON_V1 = "canonical_json_v1"
CANONICAL_JSON_V2 = "canonical_json_v2"
CanonicalJsonAlgorithm: TypeAlias = Literal["canonical_json_v1", "canonical_json_v2"]

_MAX_SAFE_INTEGER = 2**53 - 1


class CanonicalJsonErrorCode(StrEnum):
    """Stable failure categories for canonical JSON dispatch and v2 input."""

    CYCLE = "cycle"
    ENCODING_FAILURE = "encoding_failure"
    LONE_SURROGATE = "lone_surrogate"
    NON_FINITE_NUMBER = "non_finite_number"
    NON_STRING_KEY = "non_string_key"
    UNKNOWN_ALGORITHM = "unknown_algorithm"
    UNSAFE_INTEGER = "unsafe_integer"
    UNSUPPORTED_TYPE = "unsupported_type"


class CanonicalJsonError(ValueError):
    """Typed, path-aware canonical JSON failure."""

    def __init__(
        self,
        code: CanonicalJsonErrorCode,
        *,
        path: str,
        detail: str,
        value_type: str | None = None,
    ) -> None:
        self.code = code
        self.path = path
        self.detail = detail
        self.value_type = value_type
        location = path or "<root>"
        type_detail = f"; value_type={value_type}" if value_type is not None else ""
        super().__init__(f"canonical JSON {code.value} at {location}: {detail}{type_detail}")


def canonical_json_v1_bytes(value: object) -> bytes:
    """Return the permanent legacy ``canonical_json_v1`` byte spelling."""

    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def canonical_json_v2_bytes(value: object) -> bytes:
    """Return strict cross-language canonical JSON bytes.

    V2 emits UTF-8 with no whitespace, sorts object keys by UTF-16 code units,
    uses ECMAScript-compatible finite-number spelling, preserves non-ASCII text
    without Unicode normalization, and rejects values outside the strict JSON
    contract with :class:`CanonicalJsonError`.
    """

    _validate_v2_value(value, path="", active_containers=set())
    try:
        return rfc8785.dumps(cast(Any, value))
    except rfc8785.CanonicalizationError as exc:
        raise CanonicalJsonError(
            CanonicalJsonErrorCode.ENCODING_FAILURE,
            path="",
            detail=str(exc),
            value_type=type(value).__name__,
        ) from exc


def canonical_json_bytes_for_algorithm(value: object, algorithm: str) -> bytes:
    """Encode ``value`` with one explicitly named canonical JSON algorithm."""

    if algorithm == CANONICAL_JSON_V1:
        return canonical_json_v1_bytes(value)
    if algorithm == CANONICAL_JSON_V2:
        return canonical_json_v2_bytes(value)
    raise CanonicalJsonError(
        CanonicalJsonErrorCode.UNKNOWN_ALGORITHM,
        path="",
        detail=f"unsupported algorithm {algorithm!r}",
        value_type=type(value).__name__,
    )


def _validate_v2_value(value: object, *, path: str, active_containers: set[int]) -> None:
    value_type = type(value)
    if value is None or value_type is bool:
        return
    if value_type is int:
        integer = cast(int, value)
        if not -_MAX_SAFE_INTEGER <= integer <= _MAX_SAFE_INTEGER:
            raise CanonicalJsonError(
                CanonicalJsonErrorCode.UNSAFE_INTEGER,
                path=path,
                detail=(
                    f"integer must be between {-_MAX_SAFE_INTEGER} and {_MAX_SAFE_INTEGER}"
                ),
                value_type="int",
            )
        return
    if value_type is float:
        number = cast(float, value)
        if not math.isfinite(number):
            raise CanonicalJsonError(
                CanonicalJsonErrorCode.NON_FINITE_NUMBER,
                path=path,
                detail="number must be finite",
                value_type="float",
            )
        return
    if value_type is str:
        _validate_v2_string(cast(str, value), path=path)
        return
    if value_type is list:
        array = cast(list[object], value)
        _validate_container_entry(array, path=path, active_containers=active_containers)
        try:
            for index, item in enumerate(array):
                _validate_v2_value(
                    item,
                    path=f"{path}/{index}",
                    active_containers=active_containers,
                )
        finally:
            active_containers.remove(id(array))
        return
    if value_type is dict:
        mapping = cast(dict[object, object], value)
        _validate_container_entry(mapping, path=path, active_containers=active_containers)
        try:
            for key, item in mapping.items():
                if type(key) is not str:
                    raise CanonicalJsonError(
                        CanonicalJsonErrorCode.NON_STRING_KEY,
                        path=path,
                        detail="object keys must be strings",
                        value_type=type(key).__name__,
                    )
                _validate_v2_string(key, path=path)
                _validate_v2_value(
                    item,
                    path=f"{path}/{_json_pointer_token(key)}",
                    active_containers=active_containers,
                )
        finally:
            active_containers.remove(id(mapping))
        return
    raise CanonicalJsonError(
        CanonicalJsonErrorCode.UNSUPPORTED_TYPE,
        path=path,
        detail="value must be null, a boolean, string, finite number, array, or object",
        value_type=value_type.__name__,
    )


def _validate_container_entry(
    value: list[object] | dict[object, object],
    *,
    path: str,
    active_containers: set[int],
) -> None:
    identity = id(value)
    if identity in active_containers:
        raise CanonicalJsonError(
            CanonicalJsonErrorCode.CYCLE,
            path=path,
            detail="arrays and objects must not contain cycles",
            value_type=type(value).__name__,
        )
    active_containers.add(identity)


def _validate_v2_string(value: str, *, path: str) -> None:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise CanonicalJsonError(
            CanonicalJsonErrorCode.LONE_SURROGATE,
            path=path,
            detail="strings must not contain UTF-16 surrogate code points",
            value_type="str",
        )


def _json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


__all__ = [
    "CANONICAL_JSON_V1",
    "CANONICAL_JSON_V2",
    "CanonicalJsonAlgorithm",
    "CanonicalJsonError",
    "CanonicalJsonErrorCode",
    "canonical_json_bytes_for_algorithm",
    "canonical_json_v1_bytes",
    "canonical_json_v2_bytes",
]
