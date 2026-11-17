"""Cross-language canonical JSON contract and migration checks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from feedbax.contracts.canonical_json import (
    CANONICAL_JSON_V2,
    CanonicalJsonError,
    canonical_json_v1_bytes,
    canonical_json_v2_bytes,
)
from feedbax.contracts.manifest import canonical_json_bytes


VECTOR_PATH = Path(__file__).resolve().parents[1] / "tests/fixtures/canonical_json_v2.json"


class _UnsupportedLeaf:
    pass


def _special_value(name: str) -> object:
    if name == "nan":
        return math.nan
    if name == "positive_infinity":
        return math.inf
    if name == "negative_infinity":
        return -math.inf
    if name == "non_string_key":
        return {1: "value"}
    if name == "unsupported_leaf":
        return {"leaf": _UnsupportedLeaf()}
    if name == "array_cycle":
        value: list[object] = []
        value.append(value)
        return value
    raise AssertionError(f"unknown canonical JSON conformance special value: {name}")


def _vector_value(case: dict[str, Any]) -> object:
    input_spec = case["input"]
    if input_spec["form"] == "json":
        return input_spec["value"]
    return _special_value(input_spec["name"])


def test_canonical_json_v2_language_neutral_conformance_vector() -> None:
    vector = json.loads(VECTOR_PATH.read_text(encoding="utf-8"))
    assert vector["schema_id"] == "feedbax.conformance.canonical_json_v2"
    assert vector["schema_version"] == "feedbax.conformance.canonical_json_v2.v1"
    assert vector["algorithm"] == CANONICAL_JSON_V2
    observed_case_ids: set[str] = set()
    for case in vector["cases"]:
        case_id = case["case_id"]
        assert case_id not in observed_case_ids
        observed_case_ids.add(case_id)
        value = _vector_value(case)
        if "expected_utf8_hex" in case:
            assert canonical_json_v2_bytes(value).hex() == case["expected_utf8_hex"], case_id
            continue
        with pytest.raises(CanonicalJsonError) as exc_info:
            canonical_json_v2_bytes(value)
        assert exc_info.value.code.value == case["expected_error"], case_id


def test_canonical_json_v1_bytes_are_permanently_unchanged() -> None:
    value = {
        "é": "é",
        "float": 1.0,
        "negative_zero": -0.0,
        "non_finite": math.nan,
    }
    expected = (
        b'{"float":1.0,"negative_zero":-0.0,"non_finite":NaN,'
        b'"\\u00e9":"\\u00e9"}'
    )
    assert canonical_json_v1_bytes(value) == expected
    assert canonical_json_bytes(value) == expected
