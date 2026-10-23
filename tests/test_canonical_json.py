"""Cross-language canonical JSON contract and migration checks."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from feedbax.contracts.canonical_json import (
    CANONICAL_JSON_V1,
    CANONICAL_JSON_V2,
    CanonicalJsonError,
    CanonicalJsonErrorCode,
    canonical_json_bytes_for_algorithm,
    canonical_json_v1_bytes,
    canonical_json_v2_bytes,
)
from feedbax.contracts.manifest import canonical_json_bytes
from feedbax.contracts.worker import (
    CONSISTENCY_PREDICATE_GENERATOR_HASH,
    CONSISTENCY_PREDICATE_SCHEMA_ID,
    CONSISTENCY_PREDICATE_SCHEMA_VERSION,
    CONSISTENCY_PREDICATE_SCHEMA_VERSION_V2,
    ConsistencyPredicateSpec,
    derive_consistency_predicate,
    toy_minimax_method_contract,
)
from feedbax.studio.draft_hash import (
    STUDIO_DRAFT_HASH_PIN,
    STUDIO_DRAFT_HASH_SCHEMA_ID,
    STUDIO_DRAFT_HASH_SCHEMA_VERSION,
    admit_studio_draft_hashes,
    studio_draft_digest_v2,
    studio_draft_hashes,
)


VECTOR_PATH = Path(__file__).resolve().parents[1] / "conformance/canonical_json_v2.json"


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
    assert vector["draft_hash"] == {
        "schema_id": STUDIO_DRAFT_HASH_SCHEMA_ID,
        "schema_version": STUDIO_DRAFT_HASH_SCHEMA_VERSION,
        "pin": STUDIO_DRAFT_HASH_PIN,
    }

    observed_case_ids: set[str] = set()
    for case in vector["cases"]:
        case_id = case["case_id"]
        assert case_id not in observed_case_ids
        observed_case_ids.add(case_id)
        value = _vector_value(case)
        if "expected_utf8_hex" in case:
            assert canonical_json_v2_bytes(value).hex() == case["expected_utf8_hex"], case_id
            assert studio_draft_digest_v2(value) == case["expected_draft_digest"], case_id
            continue
        with pytest.raises(CanonicalJsonError) as exc_info:
            canonical_json_v2_bytes(value)
        assert exc_info.value.code.value == case["expected_error"], case_id
        with pytest.raises(CanonicalJsonError) as digest_exc_info:
            studio_draft_digest_v2(value)
        assert digest_exc_info.value.code.value == case["expected_error"], case_id


def test_studio_draft_hash_version_admission_and_json_round_trip() -> None:
    legacy = admit_studio_draft_hashes({"training_spec": "fnv1a:12345678"})
    assert legacy == {
        "schema_id": "feedbax.studio.draft_hashes",
        "schema_version": "feedbax.studio.draft_hashes.v1",
        "pin": "fnv1a32-runtime_local_json_v1",
        "rehash_required": True,
        "hashes": {"training_spec": "fnv1a:12345678"},
    }
    assert admit_studio_draft_hashes(json.loads(json.dumps(legacy))) == legacy

    current = studio_draft_hashes({"training_spec": {"integral": 1.0, "zero": -0.0}})
    assert admit_studio_draft_hashes(json.loads(json.dumps(current))) == current

    unknown = {**current, "pin": "fnv1a32-canonical_json_v999"}
    with pytest.raises(ValueError, match="unsupported Studio draft hash pin"):
        admit_studio_draft_hashes(unknown)
    future = {**current, "schema_version": "feedbax.studio.draft_hashes.v999"}
    with pytest.raises(ValueError, match="unsupported Studio draft hash schema version"):
        admit_studio_draft_hashes(future)


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


def test_algorithm_dispatch_is_explicit_and_unknown_algorithms_fail_typed() -> None:
    value = {"integral": 1.0}
    assert canonical_json_bytes_for_algorithm(value, CANONICAL_JSON_V1) == b'{"integral":1.0}'
    assert canonical_json_bytes_for_algorithm(value, CANONICAL_JSON_V2) == b'{"integral":1}'

    with pytest.raises(CanonicalJsonError) as exc_info:
        canonical_json_bytes_for_algorithm(value, "canonical_json_v3")
    assert exc_info.value.code is CanonicalJsonErrorCode.UNKNOWN_ALGORITHM


def test_consistency_predicate_v2_migration_preserves_and_pins_its_digest() -> None:
    legacy_digest = "a" * 64
    migrated = ConsistencyPredicateSpec.model_validate(
        {
            "schema_id": CONSISTENCY_PREDICATE_SCHEMA_ID,
            "schema_version": CONSISTENCY_PREDICATE_SCHEMA_VERSION_V2,
            "generator_hash": CONSISTENCY_PREDICATE_GENERATOR_HASH,
            "rules": [],
            "phase_program_digest": legacy_digest,
        }
    )
    assert migrated.schema_version == CONSISTENCY_PREDICATE_SCHEMA_VERSION
    assert migrated.phase_program_digest == legacy_digest
    assert migrated.pin_algorithm == CANONICAL_JSON_V1


def test_new_consistency_predicates_pin_canonical_json_v2() -> None:
    program = toy_minimax_method_contract().phase_program
    predicate = derive_consistency_predicate(program)
    expected = hashlib.sha256(
        canonical_json_v2_bytes(program.model_dump(mode="json", exclude_none=True))
    ).hexdigest()
    assert predicate.schema_version == CONSISTENCY_PREDICATE_SCHEMA_VERSION
    assert predicate.pin_algorithm == CANONICAL_JSON_V2
    assert predicate.phase_program_digest == expected


def test_consistency_predicate_unknown_pin_fails_closed() -> None:
    with pytest.raises(ValidationError, match="pin_algorithm"):
        ConsistencyPredicateSpec.model_validate(
            {
                "schema_id": CONSISTENCY_PREDICATE_SCHEMA_ID,
                "schema_version": CONSISTENCY_PREDICATE_SCHEMA_VERSION,
                "rules": [],
                "phase_program_digest": "a" * 64,
                "pin_algorithm": "canonical_json_v3",
            }
        )


def test_consistency_predicate_current_schema_requires_an_explicit_pin() -> None:
    with pytest.raises(ValidationError, match="pin_algorithm"):
        ConsistencyPredicateSpec.model_validate(
            {
                "schema_id": CONSISTENCY_PREDICATE_SCHEMA_ID,
                "schema_version": CONSISTENCY_PREDICATE_SCHEMA_VERSION,
                "rules": [],
                "phase_program_digest": "a" * 64,
            }
        )
