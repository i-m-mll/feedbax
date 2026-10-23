from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from feedbax.contracts import (
    VALUE_IDENTITY_SCHEMA_ID,
    VALUE_IDENTITY_SCHEMA_VERSION,
    ValueIdentityRecord,
    authored_value_sha256,
    realization_value_sha256,
    semantic_value_sha256,
    value_identity_record,
)


def _authored(encoding_kind: str, *, locator: str = "local://one") -> str:
    return authored_value_sha256(
        encoding_kind=encoding_kind,
        encoding_schema_id=f"example.{encoding_kind}",
        encoding_schema_version=f"example.{encoding_kind}.v1",
        arguments={"rows": 2},
        content_pins={"payload": "sha256:" + "a" * 64},
        movable_locators=[locator],
    )


def test_cross_encoding_semantic_equality_and_authored_inequality() -> None:
    inline = _authored("inline")
    pinned = _authored("pinned")

    assert inline != pinned
    assert semantic_value_sha256([1, 2], dtype="float32") == semantic_value_sha256(
        np.array([1.0, 2.0], dtype=np.float32),
        dtype="float32",
    )


def test_raw_authored_numeric_spelling_and_normalized_value_have_separate_tiers() -> None:
    """The d8e2fc1 raw-versus-normalized ambiguity resolves without one mixed hash."""
    raw_authored = authored_value_sha256(
        encoding_kind="inline",
        encoding_schema_id="example.inline",
        encoding_schema_version="example.inline.v1",
        arguments={"value": 1},
    )
    normalized_authored = authored_value_sha256(
        encoding_kind="inline",
        encoding_schema_id="example.inline",
        encoding_schema_version="example.inline.v1",
        arguments={"value": 1.0},
    )

    assert raw_authored != normalized_authored
    assert semantic_value_sha256(1, dtype="float64") == semantic_value_sha256(
        1.0,
        dtype="float64",
    )


def test_dtype_shape_scalar_and_list_coercion_are_explicit() -> None:
    assert semantic_value_sha256(1, dtype="int32") == semantic_value_sha256(
        np.array(1, dtype=np.int32),
        dtype="int32",
    )
    assert semantic_value_sha256([1], dtype="int32") == semantic_value_sha256(
        np.array([1], dtype=np.int32),
        dtype="int32",
    )
    assert semantic_value_sha256(1, dtype="int32") != semantic_value_sha256(
        [1],
        dtype="int32",
    )
    assert semantic_value_sha256([1], dtype="int32") != semantic_value_sha256(
        [1],
        dtype="int64",
    )


def test_c_order_signed_zero_nan_and_infinity_normalization() -> None:
    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    noncontiguous = base.T
    contiguous = np.ascontiguousarray(noncontiguous)
    assert not noncontiguous.flags.c_contiguous
    assert semantic_value_sha256(noncontiguous, dtype="float32") == semantic_value_sha256(
        contiguous, dtype="float32"
    )

    assert semantic_value_sha256([-0.0, 0.0], dtype="float64") == semantic_value_sha256(
        [0.0, -0.0], dtype="float64"
    )

    positive = semantic_value_sha256([math.inf], dtype="float64")
    negative = semantic_value_sha256([-math.inf], dtype="float64")
    assert positive != negative
    assert positive == semantic_value_sha256(
        np.array([math.inf], dtype=">f8"),
        dtype="float64",
    )


@pytest.mark.parametrize(
    ("dtype", "unsigned_dtype", "payloads"),
    [
        ("float16", np.uint16, [0x7E01, 0xFE55, 0x7D00]),
        ("float32", np.uint32, [0x7FC00001, 0xFFC12345, 0x7FA00000]),
        (
            "float64",
            np.uint64,
            [0x7FF8000000000001, 0xFFF8123456789ABC, 0x7FF4000000000000],
        ),
    ],
)
def test_every_supported_float_width_canonicalizes_all_nan_payloads(
    dtype: str,
    unsigned_dtype: type[np.unsignedinteger],
    payloads: list[int],
) -> None:
    nan_bits = np.array(payloads, dtype=unsigned_dtype).view(np.dtype(dtype))
    canonical_nans = np.array([math.nan] * len(payloads), dtype=np.dtype(dtype))

    assert semantic_value_sha256(nan_bits, dtype=dtype) == semantic_value_sha256(
        canonical_nans,
        dtype=dtype,
    )


def test_authored_locators_are_excluded_while_pins_and_constructor_are_included() -> None:
    assert _authored("pinned", locator="local://one") == _authored(
        "pinned",
        locator="local://two",
    )
    base = authored_value_sha256(
        encoding_kind="pinned",
        encoding_schema_id="example.pinned",
        encoding_schema_version="example.pinned.v1",
        arguments={},
        content_pins={"payload": "sha256:" + "a" * 64},
    )
    changed_pin = authored_value_sha256(
        encoding_kind="pinned",
        encoding_schema_id="example.pinned",
        encoding_schema_version="example.pinned.v1",
        arguments={},
        content_pins={"payload": "sha256:" + "b" * 64},
    )
    governed = authored_value_sha256(
        encoding_kind="constructor",
        encoding_schema_id="example.constructor",
        encoding_schema_version="example.constructor.v1",
        arguments={},
        governed_constructor=True,
        constructor_fingerprint="c" * 64,
    )
    changed_constructor = authored_value_sha256(
        encoding_kind="constructor",
        encoding_schema_id="example.constructor",
        encoding_schema_version="example.constructor.v1",
        arguments={},
        governed_constructor=True,
        constructor_fingerprint="d" * 64,
    )

    assert base != changed_pin
    assert governed != changed_constructor


def test_expected_semantic_mismatch_fails_closed_and_chain_is_preserved() -> None:
    authored = _authored("inline")
    parent = "b" * 64
    expected = semantic_value_sha256([1, 2], dtype="float32")
    record = value_identity_record(
        authored_sha256=authored,
        value=[1, 2],
        dtype="float32",
        inherited_authored_chain=[parent],
        expected_semantic_sha256=expected,
    )

    assert record.authored_identity_chain == (parent, authored)
    with pytest.raises(ValidationError, match="expected semantic identity mismatch"):
        value_identity_record(
            authored_sha256=authored,
            value=[1, 3],
            dtype="float32",
            inherited_authored_chain=[parent],
            expected_semantic_sha256=expected,
        )


def test_realization_uses_only_explicit_fingerprints() -> None:
    semantic = semantic_value_sha256([1, 2], dtype="float32")
    cpu = realization_value_sha256(
        semantic,
        layout_fingerprint="row-major",
        backend_fingerprint="cpu-v1",
    )
    gpu = realization_value_sha256(
        semantic,
        layout_fingerprint="row-major",
        backend_fingerprint="gpu-v1",
    )
    assert cpu != gpu

    realized_record = value_identity_record(
        authored_sha256=_authored("inline"),
        value=[1, 2],
        dtype="float32",
        layout_fingerprint="row-major",
        backend_fingerprint="cpu-v1",
    )
    assert realized_record.realization_sha256 == cpu
    assert realized_record.runtime_layout_fingerprint == "row-major"
    assert realized_record.runtime_backend_fingerprint == "cpu-v1"

    record = value_identity_record(
        authored_sha256=_authored("inline"),
        value=[1, 2],
        dtype="float32",
    )
    assert record.realization_sha256 is None
    with pytest.raises(ValueError, match="must be supplied together"):
        value_identity_record(
            authored_sha256=_authored("inline"),
            value=[1, 2],
            dtype="float32",
            layout_fingerprint="row-major",
        )
    with pytest.raises(ValidationError, match="runtime fingerprints"):
        ValueIdentityRecord(
            authored_sha256=_authored("inline"),
            semantic_sha256=semantic,
            realization_sha256=cpu,
            authored_identity_chain=(_authored("inline"),),
        )


def test_schema_is_strict_versioned_and_rejects_unsupported_inputs() -> None:
    authored = _authored("inline")
    semantic = semantic_value_sha256([1], dtype="float32")
    record = ValueIdentityRecord(
        authored_sha256=authored,
        semantic_sha256=semantic,
        authored_identity_chain=(authored,),
    )
    assert record.schema_id == VALUE_IDENTITY_SCHEMA_ID
    assert record.schema_version == VALUE_IDENTITY_SCHEMA_VERSION

    payload = record.model_dump(mode="json")
    payload["schema_version"] = "feedbax.value_identity.v0"
    with pytest.raises(ValidationError):
        ValueIdentityRecord.model_validate(payload)
    with pytest.raises(ValidationError):
        ValueIdentityRecord.model_validate({**record.model_dump(mode="json"), "unexpected": True})
    with pytest.raises(TypeError, match="object-dtype"):
        semantic_value_sha256(
            np.array([1, 2], dtype=object),
            dtype="float32",
        )
    with pytest.raises(ValueError, match="byte-order-ambiguous"):
        semantic_value_sha256([1], dtype="<f4")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        authored_value_sha256(
            encoding_kind="constructor",
            encoding_schema_id="example.constructor",
            encoding_schema_version="example.constructor.v1",
            arguments={},
            governed_constructor=True,
            constructor_fingerprint=" not-canonical ",
        )
    with pytest.raises(ValueError, match="required exactly"):
        authored_value_sha256(
            encoding_kind="inline",
            encoding_schema_id="example.inline",
            encoding_schema_version="example.inline.v1",
            arguments={},
            constructor_fingerprint="c" * 64,
        )
