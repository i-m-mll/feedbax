"""Versioned authored, semantic, and realization identities for numeric values.

The three tiers deliberately answer different questions:

* authored identity preserves the declared encoding and its immutable inputs;
* semantic identity deduplicates exact normalized numeric representations; and
* realization identity separates runtime layout/backend cache domains.

Schema/migration table:

* ``feedbax.value_identity``: new v1 fields ``schema_id``,
  ``schema_version``, the three tier digests, both runtime fingerprints,
  ``authored_identity_chain``, and ``expected_semantic_sha256``. Other versions
  and extra fields are rejected.
* Existing Feedbax manifest, specification, checkpoint, and artifact families:
  no affected fields, version bumps, or migrations. Consumers that later embed
  this record own the corresponding envelope migration.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.spec_storage import training_spec_canonical_bytes


VALUE_IDENTITY_SCHEMA_ID = "feedbax.value_identity"
VALUE_IDENTITY_SCHEMA_VERSION = f"{VALUE_IDENTITY_SCHEMA_ID}.v1"

_SEMANTIC_PREIMAGE_SCHEMA_ID = "feedbax.value_identity.semantic_preimage"
_SEMANTIC_PREIMAGE_SCHEMA_VERSION = f"{_SEMANTIC_PREIMAGE_SCHEMA_ID}.v1"
_AUTHORED_PREIMAGE_SCHEMA_ID = "feedbax.value_identity.authored_preimage"
_AUTHORED_PREIMAGE_SCHEMA_VERSION = f"{_AUTHORED_PREIMAGE_SCHEMA_ID}.v1"
_REALIZATION_PREIMAGE_SCHEMA_ID = "feedbax.value_identity.realization_preimage"
_REALIZATION_PREIMAGE_SCHEMA_VERSION = f"{_REALIZATION_PREIMAGE_SCHEMA_ID}.v1"

_CANONICAL_DTYPES = frozenset(
    {
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
    }
)
_FLOAT_BITS = {
    "float16": (np.dtype("<u2"), 0x7E00),
    "float32": (np.dtype("<u4"), 0x7FC00000),
    "float64": (np.dtype("<u8"), 0x7FF8000000000000),
}


class ValueIdentityRecord(StrictModel):
    """Strict, versioned record joining the three value-identity tiers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["feedbax.value_identity"] = VALUE_IDENTITY_SCHEMA_ID
    schema_version: Literal["feedbax.value_identity.v1"] = VALUE_IDENTITY_SCHEMA_VERSION
    authored_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    realization_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    runtime_layout_fingerprint: str | None = None
    runtime_backend_fingerprint: str | None = None
    authored_identity_chain: tuple[str, ...]
    expected_semantic_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def _validate_identity_chain_and_expectation(self) -> "ValueIdentityRecord":
        if not self.authored_identity_chain:
            raise ValueError("authored_identity_chain must include the current identity")
        if self.authored_identity_chain[-1] != self.authored_sha256:
            raise ValueError("authored_identity_chain must end with authored_sha256")
        for identity in self.authored_identity_chain:
            _validate_sha256(identity, field_name="authored_identity_chain")
        if (
            self.expected_semantic_sha256 is not None
            and self.expected_semantic_sha256 != self.semantic_sha256
        ):
            raise ValueError(
                "expected semantic identity mismatch: "
                f"expected={self.expected_semantic_sha256}, "
                f"realized={self.semantic_sha256}"
            )
        realization_fields = (
            self.realization_sha256,
            self.runtime_layout_fingerprint,
            self.runtime_backend_fingerprint,
        )
        if any(value is not None for value in realization_fields) and not all(
            value is not None for value in realization_fields
        ):
            raise ValueError(
                "realization_sha256 and runtime fingerprints must be supplied together"
            )
        if (
            self.realization_sha256 is not None
            and self.runtime_layout_fingerprint is not None
            and self.runtime_backend_fingerprint is not None
        ):
            expected_realization = realization_value_sha256(
                self.semantic_sha256,
                layout_fingerprint=self.runtime_layout_fingerprint,
                backend_fingerprint=self.runtime_backend_fingerprint,
            )
            if self.realization_sha256 != expected_realization:
                raise ValueError(
                    "realization identity mismatch: "
                    f"recorded={self.realization_sha256}, "
                    f"computed={expected_realization}"
                )
        return self


def authored_value_sha256(
    *,
    encoding_kind: str,
    encoding_schema_id: str,
    encoding_schema_version: str,
    arguments: Mapping[str, Any],
    content_pins: Mapping[str, str] | None = None,
    movable_locators: Sequence[str] = (),
    governed_constructor: bool = False,
    constructor_fingerprint: str | None = None,
) -> str:
    """Hash one authored value declaration.

    Args:
        encoding_kind: Stable encoding discriminator.
        encoding_schema_id: Stable schema family for the encoding.
        encoding_schema_version: Exact encoding schema version.
        arguments: Canonical JSON arguments that affect the produced value.
        content_pins: Immutable, named content identities.
        movable_locators: Non-authoritative lookup hints. They are validated as
            strings and intentionally excluded from the identity preimage.
        governed_constructor: Whether this encoding explicitly invokes a
            governed constructor/plugin.
        constructor_fingerprint: Lowercase SHA-256 of the declared governed
            constructor/plugin. Required exactly when ``governed_constructor``
            is true.

    Returns:
        Lowercase SHA-256 of the versioned authored preimage.
    """
    for field_name, value in (
        ("encoding_kind", encoding_kind),
        ("encoding_schema_id", encoding_schema_id),
        ("encoding_schema_version", encoding_schema_version),
    ):
        _require_nonempty_string(value, field_name=field_name)
    if not isinstance(arguments, Mapping):
        raise TypeError("arguments must be a mapping")
    pins = {} if content_pins is None else dict(content_pins)
    for key, value in pins.items():
        _require_nonempty_string(key, field_name="content pin name")
        _require_nonempty_string(value, field_name=f"content pin {key!r}")
    if isinstance(movable_locators, (str, bytes)) or not isinstance(movable_locators, Sequence):
        raise TypeError("movable_locators must be a sequence of strings")
    for locator in movable_locators:
        _require_nonempty_string(locator, field_name="movable locator")
    if not isinstance(governed_constructor, bool):
        raise TypeError("governed_constructor must be a boolean")
    if governed_constructor != (constructor_fingerprint is not None):
        raise ValueError("constructor_fingerprint is required exactly for governed constructors")
    if constructor_fingerprint is not None:
        _validate_sha256(
            constructor_fingerprint,
            field_name="constructor_fingerprint",
        )

    envelope = {
        "schema_id": _AUTHORED_PREIMAGE_SCHEMA_ID,
        "schema_version": _AUTHORED_PREIMAGE_SCHEMA_VERSION,
        "encoding_kind": encoding_kind,
        "encoding_schema_id": encoding_schema_id,
        "encoding_schema_version": encoding_schema_version,
        "arguments": dict(arguments),
        "content_pins": pins,
        "governed_constructor": governed_constructor,
        "constructor_fingerprint": constructor_fingerprint,
    }
    return hashlib.sha256(training_spec_canonical_bytes(envelope)).hexdigest()


def semantic_value_sha256(value: object, *, dtype: str) -> str:
    """Hash the exact normalized numeric representation of ``value``.

    The caller must select one canonical dtype name. Values are coerced to that
    dtype, stored in little-endian C order, and hashed with shape and dtype.
    Signed zero becomes positive zero. Every NaN payload/sign maps to the one
    quiet-NaN bit pattern defined by IEEE 754 for the selected float width.
    Positive and negative infinity retain their distinct IEEE representations.

    Args:
        value: Numeric scalar, sequence, or array-like value.
        dtype: One explicit canonical dtype name from the public v1 policy.

    Returns:
        Lowercase SHA-256 of the versioned semantic preimage.
    """
    canonical_dtype = _canonical_dtype(dtype)
    source_dtype = getattr(value, "dtype", None)
    if source_dtype is not None and np.dtype(source_dtype).hasobject:
        raise TypeError("object-dtype values are unsupported")
    try:
        array = np.asarray(value, dtype=canonical_dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"value cannot be coerced to canonical dtype {dtype!r}") from exc
    array = np.array(array, dtype=canonical_dtype, order="C", copy=True)

    if dtype in _FLOAT_BITS:
        array[array == 0] = 0.0
        nan_mask = np.isnan(array)
        if np.any(nan_mask):
            unsigned_dtype, canonical_nan = _FLOAT_BITS[dtype]
            array.view(unsigned_dtype)[nan_mask] = canonical_nan

    header = {
        "schema_id": _SEMANTIC_PREIMAGE_SCHEMA_ID,
        "schema_version": _SEMANTIC_PREIMAGE_SCHEMA_VERSION,
        "dtype": dtype,
        "byte_order": "little",
        "order": "C",
        "shape": list(array.shape),
        "signed_zero": "positive",
        "nan": "canonical_quiet",
        "infinity": "preserve_sign",
    }
    preimage = training_spec_canonical_bytes(header) + b"\n" + array.tobytes(order="C")
    return hashlib.sha256(preimage).hexdigest()


def realization_value_sha256(
    semantic_sha256: str,
    *,
    layout_fingerprint: str,
    backend_fingerprint: str,
) -> str:
    """Hash semantic identity with explicitly declared runtime fingerprints."""
    _validate_sha256(semantic_sha256, field_name="semantic_sha256")
    _require_nonempty_string(layout_fingerprint, field_name="layout_fingerprint")
    _require_nonempty_string(backend_fingerprint, field_name="backend_fingerprint")
    envelope = {
        "schema_id": _REALIZATION_PREIMAGE_SCHEMA_ID,
        "schema_version": _REALIZATION_PREIMAGE_SCHEMA_VERSION,
        "semantic_sha256": semantic_sha256,
        "layout_fingerprint": layout_fingerprint,
        "backend_fingerprint": backend_fingerprint,
    }
    return hashlib.sha256(training_spec_canonical_bytes(envelope)).hexdigest()


def value_identity_record(
    *,
    authored_sha256: str,
    value: object,
    dtype: str,
    inherited_authored_chain: Sequence[str] = (),
    expected_semantic_sha256: str | None = None,
    layout_fingerprint: str | None = None,
    backend_fingerprint: str | None = None,
) -> ValueIdentityRecord:
    """Build a strict identity record and fail closed on semantic drift.

    Runtime fingerprints must be supplied together. Their absence means no
    realization identity was requested; runtime facts are never inferred.
    """
    _validate_sha256(authored_sha256, field_name="authored_sha256")
    for identity in inherited_authored_chain:
        _validate_sha256(identity, field_name="inherited_authored_chain")
    if expected_semantic_sha256 is not None:
        _validate_sha256(
            expected_semantic_sha256,
            field_name="expected_semantic_sha256",
        )
    if (layout_fingerprint is None) != (backend_fingerprint is None):
        raise ValueError("layout_fingerprint and backend_fingerprint must be supplied together")

    semantic_sha256 = semantic_value_sha256(value, dtype=dtype)
    realization_sha256 = (
        realization_value_sha256(
            semantic_sha256,
            layout_fingerprint=layout_fingerprint,
            backend_fingerprint=backend_fingerprint,
        )
        if layout_fingerprint is not None and backend_fingerprint is not None
        else None
    )
    return ValueIdentityRecord(
        authored_sha256=authored_sha256,
        semantic_sha256=semantic_sha256,
        realization_sha256=realization_sha256,
        runtime_layout_fingerprint=layout_fingerprint,
        runtime_backend_fingerprint=backend_fingerprint,
        authored_identity_chain=(*inherited_authored_chain, authored_sha256),
        expected_semantic_sha256=expected_semantic_sha256,
    )


def _canonical_dtype(dtype: str) -> np.dtype[Any]:
    if not isinstance(dtype, str):
        raise TypeError("dtype must be a canonical dtype name")
    if dtype not in _CANONICAL_DTYPES:
        raise ValueError(
            f"unsupported or byte-order-ambiguous dtype {dtype!r}; "
            f"expected one of {sorted(_CANONICAL_DTYPES)!r}"
        )
    if dtype == "bool" or dtype.endswith("8"):
        return np.dtype(dtype)
    return np.dtype(dtype).newbyteorder("<")


def _require_nonempty_string(value: object, *, field_name: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty canonical string")


def _validate_sha256(value: object, *, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


__all__ = [
    "VALUE_IDENTITY_SCHEMA_ID",
    "VALUE_IDENTITY_SCHEMA_VERSION",
    "ValueIdentityRecord",
    "authored_value_sha256",
    "realization_value_sha256",
    "semantic_value_sha256",
    "value_identity_record",
]
