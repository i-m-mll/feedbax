"""Role-addressed array artifact schema and NPZ storage helpers.

Feedbax model artifacts separate graph structure from learned/runtime arrays.
This module stores arrays under semantic role addresses while allowing the
physical storage backend to remain replaceable and versioned.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, field_validator

from feedbax.contracts.manifest import ArrayStoreRef, StrictModel, sha256_bytes, sha256_file


ARRAY_STORE_SCHEMA_VERSION = "feedbax.manifest.array_store.v1"
ROLE_SCHEMA_VERSION = "feedbax.manifest.array_roles.v1"
NPZ_BACKEND_VERSION = "npz.v1"
METADATA_KEY = "__feedbax_array_store__.json"
ARRAY_KEY_TEMPLATE = "array_{index:06d}"
_ROLE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]*$")


class ArtifactSchemaError(ValueError):
    """Base class for durable artifact schema errors."""


class RoleAddressError(ArtifactSchemaError):
    """Raised when an array role address is not valid."""


class ArrayStoreValidationError(ArtifactSchemaError):
    """Raised when a store is missing roles or has inconsistent metadata."""


def validate_role_address(role: str) -> str:
    """Validate and return a semantic array role address."""
    if not isinstance(role, str) or not role:
        raise RoleAddressError("Array role address must be a non-empty string.")
    if role.startswith("__feedbax_"):
        raise RoleAddressError(f"Array role address {role!r} is reserved for Feedbax metadata.")
    if not _ROLE_PATTERN.match(role):
        raise RoleAddressError(f"Array role address {role!r} contains unsupported characters.")
    return role


class ArrayRecord(StrictModel):
    """Metadata for one role-addressed array inside an array store."""

    role: str
    storage_key: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str
    role_schema_version: str = ROLE_SCHEMA_VERSION
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("role")
    @classmethod
    def _validate_role(cls, value: str) -> str:
        return validate_role_address(value)


class ArrayStorePayload(StrictModel):
    """Portable metadata payload for a role-addressed array store."""

    kind: Literal["ArrayStore"] = "ArrayStore"
    schema_version: str = ARRAY_STORE_SCHEMA_VERSION
    storage_backend: Literal["npz.v1"] = NPZ_BACKEND_VERSION
    store_role: Literal["params", "state", "optimizer", "history"]
    arrays: list[ArrayRecord]
    graph_spec_ref: str | None = None
    graph_spec_sha256: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def roles(self) -> list[str]:
        """Return semantic roles in deterministic store order."""
        return [record.role for record in self.arrays]


class ArrayStore:
    """Loaded role-addressed array store."""

    def __init__(self, payload: ArrayStorePayload, arrays: Mapping[str, np.ndarray]):
        self.payload = payload
        self.arrays = dict(arrays)

    def validate_roles(
        self,
        *,
        required_roles: Sequence[str] | None = None,
        exact_roles: Sequence[str] | None = None,
    ) -> None:
        """Validate required or exact role coverage for this store."""
        validate_role_coverage(
            self.payload.roles,
            required_roles=required_roles,
            exact_roles=exact_roles,
        )


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return sha256_bytes(contiguous.tobytes(order="C"))


def _metadata_bytes(payload: ArrayStorePayload) -> bytes:
    data = payload.model_dump(mode="json", exclude_none=True)
    return json.dumps(data, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def validate_role_coverage(
    roles: Sequence[str],
    *,
    required_roles: Sequence[str] | None = None,
    exact_roles: Sequence[str] | None = None,
) -> None:
    """Validate role-address coverage with deterministic failure messages."""
    role_set = {validate_role_address(role) for role in roles}
    if len(role_set) != len(roles):
        raise ArrayStoreValidationError("Array role addresses must be unique.")

    if required_roles is not None:
        required = {validate_role_address(role) for role in required_roles}
        missing = sorted(required - role_set)
        if missing:
            raise ArrayStoreValidationError(f"Array store missing required roles: {missing}")

    if exact_roles is not None:
        expected = {validate_role_address(role) for role in exact_roles}
        missing = sorted(expected - role_set)
        unexpected = sorted(role_set - expected)
        if missing or unexpected:
            raise ArrayStoreValidationError(
                f"Array store role mismatch: missing={missing}, unexpected={unexpected}"
            )


def write_npz_array_store(
    path: Path | str,
    arrays: Mapping[str, Any],
    *,
    store_role: Literal["params", "state", "optimizer", "history"],
    graph_spec_ref: str | None = None,
    graph_spec_sha256: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ArrayStorePayload:
    """Write arrays to an NPZ file under deterministic semantic role metadata."""
    if not arrays:
        raise ArrayStoreValidationError("Array store must contain at least one array.")

    records: list[ArrayRecord] = []
    encoded_arrays: dict[str, np.ndarray] = {}
    for index, (role, value) in enumerate(sorted(arrays.items())):
        valid_role = validate_role_address(role)
        storage_key = ARRAY_KEY_TEMPLATE.format(index=index)
        array = np.asarray(value)
        encoded_arrays[storage_key] = array
        records.append(
            ArrayRecord(
                role=valid_role,
                storage_key=storage_key,
                dtype=str(array.dtype),
                shape=tuple(int(dim) for dim in array.shape),
                sha256=_array_digest(array),
            )
        )

    payload = ArrayStorePayload(
        store_role=store_role,
        arrays=records,
        graph_spec_ref=graph_spec_ref,
        graph_spec_sha256=graph_spec_sha256,
        metadata=dict(metadata or {}),
    )
    encoded_arrays[METADATA_KEY] = np.asarray(_metadata_bytes(payload))

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **encoded_arrays)
    return payload


def read_npz_array_store(path: Path | str) -> ArrayStore:
    """Read and validate an NPZ role-addressed array store."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as npz:
        if METADATA_KEY not in npz.files:
            raise ArrayStoreValidationError(
                f"NPZ array store is missing metadata member {METADATA_KEY!r}."
            )
        raw_metadata = npz[METADATA_KEY].tobytes().decode("utf-8")
        payload = ArrayStorePayload.model_validate_json(raw_metadata)
        arrays: dict[str, np.ndarray] = {}
        seen_storage_keys: set[str] = set()
        for record in payload.arrays:
            if record.storage_key not in npz.files:
                raise ArrayStoreValidationError(
                    f"Array role {record.role!r} is missing storage key {record.storage_key!r}."
                )
            array = np.asarray(npz[record.storage_key])
            seen_storage_keys.add(record.storage_key)
            if str(array.dtype) != record.dtype or tuple(array.shape) != record.shape:
                raise ArrayStoreValidationError(
                    f"Array role {record.role!r} metadata mismatch: "
                    f"expected dtype={record.dtype}, shape={record.shape}; "
                    f"found dtype={array.dtype}, shape={tuple(array.shape)}."
                )
            digest = _array_digest(array)
            if digest != record.sha256:
                raise ArrayStoreValidationError(f"Array role {record.role!r} digest mismatch.")
            arrays[record.role] = array

        extra_keys = sorted(set(npz.files) - seen_storage_keys - {METADATA_KEY})
        if extra_keys:
            raise ArrayStoreValidationError(f"NPZ array store has unknown members: {extra_keys}")

    validate_role_coverage(payload.roles)
    return ArrayStore(payload, arrays)


def array_store_ref(
    path: Path | str,
    payload: ArrayStorePayload,
    *,
    logical_name: str | None = None,
    artifact_id: str | None = None,
) -> ArrayStoreRef:
    """Build an ``ArrayStoreRef`` for a written array-store file."""
    source = Path(path)
    return ArrayStoreRef(
        role=payload.store_role,
        schema_version=payload.schema_version,
        storage_backend=payload.storage_backend,
        logical_name=logical_name or source.name,
        artifact_id=artifact_id,
        sha256=sha256_file(source),
        uri=str(source),
        array_count=len(payload.arrays),
        roles=payload.roles,
    )
