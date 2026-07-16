"""Canonical three-layer storage contracts for training-run specifications."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import Field, model_validator

from feedbax.contracts.manifest import ArtifactRef, StrictModel
from feedbax.contracts.resolved_snapshot_decoder import (
    SNAPSHOT_SCHEMA_ID,
    SNAPSHOT_SCHEMA_VERSION,
)
from feedbax.contracts.validation import validate_sha256


TRAINING_RUN_EXECUTION_CAPSULE_SCHEMA_ID = "feedbax.manifest.training_run_execution_capsule"
TRAINING_RUN_EXECUTION_CAPSULE_SCHEMA_VERSION = (
    f"{TRAINING_RUN_EXECUTION_CAPSULE_SCHEMA_ID}.v1"
)


def canonicalize_immutable_input_identities(
    identities: Sequence[Any],
) -> list[dict[str, Any]]:
    """Return the canonical JSON projection of immutable execution inputs.

    Inputs may be mappings or Pydantic models. Records are sorted by their
    complete v1 identity tuple. Repeating an input identity is rejected rather
    than silently changing the scientific identity through deduplication.
    """
    projected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for index, identity in enumerate(identities):
        if hasattr(identity, "model_dump"):
            raw = identity.model_dump(mode="json", exclude_none=True)
        elif isinstance(identity, Mapping):
            raw = dict(identity)
        else:
            raise TypeError(
                f"immutable input identity {index} must be a mapping or Pydantic model"
            )
        normalized = _normalize_json(raw)
        if not isinstance(normalized, dict):  # pragma: no cover - defensive
            raise TypeError(f"immutable input identity {index} must project to an object")
        try:
            role = normalized["role"]
            kind = normalized["kind"]
            identifier = normalized["identifier"]
            digest = normalized["digest"]
        except KeyError as exc:
            raise ValueError(
                f"immutable input identity {index} is missing {exc.args[0]!r}"
            ) from exc
        if not all(isinstance(value, str) and value for value in (role, kind, identifier)):
            raise ValueError(
                f"immutable input identity {index} requires non-empty role, kind, identifier"
            )
        if not isinstance(digest, dict):
            raise ValueError(f"immutable input identity {index} digest must be an object")
        algorithm = digest.get("algorithm")
        value = digest.get("value")
        if algorithm != "sha256" or not isinstance(value, str):
            raise ValueError(
                f"immutable input identity {index} digest must use sha256 with a string value"
            )
        validate_sha256(value, field_name=f"immutable_inputs[{index}].digest.value")
        key = (role, kind, identifier)
        if key in seen:
            raise ValueError(
                "duplicate immutable input identity for "
                f"role={role!r}, kind={kind!r}, identifier={identifier!r}"
            )
        seen.add(key)
        projected.append(normalized)
    return sorted(
        projected,
        key=lambda item: (
            item["role"],
            item["kind"],
            item["identifier"],
            item["digest"]["algorithm"],
            item["digest"]["value"],
        ),
    )


class TrainingRunExecutionCapsule(StrictModel):
    """Immutable execution and provenance record joining intent to semantics."""

    schema_id: str = TRAINING_RUN_EXECUTION_CAPSULE_SCHEMA_ID
    schema_version: str = TRAINING_RUN_EXECUTION_CAPSULE_SCHEMA_VERSION
    materializer_commit: str
    relevant_schema_versions: dict[str, str] = Field(default_factory=dict)
    dependency_lock_digest: str
    environment_digest: str | None = None
    input_data_identities: list[dict[str, Any]] = Field(default_factory=list)
    intent_hash: str
    resolved_root_hash: str
    execution_hash: str | None = None

    @model_validator(mode="after")
    def _validate_identity(self) -> "TrainingRunExecutionCapsule":
        if self.schema_id != TRAINING_RUN_EXECUTION_CAPSULE_SCHEMA_ID:
            raise ValueError("unsupported training-run execution capsule schema_id")
        if self.schema_version != TRAINING_RUN_EXECUTION_CAPSULE_SCHEMA_VERSION:
            raise ValueError("unsupported training-run execution capsule schema_version")
        if not self.materializer_commit.strip():
            raise ValueError("materializer_commit must not be empty")
        if not self.relevant_schema_versions:
            raise ValueError("relevant_schema_versions must not be empty")
        for field_name, value in (
            ("dependency_lock_digest", self.dependency_lock_digest),
            ("environment_digest", self.environment_digest),
            ("intent_hash", self.intent_hash),
            ("resolved_root_hash", self.resolved_root_hash),
            ("execution_hash", self.execution_hash),
        ):
            if value is not None:
                validate_sha256(value, field_name=field_name)
        _validate_nested_hash_fields(self.input_data_identities, path="input_data_identities")
        expected = training_run_execution_hash(
            self.resolved_root_hash,
            self.input_data_identities,
        )
        if self.execution_hash is not None and self.execution_hash != expected:
            raise ValueError(
                "materializer drift: archived execution_hash does not match resolved semantics; "
                f"archived={self.execution_hash!r}, computed={expected!r}"
            )
        self.execution_hash = expected
        return self


def training_spec_canonical_bytes(value: Any) -> bytes:
    """Encode new spec-storage surfaces with shortest round-trip-safe floats.

    Integers retain exact integer identity. Finite floats use Python's shortest
    correctly-round-tripping representation; negative zero is normalized to
    positive zero. Object keys must already be strings, preventing coercion
    collisions. Existing manifest/checkpoint canonicalizers remain unchanged.
    """
    normalized = _normalize_json(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def training_spec_sha256(value: Any) -> str:
    """Return the canonical content identity of a new spec-storage value."""
    return hashlib.sha256(training_spec_canonical_bytes(value)).hexdigest()


def training_run_intent_envelope(authored: Mapping[str, Any]) -> dict[str, Any]:
    """Return the explicit semantic allowlist hashed as authored intent."""
    envelope: dict[str, Any] = {
        key: authored[key]
        for key in ("schema_id", "schema_version")
        if key in authored
    }
    base = authored.get("base")
    if isinstance(base, Mapping):
        kind = base.get("kind")
        base_fields = {
            "inline": ("kind", "inline"),
            "authored_intent": ("kind", "content_hash", "pin_algorithm", "payload_path"),
            "resolved_output": ("kind", "resolved_root_hash", "payload_path"),
        }.get(kind, ("kind",))
        envelope["base"] = {key: base[key] for key in base_fields if key in base}
    envelope["sources"] = [
        {
            key: source[key]
            for key in ("alias", "kind", "payload_query", "optional", "missing_payload")
            if key in source
        }
        for source in authored.get("sources", [])
        if isinstance(source, Mapping)
    ]
    envelope["derivations"] = [
        {key: item[key] for key in ("output_path", "query") if key in item}
        for item in authored.get("derivations", [])
        if isinstance(item, Mapping)
    ]
    envelope["rows"] = [
        {key: row[key] for key in ("row_id", "overrides", "seed") if key in row}
        for row in authored.get("rows", [])
        if isinstance(row, Mapping)
    ]
    envelope["axes"] = [
        {
            key: axis[key]
            for key in ("id", "path", "variation", "role", "authored_parameter")
            if key in axis
        }
        for axis in authored.get("axes", [])
        if isinstance(axis, Mapping)
    ]
    combination = authored.get("combination")
    if isinstance(combination, Mapping):
        envelope["combination"] = {
            key: combination[key]
            for key in ("mode", "groups", "manual_coordinates")
            if key in combination
        }
    fork = authored.get("fork")
    if isinstance(fork, Mapping):
        envelope["fork"] = {
            key: fork[key]
            for key in ("lr_continuation", "parity", "expected_slots")
            if key in fork
        }
    return envelope


def training_run_intent_hash(authored: Mapping[str, Any]) -> str:
    """Hash authored matrix intent, retaining pins while excluding names and URIs."""
    return training_spec_sha256(training_run_intent_envelope(authored))


def training_run_authored_envelope(authored: Mapping[str, Any]) -> dict[str, Any]:
    """Return identity-bearing authored composition, excluding symbolic locators."""
    envelope = training_run_intent_envelope(authored)
    envelope["deltas"] = [
        {
            key: delta[key]
            for key in (
                "layer_id",
                "patches",
                "acknowledges_ancestor_paths",
                "schema_id",
                "schema_version",
            )
            if key in delta
        }
        for delta in authored.get("deltas", [])
        if isinstance(delta, Mapping)
    ]
    envelope["execution_dependencies"] = list(
        authored.get("execution_dependencies", [])
    )
    return envelope


def training_run_authored_envelope_hash(authored: Mapping[str, Any]) -> str:
    """Hash ordered pinned parents, local deltas, selectors, seeds, and sources."""
    return training_spec_sha256(training_run_authored_envelope(authored))


def training_run_composed_intent_hash(flattened: Mapping[str, Any]) -> str:
    """Hash canonical effective intent before execution evidence is attached."""
    return training_spec_sha256(flattened)


def training_run_execution_hash(
    resolved_root_hash: str,
    input_data_identities: list[dict[str, Any]],
) -> str:
    """Hash exact resolved semantics and immutable external input identities."""
    return training_spec_sha256(
        {
            "resolved_root_hash": resolved_root_hash,
            "input_data_identities": input_data_identities,
        }
    )


def _validate_nested_hash_fields(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key == "sha256" or key.endswith(("_hash", "_digest")):
                if not isinstance(child, str):
                    raise ValueError(f"{child_path} must be a sha256 string")
                validate_sha256(child, field_name=child_path)
            else:
                _validate_nested_hash_fields(child, path=child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_nested_hash_fields(child, path=f"{path}[{index}]")


def build_resolved_semantics_snapshot(value: Any) -> dict[str, Any]:
    """Build a structurally shared content-addressed node table for a JSON tree."""
    nodes: dict[str, dict[str, Any]] = {}

    def intern(item: Any) -> str:
        if isinstance(item, Mapping):
            if not all(isinstance(key, str) for key in item):
                raise TypeError("resolved-semantics object keys must all be strings")
            node = {
                "type": "object",
                "children": {str(key): intern(item[key]) for key in sorted(item)},
            }
        elif isinstance(item, (list, tuple)):
            node = {"type": "array", "children": [intern(child) for child in item]}
        else:
            node = {"type": "scalar", "value": _normalize_json(item)}
        digest = training_spec_sha256(node)
        nodes.setdefault(digest, node)
        return digest

    root_hash = intern(value)
    return {
        "schema_id": SNAPSHOT_SCHEMA_ID,
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "root_hash": root_hash,
        "nodes": nodes,
    }


def store_canonical_json_artifact(
    value: Any,
    *,
    root: Path,
    role: str,
    logical_name: str,
) -> ArtifactRef:
    """Store canonical JSON in a SHA-256-sharded, write-if-absent custody tree."""
    data = training_spec_canonical_bytes(value)
    digest = hashlib.sha256(data).hexdigest()
    destination = root / "spec-storage" / "sha256" / digest[:2] / f"{digest}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != data:
            raise ValueError(
                f"custody digest path exists with mismatched bytes: {destination}"
            )
    else:
        destination.write_bytes(data)
    return ArtifactRef(
        role=role,
        logical_name=logical_name,
        artifact_id=f"artifact://sha256/{digest}",
        sha256=digest,
        media_type="application/json",
        size_bytes=len(data),
        uri=str(destination),
        metadata={"relative_path": str(destination.relative_to(root))},
    )


def _normalize_json(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical training specs require finite numbers")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical training spec object keys must all be strings")
        return {key: _normalize_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(child) for child in value]
    if hasattr(value, "model_dump"):
        return _normalize_json(value.model_dump(mode="json", exclude_none=True))
    raise TypeError(f"value is not JSON-compatible: {type(value).__name__}")
