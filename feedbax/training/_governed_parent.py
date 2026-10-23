"""Shared authentication for governed training-row parent artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from feedbax.contracts.run_composition import (
    AuthoredIntentParent,
    ResolvedOutputParent,
    authored_envelope_hash,
    parse_composition_node,
)
from feedbax.contracts.run_matrix import TrainingRowParentProvenance
from feedbax.contracts.spec_storage import training_spec_sha256
from feedbax.contracts.strict_json import StrictJsonError, strict_json_loads


class _GovernedParentAuthenticationError(ValueError):
    """A governed parent failed byte or semantic authentication."""

    def __init__(self, message: str, *, byte_drift: bool = False) -> None:
        super().__init__(message)
        self.byte_drift = byte_drift


@dataclass(frozen=True)
class _AuthenticatedGovernedParent:
    provenance: TrainingRowParentProvenance
    parent: AuthoredIntentParent | ResolvedOutputParent
    payload: dict[str, Any]
    payload_bytes: bytes


def _authenticate_governed_training_row_parent(
    *,
    parent: AuthoredIntentParent | ResolvedOutputParent,
    provenance: TrainingRowParentProvenance,
    payload_bytes: bytes,
    expected_parent: AuthoredIntentParent | ResolvedOutputParent | None = None,
) -> _AuthenticatedGovernedParent:
    """Authenticate one exact typed declaration against immutable source bytes."""
    immutable_bytes = bytes(payload_bytes)
    observed_artifact_sha256 = hashlib.sha256(immutable_bytes).hexdigest()
    if observed_artifact_sha256 != provenance.artifact_sha256:
        raise _GovernedParentAuthenticationError(
            f"artifact {provenance.artifact_id!r} has byte hash "
            f"{observed_artifact_sha256} but its declaration pins "
            f"{provenance.artifact_sha256}",
            byte_drift=True,
        )

    try:
        decoded = strict_json_loads(immutable_bytes, ref=provenance.artifact_id)
    except (UnicodeDecodeError, json.JSONDecodeError, StrictJsonError) as exc:
        raise _GovernedParentAuthenticationError(
            f"artifact {provenance.artifact_id!r} is not strict JSON: {exc}",
            byte_drift=True,
        ) from exc
    if not isinstance(decoded, Mapping):
        raise _GovernedParentAuthenticationError(
            f"artifact {provenance.artifact_id!r} does not contain an object payload"
        )
    payload = dict(decoded)

    semantic_hash = (
        parent.content_hash
        if isinstance(parent, AuthoredIntentParent)
        else parent.resolved_root_hash
    )
    expected = parent if expected_parent is None else expected_parent
    if (
        parent != expected
        or provenance.parent_kind != parent.kind
        or provenance.ref != parent.ref
        or provenance.semantic_hash != semantic_hash
    ):
        raise _GovernedParentAuthenticationError(
            f"{expected.kind}:{expected.ref!r} was declared under different parent semantics"
        )

    schema_fields_present = "schema_id" in payload, "schema_version" in payload
    embedded_schema = payload.get("schema_id"), payload.get("schema_version")
    declared_schema = provenance.schema_id, provenance.schema_version
    if schema_fields_present == (False, False):
        if not isinstance(parent, ResolvedOutputParent):
            raise _GovernedParentAuthenticationError(
                f"artifact {provenance.artifact_id!r} omits the schema identity required for "
                "an authored_intent parent"
            )
    elif schema_fields_present != (True, True) or embedded_schema != declared_schema:
        raise _GovernedParentAuthenticationError(
            f"artifact {provenance.artifact_id!r} schema identity drifted from "
            f"{provenance.schema_id!r} at {provenance.schema_version!r}"
        )

    if isinstance(parent, AuthoredIntentParent):
        observed_semantic_hash = authored_envelope_hash(parse_composition_node(payload))
    else:
        observed_semantic_hash = training_spec_sha256(payload)
    if observed_semantic_hash != semantic_hash:
        raise _GovernedParentAuthenticationError(
            f"{parent.kind}:{parent.ref!r} semantic hash is {observed_semantic_hash}, but the "
            f"parent pins {semantic_hash}"
        )

    return _AuthenticatedGovernedParent(
        provenance=provenance.model_copy(deep=True),
        parent=parent.model_copy(deep=True),
        payload=payload,
        payload_bytes=immutable_bytes,
    )
