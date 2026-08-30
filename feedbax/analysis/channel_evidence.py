"""Fail-closed loading for custody-authenticated per-channel evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from types import MappingProxyType
from typing import Any

import numpy as np

from feedbax.analysis.evaluation import _resolve_staged_evaluation_prerequisite
from feedbax.analysis.execution_context import StagedExecutionContext
from feedbax.contracts.evaluation_states import EVALUATION_STATES_ARTIFACT_ROLE
from feedbax.contracts.manifest import StagedEvaluationPrerequisite
from feedbax.contracts.validation import validate_sha256


class EvaluationChannelEvidenceError(ValueError):
    """Raised when per-channel evidence does not authenticate loaded states."""


@dataclass(frozen=True, slots=True)
class EvaluationChannelEvidence:
    """Authenticated producer evidence for one channel array."""

    name: str
    index: int
    shape: tuple[int, ...]
    dtype: str
    byte_order: str
    c_contiguous: bool
    sha256: str


@dataclass(frozen=True, slots=True)
class AuthenticatedEvaluationChannels:
    """Custody-authenticated evaluation states and their channel evidence."""

    states: Any
    channels: Mapping[str, np.ndarray]
    evidence: Mapping[str, EvaluationChannelEvidence]
    manifest_sha256: str
    states_sha256: str


def resolve_authenticated_evaluation_channels(
    prerequisite: StagedEvaluationPrerequisite | Mapping[str, Any],
    *,
    execution_context: StagedExecutionContext,
) -> AuthenticatedEvaluationChannels:
    """Load custody-authenticated states and verify their per-channel evidence."""
    declared = StagedEvaluationPrerequisite.model_validate(prerequisite)
    prerequisite_result = _resolve_staged_evaluation_prerequisite(
        declared, execution_context
    )
    states = prerequisite_result.states
    cache_lookup = getattr(execution_context, "_cached_authenticated_channels", None)
    if cache_lookup is not None:
        cached = cache_lookup(states)
        if cached is not None:
            return cached
    resolved = prerequisite_result.manifest_input
    artifacts = [
        item
        for item in resolved.manifest.artifacts
        if item.role == EVALUATION_STATES_ARTIFACT_ROLE
    ]
    if len(artifacts) != 1 or not artifacts[0].sha256:
        raise EvaluationChannelEvidenceError(
            "authenticated channel states require exactly one SHA-pinned evaluation_states artifact"
        )
    if not isinstance(states, Mapping) or not isinstance(states.get("channels"), Mapping):
        raise EvaluationChannelEvidenceError(
            "authenticated evaluation states require a 'channels' mapping"
        )
    channels = states["channels"]
    raw_records = resolved.manifest.metadata.get("channels")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        raise EvaluationChannelEvidenceError(
            "evaluation manifest metadata.channels must be a sequence"
        )

    evidence: dict[str, EvaluationChannelEvidence] = {}
    for position, raw in enumerate(raw_records):
        if not isinstance(raw, Mapping):
            raise EvaluationChannelEvidenceError(
                f"evaluation channel evidence record {position} must be a mapping"
            )
        name = raw.get("name")
        if not isinstance(name, str) or not name or name in evidence:
            raise EvaluationChannelEvidenceError(
                f"evaluation channel evidence record {position} has invalid or duplicate name {name!r}"
            )
        evidence[name] = _authenticate_channel(
            name, raw, channels.get(name), expected_index=position
        )

    if set(channels) != set(evidence):
        raise EvaluationChannelEvidenceError(
            "evaluation channel keys disagree with manifest evidence: "
            f"states={sorted(channels)}, evidence={sorted(evidence)}"
        )
    authenticated = AuthenticatedEvaluationChannels(
        states=states,
        channels=MappingProxyType(dict(channels)),
        evidence=MappingProxyType(evidence),
        manifest_sha256=str(declared.parent.metadata["manifest_sha256"]),
        states_sha256=artifacts[0].sha256,
    )
    cache_store = getattr(execution_context, "_cache_authenticated_channels", None)
    if cache_store is not None:
        cache_store(states, authenticated)
    return authenticated


def _authenticate_channel(
    name: str,
    raw: Mapping[str, Any],
    value: Any,
    *,
    expected_index: int,
) -> EvaluationChannelEvidence:
    if not isinstance(value, np.ndarray):
        raise EvaluationChannelEvidenceError(f"channel {name!r} is missing or is not an array")
    expected_shape = raw.get("shape")
    if not isinstance(expected_shape, (list, tuple)) or any(
        not isinstance(size, int) or size < 0 for size in expected_shape
    ):
        raise EvaluationChannelEvidenceError(f"channel {name!r} has malformed evidence shape")
    index = raw.get("index")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise EvaluationChannelEvidenceError(f"channel {name!r} has malformed evidence index")
    if index != expected_index:
        raise EvaluationChannelEvidenceError(
            f"channel {name!r} evidence index mismatch: "
            f"expected={expected_index}, actual={index}"
        )
    expected = {
        "shape": value.shape,
        "dtype": value.dtype.str,
        "byte_order": value.dtype.str[0],
        "c_contiguous": True,
    }
    actual = {key: raw.get(key) for key in expected}
    actual["shape"] = tuple(actual["shape"]) if isinstance(actual["shape"], (list, tuple)) else actual["shape"]
    if actual != expected or not value.flags.c_contiguous:
        raise EvaluationChannelEvidenceError(
            f"channel {name!r} shape/dtype/byte-order/contiguity evidence mismatch: "
            f"expected={expected!r}, evidence={actual!r}, actual_c_contiguous={value.flags.c_contiguous!r}"
        )
    digest = raw.get("sha256")
    try:
        digest = validate_sha256(
            digest, field_name=f"channel {name!r} malformed evidence sha256"
        )
    except ValueError as exc:
        raise EvaluationChannelEvidenceError(str(exc)) from exc
    computed = hashlib.sha256(value.tobytes(order="C")).hexdigest()
    if digest != computed:
        raise EvaluationChannelEvidenceError(
            f"channel {name!r} SHA-256 mismatch: expected={digest!r}, computed={computed!r}"
        )
    return EvaluationChannelEvidence(
        name=name,
        index=index,
        shape=tuple(expected_shape),
        dtype=value.dtype.str,
        byte_order=value.dtype.str[0],
        c_contiguous=True,
        sha256=digest,
    )
