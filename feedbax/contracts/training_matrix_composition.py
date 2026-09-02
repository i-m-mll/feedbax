"""Content-pinned whole-document composition for training run matrices."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from feedbax.contracts.base import (
    StrictModel,
    canonical_json_bytes,
    sha256_bytes,
)
from feedbax.contracts.manifest import (
    TRAINING_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID,
    TRAINING_RUN_MATRIX_DELTA_SPEC_SCHEMA_VERSION,
)
from feedbax.contracts.matrix_core import ContentPinnedJsonBase, load_content_pinned_json_base
from feedbax.contracts.run_matrix import (
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
    MatrixCompositionDelta,
    TrainingRunMatrixSpec,
    apply_composition_deltas,
)


class TrainingRunMatrixDeltaSpec(StrictModel):
    """Ordered whole-document deltas over one content-pinned training matrix."""

    schema_id: str = TRAINING_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID
    schema_version: str = TRAINING_RUN_MATRIX_DELTA_SPEC_SCHEMA_VERSION
    parent: ContentPinnedJsonBase
    deltas: list[MatrixCompositionDelta] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_delta_spec(self) -> "TrainingRunMatrixDeltaSpec":
        if self.schema_id != TRAINING_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID:
            raise ValueError(
                f"unsupported TrainingRunMatrixDeltaSpec schema_id {self.schema_id!r}"
            )
        if self.schema_version != TRAINING_RUN_MATRIX_DELTA_SPEC_SCHEMA_VERSION:
            raise ValueError(
                "unsupported TrainingRunMatrixDeltaSpec schema_version "
                f"{self.schema_version!r}"
            )
        layer_ids = [delta.layer_id for delta in self.deltas]
        if len(layer_ids) != len(set(layer_ids)):
            raise ValueError("deltas layer_id values must be unique")
        return self


class TrainingMatrixCompositionLayer(StrictModel):
    """Identity of one authored matrix layer and its pinned parent."""

    envelope_sha256: str
    parent_ref: str
    parent_sha256: str
    layer_ids: list[str]
    parent_payload_path: tuple[str, ...] | None = None


class FlattenedTrainingMatrix(StrictModel):
    """Flattened training matrix with ordered identities and attribution."""

    authored: dict[str, Any]
    payload: dict[str, Any]
    attribution: dict[str, str]
    layers: list[TrainingMatrixCompositionLayer] = Field(min_length=1)

    @property
    def authored_envelope_sha256(self) -> str:
        """Identity of the authored child delta document."""
        return self.layers[-1].envelope_sha256

    @property
    def root_matrix(self) -> TrainingMatrixCompositionLayer:
        """Layer whose pinned parent is the terminal direct matrix."""
        return self.layers[0]


def is_training_run_matrix_delta_payload(payload: Any) -> bool:
    """Return whether a mapping declares the training-matrix delta schema."""
    return (
        isinstance(payload, Mapping)
        and payload.get("schema_id") == TRAINING_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID
    )


def training_matrix_delta_envelope(spec: TrainingRunMatrixDeltaSpec) -> dict[str, Any]:
    """Return the identity-bearing authored composition of a matrix delta."""
    parent = spec.parent.model_dump(mode="json", exclude_none=True)
    parent.pop("ref", None)
    return {
        "schema_id": spec.schema_id,
        "schema_version": spec.schema_version,
        "parent": parent,
        "deltas": [delta.model_dump(mode="json", exclude_none=True) for delta in spec.deltas],
    }


def training_matrix_delta_envelope_hash(spec: TrainingRunMatrixDeltaSpec) -> str:
    """Return the canonical authored envelope identity of a matrix delta."""
    return sha256_bytes(canonical_json_bytes(training_matrix_delta_envelope(spec)))


def flatten_training_run_matrix_delta(
    spec: TrainingRunMatrixDeltaSpec,
    *,
    repo_root: Path | str | None = None,
) -> FlattenedTrainingMatrix:
    """Resolve pinned parents root-to-child and validate the terminal matrix."""
    chain: list[tuple[str, TrainingRunMatrixDeltaSpec]] = []
    seen: set[str] = set()
    current = spec
    while True:
        digest = training_matrix_delta_envelope_hash(current)
        if digest in seen:
            raise ValueError("training run matrix delta parent composition cycle detected")
        seen.add(digest)
        chain.append((digest, current))
        parent_payload = load_content_pinned_json_base(current.parent, repo_root=repo_root)
        parent_schema_id = parent_payload.get("schema_id")
        if parent_schema_id == TRAINING_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID:
            current = TrainingRunMatrixDeltaSpec.model_validate(parent_payload)
            continue
        if parent_schema_id != TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID:
            raise ValueError(
                "training run matrix delta parent must declare schema_id "
                f"{TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID!r} or "
                f"{TRAINING_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID!r}, got {parent_schema_id!r}"
            )
        from feedbax.contracts.migrations import default_spec_registry

        payload = default_spec_registry.migrate(
            "TrainingRunMatrixSpec", parent_payload
        ).payload
        break

    attribution: dict[str, str] = {}
    written: set[str] = set()
    layers: list[TrainingMatrixCompositionLayer] = []
    for digest, node in reversed(chain):
        payload, local_attribution, written = apply_composition_deltas(
            payload, node.deltas, ancestor_written_paths=written
        )
        attribution.update(local_attribution)
        layers.append(
            TrainingMatrixCompositionLayer(
                envelope_sha256=digest,
                parent_ref=node.parent.ref,
                parent_sha256=node.parent.sha256,
                layer_ids=[delta.layer_id for delta in node.deltas],
                parent_payload_path=node.parent.payload_path,
            )
        )
    payload = TrainingRunMatrixSpec.model_validate(payload).model_dump(
        mode="json", exclude_none=True
    )
    return FlattenedTrainingMatrix(
        authored=spec.model_dump(mode="json", exclude_none=True),
        payload=payload,
        attribution=attribution,
        layers=layers,
    )


__all__ = [
    "FlattenedTrainingMatrix",
    "TrainingMatrixCompositionLayer",
    "TrainingRunMatrixDeltaSpec",
    "flatten_training_run_matrix_delta",
    "is_training_run_matrix_delta_payload",
    "training_matrix_delta_envelope",
    "training_matrix_delta_envelope_hash",
]
