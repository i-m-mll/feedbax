"""Content-pinned whole-document composition for evaluation run matrices.

An :class:`EvaluationRunMatrixDeltaSpec` inherits one complete evaluation matrix
document — either a direct ``EvaluationRunMatrixSpec`` or another delta spec —
and applies ordered :class:`~feedbax.contracts.run_matrix.MatrixCompositionDelta`
layers to it. Flattening reuses the training-composition idioms: canonical hash
verification through the shared content-pinned loader, cycle detection over
authored envelope identities, ancestor-write acknowledgement enforcement, and
per-path layer attribution. Strict validation of the flattened terminal document
belongs to the evaluation authoring surface that owns the matrix model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from feedbax.contracts._parent_delta import _flatten_content_pinned_parent_deltas
from feedbax.contracts.base import (
    StrictModel,
    canonical_json_bytes,
    sha256_bytes,
)
from feedbax.contracts.manifest import (
    EVALUATION_MATRIX_COMPOSITION_PROVENANCE_SCHEMA_ID,
    EVALUATION_MATRIX_COMPOSITION_PROVENANCE_SCHEMA_VERSION,
    EVALUATION_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID,
    EVALUATION_RUN_MATRIX_DELTA_SPEC_SCHEMA_VERSION,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID,
)
from feedbax.contracts.matrix_core import ContentPinnedJsonBase, load_content_pinned_json_base
from feedbax.contracts.migrations import default_spec_registry
from feedbax.contracts.run_matrix import MatrixCompositionDelta


class EvaluationRunMatrixDeltaSpec(StrictModel):
    """Ordered whole-matrix deltas over one content-pinned parent document."""

    schema_id: str = EVALUATION_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID
    schema_version: str = EVALUATION_RUN_MATRIX_DELTA_SPEC_SCHEMA_VERSION
    parent: ContentPinnedJsonBase
    deltas: list[MatrixCompositionDelta] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_delta_spec(self) -> "EvaluationRunMatrixDeltaSpec":
        if self.schema_id != EVALUATION_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID:
            raise ValueError(
                f"unsupported EvaluationRunMatrixDeltaSpec schema_id {self.schema_id!r}"
            )
        if self.schema_version != EVALUATION_RUN_MATRIX_DELTA_SPEC_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported EvaluationRunMatrixDeltaSpec schema_version {self.schema_version!r}"
            )
        layer_ids = [delta.layer_id for delta in self.deltas]
        if len(layer_ids) != len(set(layer_ids)):
            raise ValueError("deltas layer_id values must be unique")
        return self


class EvaluationMatrixCompositionLayer(StrictModel):
    """Identity of one authored layer and the parent document it inherits.

    ``parent_payload_path`` records the JSON-pointer-lite sub-document selector
    when the pinned parent inherits only a sub-document; it is absent (``None``)
    for whole-file parents so existing provenance stays byte-identical.
    """

    envelope_sha256: str
    parent_ref: str
    parent_sha256: str
    layer_ids: list[str]
    parent_payload_path: tuple[str, ...] | None = None


class FlattenedEvaluationMatrix(StrictModel):
    """Flattened matrix document with ordered layer identities and attribution."""

    authored: dict[str, Any]
    payload: dict[str, Any]
    attribution: dict[str, str]
    layers: list[EvaluationMatrixCompositionLayer] = Field(min_length=1)

    @property
    def authored_envelope_sha256(self) -> str:
        """Identity of the authored child document that was flattened."""
        return self.layers[-1].envelope_sha256

    @property
    def root_matrix(self) -> EvaluationMatrixCompositionLayer:
        """Layer whose pinned parent is the terminal direct evaluation matrix."""
        return self.layers[0]


def is_evaluation_run_matrix_delta_payload(payload: Any) -> bool:
    """Return whether a mapping declares the delta authoring schema identity."""
    return (
        isinstance(payload, Mapping)
        and payload.get("schema_id") == EVALUATION_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID
    )


def evaluation_matrix_delta_envelope(spec: EvaluationRunMatrixDeltaSpec) -> dict[str, Any]:
    """Return the identity-bearing authored composition of a delta spec.

    The parent ``ref`` is a readability locator; the pinned ``sha256`` carries the
    parent's identity, so the ref is excluded exactly as in training composition.
    """
    parent = spec.parent.model_dump(mode="json", exclude_none=True)
    parent.pop("ref", None)
    return {
        "schema_id": spec.schema_id,
        "schema_version": spec.schema_version,
        "parent": parent,
        "deltas": [delta.model_dump(mode="json", exclude_none=True) for delta in spec.deltas],
    }


def evaluation_matrix_delta_envelope_hash(spec: EvaluationRunMatrixDeltaSpec) -> str:
    """Return the canonical authored envelope identity of a delta spec."""
    return sha256_bytes(canonical_json_bytes(evaluation_matrix_delta_envelope(spec)))


def flatten_evaluation_run_matrix_delta(
    spec: EvaluationRunMatrixDeltaSpec,
    *,
    repo_root: Path | str | None = None,
) -> FlattenedEvaluationMatrix:
    """Resolve pinned parents root-to-child into one flattened matrix document."""
    flattened = _flatten_content_pinned_parent_deltas(
        spec,
        repo_root=repo_root,
        envelope_hash=evaluation_matrix_delta_envelope_hash,
        parent_of=lambda node: node.parent,
        load_parent=load_content_pinned_json_base,
        deltas_of=lambda node: node.deltas,
        parse_delta_parent=EvaluationRunMatrixDeltaSpec.model_validate,
        terminal_payload=lambda payload: default_spec_registry.migrate(
            "EvaluationRunMatrixSpec", payload
        ).payload,
        layer_from_node=lambda digest, node: EvaluationMatrixCompositionLayer(
            envelope_sha256=digest,
            parent_ref=node.parent.ref,
            parent_sha256=node.parent.sha256,
            layer_ids=[delta.layer_id for delta in node.deltas],
            parent_payload_path=node.parent.payload_path,
        ),
        delta_schema_id=EVALUATION_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID,
        terminal_schema_id=EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID,
        cycle_error="evaluation matrix delta parent composition cycle detected",
        invalid_parent_error=lambda schema_id: (
            "evaluation matrix delta parent must declare schema_id "
            f"{EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID!r} or "
            f"{EVALUATION_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID!r}, got {schema_id!r}"
        ),
    )
    return FlattenedEvaluationMatrix(
        authored=spec.model_dump(mode="json", exclude_none=True),
        payload=flattened.payload,
        attribution=flattened.attribution,
        layers=flattened.layers,
    )


def evaluation_matrix_composition_provenance(
    flattened: FlattenedEvaluationMatrix,
    *,
    flattened_matrix: Mapping[str, Any],
    canonical_row_order: Sequence[str],
    canonical_payload_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Build the single canonical composition-provenance record for a matrix run."""
    return {
        "schema_id": EVALUATION_MATRIX_COMPOSITION_PROVENANCE_SCHEMA_ID,
        "schema_version": EVALUATION_MATRIX_COMPOSITION_PROVENANCE_SCHEMA_VERSION,
        "authored_envelope_sha256": flattened.authored_envelope_sha256,
        "root_matrix": {
            "ref": flattened.root_matrix.parent_ref,
            "sha256": flattened.root_matrix.parent_sha256,
        },
        "layers": [layer.model_dump(mode="json", exclude_none=True) for layer in flattened.layers],
        "attribution": dict(flattened.attribution),
        "flattened_matrix_sha256": sha256_bytes(canonical_json_bytes(dict(flattened_matrix))),
        "canonical_row_order": list(canonical_row_order),
        "canonical_payload_sha256": dict(canonical_payload_sha256),
    }


__all__ = [
    "EvaluationMatrixCompositionLayer",
    "EvaluationRunMatrixDeltaSpec",
    "FlattenedEvaluationMatrix",
    "evaluation_matrix_composition_provenance",
    "evaluation_matrix_delta_envelope",
    "evaluation_matrix_delta_envelope_hash",
    "flatten_evaluation_run_matrix_delta",
    "is_evaluation_run_matrix_delta_payload",
]
