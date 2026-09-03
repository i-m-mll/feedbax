"""Content-pinned whole-document composition for analysis run specs.

An :class:`AnalysisRunDeltaSpec` inherits one complete analysis-run document —
either a direct ``AnalysisRunSpec`` or another delta spec — and applies ordered
:class:`~feedbax.contracts.run_matrix.MatrixCompositionDelta` layers to it. The
flattening reuses the evaluation-matrix composition idioms exactly: canonical
hash verification through the shared content-pinned loader, cycle detection over
authored envelope identities, ancestor-write acknowledgement enforcement, and
per-path layer attribution. Strict validation of the flattened terminal document
belongs to the analysis authoring surface that owns ``AnalysisRunSpec``.
"""

from __future__ import annotations

from collections.abc import Mapping
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
    ANALYSIS_COMPOSITION_PROVENANCE_SCHEMA_ID,
    ANALYSIS_COMPOSITION_PROVENANCE_SCHEMA_VERSION,
    ANALYSIS_RUN_DELTA_SPEC_SCHEMA_ID,
    ANALYSIS_RUN_DELTA_SPEC_SCHEMA_VERSION,
    ANALYSIS_RUN_SPEC_SCHEMA_ID,
)
from feedbax.contracts.matrix_core import ContentPinnedJsonBase, load_content_pinned_json_base
from feedbax.contracts.migrations import default_spec_registry
from feedbax.contracts.run_matrix import MatrixCompositionDelta


class AnalysisRunDeltaSpec(StrictModel):
    """Ordered whole-document deltas over one content-pinned parent analysis spec."""

    schema_id: str = ANALYSIS_RUN_DELTA_SPEC_SCHEMA_ID
    schema_version: str = ANALYSIS_RUN_DELTA_SPEC_SCHEMA_VERSION
    parent: ContentPinnedJsonBase
    deltas: list[MatrixCompositionDelta] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_delta_spec(self) -> "AnalysisRunDeltaSpec":
        if self.schema_id != ANALYSIS_RUN_DELTA_SPEC_SCHEMA_ID:
            raise ValueError(f"unsupported AnalysisRunDeltaSpec schema_id {self.schema_id!r}")
        if self.schema_version != ANALYSIS_RUN_DELTA_SPEC_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported AnalysisRunDeltaSpec schema_version {self.schema_version!r}"
            )
        layer_ids = [delta.layer_id for delta in self.deltas]
        if len(layer_ids) != len(set(layer_ids)):
            raise ValueError("deltas layer_id values must be unique")
        return self


class AnalysisCompositionLayer(StrictModel):
    """Identity of one authored layer and the parent document it inherits.

    ``parent_payload_path`` records the JSON-pointer-lite sub-document selector
    when the pinned parent inherits only a sub-document; it is absent (``None``)
    for whole-file parents.
    """

    envelope_sha256: str
    parent_ref: str
    parent_sha256: str
    layer_ids: list[str]
    parent_payload_path: tuple[str, ...] | None = None


class FlattenedAnalysisRun(StrictModel):
    """Flattened analysis-run document with ordered layer identities and attribution."""

    authored: dict[str, Any]
    payload: dict[str, Any]
    attribution: dict[str, str]
    layers: list[AnalysisCompositionLayer] = Field(min_length=1)

    @property
    def authored_envelope_sha256(self) -> str:
        """Identity of the authored child document that was flattened."""
        return self.layers[-1].envelope_sha256

    @property
    def root_spec(self) -> AnalysisCompositionLayer:
        """Layer whose pinned parent is the terminal direct analysis spec."""
        return self.layers[0]


def is_analysis_run_delta_payload(payload: Any) -> bool:
    """Return whether a mapping declares the delta authoring schema identity."""
    return (
        isinstance(payload, Mapping)
        and payload.get("schema_id") == ANALYSIS_RUN_DELTA_SPEC_SCHEMA_ID
    )


def analysis_run_delta_envelope(spec: AnalysisRunDeltaSpec) -> dict[str, Any]:
    """Return the identity-bearing authored composition of a delta spec.

    The parent ``ref`` is a readability locator; the pinned ``sha256`` carries the
    parent's identity, so the ref is excluded exactly as in matrix composition.
    """
    parent = spec.parent.model_dump(mode="json", exclude_none=True)
    parent.pop("ref", None)
    return {
        "schema_id": spec.schema_id,
        "schema_version": spec.schema_version,
        "parent": parent,
        "deltas": [delta.model_dump(mode="json", exclude_none=True) for delta in spec.deltas],
    }


def analysis_run_delta_envelope_hash(spec: AnalysisRunDeltaSpec) -> str:
    """Return the canonical authored envelope identity of a delta spec."""
    return sha256_bytes(canonical_json_bytes(analysis_run_delta_envelope(spec)))


def flatten_analysis_run_delta(
    spec: AnalysisRunDeltaSpec,
    *,
    repo_root: Path | str | None = None,
) -> FlattenedAnalysisRun:
    """Resolve pinned parents root-to-child into one flattened analysis document."""
    flattened = _flatten_content_pinned_parent_deltas(
        spec,
        repo_root=repo_root,
        envelope_hash=analysis_run_delta_envelope_hash,
        parent_of=lambda node: node.parent,
        load_parent=load_content_pinned_json_base,
        deltas_of=lambda node: node.deltas,
        parse_delta_parent=AnalysisRunDeltaSpec.model_validate,
        terminal_payload=lambda payload: default_spec_registry.migrate(
            "AnalysisRunSpec", payload
        ).payload,
        layer_from_node=lambda digest, node: AnalysisCompositionLayer(
            envelope_sha256=digest,
            parent_ref=node.parent.ref,
            parent_sha256=node.parent.sha256,
            layer_ids=[delta.layer_id for delta in node.deltas],
            parent_payload_path=node.parent.payload_path,
        ),
        delta_schema_id=ANALYSIS_RUN_DELTA_SPEC_SCHEMA_ID,
        terminal_schema_id=ANALYSIS_RUN_SPEC_SCHEMA_ID,
        cycle_error="analysis run delta parent composition cycle detected",
        invalid_parent_error=lambda schema_id: (
            "analysis run delta parent must declare schema_id "
            f"{ANALYSIS_RUN_SPEC_SCHEMA_ID!r} or "
            f"{ANALYSIS_RUN_DELTA_SPEC_SCHEMA_ID!r}, got {schema_id!r}"
        ),
    )
    return FlattenedAnalysisRun(
        authored=spec.model_dump(mode="json", exclude_none=True),
        payload=flattened.payload,
        attribution=flattened.attribution,
        layers=flattened.layers,
    )


def analysis_composition_provenance(flattened: FlattenedAnalysisRun) -> dict[str, Any]:
    """Build the single canonical composition-provenance record for an analysis run."""
    return {
        "schema_id": ANALYSIS_COMPOSITION_PROVENANCE_SCHEMA_ID,
        "schema_version": ANALYSIS_COMPOSITION_PROVENANCE_SCHEMA_VERSION,
        "authored_envelope_sha256": flattened.authored_envelope_sha256,
        "root_spec": {
            "ref": flattened.root_spec.parent_ref,
            "sha256": flattened.root_spec.parent_sha256,
        },
        "layers": [layer.model_dump(mode="json", exclude_none=True) for layer in flattened.layers],
        "attribution": dict(flattened.attribution),
        "flattened_spec_sha256": sha256_bytes(canonical_json_bytes(dict(flattened.payload))),
    }


__all__ = [
    "AnalysisCompositionLayer",
    "AnalysisRunDeltaSpec",
    "FlattenedAnalysisRun",
    "analysis_composition_provenance",
    "analysis_run_delta_envelope",
    "analysis_run_delta_envelope_hash",
    "flatten_analysis_run_delta",
    "is_analysis_run_delta_payload",
]
