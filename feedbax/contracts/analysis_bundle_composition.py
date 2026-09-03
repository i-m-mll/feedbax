"""Content-pinned whole-document composition for analysis bundles."""

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
    ANALYSIS_BUNDLE_COMPOSITION_PROVENANCE_SCHEMA_ID,
    ANALYSIS_BUNDLE_COMPOSITION_PROVENANCE_SCHEMA_VERSION,
    ANALYSIS_BUNDLE_DELTA_SPEC_SCHEMA_ID,
    ANALYSIS_BUNDLE_DELTA_SPEC_SCHEMA_VERSION,
)
from feedbax.contracts.matrix_core import ContentPinnedJsonBase, load_content_pinned_json_base
from feedbax.contracts.migrations import default_spec_registry
from feedbax.contracts.run_matrix import MatrixCompositionDelta

ANALYSIS_BUNDLE_SPEC_SCHEMA_ID = "feedbax.spec.analysis_bundle"


class AnalysisBundleDeltaSpec(StrictModel):
    """Ordered whole-document deltas over one content-pinned parent bundle."""

    schema_id: str = ANALYSIS_BUNDLE_DELTA_SPEC_SCHEMA_ID
    schema_version: str = ANALYSIS_BUNDLE_DELTA_SPEC_SCHEMA_VERSION
    parent: ContentPinnedJsonBase
    deltas: list[MatrixCompositionDelta] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_delta_spec(self) -> "AnalysisBundleDeltaSpec":
        if self.schema_id != ANALYSIS_BUNDLE_DELTA_SPEC_SCHEMA_ID:
            raise ValueError(f"unsupported AnalysisBundleDeltaSpec schema_id {self.schema_id!r}")
        if self.schema_version != ANALYSIS_BUNDLE_DELTA_SPEC_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported AnalysisBundleDeltaSpec schema_version {self.schema_version!r}"
            )
        layer_ids = [delta.layer_id for delta in self.deltas]
        if len(layer_ids) != len(set(layer_ids)):
            raise ValueError("deltas layer_id values must be unique")
        return self


class AnalysisBundleCompositionLayer(StrictModel):
    """Identity of one authored bundle layer and its pinned parent."""

    envelope_sha256: str
    parent_ref: str
    parent_sha256: str
    layer_ids: list[str]
    parent_payload_path: tuple[str, ...] | None = None


class FlattenedAnalysisBundle(StrictModel):
    """Flattened bundle document with ordered identities and attribution."""

    authored: dict[str, Any]
    payload: dict[str, Any]
    attribution: dict[str, str]
    layers: list[AnalysisBundleCompositionLayer] = Field(min_length=1)

    @property
    def authored_envelope_sha256(self) -> str:
        """Identity of the authored child document that was flattened."""
        return self.layers[-1].envelope_sha256

    @property
    def root_bundle(self) -> AnalysisBundleCompositionLayer:
        """Layer whose pinned parent is the terminal direct bundle."""
        return self.layers[0]


def is_analysis_bundle_delta_payload(payload: Any) -> bool:
    """Return whether a mapping declares the bundle-delta schema identity."""
    return (
        isinstance(payload, Mapping)
        and payload.get("schema_id") == ANALYSIS_BUNDLE_DELTA_SPEC_SCHEMA_ID
    )


def analysis_bundle_delta_envelope(spec: AnalysisBundleDeltaSpec) -> dict[str, Any]:
    """Return the identity-bearing authored composition of a bundle delta."""
    parent = spec.parent.model_dump(mode="json", exclude_none=True)
    parent.pop("ref", None)
    return {
        "schema_id": spec.schema_id,
        "schema_version": spec.schema_version,
        "parent": parent,
        "deltas": [delta.model_dump(mode="json", exclude_none=True) for delta in spec.deltas],
    }


def analysis_bundle_delta_envelope_hash(spec: AnalysisBundleDeltaSpec) -> str:
    """Return the canonical authored envelope identity of a bundle delta."""
    return sha256_bytes(canonical_json_bytes(analysis_bundle_delta_envelope(spec)))


def flatten_analysis_bundle_delta(
    spec: AnalysisBundleDeltaSpec,
    *,
    repo_root: Path | str | None = None,
) -> FlattenedAnalysisBundle:
    """Resolve pinned parents root-to-child into one flat bundle document."""
    flattened = _flatten_content_pinned_parent_deltas(
        spec,
        repo_root=repo_root,
        envelope_hash=analysis_bundle_delta_envelope_hash,
        parent_of=lambda node: node.parent,
        load_parent=load_content_pinned_json_base,
        deltas_of=lambda node: node.deltas,
        parse_delta_parent=AnalysisBundleDeltaSpec.model_validate,
        terminal_payload=lambda payload: default_spec_registry.migrate(
            "AnalysisBundleSpec", payload
        ).payload,
        layer_from_node=lambda digest, node: AnalysisBundleCompositionLayer(
            envelope_sha256=digest,
            parent_ref=node.parent.ref,
            parent_sha256=node.parent.sha256,
            layer_ids=[delta.layer_id for delta in node.deltas],
            parent_payload_path=node.parent.payload_path,
        ),
        delta_schema_id=ANALYSIS_BUNDLE_DELTA_SPEC_SCHEMA_ID,
        terminal_schema_id=ANALYSIS_BUNDLE_SPEC_SCHEMA_ID,
        cycle_error="analysis bundle delta parent composition cycle detected",
        invalid_parent_error=lambda schema_id: (
            "analysis bundle delta parent must declare schema_id "
            f"{ANALYSIS_BUNDLE_SPEC_SCHEMA_ID!r} or "
            f"{ANALYSIS_BUNDLE_DELTA_SPEC_SCHEMA_ID!r}, got {schema_id!r}"
        ),
    )
    return FlattenedAnalysisBundle(
        authored=spec.model_dump(mode="json", exclude_none=True),
        payload=flattened.payload,
        attribution=flattened.attribution,
        layers=flattened.layers,
    )


def analysis_bundle_composition_provenance(
    flattened: FlattenedAnalysisBundle,
) -> dict[str, Any]:
    """Build the canonical composition-provenance record for a bundle."""
    return {
        "schema_id": ANALYSIS_BUNDLE_COMPOSITION_PROVENANCE_SCHEMA_ID,
        "schema_version": ANALYSIS_BUNDLE_COMPOSITION_PROVENANCE_SCHEMA_VERSION,
        "authored_envelope_sha256": flattened.authored_envelope_sha256,
        "root_bundle": {
            "ref": flattened.root_bundle.parent_ref,
            "sha256": flattened.root_bundle.parent_sha256,
        },
        "layers": [layer.model_dump(mode="json", exclude_none=True) for layer in flattened.layers],
        "attribution": dict(flattened.attribution),
        "flattened_bundle_sha256": sha256_bytes(canonical_json_bytes(dict(flattened.payload))),
    }


__all__ = [
    "AnalysisBundleCompositionLayer",
    "AnalysisBundleDeltaSpec",
    "FlattenedAnalysisBundle",
    "analysis_bundle_composition_provenance",
    "analysis_bundle_delta_envelope",
    "analysis_bundle_delta_envelope_hash",
    "flatten_analysis_bundle_delta",
    "is_analysis_bundle_delta_payload",
]
