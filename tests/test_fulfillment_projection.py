"""Rebuild verification includes scientific interpretation and exact bytes."""

from copy import deepcopy

import pytest

from feedbax.analysis.fulfillment_custody import (
    NodeRebuildOutcome,
    OutputProjection,
    ProjectionDifference,
    compare_output_projections,
    migrate_output_projection,
    output_projection,
)
from feedbax.contracts.base import ArtifactRef, Provenance, sha256_bytes
from feedbax.contracts.manifest import (
    AnalysisDataProduct,
    AnalysisRunManifest,
    AnalysisRunSpec,
    analysis_run_manifest_id,
    spec_payload,
)


def _manifest() -> AnalysisRunManifest:
    spec = AnalysisRunSpec(analysis_type="testpkg.scientific_output")
    manifest_id = analysis_run_manifest_id(spec)
    artifact = ArtifactRef(
        role="analysis_data",
        logical_name="position.json",
        media_type="application/json",
        sha256=sha256_bytes(b'{"position": 3}'),
        size_bytes=15,
    )
    product = AnalysisDataProduct(
        role="position",
        logical_name="position",
        producer_manifest_id=manifest_id,
        product_schema_id="testpkg.position",
        product_schema_version="testpkg.position.v1",
        artifacts=[artifact],
        output_descriptors=[
            {
                "path": "position",
                "value_schema": {
                    "id": "position",
                    "label": "Position",
                    "kind": "scalar",
                    "dtype": "float64",
                    "quantity": "length",
                    "units": "m",
                    "frame": "world",
                },
            }
        ],
    )
    return AnalysisRunManifest(
        id=manifest_id,
        status="completed",
        provenance=Provenance(parents=[]),
        analysis_spec=spec_payload("AnalysisRunSpec", spec.model_dump(mode="json")),
        produced_data=[product],
        artifacts=[artifact],
    )


def test_scientific_interpretation_and_payload_changes_are_drift() -> None:
    manifest = _manifest()
    expected = output_projection(manifest, node_kind="analysis")
    assert migrate_output_projection(expected.model_dump(mode="json")) == expected
    assert (
        compare_output_projections(expected, output_projection(manifest, node_kind="analysis"))
        == ()
    )

    changes = {
        "path": "distance",
        "value_schema.units": "cm",
        "value_schema.frame": "body",
        "value_schema.quantity": "displacement",
        "value_schema.dtype": "float32",
        "value_schema.metadata.calibration": "revised",
    }
    for field, value in changes.items():
        raw = manifest.produced_data[0].model_dump(mode="json")
        raw.pop("product_identity_hash")
        descriptor = raw["output_descriptors"][0]
        target = descriptor
        parts = field.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
        product = AnalysisDataProduct.model_validate(raw)
        observed = output_projection(
            manifest.model_copy(update={"produced_data": [product]}), node_kind="analysis"
        )
        assert product.product_identity_hash != manifest.produced_data[0].product_identity_hash
        assert [d.field_path for d in compare_output_projections(expected, observed)] == [
            f"products[0].output_descriptors[0].{field}"
        ]

    raw = manifest.produced_data[0].model_dump(mode="json")
    raw.pop("product_identity_hash")
    raw["output_descriptors"][0].update(
        origin="preserved",
        preserved_from={"product_identity_hash": "a" * 64, "path": "source"},
    )
    observed = output_projection(
        manifest.model_copy(
            update={
                "produced_data": [AnalysisDataProduct.model_validate(raw)],
            }
        ),
        node_kind="analysis",
    )
    assert {d.field_path for d in compare_output_projections(expected, observed)} == {
        "products[0].output_descriptors[0].origin",
        "products[0].output_descriptors[0].preserved_from",
    }

    changed_artifact = manifest.artifacts[0].model_copy(
        update={
            "sha256": sha256_bytes(b'{"position": 9}'),
        }
    )
    observed = output_projection(
        manifest.model_copy(update={"artifacts": [changed_artifact]}), node_kind="analysis"
    )
    assert [d.field_path for d in compare_output_projections(expected, observed)] == [
        "artifacts[0].sha256"
    ], "equal-size numerical changes still require exact byte identity"


def test_v1_verification_envelopes_require_reprojection_from_retained_manifests() -> None:
    projection = output_projection(_manifest(), node_kind="analysis")
    difference = ProjectionDifference(field_path="status", expected="completed", observed="failed")
    outcome = NodeRebuildOutcome(
        node_key="analysis/position",
        node_kind="analysis",
        manifest_id=projection.manifest_id,
        matched=False,
        expected=projection,
        observed=projection,
        differences=[difference],
    )
    guidance = "regenerate v2 projections from retained manifests and rerun rebuild verification"
    for envelope in (projection, difference, outcome):
        payload = envelope.model_dump(mode="json")
        assert payload["schema_version"] == "feedbax.fulfillment.output_projection.v2"
        assert type(envelope).model_validate_json(envelope.model_dump_json()) == envelope
        payload["schema_version"] = "feedbax.fulfillment.output_projection.v1"
        with pytest.raises(ValueError, match=guidance):
            type(envelope).model_validate(payload)
        if isinstance(envelope, OutputProjection):
            with pytest.raises(ValueError, match=guidance):
                migrate_output_projection(payload)

    for field in ("expected", "observed", "differences"):
        payload = deepcopy(outcome.model_dump(mode="json"))
        nested = payload[field][0] if field == "differences" else payload[field]
        nested["schema_version"] = "feedbax.fulfillment.output_projection.v1"
        with pytest.raises(ValueError, match=guidance):
            NodeRebuildOutcome.model_validate(payload)
