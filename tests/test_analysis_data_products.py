from __future__ import annotations

import inspect

import pytest

from feedbax.contracts.graph import AnalysisDataProductRequirement
from feedbax.contracts.base import (
    ArtifactRef,
    ParentRef,
)
from feedbax.contracts.manifest import (
    ANALYSIS_DATA_PRODUCT_SCHEMA_ID,
    ANALYSIS_DATA_PRODUCT_SCHEMA_VERSION,
    AnalysisDataProduct,
    AnalysisOutputDescriptor,
    AnalysisRunManifest,
    load_manifest,
    analysis_data_product_identity_envelope,
    preserve_analysis_output_descriptor,
    spec_payload,
    write_manifest,
)
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.contracts.value_schema import ValueSchema


def _product(**updates: object) -> AnalysisDataProduct:
    payload = {
        "product_schema_id": "rlrmp.controller_feedback_scales",
        "product_schema_version": "rlrmp.controller_feedback_scales.v1",
        "role": "controller_feedback_scales",
        "logical_name": "controller_feedback_scales",
        "producer_manifest_id": "analysis-run:selected-checkpoint",
        "producer_manifest_hash": "sha256:producer",
        "parent_manifests": [
            {
                "kind": "EvaluationRunManifest",
                "id": "evaluation-run:selected",
                "role": "selected_checkpoint_nominal_rollout",
                "manifest_hash": "sha256:parent",
            }
        ],
        "checkpoint_policy": {
            "selection": "validation_selected",
            "checkpoint_id": "checkpoint:selected",
        },
        "rollout_policy": {"mode": "nominal", "task_split": "validation"},
        "parameters": {
            "component": "force_filter",
            "calibration_level": "moderate",
            "calibration_reach": "canonical_15cm",
        },
        "descriptor_basis_hash": "sha256:descriptor-basis",
        "artifacts": [
            {
                "role": "data_product",
                "logical_name": "controller_feedback_scales.json",
                "artifact_id": "artifact://sha256/abc",
                "sha256": "abc",
                "media_type": "application/json",
                "size_bytes": 123,
                "storage_backend": "feedbax-local",
                "uri": "artifacts/scales-a.json",
                "metadata": {"rows": 1},
            }
        ],
        "materialization": {"materializer": "evaluation_diagnostics", "external_custody": False},
    }
    payload.update(updates)
    return AnalysisDataProduct.model_validate(payload)


def _requirement(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "role": "controller_feedback_scales",
        "product_schema_id": "rlrmp.controller_feedback_scales",
        "exact_product_schema_version": "rlrmp.controller_feedback_scales.v1",
        "logical_name": "controller_feedback_scales",
        "descriptor_basis_hash": "sha256:descriptor-basis",
        "producer_manifest_id": "analysis-run:selected-checkpoint",
        "producer_manifest_hash": "sha256:producer",
        "parent_manifest_ids": ["evaluation-run:selected"],
        "parent_manifest_hashes": ["sha256:parent"],
        "checkpoint_policy": {
            "selection": "validation_selected",
            "checkpoint_id": "checkpoint:selected",
        },
        "rollout_policy": {"mode": "nominal", "task_split": "validation"},
        "parameters": {
            "component": "force_filter",
            "calibration_level": "moderate",
            "calibration_reach": "canonical_15cm",
        },
        "artifact_sha256": "abc",
    }
    payload.update(updates)
    return payload


def test_analysis_data_product_emits_identity_and_round_trips(tmp_path) -> None:
    product = _product()
    assert product.schema_id == ANALYSIS_DATA_PRODUCT_SCHEMA_ID
    assert product.schema_version == ANALYSIS_DATA_PRODUCT_SCHEMA_VERSION
    assert product.product_identity_hash is not None

    manifest = AnalysisRunManifest(
        id="analysis-run:selected-checkpoint",
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec",
            {
                "analysis_type": "feedbax.analysis.evaluation_diagnostics",
                "inputs": [{"kind": "EvaluationRunManifest", "id": "evaluation-run:selected"}],
            },
        ),
        inputs=[
            ParentRef(
                kind="EvaluationRunManifest",
                id="evaluation-run:selected",
                role="selected_checkpoint_nominal_rollout",
            )
        ],
        produced_data=[product],
    )

    path = write_manifest(manifest, root=tmp_path)
    loaded = load_manifest(path)

    assert isinstance(loaded, AnalysisRunManifest)
    assert loaded.produced_data[0].product_identity_hash == product.product_identity_hash
    assert loaded.produced_data[0].artifacts[0].sha256 == "abc"


def test_product_identity_hash_covers_semantic_envelope_not_mutable_uri_or_label() -> None:
    product = _product(label="Readable label")
    same_semantics = _product(
        label="Different label",
        artifacts=[
            ArtifactRef(
                role="data_product",
                logical_name="controller_feedback_scales.json",
                artifact_id="artifact://sha256/abc",
                sha256="abc",
                media_type="application/json",
                size_bytes=123,
                storage_backend="feedbax-local",
                uri="different/local/path.json",
                metadata={"rows": 1},
            )
        ],
    )
    changed_semantics = _product(parameters={**product.parameters, "calibration_level": "stress"})

    assert same_semantics.product_identity_hash == product.product_identity_hash
    assert changed_semantics.product_identity_hash != product.product_identity_hash


def _output_descriptor(
    path: str,
    *,
    quantity: str,
    units: str,
    frame: str | None = None,
) -> AnalysisOutputDescriptor:
    return AnalysisOutputDescriptor(
        path=path,
        value_schema=ValueSchema(
            id=f"science.{quantity}",
            label=quantity.title(),
            kind="array",
            quantity=quantity,
            units=units,
            frame=frame,
            origin="declared",
        ),
    )


def test_output_descriptors_are_identity_bearing_and_absence_preserves_old_envelope() -> None:
    legacy = _product()
    described = _product(
        output_descriptors=[
            _output_descriptor("profiles.force", quantity="force", units="N", frame="body")
        ]
    )

    assert "output_descriptors" not in analysis_data_product_identity_envelope(legacy)
    assert "output_descriptors" not in legacy.model_dump(mode="json", exclude_none=True)
    assert described.product_identity_hash != legacy.product_identity_hash
    assert (
        analysis_data_product_identity_envelope(described)["output_descriptors"][0]["value_schema"][
            "quantity"
        ]
        == "force"
    )


@pytest.mark.parametrize(
    ("quantity", "units"),
    [("force", "N"), ("position", "m")],
)
def test_semantics_preserving_operation_reuses_exact_input_descriptor_generically(
    quantity: str,
    units: str,
) -> None:
    source = _product(
        output_descriptors=[
            _output_descriptor("samples.value", quantity=quantity, units=units, frame="body")
        ]
    )

    preserved = preserve_analysis_output_descriptor(
        source,
        input_path="samples.value",
        output_path="aligned.value",
    )
    changed = _output_descriptor("aligned.value", quantity="time", units="s")

    assert preserved.value_schema.quantity == quantity
    assert preserved.value_schema.units == units
    assert preserved.origin == "preserved"
    assert preserved.preserved_from is not None
    assert preserved.preserved_from.product_identity_hash == source.product_identity_hash
    assert preserved.preserved_from.path == "samples.value"
    assert changed.origin == "declared"
    assert changed.preserved_from is None


def test_output_descriptor_round_trips_with_saved_product(tmp_path) -> None:
    product = _product(
        output_descriptors=[
            _output_descriptor("profiles.force", quantity="force", units="N", frame="body")
        ]
    )
    manifest = AnalysisRunManifest(
        id="analysis-run:described-output",
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec",
            {"analysis_type": "feedbax.test", "inputs": []},
        ),
        produced_data=[product],
    )

    loaded = load_manifest(write_manifest(manifest, root=tmp_path))

    assert isinstance(loaded, AnalysisRunManifest)
    assert loaded.produced_data[0] == product


@pytest.mark.parametrize(
    ("kind", "version"),
    [
        ("AnalysisDataProduct", "feedbax.manifest.analysis_data_product.v0"),
        ("AnalysisDataProductRequirement", "feedbax.spec.analysis_data_product_requirement.v0"),
    ],
)
def test_data_product_schema_rejects_old_versions(kind, version) -> None:
    with pytest.raises(UnsupportedSpecVersion):
        default_spec_registry.migrate(kind, {"schema_version": version})


def test_policy_matrix_exports_data_product_families() -> None:
    policies = default_spec_registry.policy_matrix()

    assert policies["AnalysisDataProduct"].stance == "reject"
    assert policies["AnalysisDataProductRequirement"].stance == "reject"
    assert "tests/test_analysis_data_products.py" in policies["AnalysisDataProduct"].required_tests


def test_analysis_data_product_fixture_is_rlrmp_shaped_but_feedbax_generic() -> None:
    product = _product()
    requirement = AnalysisDataProductRequirement.model_validate(_requirement())

    assert product.logical_name == "controller_feedback_scales"
    assert product.parameters["component"] == "force_filter"
    assert requirement.parameters["component"] == "force_filter"
    assert product.product_schema_id.startswith("rlrmp.")


def test_data_product_contract_does_not_redefine_graph_build_component_identity() -> None:
    forbidden_fields = {"component_id", "component_type_id", "node_id", "port_id", "graph_node_id"}
    product_fields = set(AnalysisDataProduct.model_fields)
    requirement_fields = set(AnalysisDataProductRequirement.model_fields)

    assert forbidden_fields.isdisjoint(product_fields)
    assert forbidden_fields.isdisjoint(requirement_fields)
    assert "descriptor_basis_hash" in product_fields
    assert "descriptor_selector_requirements" in requirement_fields


def test_data_product_contract_does_not_import_bundle_selection() -> None:
    import feedbax.contracts.manifest as manifest_module

    manifest_source = inspect.getsource(manifest_module)

    assert "select_bundle_manifests" not in manifest_source
