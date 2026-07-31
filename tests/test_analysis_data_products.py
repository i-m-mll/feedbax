from __future__ import annotations

import inspect

import pytest

from feedbax.analysis.context import AnalysisRunContext
from feedbax.contracts.graph import AnalysisDataProductRequirement
from feedbax.contracts.manifest import (
    ANALYSIS_DATA_PRODUCT_SCHEMA_ID,
    ANALYSIS_DATA_PRODUCT_SCHEMA_VERSION,
    AnalysisDataProduct,
    AnalysisRunManifest,
    AnalysisRunSpec,
    ArtifactRef,
    ParentRef,
    load_manifest,
    spec_payload,
    write_manifest,
)
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.integrations import provider as provider_module
from feedbax.integrations.provider import provider_manifest, validate_analysis_spec


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


def _analysis_payload(requirement: dict[str, object]) -> dict[str, object]:
    return {
        "analysis_type": "feedbax.analysis.perturbation_response_bank",
        "inputs": [{"kind": "AnalysisRunManifest", "id": "analysis-run:selected-checkpoint"}],
        "input_requirements": [{"data_product": requirement}],
    }


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


def test_analysis_data_product_requirement_validates_resolved_manifest_success(
    application_registry_bundle,
) -> None:
    product = _product()
    requirement = _requirement(product_identity_hash=product.product_identity_hash)

    result = validate_analysis_spec(
        _analysis_payload(requirement),
        resolved_manifests=[
            {
                "kind": "AnalysisRunManifest",
                "produced_data": [product.model_dump(mode="json", exclude_none=True)],
            }
        ],
        component_registry=application_registry_bundle.components,
        analysis_registry=application_registry_bundle.analysis_recipes,
    )

    assert result.valid is True


def test_context_recorded_product_satisfies_public_consumer_requirement(
    tmp_path, application_registry_bundle
) -> None:
    context = AnalysisRunContext(
        spec=AnalysisRunSpec(analysis_type="downstream.scalar_projection"),
        root=tmp_path,
        index_manifest=False,
    )
    product = context.record_data_product(
        {"value": 1.25},
        product_schema_id="downstream.scalar_projection",
        product_schema_version="downstream.scalar_projection.v1",
        role="peak_velocity_scalar",
        logical_name="peak_velocity_scalar",
        materialization={"scalar_path": "value"},
    )
    manifest, _path = context.finalize()

    result = validate_analysis_spec(
        {
            "analysis_type": "downstream.scalar_consumer",
            "inputs": [
                {
                    "kind": "AnalysisRunManifest",
                    "id": manifest.id,
                }
            ],
            "input_requirements": [
                {
                    "data_product": {
                        "role": product.role,
                        "product_schema_id": product.product_schema_id,
                        "exact_product_schema_version": product.product_schema_version,
                        "logical_name": product.logical_name,
                        "producer_manifest_id": manifest.id,
                        "artifact_sha256": product.artifacts[0].sha256,
                    }
                }
            ],
        },
        resolved_manifests=[manifest.model_dump(mode="json", exclude_none=True)],
        component_registry=application_registry_bundle.components,
        analysis_registry=application_registry_bundle.analysis_recipes,
    )

    assert result.valid is True


@pytest.mark.parametrize(
    ("name", "product", "expected_class"),
    [
        (
            "wrong-run",
            _product(producer_manifest_id="analysis-run:other"),
            "wrong-run",
        ),
        (
            "wrong-checkpoint",
            _product(checkpoint_policy={"selection": "validation_selected", "checkpoint_id": "other"}),
            "wrong-checkpoint",
        ),
        (
            "wrong-basis",
            _product(descriptor_basis_hash="sha256:other-basis"),
            "wrong-basis",
        ),
        (
            "wrong-role",
            _product(role="diagnostic_table"),
            "wrong-role",
        ),
    ],
)
def test_named_negative_fixtures_fail_closed(
    name, product, expected_class, application_registry_bundle
) -> None:
    result = validate_analysis_spec(
        _analysis_payload(_requirement()),
        resolved_manifests=[
            {
                "kind": "AnalysisRunManifest",
                "produced_data": [product.model_dump(mode="json", exclude_none=True)],
            }
        ],
        component_registry=application_registry_bundle.components,
        analysis_registry=application_registry_bundle.analysis_recipes,
    )

    assert name
    assert result.valid is False
    assert result.errors[0].type == "analysis_data_product_mismatch"
    assert result.errors[0].details["kind"] == "Mismatch"
    assert result.errors[0].details["mismatch_class"] == expected_class


def test_data_product_requirement_fails_closed_for_absence_and_incompatibility(
    application_registry_bundle,
) -> None:
    missing = validate_analysis_spec(
        _analysis_payload(_requirement()),
        resolved_manifests=[],
        component_registry=application_registry_bundle.components,
        analysis_registry=application_registry_bundle.analysis_recipes,
    )
    assert missing.valid is False
    assert missing.errors[0].type == "analysis_data_product_missing"
    assert missing.errors[0].details["kind"] == "Missing"

    incompatible = validate_analysis_spec(
        _analysis_payload(_requirement(artifact_sha256="def")),
        resolved_manifests=[
            {
                "kind": "AnalysisRunManifest",
                "produced_data": [_product().model_dump(mode="json", exclude_none=True)],
            }
        ],
        component_registry=application_registry_bundle.components,
        analysis_registry=application_registry_bundle.analysis_recipes,
    )
    assert incompatible.valid is False
    assert incompatible.errors[0].details["mismatch_class"] == "artifact-byte-hash"


def test_data_product_schema_version_range_and_old_version_rejection(
    application_registry_bundle,
) -> None:
    requirement = _requirement(
        exact_product_schema_version=None,
        min_product_schema_version="rlrmp.controller_feedback_scales.v1",
        max_product_schema_version="rlrmp.controller_feedback_scales.v1",
    )
    result = validate_analysis_spec(
        _analysis_payload(requirement),
        resolved_manifests=[
            {
                "kind": "AnalysisRunManifest",
                "produced_data": [_product().model_dump(mode="json", exclude_none=True)],
            }
        ],
        component_registry=application_registry_bundle.components,
        analysis_registry=application_registry_bundle.analysis_recipes,
    )
    assert result.valid is True

    incompatible = validate_analysis_spec(
        _analysis_payload(_requirement(exact_product_schema_version="rlrmp.controller_feedback_scales.v2")),
        resolved_manifests=[
            {
                "kind": "AnalysisRunManifest",
                "produced_data": [_product().model_dump(mode="json", exclude_none=True)],
            }
        ],
        component_registry=application_registry_bundle.components,
        analysis_registry=application_registry_bundle.analysis_recipes,
    )
    assert incompatible.valid is False
    assert incompatible.errors[0].details["mismatch_class"] == "schema-version"

    with pytest.raises(UnsupportedSpecVersion):
        default_spec_registry.migrate(
            "AnalysisDataProduct",
            {"schema_version": "feedbax.manifest.analysis_data_product.v0"},
        )
    with pytest.raises(UnsupportedSpecVersion):
        default_spec_registry.migrate(
            "AnalysisDataProductRequirement",
            {"schema_version": "feedbax.spec.analysis_data_product_requirement.v0"},
        )


def test_provider_manifest_and_policy_matrix_export_data_product_families() -> None:
    manifest = provider_manifest()
    policies = default_spec_registry.policy_matrix()

    assert "AnalysisDataProduct" in manifest.schemas
    assert "AnalysisDataProductRequirement" in manifest.schemas
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
    provider_source = inspect.getsource(provider_module)

    assert "select_bundle_manifests" not in manifest_source
    assert "select_bundle_manifests" not in provider_source
