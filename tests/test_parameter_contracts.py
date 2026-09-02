"""Sparse acceptance for the one durable parameter-contract authority."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from feedbax.contracts.experiment_envelope_dialect import (
    ANALYSIS_BUNDLE_OUTPUT,
    ANALYSIS_RUN_OUTPUT,
    FIGURE_OUTPUT,
    REPORT_OUTPUT,
)
from feedbax.contracts.manifest import ANALYSIS_RUN_SPEC_SCHEMA_VERSION_V1
from feedbax.contracts.manifest import authenticated_manifest_ref_metadata
from feedbax.contracts.migrations import (
    UnsupportedSpecVersion,
    default_spec_registry,
    migrate_structured_spec_payload,
)
from feedbax.contracts.parameter_contracts import (
    ANALYSIS_PARAMS_SCHEMA_VERSION,
    FIGURE_TRACE_PARAMS_SCHEMA_VERSION,
    AnalysisParams,
    FigureTraceParams,
    ParameterContractError,
)
from feedbax.integrations.provider import provider_manifest


@pytest.mark.parametrize(
    "params",
    [
        {
            "aggregation": {"sample_unit": "trial_velocity_profile"},
            "expected_grid": {"target_indices": [0, 1]},
        },
        {
            "fixed_point": {"loss_tol": 1e-6},
            "result_schema": {
                "schema_id": "example.analysis.fixed_points",
                "schema_version": "example.analysis.fixed_points.v1",
            },
        },
        {
            "scalar_projection": {
                "schema_id": "example.spec.scalar_projection",
                "schema_version": "example.spec.scalar_projection.v2",
                "source_schema_id": "example.analysis.source",
                "source_schema_version": "example.analysis.source.v1",
            }
        },
        {
            "payload": {
                "role": "analysis_table",
                "logical_name": "table.json",
                "sha256": "a" * 64,
                "size_bytes": 1024,
                "media_type": "application/json",
            }
        },
    ],
)
def test_analysis_corpus_shapes_have_one_declared_contract(params: dict[str, object]) -> None:
    document = {"analysis_type": "example.analysis", "params": params}

    addressed = ANALYSIS_RUN_OUTPUT.parameter_objects(document)

    assert [(path, schema.schema_version) for path, _, schema in addressed] == [
        ("params", ANALYSIS_PARAMS_SCHEMA_VERSION)
    ]
    ANALYSIS_RUN_OUTPUT.validate_parameter_objects(document)


def test_analysis_rejects_an_undeclared_artifact_reference_shape() -> None:
    document = {
        "analysis_type": "example.analysis",
        "params": {
            "payload": {
                "sha256": "a" * 64,
                "size_bytes": 22_500_000,
                "media_type": "application/octet-stream",
            }
        },
    }

    with pytest.raises(ParameterContractError, match="declared ArtifactRef contract"):
        ANALYSIS_RUN_OUTPUT.validate_parameter_objects(document)


def test_analysis_census_high_risk_shapes_have_explicit_migration_boundary() -> None:
    checkpoint = {
        "checkpoint_custody_binding": "capture-checkpoints",
        "checkpoint_transaction": {
            "kind": "TrainingCheckpointTransactionManifest",
            "id": "tx-1",
            "role": "training_checkpoint_custody",
            "uri": "transactions/tx-1/manifest.json",
            "metadata": {"manifest_sha256": "b" * 64},
        },
    }
    legacy_artifact_authorities = {
        "artifact_provider": "training_diagnostics",
        "manifest_authority": {"id": "run-1", "sha256": "a" * 64, "size_bytes": 100},
        "artifact_authority": {
            "sha256": "b" * 64,
            "size_bytes": 22_500_000,
            "media_type": "application/json",
        },
    }
    legacy_ranged = {
        **legacy_artifact_authorities,
        "first_completed_batch": 1,
        "target_completed_batch": 100,
        "replica_count": 5,
    }

    for params in ({}, checkpoint):
        ANALYSIS_RUN_OUTPUT.validate_parameter_objects(
            {"analysis_type": "example.analysis", "params": params}
        )
    for params in (legacy_artifact_authorities, legacy_ranged):
        with pytest.raises(ParameterContractError, match="declared (Parent|Artifact)Ref contract"):
            ANALYSIS_RUN_OUTPUT.validate_parameter_objects(
                {"analysis_type": "example.analysis", "params": params}
            )

    migrated = {
        **legacy_ranged,
        "manifest_authority": {
            "kind": "TrainingRunManifest",
            "id": "run-1",
            "role": "training_run",
            "metadata": authenticated_manifest_ref_metadata("a" * 64, 100),
        },
        "artifact_authority": {
            "role": "training_diagnostics",
            "logical_name": "training-diagnostics.json",
            "sha256": "b" * 64,
            "size_bytes": 22_500_000,
            "media_type": "application/json",
        },
    }
    ANALYSIS_RUN_OUTPUT.validate_parameter_objects(
        {"analysis_type": "example.analysis", "params": migrated}
    )


def test_analysis_rejects_half_a_nested_schema_identity() -> None:
    document = {
        "analysis_type": "example.analysis",
        "params": {"result_schema": {"schema_id": "example.analysis.result"}},
    }

    with pytest.raises(ParameterContractError, match="without a nonempty 'schema_version'"):
        ANALYSIS_RUN_OUTPUT.validate_parameter_objects(document)


def test_bundle_and_all_figure_parameter_locations_are_addressed() -> None:
    bundle = {"params_base": {"params": {"window": 4}}}
    figure = {
        "assembler_params": {"height": 450},
        "traces": [{"params": {"color": "blue"}}],
        "slot_bindings": {"observed": [{"params": {"showlegend": True}}]},
        "trace_families": [{"params": {"line_width": 2}}],
        "slot_families": [{"params": {"marker_size": 5}}],
    }

    ANALYSIS_BUNDLE_OUTPUT.validate_parameter_objects(bundle)
    FIGURE_OUTPUT.validate_parameter_objects(figure)

    paths = [path for path, _, _ in FIGURE_OUTPUT.parameter_objects(figure)]
    assert paths == [
        "assembler_params",
        "traces.0.params",
        "slot_bindings.observed.0.params",
        "trace_families.0.params",
        "slot_families.0.params",
    ]


def test_figure_style_is_deliberately_open_but_common_fields_are_typed() -> None:
    FigureTraceParams.model_validate({"color": "blue", "project_specific_dash": [1, 3]})

    with pytest.raises(ValidationError, match="showlegend"):
        FigureTraceParams.model_validate({"showlegend": "yes"})


def test_report_unknown_type_keeps_the_existing_external_owner_behavior() -> None:
    assert REPORT_OUTPUT.params_model({"report_type": "example.external_report"}) is None


def test_old_document_migrates_before_current_parameter_acceptance() -> None:
    migrated = migrate_structured_spec_payload(
        "AnalysisRunSpec",
        {"analysis_type": "example.analysis", "inputs": [], "params": {"window": 4}},
        source_version=ANALYSIS_RUN_SPEC_SCHEMA_VERSION_V1,
    )

    assert migrated.target_version == "feedbax.spec.analysis_run.v2"
    ANALYSIS_RUN_OUTPUT.validate_parameter_objects(migrated.payload)


def test_unknown_old_parameter_contract_version_is_explicitly_rejected() -> None:
    with pytest.raises(UnsupportedSpecVersion, match="migration_intentionally_absent"):
        migrate_structured_spec_payload(
            "AnalysisParams",
            {"window": 4},
            source_version="feedbax.spec.params.analysis.v0",
        )


def test_provider_publishes_the_same_parameter_schema_models() -> None:
    schemas = provider_manifest().schemas

    assert schemas["AnalysisParams"] == AnalysisParams.model_json_schema()
    assert schemas["AnalysisParams"]["$id"] == ANALYSIS_PARAMS_SCHEMA_VERSION
    assert schemas["FigureTraceParams"]["$id"] == FIGURE_TRACE_PARAMS_SCHEMA_VERSION
    assert default_spec_registry.current_version("FigureTraceParams") == (
        FIGURE_TRACE_PARAMS_SCHEMA_VERSION
    )
