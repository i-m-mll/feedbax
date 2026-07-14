from __future__ import annotations

import pytest

from feedbax.analysis.bundles import (
    ANALYSIS_BUNDLE_SCHEMA_ID,
    ANALYSIS_BUNDLE_SCHEMA_VERSION,
    AnalysisBundleSpec,
    BundleParamsBase,
    BundleStageSpec,
    _params_for_stage,
)
from feedbax.contracts.manifest import OverridePatch
from feedbax.config.yaml import get_yaml_loader
from feedbax.contracts.migrations import (
    UnsupportedSpecVersion,
    migrate_structured_spec_payload,
)


def test_bundle_stage_resolves_shared_base_with_ordered_patches() -> None:
    params_base = BundleParamsBase(
        params={"metric": {"window": 10, "mode": "mean"}, "seed": 3}
    )
    stage = BundleStageSpec(
        name="summary",
        kind="analysis",
        analysis_type="feedbax.test.summary",
        params_patches=[
            OverridePatch(path="metric.window", value=20),
            OverridePatch(path="metric.mode", value="median"),
            OverridePatch(path="label", op="add", value="robust"),
        ],
    )

    assert _params_for_stage(stage, params_base) == {
        "metric": {"window": 20, "mode": "median"},
        "seed": 3,
        "label": "robust",
    }
    assert params_base.params["metric"]["window"] == 10


def test_bundle_stage_local_params_is_explicit_and_exclusive() -> None:
    stage = BundleStageSpec(
        name="legacy-science",
        kind="analysis",
        analysis_type="feedbax.test.summary",
        local_params={"window": 5},
    )
    assert _params_for_stage(stage, BundleParamsBase(params={"window": 10})) == {
        "window": 5
    }

    with pytest.raises(ValueError, match="cannot combine local_params with params_patches"):
        BundleStageSpec(
            name="ambiguous",
            kind="analysis",
            analysis_type="feedbax.test.summary",
            local_params={},
            params_patches=[OverridePatch(path="window", value=5)],
        )


def test_analysis_bundle_v4_is_accepted_without_migration() -> None:
    payload = AnalysisBundleSpec(name="current").model_dump(mode="json")
    result = migrate_structured_spec_payload("AnalysisBundleSpec", payload)

    assert result.target_version == ANALYSIS_BUNDLE_SCHEMA_VERSION
    assert result.migration_records == []
    assert AnalysisBundleSpec.model_validate(result.payload).schema_id == ANALYSIS_BUNDLE_SCHEMA_ID


def test_analysis_bundle_v2_migrates_stage_params_to_explicit_local_params() -> None:
    payload = {
        "schema_id": ANALYSIS_BUNDLE_SCHEMA_ID,
        "schema_version": "feedbax.spec.analysis_bundle.v2",
        "name": "archived",
        "stages": [
            {
                "name": "analysis",
                "kind": "analysis",
                "analysis_type": "feedbax.test.summary",
                "params": {"window": 11},
            }
        ],
    }

    result = migrate_structured_spec_payload("AnalysisBundleSpec", payload)
    migrated = AnalysisBundleSpec.model_validate(result.payload)

    assert result.target_version == ANALYSIS_BUNDLE_SCHEMA_VERSION
    assert [record.migration_id for record in result.migration_records] == [
        "analysis-bundle-v2-to-v3-shared-params-base",
        "analysis-bundle-v3-to-v4-per-input-prerequisites",
    ]
    assert migrated.params_base.params == {}
    assert migrated.stages[0].local_params == {"window": 11}
    assert migrated.stages[0].params_patches == []
    assert migrated.stages[0].prerequisite_bindings == []


def test_analysis_bundle_v3_migrates_with_empty_prerequisite_bindings() -> None:
    result = migrate_structured_spec_payload(
        "AnalysisBundleSpec",
        {
            "schema_id": ANALYSIS_BUNDLE_SCHEMA_ID,
            "schema_version": "feedbax.spec.analysis_bundle.v3",
            "name": "archived-v3",
            "stages": [],
        },
    )

    migrated = AnalysisBundleSpec.model_validate(result.payload)
    assert migrated.schema_version == ANALYSIS_BUNDLE_SCHEMA_VERSION
    assert [record.migration_id for record in result.migration_records] == [
        "analysis-bundle-v3-to-v4-per-input-prerequisites"
    ]


def test_analysis_bundle_v4_yaml_round_trip_preserves_prerequisite_binding() -> None:
    payload = """
schema_id: feedbax.spec.analysis_bundle
schema_version: feedbax.spec.analysis_bundle.v4
name: portable
stages:
  - name: evaluate
    kind: evaluation
    mode: per-run
    evaluation_type: feedbax.test.portable
    prerequisite_bindings:
      - input_id: training-1
        bind_as: baseline
        artifact_provider: external
        parent:
          kind: EvaluationRunManifest
          id: prerequisite-1
          role: evaluation_run
          metadata:
            ref_schema_id: feedbax.ref.authenticated_manifest
            ref_schema_version: feedbax.ref.authenticated_manifest.v1
            manifest_sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
            size_bytes: 10
"""
    loaded = AnalysisBundleSpec.model_validate(get_yaml_loader(typ="safe").load(payload))
    round_tripped = AnalysisBundleSpec.model_validate_json(loaded.model_dump_json())
    binding = round_tripped.stages[0].prerequisite_bindings[0]
    assert binding.input_id == "training-1"
    assert binding.artifact_provider == "external"


def test_analysis_bundle_v1_is_explicitly_rejected() -> None:
    with pytest.raises(UnsupportedSpecVersion, match="migration_intentionally_absent=yes"):
        migrate_structured_spec_payload(
            "AnalysisBundleSpec",
            {
                "schema_id": ANALYSIS_BUNDLE_SCHEMA_ID,
                "schema_version": "feedbax.spec.analysis_bundle.v1",
                "name": "unsupported",
            },
        )
