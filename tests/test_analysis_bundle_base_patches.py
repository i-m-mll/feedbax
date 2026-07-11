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


def test_analysis_bundle_v3_is_accepted_without_migration() -> None:
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
        "analysis-bundle-v2-to-v3-shared-params-base"
    ]
    assert migrated.params_base.params == {}
    assert migrated.stages[0].local_params == {"window": 11}
    assert migrated.stages[0].params_patches == []


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
