from __future__ import annotations

import pytest

from feedbax.contracts.graph import GraphSpec, StudioArtifactRef, StudioSelectorRef
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.contracts.workspace_replay import (
    WORKSPACE_REPLAY_SCHEMA_ID,
    WORKSPACE_REPLAY_SCHEMA_VERSION,
    WORKSPACE_REPLAY_SCHEMA_VERSION_V0,
    WorkspaceReplayManifestRefs,
    WorkspaceReplayProduct,
    WorkspaceReplaySampleAxis,
    WorkspaceReplayTrack,
    WorkspaceReplayTrial,
    WorkspaceReplayTrialIdentity,
    WorkspaceReplayTrialSpecSnapshot,
    compile_workspace_replay_retention,
    imported_npz_workspace_replay_product,
    workspace_replay_metadata,
)
from feedbax.integrations.provider import provider_manifest
from feedbax.runtime.retained_observables import lower_retention_plan


pytestmark = [pytest.mark.feedbax_contract, pytest.mark.provider_contract]


def _selector(namespace: str, compact: str, **kwargs: object) -> StudioSelectorRef:
    return StudioSelectorRef(namespace=namespace, compact=compact, **kwargs)


def _graph() -> GraphSpec:
    return GraphSpec(
        nodes={
            "plant": {
                "type": "PointMass",
                "params": {},
                "input_ports": ["force"],
                "output_ports": ["effector"],
            }
        },
        output_ports=["effector"],
        output_bindings={"effector": ("plant", "effector")},
    )


def test_workspace_replay_required_selectors_compile_to_trajectory_retention() -> None:
    observables = compile_workspace_replay_retention(
        {
            "required_selectors": [
                {
                    "anchor_id": "effector",
                    "selector": _selector("graph_output", "graph_output:effector"),
                    "value_schema": {
                        "id": "effector",
                        "label": "Effector",
                        "kind": "array",
                        "shape": ["time", 2],
                    },
                },
                {
                    "anchor_id": "target",
                    "selector": _selector("task_data", "task_data:targets.effector"),
                },
            ]
        }
    )

    assert [item.id for item in observables] == [
        "workspace_replay:effector",
        "workspace_replay:target",
    ]
    assert {item.retention.mode for item in observables} == {"trajectory"}
    assert observables[0].target is not None
    assert observables[0].target.kind == "graph_output"
    assert observables[0].target.selector == "graph_output:effector"
    assert observables[0].metadata["workspace_replay"] is True

    graph = _graph()
    graph.retained_observables = observables[:1]
    plan = lower_retention_plan(graph)

    assert plan.by_selector["graph_output:effector"].retention.mode == "trajectory"
    assert plan.by_selector["graph_output:effector"].retention.reasons == (
        "workspace_replay",
    )


def test_workspace_replay_compile_rejects_unresolved_representation_anchor() -> None:
    with pytest.raises(ValueError, match="server-resolved"):
        compile_workspace_replay_retention(
            [
                {
                    "anchor_id": "muscle-origin",
                    "selector": _selector(
                        "mechanics_object",
                        "mechanics_object:arm.origin",
                    ),
                }
            ]
        )


def test_workspace_replay_product_records_trials_tracks_and_manifest_refs() -> None:
    product = WorkspaceReplayProduct(
        trials=[
            WorkspaceReplayTrial(
                identity=WorkspaceReplayTrialIdentity(index=0, stable_id="trial-a"),
                time=WorkspaceReplaySampleAxis(length=2, units="s", values=[0.0, 0.02]),
                tracks=[
                    WorkspaceReplayTrack(
                        anchor_id="effector",
                        selector=_selector("graph_output", "graph_output:effector"),
                        samples=[[0.0, 0.1], [0.2, 0.3]],
                        dim=2,
                        units="m",
                        frame="world",
                    )
                ],
                trial_spec=WorkspaceReplayTrialSpecSnapshot(
                    summary={"target_count": 1},
                    targets={"effector": [0.2, 0.3]},
                    timeline={"epochs": [{"id": "reach", "length": 2}]},
                ),
                manifest_refs=WorkspaceReplayManifestRefs(
                    spec_snapshot=StudioArtifactRef(
                        kind="trial_spec_snapshot",
                        id="artifact:trial-spec",
                        role="spec_snapshot",
                    ),
                    checkpoint=StudioArtifactRef(
                        kind="checkpoint",
                        id="artifact:checkpoint",
                        role="training_checkpoint",
                    ),
                    seed=7,
                    environment={"jax": "test"},
                ),
            )
        ]
    )

    dumped = product.model_dump(mode="json", exclude_none=True)

    assert dumped["schema_id"] == WORKSPACE_REPLAY_SCHEMA_ID
    assert dumped["schema_version"] == WORKSPACE_REPLAY_SCHEMA_VERSION
    assert dumped["product_kind"] == "workspace_replay"
    assert dumped["trials"][0]["identity"]["source"] == "stable_id"
    assert dumped["trials"][0]["tracks"][0]["samples"] == [[0.0, 0.1], [0.2, 0.3]]
    assert dumped["trials"][0]["manifest_refs"]["checkpoint"]["id"] == "artifact:checkpoint"


def test_workspace_replay_product_represents_index_fallback_identity() -> None:
    identity = WorkspaceReplayTrialIdentity(index=3)

    assert identity.stable_id is None
    assert identity.source == "index_fallback"


def test_imported_npz_workspace_replay_is_explicit_downgrade_with_warnings() -> None:
    product = imported_npz_workspace_replay_product(
        StudioArtifactRef(
            kind="trajectory_dataset",
            id="artifact:legacy-npz",
            role="trajectory_dataset",
            media_type="application/x-npz",
        ),
        missing_metadata=("trial ids", "manifest refs", "anchor selectors"),
    )

    assert product.source_mode == "imported_artifact"
    assert product.imported_artifact is not None
    assert product.imported_artifact.missing_metadata == [
        "trial ids",
        "manifest refs",
        "anchor selectors",
    ]
    assert [warning.code for warning in product.warnings] == [
        "npz_browser_downgrade",
        "missing_trial_metadata",
        "missing_manifest_refs",
        "missing_anchor_resolution",
    ]


def test_workspace_replay_schema_policy_accepts_current_and_rejects_v0() -> None:
    assert workspace_replay_metadata() == {
        "schema_id": WORKSPACE_REPLAY_SCHEMA_ID,
        "schema_version": WORKSPACE_REPLAY_SCHEMA_VERSION,
        "product_kind": "workspace_replay",
    }

    current = default_spec_registry.migrate(
        "WorkspaceReplayProduct",
        {"schema_version": WORKSPACE_REPLAY_SCHEMA_VERSION},
    )
    assert current.source_version == WORKSPACE_REPLAY_SCHEMA_VERSION
    assert not current.migrated

    with pytest.raises(UnsupportedSpecVersion) as exc_info:
        default_spec_registry.migrate(
            "WorkspaceReplayProduct",
            {"schema_version": WORKSPACE_REPLAY_SCHEMA_VERSION_V0},
        )

    message = str(exc_info.value)
    assert "WorkspaceReplayProduct" in message
    assert "migration_intentionally_absent=yes" in message


def test_provider_manifest_exposes_workspace_replay_product_schema() -> None:
    manifest = provider_manifest()

    assert "workspace_replay" in manifest.artifact_roles
    assert manifest.schemas["WorkspaceReplayProduct"] == WorkspaceReplayProduct.model_json_schema()
