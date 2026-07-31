from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from feedbax.analysis import (
    AnalysisBundleDeltaSpec,
    AnalysisBundleSpec,
    dry_run_staged_analysis_bundle,
    expand_analysis_bundle,
    execute_analysis_bundle,
    execute_staged_analysis_bundle,
    resolve_analysis_bundle_authoring,
)
from feedbax.analysis import bundles
from feedbax.analysis.bundles import (
    ANALYSIS_BUNDLE_EXECUTION_SCHEMA_VERSION,
    ANALYSIS_BUNDLE_EXECUTION_SCHEMA_VERSION_V1,
    BundleExpansion,
    StagedAnalysisBundleExecution,
)
from feedbax.analysis.evaluation import (
    EvaluationRecipeResult,
)
from feedbax.contracts import analysis_bundle_composition
from feedbax.contracts.analysis_bundle_composition import (
    analysis_bundle_composition_provenance,
    analysis_bundle_delta_envelope_hash,
    flatten_analysis_bundle_delta,
)
from feedbax.contracts.manifest import (
    ANALYSIS_BUNDLE_COMPOSITION_PROVENANCE_SCHEMA_ID,
    ANALYSIS_BUNDLE_COMPOSITION_PROVENANCE_SCHEMA_VERSION,
    ANALYSIS_BUNDLE_DELTA_SPEC_SCHEMA_ID,
    ANALYSIS_BUNDLE_DELTA_SPEC_SCHEMA_VERSION,
    TrainingRunManifest,
    canonical_json_bytes,
    sha256_bytes,
    write_manifest,
)
from feedbax.contracts.migrations import migrate_structured_spec_payload
from feedbax.integrations.provider import provider_manifest


def _write(tmp_path: Path, name: str, payload: dict[str, Any]) -> str:
    (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    return sha256_bytes(canonical_json_bytes(payload))


def _base_bundle(*, run_ids: list[str] | None = None) -> dict[str, Any]:
    return AnalysisBundleSpec.model_validate(
        {
            "name": "shared-reduction",
            "predicate": {
                "manifest_kind": "TrainingRunManifest",
                "run_ids": run_ids or [],
            },
            "templates": [
                {
                    "name": "grouped",
                    "mode": "grouped",
                    "analysis_type": "example.reduction",
                    "params": {"aggregation": "signed_pair"},
                }
            ],
            "metadata": {"protocol": "shared"},
        }
    ).model_dump(mode="json")


def _delta(
    parent_ref: str,
    parent_sha: str,
    deltas: list[dict[str, Any]],
    *,
    payload_path: list[str] | None = None,
) -> dict[str, Any]:
    parent: dict[str, Any] = {"ref": parent_ref, "sha256": parent_sha}
    if payload_path is not None:
        parent["payload_path"] = payload_path
    return {
        "schema_id": ANALYSIS_BUNDLE_DELTA_SPEC_SCHEMA_ID,
        "schema_version": ANALYSIS_BUNDLE_DELTA_SPEC_SCHEMA_VERSION,
        "parent": parent,
        "deltas": deltas,
    }


def test_shared_base_supports_thin_authority_bindings_and_equivalent_expansion(
    tmp_path: Path,
) -> None:
    base = _base_bundle()
    base_sha = _write(tmp_path, "base.json", base)
    child = _delta(
        "base.json",
        base_sha,
        [
            {
                "layer_id": "model-authority",
                "patches": [
                    {
                        "path": "predicate.run_ids",
                        "value": ["feedbax-training-run:target"],
                    }
                ],
            }
        ],
    )
    direct = _base_bundle(run_ids=["feedbax-training-run:target"])
    resolved, flattening = resolve_analysis_bundle_authoring(child, repo_root=tmp_path)

    assert resolved == AnalysisBundleSpec.model_validate(direct)
    assert flattening is not None
    manifest = TrainingRunManifest(
        id="feedbax-training-run:target",
        run_set_id="target-set",
    )
    composed_expansions = expand_analysis_bundle(child, [manifest], repo_root=tmp_path)
    direct_expansions = expand_analysis_bundle(direct, [manifest])
    assert [item.spec for item in composed_expansions] == [item.spec for item in direct_expansions]
    assert composed_expansions[0].bundle_composition == (
        analysis_bundle_composition_provenance(flattening)
    )
    assert direct_expansions[0].bundle_composition is None
    direct_construction = BundleExpansion(
        bundle_name="shared-reduction",
        template_name="grouped",
        mode="grouped",
        matched_run_ids=(manifest.id,),
        spec=direct_expansions[0].spec,
    )
    assert direct_construction.bundle_composition is None


def test_nested_composition_records_root_to_child_provenance(tmp_path: Path) -> None:
    base_sha = _write(tmp_path, "base.json", _base_bundle())
    middle = _delta(
        "base.json",
        base_sha,
        [
            {
                "layer_id": "middle",
                "patches": [{"path": "metadata.model", "op": "add", "value": "analytical"}],
            }
        ],
    )
    middle_sha = _write(tmp_path, "middle.json", middle)
    leaf = _delta(
        "middle.json",
        middle_sha,
        [
            {
                "layer_id": "leaf",
                "patches": [
                    {"path": "metadata.model", "value": "checkpointed"},
                    {"path": "predicate.run_ids", "value": ["run-leaf"]},
                ],
                "acknowledges_ancestor_paths": ["metadata.model"],
            }
        ],
    )

    flattened = flatten_analysis_bundle_delta(
        AnalysisBundleDeltaSpec.model_validate(leaf), repo_root=tmp_path
    )
    provenance = analysis_bundle_composition_provenance(flattened)

    assert [layer.layer_ids for layer in flattened.layers] == [["middle"], ["leaf"]]
    assert flattened.authored_envelope_sha256 == analysis_bundle_delta_envelope_hash(
        AnalysisBundleDeltaSpec.model_validate(leaf)
    )
    assert provenance["schema_id"] == ANALYSIS_BUNDLE_COMPOSITION_PROVENANCE_SCHEMA_ID
    assert provenance["schema_version"] == (ANALYSIS_BUNDLE_COMPOSITION_PROVENANCE_SCHEMA_VERSION)
    assert provenance["root_bundle"] == {"ref": "base.json", "sha256": base_sha}
    assert provenance["attribution"] == {
        "metadata.model": "leaf",
        "predicate.run_ids": "leaf",
    }
    assert provenance["flattened_bundle_sha256"] == sha256_bytes(
        canonical_json_bytes(flattened.payload)
    )


def test_parent_resolution_fails_closed_on_pin_path_schema_and_cycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _base_bundle()
    base_sha = _write(tmp_path, "base.json", base)
    child = _delta("base.json", base_sha, [{"layer_id": "child", "patches": []}])

    tampered = dict(base)
    tampered["name"] = "tampered"
    (tmp_path / "base.json").write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        resolve_analysis_bundle_authoring(child, repo_root=tmp_path)

    wrapper_sha = _write(tmp_path, "wrapper.json", {"bundle": base, "notes": "text"})
    missing_path = _delta(
        "wrapper.json",
        wrapper_sha,
        [{"layer_id": "child", "patches": []}],
        payload_path=["missing"],
    )
    with pytest.raises(ValueError, match="missing object key"):
        resolve_analysis_bundle_authoring(missing_path, repo_root=tmp_path)

    wrong_sha = _write(tmp_path, "wrong.json", {"schema_id": "example.wrong"})
    with pytest.raises(ValueError, match="must declare schema_id"):
        resolve_analysis_bundle_authoring(
            _delta("wrong.json", wrong_sha, [{"layer_id": "child", "patches": []}]),
            repo_root=tmp_path,
        )

    _write(tmp_path, "base.json", base)
    monkeypatch.setattr(
        analysis_bundle_composition,
        "load_content_pinned_json_base",
        lambda _parent, *, repo_root: dict(child),
    )
    with pytest.raises(ValueError, match="cycle detected"):
        flatten_analysis_bundle_delta(
            AnalysisBundleDeltaSpec.model_validate(child), repo_root=tmp_path
        )


def test_invalid_delta_layers_patches_and_versions_fail_closed(tmp_path: Path) -> None:
    base_sha = _write(tmp_path, "base.json", _base_bundle())

    duplicate = _delta(
        "base.json",
        base_sha,
        [{"layer_id": "same", "patches": []}, {"layer_id": "same", "patches": []}],
    )
    with pytest.raises(ValidationError, match="layer_id values must be unique"):
        AnalysisBundleDeltaSpec.model_validate(duplicate)

    unsupported = _delta("base.json", base_sha, [{"layer_id": "x", "patches": []}])
    unsupported["schema_version"] = "feedbax.spec.analysis_bundle_delta.v0"
    with pytest.raises(ValidationError, match="unsupported AnalysisBundleDeltaSpec"):
        AnalysisBundleDeltaSpec.model_validate(unsupported)

    invalid_op = _delta(
        "base.json",
        base_sha,
        [{"layer_id": "x", "patches": [{"path": "name", "op": "copy", "value": "x"}]}],
    )
    with pytest.raises(ValidationError, match="copy"):
        AnalysisBundleDeltaSpec.model_validate(invalid_op)

    missing_path = _delta(
        "base.json",
        base_sha,
        [{"layer_id": "x", "patches": [{"path": "metadata.absent.value", "value": 1}]}],
    )
    with pytest.raises(ValueError, match="/deltas/x"):
        resolve_analysis_bundle_authoring(missing_path, repo_root=tmp_path)

    unsupported_direct = _base_bundle()
    unsupported_direct["schema_version"] = "feedbax.spec.analysis_bundle.v1"
    with pytest.raises(ValueError, match="migration_intentionally_absent"):
        resolve_analysis_bundle_authoring(unsupported_direct)


def test_unacknowledged_ancestor_override_fails_closed(tmp_path: Path) -> None:
    base_sha = _write(tmp_path, "base.json", _base_bundle())
    middle = _delta(
        "base.json",
        base_sha,
        [{"layer_id": "middle", "patches": [{"path": "name", "value": "middle"}]}],
    )
    middle_sha = _write(tmp_path, "middle.json", middle)
    leaf = _delta(
        "middle.json",
        middle_sha,
        [{"layer_id": "leaf", "patches": [{"path": "name", "value": "leaf"}]}],
    )

    with pytest.raises(ValueError, match="without explicit acknowledgement"):
        resolve_analysis_bundle_authoring(leaf, repo_root=tmp_path)


def test_composed_execution_accepts_authoring_and_records_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    application_registry_bundle,
) -> None:
    run_id = "feedbax-training-run:target"
    base_sha = _write(tmp_path, "base.json", _base_bundle(run_ids=[run_id]))
    child = _delta("base.json", base_sha, [{"layer_id": "authority", "patches": []}])
    write_manifest(TrainingRunManifest(id=run_id, run_set_id="target-set"), root=tmp_path)
    captured: dict[str, Any] = {}

    def fake_execute(_spec, *, root, metadata, **_kwargs):
        captured["metadata"] = metadata
        return object(), Path(root) / "out.json"

    monkeypatch.setattr(bundles, "execute_analysis_run_spec", fake_execute)
    outputs = execute_analysis_bundle(
        child, root=tmp_path, repo_root=tmp_path, registries=application_registry_bundle
    )

    assert len(outputs) == 1
    provenance = captured["metadata"]["bundle"]["composition"]
    assert (
        provenance["authored_envelope_sha256"]
        == outputs[0][0].bundle_composition["authored_envelope_sha256"]
    )


def test_staged_execution_accepts_composed_authoring_and_keeps_direct_metadata_unchanged(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    base = AnalysisBundleSpec.model_validate(
        {
            "name": "staged-shared",
            "predicate": {"manifest_kind": "TrainingRunManifest"},
            "stages": [
                {
                    "name": "optional-reduction",
                    "kind": "analysis",
                    "skip_reason": "no selected inputs",
                }
            ],
            "metadata": {
                "protocol": "shared",
                "bundle_composition": {"authored": "must-remain-user-owned"},
            },
        }
    ).model_dump(mode="json")
    base_sha = _write(tmp_path, "staged.json", base)
    child = _delta("staged.json", base_sha, [{"layer_id": "authority", "patches": []}])

    composed = execute_staged_analysis_bundle(
        child, root=tmp_path, repo_root=tmp_path, registries=application_registry_bundle
    )
    direct = execute_staged_analysis_bundle(
        base, root=tmp_path, registries=application_registry_bundle
    )
    dry_run = dry_run_staged_analysis_bundle(child, root=tmp_path, repo_root=tmp_path)
    direct_dry_run = dry_run_staged_analysis_bundle(base, root=tmp_path)

    assert composed.bundle_composition["root_bundle"] == {
        "ref": "staged.json",
        "sha256": base_sha,
    }
    assert dry_run.bundle_composition == composed.bundle_composition
    assert composed.metadata["bundle_composition"] == {"authored": "must-remain-user-owned"}
    assert dry_run.metadata == composed.metadata
    assert direct.metadata == composed.metadata
    assert direct.bundle_composition is None
    assert "bundle_composition" not in direct.model_dump(mode="json", exclude_none=True)
    assert "bundle_composition" not in direct.model_dump(mode="json")
    assert direct_dry_run.bundle_composition is None
    assert "bundle_composition" not in direct_dry_run.model_dump(mode="json")


def test_staged_execution_v1_migrates_and_composed_v2_serializes_provenance(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    base = AnalysisBundleSpec.model_validate(
        {
            "name": "versioned-staged",
            "predicate": {"manifest_kind": "TrainingRunManifest"},
            "stages": [
                {
                    "name": "optional",
                    "kind": "analysis",
                    "skip_reason": "no selected inputs",
                }
            ],
        }
    ).model_dump(mode="json")
    direct = execute_staged_analysis_bundle(
        base, root=tmp_path, registries=application_registry_bundle
    )
    legacy_payload = direct.model_dump(mode="json")
    legacy_payload["schema_version"] = ANALYSIS_BUNDLE_EXECUTION_SCHEMA_VERSION_V1

    with pytest.raises(ValidationError, match="schema_version"):
        StagedAnalysisBundleExecution.model_validate(legacy_payload)
    migrated = migrate_structured_spec_payload(
        "StagedAnalysisBundleExecution",
        legacy_payload,
    )
    restored = StagedAnalysisBundleExecution.model_validate(migrated.payload)

    assert migrated.target_version == ANALYSIS_BUNDLE_EXECUTION_SCHEMA_VERSION
    assert [record.migration_id for record in migrated.migration_records] == [
        "analysis-bundle-execution-v1-to-v2-composition-provenance"
    ]
    assert restored.bundle_composition is None
    with pytest.raises(ValueError, match="v1 cannot declare bundle_composition"):
        migrate_structured_spec_payload(
            "StagedAnalysisBundleExecution",
            {**legacy_payload, "bundle_composition": {}},
        )

    base_sha = _write(tmp_path, "versioned-staged.json", base)
    child = _delta(
        "versioned-staged.json",
        base_sha,
        [{"layer_id": "authority", "patches": []}],
    )
    composed_payload = execute_staged_analysis_bundle(
        child,
        root=tmp_path,
        repo_root=tmp_path,
        registries=application_registry_bundle,
    ).model_dump(mode="json")

    assert composed_payload["schema_version"] == ANALYSIS_BUNDLE_EXECUTION_SCHEMA_VERSION
    assert composed_payload["bundle_composition"]["root_bundle"] == {
        "ref": "versioned-staged.json",
        "sha256": base_sha,
    }
    provider_version = provider_manifest().schemas["StagedAnalysisBundleExecution"]["properties"][
        "schema_version"
    ]
    assert provider_version["const"] == ANALYSIS_BUNDLE_EXECUTION_SCHEMA_VERSION
    assert provider_version["default"] == ANALYSIS_BUNDLE_EXECUTION_SCHEMA_VERSION


def test_real_composed_staged_child_executes_and_records_provenance(
    tmp_path: Path, application_registry_bundle
) -> None:
    evaluation_type = "feedbax.test.composed_bundle_evaluation"
    run_id = "feedbax-training-run:composed"

    def recipe(spec, _root, _states_path, _execution_context):
        return EvaluationRecipeResult(summary_metrics={"authority": spec.params["authority"]})

    application_registry_bundle.evaluation_recipes.register(evaluation_type, recipe)
    write_manifest(
        TrainingRunManifest(id=run_id, run_set_id="composed", status="completed"),
        root=tmp_path,
    )
    base = AnalysisBundleSpec.model_validate(
        {
            "name": "staged-real",
            "predicate": {
                "manifest_kind": "TrainingRunManifest",
                "run_ids": [run_id],
            },
            "stages": [
                {
                    "name": "evaluate",
                    "kind": "evaluation",
                    "evaluation_type": evaluation_type,
                    "local_params": {"authority": "base"},
                }
            ],
        }
    ).model_dump(mode="json")
    base_sha = _write(tmp_path, "staged-real.json", base)
    child = _delta(
        "staged-real.json",
        base_sha,
        [
            {
                "layer_id": "checkpoint-authority",
                "patches": [
                    {
                        "path": "stages.0.local_params.authority",
                        "value": "child",
                    }
                ],
            }
        ],
    )

    result = execute_staged_analysis_bundle(
        child,
        root=tmp_path,
        repo_root=tmp_path,
        registries=application_registry_bundle,
    )

    assert result.stages[0].status == "materialized"
    assert result.stages[0].manifest_refs[0].kind == "EvaluationRunManifest"
    assert result.bundle_composition["attribution"] == {
        "stages.0.local_params.authority": "checkpoint-authority"
    }


def test_recursive_parent_migrates_supported_older_direct_bundle(tmp_path: Path) -> None:
    older = {
        "schema_id": "feedbax.spec.analysis_bundle",
        "schema_version": "feedbax.spec.analysis_bundle.v2",
        "name": "older",
        "stages": [
            {
                "name": "analysis",
                "kind": "analysis",
                "analysis_type": "feedbax.test.summary",
                "params": {"window": 11},
            }
        ],
    }
    older_sha = _write(tmp_path, "older.json", older)
    child = _delta("older.json", older_sha, [{"layer_id": "child", "patches": []}])

    resolved, flattening = resolve_analysis_bundle_authoring(child, repo_root=tmp_path)

    assert resolved.schema_version == "feedbax.spec.analysis_bundle.v6"
    assert resolved.stages[0].local_params == {"window": 11}
    assert flattening is not None
    assert flattening.payload["schema_version"] == "feedbax.spec.analysis_bundle.v6"


def test_provider_and_migration_registry_publish_separate_authoring_schema() -> None:
    properties = provider_manifest().schemas["AnalysisBundleDeltaSpec"]["properties"]
    assert set(properties) == {"schema_id", "schema_version", "parent", "deltas"}
    assert properties["schema_version"]["default"] == ANALYSIS_BUNDLE_DELTA_SPEC_SCHEMA_VERSION

    from feedbax.contracts.migrations import default_spec_registry

    families = {family.kind: family for family in default_spec_registry.families()}
    assert families["AnalysisBundleDeltaSpec"].current_version == (
        ANALYSIS_BUNDLE_DELTA_SPEC_SCHEMA_VERSION
    )
    with pytest.raises(ValueError, match="migration_intentionally_absent"):
        default_spec_registry.migrate(
            "AnalysisBundleDeltaSpec",
            {
                "schema_id": ANALYSIS_BUNDLE_DELTA_SPEC_SCHEMA_ID,
                "schema_version": "feedbax.spec.analysis_bundle_delta.v0",
            },
        )
