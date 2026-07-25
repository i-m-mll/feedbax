"""Content-pinned composition/delta facility for ``AnalysisRunSpec``.

These tests mirror ``tests/test_evaluation_matrix_composition.py`` for the
analysis authoring surface: composed-vs-standalone equivalence, fail-closed
error cases, recorded provenance, and byte-identical handling of existing direct
``AnalysisRunSpec`` documents.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from feedbax.analysis import (
    AnalysisRecipeResult,
    AnalysisRunDeltaSpec,
    coerce_analysis_run_spec,
    execute_analysis_run_spec,
    register_analysis_recipe,
    resolve_analysis_run_authoring,
    unregister_analysis_recipe,
)
from feedbax.contracts import analysis_composition
from feedbax.contracts.analysis_composition import (
    analysis_composition_provenance,
    analysis_run_delta_envelope_hash,
    flatten_analysis_run_delta,
)
from feedbax.contracts.manifest import (
    ANALYSIS_COMPOSITION_PROVENANCE_SCHEMA_ID,
    ANALYSIS_COMPOSITION_PROVENANCE_SCHEMA_VERSION,
    ANALYSIS_RUN_DELTA_SPEC_SCHEMA_ID,
    ANALYSIS_RUN_DELTA_SPEC_SCHEMA_VERSION,
    ANALYSIS_RUN_SPEC_SCHEMA_ID,
    ANALYSIS_RUN_SPEC_SCHEMA_VERSION,
    AnalysisRunSpec,
    canonical_json_bytes,
    sha256_bytes,
)
from tests.analysis_fixtures import ToyAnalysis, build_toy_analysis_data


_EXECUTION_ANALYSIS_TYPE = "example.velocity_profiles"


def _register_toy_velocity_recipe() -> None:
    """Register a minimal recipe for the shared velocity-profile analysis type."""

    def recipe(_spec, _root, _inputs, _execution_context) -> AnalysisRecipeResult:
        return AnalysisRecipeResult(
            analyses={"velocity": ToyAnalysis(variant="velocity", cache_result=True)},
            data=build_toy_analysis_data(value=0),
        )

    register_analysis_recipe(_EXECUTION_ANALYSIS_TYPE, recipe, replace=True)


def _write(tmp_path: Path, name: str, payload: dict[str, Any]) -> str:
    (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    return sha256_bytes(canonical_json_bytes(payload))


def _analysis_base(analysis_type: str, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_id": ANALYSIS_RUN_SPEC_SCHEMA_ID,
        "schema_version": ANALYSIS_RUN_SPEC_SCHEMA_VERSION,
        "analysis_type": analysis_type,
        "inputs": [],
        "params": params,
    }


def _shared_velocity_base(tmp_path: Path) -> str:
    """Write the shared velocity-profile base carrying the duplicated params block."""
    return _write(
        tmp_path,
        "base.json",
        _analysis_base(
            "example.velocity_profiles",
            {
                "aggregation": "mean",
                "expected_grid": {
                    "policy_arms": ["control", "perturbed"],
                    "target_indices": [0, 1, 2],
                    "replicate_slots": 4,
                },
            },
        ),
    )


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
        "schema_id": ANALYSIS_RUN_DELTA_SPEC_SCHEMA_ID,
        "schema_version": ANALYSIS_RUN_DELTA_SPEC_SCHEMA_VERSION,
        "parent": parent,
        "deltas": deltas,
    }


def test_two_children_share_one_pinned_params_block(tmp_path: Path) -> None:
    """The first-consumer scenario: two wrappers inherit one shared params block."""
    base_sha = _shared_velocity_base(tmp_path)
    flat3e5 = _delta(
        "base.json",
        base_sha,
        [
            {
                "layer_id": "flat3e5",
                "patches": [
                    {
                        "path": "inputs",
                        "value": [{"kind": "evaluation_run", "id": "flat3e5-eval"}],
                    }
                ],
            }
        ],
    )
    sisu = _delta(
        "base.json",
        base_sha,
        [
            {
                "layer_id": "sisu_wave1",
                "patches": [
                    {
                        "path": "inputs",
                        "value": [{"kind": "evaluation_run", "id": "sisu-wave1-eval"}],
                    },
                    {
                        "path": "params.expected_grid.conditioning",
                        "op": "add",
                        "value": "wave1",
                    },
                ],
            }
        ],
    )

    flat_spec, flat_flat = resolve_analysis_run_authoring(flat3e5, repo_root=tmp_path)
    sisu_spec, sisu_flat = resolve_analysis_run_authoring(sisu, repo_root=tmp_path)

    # Neither child restates the shared aggregation/grid; both inherit it.
    assert flat_spec.params["aggregation"] == "mean"
    assert sisu_spec.params["aggregation"] == "mean"
    assert flat_spec.params["expected_grid"] == {
        "policy_arms": ["control", "perturbed"],
        "target_indices": [0, 1, 2],
        "replicate_slots": 4,
    }
    # SISU adds only its conditioning delta on top of the shared grid.
    assert sisu_spec.params["expected_grid"]["conditioning"] == "wave1"
    assert "conditioning" not in flat_spec.params["expected_grid"]
    assert [ref.id for ref in flat_spec.inputs] == ["flat3e5-eval"]
    assert [ref.id for ref in sisu_spec.inputs] == ["sisu-wave1-eval"]

    # Composed-vs-standalone equivalence: the flattened doc equals a hand-authored spec.
    standalone_sisu = _analysis_base(
        "example.velocity_profiles",
        {
            "aggregation": "mean",
            "expected_grid": {
                "policy_arms": ["control", "perturbed"],
                "target_indices": [0, 1, 2],
                "replicate_slots": 4,
                "conditioning": "wave1",
            },
        },
    )
    standalone_sisu["inputs"] = [{"kind": "evaluation_run", "id": "sisu-wave1-eval"}]
    assert sisu_spec == AnalysisRunSpec.model_validate(standalone_sisu)
    assert flat_flat is not None and sisu_flat is not None
    assert sisu_flat.attribution == {
        "inputs": "sisu_wave1",
        "params.expected_grid.conditioning": "sisu_wave1",
    }


def test_nested_chain_applies_layers_root_to_child_with_attribution(tmp_path: Path) -> None:
    base_sha = _shared_velocity_base(tmp_path)
    mid = _delta(
        "base.json",
        base_sha,
        [{"layer_id": "mid", "patches": [{"path": "params.aggregation", "value": "median"}]}],
    )
    mid_sha = _write(tmp_path, "mid.json", mid)
    leaf = _delta(
        "mid.json",
        mid_sha,
        [
            {
                "layer_id": "leaf",
                "patches": [
                    {
                        "path": "params.expected_grid.conditioning",
                        "op": "add",
                        "value": "wave1",
                    },
                    {"path": "analysis_type", "value": "example.velocity_profiles_leaf"},
                ],
            }
        ],
    )

    flattened = flatten_analysis_run_delta(
        AnalysisRunDeltaSpec.model_validate(leaf), repo_root=tmp_path
    )

    assert [layer.layer_ids for layer in flattened.layers] == [["mid"], ["leaf"]]
    assert flattened.layers[0].parent_ref == "base.json"
    assert flattened.layers[0].parent_sha256 == base_sha
    assert flattened.root_spec is flattened.layers[0]
    assert flattened.authored_envelope_sha256 == analysis_run_delta_envelope_hash(
        AnalysisRunDeltaSpec.model_validate(leaf)
    )
    assert flattened.attribution == {
        "params.aggregation": "mid",
        "params.expected_grid.conditioning": "leaf",
        "analysis_type": "leaf",
    }
    resolved = AnalysisRunSpec.model_validate(flattened.payload)
    assert resolved.params["aggregation"] == "median"
    assert resolved.params["expected_grid"]["conditioning"] == "wave1"
    assert resolved.analysis_type == "example.velocity_profiles_leaf"


def test_unacknowledged_ancestor_override_fails_closed(tmp_path: Path) -> None:
    base_sha = _shared_velocity_base(tmp_path)
    mid = _delta(
        "base.json",
        base_sha,
        [{"layer_id": "mid", "patches": [{"path": "params.aggregation", "value": "median"}]}],
    )
    mid_sha = _write(tmp_path, "mid.json", mid)
    leaf = _delta(
        "mid.json",
        mid_sha,
        [{"layer_id": "leaf", "patches": [{"path": "params.aggregation", "value": "max"}]}],
    )

    with pytest.raises(ValueError, match="without explicit acknowledgement"):
        flatten_analysis_run_delta(AnalysisRunDeltaSpec.model_validate(leaf), repo_root=tmp_path)

    # The same override succeeds when the layer explicitly acknowledges the path.
    leaf["deltas"][0]["acknowledges_ancestor_paths"] = ["params.aggregation"]
    flattened = flatten_analysis_run_delta(
        AnalysisRunDeltaSpec.model_validate(leaf), repo_root=tmp_path
    )
    assert flattened.payload["params"]["aggregation"] == "max"


def test_parent_resolution_is_pinned_confined_and_schema_checked(tmp_path: Path) -> None:
    base_sha = _shared_velocity_base(tmp_path)
    child = _delta(
        "base.json",
        base_sha,
        [{"layer_id": "x", "patches": [{"path": "params.aggregation", "value": "sum"}]}],
    )

    tampered = json.loads((tmp_path / "base.json").read_text(encoding="utf-8"))
    tampered["params"]["aggregation"] = "tampered"
    (tmp_path / "base.json").write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        flatten_analysis_run_delta(AnalysisRunDeltaSpec.model_validate(child), repo_root=tmp_path)
    base_sha = _shared_velocity_base(tmp_path)

    escaping = _delta("../base.json", base_sha, child["deltas"])
    with pytest.raises(ValueError, match="escapes repo_root"):
        flatten_analysis_run_delta(
            AnalysisRunDeltaSpec.model_validate(escaping), repo_root=tmp_path / "nested"
        )

    wrong_schema_sha = _write(tmp_path, "wrong.json", {"schema_id": "feedbax.spec.something_else"})
    with pytest.raises(ValueError, match="must declare schema_id"):
        flatten_analysis_run_delta(
            AnalysisRunDeltaSpec.model_validate(_delta("wrong.json", wrong_schema_sha, child["deltas"])),
            repo_root=tmp_path,
        )

    with pytest.raises(ValueError, match="requires repo_root"):
        flatten_analysis_run_delta(AnalysisRunDeltaSpec.model_validate(child))


def test_repeated_parent_document_is_rejected_as_a_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_sha = _shared_velocity_base(tmp_path)
    child = _delta(
        "base.json",
        base_sha,
        [{"layer_id": "x", "patches": [{"path": "params.aggregation", "value": "sum"}]}],
    )
    monkeypatch.setattr(
        analysis_composition,
        "load_content_pinned_json_base",
        lambda base, *, repo_root: dict(child),
    )

    with pytest.raises(ValueError, match="cycle detected"):
        flatten_analysis_run_delta(AnalysisRunDeltaSpec.model_validate(child), repo_root=tmp_path)


def test_invalid_delta_documents_fail_closed(tmp_path: Path) -> None:
    base_sha = _shared_velocity_base(tmp_path)

    missing_path = _delta(
        "base.json",
        base_sha,
        [{"layer_id": "broken", "patches": [{"path": "params.absent.key", "value": "x"}]}],
    )
    with pytest.raises(ValueError, match="/deltas/broken"):
        flatten_analysis_run_delta(
            AnalysisRunDeltaSpec.model_validate(missing_path), repo_root=tmp_path
        )

    replace_missing = _delta(
        "base.json",
        base_sha,
        [{"layer_id": "r", "patches": [{"path": "params.nonexistent", "op": "replace", "value": 1}]}],
    )
    with pytest.raises(ValueError, match="missing key"):
        flatten_analysis_run_delta(
            AnalysisRunDeltaSpec.model_validate(replace_missing), repo_root=tmp_path
        )

    duplicate_layers = _delta(
        "base.json",
        base_sha,
        [{"layer_id": "same", "patches": []}, {"layer_id": "same", "patches": []}],
    )
    with pytest.raises(ValidationError, match="layer_id values must be unique"):
        AnalysisRunDeltaSpec.model_validate(duplicate_layers)

    with pytest.raises(ValidationError, match="min_length|too_short"):
        AnalysisRunDeltaSpec.model_validate(_delta("base.json", base_sha, []))

    unsupported = _delta("base.json", base_sha, [{"layer_id": "a", "patches": []}])
    unsupported["schema_version"] = "feedbax.spec.analysis_run_delta.v0"
    with pytest.raises(ValidationError, match="unsupported AnalysisRunDeltaSpec"):
        AnalysisRunDeltaSpec.model_validate(unsupported)


def test_delta_parent_pinned_via_payload_path_selects_a_sub_document(tmp_path: Path) -> None:
    """A wrapper file's analysis sub-document is inherited without local plumbing."""
    base_doc = _analysis_base(
        "example.velocity_profiles",
        {"aggregation": "mean", "expected_grid": {"replicate_slots": 4}},
    )
    wrapper = {"notes": "shared wrapper", "analysis": base_doc}
    wrapper_sha = _write(tmp_path, "wrapper.json", wrapper)
    child = _delta(
        "wrapper.json",
        wrapper_sha,
        [{"layer_id": "child", "patches": [{"path": "params.aggregation", "value": "median"}]}],
        payload_path=["analysis"],
    )

    flattened = flatten_analysis_run_delta(
        AnalysisRunDeltaSpec.model_validate(child), repo_root=tmp_path
    )
    spec = AnalysisRunSpec.model_validate(flattened.payload)
    assert spec.params["aggregation"] == "median"
    assert spec.params["expected_grid"] == {"replicate_slots": 4}
    assert flattened.layers[0].parent_payload_path == ("analysis",)

    provenance = analysis_composition_provenance(flattened)
    assert provenance["layers"][0]["parent_payload_path"] == ["analysis"]


def test_delta_parent_payload_path_to_non_spec_fails_closed(tmp_path: Path) -> None:
    base_doc = _analysis_base("example.velocity_profiles", {"aggregation": "mean"})
    wrapper = {"notes": {"free": "text"}, "analysis": base_doc}
    wrapper_sha = _write(tmp_path, "wrapper.json", wrapper)
    child = _delta(
        "wrapper.json",
        wrapper_sha,
        [{"layer_id": "x", "patches": []}],
        payload_path=["notes"],
    )
    with pytest.raises(ValueError, match="analysis run delta parent must declare schema_id"):
        flatten_analysis_run_delta(AnalysisRunDeltaSpec.model_validate(child), repo_root=tmp_path)


def test_composition_provenance_record(tmp_path: Path) -> None:
    base_sha = _shared_velocity_base(tmp_path)
    child = _delta(
        "base.json",
        base_sha,
        [
            {
                "layer_id": "sisu",
                "patches": [
                    {"path": "params.expected_grid.conditioning", "op": "add", "value": "w"}
                ],
            }
        ],
    )
    flattened = flatten_analysis_run_delta(
        AnalysisRunDeltaSpec.model_validate(child), repo_root=tmp_path
    )
    provenance = analysis_composition_provenance(flattened)

    assert provenance["schema_id"] == ANALYSIS_COMPOSITION_PROVENANCE_SCHEMA_ID
    assert provenance["schema_version"] == ANALYSIS_COMPOSITION_PROVENANCE_SCHEMA_VERSION
    assert provenance["authored_envelope_sha256"] == flattened.authored_envelope_sha256
    assert provenance["root_spec"] == {"ref": "base.json", "sha256": base_sha}
    assert provenance["layers"] == [
        {
            "envelope_sha256": flattened.authored_envelope_sha256,
            "parent_ref": "base.json",
            "parent_sha256": base_sha,
            "layer_ids": ["sisu"],
        }
    ]
    assert provenance["attribution"] == {"params.expected_grid.conditioning": "sisu"}
    assert provenance["flattened_spec_sha256"] == sha256_bytes(
        canonical_json_bytes(dict(flattened.payload))
    )


def test_direct_spec_handling_is_unchanged(tmp_path: Path) -> None:
    direct = _analysis_base("example.velocity_profiles", {"aggregation": "mean"})
    _write(tmp_path, "direct.json", direct)

    spec, flattening = resolve_analysis_run_authoring(direct, repo_root=tmp_path)
    assert flattening is None
    assert spec == AnalysisRunSpec.model_validate(direct)

    # A direct spec needs no repo_root and is handled exactly as the landed
    # coerce path handles it (validation fills declarative defaults, unchanged).
    from_file = coerce_analysis_run_spec(tmp_path / "direct.json")
    assert from_file == spec
    assert from_file.model_dump(mode="json") == AnalysisRunSpec.model_validate(direct).model_dump(
        mode="json"
    )

    # An already-constructed AnalysisRunSpec instance passes through untouched.
    instance = AnalysisRunSpec.model_validate(direct)
    assert coerce_analysis_run_spec(instance) is instance


def test_coerce_delta_requires_repo_root(tmp_path: Path) -> None:
    base_sha = _shared_velocity_base(tmp_path)
    child = _delta("base.json", base_sha, [{"layer_id": "x", "patches": []}])
    with pytest.raises(ValueError, match="requires repo_root"):
        coerce_analysis_run_spec(child)


def test_delta_authored_execution_records_composition_provenance(tmp_path: Path) -> None:
    """A delta-authored run embeds the canonical composition record in manifest metadata."""
    base_sha = _shared_velocity_base(tmp_path)
    child = _delta(
        "base.json",
        base_sha,
        [
            {
                "layer_id": "sisu",
                "patches": [
                    {"path": "params.expected_grid.conditioning", "op": "add", "value": "w"}
                ],
            }
        ],
    )
    flattened = flatten_analysis_run_delta(
        AnalysisRunDeltaSpec.model_validate(child), repo_root=tmp_path
    )
    expected = analysis_composition_provenance(flattened)

    _register_toy_velocity_recipe()
    try:
        manifest, _path = execute_analysis_run_spec(
            child,
            root=tmp_path / "runs",
            repo_root=tmp_path,
            fig_dump_path=tmp_path / "figs",
        )
    finally:
        unregister_analysis_recipe(_EXECUTION_ANALYSIS_TYPE)

    provenance = manifest.metadata["analysis_composition"]
    assert provenance == expected
    assert provenance["schema_id"] == ANALYSIS_COMPOSITION_PROVENANCE_SCHEMA_ID
    assert provenance["schema_version"] == ANALYSIS_COMPOSITION_PROVENANCE_SCHEMA_VERSION
    assert provenance["authored_envelope_sha256"] == flattened.authored_envelope_sha256
    assert provenance["root_spec"] == {"ref": "base.json", "sha256": base_sha}
    assert provenance["layers"][0]["parent_sha256"] == base_sha
    assert provenance["layers"][0]["layer_ids"] == ["sisu"]
    assert provenance["attribution"] == {"params.expected_grid.conditioning": "sisu"}


def test_direct_spec_execution_metadata_carries_no_composition_key(tmp_path: Path) -> None:
    """A direct (non-delta) run's manifest metadata stays free of any composition key."""
    direct = _analysis_base("example.velocity_profiles", {"aggregation": "mean"})

    _register_toy_velocity_recipe()
    try:
        manifest, _path = execute_analysis_run_spec(
            direct,
            root=tmp_path / "runs",
            fig_dump_path=tmp_path / "figs",
        )
    finally:
        unregister_analysis_recipe(_EXECUTION_ANALYSIS_TYPE)

    assert "analysis_composition" not in manifest.metadata


def test_provider_manifest_publishes_the_delta_authoring_kind() -> None:
    from feedbax.integrations.provider import provider_manifest

    properties = provider_manifest().schemas["AnalysisRunDeltaSpec"]["properties"]

    assert set(properties) == {"schema_id", "schema_version", "parent", "deltas"}
    assert properties["schema_version"]["default"] == ANALYSIS_RUN_DELTA_SPEC_SCHEMA_VERSION
