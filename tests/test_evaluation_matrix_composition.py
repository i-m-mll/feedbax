from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from feedbax.analysis import harness
from feedbax.analysis.evaluation import (
    EvaluationRecipeResult,
    EvaluationRunMatrixDeltaSpec,
    EvaluationRunMatrixSpec,
    compile_evaluation_run_matrix,
    execute_evaluation_run_matrix,
    materialize_evaluation_run_matrix,
    register_evaluation_recipe,
    unregister_evaluation_recipe,
)
from feedbax.contracts import evaluation_composition
from feedbax.contracts.evaluation_composition import (
    evaluation_matrix_delta_envelope_hash,
    flatten_evaluation_run_matrix_delta,
)
from feedbax.contracts.manifest import (
    EVALUATION_MATRIX_COMPOSITION_PROVENANCE_SCHEMA_ID,
    EVALUATION_MATRIX_COMPOSITION_PROVENANCE_SCHEMA_VERSION,
    EVALUATION_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID,
    EVALUATION_RUN_MATRIX_DELTA_SPEC_SCHEMA_VERSION,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION,
    canonical_json_bytes,
    sha256_bytes,
)


def _write(tmp_path: Path, name: str, payload: dict[str, Any]) -> str:
    (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    return sha256_bytes(canonical_json_bytes(payload))


def _evaluation_base(evaluation_type: str, run_id: str) -> dict[str, Any]:
    return {
        "evaluation_type": evaluation_type,
        "training_run_ids": [run_id],
        "inputs": [],
        "params": {"target_index": 0, "replica": 0},
    }


def _shared_grid(tmp_path: Path) -> str:
    """Write the shared base and the governed target-by-replica parent matrix."""
    base_sha = _write(tmp_path, "base.json", _evaluation_base("example.compose", "train-shared"))
    parent = {
        "schema_id": EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID,
        "schema_version": EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION,
        "base": {"ref": "base.json", "sha256": base_sha},
        "axes": [
            {
                "id": "target",
                "values": [
                    {"id": f"t{index}", "deltas": [{"path": "params.target_index", "value": index}]}
                    for index in range(3)
                ],
            },
            {
                "id": "replica",
                "values": [
                    {"id": f"r{index}", "deltas": [{"path": "params.replica", "value": index}]}
                    for index in range(2)
                ],
            },
        ],
    }
    return _write(tmp_path, "parent.json", parent)


def _child(parent_ref: str, parent_sha: str, deltas: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_id": EVALUATION_RUN_MATRIX_DELTA_SPEC_SCHEMA_ID,
        "schema_version": EVALUATION_RUN_MATRIX_DELTA_SPEC_SCHEMA_VERSION,
        "parent": {"ref": parent_ref, "sha256": parent_sha},
        "deltas": deltas,
    }


def _rebased_child(
    tmp_path: Path,
    parent_sha: str,
    *,
    layer_id: str,
    evaluation_type: str,
    run_id: str,
) -> dict[str, Any]:
    base_sha = _write(tmp_path, f"base-{layer_id}.json", _evaluation_base(evaluation_type, run_id))
    return _child(
        "parent.json",
        parent_sha,
        [
            {
                "layer_id": layer_id,
                "patches": [
                    {"path": "base.ref", "value": f"base-{layer_id}.json"},
                    {"path": "base.sha256", "value": base_sha},
                ],
            }
        ],
    )


def test_two_children_share_one_pinned_grid_without_restating_axes(tmp_path: Path) -> None:
    parent_sha = _shared_grid(tmp_path)
    analytical = _rebased_child(
        tmp_path, parent_sha, layer_id="analytical", evaluation_type="example.analytical",
        run_id="train-shared",
    )
    trained = _rebased_child(
        tmp_path, parent_sha, layer_id="trained", evaluation_type="example.trained",
        run_id="train-checkpointed",
    )
    parent = json.loads((tmp_path / "parent.json").read_text(encoding="utf-8"))

    compiled = [
        compile_evaluation_run_matrix(child, repo_root=tmp_path)
        for child in (analytical, trained)
    ]
    resolved = [
        materialize_evaluation_run_matrix(child, repo_root=tmp_path)
        for child in (analytical, trained)
    ]

    for child in (analytical, trained):
        assert set(child) == {"schema_id", "schema_version", "parent", "deltas"}
    governed_axes = [
        (axis["id"], [value["id"] for value in axis["values"]]) for axis in parent["axes"]
    ]
    for matrix in (
        EvaluationRunMatrixSpec.model_validate(
            flatten_evaluation_run_matrix_delta(
                EvaluationRunMatrixDeltaSpec.model_validate(child), repo_root=tmp_path
            ).payload
        )
        for child in (analytical, trained)
    ):
        assert [(axis.id, [value.id for value in axis.values]) for axis in matrix.axes] == (
            governed_axes
        )
    assert [row.row_id for row in resolved[0]] == [row.row_id for row in resolved[1]]
    assert [row.row_id for row in resolved[0]] == [
        f"target-t{target}--replica-r{replica}" for target in range(3) for replica in range(2)
    ]
    assert [row.payload.params for row in resolved[0]] == [
        row.payload.params for row in resolved[1]
    ]
    assert {row.payload.evaluation_type for row in resolved[0]} == {"example.analytical"}
    assert {row.payload.evaluation_type for row in resolved[1]} == {"example.trained"}
    assert [row.row_id for row in compiled[0].rows] == [row.row_id for row in compiled[1].rows]


def test_nested_chain_applies_layers_root_to_child_with_attribution(tmp_path: Path) -> None:
    parent_sha = _shared_grid(tmp_path)
    mid = _rebased_child(
        tmp_path, parent_sha, layer_id="mid", evaluation_type="example.mid", run_id="train-mid"
    )
    mid_sha = _write(tmp_path, "mid.json", mid)
    leaf_base_sha = _write(
        tmp_path, "base-leaf.json", _evaluation_base("example.leaf", "train-leaf")
    )
    leaf = _child(
        "mid.json",
        mid_sha,
        [
            {
                "layer_id": "leaf",
                "patches": [
                    {"path": "base.ref", "value": "base-leaf.json"},
                    {"path": "base.sha256", "value": leaf_base_sha},
                ],
                "acknowledges_ancestor_paths": ["base.ref", "base.sha256"],
            }
        ],
    )

    flattened = flatten_evaluation_run_matrix_delta(
        EvaluationRunMatrixDeltaSpec.model_validate(leaf), repo_root=tmp_path
    )

    assert [layer.layer_ids for layer in flattened.layers] == [["mid"], ["leaf"]]
    assert flattened.layers[0].parent_ref == "parent.json"
    assert flattened.layers[0].parent_sha256 == parent_sha
    assert flattened.root_matrix is flattened.layers[0]
    assert flattened.authored_envelope_sha256 == evaluation_matrix_delta_envelope_hash(
        EvaluationRunMatrixDeltaSpec.model_validate(leaf)
    )
    assert flattened.attribution == {"base.ref": "leaf", "base.sha256": "leaf"}
    assert flattened.payload["base"] == {"ref": "base-leaf.json", "sha256": leaf_base_sha}
    assert EvaluationRunMatrixSpec.model_validate(flattened.payload).axes


def test_unacknowledged_ancestor_override_fails_closed(tmp_path: Path) -> None:
    parent_sha = _shared_grid(tmp_path)
    mid = _rebased_child(
        tmp_path, parent_sha, layer_id="mid", evaluation_type="example.mid", run_id="train-mid"
    )
    mid_sha = _write(tmp_path, "mid.json", mid)
    leaf = _child(
        "mid.json",
        mid_sha,
        [{"layer_id": "leaf", "patches": [{"path": "base.ref", "value": "base-mid.json"}]}],
    )

    with pytest.raises(ValueError, match="without explicit acknowledgement"):
        materialize_evaluation_run_matrix(leaf, repo_root=tmp_path)


def test_parent_resolution_is_pinned_confined_and_schema_checked(tmp_path: Path) -> None:
    parent_sha = _shared_grid(tmp_path)
    child = _rebased_child(
        tmp_path, parent_sha, layer_id="analytical", evaluation_type="example.analytical",
        run_id="train-shared",
    )

    tampered = json.loads((tmp_path / "parent.json").read_text(encoding="utf-8"))
    tampered["axes"][0]["values"].pop()
    (tmp_path / "parent.json").write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        materialize_evaluation_run_matrix(child, repo_root=tmp_path)
    parent_sha = _shared_grid(tmp_path)

    escaping = _child("../parent.json", parent_sha, child["deltas"])
    with pytest.raises(ValueError, match="escapes repo_root"):
        materialize_evaluation_run_matrix(escaping, repo_root=tmp_path / "nested")

    wrong_schema_sha = _write(
        tmp_path, "wrong.json", _evaluation_base("example.compose", "train-shared")
    )
    with pytest.raises(ValueError, match="must declare schema_id"):
        materialize_evaluation_run_matrix(
            _child("wrong.json", wrong_schema_sha, child["deltas"]), repo_root=tmp_path
        )

    with pytest.raises(ValueError, match="requires repo_root"):
        materialize_evaluation_run_matrix(child)


def test_repeated_parent_document_is_rejected_as_a_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_sha = _shared_grid(tmp_path)
    child = _rebased_child(
        tmp_path, parent_sha, layer_id="analytical", evaluation_type="example.analytical",
        run_id="train-shared",
    )
    monkeypatch.setattr(
        evaluation_composition,
        "load_content_pinned_json_base",
        lambda base, *, repo_root: dict(child),
    )

    with pytest.raises(ValueError, match="cycle detected"):
        flatten_evaluation_run_matrix_delta(
            EvaluationRunMatrixDeltaSpec.model_validate(child), repo_root=tmp_path
        )


def test_invalid_delta_documents_fail_closed(tmp_path: Path) -> None:
    parent_sha = _shared_grid(tmp_path)

    missing_path = _child(
        "parent.json",
        parent_sha,
        [{"layer_id": "broken", "patches": [{"path": "base.absent.ref", "value": "x"}]}],
    )
    with pytest.raises(ValueError, match="/deltas/broken"):
        materialize_evaluation_run_matrix(missing_path, repo_root=tmp_path)

    duplicate_layers = _child(
        "parent.json",
        parent_sha,
        [{"layer_id": "same", "patches": []}, {"layer_id": "same", "patches": []}],
    )
    with pytest.raises(ValidationError, match="layer_id values must be unique"):
        EvaluationRunMatrixDeltaSpec.model_validate(duplicate_layers)

    with pytest.raises(ValidationError, match="min_length|too_short"):
        EvaluationRunMatrixDeltaSpec.model_validate(_child("parent.json", parent_sha, []))

    unsupported = _child("parent.json", parent_sha, [{"layer_id": "a", "patches": []}])
    unsupported["schema_version"] = "feedbax.spec.evaluation_run_matrix_delta.v0"
    with pytest.raises(ValidationError, match="unsupported EvaluationRunMatrixDeltaSpec"):
        EvaluationRunMatrixDeltaSpec.model_validate(unsupported)


def _execute(document: dict[str, Any], *, tmp_path: Path, root: Path, **kwargs):
    def recipe(spec, _root, _states_path, _execution_context):
        return EvaluationRecipeResult(summary_metrics={"replica": spec.params["replica"]})

    register_evaluation_recipe("example.analytical", recipe, replace=True)
    register_evaluation_recipe("example.compose", recipe, replace=True)
    try:
        return execute_evaluation_run_matrix(
            document, root=root, repo_root=tmp_path, **kwargs
        )
    finally:
        unregister_evaluation_recipe("example.analytical")
        unregister_evaluation_recipe("example.compose")


def test_delta_authored_execution_records_authored_and_flattened_identities(
    tmp_path: Path,
) -> None:
    parent_sha = _shared_grid(tmp_path)
    child = _rebased_child(
        tmp_path, parent_sha, layer_id="analytical", evaluation_type="example.analytical",
        run_id="train-shared",
    )
    flattened = flatten_evaluation_run_matrix_delta(
        EvaluationRunMatrixDeltaSpec.model_validate(child), repo_root=tmp_path
    )
    flattened_matrix = EvaluationRunMatrixSpec.model_validate(flattened.payload).model_dump(
        mode="json", exclude_none=True
    )

    result = _execute(child, tmp_path=tmp_path, root=tmp_path / "runs")

    composition = result.metadata["matrix_composition"]
    assert composition["schema_id"] == EVALUATION_MATRIX_COMPOSITION_PROVENANCE_SCHEMA_ID
    assert composition["schema_version"] == EVALUATION_MATRIX_COMPOSITION_PROVENANCE_SCHEMA_VERSION
    assert composition["authored_envelope_sha256"] == flattened.authored_envelope_sha256
    assert composition["root_matrix"] == {"ref": "parent.json", "sha256": parent_sha}
    assert composition["layers"] == [
        {
            "envelope_sha256": flattened.authored_envelope_sha256,
            "parent_ref": "parent.json",
            "parent_sha256": parent_sha,
            "layer_ids": ["analytical"],
        }
    ]
    assert composition["attribution"] == {"base.ref": "analytical", "base.sha256": "analytical"}
    assert composition["flattened_matrix_sha256"] == sha256_bytes(
        canonical_json_bytes(flattened_matrix)
    )
    assert composition["canonical_row_order"] == [row.row_id for row in result.rows]
    assert set(composition["canonical_payload_sha256"]) == set(composition["canonical_row_order"])
    assert result.metadata["regeneration_parameters"]["executable_spec"] == (
        EvaluationRunMatrixDeltaSpec.model_validate(child).model_dump(
            mode="json", exclude_none=True
        )
    )
    assert result.metadata["axis_expansion"]["authored_matrix_sha256"] == sha256_bytes(
        canonical_json_bytes(flattened_matrix)
    )
    manifest_metadata = result.rows[0].result.metadata["matrix_harness"]
    assert manifest_metadata["matrix_composition"] == composition


def test_direct_matrix_execution_is_unchanged(tmp_path: Path) -> None:
    _shared_grid(tmp_path)
    direct = json.loads((tmp_path / "parent.json").read_text(encoding="utf-8"))

    result = _execute(direct, tmp_path=tmp_path, root=tmp_path / "direct")

    authored = EvaluationRunMatrixSpec.model_validate(direct).model_dump(
        mode="json", exclude_none=True
    )
    assert "matrix_composition" not in result.metadata
    assert result.metadata["regeneration_parameters"]["executable_spec"] == authored
    assert result.metadata["axis_expansion"]["authored_matrix_sha256"] == sha256_bytes(
        canonical_json_bytes(authored)
    )
    assert [row.row_id for row in result.rows] == [
        f"target-t{target}--replica-r{replica}" for target in range(3) for replica in range(2)
    ]


def test_harness_cli_executes_and_locks_a_delta_authored_matrix(tmp_path: Path) -> None:
    parent_sha = _shared_grid(tmp_path)
    child = _rebased_child(
        tmp_path, parent_sha, layer_id="analytical", evaluation_type="example.analytical",
        run_id="train-shared",
    )
    _write(tmp_path, "child.json", child)

    def recipe(*_args):
        return EvaluationRecipeResult()

    register_evaluation_recipe("example.analytical", recipe, replace=True)
    try:
        code = harness.main(
            [
                str(tmp_path / "child.json"),
                "--manifest-root",
                str(tmp_path / "cli"),
                "--repo-root",
                str(tmp_path),
                "--locked-spec",
                str(tmp_path / "child.json"),
            ]
        )
    finally:
        unregister_evaluation_recipe("example.analytical")

    assert code == 0
    manifests = sorted((tmp_path / "cli").glob("*/manifests/evaluation_runs/*.json"))
    assert len(manifests) == 6


def test_provider_manifest_publishes_the_delta_authoring_kind() -> None:
    from feedbax.integrations.provider import provider_manifest

    properties = provider_manifest().schemas["EvaluationRunMatrixDeltaSpec"]["properties"]

    assert set(properties) == {"schema_id", "schema_version", "parent", "deltas"}
    assert properties["schema_version"]["default"] == (
        EVALUATION_RUN_MATRIX_DELTA_SPEC_SCHEMA_VERSION
    )
