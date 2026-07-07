from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.contracts.checkpoints import (
    LEGACY_CHECKPOINT_LEAF_MANIFEST_SCHEMA_ID,
    LEGACY_CHECKPOINT_LEAF_MANIFEST_SCHEMA_VERSION,
    LEGACY_CHECKPOINT_LEAF_MANIFEST_SCHEMA_VERSION_V0,
)
from feedbax.contracts.training import (
    STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
    LossTermSpec,
    ObjectiveSlotSpec,
    TaskSpec,
    TrainingConfig,
    TrainingRunSpec,
    WorkerExecutionSpec,
    standard_supervised_method_payload,
    standard_supervised_method_ref,
)
from feedbax.contracts.worker import (
    EffectivePhaseSpec,
    ProgressCoordinate,
    derive_consistency_predicate,
    toy_minimax_method_contract,
)
from feedbax.training.legacy_checkpoint_adoption import (
    LegacyManifestSchemaError,
    LegacyPathMappingError,
    LegacyStreamMismatchError,
    ManifestDumpRequest,
    PathMappingRegistry,
    PathMappingRule,
    _DUMP_SCRIPT,
    accept_leaf_manifest,
    adopt_legacy_checkpoint,
    adopt_tree_from_legacy_stream,
    dump_leaf_manifests_via_worktrees,
    manifest_from_trees,
    read_raw_np_save_stream,
)


def _minimal_graph() -> dict[str, object]:
    return {
        "nodes": {
            "gain": {
                "type": "Gain",
                "params": {"gain": 1.0},
                "input_ports": ["input"],
                "output_ports": ["output"],
            }
        },
        "wires": [],
        "input_ports": ["input"],
        "output_ports": ["output"],
        "input_bindings": {"input": ("gain", "input")},
        "output_bindings": {"output": ("gain", "output")},
    }


def _run_spec() -> TrainingRunSpec:
    contract = toy_minimax_method_contract()
    program = contract.phase_program.model_copy(deep=True)
    program.checkpoint_barriers[0].metadata["consistency_mode"] = "population-barrier"
    method_contract = contract.model_copy(
        update={
            "method_ref": "feedbax/standard_supervised/v1",
            "method_payload_schema_version": STANDARD_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
            "phase_program": program,
        }
    )
    effective_phase = EffectivePhaseSpec(
        method_ref="feedbax/standard_supervised/v1",
        axes=method_contract.axes,
        state_slots=method_contract.state_slots,
        phase_program=program,
        consistency_predicate=derive_consistency_predicate(program),
    )
    return TrainingRunSpec(
        graph={"inline": _minimal_graph()},
        task=TaskSpec(type="ReachingTask", params={"n_steps": 4}),
        training_config=TrainingConfig(n_batches=4, batch_size=3),
        objective=ObjectiveSlotSpec(
            loss=LossTermSpec(type="target_state", label="target", selector="output")
        ),
        method_ref=standard_supervised_method_ref(),
        method_payload=standard_supervised_method_payload(),
        worker_execution=WorkerExecutionSpec(
            method_contract=method_contract,
            effective_phase=effective_phase,
        ),
    )


def _coordinate(step: int = 12) -> ProgressCoordinate:
    return ProgressCoordinate(
        run_id="legacy-adopt-test",
        phase="warmup",
        global_step=step,
        completed_barrier="after_warmup",
    )


def _slots() -> dict[str, object]:
    return {
        "controller": jnp.array([0.0, 0.0], dtype=jnp.float32),
        "controller_optimizer": {"count": jnp.array(0, dtype=jnp.int32)},
        "adversary_population": [jnp.array([0.1, 0.2]), jnp.array([0.3, 0.4])],
        "adversary_optimizer": {"count": jnp.array([1, 1])},
        "rng": jnp.array([11, 22], dtype=jnp.uint32),
        "loss": [0.5],
    }


def _manifest_payload(
    *,
    model_entries: list[dict[str, object]] | None = None,
    optimizer_entries: list[dict[str, object]] | None = None,
    schema_version: str = LEGACY_CHECKPOINT_LEAF_MANIFEST_SCHEMA_VERSION,
) -> dict[str, object]:
    return {
        "kind": "LegacyCheckpointLeafManifest",
        "schema_id": LEGACY_CHECKPOINT_LEAF_MANIFEST_SCHEMA_ID,
        "schema_version": schema_version,
        "model": model_entries
        if model_entries is not None
        else [
            {
                "tree_path": "/old/static",
                "kind": "static",
                "shape": [],
                "dtype": "bool",
                "static_repr_sha256": "not-current",
            },
            {
                "tree_path": "/old/controller",
                "kind": "array",
                "shape": [2],
                "dtype": "float32",
            },
        ],
        "optimizer": optimizer_entries
        if optimizer_entries is not None
        else [
            {
                "tree_path": "/old/optimizer/count",
                "kind": "array",
                "shape": [],
                "dtype": "int32",
            }
        ],
        "provenance": {
            "producing_commit": "abc123",
            "spec_ref": "spec.json",
            "spec_hash": "0" * 64,
            "dumped_at": "2026-07-07T00:00:00+00:00",
            "dumper_version": "test",
        },
    }


def _write_stream(path: Path, arrays: list[np.ndarray]) -> None:
    with path.open("wb") as stream:
        for array in arrays:
            np.save(stream, array, allow_pickle=False)


def test_leaf_manifest_accepts_current_migrates_v0_and_rejects_tampered() -> None:
    current = accept_leaf_manifest(_manifest_payload())
    assert current.schema_version == LEGACY_CHECKPOINT_LEAF_MANIFEST_SCHEMA_VERSION

    migrated = accept_leaf_manifest(
        {
            "schema_version": LEGACY_CHECKPOINT_LEAF_MANIFEST_SCHEMA_VERSION_V0,
            "leaves": {"model": _manifest_payload()["model"], "optimizer": []},
            "producing_commit": "abc123",
        }
    )
    assert migrated.schema_id == LEGACY_CHECKPOINT_LEAF_MANIFEST_SCHEMA_ID
    assert migrated.model[0].tree_path == "/old/static"
    assert migrated.model[0].shape == ()
    assert migrated.model[0].dtype == "bool"

    static_without_stream_metadata = _manifest_payload(
        model_entries=[
            {
                "tree_path": "/old/static",
                "kind": "static",
                "static_repr_sha256": "not-current",
            }
        ]
    )
    assert accept_leaf_manifest(static_without_stream_metadata).model[0].shape is None

    tampered = _manifest_payload(
        schema_version="feedbax.manifest.legacy_checkpoint_leaf_manifest.tampered"
    )
    with pytest.raises(LegacyManifestSchemaError, match="Unsupported Feedbax"):
        accept_leaf_manifest(tampered)


def test_manifest_from_trees_skips_default_equinox_ignored_static_leaves() -> None:
    manifest = manifest_from_trees(
        model={
            "weight": jnp.array([1.0], dtype=jnp.float32),
            "flag": True,
            "scale": 2.0,
            "name": "ignored",
            "opaque": object(),
        },
        optimizer=None,
        producing_commit="abc123",
    )

    paths = [entry.tree_path for entry in manifest.model]
    assert paths == ["/flag", "/scale", "/weight"]
    by_path = {entry.tree_path: entry for entry in manifest.model}
    assert by_path["/flag"].kind == "static"
    assert by_path["/flag"].shape == ()
    assert by_path["/flag"].dtype == "bool"
    assert by_path["/scale"].kind == "static"
    assert by_path["/scale"].dtype == "float64"
    assert by_path["/weight"].kind == "array"


def test_self_contained_dump_script_skips_ignored_static_leaves(tmp_path: Path) -> None:
    script = tmp_path / "dump_script.py"
    builder = tmp_path / "builder.py"
    spec = tmp_path / "spec.json"
    output = tmp_path / "manifest.json"
    script.write_text(_DUMP_SCRIPT, encoding="utf-8")
    builder.write_text(
        """
import jax.numpy as jnp
import numpy as np


def build(_spec):
    return {
        "weight": jnp.array([1.0], dtype=jnp.float32),
        "np_weight": np.array([2.0], dtype=np.float32),
        "flag": True,
        "count": 3,
        "name": "ignored",
        "opaque": object(),
    }, None
""",
        encoding="utf-8",
    )
    spec.write_text(json.dumps({"manifest_builder": "builder:build"}), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--spec",
            str(spec),
            "--output",
            str(output),
            "--commit",
            "abc123",
        ],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    paths = [entry["tree_path"] for entry in payload["model"]]
    assert paths == ["/count", "/flag", "/np_weight", "/weight"]
    by_path = {entry["tree_path"]: entry for entry in payload["model"]}
    assert by_path["/count"]["kind"] == "static"
    assert by_path["/count"]["shape"] == []
    assert by_path["/count"]["dtype"] == "int64"
    assert by_path["/flag"]["kind"] == "static"
    assert by_path["/flag"]["dtype"] == "bool"
    assert by_path["/np_weight"]["kind"] == "array"
    assert by_path["/weight"]["kind"] == "array"


@pytest.mark.parametrize(
    ("arrays", "match"),
    [
        ([], "record count mismatch"),
        (
            [
                np.array(True, dtype=np.bool_),
                np.array([1.0, 2.0], dtype=np.float32),
                np.array([3.0], dtype=np.float32),
            ],
            "extra records",
        ),
        (
            [np.array(True, dtype=np.bool_), np.array([1.0], dtype=np.float32)],
            "record 1 /old/controller: shape file=\\(1,\\)",
        ),
        (
            [np.array(True, dtype=np.bool_), np.array([1, 2], dtype=np.int32)],
            "record 1 /old/controller: dtype file=int32",
        ),
    ],
)
def test_raw_stream_reader_fails_closed_for_count_shape_and_dtype(
    tmp_path: Path,
    arrays: list[np.ndarray],
    match: str,
) -> None:
    stream = tmp_path / "model.eqx"
    _write_stream(stream, arrays)
    manifest = accept_leaf_manifest(_manifest_payload(optimizer_entries=[]))

    with pytest.raises(LegacyStreamMismatchError, match=match):
        read_raw_np_save_stream(stream, manifest.model)


def test_path_keyed_tree_adoption_rejects_ambiguous_mapping_rule(tmp_path: Path) -> None:
    stream = tmp_path / "model.eqx"
    _write_stream(
        stream,
        [np.array(True, dtype=np.bool_), np.array([1.0, 2.0], dtype=np.float32)],
    )
    manifest = accept_leaf_manifest(_manifest_payload(optimizer_entries=[]))

    with pytest.raises(LegacyPathMappingError, match="ambiguous mapping rules"):
        adopt_tree_from_legacy_stream(
            stream,
            manifest.model,
            jnp.array([0.0, 0.0], dtype=jnp.float32),
            mapping_rules=[
                PathMappingRule("/old/controller", "/"),
                PathMappingRule("/old/controller", "/"),
            ],
        )


def test_path_mapping_registry_is_versioned_and_tree_scoped() -> None:
    registry = PathMappingRegistry(
        rules=[
            {
                "tree": "model",
                "old_path": "/old/controller",
                "new_path": "/controller",
            },
            {
                "tree": "optimizer",
                "old_path": "/old/count",
                "new_path": "/count",
            },
        ]
    )

    assert registry.rules_for("model") == (PathMappingRule("/old/controller", "/controller"),)
    assert registry.rules_for("optimizer") == (PathMappingRule("/old/count", "/count"),)
    with pytest.raises(ValueError, match="unsupported path mapping schema_version"):
        PathMappingRegistry.model_validate(
            {"schema_version": "feedbax.training.legacy_checkpoint_path_mapping.v0"}
        )


def test_path_keyed_tree_adoption_lists_unmatched_and_unfilled_leaves(
    tmp_path: Path,
) -> None:
    stream = tmp_path / "model.eqx"
    _write_stream(
        stream,
        [np.array(True, dtype=np.bool_), np.array([1.0, 2.0], dtype=np.float32)],
    )
    manifest = accept_leaf_manifest(_manifest_payload(optimizer_entries=[]))

    with pytest.raises(LegacyPathMappingError) as excinfo:
        adopt_tree_from_legacy_stream(
            stream,
            manifest.model,
            jnp.array([0.0, 0.0], dtype=jnp.float32),
        )

    message = str(excinfo.value)
    assert "unmatched old array leaves" in message
    assert "unfilled current array leaves" in message


def test_static_stream_entries_verify_in_order_but_do_not_populate_current_statics(
    tmp_path: Path,
) -> None:
    stream = tmp_path / "model.eqx"
    _write_stream(
        stream,
        [
            np.array([1.0], dtype=np.float32),
            np.array(True, dtype=np.bool_),
            np.array([2.0], dtype=np.float32),
        ],
    )
    manifest = accept_leaf_manifest(
        _manifest_payload(
            model_entries=[
                {
                    "tree_path": "/left",
                    "kind": "array",
                    "shape": [1],
                    "dtype": "float32",
                },
                {
                    "tree_path": "/flag",
                    "kind": "static",
                    "shape": [],
                    "dtype": "bool",
                    "static_repr_sha256": "legacy-true",
                },
                {
                    "tree_path": "/right",
                    "kind": "array",
                    "shape": [1],
                    "dtype": "float32",
                },
            ],
            optimizer_entries=[],
        )
    )
    current = {
        "left": jnp.array([0.0], dtype=jnp.float32),
        "flag": False,
        "right": jnp.array([0.0], dtype=jnp.float32),
    }

    adopted, report = adopt_tree_from_legacy_stream(stream, manifest.model, current)

    assert adopted["left"].tolist() == [1.0]
    assert adopted["right"].tolist() == [2.0]
    assert adopted["flag"] is False
    assert report.static_paths[0].status == "different"


def test_end_to_end_synthetic_legacy_checkpoint_round_trips_through_custody(
    tmp_path: Path,
) -> None:
    model_stream = tmp_path / "model.eqx"
    optimizer_stream = tmp_path / "optimizer_state.eqx"
    _write_stream(
        model_stream,
        [np.array(True, dtype=np.bool_), np.array([5.0, 6.0], dtype=np.float32)],
    )
    _write_stream(optimizer_stream, [np.array(7, dtype=np.int32)])
    run_spec = _run_spec()
    slots = _slots()

    result = adopt_legacy_checkpoint(
        checkpoint_root=tmp_path / "checkpoints",
        run_spec=run_spec,
        phase_program=run_spec.worker_execution.method_contract.phase_program,
        barrier_name="after_warmup",
        coordinate=_coordinate(),
        current_slots=slots,
        leaf_manifest=_manifest_payload(),
        model_stream=model_stream,
        model_slot="controller",
        optimizer_stream=optimizer_stream,
        optimizer_slot="controller_optimizer",
        model_mapping_rules=[PathMappingRule("/old/controller", "/")],
        optimizer_mapping_rules=[PathMappingRule("/old/optimizer/count", "/count")],
    )

    assert result.write.latest_pointer_path.is_file()
    assert result.loaded_slots["controller"].tolist() == [5.0, 6.0]
    assert result.loaded_slots["controller_optimizer"]["count"].item() == 7
    assert result.model_report.static_paths[0].status == "missing_current_static"


def test_optimizer_adoption_allows_resize_through_resume_slot_transform(
    tmp_path: Path,
) -> None:
    model_stream = tmp_path / "model.eqx"
    optimizer_stream = tmp_path / "optimizer_state.eqx"
    _write_stream(
        model_stream,
        [np.array(True, dtype=np.bool_), np.array([5.0, 6.0], dtype=np.float32)],
    )
    _write_stream(
        optimizer_stream,
        [
            np.array(7, dtype=np.int32),
            np.ones((5, 4), dtype=np.float32),
        ],
    )
    optimizer_entries = [
        {
            "tree_path": "/old/optimizer/count",
            "kind": "array",
            "shape": [],
            "dtype": "int32",
        },
        {
            "tree_path": "/old/optimizer/diag",
            "kind": "array",
            "shape": [5, 4],
            "dtype": "float32",
        },
    ]
    run_spec = _run_spec()
    slots = _slots()
    slots["controller_optimizer"] = {
        "count": jnp.array(0, dtype=jnp.int32),
        "diag": jnp.zeros((5, 6), dtype=jnp.float32),
    }

    def resize_optimizer(slots):
        transformed = dict(slots)
        optimizer = dict(transformed["controller_optimizer"])
        optimizer["diag"] = jnp.pad(optimizer["diag"], ((0, 0), (0, 2)))
        transformed["controller_optimizer"] = optimizer
        return transformed

    result = adopt_legacy_checkpoint(
        checkpoint_root=tmp_path / "checkpoints",
        run_spec=run_spec,
        phase_program=run_spec.worker_execution.method_contract.phase_program,
        barrier_name="after_warmup",
        coordinate=_coordinate(),
        current_slots=slots,
        leaf_manifest=_manifest_payload(optimizer_entries=optimizer_entries),
        model_stream=model_stream,
        model_slot="controller",
        optimizer_stream=optimizer_stream,
        optimizer_slot="controller_optimizer",
        model_mapping_rules=[PathMappingRule("/old/controller", "/")],
        optimizer_mapping_rules=[
            PathMappingRule("/old/optimizer/count", "/count"),
            PathMappingRule("/old/optimizer/diag", "/diag"),
        ],
        resume_slot_transform=resize_optimizer,
    )

    assert result.loaded_slots["controller_optimizer"]["diag"].shape == (5, 6)
    assert result.loaded_slots["controller_optimizer"]["diag"][:, :4].tolist() == (
        np.ones((5, 4), dtype=np.float32).tolist()
    )


def test_fresh_optimizer_must_be_explicit(tmp_path: Path) -> None:
    model_stream = tmp_path / "model.eqx"
    _write_stream(
        model_stream,
        [np.array(True, dtype=np.bool_), np.array([5.0, 6.0], dtype=np.float32)],
    )
    run_spec = _run_spec()

    with pytest.raises(LegacyPathMappingError, match="fresh_optimizer=True"):
        adopt_legacy_checkpoint(
            checkpoint_root=tmp_path / "checkpoints",
            run_spec=run_spec,
            phase_program=run_spec.worker_execution.method_contract.phase_program,
            barrier_name="after_warmup",
            coordinate=_coordinate(),
            current_slots=_slots(),
            leaf_manifest=_manifest_payload(),
            model_stream=model_stream,
            model_slot="controller",
            model_mapping_rules=[PathMappingRule("/old/controller", "/")],
        )


def test_optimizer_adoption_allows_post_load_resume_transform(tmp_path: Path) -> None:
    model_stream = tmp_path / "model.eqx"
    optimizer_stream = tmp_path / "optimizer_state.eqx"
    _write_stream(
        model_stream,
        [np.array(True, dtype=np.bool_), np.array([5.0, 6.0], dtype=np.float32)],
    )
    _write_stream(optimizer_stream, [np.array([7], dtype=np.int32)])
    run_spec = _run_spec()
    slots = _slots()
    slots["controller_optimizer"] = {"count": jnp.array([0, 0], dtype=jnp.int32)}

    def resize_optimizer(loaded_slots):
        transformed = dict(loaded_slots)
        optimizer = dict(transformed["controller_optimizer"])
        optimizer["count"] = jnp.pad(optimizer["count"], (0, 1))
        transformed["controller_optimizer"] = optimizer
        return transformed

    result = adopt_legacy_checkpoint(
        checkpoint_root=tmp_path / "checkpoints",
        run_spec=run_spec,
        phase_program=run_spec.worker_execution.method_contract.phase_program,
        barrier_name="after_warmup",
        coordinate=_coordinate(),
        current_slots=slots,
        leaf_manifest=_manifest_payload(
            optimizer_entries=[
                {
                    "tree_path": "/old/optimizer/count",
                    "kind": "array",
                    "shape": [1],
                    "dtype": "int32",
                }
            ]
        ),
        model_stream=model_stream,
        model_slot="controller",
        optimizer_stream=optimizer_stream,
        optimizer_slot="controller_optimizer",
        model_mapping_rules=[PathMappingRule("/old/controller", "/")],
        optimizer_mapping_rules=[PathMappingRule("/old/optimizer/count", "/count")],
        resume_slot_transform=resize_optimizer,
    )

    assert result.loaded_slots["controller_optimizer"]["count"].tolist() == [7, 0]


def test_dump_manifest_batch_groups_worktrees_and_cleans_up(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_runner(args, **kwargs):
        del kwargs
        calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, "", "")

    requests = (
        ManifestDumpRequest("abc123", tmp_path / "a.json", tmp_path / "a.out.json"),
        ManifestDumpRequest("abc123", tmp_path / "b.json", tmp_path / "b.out.json"),
        ManifestDumpRequest("def456", tmp_path / "c.json", tmp_path / "c.out.json"),
    )

    results = dump_leaf_manifests_via_worktrees(
        requests,
        repo=tmp_path,
        builder="old.builder:build",
        run_uv_sync=False,
        runner=fake_runner,
    )

    assert len(results) == 3
    assert sum(call[:3] == ["git", "-C", str(tmp_path)] and call[3:6] == [
        "cat-file",
        "-e",
        "abc123^{commit}",
    ] for call in calls) == 1
    add_calls = [call for call in calls if call[3:6] == ["worktree", "add", "--detach"]]
    remove_calls = [call for call in calls if call[3:6] == ["worktree", "remove", "--force"]]
    run_calls = [call for call in calls if call[:3] == ["uv", "run", "--no-sync"]]
    assert len(add_calls) == 2
    assert len(remove_calls) == 2
    assert len(run_calls) == 3
