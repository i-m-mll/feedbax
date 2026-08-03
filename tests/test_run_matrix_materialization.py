from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from feedbax.contracts.expressions import Coalesce, ValueQuery
from feedbax.contracts.spec_storage import training_spec_sha256
from feedbax.contracts.checkpoints import CheckpointSegmentLineage
from feedbax.contracts.run_matrix import (
    RUN_MATRIX_MATERIALIZATION_SCHEMA_VERSION,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION_V5,
    MatrixCompositionDelta,
    TrainingRunMatrixSpec,
    TrainingRunMatrixSpecV5,
    apply_composition_deltas,
    apply_override_patches,
)
from feedbax.contracts.training import (
    LossTermSpec,
    ObjectiveSlotSpec,
    TaskSpec,
    TrainingConfig,
    TrainingRunSpec,
    WorkerExecutionSpec,
    default_training_method_registry,
    standard_supervised_effective_phase_spec,
    standard_supervised_method_contract,
    standard_supervised_method_payload,
    standard_supervised_method_ref,
)
from feedbax.training.run_matrix import (
    RunMatrixError,
    materialize_run_matrix,
    render_spec_lock_table,
    write_materialized_matrix,
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


def _training_run_payload() -> dict[str, object]:
    worker = WorkerExecutionSpec(
        method_contract=standard_supervised_method_contract(),
        effective_phase=standard_supervised_effective_phase_spec(),
    )
    spec = TrainingRunSpec(
        graph={"inline": _minimal_graph()},
        task=TaskSpec(type="ReachingTask", params={"n_steps": 4}),
        training_config=TrainingConfig(n_batches=2, batch_size=3, learning_rate=0.01),
        objective=ObjectiveSlotSpec(
            loss=LossTermSpec(type="target_state", label="target", selector="output")
        ),
        method_ref=standard_supervised_method_ref(),
        method_payload=standard_supervised_method_payload(),
        worker_execution=worker,
    )
    return spec.model_dump(mode="json")


def _matrix(base: dict[str, object]) -> TrainingRunMatrixSpec:
    return TrainingRunMatrixSpec.model_validate(
        {
            "schema_id": TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
            "schema_version": TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
            "name": "lr rows",
            "issue": "60c12f1",
            "base": {"kind": "inline", "inline": base},
            "fork": {"lr_continuation": "continue"},
            "rows": [
                {
                    "row_id": "lr_hi",
                    "seed": 11,
                    "overrides": [
                        {"path": "training_config.learning_rate", "op": "replace", "value": 0.02},
                        {"path": "metadata.row", "op": "add", "value": "hi"},
                    ],
                },
                {
                    "row_id": "lr_lo",
                    "seed": 12,
                    "overrides": [
                        {"path": "training_config.learning_rate", "op": "replace", "value": 0.002}
                    ],
                },
            ],
        }
    )


def test_matrix_v5_object_materializes_without_v6_reinterpretation(tmp_path: Path) -> None:
    current = _matrix(_training_run_payload()).model_dump(mode="json", exclude_none=True)
    current["schema_version"] = TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION_V5
    legacy = TrainingRunMatrixSpecV5.model_validate(current)

    materialized = materialize_run_matrix(
        legacy,
        repo_root=tmp_path,
        method_registry=default_training_method_registry(),
    )

    assert materialized.run_set_manifest.metadata["matrix_schema_version"] == (
        TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION_V5
    )


def test_apply_override_patches_is_fail_closed() -> None:
    base = {"a": {"b": 1}, "items": [0, 1]}

    patched = apply_override_patches(
        base,
        [
            {"path": "a.b", "op": "replace", "value": 2},  # type: ignore[list-item]
            {"path": "a.c", "op": "add", "value": 3},  # type: ignore[list-item]
            {"path": "items.0", "op": "remove"},  # type: ignore[list-item]
        ],
    )

    assert patched == {"a": {"b": 2, "c": 3}, "items": [1]}
    assert base == {"a": {"b": 1}, "items": [0, 1]}
    with pytest.raises(ValueError, match="missing"):
        apply_override_patches(base, [{"path": "a.x", "op": "replace", "value": 2}])  # type: ignore[list-item]


def test_apply_override_patches_append_via_numeric_index() -> None:
    base = {"items": [0, 1]}

    patched = apply_override_patches(
        base,
        [{"path": "items.2", "op": "add", "value": 2}],  # type: ignore[list-item]
    )

    assert patched == {"items": [0, 1, 2]}
    assert base == {"items": [0, 1]}


def test_apply_override_patches_append_via_dash_token() -> None:
    base = {"items": [0, 1]}

    patched = apply_override_patches(
        base,
        [{"path": "items.-", "op": "add", "value": 2}],  # type: ignore[list-item]
    )

    assert patched == {"items": [0, 1, 2]}


def test_apply_override_patches_nested_append_into_deltas_patches_list() -> None:
    base = {
        "deltas": [
            {
                "layer_id": "warm",
                "patches": [{"path": "lr", "op": "replace", "value": 0.1}],
            }
        ]
    }

    patched = apply_override_patches(
        base,
        [
            {
                "path": "deltas.0.patches.1",
                "op": "add",
                "value": {"path": "batch_size", "op": "add", "value": 64},
            }
        ],  # type: ignore[list-item]
    )

    assert patched["deltas"][0]["patches"] == [
        {"path": "lr", "op": "replace", "value": 0.1},
        {"path": "batch_size", "op": "add", "value": 64},
    ]
    assert len(base["deltas"][0]["patches"]) == 1


def test_apply_override_patches_beyond_range_index_still_rejected() -> None:
    base = {"items": [0, 1]}

    with pytest.raises(ValueError, match="items.5"):
        apply_override_patches(
            base,
            [{"path": "items.5", "op": "add", "value": 2}],  # type: ignore[list-item]
        )


def test_apply_override_patches_replace_and_remove_semantics_unchanged() -> None:
    base = {"items": [0, 1, 2]}

    replaced = apply_override_patches(
        base,
        [{"path": "items.1", "op": "replace", "value": 9}],  # type: ignore[list-item]
    )
    assert replaced == {"items": [0, 9, 2]}

    removed = apply_override_patches(base, [{"path": "items.2", "op": "remove"}])  # type: ignore[list-item]
    assert removed == {"items": [0, 1]}

    with pytest.raises(ValueError, match="missing key/index"):
        apply_override_patches(base, [{"path": "items.9", "op": "replace", "value": 1}])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="missing key/index"):
        apply_override_patches(base, [{"path": "items.9", "op": "remove"}])  # type: ignore[list-item]


def test_continuation_matrix_authoring_contract_worked_example() -> None:
    """Runnable twin of the worked example in docs/design/run_matrix_continuation_contract.md.

    Keep this test and that document's example in sync: this is the "maintained
    minimal example" the continuation matrix-authoring contract doc points to.
    """
    # An ancestor matrix spec's `deltas`, as a plain serialized JSON document.
    ancestor_spec_document = {
        "deltas": [
            {
                "layer_id": "warm_restart",
                "patches": [
                    {"path": "training_config.n_batches", "op": "replace", "value": 150},
                ],
            }
        ],
    }

    # Author the continuation by appending a new patch to the existing delta's
    # `patches` list instead of replacing the whole list. This is the capability
    # this contract adds: an `add` targeting index == len(list), or "-".
    continuation_spec_document = apply_override_patches(
        ancestor_spec_document,
        [
            {
                "path": "deltas.0.patches.1",
                "op": "add",
                "value": {"path": "training_config.batch_size", "op": "add", "value": 64},
            }
        ],
    )
    assert continuation_spec_document["deltas"][0]["patches"] == [
        {"path": "training_config.n_batches", "op": "replace", "value": 150},
        {"path": "training_config.batch_size", "op": "add", "value": 64},
    ]
    assert ancestor_spec_document["deltas"][0]["patches"] == [
        {"path": "training_config.n_batches", "op": "replace", "value": 150},
    ]  # the ancestor document is untouched; the continuation is a distinct document

    # Canonicalize/hash/pin the base (elided here), then flatten the deltas onto it.
    deltas = [
        MatrixCompositionDelta.model_validate(delta)
        for delta in continuation_spec_document["deltas"]
    ]
    base_payload = {"training_config": {"learning_rate": 0.01, "n_batches": 100}}
    resolved_base, _attribution, _written = apply_composition_deltas(base_payload, deltas)
    assert resolved_base == {
        "training_config": {"learning_rate": 0.01, "n_batches": 150, "batch_size": 64},
    }

    # Each row applies its own override on top of the pinned base; the row's
    # identity derives from this patched result, not from the base hash unchanged.
    row_payload = apply_override_patches(
        resolved_base,
        [{"path": "training_config.learning_rate", "op": "replace", "value": 0.02}],
    )
    assert row_payload == {
        "training_config": {"learning_rate": 0.02, "n_batches": 150, "batch_size": 64},
    }
    assert training_spec_sha256(row_payload) != training_spec_sha256(resolved_base)


def test_materialize_explicit_rows_plans_stable_ids_and_writes_deterministic_bytes(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    materialized = materialize_run_matrix(
        _matrix(_training_run_payload()),
        repo_root=tmp_path,
        method_registry=application_registry_bundle.training_methods,
        row_lowerer=application_registry_bundle.row_lowerers.lower,
    )
    second = materialize_run_matrix(
        _matrix(_training_run_payload()),
        repo_root=tmp_path,
        method_registry=application_registry_bundle.training_methods,
        row_lowerer=application_registry_bundle.row_lowerers.lower,
    )

    assert [row.row_id for row in materialized.rows] == ["lr_hi", "lr_lo"]
    assert materialized.rows[0].payload["training_config"]["learning_rate"] == 0.02
    assert materialized.rows[0].payload["metadata"]["row"] == "hi"
    assert materialized.rows[0].planned_run_id != materialized.rows[1].planned_run_id
    assert [row.planned_run_id for row in materialized.rows] == [
        row.planned_run_id for row in second.rows
    ]
    first_manifest = write_materialized_matrix(materialized, tmp_path / "first", wrap_key="wrapped")
    second_manifest = write_materialized_matrix(second, tmp_path / "second", wrap_key="wrapped")

    assert first_manifest == second_manifest
    assert first_manifest["schema_version"] == RUN_MATRIX_MATERIALIZATION_SCHEMA_VERSION
    assert first_manifest["rows"][0]["row_provenance"]["planned_run_id"] == (
        materialized.rows[0].planned_run_id
    )
    assert first_manifest["rows"][0]["row_provenance"][
        "lowered_execution_payload_hash"
    ] == training_spec_sha256(materialized.rows[0].payload)
    assert (tmp_path / "first" / "lr_hi.json").read_text().startswith('{"wrapped"')


def test_materialize_explicit_row_override_failure_names_row_and_violation(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    matrix = _matrix(_training_run_payload())
    broken_row = matrix.rows[0].model_copy(
        update={
            "overrides": [{"path": "training_config.missing_field", "op": "replace", "value": 1}]
        }
    )
    matrix = matrix.model_copy(update={"rows": [broken_row, matrix.rows[1]]})

    with pytest.raises(RunMatrixError, match=r"/rows/lr_hi/overrides.*missing key/index"):
        materialize_run_matrix(
            matrix,
            repo_root=tmp_path,
            method_registry=application_registry_bundle.training_methods,
            row_lowerer=application_registry_bundle.row_lowerers.lower,
        )


def test_materialize_sweep_mode_uses_shared_axes_and_coordinates(
    tmp_path: Path, application_registry_bundle
) -> None:
    payload = {
        "schema_id": TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
        "schema_version": TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
        "name": "sweep",
        "base": {"kind": "inline", "inline": _training_run_payload()},
        "derivations": [
            {
                "output_path": "metadata.derived_learning_rate",
                "query": ValueQuery(item="row", path="training_config.learning_rate"),
            }
        ],
        "axes": [
            {
                "id": "lr",
                "path": "training_config.learning_rate",
                "variation": {"kind": "explicit", "values": [0.1, 0.01]},
            },
            {"id": "seed", "path": "seed", "variation": {"kind": "explicit", "values": [1, 2]}},
        ],
        "combination": {"mode": "zip"},
    }

    materialized = materialize_run_matrix(
        payload,
        repo_root=tmp_path,
        method_registry=application_registry_bundle.training_methods,
        row_lowerer=application_registry_bundle.row_lowerers.lower,
    )

    assert [row.payload["training_config"]["learning_rate"] for row in materialized.rows] == [
        0.1,
        0.01,
    ]
    assert [row.payload["metadata"]["derived_learning_rate"] for row in materialized.rows] == [
        0.1,
        0.01,
    ]
    assert [row.seed for row in materialized.rows] == [1, 2]
    assert materialized.run_set_manifest.axes.runs[1].values == {"lr": 0.01, "seed": 2}


def test_derivations_and_base_ref_sha_pin_are_fail_closed(
    tmp_path: Path, application_registry_bundle
) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"lr": 0.03}), encoding="utf-8")
    base = tmp_path / "base.json"
    base_payload = {"envelope": {"feedbax_training_run_spec": _training_run_payload()}}
    base_bytes = json.dumps(base_payload).encode("utf-8")
    base.write_bytes(base_bytes)
    payload = {
        "schema_id": TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
        "schema_version": TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
        "name": "derived",
        "base": {
            "kind": "authored_intent",
            "ref": "base.json",
            "payload_path": "envelope.feedbax_training_run_spec",
            "content_hash": training_spec_sha256(base_payload),
        },
        "sources": [{"alias": "src", "kind": "manifest", "uri": "source.json"}],
        "derivations": [
            {
                "output_path": "metadata.derived_learning_rate",
                "query": Coalesce(queries=[ValueQuery(item="src", path="missing")], default=0.04),
            }
        ],
        "rows": [{"row_id": "derived", "overrides": []}],
    }

    materialized = materialize_run_matrix(
        payload,
        repo_root=tmp_path,
        method_registry=application_registry_bundle.training_methods,
        row_lowerer=application_registry_bundle.row_lowerers.lower,
    )
    assert materialized.rows[0].payload["metadata"]["derived_learning_rate"] == 0.04

    payload["base"]["content_hash"] = "0" * 64
    with pytest.raises(RunMatrixError, match="canonical content hash mismatch"):
        materialize_run_matrix(
            payload,
            repo_root=tmp_path,
            method_registry=application_registry_bundle.training_methods,
            row_lowerer=application_registry_bundle.row_lowerers.lower,
        )


def test_derivations_use_each_delta_applied_row_and_preserve_authored_fields(
    tmp_path: Path,
    application_registry_bundle,
) -> None:
    payload = {
        "schema_id": TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
        "schema_version": TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
        "name": "per-row derivation",
        "base": {
            "kind": "inline",
            "inline": {
                **_training_run_payload(),
                "metadata": {"derived_learning_rate": None},
            },
        },
        "derivations": [
            {
                "output_path": "metadata.derived_learning_rate",
                "query": ValueQuery(item="row", path="training_config.learning_rate"),
            }
        ],
        "rows": [
            {
                "row_id": "fast",
                "overrides": [
                    {"path": "training_config.learning_rate", "op": "replace", "value": 0.02}
                ],
            },
            {
                "row_id": "slow",
                "overrides": [
                    {"path": "training_config.learning_rate", "op": "replace", "value": 0.002}
                ],
            },
        ],
    }

    materialized = materialize_run_matrix(
        payload,
        repo_root=tmp_path,
        method_registry=application_registry_bundle.training_methods,
        row_lowerer=application_registry_bundle.row_lowerers.lower,
    )

    assert [row.payload["metadata"]["derived_learning_rate"] for row in materialized.rows] == [
        0.02,
        0.002,
    ]
    assert materialized.base_payload["metadata"]["derived_learning_rate"] is None

    payload["base"]["inline"]["metadata"]["derived_learning_rate"] = 0.01
    with pytest.raises(RunMatrixError, match="cannot change authored non-null field"):
        materialize_run_matrix(
            payload,
            repo_root=tmp_path,
            method_registry=application_registry_bundle.training_methods,
            row_lowerer=application_registry_bundle.row_lowerers.lower,
        )

    payload["base"]["inline"]["metadata"]["derived_learning_rate"] = None
    payload["derivations"][0]["query"] = {"item": "row", "path": "training_config.missing"}
    with pytest.raises(RunMatrixError, match="derivation failed"):
        materialize_run_matrix(
            payload,
            repo_root=tmp_path,
            method_registry=application_registry_bundle.training_methods,
            row_lowerer=application_registry_bundle.row_lowerers.lower,
        )


def test_v1_pretty_json_ref_verifies_legacy_raw_pin_then_materializes(
    tmp_path: Path, application_registry_bundle
) -> None:
    base_payload = _training_run_payload()
    pretty_bytes = (json.dumps(base_payload, indent=2, sort_keys=False) + "\n").encode()
    (tmp_path / "pretty-base.json").write_bytes(pretty_bytes)
    legacy = {
        "schema_id": TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
        "schema_version": "feedbax.spec.training_run_matrix.v1",
        "name": "legacy pretty ref",
        "base": {
            "ref": "pretty-base.json",
            "sha256": hashlib.sha256(pretty_bytes).hexdigest(),
        },
        "rows": [{"row_id": "row", "overrides": []}],
    }

    materialized = materialize_run_matrix(
        legacy,
        repo_root=tmp_path,
        method_registry=application_registry_bundle.training_methods,
        row_lowerer=application_registry_bundle.row_lowerers.lower,
    )

    assert materialized.rows[0].payload["training_config"]["n_batches"] == 2


def test_spec_lock_render_includes_legacy_lr_phrase(
    tmp_path: Path, application_registry_bundle
) -> None:
    matrix = _matrix(_training_run_payload())
    materialized = materialize_run_matrix(
        matrix,
        repo_root=tmp_path,
        method_registry=application_registry_bundle.training_methods,
        row_lowerer=application_registry_bundle.row_lowerers.lower,
    )

    rendered = render_spec_lock_table(
        matrix, materialized, method_registry=application_registry_bundle.training_methods
    )

    assert "LR continuation schedule: continue" in rendered
    assert "training_config.learning_rate" in rendered
    assert "lr_hi" in rendered


def test_spec_lock_renders_resolved_windows_for_every_active_schedule(
    tmp_path: Path, application_registry_bundle
) -> None:
    payload = _training_run_payload()
    payload["method_payload"]["payload"]["optimizer"]["lr_schedule"] = {
        "kind": "warmup_cosine",
        "learning_rate_0": 0.01,
        "total_steps": 1_000,
        "constant_lr_iterations": 500,
        "origin": {"kind": "segment_start"},
    }
    matrix = _matrix(payload)
    materialized = materialize_run_matrix(
        matrix,
        repo_root=tmp_path,
        method_registry=application_registry_bundle.training_methods,
        row_lowerer=application_registry_bundle.row_lowerers.lower,
    )
    lineages = {
        row.row_id: CheckpointSegmentLineage(
            start_batch=12_000,
            segment_batch_count=1_000,
            parent_transaction_id="parent",
        )
        for row in materialized.rows
    }

    rendered = render_spec_lock_table(
        matrix,
        materialized,
        segment_lineages=lineages,
        method_registry=application_registry_bundle.training_methods,
    )

    assert "lr_hi LR schedule: batches 12,000 -> 13,000" in rendered
    assert "lr_lo LR schedule: batches 12,000 -> 13,000" in rendered
