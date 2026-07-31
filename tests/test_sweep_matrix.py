from __future__ import annotations

import json
from pathlib import Path

import pytest

from feedbax.contracts.manifest import (
    TRAINING_RUN_SET_SCHEMA_VERSION,
    TRAINING_RUN_SET_SCHEMA_VERSION_V1,
    TrainingRunSetManifest,
    TrainingRunSetAxes,
    TrainingSweepAxis,
    TrainingSweepAxisVariation,
    load_manifest,
    spec_payload,
    write_manifest,
)
from feedbax.contracts.migrations import default_spec_registry
from feedbax.studio.sweep_matrix import (
    SweepMatrixError,
    expand_sweep_matrix,
    materialize_sweep_matrix,
)


def _base_specs() -> tuple[dict, dict, dict]:
    return (
        {"nodes": {"network": {"type": "Gain", "params": {"gain": 1.0}}}},
        {
            "optimizer": {"type": "adam", "params": {"learning_rate": 0.001}},
            "loss": {"weight": 1.0},
            "n_batches": 25,
            "batch_size": 8,
        },
        {"type": "ReachingTask", "params": {"n_targets": 4, "duration": 80}},
    )


def test_sweep_matrix_cross_expands_cartesian_coordinates(application_registry_bundle) -> None:
    graph_spec, training_spec, task_spec = _base_specs()

    expanded = expand_sweep_matrix(
        {
            "axes": [
                {"id": "loss_weight", "path": "training_spec.loss.weight", "values": [0, 1e-5]},
                {"id": "seed", "path": "seed", "values": [1, 2, 3]},
            ],
            "mode": "cross",
        },
        graph_spec=graph_spec,
        training_spec=training_spec,
        task_spec=task_spec,
        task_binding_spec=None,
        default_name="Cross",
        method_registry=application_registry_bundle.training_methods,
        row_lowerer_registry=application_registry_bundle.row_lowerers,
    )

    assert len(expanded.runs) == 6
    assert expanded.runs[0].training_spec["loss"]["weight"] == 0
    assert expanded.runs[0].training_spec["seed"] == 1
    assert expanded.runs[-1].training_spec["loss"]["weight"] == 1e-5
    assert expanded.runs[-1].training_spec["seed"] == 3
    assert expanded.axes.runs[-1].values == {"loss_weight": 1e-5, "seed": 3}


def test_studio_sweep_adapter_returns_shared_materialized_run_set(
    application_registry_bundle,
) -> None:
    graph_spec, training_spec, task_spec = _base_specs()

    materialized = materialize_sweep_matrix(
        {
            "name": "Shared engine",
            "axes": [
                {"id": "loss_weight", "path": "training_spec.loss.weight", "values": [0, 1e-5]}
            ],
            "mode": "cross",
        },
        graph_spec=graph_spec,
        training_spec=training_spec,
        task_spec=task_spec,
        task_binding_spec=None,
        default_name="Fallback",
        method_registry=application_registry_bundle.training_methods,
        row_lowerer_registry=application_registry_bundle.row_lowerers,
    )

    assert materialized.run_set_manifest.id == materialized.run_set_id
    assert materialized.run_set_manifest.run_ids == [
        row.planned_run_id for row in materialized.rows
    ]
    assert materialized.run_set_manifest.metadata["matrix_schema_version"] == (
        "feedbax.spec.training_run_matrix.v1"
    )
    assert materialized.run_set_manifest.metadata["studio_legacy_adapter"] is True


def test_sweep_matrix_zip_rejects_mismatched_lengths(application_registry_bundle) -> None:
    graph_spec, training_spec, task_spec = _base_specs()

    with pytest.raises(SweepMatrixError, match="mismatched lengths"):
        expand_sweep_matrix(
            {
                "axes": [
                    {
                        "id": "loss_weight",
                        "path": "training_spec.loss.weight",
                        "values": [0, 1e-5],
                    },
                    {"id": "seed", "path": "seed", "values": [1, 2, 3]},
                ],
                "mode": "zip",
            },
            graph_spec=graph_spec,
            training_spec=training_spec,
            task_spec=task_spec,
            task_binding_spec=None,
            default_name="Zip",
            method_registry=application_registry_bundle.training_methods,
            row_lowerer_registry=application_registry_bundle.row_lowerers,
        )


def test_sweep_matrix_groups_zip_internally_and_cross_across_groups(
    application_registry_bundle,
) -> None:
    graph_spec, training_spec, task_spec = _base_specs()

    expanded = expand_sweep_matrix(
        {
            "axes": [
                {"id": "lr", "path": "training_spec.optimizer.params.learning_rate", "values": [1e-3, 1e-4]},
                {"id": "seed", "path": "seed", "values": [10, 20]},
                {"id": "weight", "path": "training_spec.loss.weight", "values": [0, 1e-5, 1e-4]},
            ],
            "combination": {
                "mode": "cross",
                "groups": [
                    {"id": "lr_seed", "mode": "zip", "axes": ["lr", "seed"]},
                    {"id": "weight", "mode": "cross", "axes": ["weight"]},
                ],
            },
        },
        graph_spec=graph_spec,
        training_spec=training_spec,
        task_spec=task_spec,
        task_binding_spec=None,
        default_name="Grouped",
        method_registry=application_registry_bundle.training_methods,
        row_lowerer_registry=application_registry_bundle.row_lowerers,
    )

    assert len(expanded.runs) == 6
    assert expanded.axes.runs[0].values == {"lr": 1e-3, "seed": 10, "weight": 0}
    assert expanded.axes.runs[3].values == {"lr": 1e-4, "seed": 20, "weight": 0}


def test_sweep_matrix_rejects_grouped_designs_that_omit_declared_axes(
    application_registry_bundle,
) -> None:
    graph_spec, training_spec, task_spec = _base_specs()

    with pytest.raises(SweepMatrixError, match="must cover every declared axis"):
        expand_sweep_matrix(
            {
                "axes": [
                    {
                        "id": "lr",
                        "path": "training_spec.optimizer.params.learning_rate",
                        "values": [1e-3, 1e-4],
                    },
                    {"id": "seed", "path": "seed", "values": [10, 20]},
                    {"id": "weight", "path": "training_spec.loss.weight", "values": [0, 1e-5]},
                ],
                "combination": {
                    "mode": "cross",
                    "groups": [
                        {"id": "lr_seed", "mode": "zip", "axes": ["lr", "seed"]},
                    ],
                },
            },
            graph_spec=graph_spec,
            training_spec=training_spec,
            task_spec=task_spec,
            task_binding_spec=None,
            default_name="Grouped",
            method_registry=application_registry_bundle.training_methods,
            row_lowerer_registry=application_registry_bundle.row_lowerers,
        )


def test_sweep_matrix_rejects_typo_axis_path_without_creating_fields(
    application_registry_bundle,
) -> None:
    graph_spec, training_spec, task_spec = _base_specs()

    with pytest.raises(SweepMatrixError, match="targets a missing key/index.*weigth"):
        expand_sweep_matrix(
            {
                "axes": [
                    {
                        "id": "loss_weight",
                        "path": "training_spec.loss.weigth",
                        "values": [0, 1e-5],
                    }
                ],
                "mode": "cross",
            },
            graph_spec=graph_spec,
            training_spec=training_spec,
            task_spec=task_spec,
            task_binding_spec=None,
            default_name="Typo",
            method_registry=application_registry_bundle.training_methods,
            row_lowerer_registry=application_registry_bundle.row_lowerers,
        )
    assert "weigth" not in training_spec["loss"]


def test_sweep_matrix_expands_ranges_and_seeded_sampler(application_registry_bundle) -> None:
    graph_spec, training_spec, task_spec = _base_specs()

    expanded = expand_sweep_matrix(
        {
            "axes": [
                {
                    "id": "duration",
                    "path": "task_spec.params.duration",
                    "variation": {"kind": "linspace", "min": 10, "max": 20, "n": 3},
                },
                {
                    "id": "lr",
                    "path": "training_spec.optimizer.params.learning_rate",
                    "variation": {"kind": "logspace", "min": 1e-4, "max": 1e-2, "n": 3},
                },
                {
                    "id": "seed",
                    "path": "seed",
                    "variation": {
                        "kind": "sampler",
                        "sampler": "uniform",
                        "n": 3,
                        "seed": 7,
                        "params": {"min": 0, "max": 1},
                    },
                },
            ],
            "mode": "zip",
        },
        graph_spec=graph_spec,
        training_spec=training_spec,
        task_spec=task_spec,
        task_binding_spec=None,
        default_name="Ranges",
        method_registry=application_registry_bundle.training_methods,
        row_lowerer_registry=application_registry_bundle.row_lowerers,
    )

    assert [run.task_spec["params"]["duration"] for run in expanded.runs] == [10, 15, 20]
    assert pytest.approx([run.training_spec["optimizer"]["params"]["learning_rate"] for run in expanded.runs]) == [
        1e-4,
        1e-3,
        1e-2,
    ]
    assert [run.training_spec["seed"] for run in expanded.runs] == [
        pytest.approx(0.32383276483316237),
        pytest.approx(0.15084917392450192),
        pytest.approx(0.6509344730398537),
    ]


def test_sweep_matrix_manual_coordinates_prune_expanded_design(application_registry_bundle) -> None:
    graph_spec, training_spec, task_spec = _base_specs()

    expanded = expand_sweep_matrix(
        {
            "axes": [
                {"id": "loss_weight", "path": "training_spec.loss.weight", "values": [0, 1e-5]},
                {"id": "seed", "path": "seed", "values": [1, 2, 3]},
            ],
            "combination": {
                "mode": "manual",
                "manual_coordinates": [
                    {"loss_weight": 0, "seed": 2},
                    {"loss_weight": 1, "seed": 0},
                ],
            },
        },
        graph_spec=graph_spec,
        training_spec=training_spec,
        task_spec=task_spec,
        task_binding_spec=None,
        default_name="Manual",
        method_registry=application_registry_bundle.training_methods,
        row_lowerer_registry=application_registry_bundle.row_lowerers,
    )

    assert len(expanded.runs) == 2
    assert expanded.axes.runs[0].values == {"loss_weight": 0, "seed": 3}
    assert expanded.axes.runs[1].values == {"loss_weight": 1e-5, "seed": 1}


def test_training_run_set_manifest_round_trips_axes(tmp_path: Path) -> None:
    manifest = TrainingRunSetManifest(
        id="feedbax-training-run-set:demo",
        name="Demo",
        run_ids=["feedbax-training-run:a"],
        graph_spec=spec_payload("GraphSpec", {"nodes": {}}),
        axes=TrainingRunSetAxes(
            axes=[
                TrainingSweepAxis(
                    id="loss_weight",
                    path="training_spec.loss.weight",
                    variation=TrainingSweepAxisVariation(kind="explicit", values=[0, 1e-5]),
                    values=[0, 1e-5],
                )
            ]
        ),
    )
    path = write_manifest(manifest, root=tmp_path)

    loaded = load_manifest(path)

    assert isinstance(loaded, TrainingRunSetManifest)
    assert loaded.schema_version == TRAINING_RUN_SET_SCHEMA_VERSION
    assert loaded.axes.axes[0].role == "authored_sweep"
    assert loaded.axes.axes[0].values == [0, 1e-5]


def test_training_run_set_manifest_migrates_legacy_v1_axes_block(tmp_path: Path) -> None:
    path = tmp_path / "legacy-run-set.json"
    path.write_text(
        json.dumps(
            {
                "kind": "TrainingRunSetManifest",
                "schema_version": TRAINING_RUN_SET_SCHEMA_VERSION_V1,
                "id": "feedbax-training-run-set:legacy",
                "name": "Legacy",
                "run_ids": ["feedbax-training-run:a"],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_manifest(path)

    assert isinstance(loaded, TrainingRunSetManifest)
    assert loaded.schema_version == TRAINING_RUN_SET_SCHEMA_VERSION
    assert loaded.axes.axes == []
    assert [record.migration_id for record in loaded.migration_records] == [
        "training-run-set-manifest-v1-to-v2-axes"
    ]


def test_training_run_set_manifest_registry_migrates_v1_axes_block() -> None:
    result = default_spec_registry.migrate(
        "TrainingRunSetManifest",
        {
            "kind": "TrainingRunSetManifest",
            "schema_version": TRAINING_RUN_SET_SCHEMA_VERSION_V1,
            "id": "feedbax-training-run-set:legacy",
            "name": "Legacy",
        },
    )

    assert result.target_version == TRAINING_RUN_SET_SCHEMA_VERSION
    assert result.payload["schema_version"] == TRAINING_RUN_SET_SCHEMA_VERSION
    assert result.payload["axes"] == {}
    assert [record.migration_id for record in result.migration_records] == [
        "training-run-set-manifest-v1-to-v2-axes"
    ]


def test_training_run_set_manifest_rejects_unknown_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "future-run-set.json"
    path.write_text(
        json.dumps(
            {
                "kind": "TrainingRunSetManifest",
                "schema_version": "feedbax.manifest.training_run_set.v99",
                "id": "feedbax-training-run-set:future",
                "name": "Future",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported TrainingRunSetManifest schema_version"):
        load_manifest(path)
