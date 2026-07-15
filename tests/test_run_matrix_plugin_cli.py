from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import feedbax.plugins
import feedbax.training.run_matrix as run_matrix
from feedbax import __main__ as feedbax_main
from feedbax.analysis import evaluation
from feedbax.analysis import harness
from feedbax.contracts.training import default_training_method_registry
from feedbax.plugins.discovery import load_training_method_plugins
from feedbax.training.preparation import ExecutionPreparationProviderRegistry


def test_plugin_analysis_recipe_hook_is_loaded_fail_closed() -> None:
    called: list[str] = []
    plugin = SimpleNamespace(register_feedbax_analysis_recipes=lambda: called.append("recipes"))

    load_training_method_plugins(
        registry=default_training_method_registry(),
        preparation_registry=ExecutionPreparationProviderRegistry(),
        entry_points=[SimpleNamespace(name="recipes", load=lambda: plugin)],
    )

    assert called == ["recipes"]


def test_materialize_cli_loads_repeated_plugins_before_spec(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[object] = []

    monkeypatch.setattr(
        feedbax.plugins,
        "load_training_method_plugins",
        lambda *, modules: events.append(("plugins", modules)),
    )
    monkeypatch.setattr(
        run_matrix,
        "_load_spec",
        lambda path: events.append(("spec", path)) or object(),
    )
    monkeypatch.setattr(
        run_matrix,
        "materialize_run_matrix",
        lambda spec, *, repo_root: events.append(("materialize", repo_root)) or object(),
    )
    monkeypatch.setattr(
        run_matrix,
        "write_materialized_matrix",
        lambda materialized, out_dir, *, wrap_key: events.append(("write", out_dir)),
    )

    result = run_matrix.main(
        [
            "materialize",
            str(tmp_path / "matrix.json"),
            "--repo-root",
            str(tmp_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--plugin",
            "downstream.first",
            "--plugin",
            "downstream.second",
        ]
    )

    assert result == 0
    assert events[:2] == [
        ("plugins", ["downstream.first", "downstream.second"]),
        ("spec", tmp_path / "matrix.json"),
    ]


def test_top_level_harness_cli_forwards_plugins_lazily(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(harness, "main", lambda argv: calls.append(argv) or 0)

    result = feedbax_main.main(
        [
            "matrix-harness",
            str(tmp_path / "matrix.json"),
            "--manifest-root",
            str(tmp_path / "manifests"),
            "--plugin",
            "downstream.recipes",
        ]
    )

    assert result == 0
    assert calls == [
        [
            str(tmp_path / "matrix.json"),
            "--manifest-root",
            str(tmp_path / "manifests"),
            "--plugin",
            "downstream.recipes",
        ]
    ]


def test_top_level_harness_cli_forwards_explicit_staged_runtime_bindings(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(harness, "main", lambda argv: calls.append(argv) or 0)
    spec = tmp_path / "matrix.json"
    manifest_root = tmp_path / "manifests"
    parent_root = tmp_path / "parents"
    descriptor = tmp_path / "descriptor.json"

    result = feedbax_main.main(
        [
            "matrix-harness",
            str(spec),
            "--manifest-root",
            str(manifest_root),
            "--parent-manifest-root",
            str(parent_root),
            "--execution-descriptor",
            str(descriptor),
            "--artifact-provider",
            f"shared={tmp_path / 'provider'}",
            "--checkpoint-custody",
            f"checkpoints={tmp_path / 'checkpoints'}",
        ]
    )

    assert result == 0
    assert calls == [
        [
            str(spec),
            "--manifest-root",
            str(manifest_root),
            "--parent-manifest-root",
            str(parent_root),
            "--execution-descriptor",
            str(descriptor),
            "--artifact-provider",
            f"shared={tmp_path / 'provider'}",
            "--checkpoint-custody",
            f"checkpoints={tmp_path / 'checkpoints'}",
        ]
    ]


def test_harness_cli_parses_staged_runtime_bindings(monkeypatch, tmp_path: Path) -> None:
    captured: list[tuple[dict[str, object], dict[str, object]]] = []
    monkeypatch.setattr(
        feedbax.plugins,
        "load_training_method_plugins",
        lambda *, modules: None,
    )
    monkeypatch.setattr(
        evaluation,
        "execute_evaluation_run_matrix",
        lambda payload, **kwargs: captured.append((payload, kwargs)),
    )
    spec = tmp_path / "matrix.json"
    descriptor = tmp_path / "descriptor.json"
    spec.write_text(json.dumps({"schema_id": "matrix"}), encoding="utf-8")
    descriptor.write_text(json.dumps({"schema_id": "descriptor"}), encoding="utf-8")

    result = harness.main(
        [
            str(spec),
            "--manifest-root",
            str(tmp_path / "rows"),
            "--parent-manifest-root",
            str(tmp_path / "parents"),
            "--execution-descriptor",
            str(descriptor),
            "--artifact-provider",
            f"shared={tmp_path / 'provider'}",
            "--checkpoint-custody",
            f"checkpoints={tmp_path / 'checkpoints'}",
        ]
    )

    assert result == 0
    payload, kwargs = captured[0]
    assert payload == {"schema_id": "matrix"}
    assert kwargs["execution_descriptor"] == {"schema_id": "descriptor"}
    assert kwargs["parent_manifest_root"] == str(tmp_path / "parents")
    assert kwargs["artifact_provider_bindings"][0].name == "shared"
    assert kwargs["artifact_provider_bindings"][0].root == str(tmp_path / "provider")
    assert kwargs["checkpoint_custody_bindings"][0].name == "checkpoints"
