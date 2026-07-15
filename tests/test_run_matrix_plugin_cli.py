from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import feedbax.plugins
import feedbax.training.run_matrix as run_matrix
from feedbax import __main__ as feedbax_main
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


def test_matrix_harness_cli_preserves_serialized_v1_and_v2_payloads(
    monkeypatch, tmp_path: Path
) -> None:
    seen: list[dict] = []
    monkeypatch.setattr(
        feedbax.plugins,
        "load_training_method_plugins",
        lambda *, modules: None,
    )
    monkeypatch.setattr(
        "feedbax.analysis.evaluation.execute_evaluation_run_matrix",
        lambda payload, **_kwargs: seen.append(payload),
    )
    for version in ("feedbax.spec.evaluation_run_matrix.v1", "feedbax.spec.evaluation_run_matrix.v2"):
        payload = {"schema_version": version, "base": {}, "rows": []}
        path = tmp_path / f"{version.rsplit('.', 1)[-1]}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert harness.main([str(path), "--manifest-root", str(tmp_path / "runs")]) == 0

    assert seen == [
        {"schema_version": "feedbax.spec.evaluation_run_matrix.v1", "base": {}, "rows": []},
        {"schema_version": "feedbax.spec.evaluation_run_matrix.v2", "base": {}, "rows": []},
    ]
