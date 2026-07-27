from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import feedbax.plugins
import feedbax.training.run_matrix as run_matrix
from feedbax import __main__ as feedbax_main
from feedbax.analysis import evaluation
from feedbax.analysis import harness
from feedbax.analysis.evaluation import EvaluationRecipeResult
from feedbax.contracts.manifest import canonical_json_bytes, load_manifest, sha256_bytes
from feedbax.contracts.training import default_training_method_registry
from feedbax.plugins.discovery import load_training_method_plugins
from feedbax.training.preparation import ExecutionPreparationProviderRegistry


def _evaluation_matrix_payload(*row_ids: str) -> dict[str, object]:
    return {
        "schema_id": "feedbax.spec.evaluation_run_matrix",
        "schema_version": "feedbax.spec.evaluation_run_matrix.v3",
        "base": {"evaluation_type": "example.cli_gate"},
        "rows": [{"row_id": row_id} for row_id in row_ids],
    }


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


def test_top_level_harness_cli_forwards_batch_and_gate_options(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(harness, "main", lambda argv: calls.append(argv) or 0)
    spec = tmp_path / "matrix.json"
    locked_spec = tmp_path / "locked.json"
    policy = tmp_path / "policy.json"

    result = feedbax_main.main(
        [
            "matrix-harness",
            str(spec),
            "--manifest-root",
            str(tmp_path / "manifests"),
            "--batch",
            "--locked-spec",
            str(locked_spec),
            "--execution-policy",
            str(policy),
            "--allow-large-per-row",
        ]
    )

    assert result == 0
    assert calls == [
        [
            str(spec),
            "--manifest-root",
            str(tmp_path / "manifests"),
            "--batch",
            "--locked-spec",
            str(locked_spec),
            "--execution-policy",
            str(policy),
            "--allow-large-per-row",
        ]
    ]


def test_matrix_harness_gate_options_have_public_help(capsys) -> None:
    for entrypoint in (feedbax_main.main, harness.main):
        with pytest.raises(SystemExit, match="0"):
            entrypoint(
                ["matrix-harness", "--help"] if entrypoint is feedbax_main.main else ["--help"]
            )
        help_text = " ".join(capsys.readouterr().out.split())
        assert "exact ordered row_id sequence" in help_text
        assert "positive-integer per_row_max_rows" in help_text
        assert "registered batch recipe" in help_text
        assert "decision is recorded" in help_text


def test_top_level_matrix_harness_executes_v3_axis_matrix(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        feedbax.plugins,
        "load_training_method_plugins",
        lambda *, modules: None,
    )
    base = {"evaluation_type": "example.cli_axis", "params": {"gain": 0}}
    (tmp_path / "base.json").write_text(json.dumps(base), encoding="utf-8")
    payload = {
        "schema_id": "feedbax.spec.evaluation_run_matrix",
        "schema_version": "feedbax.spec.evaluation_run_matrix.v3",
        "base": {"ref": "base.json", "sha256": sha256_bytes(canonical_json_bytes(base))},
        "axes": [
            {
                "id": "gain",
                "values": [
                    {"id": "one", "deltas": [{"path": "params.gain", "value": 1}]}
                ],
            }
        ],
    }
    spec = tmp_path / "matrix.json"
    spec.write_text(json.dumps(payload), encoding="utf-8")
    evaluation.register_evaluation_recipe(
        "example.cli_axis", lambda *_args: EvaluationRecipeResult()
    )
    try:
        result = feedbax_main.main(
            [
                "matrix-harness",
                str(spec),
                "--manifest-root",
                str(tmp_path / "runs"),
                "--repo-root",
                str(tmp_path),
            ]
        )
    finally:
        evaluation.unregister_evaluation_recipe("example.cli_axis")

    assert result == 0
    manifest_path = next(
        (tmp_path / "runs" / "gain-one" / "manifests" / "evaluation_runs").glob("*.json")
    )
    manifest = load_manifest(manifest_path)
    assert manifest.metadata["matrix_harness"]["axis_expansion"]["pinned_base"]["ref"] == (
        "base.json"
    )


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
            "--repo-root",
            str(tmp_path),
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
            "--repo-root",
            str(tmp_path),
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
            "--repo-root",
            str(tmp_path),
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
    assert kwargs["repo_root"] == str(tmp_path)
    assert kwargs["execution_descriptor"] == {"schema_id": "descriptor"}
    assert kwargs["parent_manifest_root"] == str(tmp_path / "parents")
    assert kwargs["artifact_provider_bindings"][0].name == "shared"
    assert kwargs["artifact_provider_bindings"][0].root == str(tmp_path / "provider")
    assert kwargs["checkpoint_custody_bindings"][0].name == "checkpoints"


def test_harness_cli_passes_batch_execution(monkeypatch, tmp_path: Path) -> None:
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        feedbax.plugins,
        "load_training_method_plugins",
        lambda *, modules: None,
    )
    monkeypatch.setattr(
        evaluation,
        "execute_evaluation_run_matrix",
        lambda _payload, **kwargs: captured.append(kwargs),
    )
    spec = tmp_path / "matrix.json"
    spec.write_text(json.dumps(_evaluation_matrix_payload("one")), encoding="utf-8")

    result = harness.main(
        [str(spec), "--manifest-root", str(tmp_path / "rows"), "--batch"]
    )

    assert result == 0
    assert isinstance(captured[0]["batch"], evaluation.EvaluationBatchExecution)


def test_harness_cli_locked_spec_requires_exact_row_id_sequence(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    executions: list[dict[str, object]] = []
    monkeypatch.setattr(
        feedbax.plugins,
        "load_training_method_plugins",
        lambda *, modules: None,
    )
    monkeypatch.setattr(
        evaluation,
        "execute_evaluation_run_matrix",
        lambda payload, **_kwargs: executions.append(payload),
    )
    spec = tmp_path / "matrix.json"
    locked = tmp_path / "locked.json"
    spec.write_text(json.dumps(_evaluation_matrix_payload("one", "two")), encoding="utf-8")
    locked.write_text(
        json.dumps({"evaluation_matrix": _evaluation_matrix_payload("one", "two")}),
        encoding="utf-8",
    )

    result = harness.main(
        [
            str(spec),
            "--manifest-root",
            str(tmp_path / "matching"),
            "--locked-spec",
            str(locked),
        ]
    )
    assert result == 0
    assert len(executions) == 1

    locked.write_text(
        json.dumps(_evaluation_matrix_payload("two", "one")),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="2"):
        harness.main(
            [
                str(spec),
                "--manifest-root",
                str(tmp_path / "mismatch"),
                "--locked-spec",
                str(locked),
            ]
        )

    assert len(executions) == 1
    assert "row_id sequence does not match locked spec" in capsys.readouterr().err


def test_harness_cli_policy_blocks_large_per_row_execution(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    executions: list[dict[str, object]] = []
    monkeypatch.setattr(
        feedbax.plugins,
        "load_training_method_plugins",
        lambda *, modules: None,
    )
    monkeypatch.setattr(
        evaluation,
        "execute_evaluation_run_matrix",
        lambda payload, **_kwargs: executions.append(payload),
    )
    spec = tmp_path / "matrix.json"
    policy = tmp_path / "policy.json"
    spec.write_text(json.dumps(_evaluation_matrix_payload("one", "two")), encoding="utf-8")
    policy.write_text(
        json.dumps(
            {
                "schema_id": "caller.execution-policy",
                "per_row_max_rows": 1,
                "threshold_source": "authored memory limit",
                "override_flag": "--allow-large-per-row",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="2"):
        feedbax_main.main(
            [
                "matrix-harness",
                str(spec),
                "--manifest-root",
                str(tmp_path / "rows"),
                "--execution-policy",
                str(policy),
            ]
        )

    assert executions == []
    error = capsys.readouterr().err
    assert "2 rows" in error
    assert "1 rows" in error
    assert "authored memory limit" in error
    assert "--allow-large-per-row" in error


def test_harness_cli_rejects_unsupported_authored_override_before_outputs(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    monkeypatch.setattr(
        feedbax.plugins,
        "load_training_method_plugins",
        lambda *, modules: None,
    )
    monkeypatch.setattr(
        evaluation,
        "execute_evaluation_run_matrix",
        lambda *_args, **_kwargs: pytest.fail("invalid override surface must not execute"),
    )
    spec = tmp_path / "matrix.json"
    policy = tmp_path / "policy.json"
    manifest_root = tmp_path / "rows"
    spec.write_text(json.dumps(_evaluation_matrix_payload("one", "two")), encoding="utf-8")
    policy.write_text(
        json.dumps(
            {
                "per_row_max_rows": 1,
                "threshold_source": "authored memory limit",
                "override_flag": "--policy-forbids-cli-override",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="2"):
        feedbax_main.main(
            [
                "matrix-harness",
                str(spec),
                "--manifest-root",
                str(manifest_root),
                "--execution-policy",
                str(policy),
                "--allow-large-per-row",
            ]
        )

    assert not manifest_root.exists()
    error = capsys.readouterr().err
    assert "--policy-forbids-cli-override" in error
    assert "--allow-large-per-row" in error


@pytest.mark.parametrize("bypass", [["--batch"], ["--allow-large-per-row"]])
def test_harness_cli_policy_allows_batch_or_explicit_override(
    monkeypatch,
    tmp_path: Path,
    bypass: list[str],
) -> None:
    executions: list[dict[str, object]] = []
    monkeypatch.setattr(
        feedbax.plugins,
        "load_training_method_plugins",
        lambda *, modules: None,
    )
    monkeypatch.setattr(
        evaluation,
        "execute_evaluation_run_matrix",
        lambda payload, **kwargs: executions.append(kwargs),
    )
    spec = tmp_path / "matrix.json"
    policy = tmp_path / "policy.json"
    spec.write_text(json.dumps(_evaluation_matrix_payload("one", "two")), encoding="utf-8")
    policy.write_text(
        json.dumps(
            {
                "per_row_max_rows": 1,
                "threshold_source": "authored memory limit",
                "override_flag": "--allow-large-per-row",
            }
        ),
        encoding="utf-8",
    )

    result = harness.main(
        [
            str(spec),
            "--manifest-root",
            str(tmp_path / "rows"),
            "--execution-policy",
            str(policy),
            *bypass,
        ]
    )

    assert result == 0
    assert len(executions) == 1
    assert (executions[0]["batch"] is not None) is (bypass == ["--batch"])


@pytest.mark.parametrize(
    ("bypass", "execution_mode", "explicit_override_used"),
    [
        (["--batch"], "batch", False),
        (["--allow-large-per-row"], "per-row", True),
    ],
)
def test_harness_cli_persists_execution_policy_evidence(
    tmp_path: Path,
    bypass: list[str],
    execution_mode: str,
    explicit_override_used: bool,
) -> None:
    spec = tmp_path / f"matrix-{execution_mode}.json"
    policy = tmp_path / f"policy-{execution_mode}.json"
    manifest_root = tmp_path / f"rows-{execution_mode}"
    matrix_payload = _evaluation_matrix_payload("one", "two")
    matrix_payload["base"] = {
        "evaluation_type": "example.cli_gate",
        "params": {"fixture_identity": ""},
    }
    matrix_payload["rows"] = [
        {
            "row_id": row_id,
            "deltas": [{"path": "params.fixture_identity", "value": row_id}],
        }
        for row_id in ("one", "two")
    ]
    spec.write_text(json.dumps(matrix_payload), encoding="utf-8")
    policy_content = json.dumps(
        {
            "per_row_max_rows": 1,
            "threshold_source": "authored memory limit",
            "override_flag": "--allow-large-per-row",
        }
    ).encode()
    policy.write_bytes(policy_content)
    evaluation.register_evaluation_recipe(
        "example.cli_gate",
        lambda *_args: EvaluationRecipeResult(),
        batch_recipe=lambda items, _context: [EvaluationRecipeResult() for _item in items],
        replace=True,
    )
    try:
        result = harness.main(
            [
                str(spec),
                "--manifest-root",
                str(manifest_root),
                "--execution-policy",
                str(policy),
                *bypass,
            ]
        )
    finally:
        evaluation.unregister_evaluation_recipe("example.cli_gate")

    assert result == 0
    store_root = manifest_root if execution_mode == "batch" else manifest_root / "one"
    manifest_path = next((store_root / "manifests" / "evaluation_runs").glob("*.json"))
    manifest = load_manifest(manifest_path)
    expected = {
        "content_sha256": sha256_bytes(policy_content),
        "per_row_max_rows": 1,
        "threshold_source": "authored memory limit",
        "override_flag": "--allow-large-per-row",
        "execution_mode": execution_mode,
        "explicit_override_used": explicit_override_used,
    }
    harness_metadata = manifest.metadata["matrix_harness"]
    assert harness_metadata["execution_policy"] == expected
    assert harness_metadata["regeneration_spec"]["metadata"]["execution_policy"] == expected


def test_harness_cli_without_policy_omits_policy_evidence(tmp_path: Path) -> None:
    spec = tmp_path / "matrix.json"
    manifest_root = tmp_path / "rows"
    spec.write_text(json.dumps(_evaluation_matrix_payload("one")), encoding="utf-8")
    evaluation.register_evaluation_recipe(
        "example.cli_gate",
        lambda *_args: EvaluationRecipeResult(),
        replace=True,
    )
    try:
        result = harness.main([str(spec), "--manifest-root", str(manifest_root)])
    finally:
        evaluation.unregister_evaluation_recipe("example.cli_gate")

    assert result == 0
    manifest_path = next((manifest_root / "one" / "manifests" / "evaluation_runs").glob("*.json"))
    harness_metadata = load_manifest(manifest_path).metadata["matrix_harness"]
    assert "execution_policy" not in harness_metadata
    assert "execution_policy" not in harness_metadata["regeneration_spec"]["metadata"]


@pytest.mark.parametrize(
    "policy_payload",
    [
        {"per_row_max_rows": True, "threshold_source": "source", "override_flag": "--flag"},
        {"per_row_max_rows": 0, "threshold_source": "source", "override_flag": "--flag"},
        {"per_row_max_rows": 1, "threshold_source": " ", "override_flag": "--flag"},
        {"per_row_max_rows": 1, "threshold_source": "source", "override_flag": ""},
    ],
)
def test_harness_cli_rejects_malformed_execution_policy(
    monkeypatch,
    tmp_path: Path,
    policy_payload: dict[str, object],
) -> None:
    monkeypatch.setattr(
        feedbax.plugins,
        "load_training_method_plugins",
        lambda *, modules: None,
    )
    monkeypatch.setattr(
        evaluation,
        "execute_evaluation_run_matrix",
        lambda *_args, **_kwargs: pytest.fail("malformed policy must abort before execution"),
    )
    spec = tmp_path / "matrix.json"
    policy = tmp_path / "policy.json"
    spec.write_text(json.dumps(_evaluation_matrix_payload("one")), encoding="utf-8")
    policy.write_text(json.dumps(policy_payload), encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        harness.main(
            [
                str(spec),
                "--manifest-root",
                str(tmp_path / "rows"),
                "--execution-policy",
                str(policy),
            ]
        )
