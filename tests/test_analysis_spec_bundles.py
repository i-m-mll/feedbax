from __future__ import annotations

import importlib
import json
from pathlib import Path

from feedbax.analysis.bundles import (
    ManifestPredicate,
    expand_analysis_bundle,
    load_analysis_bundle,
    predicate_matches_manifest,
    select_bundle_manifests,
)
from feedbax.analysis.evaluation import (
    EvaluationRecipeResult,
    execute_evaluation_run_spec,
    register_evaluation_recipe,
    unregister_evaluation_recipe,
)
from feedbax.analysis.specs import (
    AnalysisRecipeResult,
    execute_analysis_run_spec,
    register_analysis_recipe,
    unregister_analysis_recipe,
)
from feedbax.manifest import AnalysisRunSpec, EvaluationRunSpec, ParentRef, load_manifest
from feedbax.plugins.registry import ExperimentRegistry
from tests.analysis_fixtures import ToyAnalysis, build_toy_analysis_data


TOY_ANALYSIS_TYPE = "feedbax_test_toy_analysis"
TOY_EVALUATION_TYPE = "feedbax_test_bundle_eval"


def _register_toy_analysis_recipe() -> None:
    def recipe(spec, _root, inputs):
        value = sum(int(resolved.states["value"]) for resolved in inputs)
        return AnalysisRecipeResult(
            analyses={"toy": ToyAnalysis(variant="toy", cache_result=True)},
            data=build_toy_analysis_data(value=value),
            common_inputs={"presentation": spec.params.get("presentation", {})},
        )

    register_analysis_recipe(TOY_ANALYSIS_TYPE, recipe, replace=True)


def _register_toy_evaluation_recipe() -> None:
    def recipe(run_spec: EvaluationRunSpec, _root: Path, _states_path: Path):
        return EvaluationRecipeResult(
            states={"value": run_spec.params["n_trials"]},
            summary_metrics={"n_trials": run_spec.params["n_trials"]},
        )

    register_evaluation_recipe(TOY_EVALUATION_TYPE, recipe, replace=True)


def _execute_toy_eval(root: Path, *, n_trials: int, method: str):
    parent = ParentRef(
        kind="TrainingRunManifest",
        id=f"feedbax-training-run:{method}-{n_trials}",
        role="training_run",
    )
    spec = EvaluationRunSpec(
        evaluation_type=TOY_EVALUATION_TYPE,
        inputs=[parent],
        params={"n_trials": n_trials},
    )
    return execute_evaluation_run_spec(
        spec,
        root=root,
        metadata={"method": method, "cell": f"{method}-{n_trials}"},
        force=True,
    )


def _write_bundle_package(tmp_path: Path, monkeypatch) -> ExperimentRegistry:
    package_root = tmp_path / "toy_bundle_pkg"
    bundle_root = package_root / "config" / "analysis_bundles"
    bundle_root.mkdir(parents=True)
    for path in (
        package_root / "__init__.py",
        package_root / "config" / "__init__.py",
        bundle_root / "__init__.py",
    ):
        path.write_text("", encoding="utf-8")
    (bundle_root / "matrix.yml").write_text(
        f"""
name: toy_matrix
description: Toy matrix bundle
predicate:
  manifest_kind: EvaluationRunManifest
  metadata_equals:
    method: minimax
templates:
  - name: per_cell
    mode: per-run
    analysis_type: {TOY_ANALYSIS_TYPE}
    requested_outputs: [toy]
    params:
      presentation:
        color: red
  - name: grouped_cells
    mode: grouped
    analysis_type: {TOY_ANALYSIS_TYPE}
    requested_outputs: [toy]
""",
        encoding="utf-8",
    )
    (bundle_root / "params_match.yml").write_text(
        f"""
name: toy_params_match
predicate:
  manifest_kind: EvaluationRunManifest
  metadata_equals:
    method: minimax
  params_equals:
    n_trials: 2
templates:
  - name: only_matching_params
    mode: per-run
    analysis_type: {TOY_ANALYSIS_TYPE}
    requested_outputs: [toy]
""",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    package = importlib.import_module("toy_bundle_pkg")
    registry = ExperimentRegistry()
    registry.register_package(
        "toy",
        package,
        parts=[],
        analysis_module_root="analysis",
        training_module_root="training",
        config_resource_root="config",
    )
    return registry


def test_analysis_run_spec_executes_registered_recipe_and_records_manifest(tmp_path: Path):
    _register_toy_evaluation_recipe()
    _register_toy_analysis_recipe()
    try:
        eval_manifest, eval_path = _execute_toy_eval(tmp_path, n_trials=2, method="minimax")
        spec = AnalysisRunSpec(
            analysis_type=TOY_ANALYSIS_TYPE,
            inputs=[
                ParentRef(
                    kind="EvaluationRunManifest",
                    id=eval_manifest.id,
                    role="evaluation_run",
                    uri=str(eval_path),
                )
            ],
            params={
                "requested_outputs": ["toy"],
                "presentation": {"display_name": "Toy"},
            },
        )

        manifest, path = execute_analysis_run_spec(
            spec,
            root=tmp_path,
            issues=["81c7149"],
            fig_dump_formats=("json",),
        )

        assert path.exists()
        assert manifest.kind == "AnalysisRunManifest"
        assert manifest.status == "completed"
        assert manifest.inputs[0].id == eval_manifest.id
        assert manifest.analysis_spec.inline["analysis_type"] == TOY_ANALYSIS_TYPE
        assert manifest.analysis_spec.inline["params"]["requested_outputs"] == ["toy"]
        assert manifest.provenance.parents == spec.inputs
        assert manifest.provenance.issues == ["81c7149"]
        assert manifest.summary_metrics["analysis_count"] == 1
        assert manifest.summary_metrics["figure_count"] == 1
        assert load_manifest(path).id == manifest.id
    finally:
        unregister_analysis_recipe(TOY_ANALYSIS_TYPE)
        unregister_evaluation_recipe(TOY_EVALUATION_TYPE)


def test_bundle_loading_predicates_and_per_run_grouped_expansion(tmp_path: Path, monkeypatch):
    _register_toy_evaluation_recipe()
    try:
        first, _first_path = _execute_toy_eval(tmp_path, n_trials=2, method="minimax")
        second, _second_path = _execute_toy_eval(tmp_path, n_trials=4, method="minimax")
        other, _other_path = _execute_toy_eval(tmp_path, n_trials=2, method="baseline")
        registry = _write_bundle_package(tmp_path, monkeypatch)

        bundle = load_analysis_bundle("toy/matrix", registry=registry)
        matched = select_bundle_manifests(bundle, tmp_path)
        matched_ids = [manifest.id for manifest in matched]
        assert set(matched_ids) == {first.id, second.id}
        assert [manifest.id for manifest in select_bundle_manifests(
            bundle,
            tmp_path,
            run_ids=[first.id],
        )] == [first.id]

        params_bundle = load_analysis_bundle("toy/params_match", registry=registry)
        assert [manifest.id for manifest in select_bundle_manifests(
            params_bundle,
            tmp_path,
        )] == [first.id]
        assert predicate_matches_manifest(
            ManifestPredicate(run_ids=[other.id]),
            other,
        )

        expansions = expand_analysis_bundle(bundle, matched)
        assert [(item.template_name, item.matched_run_ids) for item in expansions] == [
            ("per_cell", (matched_ids[0],)),
            ("per_cell", (matched_ids[1],)),
            ("grouped_cells", tuple(matched_ids)),
        ]
        assert expansions[0].spec.inputs[0].id == matched_ids[0]
        assert [ref.id for ref in expansions[2].spec.inputs] == matched_ids
        assert expansions[0].spec.params["requested_outputs"] == ["toy"]
    finally:
        unregister_evaluation_recipe(TOY_EVALUATION_TYPE)


def test_analysis_cli_runs_bundle_against_manifest_root(tmp_path: Path, monkeypatch, capsys):
    _register_toy_evaluation_recipe()
    _register_toy_analysis_recipe()
    try:
        first, _first_path = _execute_toy_eval(tmp_path, n_trials=2, method="minimax")
        second, _second_path = _execute_toy_eval(tmp_path, n_trials=4, method="minimax")
        registry = _write_bundle_package(tmp_path, monkeypatch)

        from feedbax.bin import analysis as analysis_cli

        monkeypatch.setattr(analysis_cli, "EXPERIMENT_REGISTRY", registry)
        analysis_cli.main(
            [
                "--bundle",
                "toy/matrix",
                "--manifest-root",
                str(tmp_path),
                "--fig-dump-dir",
                str(tmp_path / "figures"),
                "--fig-dump-formats",
                "json",
                "--issue",
                "81c7149",
            ]
        )

        output = json.loads(capsys.readouterr().out)
        expected_ids = [
            item["matched_run_ids"][0]
            for item in output
            if item["template"] == "per_cell"
        ]
        assert set(expected_ids) == {first.id, second.id}
        assert [(item["template"], item["matched_run_ids"]) for item in output] == [
            ("per_cell", [expected_ids[0]]),
            ("per_cell", [expected_ids[1]]),
            ("grouped_cells", expected_ids),
        ]
        for item in output:
            manifest = load_manifest(item["manifest_path"])
            assert manifest.kind == "AnalysisRunManifest"
            assert manifest.status == "completed"
            assert manifest.metadata["bundle"]["name"] == "toy_matrix"
            assert manifest.provenance.issues == ["81c7149"]
    finally:
        unregister_analysis_recipe(TOY_ANALYSIS_TYPE)
        unregister_evaluation_recipe(TOY_EVALUATION_TYPE)
