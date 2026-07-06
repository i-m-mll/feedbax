from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from feedbax.analysis.bundles import (
    ANALYSIS_BUNDLE_SCHEMA_VERSION,
    AnalysisBundleSpec,
    BundleStageOutputSpec,
    BundleStageSpec,
    StageArtifactDependency,
    execute_analysis_bundle,
    execute_staged_analysis_bundle,
    ManifestPredicate,
    expand_analysis_bundle,
    load_analysis_bundle,
    predicate_matches_manifest,
    select_bundle_manifests,
)
from feedbax.analysis.context import AnalysisRunContext
from feedbax.analysis.evaluation import (
    EvaluationRecipeExecutionError as EvaluationRunExecutionError,
    EvaluationRecipeResult,
    EvaluationStatesCacheCorruption,
    execute_evaluation_run_spec,
    register_evaluation_recipe,
    unregister_evaluation_recipe,
)
from feedbax.analysis.reports import BUNDLE_SUMMARY_REPORT_TYPE, REPORT_RENDER_ROLE
from feedbax.analysis.specs import (
    AnalysisRecipeExecutionError,
    AnalysisRecipeResult,
    execute_analysis_run_spec,
    find_manifest_by_id,
    register_analysis_recipe,
    unregister_analysis_recipe,
)
from feedbax.analysis.materialization import ContextMaterializer
from feedbax.contracts.expressions import Compare
from feedbax.contracts.manifest import (
    AnalysisRunSpec,
    EvaluationRunSpec,
    ParentRef,
    TrainingRunManifest,
    analysis_run_manifest_id,
    evaluation_states_cache_path,
    load_manifest,
    write_manifest,
)
from feedbax.plugins.registry import ExperimentRegistry
from tests.analysis_fixtures import ToyAnalysis, ToyArtifactProducer, build_toy_analysis_data


TOY_ANALYSIS_TYPE = "feedbax.test.toy_analysis"
TOY_EVALUATION_TYPE = "feedbax.test.bundle_eval"
TOY_ARTIFACT_ANALYSIS_TYPE = "feedbax.test.bundle_artifact_analysis"
TOY_MATERIALIZER_TYPE = "feedbax.test.bundle_materializer"


def _register_toy_analysis_recipe() -> None:
    def recipe(spec, _root, inputs):
        value = sum(int(resolved.states["value"]) for resolved in inputs)
        return AnalysisRecipeResult(
            analyses={"toy": ToyAnalysis(variant="toy", cache_result=True)},
            data=build_toy_analysis_data(value=value),
            common_inputs={"presentation": spec.params.get("presentation", {})},
        )

    register_analysis_recipe(TOY_ANALYSIS_TYPE, recipe, replace=True)


def _register_toy_artifact_analysis_recipe() -> None:
    def recipe(spec, _root, inputs):
        values = [
            int(resolved.states["value"])
            for resolved in inputs
            if resolved.ref.kind == "EvaluationRunManifest"
        ]
        return AnalysisRecipeResult(
            analyses={"artifact_producer": ToyArtifactProducer(variant="toy", cache_result=True)},
            data=build_toy_analysis_data(value=sum(values)),
            common_inputs={"presentation": spec.params.get("presentation", {})},
        )

    register_analysis_recipe(TOY_ARTIFACT_ANALYSIS_TYPE, recipe, replace=True)


def _register_toy_materializer_recipe() -> None:
    def materialize(context: AnalysisRunContext) -> dict[str, object]:
        return {
            "kind": "toy.bundle-materialized.v1",
            "manifest_id": context.manifest_id,
            "value": 23,
        }

    def recipe(_spec, _root, _inputs):
        return AnalysisRecipeResult(
            analyses={
                "materializer": ContextMaterializer(
                    materializer=materialize,
                    artifact_role="toy_materialized_payload",
                    logical_name="toy/bundle-materialized.json",
                    schema_boundary="toy.bundle-materialized.v1",
                )
            },
            data=build_toy_analysis_data(value=0),
        )

    register_analysis_recipe(TOY_MATERIALIZER_TYPE, recipe, replace=True)


def _register_toy_evaluation_recipe() -> None:
    def recipe(run_spec: EvaluationRunSpec, _root: Path, _states_path: Path):
        return EvaluationRecipeResult(
            states={"value": np.asarray(run_spec.params["n_trials"], dtype=np.int32)},
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


def _write_toy_training(root: Path, *, method: str, run_id: str = "toy") -> TrainingRunManifest:
    manifest = TrainingRunManifest(
        id=f"feedbax-training-run:{run_id}",
        status="completed",
        metadata={"method": method},
    )
    write_manifest(manifest, root=root)
    return manifest


def _write_bundle_package(tmp_path: Path, monkeypatch) -> ExperimentRegistry:
    package_root = tmp_path / "toy_bundle_pkg"
    bundle_root = package_root / "config" / "analysis_bundles"
    bundle_root.mkdir(parents=True)
    (package_root / ".git").mkdir()
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
    (bundle_root / "missing_output.yml").write_text(
        f"""
name: toy_missing_output
predicate:
  manifest_kind: EvaluationRunManifest
  metadata_equals:
    method: minimax
templates:
  - name: mismatched_output
    mode: per-run
    analysis_type: {TOY_ANALYSIS_TYPE}
    requested_outputs: [missing]
""",
        encoding="utf-8",
    )
    (bundle_root / "routed.yml").write_text(
        f"""
name: toy_routed
description: Toy routed figure bundle
predicate:
  manifest_kind: EvaluationRunManifest
  metadata_equals:
    method: minimax
metadata:
  figure_routing:
    package: toy
    experiment: toy_experiment
    topic: toy_topic
    spec:
      transform:
        - name: toy-analysis
templates:
  - name: routed_cell
    mode: per-run
    analysis_type: {TOY_ANALYSIS_TYPE}
    requested_outputs: [toy]
""",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    for module_name in [
        name
        for name in sys.modules
        if name == "toy_bundle_pkg" or name.startswith("toy_bundle_pkg.")
    ]:
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    package = importlib.import_module("toy_bundle_pkg")
    registry = ExperimentRegistry()
    registry.register_package(
        "toy",
        package,
        parts=[],
        analysis_module_root="analysis",
        training_module_root="training",
        config_resource_root="config",
        figure_routing={
            "spec_dir_template": "results/{experiment}/figures/{topic}",
            "render_dir_template": "_artifacts/{experiment}/figures/{topic}",
            "render_format": "json",
            "create_symlink_in_spec_dir": True,
        },
    )
    return registry


def _register_empty_bundle_package(
    registry: ExperimentRegistry,
    tmp_path: Path,
    monkeypatch,
) -> None:
    package_root = tmp_path / "empty_bundle_pkg"
    bundle_root = package_root / "config" / "analysis_bundles"
    bundle_root.mkdir(parents=True)
    for path in (
        package_root / "__init__.py",
        package_root / "config" / "__init__.py",
        bundle_root / "__init__.py",
    ):
        path.write_text("", encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    package = importlib.import_module("empty_bundle_pkg")
    registry.register_package(
        "empty",
        package,
        parts=[],
        analysis_module_root="analysis",
        training_module_root="training",
        config_resource_root="config",
    )


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


def test_analysis_run_spec_reuses_completed_manifest_without_recipe_call(tmp_path: Path):
    _register_toy_evaluation_recipe()
    calls: list[int] = []

    def recipe(spec: AnalysisRunSpec, _root: Path, inputs):
        calls.append(int(inputs[0].states["value"]))
        return AnalysisRecipeResult(
            analyses={"toy": ToyAnalysis(variant="toy", cache_result=True)},
            data=build_toy_analysis_data(value=int(inputs[0].states["value"])),
        )

    register_analysis_recipe(TOY_ANALYSIS_TYPE, recipe, replace=True)
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
            params={"requested_outputs": ["toy"]},
        )

        manifest, path = execute_analysis_run_spec(
            spec,
            root=tmp_path,
            fig_dump_formats=("json",),
        )
        with patch("feedbax.analysis.specs.iter_manifest_files") as iter_files:
            iter_files.side_effect = AssertionError("filesystem fallback should not run")
            rerun_manifest, rerun_path = execute_analysis_run_spec(
                spec,
                root=tmp_path,
                fig_dump_formats=("json",),
            )
            indexed_manifest, indexed_path = find_manifest_by_id(
                analysis_run_manifest_id(spec),
                root=tmp_path,
            )

        assert calls == [2]
        assert rerun_manifest.id == manifest.id
        assert rerun_path == path
        assert indexed_manifest.id == manifest.id
        assert indexed_path == path
    finally:
        unregister_analysis_recipe(TOY_ANALYSIS_TYPE)
        unregister_evaluation_recipe(TOY_EVALUATION_TYPE)


def test_analysis_run_spec_rederives_missing_evaluation_states_cache(tmp_path: Path):
    _register_toy_evaluation_recipe()
    _register_toy_analysis_recipe()
    try:
        eval_manifest, eval_path = _execute_toy_eval(tmp_path, n_trials=5, method="minimax")
        states_path = evaluation_states_cache_path(eval_manifest.id, root=tmp_path)
        assert states_path.exists()
        states_path.unlink()

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
            params={"requested_outputs": ["toy"]},
        )

        manifest, path = execute_analysis_run_spec(
            spec,
            root=tmp_path,
            issues=["ad32279"],
            fig_dump_formats=("json",),
        )

        assert path.exists()
        assert states_path.exists()
        assert manifest.status == "completed"
        assert manifest.summary_metrics["analysis_count"] == 1
        assert manifest.summary_metrics["figure_count"] == 1
        assert load_manifest(eval_path).summary_metrics["n_trials"] == 5
    finally:
        unregister_analysis_recipe(TOY_ANALYSIS_TYPE)
        unregister_evaluation_recipe(TOY_EVALUATION_TYPE)


def test_evaluation_states_cache_corruption_fails_closed(tmp_path: Path):
    _register_toy_evaluation_recipe()
    try:
        parent = ParentRef(
            kind="TrainingRunManifest",
            id="feedbax-training-run:corrupt-states-cache",
            role="training_run",
        )
        spec = EvaluationRunSpec(
            evaluation_type=TOY_EVALUATION_TYPE,
            inputs=[parent],
            params={"n_trials": 4},
        )
        manifest, _path = execute_evaluation_run_spec(spec, root=tmp_path)
        states_path = evaluation_states_cache_path(manifest.id, root=tmp_path)
        states_path.write_bytes(b"not a pickle payload")

        with pytest.raises(EvaluationRunExecutionError) as excinfo:
            execute_evaluation_run_spec(spec, root=tmp_path)

        assert isinstance(excinfo.value.__cause__, EvaluationStatesCacheCorruption)
        failed = load_manifest(excinfo.value.path)
        assert failed.status == "failed"
        assert "EvaluationStatesCacheCorruption" in failed.metadata["error"]["type"]
    finally:
        unregister_evaluation_recipe(TOY_EVALUATION_TYPE)


def test_analysis_run_spec_prefers_durable_states_on_cache_miss(tmp_path: Path):
    calls: list[int] = []

    def recipe(run_spec: EvaluationRunSpec, _root: Path, _states_path: Path):
        calls.append(int(run_spec.params["n_trials"]))
        return EvaluationRecipeResult(
            states={"value": np.asarray(run_spec.params["n_trials"], dtype=np.int32)},
            summary_metrics={"n_trials": run_spec.params["n_trials"]},
        )

    register_evaluation_recipe(TOY_EVALUATION_TYPE, recipe, replace=True)
    _register_toy_analysis_recipe()
    try:
        parent = ParentRef(
            kind="TrainingRunManifest",
            id="feedbax-training-run:durable-cache-miss",
            role="training_run",
        )
        eval_spec = EvaluationRunSpec(
            evaluation_type=TOY_EVALUATION_TYPE,
            inputs=[parent],
            params={"n_trials": 6, "states_custody": "durable"},
        )
        eval_manifest, eval_path = execute_evaluation_run_spec(
            eval_spec,
            root=tmp_path,
            force=True,
        )
        states_path = evaluation_states_cache_path(eval_manifest.id, root=tmp_path)
        assert states_path.exists()
        states_path.unlink()

        analysis_spec = AnalysisRunSpec(
            analysis_type=TOY_ANALYSIS_TYPE,
            inputs=[
                ParentRef(
                    kind="EvaluationRunManifest",
                    id=eval_manifest.id,
                    role="evaluation_run",
                    uri=str(eval_path),
                )
            ],
            params={"requested_outputs": ["toy"]},
        )
        manifest, _path = execute_analysis_run_spec(
            analysis_spec,
            root=tmp_path,
            fig_dump_formats=("json",),
        )

        assert manifest.status == "completed"
        assert calls == [6]
        assert states_path.exists()
    finally:
        unregister_analysis_recipe(TOY_ANALYSIS_TYPE)
        unregister_evaluation_recipe(TOY_EVALUATION_TYPE)


def test_analysis_run_spec_records_failed_manifest_for_unknown_requested_output(
    tmp_path: Path,
) -> None:
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
            params={"requested_outputs": ["missing"]},
        )

        with pytest.raises(AnalysisRecipeExecutionError) as excinfo:
            execute_analysis_run_spec(
                spec,
                root=tmp_path,
                fig_dump_formats=("json",),
            )

        assert "requested_outputs=['missing']" in str(excinfo.value.__cause__)
        assert "available_analysis_keys=['toy']" in str(excinfo.value.__cause__)
        failed_manifest = load_manifest(excinfo.value.path)
        assert failed_manifest.kind == "AnalysisRunManifest"
        assert failed_manifest.status == "failed"
        assert failed_manifest.metadata["error"]["type"] == "ValueError"
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
        assert bundle.schema_version == ANALYSIS_BUNDLE_SCHEMA_VERSION
        matched = select_bundle_manifests(bundle, tmp_path)
        matched_ids = [manifest.id for manifest in matched]
        assert set(matched_ids) == {first.id, second.id}
        assert [
            manifest.id
            for manifest in select_bundle_manifests(
                bundle,
                tmp_path,
                run_ids=[first.id],
            )
        ] == [first.id]

        params_bundle = load_analysis_bundle("toy/params_match", registry=registry)
        assert [
            manifest.id
            for manifest in select_bundle_manifests(
                params_bundle,
                tmp_path,
            )
        ] == [first.id]
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


def test_simple_bundle_rejects_explicit_unsupported_old_schema_version() -> None:
    with pytest.raises(ValueError, match="unsupported AnalysisBundleSpec schema_version"):
        AnalysisBundleSpec.model_validate(
            {
                "schema_version": "feedbax.spec.analysis_bundle.v1",
                "name": "old_bundle",
                "templates": [
                    {
                        "name": "legacy",
                        "analysis_type": TOY_ANALYSIS_TYPE,
                    }
                ],
            }
        )


def test_staged_bundle_executes_eval_two_analyses_and_report_with_lineage(
    tmp_path: Path,
) -> None:
    _register_toy_evaluation_recipe()
    _register_toy_analysis_recipe()
    try:
        training = _write_toy_training(tmp_path, method="minimax")
        bundle = AnalysisBundleSpec(
            name="toy_staged",
            predicate=ManifestPredicate(
                manifest_kind="TrainingRunManifest",
                metadata_equals={"method": "minimax"},
            ),
            stages=[
                BundleStageSpec(
                    name="eval",
                    kind="evaluation",
                    evaluation_type=TOY_EVALUATION_TYPE,
                    params={"n_trials": 3},
                    outputs=[BundleStageOutputSpec(role="manifest")],
                ),
                BundleStageSpec(
                    name="summary",
                    kind="analysis",
                    depends_on=["eval"],
                    analysis_type=TOY_ANALYSIS_TYPE,
                    requested_outputs=["toy"],
                    params={"variant": "summary"},
                    outputs=[BundleStageOutputSpec(role="manifest")],
                ),
                BundleStageSpec(
                    name="detail",
                    kind="analysis",
                    depends_on=["eval"],
                    analysis_type=TOY_ANALYSIS_TYPE,
                    requested_outputs=["toy"],
                    params={"variant": "detail"},
                    outputs=[BundleStageOutputSpec(role="manifest")],
                ),
                BundleStageSpec(
                    name="report",
                    kind="report",
                    depends_on=["summary", "detail"],
                    report_type=BUNDLE_SUMMARY_REPORT_TYPE,
                    outputs=[BundleStageOutputSpec(role="report")],
                ),
            ],
        )

        result = execute_staged_analysis_bundle(
            bundle,
            root=tmp_path,
            issues=["a6af537"],
            fig_dump_formats=("json",),
        )

        assert result.matched_run_ids == [training.id]
        assert [stage.name for stage in result.stages] == [
            "eval",
            "summary",
            "detail",
            "report",
        ]
        eval_ref = result.stages[0].manifest_refs[0]
        assert eval_ref.kind == "EvaluationRunManifest"
        assert result.stages[1].inputs == [eval_ref]
        assert result.stages[2].inputs == [eval_ref]
        report_inputs = result.stages[3].inputs
        assert [ref.kind for ref in report_inputs] == [
            "AnalysisRunManifest",
            "AnalysisRunManifest",
        ]
        assert report_inputs == [
            result.stages[1].manifest_refs[0],
            result.stages[2].manifest_refs[0],
        ]
        assert result.stages[1].regeneration_specs[0].kind == "RegenerationSpec"
        assert result.stages[3].regeneration_specs[0].kind == "RegenerationSpec"
        assert result.report_outputs[0].status == "materialized"
        report_manifest = load_manifest(result.stages[3].manifest_refs[0].uri)
        assert report_manifest.kind == "ReportManifest"
        assert report_manifest.regeneration_specs[0].kind == "RegenerationSpec"
        report_artifacts = {artifact.role: artifact for artifact in report_manifest.artifacts}
        assert set(report_artifacts) == {"report", REPORT_RENDER_ROLE}
        assert report_artifacts[REPORT_RENDER_ROLE].media_type == "text/markdown"
        assert report_artifacts[REPORT_RENDER_ROLE].sha256 is not None
        assert Path(report_artifacts[REPORT_RENDER_ROLE].uri or "").read_text(
            encoding="utf-8"
        ).startswith("# toy_staged / report")
    finally:
        unregister_analysis_recipe(TOY_ANALYSIS_TYPE)
        unregister_evaluation_recipe(TOY_EVALUATION_TYPE)


def test_staged_bundle_grouped_analysis_can_compose_bundle_and_dependency_inputs(
    tmp_path: Path,
) -> None:
    paired_analysis_type = "feedbax.test.paired_bundle_analysis"
    observed_inputs: list[list[tuple[str, str]]] = []

    def recipe(spec: AnalysisRunSpec, _root: Path, inputs):
        observed_inputs.append([(item.ref.kind, item.ref.id) for item in inputs])
        eval_values = [
            int(item.states["value"])
            for item in inputs
            if item.ref.kind == "EvaluationRunManifest"
        ]
        return AnalysisRecipeResult(
            analyses={"toy": ToyAnalysis(variant="toy", cache_result=True)},
            data=build_toy_analysis_data(value=sum(eval_values)),
            common_inputs={"presentation": spec.params.get("presentation", {})},
        )

    _register_toy_evaluation_recipe()
    register_analysis_recipe(paired_analysis_type, recipe, replace=True)
    try:
        training = _write_toy_training(tmp_path, method="minimax")
        bundle = AnalysisBundleSpec(
            name="toy_paired_eval",
            predicate=ManifestPredicate(
                manifest_kind="TrainingRunManifest",
                metadata_equals={"method": "minimax"},
            ),
            stages=[
                BundleStageSpec(
                    name="short_eval",
                    kind="evaluation",
                    evaluation_type=TOY_EVALUATION_TYPE,
                    params={"n_trials": 3},
                ),
                BundleStageSpec(
                    name="long_eval",
                    kind="evaluation",
                    evaluation_type=TOY_EVALUATION_TYPE,
                    params={"n_trials": 5},
                ),
                BundleStageSpec(
                    name="comparison",
                    kind="analysis",
                    depends_on=["short_eval", "long_eval"],
                    include_bundle_inputs=True,
                    analysis_type=paired_analysis_type,
                    requested_outputs=["toy"],
                    params={"variant": "paired"},
                    outputs=[BundleStageOutputSpec(role="manifest")],
                ),
            ],
        )

        result = execute_staged_analysis_bundle(
            bundle,
            root=tmp_path,
            fig_dump_formats=("json",),
        )

        short_eval_ref = result.stages[0].manifest_refs[0]
        long_eval_ref = result.stages[1].manifest_refs[0]
        comparison_stage = result.stages[2]

        assert comparison_stage.inputs == [
            ParentRef(kind="TrainingRunManifest", id=training.id, role="training_run"),
            short_eval_ref,
            long_eval_ref,
        ]
        assert observed_inputs == [
            [
                ("TrainingRunManifest", training.id),
                ("EvaluationRunManifest", short_eval_ref.id),
                ("EvaluationRunManifest", long_eval_ref.id),
            ]
        ]
        analysis_manifest = load_manifest(comparison_stage.manifest_refs[0].uri)
        assert analysis_manifest.inputs == comparison_stage.inputs
        assert analysis_manifest.provenance.parents == comparison_stage.inputs
    finally:
        unregister_analysis_recipe(paired_analysis_type)
        unregister_evaluation_recipe(TOY_EVALUATION_TYPE)


def test_staged_bundle_rerun_reuses_eval_and_analysis_manifests(tmp_path: Path) -> None:
    eval_calls: list[int] = []
    analysis_calls: list[int] = []

    def eval_recipe(run_spec: EvaluationRunSpec, _root: Path, _states_path: Path):
        n_trials = int(run_spec.params["n_trials"])
        eval_calls.append(n_trials)
        return EvaluationRecipeResult(
            states={"value": np.asarray(n_trials, dtype=np.int32)},
            summary_metrics={"n_trials": n_trials},
        )

    def analysis_recipe(_spec: AnalysisRunSpec, _root: Path, inputs):
        value = int(inputs[0].states["value"])
        analysis_calls.append(value)
        return AnalysisRecipeResult(
            analyses={"toy": ToyAnalysis(variant="toy", cache_result=True)},
            data=build_toy_analysis_data(value=value),
        )

    register_evaluation_recipe(TOY_EVALUATION_TYPE, eval_recipe, replace=True)
    register_analysis_recipe(TOY_ANALYSIS_TYPE, analysis_recipe, replace=True)
    try:
        _write_toy_training(tmp_path, method="minimax")
        bundle = AnalysisBundleSpec(
            name="toy_cache_reuse",
            predicate=ManifestPredicate(
                manifest_kind="TrainingRunManifest",
                metadata_equals={"method": "minimax"},
            ),
            stages=[
                BundleStageSpec(
                    name="eval",
                    kind="evaluation",
                    evaluation_type=TOY_EVALUATION_TYPE,
                    params={"n_trials": 7},
                ),
                BundleStageSpec(
                    name="summary",
                    kind="analysis",
                    depends_on=["eval"],
                    analysis_type=TOY_ANALYSIS_TYPE,
                    requested_outputs=["toy"],
                    outputs=[BundleStageOutputSpec(role="manifest")],
                ),
            ],
        )

        first = execute_staged_analysis_bundle(
            bundle,
            root=tmp_path,
            fig_dump_formats=("json",),
        )
        second = execute_staged_analysis_bundle(
            bundle,
            root=tmp_path,
            fig_dump_formats=("json",),
        )

        assert eval_calls == [7]
        assert analysis_calls == [7]
        assert second.stages[0].manifest_refs == first.stages[0].manifest_refs
        assert second.stages[1].manifest_refs == first.stages[1].manifest_refs
    finally:
        unregister_analysis_recipe(TOY_ANALYSIS_TYPE)
        unregister_evaluation_recipe(TOY_EVALUATION_TYPE)


def test_staged_bundle_evaluation_stage_can_request_durable_states(
    tmp_path: Path,
) -> None:
    _register_toy_evaluation_recipe()
    try:
        training = _write_toy_training(tmp_path, method="minimax")
        bundle = AnalysisBundleSpec(
            name="toy_durable_eval_stage",
            predicate=ManifestPredicate(
                manifest_kind="TrainingRunManifest",
                metadata_equals={"method": "minimax"},
            ),
            stages=[
                BundleStageSpec(
                    name="eval",
                    kind="evaluation",
                    evaluation_type=TOY_EVALUATION_TYPE,
                    params={"n_trials": 3},
                    states_custody="durable",
                    outputs=[
                        BundleStageOutputSpec(role="manifest"),
                        BundleStageOutputSpec(role="evaluation_states"),
                    ],
                ),
            ],
        )

        result = execute_staged_analysis_bundle(bundle, root=tmp_path)

        assert result.matched_run_ids == [training.id]
        eval_stage = result.stages[0]
        assert eval_stage.outputs[0].status == "materialized"
        assert eval_stage.outputs[1].status == "materialized"
        assert eval_stage.artifact_groups["evaluation_states"][0].role == (
            "evaluation_states"
        )
        eval_manifest = load_manifest(eval_stage.manifest_refs[0].uri)
        assert eval_manifest.evaluation_spec.inline["params"]["states_custody"] == "durable"
        assert eval_manifest.artifacts[0].role == "evaluation_states"
    finally:
        unregister_evaluation_recipe(TOY_EVALUATION_TYPE)


def test_staged_bundle_records_optional_output_statuses(tmp_path: Path) -> None:
    _register_toy_evaluation_recipe()
    _register_toy_analysis_recipe()
    try:
        _write_toy_training(tmp_path, method="minimax")
        bundle = AnalysisBundleSpec(
            name="toy_optional_statuses",
            predicate=ManifestPredicate(
                manifest_kind="TrainingRunManifest",
                metadata_equals={"method": "minimax"},
            ),
            stages=[
                BundleStageSpec(
                    name="eval",
                    kind="evaluation",
                    evaluation_type=TOY_EVALUATION_TYPE,
                    params={"n_trials": 1},
                ),
                BundleStageSpec(
                    name="analysis",
                    kind="analysis",
                    depends_on=["eval"],
                    analysis_type=TOY_ANALYSIS_TYPE,
                    requested_outputs=["toy"],
                    outputs=[
                        BundleStageOutputSpec(role="manifest"),
                        BundleStageOutputSpec(role="sidecar", required=False),
                    ],
                ),
                BundleStageSpec(
                    name="optional_sidecar",
                    kind="analysis",
                    depends_on=["analysis"],
                    analysis_type=TOY_ANALYSIS_TYPE,
                    outputs=[BundleStageOutputSpec(role="sidecar", required=False)],
                    skip_reason="sidecar disabled for this bundle",
                ),
                BundleStageSpec(
                    name="materializer",
                    kind="materialization",
                    depends_on=["analysis"],
                    outputs=[BundleStageOutputSpec(role="materialized_sidecar", required=False)],
                    not_applicable_reason="context-bound materializer is owned by sibling issue",
                ),
            ],
        )

        result = execute_staged_analysis_bundle(
            bundle,
            root=tmp_path,
            fig_dump_formats=("json",),
        )

        analysis_outputs = {output.role: output for output in result.stages[1].outputs}
        assert analysis_outputs["manifest"].status == "materialized"
        assert analysis_outputs["sidecar"].status == "missing"
        assert "optional output role" in analysis_outputs["sidecar"].reason
        assert result.stages[2].outputs[0].status == "skipped"
        assert result.stages[2].outputs[0].reason == "sidecar disabled for this bundle"
        assert result.stages[3].outputs[0].status == "not_applicable"
        assert "sibling issue" in result.stages[3].outputs[0].reason
    finally:
        unregister_analysis_recipe(TOY_ANALYSIS_TYPE)
        unregister_evaluation_recipe(TOY_EVALUATION_TYPE)


def test_staged_bundle_runtime_condition_skips_required_outputs_and_optional_role_omits(
    tmp_path: Path,
) -> None:
    observed_inputs: list[list[ParentRef]] = []
    consumer_type = "feedbax.test.bundle_optional_role_consumer"

    def consumer_recipe(_spec: AnalysisRunSpec, _root: Path, inputs):
        observed_inputs.append([resolved.ref for resolved in inputs])
        return AnalysisRecipeResult(
            analyses={"toy": ToyAnalysis(variant="toy", cache_result=True)},
            data=build_toy_analysis_data(value=0),
        )

    register_analysis_recipe(consumer_type, consumer_recipe, replace=True)
    try:
        _write_toy_training(tmp_path, method="minimax")
        condition = Compare(
            item="params",
            path="enabled",
            op="eq",
            value=True,
        )
        bundle = AnalysisBundleSpec(
            name="toy_condition_skip",
            predicate=ManifestPredicate(
                manifest_kind="TrainingRunManifest",
                metadata_equals={"method": "minimax"},
            ),
            stages=[
                BundleStageSpec(
                    name="gated",
                    kind="analysis",
                    analysis_type=TOY_ARTIFACT_ANALYSIS_TYPE,
                    params={"enabled": False},
                    run_condition=condition,
                    outputs=[
                        BundleStageOutputSpec(role="manifest"),
                        BundleStageOutputSpec(role="analysis_summary"),
                    ],
                ),
                BundleStageSpec(
                    name="downstream",
                    kind="analysis",
                    analysis_type=consumer_type,
                    depends_on_roles=[
                        StageArtifactDependency(
                            stage="gated",
                            role="analysis_summary",
                            required=False,
                        )
                    ],
                    outputs=[BundleStageOutputSpec(role="manifest")],
                ),
            ],
        )

        result = execute_staged_analysis_bundle(
            bundle,
            root=tmp_path,
            fig_dump_formats=("json",),
        )

        gated_outputs = result.stages[0].outputs
        assert [output.status for output in gated_outputs] == ["skipped", "skipped"]
        assert all(output.required for output in gated_outputs)
        assert "run_condition evaluated false" in gated_outputs[0].reason
        assert '"enabled"' in gated_outputs[0].reason
        assert observed_inputs == [[]]
        assert result.stages[1].outputs[0].status == "materialized"
    finally:
        unregister_analysis_recipe(consumer_type)
        unregister_analysis_recipe(TOY_ARTIFACT_ANALYSIS_TYPE)


def test_staged_bundle_required_role_dependency_fails_closed(tmp_path: Path) -> None:
    _register_toy_evaluation_recipe()
    _register_toy_artifact_analysis_recipe()
    try:
        _write_toy_training(tmp_path, method="minimax")
        bundle = AnalysisBundleSpec(
            name="toy_missing_required_role",
            predicate=ManifestPredicate(
                manifest_kind="TrainingRunManifest",
                metadata_equals={"method": "minimax"},
            ),
            stages=[
                BundleStageSpec(
                    name="eval",
                    kind="evaluation",
                    evaluation_type=TOY_EVALUATION_TYPE,
                    params={"n_trials": 2},
                ),
                BundleStageSpec(
                    name="producer",
                    kind="analysis",
                    depends_on=["eval"],
                    analysis_type=TOY_ARTIFACT_ANALYSIS_TYPE,
                    outputs=[BundleStageOutputSpec(role="analysis_summary")],
                ),
                BundleStageSpec(
                    name="consumer",
                    kind="analysis",
                    analysis_type=TOY_ANALYSIS_TYPE,
                    depends_on_roles=[
                        StageArtifactDependency(
                            stage="producer",
                            role="does_not_exist",
                        )
                    ],
                    outputs=[BundleStageOutputSpec(role="manifest")],
                ),
            ],
        )

        with pytest.raises(ValueError) as excinfo:
            execute_staged_analysis_bundle(
                bundle,
                root=tmp_path,
                fig_dump_formats=("json",),
            )

        message = str(excinfo.value)
        assert "consumer" in message
        assert "producer" in message
        assert "does_not_exist" in message
    finally:
        unregister_analysis_recipe(TOY_ARTIFACT_ANALYSIS_TYPE)
        unregister_analysis_recipe(TOY_ANALYSIS_TYPE)
        unregister_evaluation_recipe(TOY_EVALUATION_TYPE)


def test_staged_bundle_role_dependency_binds_artifact_input_alias(tmp_path: Path) -> None:
    observed_inputs: list[list[ParentRef]] = []
    consumer_type = "feedbax.test.bundle_role_consumer"

    def consumer_recipe(_spec: AnalysisRunSpec, _root: Path, inputs):
        observed_inputs.append([resolved.ref for resolved in inputs])
        return AnalysisRecipeResult(
            analyses={"toy": ToyAnalysis(variant="toy", cache_result=True)},
            data=build_toy_analysis_data(value=0),
        )

    _register_toy_evaluation_recipe()
    _register_toy_artifact_analysis_recipe()
    register_analysis_recipe(consumer_type, consumer_recipe, replace=True)
    try:
        _write_toy_training(tmp_path, method="minimax")
        bundle = AnalysisBundleSpec(
            name="toy_role_dependency",
            predicate=ManifestPredicate(
                manifest_kind="TrainingRunManifest",
                metadata_equals={"method": "minimax"},
            ),
            stages=[
                BundleStageSpec(
                    name="eval",
                    kind="evaluation",
                    evaluation_type=TOY_EVALUATION_TYPE,
                    params={"n_trials": 4},
                ),
                BundleStageSpec(
                    name="producer",
                    kind="analysis",
                    depends_on=["eval"],
                    analysis_type=TOY_ARTIFACT_ANALYSIS_TYPE,
                    outputs=[BundleStageOutputSpec(role="analysis_summary")],
                ),
                BundleStageSpec(
                    name="consumer",
                    kind="analysis",
                    analysis_type=consumer_type,
                    depends_on_roles=[
                        StageArtifactDependency(
                            stage="producer",
                            role="analysis_summary",
                            bind_as="summary_input",
                        )
                    ],
                    run_condition=Compare(
                        item="summary_input",
                        op="has_type",
                        value="artifact_role",
                    ),
                    outputs=[BundleStageOutputSpec(role="manifest")],
                ),
            ],
        )

        result = execute_staged_analysis_bundle(
            bundle,
            root=tmp_path,
            fig_dump_formats=("json",),
        )

        assert result.stages[2].outputs[0].status == "materialized"
        assert len(observed_inputs) == 1
        assert len(observed_inputs[0]) == 1
        artifact_input = observed_inputs[0][0]
        assert artifact_input.kind == "ArtifactRef"
        assert artifact_input.role == "summary_input"
        assert artifact_input.metadata["source_role"] == "analysis_summary"
    finally:
        unregister_analysis_recipe(consumer_type)
        unregister_analysis_recipe(TOY_ARTIFACT_ANALYSIS_TYPE)
        unregister_evaluation_recipe(TOY_EVALUATION_TYPE)


def test_staged_bundle_materialization_stage_emits_artifact_and_regeneration(
    tmp_path: Path,
) -> None:
    _register_toy_materializer_recipe()
    try:
        _write_toy_training(tmp_path, method="minimax")
        bundle = AnalysisBundleSpec(
            name="toy_materialization",
            predicate=ManifestPredicate(
                manifest_kind="TrainingRunManifest",
                metadata_equals={"method": "minimax"},
            ),
            stages=[
                BundleStageSpec(
                    name="materialize",
                    kind="materialization",
                    analysis_type=TOY_MATERIALIZER_TYPE,
                    outputs=[BundleStageOutputSpec(role="toy_materialized_payload")],
                ),
            ],
        )

        result = execute_staged_analysis_bundle(bundle, root=tmp_path)

        stage = result.stages[0]
        assert stage.status == "materialized"
        assert stage.outputs[0].status == "materialized"
        payload_ref = stage.outputs[0].artifacts[0]
        assert payload_ref.role == "toy_materialized_payload"
        assert json.loads(Path(payload_ref.uri).read_text(encoding="utf-8"))["value"] == 23
        assert stage.regeneration_specs[0].kind == "RegenerationSpec"
        manifest = load_manifest(stage.manifest_refs[0].uri)
        assert manifest.artifacts[0].role == "toy_materialized_payload"
        assert manifest.regeneration_specs[-1].kind == "RegenerationSpec"
        assert manifest.regeneration_specs[-1].inline["parameters"]["stage"]["kind"] == (
            "materialization"
        )
    finally:
        unregister_analysis_recipe(TOY_MATERIALIZER_TYPE)


def test_staged_bundle_materialization_stage_rejects_non_materializer_node(
    tmp_path: Path,
) -> None:
    bad_materializer_type = "feedbax.test.bundle_bad_materializer"

    def recipe(_spec: AnalysisRunSpec, _root: Path, _inputs):
        return AnalysisRecipeResult(
            analyses={"toy": ToyAnalysis(variant="toy", cache_result=True)},
            data=build_toy_analysis_data(value=0),
        )

    register_analysis_recipe(bad_materializer_type, recipe, replace=True)
    try:
        _write_toy_training(tmp_path, method="minimax")
        bundle = AnalysisBundleSpec(
            name="toy_bad_materialization",
            predicate=ManifestPredicate(
                manifest_kind="TrainingRunManifest",
                metadata_equals={"method": "minimax"},
            ),
            stages=[
                BundleStageSpec(
                    name="bad_materialize",
                    kind="materialization",
                    analysis_type=bad_materializer_type,
                    outputs=[BundleStageOutputSpec(role="manifest")],
                ),
            ],
        )

        with pytest.raises(AnalysisRecipeExecutionError) as excinfo:
            execute_staged_analysis_bundle(
                bundle,
                root=tmp_path,
                fig_dump_formats=("json",),
            )

        assert "bad_materialize" in str(excinfo.value.__cause__)
        assert "materialization stages require" in str(excinfo.value.__cause__)
    finally:
        unregister_analysis_recipe(bad_materializer_type)


def test_staged_bundle_existing_execution_record_omits_new_fields(tmp_path: Path) -> None:
    _register_toy_evaluation_recipe()
    _register_toy_analysis_recipe()
    try:
        _write_toy_training(tmp_path, method="minimax")
        bundle = AnalysisBundleSpec(
            name="toy_unchanged_record",
            predicate=ManifestPredicate(
                manifest_kind="TrainingRunManifest",
                metadata_equals={"method": "minimax"},
            ),
            stages=[
                BundleStageSpec(
                    name="eval",
                    kind="evaluation",
                    evaluation_type=TOY_EVALUATION_TYPE,
                    params={"n_trials": 3},
                    outputs=[BundleStageOutputSpec(role="manifest")],
                ),
                BundleStageSpec(
                    name="summary",
                    kind="analysis",
                    depends_on=["eval"],
                    analysis_type=TOY_ANALYSIS_TYPE,
                    requested_outputs=["toy"],
                    outputs=[BundleStageOutputSpec(role="manifest")],
                ),
            ],
        )

        result = execute_staged_analysis_bundle(
            bundle,
            root=tmp_path,
            fig_dump_formats=("json",),
        )
        payload = result.model_dump(mode="json", exclude_none=True)

        assert "depends_on_roles" not in json.dumps(payload, sort_keys=True)
        assert "input_artifacts" not in json.dumps(payload, sort_keys=True)
        stage_payload = payload["stages"][1]["regeneration_specs"][0]["inline"][
            "parameters"
        ]["stage"]
        assert "depends_on_roles" not in stage_payload
        assert "run_condition" not in stage_payload
    finally:
        unregister_analysis_recipe(TOY_ANALYSIS_TYPE)
        unregister_evaluation_recipe(TOY_EVALUATION_TYPE)


def test_unqualified_bundle_lookup_uses_public_registry_metadata(tmp_path: Path, monkeypatch):
    registry = _write_bundle_package(tmp_path, monkeypatch)
    _register_empty_bundle_package(registry, tmp_path, monkeypatch)

    with patch.object(
        registry,
        "iter_package_metadata",
        wraps=registry.iter_package_metadata,
    ) as iter_metadata:
        bundle = load_analysis_bundle("matrix", registry=registry)

    assert iter_metadata.called
    assert bundle.name == "toy_matrix"


def test_analysis_bundle_fails_on_unknown_requested_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _register_toy_evaluation_recipe()
    _register_toy_analysis_recipe()
    try:
        _execute_toy_eval(tmp_path, n_trials=2, method="minimax")
        registry = _write_bundle_package(tmp_path, monkeypatch)
        bundle = load_analysis_bundle("toy/missing_output", registry=registry)

        with pytest.raises(AnalysisRecipeExecutionError) as excinfo:
            execute_analysis_bundle(
                bundle,
                root=tmp_path,
                fig_dump_formats=("json",),
            )

        assert "requested_outputs=['missing']" in str(excinfo.value.__cause__)
        assert "available_analysis_keys=['toy']" in str(excinfo.value.__cause__)
    finally:
        unregister_analysis_recipe(TOY_ANALYSIS_TYPE)
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
            item["matched_run_ids"][0] for item in output if item["template"] == "per_cell"
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


def test_analysis_cli_keeps_bundle_progress_off_json_stdout(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from feedbax.bin import analysis as analysis_cli

    def fake_execute_analysis_bundle(*_args, **_kwargs):
        analysis_cli.rich.get_console().print("Computing analysis nodes...")
        return [
            (
                SimpleNamespace(
                    bundle_name="toy_matrix",
                    template_name="per_cell",
                    mode="per-run",
                    matched_run_ids=("feedbax-evaluation-run:toy",),
                ),
                SimpleNamespace(id="feedbax-analysis-run:toy"),
                tmp_path / "manifests" / "analysis_runs" / "toy.json",
            )
        ]

    monkeypatch.setattr(analysis_cli, "load_analysis_bundle", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(analysis_cli, "execute_analysis_bundle", fake_execute_analysis_bundle)

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
        ]
    )

    captured = capsys.readouterr()
    assert "Computing analysis nodes" not in captured.out
    assert "Computing analysis nodes" in captured.err
    assert json.loads(captured.out) == [
        {
            "bundle": "toy_matrix",
            "template": "per_cell",
            "mode": "per-run",
            "matched_run_ids": ["feedbax-evaluation-run:toy"],
            "manifest_id": "feedbax-analysis-run:toy",
            "manifest_path": str(tmp_path / "manifests" / "analysis_runs" / "toy.json"),
        }
    ]


def test_bundle_context_projects_figures_through_registered_routing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _register_toy_evaluation_recipe()
    _register_toy_analysis_recipe()
    try:
        _eval_manifest, _eval_path = _execute_toy_eval(
            tmp_path,
            n_trials=2,
            method="minimax",
        )
        registry = _write_bundle_package(tmp_path, monkeypatch)
        bundle = load_analysis_bundle("toy/routed", registry=registry)

        import feedbax.plugins as plugins

        monkeypatch.setattr(plugins, "EXPERIMENT_REGISTRY", registry)
        outputs = execute_analysis_bundle(
            bundle,
            root=tmp_path,
            issues=["3fb7e70"],
            fig_dump_formats=("json",),
        )

        assert len(outputs) == 1
        _expansion, manifest, _path = outputs[0]
        assert manifest.summary_metrics["figure_count"] == 1
        assert manifest.summary_metrics["artifact_count"] == 1
        canonical = manifest.artifacts[0]
        assert canonical.role == "figure"
        assert Path(canonical.uri).exists()
        assert canonical.metadata["relative_path"].startswith("artifacts/")

        projection = canonical.metadata["figure_routing"]
        spec_path = Path(projection["spec_path"])
        render_path = Path(projection["render_path"])
        symlink_path = Path(projection["symlink_path"])
        assert spec_path == (
            tmp_path / "toy_bundle_pkg" / "results" / "toy_experiment" / "figures"
            / "toy_topic" / "spec.json"
        )
        assert render_path == (
            tmp_path / "toy_bundle_pkg" / "_artifacts" / "toy_experiment" / "figures"
            / "toy_topic" / "figure.fig.json"
        )
        assert spec_path.exists()
        assert render_path.exists()
        assert symlink_path.is_symlink()
        assert symlink_path.resolve() == render_path

        routed_spec = json.loads(spec_path.read_text(encoding="utf-8"))
        assert routed_spec["analysis"]["manifest_id"] == manifest.id
        assert routed_spec["analysis"]["analysis_type"] == TOY_ANALYSIS_TYPE
        assert routed_spec["analysis"]["analysis_name"] == "toy_analysis"
        assert routed_spec["plot_kwargs"]["params"]["result_value"] == 3
        assert routed_spec["transform"] == [{"name": "toy-analysis"}]
        assert manifest.metadata["bundle"]["metadata"]["figure_routing"]["package"] == "toy"
    finally:
        unregister_analysis_recipe(TOY_ANALYSIS_TYPE)
        unregister_evaluation_recipe(TOY_EVALUATION_TYPE)
