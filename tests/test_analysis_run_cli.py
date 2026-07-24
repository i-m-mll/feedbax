"""CLI exposure of `execute_analysis_run_spec` through `feedbax-analysis run`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from feedbax.analysis.specs import (
    AnalysisRecipeResult,
    register_analysis_recipe,
    unregister_analysis_recipe,
)
from feedbax.bin import analysis as analysis_cli
from feedbax.config.yaml import get_yaml_loader
from feedbax.contracts.manifest import AnalysisRunSpec, ParentRef, load_manifest
from tests.analysis_fixtures import (
    ToyAnalysis,
    build_toy_analysis_data,
    execute_toy_evaluation,
)

pytestmark = [pytest.mark.feedbax_contract]

CLI_ANALYSIS_TYPE = "feedbax.test.run_cli_analysis"


@pytest.fixture
def toy_analysis_recipe():
    def recipe(spec: AnalysisRunSpec, _root: Path, _inputs, _execution_context):
        return AnalysisRecipeResult(
            analyses={"toy": ToyAnalysis(variant="toy", cache_result=True)},
            data=build_toy_analysis_data(value=int(spec.params["value"])),
        )

    register_analysis_recipe(CLI_ANALYSIS_TYPE, recipe, replace=True)
    try:
        yield
    finally:
        unregister_analysis_recipe(CLI_ANALYSIS_TYPE)


def _toy_spec_payload(root: Path) -> dict:
    eval_manifest, eval_path = execute_toy_evaluation(root)
    spec = AnalysisRunSpec(
        analysis_type=CLI_ANALYSIS_TYPE,
        inputs=[
            ParentRef(
                kind="EvaluationRunManifest",
                id=eval_manifest.id,
                role="evaluation_run",
                uri=str(eval_path),
            )
        ],
        params={"requested_outputs": ["toy"], "value": 3},
    )
    return spec.model_dump(mode="json", exclude_none=True)


def test_run_subcommand_executes_json_spec_file(tmp_path: Path, capsys, toy_analysis_recipe):
    payload = _toy_spec_payload(tmp_path)
    spec_path = tmp_path / "analysis_spec.json"
    spec_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    analysis_cli.main(["run", str(spec_path), "--root", str(tmp_path)])

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed"
    manifest_path = Path(result["manifest_path"])
    assert manifest_path.exists()
    assert manifest_path.is_relative_to(tmp_path)
    manifest = load_manifest(manifest_path)
    assert manifest.kind == "AnalysisRunManifest"
    assert manifest.id == result["manifest_id"]
    assert manifest.analysis_spec.inline["analysis_type"] == CLI_ANALYSIS_TYPE
    assert result["artifacts"] == [
        artifact.model_dump(mode="json", exclude_none=True) for artifact in manifest.artifacts
    ]


def test_run_subcommand_executes_yaml_spec_file(tmp_path: Path, capsys, toy_analysis_recipe):
    payload = _toy_spec_payload(tmp_path)
    spec_path = tmp_path / "analysis_spec.yaml"
    with spec_path.open("w", encoding="utf-8") as handle:
        get_yaml_loader().dump(payload, handle)

    analysis_cli.main(["run", str(spec_path), "--root", str(tmp_path)])

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed"
    assert Path(result["manifest_path"]).exists()


def test_run_subcommand_rejects_missing_spec_argument():
    with pytest.raises(SystemExit):
        analysis_cli.main(["run"])


def test_existing_analysis_cli_arguments_are_unchanged():
    parser = analysis_cli.build_arg_parser()
    assert not parser._subparsers

    bundle_args = parser.parse_args(["--bundle", "rlrmp/standard_matrix", "--manifest-root", "/m"])
    assert bundle_args.bundle == "rlrmp/standard_matrix"
    assert bundle_args.manifest_root == "/m"
    assert bundle_args.single is None

    single_args = parser.parse_args(["--single", "part2.plant_perts"])
    assert single_args.single == "part2.plant_perts"

    with pytest.raises(SystemExit):
        parser.parse_args([])
