"""CLI exposure of `execute_analysis_run_spec` through `feedbax-analysis run`."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import plotly.io as pio
import pytest

from feedbax.plugins import discovery as plugin_discovery
from feedbax.analysis.specs import (
    AnalysisRecipeResult,
    execute_analysis_run_spec,
    register_analysis_recipe,
    unregister_analysis_recipe,
)
from feedbax.analysis.execution_context import (
    StagedCheckpointCustodyRootBinding,
    StagedExecutionContextError,
)
from feedbax.contracts.analysis_composition import AnalysisRunDeltaSpec
from feedbax.bin import analysis as analysis_cli
from feedbax.config import PLOTLY_CONFIG
from feedbax.config.yaml import get_yaml_loader
from feedbax.contracts.manifest import (
    ANALYSIS_RUN_DELTA_SPEC_SCHEMA_ID,
    ANALYSIS_RUN_DELTA_SPEC_SCHEMA_VERSION,
    AnalysisRunSpec,
    ParentRef,
    TrainingRunManifest,
    canonical_json_bytes,
    load_manifest,
    sha256_bytes,
)
from feedbax.contracts.staged_execution import (
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
    STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
    StagedCheckpointCustodySpec,
    StagedExecutionDescriptor,
)
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


@pytest.fixture
def restore_plotly_template():
    """Restore the process-global Plotly template default after the test."""
    previous = pio.templates.default
    try:
        yield
    finally:
        pio.templates.default = previous


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


def test_run_subcommand_loads_installed_plugin_before_recipe_execution(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    analysis_type = "feedbax.test.installed_direct_plugin"
    calls: list[str] = []

    def recipe(spec, _root, _inputs, _execution_context):
        calls.append(spec.analysis_type)
        return AnalysisRecipeResult(
            analyses={"toy": ToyAnalysis(variant="toy", cache_result=True)},
            data=build_toy_analysis_data(value=2),
        )

    plugin = SimpleNamespace(
        register_feedbax_analysis_recipes=lambda: register_analysis_recipe(
            analysis_type,
            recipe,
            replace=True,
        )
    )
    entry_point = SimpleNamespace(
        name="downstream-analysis",
        module="downstream.analysis",
        load=lambda: plugin,
    )
    monkeypatch.setattr(
        plugin_discovery,
        "feedbax_plugin_entry_points",
        lambda _group: [entry_point],
    )
    spec_path = tmp_path / "plugin-analysis.json"
    spec_path.write_text(
        AnalysisRunSpec(
            analysis_type=analysis_type,
            params={"requested_outputs": ["toy"]},
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    try:
        analysis_cli.main(["run", str(spec_path), "--root", str(tmp_path / "output")])
    finally:
        unregister_analysis_recipe(analysis_type)

    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    assert calls == [analysis_type]


def test_run_subcommand_fails_before_spec_when_plugin_registration_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail_registration() -> None:
        raise RuntimeError("plugin registration exploded")

    entry_point = SimpleNamespace(
        name="broken-analysis",
        module="downstream.broken",
        load=lambda: SimpleNamespace(register_feedbax_analysis_recipes=fail_registration),
    )
    monkeypatch.setattr(
        plugin_discovery,
        "feedbax_plugin_entry_points",
        lambda _group: [entry_point],
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Failed to register Feedbax analysis recipes from "
            "entry-point:broken-analysis: plugin registration exploded"
        ),
    ):
        analysis_cli.main(["run", str(tmp_path / "missing-spec.json")])


def test_run_subcommand_executes_yaml_spec_file(tmp_path: Path, capsys, toy_analysis_recipe):
    payload = _toy_spec_payload(tmp_path)
    spec_path = tmp_path / "analysis_spec.yaml"
    with spec_path.open("w", encoding="utf-8") as handle:
        get_yaml_loader().dump(payload, handle)

    analysis_cli.main(["run", str(spec_path), "--root", str(tmp_path)])

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed"
    assert Path(result["manifest_path"]).exists()


def test_run_subcommand_applies_project_plotly_template(
    tmp_path: Path, capsys, toy_analysis_recipe, restore_plotly_template
):
    """The run path applies the same template default as the `--bundle` path."""
    payload = _toy_spec_payload(tmp_path)
    spec_path = tmp_path / "analysis_spec.json"
    spec_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    expected = PLOTLY_CONFIG.templates.default
    sentinel = "plotly" if expected != "plotly" else "none"
    pio.templates.default = sentinel

    analysis_cli.main(["run", str(spec_path), "--root", str(tmp_path)])

    capsys.readouterr()
    assert pio.templates.default == expected


def test_plotly_template_default_honours_explicit_request(restore_plotly_template):
    """The shared helper used by both CLI paths prefers an explicit template name."""
    analysis_cli._apply_plotly_template_default("plotly")
    assert pio.templates.default == "plotly"

    analysis_cli._apply_plotly_template_default()
    assert pio.templates.default == PLOTLY_CONFIG.templates.default


def test_run_subcommand_delta_binds_nested_manifest_and_checkpoint_without_copy(
    tmp_path: Path,
    capsys,
):
    analysis_type = "feedbax.test.direct_runtime_bindings"
    provider_root = tmp_path / "retained" / "parent" / "store"
    checkpoint_root = tmp_path / "retained" / "parent" / "checkpoints"
    output_root = tmp_path / "analysis-output"
    manifest_dir = provider_root / "manifests" / "training_runs"
    manifest_dir.mkdir(parents=True)
    checkpoint_root.mkdir(parents=True)
    training = TrainingRunManifest(
        id="feedbax-training-run:nested-retained",
        status="completed",
    )
    raw = training.model_dump_json(indent=2).encode()
    source_path = manifest_dir / "feedbax-training-run_nested-retained.json"
    source_path.write_bytes(raw)
    source_stat = source_path.stat()
    parent = ParentRef(
        kind="TrainingRunManifest",
        id=training.id,
        role="training_run",
        metadata={
            "ref_schema_id": "feedbax.ref.authenticated_manifest",
            "ref_schema_version": "feedbax.ref.authenticated_manifest.v1",
            "manifest_sha256": sha256_bytes(raw),
            "size_bytes": len(raw),
        },
    )
    base = AnalysisRunSpec(
        analysis_type=analysis_type,
        inputs=[parent],
        params={"requested_outputs": ["toy"], "value": 1},
    ).model_dump(mode="json", exclude_none=True)
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base, sort_keys=True), encoding="utf-8")
    delta = AnalysisRunDeltaSpec.model_validate(
        {
            "schema_id": ANALYSIS_RUN_DELTA_SPEC_SCHEMA_ID,
            "schema_version": ANALYSIS_RUN_DELTA_SPEC_SCHEMA_VERSION,
            "parent": {
                "ref": base_path.name,
                "sha256": sha256_bytes(canonical_json_bytes(base)),
            },
            "deltas": [
                {
                    "layer_id": "instance",
                    "patches": [{"path": "params.value", "value": 7}],
                }
            ],
        }
    )
    spec_path = tmp_path / "analysis.delta.json"
    spec_path.write_text(delta.model_dump_json(indent=2), encoding="utf-8")
    descriptor = StagedExecutionDescriptor(
        schema_id=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
        schema_version=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
        artifact_providers={},
        checkpoint_custody={
            "capture-checkpoints": StagedCheckpointCustodySpec(
                backend="feedbax-checkpoint-transaction-tree"
            )
        },
    )
    descriptor_path = tmp_path / "execution.json"
    descriptor_path.write_text(descriptor.model_dump_json(indent=2), encoding="utf-8")
    recipe_calls = []

    def recipe(spec, _root, inputs, execution_context):
        recipe_calls.append(spec)
        assert inputs[0].manifest == training
        assert inputs[0].path == source_path
        assert execution_context.checkpoint_custody_root("capture-checkpoints") == checkpoint_root
        return AnalysisRecipeResult(
            analyses={"toy": ToyAnalysis(variant="toy", cache_result=True)},
            data=build_toy_analysis_data(value=int(spec.params["value"])),
        )

    register_analysis_recipe(analysis_type, recipe, replace=True)
    try:
        analysis_cli.main(
            [
                "run",
                str(spec_path),
                "--root",
                str(output_root),
                "--repo-root",
                str(tmp_path),
                "--execution-descriptor",
                str(descriptor_path),
                "--manifest-root",
                f"retained={provider_root}",
                "--checkpoint-custody",
                f"capture-checkpoints={checkpoint_root}",
            ]
        )
    finally:
        unregister_analysis_recipe(analysis_type)

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed"
    assert len(recipe_calls) == 1
    assert recipe_calls[0].params["value"] == 7
    assert source_path.stat().st_ino == source_stat.st_ino
    assert source_path.stat().st_size == source_stat.st_size
    assert not (output_root / "manifests" / "training_runs").exists()


def test_direct_python_entrypoint_rejects_unknown_checkpoint_binding_before_recipe(
    tmp_path: Path,
) -> None:
    analysis_type = "feedbax.test.unknown_direct_checkpoint_binding"
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    calls = []

    def recipe(*args):
        calls.append(args)
        raise AssertionError("recipe must not run")

    register_analysis_recipe(analysis_type, recipe, replace=True)
    try:
        with pytest.raises(StagedExecutionContextError, match="unavailable.*unknown"):
            execute_analysis_run_spec(
                AnalysisRunSpec(
                    analysis_type=analysis_type,
                    params={"checkpoint_custody_binding": "unknown"},
                ),
                root=tmp_path / "output",
                execution_descriptor=StagedExecutionDescriptor(
                    schema_id=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_ID,
                    schema_version=STAGED_EXECUTION_DESCRIPTOR_SCHEMA_VERSION,
                    artifact_providers={},
                    checkpoint_custody={
                        "known": StagedCheckpointCustodySpec(
                            backend="feedbax-checkpoint-transaction-tree"
                        )
                    },
                ),
                checkpoint_custody_bindings=[
                    StagedCheckpointCustodyRootBinding("known", checkpoint_root)
                ],
            )
    finally:
        unregister_analysis_recipe(analysis_type)

    assert calls == []
    assert not (tmp_path / "output").exists()


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
