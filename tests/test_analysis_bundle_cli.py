"""File-authored analysis bundle execution through `feedbax-analysis bundle`.

These cover the two halves of the same gap: a path-accepting bundle entrypoint
that needs no registered experiment package, and staged execution bindings on
`execute_analysis_bundle` so a template whose recipe needs checkpoint custody
can run at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from feedbax.analysis.bundles import (
    AnalysisBundleSpec,
    AnalysisSpecTemplate,
    BundleStageOutputSpec,
    BundleStageSpec,
    ManifestPredicate,
)
from feedbax.analysis.evaluation import (
    EvaluationRecipeResult,
    execute_evaluation_run_spec,
)
from feedbax.analysis.specs import (
    AnalysisRecipeResult,
)
from feedbax.bin import analysis as analysis_cli
from feedbax.config.yaml import get_yaml_loader
from feedbax.contracts.analysis_bundle_composition import (
    ANALYSIS_BUNDLE_DELTA_SPEC_SCHEMA_ID,
    ANALYSIS_BUNDLE_DELTA_SPEC_SCHEMA_VERSION,
)
from feedbax.contracts.base import (
    ParentRef,
    canonical_json_bytes,
    sha256_bytes,
)
from feedbax.contracts.manifest import (
    EvaluationRunSpec,
    TrainingRunManifest,
    load_manifest,
    write_manifest,
)
from feedbax.plugins.bootstrap import BootstrapState
from tests.support.analysis_fixtures import ToyAnalysis, build_toy_analysis_data
from tests.support.application_registry import registry_bundle_fixture as registry_bundle_fixture

pytestmark = [pytest.mark.feedbax_contract, pytest.mark.analysis_recipe_contract]

BUNDLE_CLI_ANALYSIS_TYPE = "feedbax.test.bundle_cli_analysis"
BUNDLE_CLI_EVALUATION_TYPE = "feedbax.test.bundle_cli_eval"


@pytest.fixture
def toy_evaluation_recipe(application_registry_bundle):
    def recipe(run_spec: EvaluationRunSpec, _root: Path, _states_path: Path, _context):
        return EvaluationRecipeResult(
            states={"value": np.asarray(run_spec.params["n_trials"], dtype=np.int32)},
            summary_metrics={"n_trials": run_spec.params["n_trials"]},
        )

    application_registry_bundle.evaluation_recipes.register(BUNDLE_CLI_EVALUATION_TYPE, recipe)
    return application_registry_bundle.evaluation_recipes


@pytest.fixture
def toy_analysis_recipe(application_registry_bundle, monkeypatch):
    def recipe(_spec, _root, inputs, _execution_context):
        value = sum(int(resolved.states["value"]) for resolved in inputs)
        return AnalysisRecipeResult(
            analyses={"toy": ToyAnalysis(variant="toy", cache_result=True)},
            data=build_toy_analysis_data(value=value),
        )

    application_registry_bundle.analysis_recipes.register(BUNDLE_CLI_ANALYSIS_TYPE, recipe)

    async def compose_application(**_kwargs):
        return BootstrapState(application_registry_bundle, ())

    monkeypatch.setattr(analysis_cli, "compose_application", compose_application)
    return application_registry_bundle.analysis_recipes


def _execute_toy_eval(root: Path, registry, *, n_trials: int = 4):
    spec = EvaluationRunSpec(
        evaluation_type=BUNDLE_CLI_EVALUATION_TYPE,
        inputs=[
            ParentRef(
                kind="TrainingRunManifest",
                id="feedbax-training-run:bundle-cli",
                role="training_run",
            )
        ],
        params={"n_trials": n_trials},
    )
    return execute_evaluation_run_spec(spec, registry=registry, root=root, force=True)


def _template_bundle_payload(
    evaluation_id: str,
    *,
    analysis_type: str = BUNDLE_CLI_ANALYSIS_TYPE,
    params: dict | None = None,
) -> dict:
    return AnalysisBundleSpec(
        name="bundle_cli_templates",
        predicate=ManifestPredicate(
            manifest_kind="EvaluationRunManifest",
            run_ids=[evaluation_id],
        ),
        templates=[
            AnalysisSpecTemplate(
                name="per_run",
                mode="per-run",
                analysis_type=analysis_type,
                params=params or {},
            )
        ],
    ).model_dump(mode="json", exclude_none=True)


# --- Half one: a path-accepting bundle entrypoint -------------------------------------


def test_bundle_subcommand_executes_json_bundle_file_without_registered_package(
    tmp_path: Path,
    capsys,
    toy_evaluation_recipe,
    toy_analysis_recipe,
) -> None:
    """The motivating downstream case: tracked JSON bundle, no experiment package."""
    evaluation, _path = _execute_toy_eval(tmp_path, toy_evaluation_recipe, n_trials=4)
    spec_path = tmp_path / "bundle.json"
    spec_path.write_text(
        json.dumps(_template_bundle_payload(evaluation.id), sort_keys=True),
        encoding="utf-8",
    )

    analysis_cli.main(
        [
            "bundle",
            str(spec_path),
            "--root",
            str(tmp_path),
            "--fig-dump-dir",
            str(tmp_path / "figures"),
            "--fig-dump-formats",
            "json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    entry = payload[0]
    assert entry["bundle"] == "bundle_cli_templates"
    assert entry["template"] == "per_run"
    assert entry["mode"] == "per-run"
    assert entry["matched_run_ids"] == [evaluation.id]
    manifest = load_manifest(Path(entry["manifest_path"]))
    assert manifest.kind == "AnalysisRunManifest"
    assert manifest.id == entry["manifest_id"]
    assert manifest.status == "completed"


def test_bundle_subcommand_executes_yaml_bundle_file(
    tmp_path: Path,
    capsys,
    toy_evaluation_recipe,
    toy_analysis_recipe,
) -> None:
    evaluation, _path = _execute_toy_eval(tmp_path, toy_evaluation_recipe, n_trials=5)
    spec_path = tmp_path / "bundle.yaml"
    with spec_path.open("w", encoding="utf-8") as handle:
        get_yaml_loader().dump(_template_bundle_payload(evaluation.id), handle)

    analysis_cli.main(
        [
            "bundle",
            str(spec_path),
            "--root",
            str(tmp_path),
            "--fig-dump-dir",
            str(tmp_path / "figures"),
            "--fig-dump-formats",
            "json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert [entry["matched_run_ids"] for entry in payload] == [[evaluation.id]]


def test_bundle_subcommand_resolves_delta_authored_bundle_file(
    tmp_path: Path,
    capsys,
    toy_evaluation_recipe,
    toy_analysis_recipe,
) -> None:
    evaluation, _path = _execute_toy_eval(tmp_path, toy_evaluation_recipe, n_trials=6)
    base = _template_bundle_payload(evaluation.id)
    (tmp_path / "base.json").write_text(json.dumps(base), encoding="utf-8")
    delta = {
        "schema_id": ANALYSIS_BUNDLE_DELTA_SPEC_SCHEMA_ID,
        "schema_version": ANALYSIS_BUNDLE_DELTA_SPEC_SCHEMA_VERSION,
        "parent": {
            "ref": "base.json",
            "sha256": sha256_bytes(canonical_json_bytes(base)),
        },
        "deltas": [
            {
                "layer_id": "instance",
                "patches": [{"path": "name", "value": "bundle_cli_delta"}],
            }
        ],
    }
    spec_path = tmp_path / "bundle.delta.json"
    spec_path.write_text(json.dumps(delta), encoding="utf-8")

    analysis_cli.main(
        [
            "bundle",
            str(spec_path),
            "--root",
            str(tmp_path),
            "--repo-root",
            str(tmp_path),
            "--fig-dump-dir",
            str(tmp_path / "figures"),
            "--fig-dump-formats",
            "json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert [entry["bundle"] for entry in payload] == ["bundle_cli_delta"]


def test_bundle_subcommand_dry_runs_file_authored_staged_bundle(tmp_path: Path, capsys) -> None:
    """The path form dispatches on execution shape, exactly as the registry-key form."""
    write_manifest(
        TrainingRunManifest(
            id="feedbax-training-run:staged-cli",
            status="completed",
            metadata={"method": "minimax"},
        ),
        root=tmp_path,
    )
    payload = AnalysisBundleSpec(
        name="bundle_cli_staged",
        predicate=ManifestPredicate(
            manifest_kind="TrainingRunManifest",
            metadata_equals={"method": "minimax"},
        ),
        stages=[
            BundleStageSpec(
                name="only",
                kind="analysis",
                analysis_type=BUNDLE_CLI_ANALYSIS_TYPE,
                outputs=[BundleStageOutputSpec(role="analysis_summary")],
            )
        ],
    ).model_dump(mode="json", exclude_none=True)
    spec_path = tmp_path / "staged-bundle.json"
    spec_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*.json"))

    analysis_cli.main(["bundle", str(spec_path), "--root", str(tmp_path), "--dry-run"])

    result = json.loads(capsys.readouterr().out)
    assert result["match_preview"]["match_count"] == 1
    assert [stage["name"] for stage in result["stages"]] == ["only"]
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*.json")) == before


def test_bundle_subcommand_rejects_dry_run_for_template_bundle(tmp_path: Path) -> None:
    spec_path = tmp_path / "bundle.json"
    spec_path.write_text(
        json.dumps(_template_bundle_payload("feedbax-evaluation-run:absent")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="--dry-run is only valid for staged analysis bundles"):
        analysis_cli.main(["bundle", str(spec_path), "--root", str(tmp_path), "--dry-run"])


def test_bundle_subcommand_requires_execution_descriptor_for_bindings(tmp_path: Path) -> None:
    """Binding error handling matches `feedbax-analysis run`."""
    spec_path = tmp_path / "bundle.json"
    spec_path.write_text(
        json.dumps(_template_bundle_payload("feedbax-evaluation-run:absent")),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="--artifact-provider requires --execution-descriptor",
    ):
        analysis_cli.main(
            [
                "bundle",
                str(spec_path),
                "--root",
                str(tmp_path),
                "--artifact-provider",
                f"evidence={tmp_path}",
            ]
        )


def test_bundle_subcommand_rejects_missing_spec_argument() -> None:
    with pytest.raises(SystemExit):
        analysis_cli.main(["bundle"])


def test_existing_bundle_registry_key_form_is_unchanged() -> None:
    parser = analysis_cli.build_arg_parser()
    args = parser.parse_args(["--bundle", "rlrmp/standard_matrix", "--manifest-root", "/m"])
    assert args.bundle == "rlrmp/standard_matrix"
    assert args.manifest_root == "/m"


# --- Half two: staged execution bindings on `execute_analysis_bundle` -----------------
