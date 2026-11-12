"""Migrate actual saved Studio files without opening a Host or the old Store."""

import json
from pathlib import Path

import pytest

from feedbax.cli import main
from feedbax.contracts.figure_document import PLOTLY_FIGURE_ROLE
from feedbax.contracts.graph import (
    ANALYSIS_CANVAS_LAYOUT_SCHEMA_ID,
    ANALYSIS_CANVAS_LAYOUT_SCHEMA_VERSION,
    AnalysisCanvasLayoutDocument,
    AnalysisPageSpec,
    GraphProject,
)
from feedbax.contracts.project_inputs import RunReference
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.training import TaskSpec
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.web.services.project_folder import ProjectFolderService


_FIXTURES = Path(__file__).with_name("fixtures")


@pytest.fixture
def saved_workspace(tmp_path, monkeypatch):
    root = tmp_path / "saved"
    root.mkdir()
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(tmp_path / "host"))
    return root


def _save(root, change=None):
    raw = (_FIXTURES / "studio_legacy_reaching_project.json").read_bytes()
    project = GraphProject.model_validate_json(raw)
    if change is not None:
        change(project)
        raw = project.model_dump_json(exclude_none=True).encode()
    source = root / "legacy.json"
    source.write_bytes(raw)
    return source, project


def test_migrate_saved_workspace_opens_and_plans_the_same_train(tmp_path, saved_workspace, capsys):
    source, saved = _save(saved_workspace)
    before = source.read_bytes()
    assert main(["migrate-project", str(source)]) == 0
    result = capsys.readouterr()
    destination = Path(result.out.strip())
    assert destination == source.with_suffix("")
    assert not result.err
    assert source.read_bytes() == before
    assert not (tmp_path / "host").exists()
    assert not (destination / "results").exists()

    registries = new_application_registry_bundle()
    registries.seal()
    folders = ProjectFolderService(destination.parent, operations=registries.operations)
    opened = folders.open(destination.name)
    project = opened.snapshot.project
    assert project.documents["model"].payload == saved.graph.model_dump(mode="json")
    scenario = saved.workspace.scenarios["scenario:train"]
    assert project.documents["task"].payload == TaskSpec.model_validate(scenario.task_spec).model_dump(mode="json", exclude_none=True)
    assert {step.kind for step in project.steps.values()} == {"train", "evaluate", "analyze", "report"}
    train = project.steps["train"].base.inline
    assert train["graph"] == {"document": "model"} and train["task"] == {"document": "task"}
    assert train["task_binding_spec"] == scenario.task_binding_spec.model_dump(mode="json", exclude_none=True)
    assert train["objective"]["loss"]["selector"] == scenario.training_spec["loss"]["selector"]
    model = opened.snapshot.presentation.documents["model"]
    assert model.camera.model_dump() == saved.workspace_document.graph_ui_state.viewport.model_dump()
    assert {
        name: position.model_dump() for name, position in model.positions.items()
    } == {
        name: node.position.model_dump()
        for name, node in saved.workspace_document.graph_ui_state.node_states.items()
    }

    settled = folders.read_plan(destination.name, revision=opened.snapshot.revision)
    (planned,) = settled.plan.runs
    assert planned.key == RunKey("default", "train", "default")
    request = decode_request(settled.plan.requests[planned.request_id], settled.plan.documents.__getitem__)
    expected = json.loads((_FIXTURES / "studio_legacy_reaching_train.json").read_bytes())
    actual = request.spec.model_dump(mode="json")
    assert {field: actual[field] for field in expected} == expected
    assert planned.admission_problem is None
    assert {problem.step for problem in settled.plan.diagnostics} == {"eval", "analysis", "report"}

    assert main(["plan", str(destination)]) == 2
    cli_plan = capsys.readouterr()
    assert f"default/train/default {planned.request_id}" in cli_plan.out
    assert "steps/eval.json inputs.subject" in cli_plan.err
    assert not (destination / "results").exists()

    before_files = {path.relative_to(destination): path.read_bytes() for path in destination.rglob("*") if path.is_file()}
    assert main(["migrate-project", str(source), "--out", str(destination)]) == 2
    assert "Destination already exists" in capsys.readouterr().err
    assert {path.relative_to(destination): path.read_bytes() for path in destination.rglob("*") if path.is_file()} == before_files


def test_migrate_keeps_historical_subject_and_reports_its_source_path(tmp_path, saved_workspace, capsys):
    def change(draft):
        stage = draft.workspace.stages[1]
        stage.selection_spec = {
            "training_run_ids": ["feedbax-training-run:retained"],
            "eval_params": {"perturbation_type": "curl_field"},
        }

    source, _ = _save(saved_workspace, change)
    destination = tmp_path / "converted"
    assert main(["migrate-project", str(source), "--out", str(destination)]) == 0
    assert "/workspace/stages/1/selection_spec/training_run_ids/0" in capsys.readouterr().err
    opened = ProjectFolderService(tmp_path).open("converted")
    step = opened.snapshot.project.steps["eval"]
    assert step.inputs["subject"] == RunReference(run="feedbax-training-run:retained")
    assert step.base.inline["params"] == {"perturbation_type": "curl_field"}
    assert not (destination / "results").exists()


def _analysis_layout(draft, *, blank=False, eval_run_id=None):
    nodes = {} if blank else {"align": {}}
    page = AnalysisPageSpec(
        id="page", name="Velocity", graph_spec={"nodes": nodes, "dataSourceId": "input"},
        eval_run_id=eval_run_id,
    )
    if blank:
        spec = {"pages": [page.model_dump(mode="json")]}
    else:
        spec = {"operations": {"align": {"operation": "feedbax.aligned_velocity", "inputs": {}}}, "views": {}}
    draft.workspace.scenarios["scenario:analysis"].analysis_spec = spec
    draft.workspace_document.analysis_pages = [page]
    positions = {"input": {"x": 0, "y": -20}}
    if not blank:
        positions["align"] = {"x": 42, "y": -20}
    draft.workspace_document.analysis_canvas_layout = AnalysisCanvasLayoutDocument.model_validate({
        "schema_id": ANALYSIS_CANVAS_LAYOUT_SCHEMA_ID,
        "schema_version": ANALYSIS_CANVAS_LAYOUT_SCHEMA_VERSION,
        "stages": {"stage:analysis": {"pages": {"page": {
            "node_positions": positions, "viewport": {"x": 17, "y": -5, "zoom": 0.75},
        }}}},
    })


@pytest.mark.parametrize("blank", [False, True])
def test_migrate_maps_analysis_canvas_to_one_document(tmp_path, saved_workspace, blank):
    source, _ = _save(saved_workspace, lambda draft: _analysis_layout(draft, blank=blank))
    destination = tmp_path / "converted"
    assert main(["migrate-project", str(source), "--out", str(destination)]) == 0
    snapshot = ProjectFolderService(tmp_path).open("converted").snapshot
    document = snapshot.project.steps["analysis"].base.document
    operations = snapshot.project.documents[document].payload["operations"]
    if blank:
        assert operations == {}
    else:
        assert operations["align"]["operation"] == "feedbax.aligned_velocity"
    presentation = snapshot.presentation.documents[document]
    assert presentation.camera.model_dump() == {"x": 17, "y": -5, "zoom": 0.75}
    expected = {"states": 0} if blank else {"align": 42, "states": 0}
    assert {node: position.x for node, position in presentation.positions.items()} == expected


@pytest.mark.parametrize(
    "unsupported", [
        "stage", "figure", "legacy-analysis", "analysis-page-reference",
        "legacy-report", "early-stopping", "authored-objective",
    ],
)
def test_migrate_refuses_unrepresentable_authoring_at_its_saved_path(tmp_path, saved_workspace, capsys, unsupported):
    def change(draft):
        workspace = draft.workspace
        if unsupported == "stage":
            workspace.stages[1].selection_spec = {"inputs": {"subject": {"step": "stage:missing"}}}
            return "/workspace/stages/1/selection_spec/inputs/subject/step"
        if unsupported == "figure":
            workspace.scenarios["scenario:report"].report_spec = {"title": "Reach", "blocks": [{"figure": "velocity"}]}
            workspace.stages[3].selection_spec = {"inputs": {
                "velocity": {"step": "stage:analysis", "element": "missing", "product": PLOTLY_FIGURE_ROLE},
            }}
            return "/workspace/stages/3/selection_spec/inputs/velocity/element"
        if unsupported == "legacy-analysis":
            workspace.scenarios["scenario:analysis"].analysis_spec = {"pages": [{"id": "page", "name": "Old canvas", "graph_spec": {"nodes": {"x": {}}}}]}
            return "/workspace/scenarios/scenario:analysis/analysis_spec/pages/0/graph_spec"
        if unsupported == "analysis-page-reference":
            _analysis_layout(draft, eval_run_id="feedbax-evaluation-run:historical")
            return "/workspace_document/analysis_pages/0/eval_run_id"
        if unsupported == "legacy-report":
            workspace.scenarios["scenario:report"].report_spec = {"report_type": "feedbax.studio_report"}
            return "/workspace/scenarios/scenario:report/report_spec/report_type"
        if unsupported == "authored-objective":
            workspace.scenarios["scenario:train"].objective_spec["terms"][0]["weight"] = 2.0
            return "/workspace/scenarios/scenario:train/objective_spec/terms"
        workspace.scenarios["scenario:train"].training_spec["early_stopping"] = {
            "metric": "loss", "patience": 2, "min_delta": 0.01,
        }
        return "/workspace/scenarios/scenario:train/training_spec/early_stopping"

    expected = []

    def configure(draft):
        expected.append(change(draft))

    source, _ = _save(saved_workspace, configure)
    destination = tmp_path / "converted"
    before = source.read_bytes()
    assert main(["migrate-project", str(source), "--out", str(destination)]) == 2
    assert expected[0] in capsys.readouterr().err
    assert not destination.exists()
    assert source.read_bytes() == before
