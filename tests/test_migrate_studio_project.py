"""Migrate actual saved Studio files without opening a Host or the old Store."""

from pathlib import Path

import pytest

from feedbax.cli import main
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.figure_document import PLOTLY_FIGURE_ROLE
from feedbax.contracts.graph import (
    ANALYSIS_CANVAS_LAYOUT_SCHEMA_ID,
    ANALYSIS_CANVAS_LAYOUT_SCHEMA_VERSION,
    AnalysisCanvasLayoutDocument,
    AnalysisPageSpec,
    CanvasViewportSpec,
)
from feedbax.contracts.migrations import admit_studio_persistence_document
from feedbax.contracts.project_inputs import RunReference
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.training import TaskSpec
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.studio.training_requests import lower_studio_training_runs
from feedbax.web.services import graph_service
from feedbax.web.services.project_folder import ProjectFolderService
from web.e2e.studio.fixtures.reaching_project import reaching_project


@pytest.fixture
def studio_writer(tmp_path, monkeypatch):
    root = tmp_path / "saved"
    root.mkdir()
    monkeypatch.setattr(graph_service, "ensure_dirs", lambda: None)
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(tmp_path / "host"))
    return graph_service.GraphService(storage_dir=root), ComponentRegistry(), root


def _save(studio_writer, change=None):
    writer, registry, root = studio_writer
    draft = reaching_project(batches=2)
    draft.workspace_document.graph_ui_state.viewport = CanvasViewportSpec(x=41, y=-18, zoom=0.8)
    scenario = draft.workspace.scenarios["scenario:train"]
    # Wire shape emitted by objectiveSpecFromLossSpec for the reaching fixture.
    scenario.objective_spec = {
        "schema_version": "feedbax.spec.studio.objective.v1",
        "legacy_loss_spec": scenario.training_spec["loss"],
        "metadata": {"lowered_from": "training_spec.loss"},
        "terms": [{
            "id": "objective:root",
            "type_id": "TargetStateLoss",
            "label": "Endpoint position",
            "role": "loss",
            "source_selector": {
                "namespace": "state_path", "compact": "path:plant.effector.pos",
                "target_id": None, "path": "plant.effector.pos",
                "metadata": {"source": "legacy_loss_selector"},
            },
            "target_selector": None,
            "operator": "minimize",
            "penalty": "squared_l2",
            "temporal_selector": {"mode": "final"},
            "weight": 1.0,
            "metadata": {"legacy_loss_path": [], "legacy_loss_type": "TargetStateLoss"},
        }],
    }
    if change is not None:
        change(draft)
    admitted = admit_studio_persistence_document(draft.model_dump(mode="json", exclude_none=True))
    record = writer.create_graph(
        admitted.graph, workspace=admitted.workspace, workspace_document=admitted.workspace_document,
        component_registry=registry,
    )
    return root / f"{record.graph_id}.json", record.project


def test_migrate_saved_workspace_opens_and_plans_the_same_train(tmp_path, studio_writer, capsys):
    source, saved = _save(studio_writer)
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
    (legacy,) = lower_studio_training_runs(
        saved.workspace, saved.graph, stage_id="stage:train", registry_bundle=registries,
    )
    assert request.spec.training_config == legacy.spec.training_config
    assert request.spec.task_binding_spec == legacy.spec.task_binding_spec
    assert request.spec.method_payload == legacy.spec.method_payload
    assert request.spec.objective == legacy.spec.objective
    assert request.spec.task == legacy.spec.task
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


def test_migrate_keeps_historical_subject_and_reports_its_source_path(tmp_path, studio_writer, capsys):
    def change(draft):
        stage = draft.workspace.stages[1]
        stage.selection_spec = {
            "training_run_ids": ["feedbax-training-run:retained"],
            "eval_params": {"perturbation_type": "curl_field"},
        }

    source, _ = _save(studio_writer, change)
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
def test_migrate_maps_analysis_canvas_to_one_document(tmp_path, studio_writer, blank):
    source, _ = _save(studio_writer, lambda draft: _analysis_layout(draft, blank=blank))
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
def test_migrate_refuses_unrepresentable_authoring_at_its_saved_path(tmp_path, studio_writer, capsys, unsupported):
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

    source, _ = _save(studio_writer, configure)
    destination = tmp_path / "converted"
    before = source.read_bytes()
    assert main(["migrate-project", str(source), "--out", str(destination)]) == 2
    assert expected[0] in capsys.readouterr().err
    assert not destination.exists()
    assert source.read_bytes() == before
