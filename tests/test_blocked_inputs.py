"""A row whose input's producer cannot run is kept and blocked; it never removes the rows that read sound producers."""

from fastapi.testclient import TestClient

from feedbax.contracts.project import Project
from feedbax.planner.project_plan import BlockedRow, PlannedRun, ProjectPlanMemo, RunKey, compile_project
from feedbax.contracts.project_resolution import ResolutionCode
from feedbax.jobs import JobService, _planned_checkpoint_catalogs
from feedbax.application.bundle import new_application_registry_bundle
from feedbax.application.composition import application_python_tasks
from feedbax.web.app import create_app
from feedbax.web.services import project_plan as window_service
from feedbax.web.services.project_folder import ProjectFolderService
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_document_train_pair import _two_optimizer_rates, authored_documents, document_project
from tests.test_model_document_plan import _end, _model_controller, _wire
from tests.test_studio_trial_authoring import _python_project
from tests.support.host import token

BAD_LINEAR = {"kind": "leaf", "name": "Left", "type": "feedbax.equinox.Linear", "fields": {}}


def _one_bad_train_row() -> dict:
    """A two-row Train step whose ``rate=low`` row adds a Linear without its required parameters, read by an
    Evaluate over every Train row."""
    raw = document_project(authored_documents(), evaluation=True)
    _two_optimizer_rates(raw)
    raw["steps"]["train"]["sweep"]["exceptions"] = {"broken": {
        "where": {"rule": "rate=low"},
        "patches": {"trial_documents.model.interior.components.left": {"value": BAD_LINEAR, "op": "add"}},
    }}
    raw["steps"]["baseline"]["sweep"] = {"axes": {"model": {"input": "subject"}}}
    return raw


def test_one_bad_train_row_blocks_only_the_evaluate_row_that_reads_it():
    plan = compile_project(Project.model_validate(_one_bad_train_row()), memo=ProjectPlanMemo())
    bad = RunKey("default", "train", "rate=low")
    evaluations = [row for row in plan.rows if row.key.step == "baseline"]
    assert len(evaluations) == 2
    (sound,) = [row for row in evaluations if isinstance(row, PlannedRun)]
    assert sound.inputs["subject"].producer == RunKey("default", "train", "rate=high")
    (blocked,) = [row for row in evaluations if not isinstance(row, PlannedRun)]
    assert isinstance(blocked, BlockedRow) and blocked.blocked_by == bad
    problems = [problem for problem in plan.diagnostics if problem.step == "baseline"]
    assert [(problem.code, problem.row, problem.path, problem.blocked_by) for problem in problems] == [
        (ResolutionCode.BLOCKED_INPUT, blocked.key.row, "inputs.subject", bad),
    ]


def test_an_evaluate_of_an_unadmissible_train_keeps_its_own_problem():
    raw = document_project(authored_documents(), evaluation=True)
    for term in raw["documents"]["objective"]["payload"]["terms"]:
        term["enabled"] = False
    controller = _model_controller(raw)
    controller["components"]["passthrough"] = {
        "kind": "leaf", "name": "Passthrough", "type": "feedbax.Gain", "fields": {"gain": 1.0},
    }
    original = controller["wires"]["read"]
    original["target"], output = _end("passthrough", "input"), original["target"]
    controller["wires"]["read-output"] = _wire(_end("passthrough", "output"), output)
    plan = compile_project(Project.model_validate(raw), memo=ProjectPlanMemo())
    (train,) = [run for run in plan.runs if run.key.step == "train"]
    assert train.admission_problem is not None and "no enabled terms" in train.admission_problem.detail
    problems = [problem for problem in plan.diagnostics if problem.step == "baseline"]
    assert {problem.row for problem in problems} == {"default"}
    blocked = [problem for problem in problems if problem.code == ResolutionCode.BLOCKED_INPUT]
    assert [problem.blocked_by for problem in blocked] == [train.key]
    assert any("no declared Evaluate layout" in problem.detail for problem in problems)


def test_window_keeps_conditions_and_batches_of_a_train_row_that_cannot_run(tmp_path):
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, "blocked", Project.model_validate(_one_bad_train_row()))
    view = window_service.ProjectPlanRead.without_history(folders.read_plan("blocked", revision=None))
    window = view.window_map(evaluation_tasks=application_python_tasks())
    trains = {row.key.row: row for row in window.rows if row.key.step == "train"}
    assert trains["rate=low"].status == "cannot_run"
    assert trains["rate=low"].evaluation_conditions
    assert trains["rate=low"].evaluation_conditions == trains["rate=high"].evaluation_conditions
    assert trains["rate=low"].target_total == trains["rate=high"].target_total == 2
    assert trains["rate=low"].origin == trains["rate=high"].origin == 0
    (blocked,) = [problem for problem in window.problems if problem.code == ResolutionCode.BLOCKED_INPUT]
    assert blocked.blocked_by is not None and blocked.blocked_by.row == "rate=low"
    assert {(edge.from_key.row, edge.to_key.row) for edge in window.edges if edge.to_key.row == blocked.row} == {
        ("rate=low", blocked.row),
    }


def _at(at: int, *, origin: int = 0):
    """An Evaluate at update ``at`` of a 4-update Train with snapshot interval 2, continuing from ``origin``."""
    raw = document_project(authored_documents(), evaluation=True)
    raw["steps"]["train"]["base"]["inline"]["training_config"].update(n_batches=4, snapshot_interval=2)
    source = "train"
    if origin:
        raw["steps"]["successor"] = {
            "kind": "train",
            "base": {"step": "train"},
            "inputs": {"checkpoint": {"step": "train", "checkpoint": "last"}},
            "deltas": [{"layer_id": "continue", "patches": [{
                "path": "checkpoint_initialization", "op": "add", "value": {
                    "mode": "continue_from", "source_completed_updates": origin,
                    "state": "preserve", "schedule": "continue", "rng": "carry",
                },
            }]}],
        }
        source = "successor"
    raw["steps"]["baseline"]["inputs"]["subject"] = {"step": source, "checkpoint": {"at": at}}
    return compile_project(Project.model_validate(raw), memo=ProjectPlanMemo())


def test_a_checkpoint_in_the_subjects_range_plans_and_is_demanded_of_the_train():
    for at in (2, 3, 4):
        plan = _at(at)
        assert not plan.diagnostics
        by_step = {run.key.step: run for run in plan.runs}
        assert _planned_checkpoint_catalogs(plan) == {by_step["train"].request_id: {at}}


def test_a_checkpoint_outside_the_subjects_range_is_a_problem_on_the_row():
    for plan, at, step, first, last in (
        (_at(5), 5, "'train'", "1", "4"), (_at(0), 0, "'train'", "1", "4"),
        (_at(3, origin=4), 3, "'successor'", "5", "8"),
    ):
        assert not any(run.key.step == "baseline" for run in plan.runs)
        (problem,) = [problem for problem in plan.diagnostics if problem.step == "baseline"]
        assert (problem.code, problem.path, problem.row) == (
            ResolutionCode.BAD_REFERENCE, "inputs.subject.checkpoint", "default",
        )
        assert all(part in problem.detail for part in (f"Update {at} ", step, first, last)), problem.detail


def test_objective_expressions_stay_editable_when_the_model_does_not_pair(tmp_path):
    raw = document_project(authored_documents())
    raw["documents"]["reach-model"]["payload"]["interior"]["components"]["left"] = BAD_LINEAR
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, "unpaired", Project.model_validate(raw))
    settled = folders.read_plan("unpaired", revision=None)
    (row,) = [row for row in settled.plan.rows if row.key.step == "train"]
    assert not isinstance(row, PlannedRun)
    body = {"revision": settled.revision, "run": {"variant": row.key.variant, "step": row.key.step, "row": row.key.row}}
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        reply = client.post("/api/projects/unpaired/authoring/trials", json=body)
        assert reply.status_code == 200, reply.text
        fields = {field["path"]: field["text"] for field in reply.json()["fields"]}
        assert fields["/objective/terms/effort/measure"] == "sqnorm(command)"
        assert fields["/objective/terms/effort/weight"] == "1"
        edit = client.post("/api/projects/unpaired/authoring/trials", json=body | {
            "change": {"kind": "expression", "path": "/objective/terms/effort/weight", "edit": {"kind": "text", "text": "2"}},
        })
        assert edit.status_code == 200, edit.text
        assert edit.json()["files"], edit.text
        assert not [problem for problem in edit.json()["problems"] if problem["path"][:1] == ["objective"]]


def test_a_lowered_row_waiting_on_its_producer_cannot_run_and_is_never_submitted(host, tmp_path):
    folders = ProjectFolderService(tmp_path / "projects")
    _save_project(folders, "waiting", Project.model_validate(_python_project()))
    snapshot = folders.read_plan("waiting", revision=None)
    (train,) = [run for run in snapshot.plan.runs if run.key.step == "train"]
    (evaluate,) = [run for run in snapshot.plan.runs if run.key.step == "baseline"]
    assert train.admission_problem is not None
    view = window_service.ProjectPlanRead.without_history(snapshot)
    assert next(row for row in view.status_rows() if row.key == evaluate.key).status == "cannot_run"
    jobs = JobService(host, new_application_registry_bundle())
    (item,) = jobs.admit_plan(folders, snapshot, run_keys=(evaluate.key,), token=token("waiting")).items
    assert item.disposition == "rejected" and "inputs.subject" in (item.reason or ""), item
    assert not host.jobs(project=snapshot.project_id)
