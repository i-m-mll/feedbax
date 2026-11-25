"""Choose exact Train products by final metrics through saved Plan and Host."""

from feedbax.application.bundle import new_application_registry_bundle
from feedbax.contracts.jobs import JobStatus
from feedbax.contracts.project import Project
from feedbax.contracts.run_rule import metric_condition_holds
from feedbax.contracts.store_closure import ReceiptRoot, StoreClosureRef
from feedbax.jobs import JobService
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView
from feedbax.persistence.receipt_closure import ReceiptClosure
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanService
from tests.support.host import job_row, token, wait_until
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_saved_update_series import definition


def test_final_metric_condition_excludes_one_replicated_candidate(host, tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    bundle = new_application_registry_bundle()
    folders = ProjectFolderService(tmp_path / "projects", operations=bundle.operations)
    jobs = JobService(host, bundle)
    service = ProjectPlanService(folders, jobs)
    raw = definition("last")
    train = raw["steps"]["train"]
    train["base"]["inline"]["seed"] = 0
    train["base"]["inline"]["training_config"]["replicates"] = 2
    train["sweep"] = {"axes": {"seed": {"values": {
        key: {"value": seed, "patches": {"seed": {"value": seed}}}
        for key, seed in (("zero", 0), ("high", 1000))
    }}}}
    raw["steps"]["baseline"]["inputs"]["subject"]["where"] = {"rule": "train_loss < seed"}
    raw["steps"]["speed"] = {
        "kind": "analyze",
        "base": {"inline": {
            "inputs": {"states": {"kind": "dataset"}},
            "operations": {"speed": {
                "operation": "feedbax.trial_speed",
                "params": {"velocity": "/v1/model/state/plant/effector~1velocity"},
                "inputs": {"states": {"input": "states"}},
            }},
        }},
        "inputs": {"states": {"step": "baseline"}},
    }
    project = "metric-condition"
    _save_project(folders, project, Project.model_validate(raw))
    saved = folders.read_plan(project, revision=None)
    assert not saved.plan.diagnostics, saved.plan.diagnostics
    trains = [run for run in saved.plan.runs if run.kind == "train"]
    evaluations = [run for run in saved.plan.runs if run.kind == "evaluate"]
    gathered, = saved.plan.elements.values()
    assert len(trains) == len(evaluations) == len(gathered.inputs["states"]) == 2

    def settle(admitted):
        attempts = {}
        for item in admitted.items:
            assert item.disposition == "admitted", item
            row = wait_until(lambda: (
                current if (current := job_row(host, item.job_id))["outcome"] is not None else None
            ))
            attempts[row["request"]] = row
        return attempts

    trained = settle(jobs.admit_plan(
        folders, saved, run_keys=tuple(run.key for run in trains), token=token("metric-train"),
    ))
    assert all(row["state"] == "succeeded" for row in trained.values()), [row["outcome"] for row in trained.values()]
    status = {row.key: row.status for row in service.read(project, revision=None).status_rows()}
    zero, = [run.key for run in evaluations if run.coordinates["seed"].value == 0]
    assert status[zero] is JobStatus.NOT_CHOSEN
    pass_run, = [run for run in evaluations if run.key != zero]
    results = settle(jobs.admit_plan(
        folders, saved, run_keys=(pass_run.key, gathered.key.run), token=token("metric-chosen-only"),
    ))
    assert not any(row["request"] == next(run.request_id for run in evaluations if run.key == zero)
                   for row in host.jobs(project=saved.project_id))
    results.update(settle(jobs.admit_plan(folders, saved, run_keys=(zero,), token=token("metric-false"))))
    view = SuccessView(ImmutableArtifactBlobProvider(saved.store_path))
    rows = {row.key: row for row in service.read(project, revision=None).status_rows()}
    for run in evaluations:
        accepted = run.coordinates["seed"].value == 1000
        result = results[run.request_id]
        if accepted:
            assert result["state"] == "succeeded", result["outcome"]
            assert rows[run.key].status is JobStatus.SUCCEEDED
            assert view.read_attempt(result["id"]) is not None
        else:
            assert result["state"] == "not_chosen", result["outcome"]
            assert rows[run.key].status is JobStatus.NOT_CHOSEN
            assert not any(event["kind"] == "assigned" for event in host.events(projects=[saved.project_id], limit=1000)["events"]
                           if event["job_id"] == result["id"])
        assert metric_condition_holds(
            run.inputs["subject"].member.reference.conditions, lambda _: 0.1,
        ) is accepted


    result = results[gathered.request_id]
    assert result["state"] == "succeeded", result["outcome"]
    success = view.read_attempt(result["id"])
    assert success is not None
    (chosen,) = success.receipt.inputs["states"]
    (passed,) = [run for run in evaluations if run.key != zero]
    chosen_result = view.read_attempt(results[passed.request_id]["id"])
    assert chosen_result is not None and chosen.selection.receipt == chosen_result.ref
    excluded, = success.receipt.excluded["states"].values()
    rejected, = [run for run in evaluations if run.key == zero]
    assert excluded.request.id == rejected.request_id
    assert excluded.cause.train in ReceiptClosure(view.store).walk(StoreClosureRef(
        roots=(ReceiptRoot(receipt=success.ref),), include_ancestry=True,
    ))
