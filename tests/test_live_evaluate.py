"""Normal saved admission owns live products across an interrupted Train delivery."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import signal

from feedbax.contracts.project import Project
from feedbax.jobs import JobService
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.training.validation_placement import ValidationBudget, ValidationQualification, validation_runtime
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanService
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_local_job_host import host as host_fixture, job_row, restart, wait_until
from tests.test_saved_update_series import definition

host = host_fixture


def test_live_evaluate_replays_prefix_and_backfills_exact_interrupted_checkpoint(host, tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    raw = definition({"every": 2})
    train = raw["steps"]["train"]["base"]["inline"]
    train["training_config"]["n_batches"] = 8
    train["method_payload"]["payload"]["target_total"] = 8
    raw["steps"]["other"] = deepcopy(raw["steps"]["baseline"])
    raw["steps"]["other"]["base"]["inline"]["params"]["seed"] += 1
    project = "delegated-evaluate"
    folders = ProjectFolderService(tmp_path / "projects")
    _save_project(folders, project, Project.model_validate(raw))
    plan = folders.read_plan(project, revision=None)
    assert not plan.plan.diagnostics
    rows = {run.key.step: run for run in plan.plan.runs}
    budget = ValidationBudget(durable_point_p95_ms=3, steady_update_ms=20, max_point_ms=10, max_update_fraction=.1)
    jobs = JobService(host, new_application_registry_bundle())
    host.policy(compute=1)
    (tmp_path / "stop_at_publication.py").write_text("""
import os
from pathlib import Path
from time import monotonic, sleep

from feedbax.training_host_worker import _Events

original = _Events.append

def append(self, kind, data, **kwargs):
    original(self, kind, data, **kwargs)
    if kind == "evaluation_point":
        original(self, kind, data, **kwargs)
    if kind != "publication" or data.get("key") != "checkpoint:6":
        return
    signal = Path(os.environ["CUT_MARKER"])
    signal.touch()
    deadline = monotonic() + 20
    while monotonic() < deadline:
        sleep(.01)
    raise TimeoutError("expected worker-loss injection")

_Events.append = append
""")
    jobs._training_worker = replace(
        jobs._training_worker,
        env=jobs._training_worker.env | {
            "PYTHONPATH": f"{tmp_path}:{Path.cwd()}",
            "CUT_MARKER": str(tmp_path / "before-point"),
        },
        preload=(*jobs._training_worker.preload, "stop_at_publication"),
    )
    jobs._live_validation_qualifications = {
        (rows["train"].request_id, rows[step].request_id): ValidationQualification(
            train_request=rows["train"].request_id, evaluate_request=rows[step].request_id,
            runtime=validation_runtime(asdict(jobs._training_worker)), budget=budget,
            evidence="synthetic lifecycle qualification",
        )
        for step in ("baseline", "other")
    }
    run_keys = tuple(rows[step].key for step in ("train", "baseline", "other"))
    admission = jobs.admit_plan(folders, plan, run_keys=run_keys, token="joint")
    assert all(item.disposition == "admitted" for item in admission.items), admission
    attempts = {json.loads(item.run_key)[1]: item.job_id for item in admission.items}
    producer, evaluation, ordinary = (attempts[step] for step in ("train", "baseline", "other"))

    def points(job_id):
        return [
            event["data"] for event in host.events(0, projects=(project,))["events"]
            if event["job_id"] == job_id and event["kind"] == "evaluation_point"
        ]

    wait_until(lambda: (tmp_path / "before-point").exists(), timeout=20)
    prefix = wait_until(lambda: found if len(found := points(evaluation)) == 2 else None)
    assert [point["checkpoint"]["completed_updates"] for point in prefix] == [2, 4]
    service = ProjectPlanService(folders, jobs)

    def saved_attempt(step="baseline"):
        return service.read(project, revision=plan.revision).evaluations[rows[step].key].checkpoint_attempt

    attempt = saved_attempt()
    assert attempt.job_id == evaluation
    assert attempt.placement.location == "live"
    assert attempt.pending.producer_job_id == producer
    assert attempt.pending.next_update == 6
    assert [point.model_dump(mode="json") for point in attempt.points] == [
        {
            "job_id": producer,
            "request_id": rows["train"].request_id,
            "completed_updates": point["checkpoint"]["completed_updates"],
            "publication_cursor": point["publication_cursor"],
            "transaction_root_sha256": point["checkpoint"]["pointer"]["transaction_root_sha256"],
            "status": "scored",
        }
        for point in prefix
    ]
    ordinary_attempt = saved_attempt("other")
    assert ordinary_attempt.placement.model_dump(mode="json") == {
        "location": "ordinary", "reason": "driver_occupied",
    }
    assert not ordinary_attempt.points
    last_point_cursor = max(
        event["cursor"] for event in host.events(after=0, projects=(project,))["events"]
        if event["job_id"] == evaluation and event["kind"] == "evaluation_point"
    )
    _, activity = jobs.events(after=last_point_cursor - 1, projects=(project,), wait=0, machine=True)
    assert any(
        event.cursor == last_point_cursor and event.job_id == evaluation
        and event.kind == "activity" and event.activity is not None and not event.data
        for event in activity.events
    )
    assert job_row(host, evaluation)["data"]["placement"] == "live"
    assert job_row(host, ordinary)["data"]["placement_detail"]["reason"] == "driver_occupied"
    assert {item.job_id: item.placement for item in jobs.jobs(project=project).jobs}[evaluation] == "live"
    running = job_row(host, producer)
    assert running["request"] == rows["train"].request_id
    os.killpg(running["worker_pid"], signal.SIGKILL)
    interrupted = wait_until(lambda: job_row(host, producer, "interrupted"))
    assert interrupted["checkpoint"]["completed_updates"] == 6
    restart(host)
    assert points(evaluation) == prefix
    assert job_row(host, evaluation, "queued") is not None
    restarted = saved_attempt()
    assert restarted.pending.state == "resumable"
    assert restarted.pending.recovery_update == 6
    assert restarted.pending.next_update == 8
    assert restarted.points == attempt.points
    assert jobs.admit_plan(folders, plan, run_keys=run_keys, token="joint") == admission

    resumed = jobs.resume(producer, "exact-resume")
    wait_until(lambda: job_row(host, resumed.job_id, "succeeded"), timeout=22)
    result = wait_until(lambda: job_row(host, evaluation, "succeeded"), timeout=22)
    wait_until(lambda: job_row(host, ordinary, "succeeded"), timeout=22)
    bound = result["edges"][0]["binding"]
    assert bound["checkpoint_selection"]["lineage"] == [producer, resumed.job_id]
    assert [point["checkpoint"]["completed_updates"] for point in points(evaluation)] == [2, 4, 8]
    assert bound["live_evaluation_points"] == points(evaluation)
    assert points(ordinary) == []

    delegated = [
        event["data"] for event in host.events(0, projects=(project,))["events"]
        if event["job_id"] == evaluation and event["kind"] == "evaluation_delegated"
    ]
    assert [entry["producer_job_id"] for entry in delegated] == [producer, resumed.job_id]

    live = jobs.read_evaluation_result(project=project, store=plan.store_path, job_id=evaluation)
    assert live._manifest().metadata["placement"] == "live"
    with live.open_states() as dataset:
        assert dataset.data("update").tolist() == [2, 4, 6, 8]
    (late,) = jobs.admit_plan(folders, plan, run_keys=(rows["baseline"].key,), token="late").items
    wait_until(lambda: job_row(host, late.job_id, "succeeded"), timeout=22)
    ordinary_result = jobs.read_evaluation_result(project=project, store=plan.store_path, job_id=late.job_id)
    assert job_row(host, late.job_id)["data"]["placement_detail"]["reason"] == "late"
    assert ordinary_result._manifest().metadata["placement"] == "ordinary"
    assert ordinary_result._manifest().artifacts[0].sha256 == live._manifest().artifacts[0].sha256
    assert job_row(host, late.job_id)["request"] == result["request"] == rows["baseline"].request_id
    view = SuccessView(ImmutableArtifactBlobProvider(plan.store_path))
    first, second = view.read_attempt(evaluation), view.read_attempt(late.job_id)
    assert first.receipt.effective == second.receipt.effective
    assert first.receipt.products == second.receipt.products
    assert first.receipt.evaluation_manifest != second.receipt.evaluation_manifest
    assert second.receipt.supersedes == first.ref


def test_lost_pinned_continuation_does_not_wait_for_a_sibling(tmp_path, monkeypatch):
    from feedbax.host.checkpoint_selection import CheckpointSelection
    from feedbax.host.live_evaluate import choose_evaluation
    from feedbax.host.store import Store, dump

    store = Store(tmp_path / "host.sqlite")
    checkpoint = {"completed_updates": 2}
    parent_launch = {"payload": {}, "warm_entrypoint": "feedbax.training_host_worker:main"}
    resume_launch = parent_launch | {"payload": {"resume": checkpoint}}
    selector = {"kind": "update_series", "updates": [4]}
    evaluate_launch = {"payload": {"canonical_request": {"checkpoint": {"select": selector}}}}
    with store.db:
        for job, request, launch, state, parent, pointer in (
            ("parent", "train:request", parent_launch, "interrupted", None, checkpoint),
            ("resumed", "train:request", resume_launch, "failed", "parent", None),
            ("sibling", "train:request", resume_launch, "queued", "parent", None),
            ("evaluate", "evaluate:request", evaluate_launch, "queued", None, None),
        ):
            store.db.execute(
                """INSERT INTO jobs(id,project,request,launch,slot,data,state,continues,checkpoint,delivery)
                   VALUES (?, 'project', ?, ?, 'compute', ?, ?, ?, ?, ?)""",
                (job, request, dump(launch), dump({"placement": "live"} if job == "evaluate" else {}),
                 state, parent, dump(pointer) if pointer else None, int(state != "queued")),
            )
        store.db.execute("INSERT INTO edges(job,parent,publication,authenticated_completion) VALUES ('evaluate','parent','',1)")
        store.db.executemany("INSERT INTO live_evaluations VALUES ('evaluate',?,1)", [("parent",), ("resumed",)])

    assert choose_evaluation(store.db, store.get("sibling")) is None

    def closure(attempt, path):
        return {"kind": "recovery", "product": {"checkpoint": checkpoint}} if attempt["id"] == "parent" else None

    monkeypatch.setattr(store, "_series_closure", closure)
    store.resolve_completions()
    attempt = store.get("evaluate")
    assert attempt["state"] == "interrupted"
    assert json.loads(attempt["outcome"])["error"]["code"] == "input_unavailable"

    with store.db:
        store.db.execute("UPDATE jobs SET checkpoint=? WHERE id='resumed'", (dump(checkpoint),))
    selection = CheckpointSelection((4,), requires_catalog=False, allows_gaps=True)
    assert store._completion_candidate("parent", selection, observe=True).pending is None
    assert store._completion_candidate("parent", selection, tip="resumed", observe=True).pending == {
        "producer_job_id": "resumed",
        "producer_request_id": "train:request",
        "state": "resumable",
        "next_update": None,
        "recovery_update": 2,
    }
