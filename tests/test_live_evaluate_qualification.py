"""Managed Host qualification for delegated Evaluate stop, refusal and timing."""

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from statistics import mean

import pytest

from feedbax.contracts.jobs import StudioSavedEvaluationRead
from feedbax.contracts.project import InlineBase, Project
from feedbax.contracts.task_document import TaskDocument
from feedbax.jobs import JobService
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.training.validation_placement import ValidationBudget, ValidationQualification, validation_runtime
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanService
from tests.support.managed_train_fixture import managed_train_project
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_local_job_host import host as host_fixture, job_row, restart, wait_until
from tests.test_saved_update_series import definition

host = host_fixture


@pytest.fixture
def managed_jobs(host, tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    host.policy(compute=1)
    return JobService(host, new_application_registry_bundle())


def instrument(jobs, mode, tmp_path):
    worker = jobs._training_worker
    jobs._training_worker = replace(
        worker,
        env=worker.env | {
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            "FEEDBAX_QUALIFICATION_MODE": mode,
            "FEEDBAX_QUALIFICATION_GATE": str(tmp_path),
        },
        preload=(*worker.preload, "tests.support.live_evaluate_qualification"),
    )


def admit(jobs, folders, project, definition, *, qualified=True, train_only=False):
    _save_project(folders, project, definition)
    plan = folders.read_plan(project, revision=None)
    assert not plan.plan.diagnostics, plan.plan.diagnostics
    rows = {run.kind: run for run in plan.plan.runs}
    train, evaluate = rows["train"], rows["evaluate"]
    # This synthetic record only exercises dispatch. No application budget is installed.
    if qualified:
        jobs._live_validation_qualifications[(train.request_id, evaluate.request_id)] = ValidationQualification(
            train_request=train.request_id, evaluate_request=evaluate.request_id,
            runtime=validation_runtime(asdict(jobs._training_worker)), evidence="test-only dispatch",
            budget=ValidationBudget(
                durable_point_p95_ms=1, steady_update_ms=10, max_point_ms=10, max_update_fraction=.2,
            ),
        )
    keys = (train.key,) if train_only else (train.key, evaluate.key)
    submission = jobs.admit_plan(folders, plan, run_keys=keys, token=project)
    assert all(item.disposition == "admitted" for item in submission.items), submission
    return plan, tuple(item.job_id for item in submission.items)


def events(host, project, job_id, kind):
    cursor = 0
    found = []
    while True:
        page = host.events(cursor, projects=(project,))
        found.extend(event["data"] for event in page["events"] if event["job_id"] == job_id and event["kind"] == kind)
        cursor = page["cursor"]
        if cursor >= page["head"]:
            return found


def terminal(host, job_id, state):
    row = wait_until(lambda: (
        current if (current := job_row(host, job_id)) and current["outcome"] is not None else None
    ), timeout=24)
    assert row["state"] == state, row["outcome"]
    return row


def series_definition(updates):
    raw = definition({"every": 2})
    source = raw["steps"]["train"]["base"]["inline"]
    source["training_config"]["n_batches"] = updates
    source["method_payload"]["payload"]["target_total"] = updates
    return Project.model_validate(raw)


def test_live_evaluate_train_stop_and_independent_cancellation(host, tmp_path, managed_jobs):
    jobs = managed_jobs
    instrument(jobs, "lifecycle", tmp_path)
    project = "live-lifecycle"
    folders = ProjectFolderService(tmp_path / "projects")
    plan, (producer, evaluation) = admit(jobs, folders, project, series_definition(8))
    service = ProjectPlanService(folders, jobs)
    (row,) = service.read(project, revision=plan.revision).evaluations.values()
    read = StudioSavedEvaluationRead(revision=plan.revision, key=row.key, job_id=evaluation)

    def stop_signal(job_id, delivery):
        return host.root / "deliveries" / job_id / str(delivery) / "stop"

    try:
        wait_until(lambda: (tmp_path / "stop.reached").exists(), timeout=20)
        assert job_row(host, evaluation)["data"]["placement"] == "live"
        assert jobs.stop([producer], "checkpoint")[0].job.status == "stopping"
        wait_until(lambda: stop_signal(producer, 1).exists())
    finally:
        (tmp_path / "stop.release").touch()
    stopped = terminal(host, producer, "stopped")
    assert stopped["checkpoint"]["completed_updates"] == 4
    assert stopped["stop_mode"] == "checkpoint"
    assert stopped["outcome"].get("signal") is None
    prefix = events(host, project, evaluation, "evaluation_point")
    assert [point["checkpoint"]["completed_updates"] for point in prefix] == [2]
    assert job_row(host, evaluation, "queued")["edges"][0]["binding"] is None
    paused_curve = service.evaluation_objective(project, read)
    assert paused_curve.receipt is None and paused_curve.gaps == []
    assert paused_curve.lineage == [producer]
    assert [point.checkpoint.completed_updates for point in paused_curve.points] == [2]

    restart(host)
    assert events(host, project, evaluation, "evaluation_point") == prefix
    assert service.evaluation_objective(project, read) == paused_curve
    resumed = jobs.resume(producer, "resume")
    try:
        wait_until(lambda: (tmp_path / "cancel.reached").exists(), timeout=20)
        continued_curve = service.evaluation_objective(project, read)
        assert continued_curve.lineage == [producer, resumed.job_id]
        assert continued_curve.points == paused_curve.points
        assert continued_curve.gaps == []
        assert jobs.stop([evaluation], "force")[0].job.status == "stopped"
        wait_until(lambda: stop_signal(evaluation, 0).exists())
        assert not stop_signal(resumed.job_id, 1).exists()
    finally:
        (tmp_path / "cancel.release").touch()
    succeeded = terminal(host, resumed.job_id, "succeeded")
    assert succeeded["stop_mode"] is None
    assert succeeded["checkpoint"]["completed_updates"] == 8
    assert succeeded["request"] == stopped["request"]
    assert succeeded["continues"] == producer
    cancelled = terminal(host, evaluation, "stopped")
    assert cancelled["delivery"] == 0 and cancelled["outcome"].get("receipt") is None
    assert events(host, project, evaluation, "evaluation_point") == prefix
    assert [row["producer_job_id"] for row in events(host, project, evaluation, "evaluation_delegated")] == [
        producer, resumed.job_id,
    ]


def test_live_evaluate_refuses_tampered_assignment_without_failing_train(host, tmp_path, managed_jobs):
    jobs = managed_jobs
    instrument(jobs, "tamper", tmp_path)
    project = "changed-assignment"
    folders = ProjectFolderService(tmp_path / "projects")
    _, (producer, evaluation) = admit(jobs, folders, project, series_definition(2))
    failed = terminal(host, evaluation, "failed")
    assert failed["data"]["placement"] == "live"
    assert failed["delivery"] == 0 and failed["outcome"].get("receipt") is None
    assert failed["outcome"]["error"] == {
        "code": "live_evaluation_protocol", "message": "Live Evaluate assignment changed after delivery.",
    }
    assert events(host, project, evaluation, "evaluation_point") == []
    succeeded = terminal(host, producer, "succeeded")
    assert succeeded["checkpoint"]["completed_updates"] == 2
    assert succeeded["outcome"].get("receipt") is not None


def timing_definition(scenario, batch_size, trial_count):
    project = managed_train_project(batches=36, scenario=scenario)
    train = project.steps["train"].base
    assert isinstance(train, InlineBase)
    train.inline["training_config"].update(batch_size=batch_size, snapshot_interval=1)
    task_id = train.inline["trial_documents"]["task"]["document"]
    task = TaskDocument.model_validate(project.documents[task_id].payload)
    assert len(task.condition_ids) >= trial_count
    raw = project.model_dump(mode="json", by_alias=True, exclude_unset=True)
    raw["steps"]["check"] = {
        "kind": "evaluate",
        "inputs": {"subject": {"step": "train", "checkpoint": {"every": 1}}},
        "base": {"inline": {
            "evaluation_type": "feedbax.rollout",
            "params": {
                "seed": 731, "trial_policy": "paired", "variation": {},
                "trial_set": {"kind": "conditions", "ids": list(task.condition_ids[:trial_count])},
            },
        }},
    }
    return Project.model_validate(raw), task


@pytest.mark.skipif(not os.environ.get("FEEDBAX_LIVE_TIMING"), reason="scoped local CPU measurement")
@pytest.mark.parametrize("scenario,batch_size,trial_count", [
    ("simple-reaches", 1, 1),
    ("delayed-reaches", 4, 28),
])
def test_live_evaluate_fresh_points_and_paired_driver_windows(
    host, tmp_path, managed_jobs, scenario, batch_size, trial_count,
):
    jobs = managed_jobs
    instrument(jobs, "timing", tmp_path)
    definition, task = timing_definition(scenario, batch_size, trial_count)
    folders = ProjectFolderService(tmp_path / "projects")

    def arm(project, mode):
        live = mode == "live"
        plan, attempts = admit(jobs, folders, project, definition, qualified=live, train_only=not live)
        producer = attempts[0]
        train = terminal(host, producer, "succeeded")
        clocks = {entry["update"]: entry["ns"] for entry in events(host, project, producer, "qualification_checkpoint_clock")}
        assert set(clocks) == set(range(1, 37))
        window, = events(host, project, producer, "qualification_train_window")
        checkpoint = train["checkpoint"]
        manifest = Path(checkpoint["checkpoint_root"]) / checkpoint["pointer"]["manifest_relative_path"]
        final_slots = {
            slot["slot"]: slot["semantic_state_root_sha256"]
            for slot in json.loads(manifest.read_bytes())["content_integrity_digest"]["slots"]
        }
        assert {"model", "optimizer", "prng", "batch_counter"} <= final_slots.keys()
        result = {
            "mode": mode, "producer": producer, "request": train["request"],
            "final_slot_roots": final_slots,
            "checkpoint_clocks_ns": clocks, "whole_driver_ns": window["ns"],
            "steady_ns": clocks[36] - clocks[4], "steady_updates": 32,
        }
        if live:
            evaluation = attempts[1]
            completed = terminal(host, evaluation, "succeeded")
            assert completed["data"]["placement"] == "live"
            points = events(host, project, evaluation, "evaluation_point")
            assert [point["checkpoint"]["completed_updates"] for point in points] == list(range(1, 37))
            hashes = [point["result"]["states"]["content"]["sha256"] for point in points]
            assert len(set(hashes)) == 36
            times = events(host, project, producer, "qualification_point_window")
            assert [point["update"] for point in times] == list(range(1, 37))
            output = jobs.read_evaluation_result(project=project, store=plan.store_path, job_id=evaluation)
            with output.open_states() as dataset:
                assert dataset.data("update").tolist() == list(range(1, 37))
            result |= {
                "evaluation": evaluation, "evaluation_request": completed["request"],
                "point_windows": times, "point_sha256": hashes,
            }
        return result

    measurements = []
    for index, order in enumerate((("control", "live"), ("live", "control"))):
        pair = {mode: arm(f"{scenario}-{index}-{mode}", mode) for mode in order}
        assert pair["live"]["request"] == pair["control"]["request"]
        assert pair["live"]["final_slot_roots"] == pair["control"]["final_slot_roots"]
        measurements.append({
            "order": list(order), **pair,
            "steady_slowdown": pair["live"]["steady_ns"] / pair["control"]["steady_ns"] - 1,
        })
    points_ms = [entry["ns"] / 1e6 for pair in measurements for entry in pair["live"]["point_windows"][1:]]
    points_ms.sort()
    result = {
        "scenario": scenario, "train_batch_size": batch_size, "held_out_trials": trial_count,
        "transitions": task.fixed_steps, "pairs": measurements, "warm_samples": len(points_ms),
        "warm_min_ms": points_ms[0], "warm_mean_ms": mean(points_ms),
        "warm_p95_ms": points_ms[(95 * len(points_ms) + 99) // 100 - 1], "warm_max_ms": points_ms[-1],
    }
    destination = Path(os.environ["FEEDBAX_LIVE_TIMING"])
    (destination / f"{scenario}.json").write_text(json.dumps(result, indent=2) + "\n")
