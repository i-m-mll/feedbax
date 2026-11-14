"""Compare one saved Evaluate request across real Host and the Train driver seam."""

from dataclasses import replace
import json
import os
from pathlib import Path

import pytest

from feedbax.contracts.project import InlineBase, Project
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.task_document import TaskDocument
from feedbax.jobs import JobService
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.training.validation_placement import ValidationBudget, place_validation
from feedbax.web.services.project_folder import ProjectFolderService
from tests.support import evaluate_placement_probe as probe
from tests.support.managed_train_fixture import managed_train_project
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_local_job_host import host as host_fixture, job_row, wait_until

host = host_fixture


@pytest.mark.parametrize("scenario,batch_size,trial_count", [
    ("simple-reaches", 1, 1),
    ("delayed-reaches", 4, 28),
])
def test_same_evaluate_products_inside_driver_and_ordinary_job(host, tmp_path, monkeypatch, scenario, batch_size, trial_count):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    registry = new_application_registry_bundle()
    folders = ProjectFolderService(tmp_path / "projects")
    project = managed_train_project(batches=4, scenario=scenario)
    train = project.steps["train"].base
    assert isinstance(train, InlineBase)
    train.inline["training_config"].update(batch_size=batch_size, snapshot_interval=2)
    task_id = train.inline["trial_documents"]["task"]["document"]
    task = TaskDocument.model_validate(project.documents[task_id].payload)
    project = Project.model_validate(project.model_dump(mode="json", by_alias=True, exclude_unset=True) | {
        "steps": {
            "train": project.steps["train"].model_dump(mode="json", by_alias=True, exclude_unset=True),
            "check": {
                "kind": "evaluate",
                "inputs": {"subject": {"step": "train", "checkpoint": [{"at": 2}]}},
                "base": {"inline": {
                    "evaluation_type": "feedbax.rollout",
                    "params": {
                        "seed": 731, "trial_policy": "paired", "variation": {},
                        "trial_set": {"kind": "conditions", "ids": list(task.condition_ids[:trial_count])},
                    },
                }},
            },
        },
    })
    _save_project(folders, scenario, project)
    plan = folders.read_plan(scenario, revision=None)
    assert not plan.plan.diagnostics, plan.plan.diagnostics
    runs = {run.key.step: run for run in plan.plan.runs}
    train_key = RunKey("default", "train", "default")
    evaluate = runs["check"]
    assert len(task.condition_ids) >= trial_count
    store = ImmutableArtifactBlobProvider(plan.store_path)
    request = store.store_rooted_object(plan.plan.requests[evaluate.request_id], namespace="qualification-evaluate")
    jobs = JobService(host, registry)
    template = jobs._training_worker
    env = template.env | {
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        "EVALUATE_PLACEMENT_REQUEST": request.model_dump_json(),
        "EVALUATE_PLACEMENT_STORE": str(plan.store_path),
        "EVALUATE_PLACEMENT_ID": evaluate.request_id,
    }

    def settled(item):
        assert item.disposition == "admitted", item
        result = wait_until(lambda: (
            row if (row := job_row(host, item.job_id))["outcome"] is not None else None
        ), timeout=25)
        assert result["state"] == "succeeded", result["outcome"]
        return result

    def events(attempt, kind):
        cursor = 0
        found = []
        while True:
            page = host.events(cursor, projects=(scenario, f"{scenario}-control"))
            found.extend(
                event["data"] for event in page["events"]
                if event["job_id"] == attempt and event["kind"] == kind
            )
            cursor = page["cursor"]
            if cursor >= page["head"]:
                return found

    def train_arm(mode, producer_plan):
        jobs._training_worker = replace(
            template, env=env | {"EVALUATE_PLACEMENT": mode}, warm_entrypoint=f"{probe.__name__}:main",
        )
        (item,) = jobs.admit_plan(folders, producer_plan, run_keys=(train_key,), token=mode).items
        return settled(item)

    host.policy(compute=1)
    control_project = f"{scenario}-control"
    _save_project(folders, control_project, project)
    control = train_arm("disabled", folders.read_plan(control_project, revision=None))
    live = train_arm("live", plan)
    jobs._training_worker = template
    (item,) = jobs.admit_plan(folders, plan, run_keys=(evaluate.key,), token="ordinary").items
    assert control["request"] == live["request"] == runs["train"].request_id
    ordinary = settled(item)
    binding = ordinary["edges"][0]["binding"]["checkpoint_selection"]
    (member,) = binding["members"]
    assert member["producer_job_id"] == live["id"]
    result = jobs.read_evaluation_result(project=scenario, store=plan.store_path, job_id=ordinary["id"])
    artifact = result._manifest().artifacts[0]
    points = events(live["id"], "qualification_evaluate_point")
    assert len(points) == 3
    assert all(point["request"] == ordinary["request"] == evaluate.request_id for point in points)
    assert all(point["checkpoint"] == member["checkpoint"] for point in points)
    assert {point["ref"]["sha256"] for point in points} == {artifact.sha256}
    timing, = events(live["id"], "qualification_evaluate_timing")
    timing["driver_ms"] = {
        "disabled": events(control["id"], "qualification_driver_timing")[0]["window_ms"],
        "live_three_measurements": events(live["id"], "qualification_driver_timing")[0]["window_ms"],
    }
    timing["scenario"] = scenario
    timing["request"] = ordinary["request"]
    timing["product"] = artifact.sha256
    if destination := os.environ.get("FEEDBAX_PLACEMENT_EVIDENCE"):
        (Path(destination) / f"{scenario}.json").write_text(json.dumps(timing, indent=2) + "\n")


def test_placement_respects_readiness_and_full_product_budget():
    budget = ValidationBudget(durable_point_p95_ms=4, steady_update_ms=10, max_point_ms=10, max_update_fraction=0.10)

    def place(updates=(7, 14), **facts):
        return place_validation(updates, **({
            "producer_started": False, "live_products_supported": True, "budget": budget,
        } | facts))

    assert place().location == "live"
    assert place((1, 7)).reason == "over_budget"
    assert place(producer_started=True).reason == "late"
    assert place(live_products_supported=False).reason == "unqualified"
    assert place(budget=None).reason == "unmeasured"
