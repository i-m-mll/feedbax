"""Ordinary folder inputs and read-only custody evidence for the mounted Train test."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project import Project
from feedbax.contracts.project_sweep import model_field_path
from feedbax.application.project_writer import dump_text_project
from feedbax.contracts.project_io import ProjectConfiguration
from feedbax.planner.project_plan import RunKey
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.desktop.release import AppPaths
from feedbax.host import HostClient
from feedbax.host.plan_history import plan_run_key
from feedbax.host.job_completion import verify_bound_training_completion
from feedbax.jobs import JobService
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView
from feedbax.persistence.method_checkpoint import read_method_checkpoint_manifest
from feedbax.application.bundle import new_application_registry_bundle
from feedbax.persistence.canonical_train import BoundStudioTrainingJobInput
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanService
from tests.support.browser_seed_actions import FileSeed, HomeSeed, NonSeed, print_action
from tests.support.managed_train_fixture import managed_train_project


def successor_file():
    return {"kind": "step", "id": "successor", "step": {
        "kind": "train", "label": "Successor", "base": {"step": "train"},
        "inputs": {"checkpoint": {"step": "train", "checkpoint": "last"}},
        "deltas": [{
            "layer_id": "initialize",
            "patches": [
                {
                    "path": "checkpoint_initialization",
                    "op": "add",
                    "value": "initialize_from",
                },
                {"path": "training_config.n_batches", "value": 1200},
            ],
        }],
    }}


def ordinary_files(batches: str, *rates: str, scenario: str = "simple-reaches"):
    example = managed_train_project(batches=int(batches), scenario=scenario)
    step = example.steps["train"].model_dump(mode="json", by_alias=True, exclude_unset=True)
    if rates:
        step["sweep"] = {
            "axes": {"rate": {"values": {
                f"rate-{index}": {"label": rate, "patches": {
                    "settings.optimizer.learning_rate.value": {"value": float(rate)},
                }}
                for index, rate in enumerate(rates)
            }}},
        }
    trial = step["base"]["inline"]["trial_documents"]
    names = (trial[slot]["document"] for slot in ("model", "task", "objective"))
    files = [
        {"kind": "document", "id": name, "document": example.documents[name].model_dump(
            mode="json", by_alias=True, exclude_unset=True,
        )}
        for name in names
    ]
    files.append({"kind": "step", "id": "train", "step": step})
    return files


def hold_campaign_files():
    """The delayed-reaches files in the shape of a hold campaign, with no runs, and every key differing from its label.

    A trial Set holds a plain entry and a family over two indices, each with a Task variable and an attachment
    difference. A Train step samples it with a condition and trains a based Model whose authored input ``level`` only
    one of the two Tasks feeds; an Evaluate step lists the Set as an axis. The Task's validation trials are one family and its
    target draws a ``choice``; the Objective's one term is a family.
    """
    model, task, objective, train = ordinary_files("2", scenario="delayed-reaches")
    model_ports = model["document"]["payload"]["ports"]
    model_ports["inputs"]["target_pos"]["label"] = "Target position"
    model_ports["outputs"]["command"]["label"] = "Command"
    payload = task["document"]["payload"]
    for collection in ("variables", "epochs", "signals"):
        for member in payload[collection].values():
            member["label"] = member["label"].capitalize()
    payload["variables"]["target"]["value"] = "start + polar(0.2 m, choice(0°, 90°, 180°, 270°))"
    payload["validation_trials"] = {"reach": {
        "label": "Reach {direction}, draw {repeat}",
        "over": {"direction": "0..3", "repeat": "0..1"},
        "variables": {"start": "[0, 0]", "target": "polar(0.2 m, direction * 90°)", "catch": False},
        "epoch_steps": {"hold": "7 + repeat", "cue": 10},
    }}
    level_task = {"kind": "document", "id": "level-task", "document": {
        "schema": "feedbax.spec.task_document.v2", "label": "Level task", "base": {"document": task["id"]},
        "differences": {"/signals/level": {"value": {
            "label": "Level", "type": {"base": "real", "shape": [1]},
            "cells": {"all": {"first_epoch": "hold", "last_epoch": "move", "value": "[1]"}},
        }}},
    }}
    objective["document"]["payload"]["terms"] = {"effort": {
        "label": "Command effort", "over": {"k": "0..2"},
        "measure": "sqnorm(outputs.command)", "weight": "(k + 1) / 6", "reduction": "last",
    }}
    heavy = {"kind": "document", "id": "heavy-point-mass", "document": {
        "schema": "feedbax.spec.model.v2", "label": "Heavy point mass", "base": {"document": model["id"]},
        "differences": {
            "/interior/components/plant/fields/mass": {"value": 2},
            "/ports/inputs/level": {"value": {"label": "Level", "type": {"shape": [1]}}},
            "/ports/outputs/level-out": {"value": {"label": "Level out", "type": {"shape": [1]}}},
            "/interior/components/level-gain": {"value": {
                "kind": "leaf", "label": "Level gain", "type": "feedbax.Gain", "fields": {"gain": 1},
            }},
            "/interior/wires/level-in": {"value": {
                "source": {"owner": "inputs", "port": "level"},
                "target": {"owner": "component", "component": "level-gain", "port": "input"},
            }},
            "/interior/wires/level-echo": {"value": {
                "source": {"owner": "component", "component": "level-gain", "port": "output"},
                "target": {"owner": "outputs", "port": "level-out"},
            }},
        },
    }}

    def kick(variable: str, label: str, value: str, onset: str):
        return {
            "task_variables": {variable: {"label": label, "type": {"base": "real", "shape": [2]}, "value": value}},
            "differences": {f"/interior/wires/drive/attachments/{variable}-pulse": {"value": {
                "kind": "intervention", "operation": "add",
                "operands": {"value": {"kind": "task", "variable": variable}},
                "window": {"spans": [{
                    "kind": "interval", "start": {"anchor": {"kind": "trial", "edge": "start"}, "offset": onset},
                    "end": {"offset": "5 st"},
                }]},
            }}},
        }

    kicks = {"kind": "document", "id": "command-kicks", "document": {
        "schema": "feedbax.set_document.v2", "label": "Command kicks", "payload": {"entries": {
            "drift": {"label": "Drift", **kick("bias", "Bias", "[0.1, 0]", "5 st")},
            "pulse": {
                "label": "Command pulse {axis} {sign}",
                "over": {
                    "axis": {"x": "[1, 0]", "y": "[0, 1]"},
                    "sign": {"minus": {"value": -1, "label": "Minus"}, "plus": {"value": 1, "label": "Plus"}},
                },
                **kick("kick", "Kick", "0.5 * sign * axis", "10 st"),
            },
        }},
    }}
    train["step"]["label"] = "Kicked training"
    train["step"]["base"]["inline"]["trial_documents"].update(
        model={"document": heavy["id"]}, task={"document": level_task["id"]},
    )
    train["step"]["set_fields"] = {"kicks": {
        "axis": {"set": kicks["id"]}, "sampling": {"kind": "weighted"}, "condition": "not catch",
    }}
    check = {"kind": "step", "id": "kick-check", "step": {
        "kind": "evaluate", "label": "Kick check",
        "base": {"inline": {
            "evaluation_type": "feedbax.rollout",
            "params": {"seed": 1, "trial_set": {"kind": "every_condition"}},
        }},
        "set_fields": {"kick-bank": {"axis": {"set": kicks["id"]}, "sampling": {"kind": "listed"}}},
        "inputs": {"subject": {"step": "train", "checkpoint": "last"}},
    }}
    return [model, heavy, task, level_task, objective, kicks, train, check]


def _field(component: str, *path: str) -> ModelAddress:
    return ModelAddress.model_validate({
        "composites": [], "element": {"kind": "field", "parent": {"kind": "component", "id": component}, "path": list(path)},
    })


def mass_range_files(start: str, stop: str, count: str):
    """Ordinary files whose Train sweeps the point mass's mass over a range."""
    *documents, train = ordinary_files("2")
    train["step"]["sweep"] = {"axes": {"mass": {
        "path": model_field_path(_field("plant", "fields", "mass")),
        "range": {"start": float(start), "stop": float(stop), "count": int(count)},
    }}}
    return [*documents, train]


def task_link_files(scenario: str, variable: str, component: str, *path: str):
    """Ordinary files whose Task links only ``variable`` to the field at ``path`` of ``component``."""
    files = ordinary_files("2", scenario=scenario)
    for file in files:
        payload = file.get("document", {}).get("payload", {})
        if "variables" in payload:
            payload["links"] = {_field(component, *path).to_authored_text(): variable}
    return files


def seed_variants(home: str, project_id: str):
    *documents, train = ordinary_files("2")
    project = Project.model_validate({
        "documents": {file["id"]: file["document"] for file in documents},
        "steps": {"train": train["step"]},
        "sweep": {"rows": {
            "base": {"label": "Baseline"},
            "compare": {"from": "base", "label": "Comparison", "value": "comparison"},
            "control": {"from": "base", "label": "Control"},
        }},
    })
    return write_project(home, project_id, project)


def write_project(home: str, project_id: str, project: Project) -> str:
    """Write ``project`` as the folder ``project_id`` of the Studio home ``home``."""
    root = AppPaths.resolve(instance_root=Path(home)).web / "projects" / project_id
    for name, data in dump_text_project(
        project, ProjectConfiguration(project=project_id, store="results"),
    ).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return ProjectFolderService(root.parent).open(project_id).snapshot.project_id


def last_train_interval(events, attempt: str):
    return next(
        event["data"]["metric_interval"]
        for event in reversed(events)
        if event["job_id"] == attempt
        and event["kind"] == "progress"
        and "metric_interval" in event["data"]
    )


def train_success(job, store):
    ref = ContentObjectRef.model_validate(job["outcome"]["store_success"])
    receipt = SuccessView(store).read_owned_success(ref).receipt
    assert (receipt.attempt, receipt.request.id) == (job["id"], job["request"])
    return ref, receipt


def observe(
    home: Path, project: str, revision: str, attempt: str, stopped_attempt: str, successor_attempt: str,
):
    paths = AppPaths.resolve(instance_root=home)
    folders = ProjectFolderService(paths.web / "projects")
    host = HostClient(paths.host)
    service = ProjectPlanService(folders, JobService(host, new_application_registry_bundle()))
    train = RunKey("default", "train", "default")
    successor = RunKey("default", "successor", "default")
    view = service.read(project, revision=revision, source_runs=(train, successor))
    plan = view.settled.plan
    rows = {run.key: run for run in plan.runs}
    assert not plan.diagnostics
    snapshot = host.jobs_snapshot(project=project, include_launch=True, start_host=False)
    jobs = {job["id"]: job for job in snapshot["jobs"]}
    assert set(jobs) == {attempt, stopped_attempt, successor_attempt}
    job = jobs[attempt]
    assert (job["id"], job["request"], job["delivery"]) == (attempt, rows[train].request_id, 1)
    launch = BoundStudioTrainingJobInput.model_validate(job["launch"]["payload"])
    assert launch.canonical_request is not None and launch.resume is None
    store = ImmutableArtifactBlobProvider(view.settled.store_path)
    assert launch.manifest_root == view.settled.store_path
    assert store.get_object(launch.canonical_request.request) == plan.requests[launch.request_id]
    assert all(
        store.get_object(ref) == plan.documents[ref.sha256]
        for ref in launch.canonical_request.documents
    )
    completion = job["outcome"]["completion"]
    verify_bound_training_completion(completion)
    assert completion["request_id"] == launch.request_id
    assert completion["producer_job_id"] == attempt
    (joined,) = view.attempts_by_run[train]
    assert joined.job_id == attempt
    assert joined.successful_receipt.model_dump(mode="json") == completion["receipt"]
    child = rows[successor]
    assert child.inputs["checkpoint"].producer == train
    decoded = decode_request(plan.requests[child.request_id], plan.documents.__getitem__)
    (checkpoint,) = decoded.inputs["checkpoint"]
    assert checkpoint.reference.request_id == launch.request_id
    child_job = jobs[successor_attempt]
    assert child_job["request"] == child.request_id
    stopped = jobs[stopped_attempt]
    assert (stopped["state"], child_job["continues"]) == ("stopped", stopped_attempt)
    assert child_job["launch"]["payload"]["resume"] == stopped["checkpoint"]
    (edge,) = child_job["edges"]
    assert edge["satisfied"] and edge["authenticated_completion"]
    assert edge["binding"] == {"dependency_job_id": attempt, "completion": completion}
    assert stopped["edges"] == child_job["edges"]
    child_completion = child_job["outcome"]["completion"]
    child_binding = verify_bound_training_completion(child_completion)
    assert child_completion["producer_job_id"] == successor_attempt
    assert child_completion["request_id"] == child.request_id
    joined_stopped, joined_child = view.attempts_by_run[successor]
    assert joined_child.job_id == successor_attempt
    assert joined_stopped.job_id == stopped_attempt
    assert joined_child.checkpoint_source == joined_stopped.checkpoint_source
    assert joined_child.successful_receipt.model_dump(mode="json") == child_completion["receipt"]
    assert view.history[successor].result.receipt == joined_child.successful_receipt

    predecessor = PublishedTrainingCheckpoint.model_validate(completion["checkpoint"])
    stopped_checkpoint = PublishedTrainingCheckpoint.model_validate(stopped["checkpoint"])
    assert child_binding.spec.checkpoint_initialization == "initialize_from"
    assert child_binding.checkpoint_input is not None
    assert child_binding.checkpoint_input.checkpoint == predecessor
    assert child_binding.resume == stopped_checkpoint
    source_ref, _ = train_success(job, store)
    _, child_receipt = train_success(child_job, store)
    (source,) = child_receipt.inputs["checkpoint"]
    assert source.selection is not None and source.selection.receipt == source_ref
    assert child_receipt.recovery is not None
    assert (child_receipt.recovery.run_id, child_receipt.recovery.manifest_sha256) == (
        stopped_attempt, stopped_checkpoint.pointer.manifest_sha256,
    )

    ref = view.jobs.plan_ref
    assert ref is not None
    admitted_plan = json.loads(store.get_object(ContentObjectRef.model_validate(ref["plan"])))
    assert admitted_plan == {
        "revision": revision,
        "semantic_root": view.settled.semantic_root,
        "requests": {
            plan_run_key(key.variant, key.step, key.row): row.request_id for key, row in rows.items()
        },
    }
    terminal_loss = last_train_interval(
        host.events(projects=(project,), limit=1024)["events"], successor_attempt,
    )
    return {
        "project": project, "revision": revision, "semantic_root": view.settled.semantic_root,
        "request": launch.request_id, "attempt": attempt, "delivery": job["delivery"],
        "completion_updates": completion["checkpoint"]["completed_updates"],
        "receipt": completion["receipt"], "joined_status": joined.status,
        "worker_request_object": launch.canonical_request.request.sha256,
        "successor": {
            "request": child.request_id, "attempt": successor_attempt, "delivery": child_job["delivery"],
            "completion_updates": child_completion["checkpoint"]["completed_updates"],
            "receipt": child_completion["receipt"], "joined_status": joined_child.status,
            "initialization": child_binding.spec.checkpoint_initialization,
            "resumed_from": stopped["checkpoint"]["completed_updates"],
            "terminal_loss": terminal_loss,
        },
        "admitted_plan": admitted_plan,
        "checkpoint_input": checkpoint.reference.model_dump(mode="json"),
    }


def history_files():
    *documents, train = ordinary_files("2")
    sibling = deepcopy(train)
    sibling["id"] = "sibling"
    sibling["step"]["label"] = "Sibling"
    sibling["step"]["base"]["inline"]["seed"] = 91
    return [*documents, train, sibling]


def observe_swept(home: Path, project: str, successors: list[str]):
    host = HostClient(AppPaths.resolve(instance_root=home).host)
    jobs = {job["id"]: job for job in host.jobs_snapshot(project=project, start_host=False)["jobs"]}
    events = host.events(projects=(project,), limit=1024)["events"]
    result = {}
    for attempt in successors:
        job = jobs[attempt]
        (edge,) = job["edges"]
        assert edge["satisfied"] and edge["authenticated_completion"]
        source = edge["binding"]["completion"]
        producer = source["producer_job_id"]
        assert source == jobs[producer]["outcome"]["completion"]
        completion = job["outcome"]["completion"]
        assert (completion["producer_job_id"], completion["request_id"]) == (attempt, job["request"])
        verify_bound_training_completion(source)
        binding = verify_bound_training_completion(completion)
        policy = binding.spec.continuation
        assert policy is not None and binding.checkpoint_input is not None
        predecessor = PublishedTrainingCheckpoint.model_validate(source["checkpoint"])
        assert binding.checkpoint_input.checkpoint == predecessor
        assert policy.source_completed_updates == predecessor.completed_updates
        store = ImmutableArtifactBlobProvider(binding.manifest_root)
        source_ref, _ = train_success(jobs[producer], store)
        _, receipt = train_success(job, store)
        (selected,) = receipt.inputs["checkpoint"]
        assert selected.selection is not None and selected.selection.receipt == source_ref
        checkpoint = PublishedTrainingCheckpoint.model_validate(completion["checkpoint"])
        manifest = read_method_checkpoint_manifest(checkpoint.checkpoint_root, checkpoint.pointer)
        assert manifest.clock.segment_start == predecessor.completed_updates
        interval = last_train_interval(events, attempt)
        result[attempt] = {
            "producer": producer,
            "receipt": completion["receipt"],
            "completed_updates": manifest.clock.completed_updates,
            "source_checkpoint": {
                "completed_updates": predecessor.completed_updates,
                "transaction_id": predecessor.pointer.transaction_id,
                "manifest_sha256": predecessor.pointer.manifest_sha256,
            },
            "schedule_origin": manifest.clock.schedule_origin,
            "initialization": policy.model_dump(mode="json"),
            "last_loss": interval["metrics"]["train_loss"]["last"],
        }
    return result


def observe_history(home: Path, project: str, revision: str):
    paths = AppPaths.resolve(instance_root=home)
    folders = ProjectFolderService(paths.web / "projects")
    host = HostClient(paths.host)
    service = ProjectPlanService(folders, JobService(host, new_application_registry_bundle()))
    view = service.read(project, revision=revision)
    history = view.history[RunKey("default", "train", "default")]
    snapshot = host.jobs_snapshot(project=project, start_host=False)
    jobs = {job["id"]: job for job in snapshot["jobs"]}
    output = {}
    for role, result in (("current", history.result), ("previous", history.previous)):
        if result is None:
            output[role] = None
            continue
        job = jobs[result.job_id]
        completion = job["outcome"]["completion"]
        verify_bound_training_completion(completion)
        assert completion["request_id"] == result.request_id
        assert completion["producer_job_id"] == result.job_id
        assert completion["receipt"] == result.receipt.model_dump(mode="json")
        output[role] = result.model_dump(mode="json")
        assert job["continues"] is None and result.progress is not None
        assert completion["checkpoint"]["completed_updates"] == result.progress.completed_updates
    ref = view.jobs.plan_ref
    assert ref is not None
    store = ImmutableArtifactBlobProvider(view.settled.store_path)
    output["admitted_plan"] = json.loads(store.get_object(ContentObjectRef.model_validate(ref["plan"])))
    return output


def run_saved_cli(home: str, folder_name: str, label: str) -> None:
    paths = AppPaths.resolve(instance_root=Path(home))
    paths.configure_studio(desktop=False)
    project = paths.web / "projects" / folder_name
    os.execv(sys.executable, [sys.executable, "-m", "feedbax.cli", "run", str(project), "train", "--label", label])


ACTIONS = {
    "managed-files": FileSeed(lambda: [*ordinary_files("2"), successor_file()]),
    "ordinary-files": FileSeed(ordinary_files, cases=(("2",), ("2", "0.001", "0.01"))),
    "delayed-reaches-files": FileSeed(lambda: ordinary_files("2", scenario="delayed-reaches")),
    "hold-campaign-files": FileSeed(hold_campaign_files),
    "mass-range-files": FileSeed(mass_range_files, cases=(("1.000123456789", "2.000987654321", "3"),)),
    "task-link-files": FileSeed(task_link_files, cases=(
        ("delayed-reaches", "start", "plant", "reset", "effector", "elsewhere"),
        ("simple-reaches", "start", "nowhere", "state", "effector/position"),
        ("simple-reaches", "ghost", "plant", "state", "effector/position"),
    )),
    "run-saved-cli": NonSeed(run_saved_cli, "executes the CLI to submit a saved Train"),
    "seed-variants": HomeSeed(seed_variants),
    "history-files": FileSeed(history_files),
    "observe": NonSeed(lambda home, *args: observe(Path(home), *args), "reads saved Train attempts"),
    "observe-history": NonSeed(lambda home, *args: observe_history(Path(home), *args), "reads Train attempt history"),
    "observe-swept": NonSeed(lambda home, project, *args: observe_swept(Path(home), project, args),
                             "reads swept Train attempts"),
}


if __name__ == "__main__":
    print_action(ACTIONS)
