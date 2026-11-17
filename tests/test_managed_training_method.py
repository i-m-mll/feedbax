"""Run a discovered method through saved plans, Host checkpoint stop, and collected Resume."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import signal

import numpy as np

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.checkpoints import CheckpointLatestPointer
from feedbax.contracts.effective_success import success_content_ref
from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.host import HostClient
from feedbax.jobs import JobService
from feedbax.persistence.array_container import decode_array_container
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView
from feedbax.persistence.method_checkpoint import read_method_checkpoint_manifest
from feedbax.persistence.train_job import read_train_job
from feedbax.plugins.composition import load_application
from feedbax.project_folder import ProjectFolder
from tests.support import external_method_witness
from tests.support.installed_package import install_test_package
from tests.support.managed_train_fixture import managed_evaluation_step, managed_train_project
from tests.test_local_job_host import host, job_row, wait_until  # noqa: F401 -- pytest fixture


_TOTAL = 4
_REPLICATES = 2


def _install_methods(directory, monkeypatch):
    project = directory / "project"
    package = project / "managed_method_probe"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    shutil.copyfile(external_method_witness.__file__, package / "methods.py")
    (project / "pyproject.toml").write_text('''\
[project]
name = "managed-method-probe"
version = "0.1"

[project.entry-points."feedbax.plugins"]
probe = "managed_method_probe"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["managed_method_probe"]
''')
    site = install_test_package(project, directory)
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, (str(site), os.environ.get("PYTHONPATH")))))


def _write_project(folders, project, store):
    raw = managed_train_project(batches=_TOTAL).model_dump(mode="json", by_alias=True, exclude_unset=True)
    authored = raw["steps"]["train"]["base"]["inline"]
    authored.update(
        method="managed_method_probe.adversarial",
        settings={"learning_rate": 0.03, "limit": 0.25},
        seed=19,
    )
    authored["training_config"].update(n_batches=_TOTAL, batch_size=2, snapshot_interval=2, replicates=_REPLICATES)
    task = raw["documents"][authored["trial_documents"]["task"]["document"]]["payload"]
    task["epochs"][0]["length"]["value"] = 6
    receiver = deepcopy(authored)
    receiver.update(
        method="managed_method_probe.receiver", settings={"initial_count": 41},
        seed=73, checkpoint_initialization="initialize_from",
    )
    receiver["training_config"]["n_batches"] = 1
    raw["steps"]["receiver"] = {
        "kind": "train", "base": {"inline": receiver},
        "inputs": {"checkpoint": {"step": "train", "checkpoint": "last"}},
    }
    raw["steps"]["evaluate"] = managed_evaluation_step("train")
    for name, data in dump_text_project(
        Project.model_validate(raw), ProjectConfiguration(project=project, store=str(store)),
    ).items():
        path = folders.root / project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    plan = folders.read_plan(project, revision=None)
    assert not plan.plan.diagnostics, plan.plan.diagnostics
    return plan


def _launch(jobs, folders, plan, step, token):
    key = next(row.key for row in plan.plan.runs if row.key.step == step)
    (item,) = jobs.admit_plan(folders, plan, run_keys=(key,), token=token).items
    assert item.disposition == "admitted", item.reason
    assert item.job_id is not None
    return item.job_id


def _terminal(client, job_id, expected):
    def read():
        row = job_row(client, job_id)
        return row if row and row["outcome"] is not None else None

    row = wait_until(read, timeout=28)
    assert row["state"] == expected, row["outcome"]
    return row


def _checkpoint(row):
    checkpoint = row["outcome"]["checkpoint"]
    pointer = CheckpointLatestPointer.model_validate(checkpoint["pointer"])
    return read_method_checkpoint_manifest(Path(checkpoint["checkpoint_root"]), pointer)


def _extra(store, checkpoint):
    return decode_array_container(
        store.get_object(checkpoint.extra.object),
        expected_descriptor_sha256=checkpoint.extra.descriptor_sha256,
    ).tensors


def _success(store, row):
    ref = ContentObjectRef.model_validate(row["outcome"]["store_success"])
    return SuccessView(store).read_owned_success(ref)


def _retained_metrics(store, success):
    (product,) = success.receipt.products["metrics"]
    return store.get_object(success_content_ref(product.content))


def _await_stop_gate(venue, job_id, gate):
    def read():
        row = job_row(venue, job_id)
        assert row["outcome"] is None, row["outcome"]
        return gate.with_suffix(".ready").exists()

    wait_until(read, timeout=28)


def test_external_adversarial_method_resumes_and_supplies_its_model(request, tmp_path, monkeypatch):
    _install_methods(tmp_path / "installed", monkeypatch)
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    gate = tmp_path / "stop.armed"
    monkeypatch.setenv("FEEDBAX_TEST_ADVERSARIAL_STOP", str(gate))
    local = request.getfixturevalue("host")
    registries = load_application().bundle
    folders = ProjectFolder(
        tmp_path / "projects", components=registries.components, python_tasks=registries.python_tasks,
        operations=registries.operations, functions=registries.expression_functions,
        training_methods=registries.training_methods,
    )
    project = "adversarial"
    plan = _write_project(folders, project, tmp_path / "source-results")
    source = ImmutableArtifactBlobProvider(plan.store_path)
    venue = HostClient(tmp_path / "venue-host")
    venue_jobs = JobService(venue, registries)
    try:
        venue.policy(compute=1)
        uninterrupted_id = _launch(venue_jobs, folders, plan, "train", "uninterrupted")
        uninterrupted = _terminal(venue, uninterrupted_id, "succeeded")
        gate.touch()
        stopped_id = _launch(venue_jobs, folders, plan, "train", "stop-and-collect")
        _await_stop_gate(venue, stopped_id, gate)
        (stop,) = venue_jobs.stop((stopped_id,), "checkpoint")
        assert stop.error is None
        stopped = _terminal(venue, stopped_id, "stopped")
        assert 0 < stopped["outcome"]["checkpoint"]["completed_updates"] < _TOTAL
    finally:
        if venue.address.exists():
            live = [row["id"] for row in venue.jobs() if row["outcome"] is None]
            if live:
                venue.stop(live, "force")
                wait_until(lambda: all(row["outcome"] is not None for row in venue.jobs()))
            os.kill(venue.ping()["pid"], signal.SIGTERM)
            wait_until(lambda: not venue.address.exists())

    control = _checkpoint(uninterrupted)
    uninterrupted_metrics = _retained_metrics(source, _success(source, uninterrupted))
    local_plan = _write_project(folders, project, tmp_path / "local-results")
    destination = ImmutableArtifactBlobProvider(local_plan.store_path)
    local_jobs = JobService(local, registries)
    collected = local_jobs.collect_train_attempt(source, stopped, project=plan.project_id, store=destination.root)
    assert collected.can_resume and collected.job_id == stopped_id
    stopped_entry = read_train_job(destination, stopped_id)
    assert stopped_entry.metrics is not None
    prefix_metrics = destination.get_object(stopped_entry.metrics)
    shutil.rmtree(source.root)
    attempt = local_jobs.resume(stopped_id, "resume-collected")
    assert attempt.continues == stopped_id and attempt.job_id != stopped_id
    resumed = _terminal(local, attempt.job_id, "succeeded")
    assert resumed["request"] == uninterrupted["request"]
    restored = _checkpoint(resumed)
    assert restored.clock == control.clock and restored.clock.completed_updates == _TOTAL
    assert restored.model.tree == control.model.tree
    assert restored.extra.object == control.extra.object

    resumed_success = _success(destination, resumed)
    assert resumed_success.receipt.recovery.run_id == stopped_id
    assert uninterrupted_metrics == prefix_metrics + _retained_metrics(destination, resumed_success)
    history = [json.loads(line) for line in uninterrupted_metrics.splitlines()]
    assert [row["update"] for row in history] == list(range(1, _TOTAL + 1))
    initial_prices = np.asarray(history[0]["metrics"]["price"])
    assert initial_prices.shape == (_REPLICATES,) and initial_prices[0] != initial_prices[1]
    assert np.all(np.asarray(history[-1]["metrics"]["price"]) > initial_prices)

    events = local.events(projects=(local_plan.project_id,))["events"]
    live = [
        event
        for event in events
        if event["job_id"] == attempt.job_id and event["kind"] == "progress"
        and "losses" in event["data"].get("metrics", {})
    ]
    assert live
    last_live = live[-1]["data"]
    losses = np.asarray(last_live["metrics"]["losses"])
    assert losses.shape == (_REPLICATES, 2)
    np.testing.assert_array_equal(losses, history[last_live["completed_updates"] - 1]["metrics"]["losses"])
    assert np.any(losses[:, 1] > losses[:, 0])
    terminal = next(event for event in events if event["job_id"] == attempt.job_id and event["kind"] == "terminal")
    assert live[-1]["cursor"] < terminal["cursor"]

    gate.with_suffix(".model-only").touch()
    evaluator_id = _launch(local_jobs, folders, local_plan, "evaluate", "evaluate")
    evaluation = _terminal(local, evaluator_id, "succeeded")
    result = local_jobs.read_evaluation_result(
        project=local_plan.project_id, store=local_plan.store_path, job_id=evaluator_id,
    )
    with result.open_states(verify=True) as dataset:
        assert dataset.data("checkpoint").tolist() == [_TOTAL]
        assert dataset.data("replicate").tolist() == ["0", "1"]
    (evaluated_subject,) = _success(destination, evaluation).receipt.inputs["subject"]
    assert evaluated_subject.evaluation_subject.terminal_receipt == resumed_success.ref

    receiver_id = _launch(local_jobs, folders, local_plan, "receiver", "receiver")
    received = _terminal(local, receiver_id, "succeeded")
    receiver_checkpoint = _checkpoint(received)
    assert receiver_checkpoint.model.tree == restored.model.tree
    (counts,) = _extra(destination, receiver_checkpoint).values()
    np.testing.assert_array_equal(counts, [42, 42])
    (source_checkpoint,) = _success(destination, received).receipt.inputs["checkpoint"]
    assert source_checkpoint.selection.receipt == resumed_success.ref
