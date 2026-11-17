"""Reconstructed graphs, rollout trajectories, and task inputs in saved Analyze jobs."""

import asyncio
from datetime import timedelta
import json
import os
from pathlib import Path

import numpy as np
import pytest

from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID
from feedbax.contracts.project import Project
from feedbax.contracts.project_plan import ElementKey, RunKey
from feedbax.contracts.rollout import ROLLOUT_TYPE
from feedbax.contracts.task_document import PYTHON_TASK_SCHEMA_ID, TASK_DOCUMENT_SCHEMA_ID
from feedbax.contracts.jobs import JobStatus
from feedbax.jobs import JobService, plan_attempt_key
from feedbax.light_jobs import dataset_member
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import open_stored_dataset
from feedbax.persistence.effective_success import SuccessView
from feedbax.plugins.composition import compose_application
from feedbax.validity.environment import installed_versions
from feedbax.validity.source import plugin_packages
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import read_settled_project_plan
from tests.support import derivative_witness as witness
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_local_job_host import host as host_fixture, restart, wait_until

host = host_fixture
_CAPTURE = Path(__file__).parent / "fixtures" / "derivative_fixed_points.json"
_CALIBRATION_PLUGIN = "feedbax_witness_calibration"


def _port(name, width):
    return {"id": name, "name": name, "type": {"base": "real", "shape": [width]}}


def _leaf(name, owner, **fields):
    return {"kind": "leaf", "name": name, "type": owner, "fields": fields}


def _input(name):
    return {"owner": "inputs", "port": name}


def _output(name):
    return {"owner": "outputs", "port": name}


def _component(name, port):
    return {"owner": "component", "component": name, "port": port}


def _wire(source, target):
    return {"source": source, "target": target}


def _model(inputs, outputs, components, wires):
    return {
        "step_seconds": 0.01,
        "ports": {"inputs": inputs, "outputs": outputs},
        "interior": {"components": components, "wires": wires},
    }


def _fixed_point_model():
    return _model(
        [_port("feedback", 3), _port("memory", 4)],
        [_port("hidden", 4), _port("command", 2), _port("start", 4)],
        {
            "cell": _leaf("Cell", f"{witness.PACKAGE}.PinnedGRU", input_size=3, hidden_size=4),
            "readout": _leaf("Readout", f"{witness.PACKAGE}.PinnedReadout", input_size=4, output_size=2),
            "encoder": _leaf("Encoder", "feedbax.Linear", input_size=3, output_size=4, initialization="zeros"),
        },
        {
            "drive": _wire(_input("feedback"), _component("cell", "input")),
            "context": _wire(_input("feedback"), _component("encoder", "input")),
            "carry": _wire(_input("memory"), _component("cell", "hidden")),
            "decode": _wire(_component("cell", "hidden"), _component("readout", "input")),
            "hidden": _wire(_component("cell", "hidden"), _output("hidden")),
            "command": _wire(_component("readout", "output"), _output("command")),
            "start": _wire(_component("encoder", "output"), _output("start")),
        },
    )


def _feedback_model():
    return _model(
        [_port("drive", 1)], [_port("value", 1)],
        {"feedback": _leaf("Feedback", f"{witness.PACKAGE}.QuadraticFeedback")},
        {
            "input": _wire(_input("drive"), _component("feedback", "input")),
            "output": _wire(_component("feedback", "output"), _output("value")),
        },
    )


def _literal(value):
    return {"kind": "literal", "value": value}


def _variable(name, value):
    return {"id": name, "name": name, "type": {"base": "real"}, "value": _literal(value)}


def _vector_signal(name, terms):
    return {
        "id": name, "name": name, "type": {"base": "real", "shape": [len(terms)]},
        "cells": [{
            "first_epoch": "rollout", "last_epoch": "rollout",
            "value": {"kind": "call", "function": "vector", "args": terms},
        }],
    }


def _conditioning_task():
    zero = _literal(0.0)
    return {
        "id": "conditioning", "variables": [_variable("conditioning", 0.0)],
        "epochs": [{"id": "rollout", "name": "Rollout"}], "fixed_steps": 2,
        "signals": [
            _vector_signal("feedback", [zero, zero, {"kind": "name", "reference": "conditioning"}]),
            _vector_signal("memory", [zero] * 4),
        ],
        "validation_trials": [{"variables": {"conditioning": level}} for level in (0.0, 0.5, 1.0)],
    }


def _initial(document):
    return {"kind": "initial_model", "source": {"document": document}}


def _task_binding(document):
    return {"kind": "task", "source": {"document": document}}


def _operation(name, **inputs):
    return {"operation": f"{witness.PACKAGE}.{name}", "inputs": inputs}


def _analyze(document, **inputs):
    return {"kind": "analyze", "base": {"document": document}, "inputs": inputs}


def _project():
    feedback_task = {
        "type": f"{witness.PACKAGE}.DriveTask",
        "params": {"nominal": 2.0, "disturbance_matrix": [[3.0]]},
    }
    fixed_analysis = {
        "inputs": {"model": {"kind": "model"}, "task": {"kind": "task"}},
        "operations": {
            "fixed": _operation("fixed_points", model={"input": "model"}, task={"input": "task"}),
        },
    }
    gain_analysis = {
        "inputs": {"model": {"kind": "model"}, "task": {"kind": "task"}, "states": {"kind": "dataset"}},
        "operations": {
            "linearise": _operation("linearise", model={"input": "model"}, states={"input": "states"}),
            "assemble": _operation(
                "assemble", jacobians={"element": "linearise", "product": "jacobians"}, task={"input": "task"},
            ),
            "gain": _operation("power_iteration", system={"element": "assemble", "product": "system"}),
        },
    }
    return Project.model_validate({
        "documents": {
            name: {"schema": schema, "payload": payload}
            for name, schema, payload in (
                ("fixed-model", MODEL_DOCUMENT_SCHEMA_ID, _fixed_point_model()),
                ("feedback-model", MODEL_DOCUMENT_SCHEMA_ID, _feedback_model()),
                ("fixed-task", TASK_DOCUMENT_SCHEMA_ID, _conditioning_task()),
                ("feedback-task", PYTHON_TASK_SCHEMA_ID, feedback_task),
                ("fixed-analysis", "feedbax.analysis_document.v1", fixed_analysis),
                ("gain-analysis", "feedbax.analysis_document.v1", gain_analysis),
            )
        },
        "steps": {
            "rollout": {
                "kind": "evaluate",
                "base": {"inline": {"evaluation_type": ROLLOUT_TYPE, "params": {
                    "seed": 17, "trial_policy": "paired", "variation": {},
                    "trial_set": {"kind": "sample", "count": 1},
                }}},
                "inputs": {"subject": _initial("feedback-model"), "task": _task_binding("feedback-task")},
            },
            "fixed": _analyze("fixed-analysis", model=_initial("fixed-model"), task=_task_binding("fixed-task")),
            "gain": _analyze(
                "gain-analysis", model=_initial("feedback-model"), task=_task_binding("feedback-task"),
                states={"step": "rollout"},
            ),
        },
    })


def _install_plugin(root, package, distribution, source):
    package_root = root / package
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_bytes(source)
    metadata = root / f"{package}-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(f"Name: {distribution}\nVersion: 1.0\n")
    (metadata / "entry_points.txt").write_text(f"[feedbax.plugins]\n{package} = {package}\n")


def _install_external_package(root, monkeypatch):
    _install_plugin(root, _CALIBRATION_PLUGIN, "feedbax-witness-calibration", b"GAIN_SCALE = 1\n")
    source = Path(witness.__file__).read_bytes().replace(
        b"from feedbax.analysis.eig import eig",
        b"from feedbax_witness_calibration import GAIN_SCALE\n"
        b"from feedbax.analysis.eig import eig",
    ).replace(b"jnp.linalg.norm(forward(unit_vector))", b"GAIN_SCALE * jnp.linalg.norm(forward(unit_vector))")
    _install_plugin(root, witness.PACKAGE, "feedbax-witness", source)
    monkeypatch.syspath_prepend(str(root))
    pythonpaths = (str(root), str(Path.cwd()), os.environ.get("PYTHONPATH"))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(path for path in pythonpaths if path))


def _assert_fixed_point_products(dataset):
    # Captured at rlrmp2 50993e39 and Feedbax 6e3491e2, per proof commit 61f3bccb.
    reference = json.loads(_CAPTURE.read_bytes())
    rows, summary = reference["results"], reference["summary"]
    replicated = {
        **{name: value for name, value in rows.items() if name != "spectrum"},
        **{f"spectrum.{name}": value for name, value in rows["spectrum"].items()},
        **{f"replica_summary.{name}": value for name, value in summary["per_replica"].items()},
    }
    single = {
        **{
            f"summary.{metric}.{stat}": value
            for metric, stats in summary.items() if metric != "per_replica"
            for stat, value in stats.items()
        },
        "candidate_count": reference["candidate_count"],
    }
    assert dataset.descriptor.attrs["candidate_strategy"] == reference["candidate_strategy"]
    assert list(map(str, dataset.data("replicate"))) == list(map(str, reference["replica_slots"]))
    np.testing.assert_array_equal(dataset.data("conditioning_level"), reference["conditioning_levels"])
    assert dataset.descriptor.variables.keys() == replicated.keys() | single.keys()
    for name in replicated.keys() | single.keys():
        value = dataset.data(name)
        expected = replicated[name] if name in replicated else single[name]
        if name in single:
            value = value[0]
        np.testing.assert_allclose(value, expected, rtol=1e-6, atol=1e-8, err_msg=name)


def test_saved_derivative_operations_use_exact_model_task_and_rollout(host, tmp_path, monkeypatch):
    plugin_packages()
    _install_external_package(tmp_path / "installed", monkeypatch)
    restart(host)
    host.policy(compute=1, light=1)
    registry = asyncio.run(compose_application()).bundle
    folders = ProjectFolderService(
        tmp_path / "projects", components=registry.components, operations=registry.operations,
        functions=registry.expression_functions, python_tasks=registry.python_tasks,
    )
    project = "derivatives"
    _save_project(folders, project, _project())
    review = folders.read_plan(project, revision=None)
    assert not review.plan.diagnostics, review.plan.diagnostics
    jobs = JobService(host, registry, quick_duration=timedelta(seconds=60))
    selected = tuple(RunKey("default", step, "default") for step in ("rollout", "fixed", "gain"))
    launch = jobs.admit_plan(folders, review, run_keys=selected, token="derivative-witness")
    assert len(launch.items) == 5, launch
    job_ids = {item.job_id for item in launch.items}

    def finished():
        rows = [row for row in host.jobs(project=review.project_id) if row["id"] in job_ids]
        terminal = [row for row in rows if row["outcome"] is not None]
        if any(row["state"] != "succeeded" for row in terminal) or len(terminal) == len(job_ids):
            return terminal

    rows = wait_until(finished, timeout=45)
    assert all(row["state"] == "succeeded" for row in rows), {
        row["request"]: row["outcome"] for row in rows if row["state"] != "succeeded"
    }
    by_request = {row["request"]: row for row in rows}

    def identify(name, key):
        return review.plan.elements[ElementKey(key, name)].request_id

    fixed = identify("fixed", selected[1])
    linearise = identify("linearise", selected[2])
    assemble = identify("assemble", selected[2])
    gain = identify("gain", selected[2])
    assert [by_request[ref]["slot"] for ref in (fixed, linearise, assemble, gain)] == [
        "compute", "compute", "light", "compute",
    ]

    store = ImmutableArtifactBlobProvider(review.store_path)
    view = SuccessView(store)

    def results(request, role):
        success = view.authored(request)
        provider = success.receipt.provider
        assert (provider.distribution, provider.version) == ("feedbax-witness", "1.0")
        (product,) = success.receipt.products[role]
        return open_stored_dataset(store, dataset_member(product))

    with results(fixed, "roots") as roots:
        accepted = roots.data("accepted")
        np.testing.assert_array_equal(accepted.sum(axis=-1), [[1, 1, 1]])
        assert np.all(roots.data("losses")[accepted] < 1e-6)
        assert np.all(roots.data("spectrum.magnitude")[accepted] < 1.0)
        _assert_fixed_point_products(roots)
    with results(linearise, "jacobians") as jacobians:
        np.testing.assert_allclose(jacobians.data("feedback").ravel(), [0.0, 2.0])
        np.testing.assert_allclose(jacobians.data("actuation").ravel(), [1.0, 1.0])
    with results(gain, "gain") as dataset:
        estimate = dataset.data("gain")
        assert estimate.dtype == np.dtype("float64")
        assert estimate.item() == pytest.approx(3.0 * (1 + np.sqrt(2)), rel=1e-10)

    gain_key = ElementKey(selected[2], "gain")

    def retained():
        return read_settled_project_plan(review, jobs).light[gain_key]

    initial = retained()
    assert initial.status is JobStatus.SUCCEEDED
    assert initial.result.validity.state == "current"
    calibration = tmp_path / "installed" / _CALIBRATION_PLUGIN / "__init__.py"
    baseline_calibration = calibration.read_bytes()
    calibration.write_text("GAIN_SCALE = 2\n")
    edited = retained().result.validity
    assert edited.state == "stale"
    assert {
        (change.path.scope, change.path.path, change.status)
        for reason in edited.reasons if reason.kind == "code"
        for change in reason.differences
    } == {("plugin:feedbax-witness-calibration", "__init__.py", "changed")}
    calibration.write_bytes(baseline_calibration)
    restored = retained()
    assert restored.status is JobStatus.SUCCEEDED
    assert restored.result.receipt == initial.result.receipt
    source = tmp_path / "installed" / witness.PACKAGE / "__init__.py"
    original = source.read_text()
    body = "    def forward(disturbance):"
    assert body in original
    source.write_text(original.replace(body, "    scale = 2\n" + body, 1))
    stale = retained()
    assert stale.status is JobStatus.STALE
    assert stale.result.receipt == initial.result.receipt
    (reason,) = stale.result.validity.reasons
    assert reason.kind == "code"
    assert any("power_iteration" in change.functions for change in reason.differences)

    judgement = jobs.result_validity_action(
        review, gain, stale.result.validity.receipt,
        judgement=(reason.differences, "Scientist", "Reviewed an unused local."),
    )
    assert judgement.state == "current"
    assert retained().result.receipt == initial.result.receipt
    source.write_text(original.replace(body, "    scale = 3\n" + body, 1))
    assert retained().status is JobStatus.STALE
    assert set(row["id"] for row in host.jobs(project=review.project_id)) == job_ids

    source.write_text(original)
    assert retained().status is JobStatus.SUCCEEDED
    versions = installed_versions() | {"numpy": "0.0"}
    lock = review.project_root / "uv.lock"
    lock.write_text("".join(
        f'[[package]]\nname = "{name}"\nversion = "{version}"\n'
        for name, version in versions.items()
    ))
    observed = retained().result.validity
    assert observed.state == "current"
    assert observed.environment.check == "pending"
    jobs.result_validity_action(review, gain, observed.receipt)

    def completed_check():
        return next((row for row in host.jobs(project=review.project_id) if (
            row["request"].startswith("validity-check:") and row["outcome"] is not None
        )), None)

    check = wait_until(completed_check, timeout=15)
    assert "store_success" not in check["outcome"]
    decided = jobs.result_validity_action(review, gain, observed.receipt)
    assert decided.state == "current" and decided.environment.check == "not_checked"
    assert "numpy" in decided.environment.detail
    assert set(row["id"] for row in host.jobs(project=review.project_id)) == job_ids | {check["id"]}

    lock.unlink()
    settings = registry.operations.resolve(f"{witness.PACKAGE}.power_iteration").value.parameter_model

    def unexpected_settings_validation(*args, **kwargs):
        pytest.fail("Unchanged Run executed the plugin's settings validation.")

    with monkeypatch.context() as unchanged:
        unchanged.setattr(settings, "model_validate", unexpected_settings_validation)
        acknowledged = folders.read_plan(project, revision=review.revision)
        unchanged_run = jobs.submit_plan(
            folders, acknowledged, run_keys=(selected[2],), token="reuse-unchanged-gain", reuse_completed=True,
        )
    assert {item.disposition for item in unchanged_run.items} == {"reused"}

    old_system = view.authored(assemble).receipt.products["system"]
    old_gain = view.authored(gain)
    retained_id = old_gain.receipt.attempt
    data_file = review.project_root / "assemble-calibration.txt"
    data_file.write_text("first calibration")
    marker = "    ((actuation,),) = task.parameters.disturbance_matrix"
    assert marker in original
    same_output = original.replace(
        marker, f"    with open({str(data_file)!r}) as calibration:\n        calibration.read()\n{marker}", 1,
    )
    source.write_text(same_output)

    assemble_key = ElementKey(selected[2], "assemble")
    changed_assemble = read_settled_project_plan(review, jobs).light[assemble_key].result
    assert changed_assemble.validity.state == "stale"
    assert any(
        "assemble" in difference.functions
        for reason in changed_assemble.validity.reasons if reason.kind == "code"
        for difference in reason.differences
    )
    assert retained().status is JobStatus.SUCCEEDED

    def run_together(token):
        submission = jobs.submit_plan(
            folders, review, run_keys=(selected[2],), token=token, reuse_completed=True,
        )
        items = {item.run_key: item for item in submission.items}
        producer = items[plan_attempt_key(assemble_key)]
        assert producer.disposition == "admitted"
        assert items[plan_attempt_key(gain_key)].disposition != "rejected"

        def settled_jobs():
            rows = [row for row in host.jobs(project=review.project_id) if row["request"] in (assemble, gain)]
            if all(row["state"] not in {"queued", "assigned", "running", "stopping"} for row in rows):
                return rows

        rows = wait_until(settled_jobs, timeout=20)
        rebuilt = next(row for row in rows if row["id"] == producer.job_id)
        assert rebuilt["state"] == "succeeded", rebuilt["outcome"]
        return producer.job_id, [row for row in rows if row["request"] == gain]

    producer_id, gain_attempts = run_together("refresh-assemble-same-output")
    equal = SuccessView(store)
    unchanged_system = equal.authored(assemble).receipt
    assert unchanged_system.attempt == producer_id
    assert unchanged_system.products["system"] == old_system
    assert {row["id"] for row in gain_attempts if row["delivery"]} == {retained_id}
    assert {row["id"] for row in gain_attempts if equal.attempt_ref(row["id"])} == {retained_id}
    assert equal.authored_ref(gain) == old_gain.ref
    still_current = retained()
    assert still_current.status is JobStatus.SUCCEEDED
    assert still_current.result.job_id == retained_id
    assert still_current.result.receipt == initial.result.receipt

    source.write_text(same_output.replace(marker, f"{marker}\n    actuation *= 2", 1))
    assert read_settled_project_plan(review, jobs).light[assemble_key].status is JobStatus.STALE
    assert retained().status is JobStatus.SUCCEEDED
    producer_id, gain_attempts = run_together("refresh-assemble-changed-output")
    changed = SuccessView(store)
    changed_system = changed.authored(assemble).receipt
    rebuilt_gain = changed.authored(gain).receipt
    assert changed_system.attempt == producer_id
    assert changed_system.products["system"] != unchanged_system.products["system"]
    assert rebuilt_gain.attempt != retained_id
    assert {row["id"] for row in gain_attempts if row["delivery"]} == {retained_id, rebuilt_gain.attempt}
    current = retained()
    assert current.status is JobStatus.SUCCEEDED
    assert current.result.job_id == rebuilt_gain.attempt
    (product,) = rebuilt_gain.products["gain"]
    with open_stored_dataset(store, dataset_member(product)) as dataset:
        assert dataset.data("gain").item() == pytest.approx(6.0 * (1 + np.sqrt(2)), rel=1e-10)

    data_file.write_text("second calibration")
    file_change = read_settled_project_plan(review, jobs).light[assemble_key].result.validity
    assert file_change.state == "stale"
    assert {(reason.kind, reason.path.path) for reason in file_change.reasons} == {("file", data_file.name)}
