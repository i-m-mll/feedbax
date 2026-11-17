"""Reconstructed graphs, rollout trajectories, and task inputs in saved Analyze jobs."""

import asyncio
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID
from feedbax.contracts.project import Project
from feedbax.contracts.project_plan import ElementKey, RunKey
from feedbax.contracts.rollout import ROLLOUT_TYPE
from feedbax.contracts.task_document import PYTHON_TASK_SCHEMA_ID, TASK_DOCUMENT_SCHEMA_ID
from feedbax.jobs import JobService
from feedbax.light_jobs import dataset_member
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import open_stored_dataset
from feedbax.persistence.effective_success import SuccessView
from feedbax.plugins.composition import compose_application
from feedbax.web.services.project_folder import ProjectFolderService
from tests.support import derivative_witness as witness
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_local_job_host import host as host_fixture, restart, wait_until

host = host_fixture
_CAPTURE = Path(__file__).parent / "fixtures" / "derivative_fixed_points.json"


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


def _install_external_package(root, monkeypatch):
    package = root / witness.PACKAGE
    package.mkdir(parents=True)
    shutil.copyfile(witness.__file__, package / "__init__.py")
    metadata = root / f"{witness.PACKAGE}-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: feedbax-witness\nVersion: 1.0\n")
    (metadata / "entry_points.txt").write_text(f"[feedbax.plugins]\nwitness = {witness.PACKAGE}\n")
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
    jobs = JobService(host, registry)
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
