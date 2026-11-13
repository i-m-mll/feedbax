"""Qualify saved task and connection variants on one real Train checkpoint."""

from copy import deepcopy

import numpy as np
import pytest

from feedbax.compiler.model_document import lower_model_document
from feedbax.contracts.project import Project
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.rollout import RolloutParameters, authored_variation
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.host import HostClient
from feedbax.jobs import JobService, PlanJobReader
from feedbax.persistence.array_container import ArrayContainerReader
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.studio.canonical_evaluate import RAW_STATES_SCHEMA, CannotRun
from feedbax.training.evaluation_variant import resolve_evaluation_variant
from feedbax.web.services.project_folder import ProjectFolderService
from tests.test_canonical_evaluate_admission import _save_project, managed_evaluation_definition
from tests.test_local_job_host import host as host_fixture, job_row, wait_until
from tests.test_model_document_plan import _add_attachment, _attachment_owner

host = host_fixture


def _patch(path, value, *, op="replace"):
    return {"path": path, "value": value, "op": op}


def _saved_variants():
    definition = managed_evaluation_definition()
    steps = definition["steps"]
    baseline = steps["baseline"]
    task = deepcopy(definition["documents"]["task"])
    signal = next(signal for signal in task["payload"]["signals"] if signal["id"] == "target_pos")
    signal["cells"][0]["value"] = {
        "kind": "call",
        "function": "vector",
        "args": [{"kind": "literal", "value": coordinate, "unit": "m"} for coordinate in (-0.05, 0.10)],
    }
    definition["documents"]["replacement-task"] = task

    def step(variation):
        result = deepcopy(baseline)
        result["base"]["inline"]["params"]["variation"] = variation
        return result

    selection = _patch("task", {"document": "replacement-task"})
    replacement = step([selection])
    perturbed = step([selection])
    perturbed["model_attachments"] = {"wire": [_add_attachment(_attachment_owner("read"), "scale", None, 0.5)]}
    incompatible = deepcopy(task["payload"])
    incompatible["signals"] = [signal for signal in incompatible["signals"] if signal["id"] != "target_pos"]
    nested = "graph.subgraphs.wrapper.subgraphs.controller.nodes"
    definition["steps"] = {
        "train": steps["train"], "baseline": baseline, "replacement": replacement, "perturbed": perturbed,
        "different-dtype": step([_patch(f"{nested}.readout.params.dtype", "float16")]),
        "incompatible": step([_patch("task", incompatible)]),
    }
    return Project.model_validate(definition)


def test_two_crossed_wire_insertions_compose_through_saved_plan(tmp_path):
    definition = managed_evaluation_definition()
    registries = new_application_registry_bundle()
    from feedbax.contracts.model_document import ModelDocument
    from feedbax.contracts.task_document import TaskDocument

    model = ModelDocument.model_validate(definition["documents"]["reach-model"]["payload"])
    task = TaskDocument.model_validate(definition["documents"]["task"]["payload"])
    graph = lower_model_document("reach-model", model, registries.components).graph
    first = next(wire for wire in graph.wires if wire.target_node == "plant")
    second = next(wire for wire in graph.wires if wire.target_node == "wrapper")
    baseline = deepcopy(definition["steps"]["baseline"])
    baseline["base"]["inline"]["params"]["variation"] = {}
    axes, ordered = {}, []
    for name, wire in (("first", first), ("second", second)):
        wire_id = wire.authored_id()
        node = f"scale_{name}"
        edits = [
            _patch(f"graph.nodes.{node}", {
                "type": "feedbax.Scale", "params": {"scale": 0.5, "fixed": True},
                "input_ports": ["input"], "output_ports": ["output"],
            }, op="add"),
            _patch(f"graph.wires.{wire_id}.target_node", node),
            _patch(f"graph.wires.{wire_id}.target_port", "input"),
            _patch(f"graph.wires.{name}-out", {
                "source_node": node, "source_port": "output",
                "target_node": wire.target_node, "target_port": wire.target_port,
            }, op="add"),
        ]
        ordered.extend(edits)
        axes[name] = {"values": {"inserted": {"patches": {
            f"params.variation.{name}-{index}": {"op": "add", "value": edit}
            for index, edit in enumerate(edits)
        }}}}
    baseline["sweep"] = {"axes": axes}
    invalid = deepcopy(definition["steps"]["baseline"])
    invalid["base"]["inline"]["params"]["variation"] = {
        "bad-port": _patch(f"graph.wires.{first.authored_id()}.target_port", "unavailable"),
    }
    project = Project.model_validate(definition | {"steps": {
        "train": definition["steps"]["train"], "crossed": baseline, "invalid": invalid,
    }})
    folders = ProjectFolderService(tmp_path / "projects")
    _save_project(folders, "crossed", project)
    plan = folders.read_plan("crossed", revision=None).plan
    assert not plan.diagnostics
    (run,) = (run for run in plan.runs if run.key.step == "crossed")
    request = decode_request(plan.requests[run.request_id], plan.documents.__getitem__)
    named = RolloutParameters.model_validate(request.spec.params).variation
    listed = RolloutParameters.model_validate(request.spec.params | {"variation": ordered}).variation
    variant = resolve_evaluation_variant(graph, task, named)
    assert variant == resolve_evaluation_variant(graph, task, listed)
    unchanged = [wire for wire in graph.wires if wire not in (first, second)]
    assert all(wire in variant.graph.wires for wire in unchanged)
    assert len(named) == 8 and all(edit.path != "graph.wires" for _, edit in authored_variation(named))
    wires = {wire.authored_id(): wire for wire in variant.graph.wires}
    for name, original in (("first", first), ("second", second)):
        before, after = wires[original.authored_id()], wires[f"{name}-out"]
        assert (before.target_node, before.target_port) == (f"scale_{name}", "input")
        assert (after.source_node, after.source_port) == (f"scale_{name}", "output")
        assert (after.target_node, after.target_port) == (original.target_node, original.target_port)
    reader = PlanJobReader(HostClient(), enable_evaluation_candidate=True)
    reader.evaluation_parameters(plan, run)
    (bad,) = (run for run in plan.runs if run.key.step == "invalid")
    params, _ = reader.evaluation_parameters(plan, bad)
    with pytest.raises(CannotRun, match="/params/variation/bad-port: .*does not exist"):
        resolve_evaluation_variant(graph, task, params.variation)


def test_saved_task_replacement_and_scale_use_one_exact_trained_checkpoint(host, tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    registry = new_application_registry_bundle()
    jobs = JobService(host, registry, enable_evaluation_candidate=True)
    folders = ProjectFolderService(tmp_path / "projects")
    project = "compatibility"
    _save_project(folders, project, _saved_variants())
    reviewed = folders.read_plan(project, revision=None)
    assert not reviewed.plan.diagnostics
    planned = {run.key.step: run for run in reviewed.plan.runs}
    cases = ["baseline", "replacement", "perturbed", "different-dtype", "incompatible"]
    assert len({planned[name].request_id for name in cases}) == len(cases)
    for name in cases:
        assert planned[name].inputs["subject"].producer == planned["train"].key
    decoded = decode_request(
        reviewed.plan.requests[planned["replacement"].request_id],
        reviewed.plan.documents.__getitem__,
    )
    replacement_task = decoded.spec.params["variation"][0]["value"]
    signal = next(signal for signal in replacement_task["signals"] if signal["id"] == "target_pos")
    assert signal["cells"][0]["value"]["args"] == [
        {"kind": "literal", "value": coordinate, "unit": "m"} for coordinate in (-0.05, 0.10)
    ]

    (train,) = jobs.admit_plan(
        folders, reviewed, run_keys=(RunKey("default", "train", "default"),), token="train",
    ).items
    started = jobs.admit_plan(
        folders, reviewed, run_keys=tuple(planned[name].key for name in cases), token="variants",
    )
    assert [item.disposition for item in started.items] == ["admitted"] * len(cases)

    def settled(job_id):
        return wait_until(lambda: (
            job_row(host, job_id, "succeeded") or job_row(host, job_id, "failed")
        ), timeout=25)

    trained = settled(train.job_id)
    assert trained["state"] == "succeeded", trained["outcome"]
    completion = trained["outcome"]["completion"]
    checkpoint = PublishedTrainingCheckpoint.model_validate(completion["checkpoint"])
    position = "/v1/model/state/plant/effector~1position"
    velocity = "/v1/model/state/plant/effector~1velocity"
    carry = "/v1/model/carry/wrapper/controller/carry/carry"
    hidden = "/v1/model/state/wrapper/controller/cell/hidden"
    target = "/v1/task/output/target_pos"
    data, descriptors, results = {}, {}, {}
    for name, item in zip(cases, started.items, strict=True):
        row = settled(item.job_id)
        assert row["edges"][0]["binding"]["completion"] == completion
        if name in {"incompatible", "different-dtype"}:
            assert row["state"] == "failed"
            error = row["outcome"]["error"]
            assert error["code"] == "cannot_run"
            if name == "different-dtype":
                assert error["path"].startswith("/inputs/subject/checkpoint/model/")
                assert "dtype" in error["message"]
            else:
                assert error["path"].startswith("/inputs/subject/model/inputs")
                assert "target_pos" in error["message"]
            assert "receipt" not in row["outcome"]
            continue
        assert row["state"] == "succeeded", row["outcome"]
        result = jobs.read_evaluation_result(project=project, store=reviewed.store_path, job_id=item.job_id)
        results[name] = result
        assert result._manifest().provenance.parents[0].id == checkpoint.pointer.transaction_id
        with result.open_states() as states:
            descriptors[name] = states.descriptor
            assert states.descriptor.schema_id == RAW_STATES_SCHEMA
            assert states.data("trial").tolist() == list(range(28))
            assert states.data("objective/coverage/effort").item() == 28
            data[name] = {
                key: states.data(key)[0, 0] for key in (
                    position, velocity, carry, hidden, target, "objective/total",
                )
            }
            data[name]["condition"] = states.data("condition")[0]
    baseline, replacement, perturbed = (data[name] for name in cases[:3])
    assert len({item["condition"] for item in data.values()}) == 3
    assert all(d.variables == descriptors["baseline"].variables for d in descriptors.values())
    assert len({d.attrs["source_graph_sha256"] for d in descriptors.values()}) == 1
    b, c = (descriptors[name].attrs for name in ("replacement", "perturbed"))
    assert b["effective_task_sha256"] == c["effective_task_sha256"]
    assert b["effective_graph_sha256"] != c["effective_graph_sha256"]
    assert b["variation_sha256"] != c["variation_sha256"]

    visible = np.any(replacement[target] != 0.0, axis=-1)
    assert len({tuple(mask) for mask in visible}) > 1
    np.testing.assert_array_equal(
        replacement[target][visible],
        np.broadcast_to(np.array([-0.05, 0.10], dtype=np.float32), replacement[target][visible].shape),
    )
    assert not np.array_equal(replacement[target], baseline[target])
    for key in (target, hidden, carry):
        np.testing.assert_array_equal(perturbed[key], replacement[key])
    displacement = replacement[position] - replacement[position][:, :1]
    assert np.max(np.abs(displacement)) > 0
    np.testing.assert_allclose(
        perturbed[position] - perturbed[position][:, :1], 0.5 * displacement, rtol=3e-5, atol=1e-7,
    )
    np.testing.assert_allclose(perturbed[velocity], 0.5 * replacement[velocity], rtol=3e-5, atol=1e-7)
    assert replacement["objective/total"] > 0
    np.testing.assert_allclose(perturbed["objective/total"], 0.25 * replacement["objective/total"], rtol=2e-5)

    read_tensor = ArrayContainerReader.read_tensor
    read = []

    def observe(reader, name):
        read.append(name)
        return read_tensor(reader, name)

    with monkeypatch.context() as context:
        context.setattr(ArrayContainerReader, "read_tensor", observe)
        with results["perturbed"].open_states() as states:
            position_samples = states.select(position, labels={"trial": 0})[0, 0]
        assert read == ["trial", position]
    np.testing.assert_array_equal(position_samples, perturbed[position][0])
