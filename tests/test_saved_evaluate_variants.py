"""Qualify saved task and connection variants on one real Train checkpoint."""

from copy import deepcopy

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from feedbax.compiler import GraphDocument, compile_graph
from feedbax.contracts.project import Project
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint, StudioGraphMethodPayload
from feedbax.jobs import JobService
from feedbax.persistence.array_container import ArrayContainerReader
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.runtime._graph_key_schedule import derive_key
from feedbax.studio.canonical_evaluate import RAW_STATES_SCHEMA
from feedbax.training.checkpoint_custody import open_published_model_slot
from feedbax.training.studio_executor import studio_training_key
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.worker.execution import prototypes_from_task_bindings
from tests.test_canonical_evaluate_admission import _save_project, authored_evaluation_project
from tests.test_local_job_host import host as host_fixture, job_row, wait_until

host = host_fixture


def _patch(path, value, *, op="replace"):
    return {"path": path, "value": value, "op": op}


def _saved_variants():
    authored, source = authored_evaluation_project()
    definition = authored.model_dump(mode="json", by_alias=True, exclude_unset=True)
    original, baseline = (definition["steps"][name] for name in ("train", "baseline"))
    graph = original["base"]["inline"]["graph"]
    replacement = deepcopy(original["base"]["inline"]["task"])
    replacement["timeline"]["epoch_value_specs"].append({
        "target_id": "reach_target",
        "epoch_id": "reach",
        "value_spec": {"mode": "constant", "value": [-0.05, 0.10]},
    })
    task = _patch("task", {"document": "replacement-task"})
    different_horizon = deepcopy(replacement)
    different_horizon["params"]["n_steps"] = 80
    wires = deepcopy(graph["wires"])
    force = next(wire for wire in wires if wire["target_node"] == "plant")
    force.update(target_node="force_scale", target_port="input")
    wires.append({
        "source_node": "force_scale", "source_port": "output",
        "target_node": "plant", "target_port": "force",
    })
    scale = [
        _patch("graph.nodes.force_scale", {
            "type": "feedbax.Scale", "params": {"scale": 0.5, "fixed": True},
            "input_ports": ["input"], "output_ports": ["output"],
        }, op="add"),
        _patch("graph.wires", wires),
    ]
    wider = deepcopy(graph["wires"])
    recurrent = next(wire for wire in wider if wire["temporality"] == "recurrent")
    recurrent["recurrent_initializer"]["shape"] = [101]

    def step(variation):
        result = deepcopy(baseline)
        result["base"]["inline"]["params"]["variation"] = variation
        return result

    return Project.model_validate({
        "documents": {
            "replacement-task": {"schema": "feedbax.spec.task.v1", "payload": replacement},
        },
        "steps": {
            "train": original,
            "baseline": baseline,
            "replacement": step([task]),
            "perturbed": step([task, *scale]),
            "different-dtype": step([_patch("graph.nodes.readout.params.dtype", "float16")]),
            "different-horizon": step([_patch("task", different_horizon)]),
            "incompatible": step([
                _patch("graph.nodes.cell.params.hidden_size", 101),
                _patch("graph.nodes.readout.params.input_size", 101),
                _patch("graph.wires", wider),
            ]),
        },
    }), source


def test_saved_task_replacement_and_scale_use_one_exact_trained_checkpoint(host, tmp_path, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_ENABLE_X64", "0")
    registry = new_application_registry_bundle()
    jobs = JobService(host, registry, enable_evaluation_candidate=True)
    folders = ProjectFolderService(tmp_path / "projects")
    project = "compatibility"
    authored, source = _saved_variants()
    _save_project(folders, project, authored)
    reviewed = folders.read_plan(project, revision=None)
    assert not reviewed.plan.diagnostics
    planned = {run.key.step: run for run in reviewed.plan.runs}
    cases = [
        "baseline", "replacement", "perturbed", "incompatible", "different-dtype", "different-horizon",
    ]
    assert len({planned[name].request_id for name in cases}) == len(cases)
    for name in cases:
        assert planned[name].inputs["subject"].producer == planned["train"].key
    decoded = decode_request(
        reviewed.plan.requests[planned["replacement"].request_id],
        reviewed.plan.documents.__getitem__,
    )
    assert decoded.spec.params["variation"][0]["value"]["timeline"]["epoch_value_specs"][-1][
        "value_spec"
    ]["value"] == [-0.05, 0.10]

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
    carry = "/v1/model/carry/cell:hidden/hidden"
    hidden = "/v1/model/state/cell/hidden"
    target = "/v1/task/output/reach_target"
    data, descriptors, results = {}, {}, {}
    for name, item in zip(cases, started.items, strict=True):
        row = settled(item.job_id)
        assert row["edges"][0]["binding"]["completion"] == completion
        if name == "different-horizon":
            assert row["state"] == "failed"
            error = row["outcome"]["error"]
            assert error["code"] == "cannot_run"
            assert error["path"] == "/params/variation/0/value"
            assert "preserve the horizon" in error["message"]
            assert "receipt" not in row["outcome"]
            continue
        if name in {"incompatible", "different-dtype"}:
            assert row["state"] == "failed"
            error = row["outcome"]["error"]
            assert error["code"] == "cannot_run"
            address = "cell/candidate_bias" if name == "incompatible" else "readout/bias"
            assert error["path"] == f"/inputs/subject/checkpoint/model/v1/model/value/{address}"
            assert "shape or dtype differs" in error["message"]
            if name == "different-dtype":
                assert "float16" in error["message"]
            assert "receipt" not in row["outcome"]
            continue
        assert row["state"] == "succeeded", row["outcome"]
        result = jobs.read_evaluation_result(project=project, store=reviewed.store_path, job_id=item.job_id)
        results[name] = result
        assert result._manifest().provenance.parents[0].id == checkpoint.pointer.transaction_id
        with result.open_states() as states:
            descriptors[name] = states.descriptor
            assert states.descriptor.schema_id == RAW_STATES_SCHEMA
            assert states.data("trial").tolist() == ["alpha", "beta", "gamma"]
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

    hold = np.all(baseline[target] == 0.0, axis=-1)
    assert len({tuple(mask) for mask in hold}) > 1
    np.testing.assert_array_equal(hold, np.all(replacement[target] == 0.0, axis=-1))
    np.testing.assert_array_equal(
        replacement[target][~hold],
        np.broadcast_to(np.array([-0.05, 0.10], dtype=np.float32), replacement[target][~hold].shape),
    )
    assert not np.array_equal(replacement[target], baseline[target])
    for key in (target, hidden, carry):
        np.testing.assert_array_equal(perturbed[key], replacement[key])
    assert np.max(np.abs(replacement[position])) > 0.001
    np.testing.assert_allclose(perturbed[position], 0.5 * replacement[position], rtol=3e-5, atol=1e-8)
    np.testing.assert_allclose(perturbed[velocity], 0.5 * replacement[velocity], rtol=3e-5, atol=1e-8)
    for values in data.values():
        np.testing.assert_allclose(
            values["objective/total"],
            np.sum((values[position][:, -1] - values[target][:, -1]) ** 2, axis=-1),
            rtol=2e-5,
        )

    slot = open_published_model_slot(checkpoint.checkpoint_root, checkpoint.pointer)
    assert slot is not None
    seed = StudioGraphMethodPayload.model_validate(source.method_payload.payload).seed
    template = compile_graph(
        GraphDocument(graph=slot.graph), registry.components,
        input_prototypes=prototypes_from_task_bindings(source.task_binding_spec),
        initialization_key=studio_training_key(seed + 1, "model"), artifact_provider=slot.provider,
    ).graph
    restored = slot.restore(template)
    assert not eqx.tree_equal(restored.nodes["cell"].cell, template.nodes["cell"].cell)
    inputs = {
        next(iter(restored.input_bindings)): jnp.asarray(replacement[target][0]),
    }
    key = derive_key(
        derive_key(jax.random.PRNGKey(731), "evaluate-trial", "alpha"), "studio-trial", "graph",
    )
    outputs, _ = restored(inputs, restored.init_state(key=key), key=key, n_steps=100)
    np.testing.assert_allclose(outputs["effector"].pos, replacement[position][0, 1:], atol=1e-7)

    read_tensor = ArrayContainerReader.read_tensor
    read = []

    def observe(reader, name):
        read.append(name)
        return read_tensor(reader, name)

    with monkeypatch.context() as context:
        context.setattr(ArrayContainerReader, "read_tensor", observe)
        replay = results["perturbed"].replay_trial("alpha", anchor_id="effector")
        assert read == ["state_time", position]
    np.testing.assert_array_equal(replay.tracks[0].samples, perturbed[position][0])
