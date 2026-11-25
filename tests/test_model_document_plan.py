"""Plan and shared graph compilation for a model document with nested owners."""

from copy import deepcopy
import json
from unittest.mock import Mock

import jax
import numpy as np
import equinox as eqx
import jax.numpy as jnp
import pytest
from fastapi.testclient import TestClient

from feedbax.cli import main
from feedbax.compiler import GraphDocument, compile_graph
from feedbax.execution.trial_resolution import PreparedTrialExecution, compile_trial_execution
from feedbax.compiler.model_document import ModelCompilationProblem, lower_model_document, model_input_prototypes
from feedbax.compiler.model_attachment import resolve_effective_model
from feedbax.builtins.templates.mechanics_templates import two_link_arm_6muscle_template_graph
from feedbax.contracts import request_identity
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.studio_place import StudioModelPlace
from feedbax.contracts.model_attachment import MODEL_ATTACHMENT_BATCHES
from feedbax.contracts.expression_values import LENGTH, TIME, ValueBase, ValueType
from feedbax.contracts.graph import StudioTaskBindingSpec
from feedbax.examples import load_shipped_projects
from feedbax.contracts.jobs import JobStatus
from feedbax.component_registry import ComponentRegistry
from tests.support.component_owners import register_owners
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.model_edit import ModelEditCommand, ModelEditResult, ModelEditState
from feedbax.contracts.project import Project
from feedbax.planner.project_plan import ProjectPlanMemo, RunKey, compile_project
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.rollout import RolloutParameters
from feedbax.contracts.train_settings import normalize_train_method
from feedbax.jobs import JobService
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView
from feedbax.application.bundle import new_application_registry_bundle
from feedbax.compiler.model_edit import apply_model_edit
from feedbax.compiler.model_ports import ModelRegistryPorts
from feedbax.runtime.graph import GraphTraceRequest
from feedbax.builtins.runtime.components import Input
from feedbax.runtime._signal_schedule import SignalScheduleError, resolve_signal_schedule
from feedbax.host.method_checkpoint import read_method_checkpoint
from feedbax.training.evaluation_variant import resolve_evaluation_variant
from feedbax.runtime._graph_key_schedule import model_initialization_key
from feedbax.training.method_runtime import TrainingLoop, initialize_method, replica_value
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.contracts.model_edit import ModelEditRequest
from feedbax.web.services.project_plan import ProjectPlanRead, ProjectPlanService
from feedbax.web.app import create_app
from tests.support.managed_train_fixture import managed_train_project
from tests.test_canonical_evaluate_admission import _save_project
from tests.support.model_documents import read_changed_model
from tests.support.host import job_row, token, wait_until



def _prepare_request(request, registry) -> PreparedTrialExecution:
    documents = request.trial_documents
    paired = compile_trial_execution(
        documents.model, documents.task_source,
        registry.components, registry.expression_functions, python_tasks=registry.python_tasks,
        objective=documents.objective, input_bindings=documents.input_bindings,
        initialization_key=model_initialization_key(request.seed or 0),
    )
    assert isinstance(paired, PreparedTrialExecution), paired
    return paired


def _end(component, port):
    return {"owner": "component", "component": component, "port": port}


def _wire(source, target, attachments=()):
    return {"source": source, "target": target, "attachments": list(attachments)}


def _port(id_, name, *, shape=None):
    return {"id": id_, "name": name, "type": {"shape": shape}}


@pytest.mark.parametrize(("delay", "initial_value", "expected", "gradient"), [
    ({"value": 1, "unit": "st"}, 0, [0, 24, 48, 72], [24, 24, 24, 0]),
    ({"value": 0.02, "unit": "s"}, 5, [120, 120, 24, 48], [24, 24, 0, 0]),
])
def test_noncontiguous_group_preserves_rollout_and_gradient(delay, initial_value, expected, gradient):
    scale = {
        "kind": "leaf", "name": "Scale", "type": "feedbax.Scale",
        "fields": {"scale": 2.0, "fixed": True},
    }
    boundary = {"owner": "inputs", "port": "signal"}
    source_wire = _wire(boundary, _end("a", "input"), [
        {"kind": "delay", "id": "delay", "delay": delay, "initial_source": {"kind": "value", "value": initial_value}},
        {"kind": "intervention", "id": "scale", "operation": "Scale", "operands": {"factor": {"kind": "value", "value": 3.0}}},
    ])
    model = ModelDocument.model_validate({
        "step_seconds": 0.01,
        "ports": {
            "inputs": [_port("signal", "signal", shape=[])],
            "outputs": [_port("result", "result", shape=[])],
        },
        "interior": {
            "components": {name: scale for name in ("a", "b", "c")},
            "wires": {
                "input": source_wire,
                "ab": _wire(_end("a", "output"), _end("b", "input")),
                "bc": _wire(_end("b", "output"), _end("c", "input")),
                "result": _wire(_end("c", "output"), {"owner": "outputs", "port": "result"}),
            },
        },
    })
    registry = ComponentRegistry()
    ports = ModelRegistryPorts(registry)
    command = ModelEditCommand.model_validate({"operation": {
        "op": "group",
        "name": "Group",
        "selection": [
            {"composites": [], "element": {"kind": "component", "id": name}}
            for name in ("a", "c")
        ],
    }})
    grouped = apply_model_edit(
        ModelEditState(document=model), command, ports=ports.port, component_problem=ports.placement_problem,
    )
    assert isinstance(grouped, ModelEditResult), grouped
    group = grouped.changed[-1].element.id
    flat = lower_model_document(model, registry).graph
    lowered = lower_model_document(grouped.state.document, registry).graph
    assert [scope for scope in resolve_signal_schedule(lowered).order if scope[-1] in {"a", "b", "c"}] == [
        (group, "a"), ("b",), (group, "c"),
    ]
    initial = {flat.input_bindings["signal"]: jax.ShapeDtypeStruct((), jnp.float32)}
    first = compile_graph(GraphDocument(graph=flat), registry, input_prototypes=initial)
    second = compile_graph(GraphDocument(graph=lowered), registry, input_prototypes=initial)
    assert second.record.key_schedule.scoped_node_order == resolve_signal_schedule(lowered).order

    def run(graph, signal):
        requests = (GraphTraceRequest("graph_output", "result", port="result"),)
        if graph is second.graph:
            requests += (GraphTraceRequest("port", "first scale", node="a", port="output", scope=(group,)),)
        outputs, _, trace = graph.rollout_with_trace(
            {"signal": signal}, graph.init_state(key=jax.random.key(5)), n_steps=signal.size,
            key=jax.random.key(5),
            trace=requests,
        )
        return outputs["result"], trace

    signal = jnp.arange(1, 5, dtype=jnp.float32)
    for graph in (first.graph, second.graph):
        output, trace = run(graph, signal)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(trace["result"], output)
        if graph is second.graph:
            np.testing.assert_array_equal(trace["first scale"], np.asarray(expected) / 4)
        derivative = jax.grad(lambda data: run(graph, data)[0].sum())(signal)
        np.testing.assert_array_equal(derivative, gradient)

    command = ModelEditCommand.model_validate({"operation": {
        "op": "ungroup", "address": grouped.changed[-1].model_dump(mode="json"),
    }})
    ungrouped = apply_model_edit(
        grouped.state, command, ports=ports.port, component_problem=ports.placement_problem,
    )
    assert isinstance(ungrouped, ModelEditResult), ungrouped
    assert ungrouped.state.document == model

    cycle = lower_model_document(grouped.state.document, registry).graph
    first_cut = next(wire for wire in cycle.subgraphs[group].wires if wire.is_recurrent)
    first_cut.recurrent_initializer = None
    upstream = next(wire for wire in cycle.wires if wire.source_node == flat.input_bindings["signal"][0])
    last = cycle.output_bindings["result"]
    upstream.source_node, upstream.source_port = last
    with pytest.raises(SignalScheduleError, match="same-step cycle") as exc:
        resolve_signal_schedule(cycle)
    assert exc.value.authored_path.startswith("/graph/")


def _nested_model() -> dict:
    leaves = {
        "cell": {"kind": "leaf", "name": "Cell", "type": "feedbax.GRU", "fields": {
            "hidden_size": 100, "dtype": "float32",
        }},
        "readout": {"kind": "leaf", "name": "Readout", "type": "feedbax.Linear", "fields": {
            "output_size": 2, "use_bias": True, "activation": "identity",
            "initialization": "random", "dtype": "float32", "trainable": False,
        }},
        "plant": {"kind": "leaf", "name": "Plant", "type": "feedbax.PointMass", "fields": {"mass": 1.0}},
    }
    delay = {"kind": "delay", "id": "carry", "delay": {"value": 1, "unit": "st"}}
    network = {"kind": "composite", "name": "Controller", "ports": {
        "inputs": [_port("target", "Target", shape=[2])],
        "outputs": [_port("force", "Force", shape=[2])],
    }, "interior": {"components": {key: leaves[key] for key in ("cell", "readout")}, "wires": {
        "target": _wire({"owner": "inputs", "port": "target"}, _end("cell", "input")),
        "carry": _wire(_end("cell", "hidden"), _end("cell", "hidden"), (delay,)),
        "decode": _wire(_end("cell", "output"), _end("readout", "input")),
        "read": _wire(_end("readout", "output"), {"owner": "outputs", "port": "force"}),
    }}}
    wrapper = {"kind": "composite", "name": "Wrapper", "ports": deepcopy(network["ports"]), "interior": {
        "components": {"controller": network}, "wires": {
            "pass-target": _wire({"owner": "inputs", "port": "target"}, _end("controller", "target")),
            "pass-force": _wire(_end("controller", "force"), {"owner": "outputs", "port": "force"}),
        },
    }}
    return {
        "step_seconds": 0.01,
        "ports": {"inputs": [_port("task-input", "reach_target", shape=[2])],
                  "outputs": [_port("effector", "effector")]},
        "interior": {"components": {"wrapper": wrapper, "plant": leaves["plant"]}, "wires": {
            "task-wire": _wire({"owner": "inputs", "port": "task-input"}, _end("wrapper", "target")),
            "plant-wire": _wire(_end("wrapper", "force"), _end("plant", "force")),
            "effector-wire": _wire(_end("plant", "effector"), {"owner": "outputs", "port": "effector"}),
        }},
    }

def _model_project() -> dict:
    from tests.test_document_train_pair import authored_documents, document_project

    return document_project(authored_documents(), evaluation=True)


def _rename_components(model):
    for component in model["interior"]["components"].values():
        component["name"] += " display name"
        if component["kind"] == "composite":
            for ports in component["ports"].values():
                for port in ports:
                    port["name"] += " display name"
            _rename_components(component)


@pytest.mark.parametrize(("delay", "step", "conversion", "delay_steps"), [
    ({"value": 0.05, "unit": "s"}, 0.02, {"value": 3, "unit": "st"}, 3),
    ({"value": 0.05, "unit": "s"}, 0.01, {"value": 5, "unit": "st"}, 5),
    ({"value": 3, "unit": "st"}, 0.01, {"value": 0.03, "unit": "s"}, 3),
    ({"value": 0.001, "unit": "s"}, 0.01, None, None),
])
def test_a_projected_delay_carries_its_conversion_to_the_other_time_unit(tmp_path, delay, step, conversion, delay_steps):
    raw = _model_project()
    model = raw["documents"]["reach-model"]["payload"]
    model["step_seconds"] = step
    controller = model["interior"]["components"]["wrapper"]["interior"]["components"]["controller"]
    controller["interior"]["wires"]["carry"]["attachments"][0]["delay"] = delay
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, "delay", Project.model_validate(raw))
    url = "/api/projects/delay"
    owner = {"composites": ["wrapper", "controller"],
             "element": {"kind": "attachment", "parent": {"kind": "wire", "id": "carry"}, "id": "carry"}}
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        revision = client.get(url).json()["revision"]
        fields = client.post(url + "/model/fields", json={
            "revision": revision, "document": "reach-model", "owner": owner,
            "run": {"variant": "default", "step": "train", "row": "default"},
        })
        assert fields.status_code == 200, fields.text
        (shown,) = [field for field in fields.json()["fields"] if field["label"] == "Delay"]
        assert shown.get("conversion") == conversion
        if delay["unit"] == "st":
            assert shown["text"] == "3 st"
        projected = client.post(url + "/model/projection", json={"revision": revision, "document": "reach-model"})
        assert projected.status_code == 200, projected.text
        assert projected.json()["delay_steps"][ModelAddress.model_validate(owner).to_text()] == delay_steps


def test_intervention_window_field_edits_and_sweeps_in_the_authored_unit(tmp_path):
    raw = managed_train_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    model = raw["documents"]["point-mass"]["payload"]
    pulse = {
        "kind": "intervention", "id": "pulse", "operation": "add", "operands": {"value": {"kind": "value", "value": [0.5, 0.0]}},
        "window": {"spans": [{
            "kind": "interval", "start": {"anchor": {"kind": "trial", "edge": "start"}},
            "end": {"offset": {"value": 2, "unit": "st"}},
        }]},
    }
    model["interior"]["wires"]["drive"]["attachments"] = [pulse]
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, "window", Project.model_validate(raw))
    url = "/api/projects/window"
    owner = {"composites": [], "element": {
        "kind": "attachment", "parent": {"kind": "wire", "id": "drive"}, "id": "pulse",
    }}

    with TestClient(create_app(projects_dir=tmp_path)) as client:
        snapshot = client.get(url).json()
        body = {"revision": snapshot["revision"], "document": "point-mass", "owner": owner,
                "run": {"variant": "default", "step": "train", "row": "default"}}
        projection = client.post(url + "/model/projection", json={
            "revision": snapshot["revision"], "document": "point-mass", "run": body["run"],
        })
        assert projection.status_code == 200, projection.text
        shown_model = projection.json()
        assert not shown_model["problems"], shown_model["problems"]
        shown = shown_model["resolved_model"]
        assert shown["interior"]["wires"]["drive"]["attachments"][0]["id"] == "pulse"
        assert set(shown["interior"]["components"]) == set(model["interior"]["components"])
        response = client.post(url + "/model/fields", json=body)
        assert response.status_code == 200, response.text
        fields = {tuple(field["address"]["element"]["path"]): field for field in response.json()["fields"]}
        offset = fields["window", "spans", 0, "start", "offset"]
        assert offset["text"] == "0 st" and offset["can_sweep"]
        assert fields["operands", "value", "value"]["structure"]["value"]["function"] == "vector"
        edit = {"address": offset["address"], "edit": {"kind": "text", "text": "0.1 s"}}
        value = client.post(url + "/model/fields", json=body | edit).json()
        assert value["command"]["operation"]["value"] == {"value": 0.1, "unit": "s"}
        applied = client.post(url + "/model/edit", json={
            "revision": snapshot["revision"], "presentation_revision": snapshot["presentation_revision"],
            "document": "point-mass", "command": value["command"],
        })
        assert applied.status_code == 200, applied.text
        snapshot = applied.json()["result"]["snapshot"]
        body["revision"] = snapshot["revision"]
        fields = client.post(url + "/model/fields", json=body).json()["fields"]
        shown = next(field for field in fields if field["address"] == offset["address"])
        assert shown["text"] == "0.1 s" and shown["conversion"] == {"value": 10.0, "unit": "st"}
        response = client.post(url + "/model/fields", json=body | {
            "address": offset["address"], "edit": {"kind": "text", "text": "0.1 s, 0.2 s"},
        })
        assert response.status_code == 200, response.text
        prepared = response.json()
        assert not prepared["problems"], prepared
        saved = client.patch(url, json={"revision": snapshot["revision"], "files": prepared["files"]})
        assert saved.status_code == 200, saved.text
        settled = folders.read_plan("window", revision=None)
        plan = settled.plan
        assert len(plan.runs) == 2 and all(run.admission_problem is None for run in plan.runs), (
            plan.diagnostics, tuple(run.problems for run in plan.runs),
        )
        assert {run.label for run in plan.runs} == {"Offset 0.1 s", "Offset 0.2 s"}
        window = ProjectPlanRead.without_history(settled).response(include_window_map=True).window_map
        assert window is not None
        (axis,) = window.steps[0].axes
        assert axis.label == "Offset"
        assert [value.value_label for value in axis.values] == ["0.1 s", "0.2 s"]
        values = set()
        for run in plan.runs:
            documents = decode_request(plan.requests[run.request_id], plan.documents.__getitem__).spec.trial_documents
            attachment = documents.model.interior.wires["drive"].attachments[0]
            start = attachment.window.spans[0].start.offset
            assert start.unit == "s"
            values.add(start.value)
            body = {
                "project_id": "window", "revision": saved.json()["revision"], "count": 2,
                "run": {"variant": run.key.variant, "step": run.key.step, "row": run.key.row},
            }
            route = "/api/execution/task-trials/sample"
            sampled = client.post(route, json=body)
            assert sampled.status_code == 200, sampled.text
            draw = sampled.json()
            address = ModelAddress.model_validate(owner).to_text()
            first = round(start.value / documents.model.step_seconds)
            active = [first, first + 1]
            for trial in draw["drawn"]["trials"]:
                assert np.flatnonzero(trial["intervention_windows"][address]).tolist() == active
            batch = client.post(route, json=body | {"aggregate": True})
            assert batch.status_code == 200, batch.text
            shares = batch.json()["drawn"]["interventions"][address]
            assert shares == [float(i in active) for i in range(draw["n_steps"])]
        assert values == {0.1, 0.2}


def test_scale_operand_sweep_has_the_field_name_without_changing_the_axis_key(tmp_path):
    raw = managed_train_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    wire = raw["documents"]["point-mass"]["payload"]["interior"]["wires"]["drive"]
    wire["attachments"] = [{
        "kind": "intervention", "id": "scale", "operation": "Scale",
        "operands": {"factor": {"kind": "value", "value": 0.5}},
    }]
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, "scale", Project.model_validate(raw))
    url = "/api/projects/scale"
    owner = {"composites": [], "element": {
        "kind": "attachment", "parent": {"kind": "wire", "id": "drive"}, "id": "scale",
    }}
    revision = folders.read("scale").revision
    body = {
        "revision": revision, "document": "point-mass", "owner": owner,
        "run": {"variant": "default", "step": "train", "row": "default"},
    }
    with TestClient(create_app(projects_dir=tmp_path, automatic_run="off")) as client:
        fields = client.post(f"{url}/model/fields", json=body)
        assert fields.status_code == 200, fields.text
        (factor,) = fields.json()["fields"]
        assert factor["label"] == "Factor"
        assert factor["address"]["element"]["path"] == ["operands", "factor", "value"]
        proposal = client.post(f"{url}/model/fields", json=body | {
            "address": factor["address"], "edit": {"kind": "text", "text": "0.5, 1"},
        })
        assert proposal.status_code == 200 and not proposal.json()["problems"], proposal.text
        saved = client.patch(url, json={"revision": revision, "files": proposal.json()["files"]})
        assert saved.status_code == 200, saved.text

    settled = folders.read_plan("scale", revision=None)
    window = ProjectPlanRead.without_history(settled).response(include_window_map=True).window_map
    assert window is not None
    (axis,) = window.steps[0].axes
    assert axis.id == "value"
    assert axis.label == "Factor"
    assert [(value.id, value.label, value.value_label) for value in axis.values] == [
        ("value-1", "Factor 0.5", "0.5"), ("value-2", "Factor 1", "1"),
    ]
    assert [(row.key.row, row.label) for row in window.rows] == [
        ("value=value-1", "Factor 0.5"), ("value=value-2", "Factor 1"),
    ]


def test_addressed_field_range_sample_and_row_provenance(tmp_path):
    raw = _model_project()
    raw["steps"].pop("baseline")
    model = raw["documents"]["reach-model"]["payload"]
    raw["steps"]["train"]["base"]["inline"]["alias"] = {"document": "reach-model", "pointer": "/step_seconds"}
    model["step_seconds"] = 0.02
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, "fields", Project.model_validate(raw))
    url = "/api/projects/fields"
    owner = {"composites": [], "element": {"kind": "graph"}}

    with TestClient(create_app(projects_dir=tmp_path)) as client:
        initial = client.get(url).json()
        body = {"revision": initial["revision"], "document": "reach-model", "owner": owner,
                "run": {"variant": "default", "step": "train", "row": "default"}}
        state = client.post(url + "/model/fields", json=body).json()["fields"][0]
        assert state["text"] == "0.02 s"
        assert state["site"]["kinds"] == ["value", "sweep"]
        response = client.post(url + "/model/fields", json=body | {
            "address": state["address"], "edit": {"kind": "text", "text": "linspace(10 ms, 30 ms, 3)"},
        })
        assert response.status_code == 200, response.text
        recipe = response.json()
        assert not recipe["problems"], recipe
        saved = client.patch(url, json={"revision": initial["revision"], "files": recipe["files"]})
        assert saved.status_code == 200, saved.text
        assert client.post(url + "/model/fields", json=body).status_code == 409
        snapshot = folders.read("fields")
        plan = folders.read_plan("fields", revision=snapshot.revision).plan
        keys = tuple(row.key for row in plan.rows)
        assert len(keys) == 3
        plan = folders.read_plan("fields", revision=snapshot.revision, source_runs=keys).plan
        assert not plan.runs and not plan.requests
        assert {(problem.step, problem.row, problem.path) for problem in plan.diagnostics} == {
            (key.step, key.row, "payload.alias") for key in keys
        }
        for key, period in zip(keys, (0.01, 0.02, 0.03), strict=True):
            definition = plan.selected_definitions[key]
            assert definition.payload["trial_documents"]["model"]["step_seconds"] == pytest.approx(period)
            assert definition.payload["alias"] == pytest.approx(period)
            assert definition.document_ids["payload.trial_documents.model"] == "reach-model"
            source = definition.sources["/payload/trial_documents/model/step_seconds"]
            assert source.kind == "sweep_value"
            assert definition.sources["/payload/alias"] == source
        assert snapshot.project.documents["reach-model"].payload == model

        component_owner = {"composites": [],
                           "element": {"kind": "component", "id": "plant"}}
        selected = body | {
            "owner": component_owner, "revision": snapshot.revision,
            "run": body["run"] | {"row": keys[0].row},
        }
        plant = client.post(url + "/model/fields", json=selected).json()["fields"]
        mass = next(field for field in plant if field["label"] == "Mass")
        prepared = client.post(url + "/model/fields", json=selected | {
            "address": mass["address"], "edit": {"kind": "text", "text": "2"},
            "rows": [keys[0].row],
        }).json()
        assert not prepared["problems"], prepared
        scoped = client.patch(url, json={"revision": snapshot.revision, "files": prepared["files"]})
        assert scoped.status_code == 200, scoped.text
        scoped_plan = folders.read_plan("fields", revision=scoped.json()["revision"], source_runs=keys).plan
        mass_path = "/payload/trial_documents/model/interior/components/plant/fields/mass"
        assert scoped_plan.selected_definitions[keys[0]].sources[mass_path].kind == "exception"
        assert scoped_plan.selected_definitions[keys[1]].sources[mass_path].kind == "document"
        restored = client.patch(url, json={
            "revision": scoped.json()["revision"], "address": {"kind": "step", "id": "train"},
            "patches": [{"op": "remove", "path": "/sweep/exceptions"}],
        })
        assert restored.status_code == 200, restored.text
        snapshot = folders.read("fields")

        body["revision"] = snapshot.revision
        body["run"]["row"] = keys[0].row
        sampled = client.post(url + "/model/fields", json=body | {
            "address": state["address"], "edit": {
                "kind": "text", "text": "sample(U(0.02 s, 0.03 s), 3)", "seed": 8,
            },
        })
        assert sampled.status_code == 200, sampled.text
        recipe = sampled.json()
        assert not recipe["problems"], recipe
        saved = client.patch(url, json={"revision": snapshot.revision, "files": recipe["files"]})
        assert saved.status_code == 200, saved.text
        sampled_axis = folders.read("fields").project.steps["train"].sweep.axes["step-seconds"]
        assert sampled_axis.sampler.distribution.family == "uniform"
        assert sampled_axis.sampler.seed == 8
        expected = [float(jax.random.uniform(jax.random.fold_in(jax.random.key(8), index),
                                             minval=0.02, maxval=0.03)) for index in range(3)]
        assert list(sampled_axis.draws.values()) == expected


def test_model_field_tau_keeps_units_and_allows_subsequent_leaf_edits(tmp_path):
    raw = _model_project()
    model = raw["documents"]["reach-model"]["payload"]
    components = model["interior"]["components"]
    components["rnn"] = {
        "kind": "leaf", "name": "RNN", "type": "feedbax.VanillaRNN",
        "fields": {"tau": 2, "hidden_size": 4},
    }
    _save_project(ProjectFolderService(tmp_path), "duration", Project.model_validate(raw))
    url = "/api/projects/duration"
    owner = {"composites": [], "element": {"kind": "component", "id": "rnn"}}

    with TestClient(create_app(projects_dir=tmp_path)) as client:
        def prepare(label, text, row="default", *, files=False):
            snapshot = client.get(url).json()
            body = {"revision": snapshot["revision"], "document": "reach-model", "owner": owner, "chosen_steps": ["train"],
                    "run": {"variant": "default", "step": "train", "row": row}}
            response = client.post(url + "/model/fields", json=body)
            assert response.status_code == 200, response.text
            field = next(item for item in response.json()["fields"] if item["label"] == label)
            result = client.post(url + "/model/fields", json=body | {
                "address": field["address"], "edit": {"kind": "text", "text": text},
            })
            assert result.status_code == 200, result.text
            recipe = result.json()
            assert not recipe["problems"], recipe
            if files:
                changed = client.patch(url, json={"revision": snapshot["revision"], "files": recipe["files"]})
                assert changed.status_code == 200, changed.text
                return field, changed.json()
            changed = client.post(url + "/model/edit", json={
                "revision": snapshot["revision"], "presentation_revision": snapshot["presentation_revision"],
                "document": "reach-model", "command": recipe["command"],
            })
            assert changed.status_code == 200, changed.text
            assert changed.json()["result"]["status"] == "applied", changed.text
            return field, recipe["command"]["operation"]["value"]

        field, value = prepare("Tau", "30 ms")
        assert field["text"] == "2 st"
        assert field["conversion"] == {"value": 0.02, "unit": "s"}
        assert value == {"kind": "literal", "value": 0.03, "unit": "s"}
        field, value = prepare("Hidden size", "8")
        assert field["text"] == "4"
        assert value == 8
        field, snapshot = prepare("Tau", "linspace(20 ms, 40 ms, 3)", files=True)
        assert field["text"] == "0.03 s"
        assert field["conversion"] == {"value": 3, "unit": "st"}
        axis = snapshot["project"]["steps"]["train"]["sweep"]["axes"]["tau"]
        assert ModelAddress.from_text(axis["path"]).element.path == ("fields", "tau")
        assert axis["range"] == {"start": 0.02, "stop": 0.04, "count": 3}
        folders = ProjectFolderService(tmp_path)
        revision = folders.read("duration").revision
        keys = tuple(row.key for row in folders.read_plan("duration", revision=revision).plan.rows if row.key.step == "train")
        definitions = folders.read_plan("duration", revision=revision, source_runs=keys).plan.selected_definitions
        assert [item.payload["trial_documents"]["model"]["interior"]["components"]["rnn"]["fields"]["tau"]
                for item in definitions.values()] == [
            {"kind": "literal", "value": value, "unit": "s"} for value in (0.02, 0.03, 0.04)
        ]
        _, snapshot = prepare("Hidden size", "sample(choice(8, 12), 4)", keys[0].row, files=True)
        sampled = snapshot["project"]["steps"]["train"]["sweep"]["axes"]["hidden-size"]
        assert len(sampled["draws"]) == 4
        assert all(type(value) is int and value in (8, 12) for value in sampled["draws"].values())
        assert sampled["sampler"]["distribution"]["family"] == "choice"


def test_class_parameter_axes_are_named_by_the_field_holding_the_class(tmp_path):
    raw = _model_project()
    raw["steps"].pop("baseline")
    raw["steps"]["train"]["base"]["inline"]["settings"]["optimizer"]["learning_rate"] = {
        "type": "warmup_cosine_decay_schedule", "init_value": 0.0, "peak_value": 1e-3, "warmup_steps": 10, "decay_steps": 100,
    }
    _save_project(ProjectFolderService(tmp_path), "class-axes", Project.model_validate(raw))
    url = "/api/projects/class-axes"
    endpoint = url + "/authoring/train/train/fields"
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        body = {"revision": client.get(url).json()["revision"]}

        recipe = client.post(endpoint, json=body | {
            "field": "/settings/optimizer/learning_rate/peak_value", "edit": {"kind": "text", "text": "0.001, 0.003"},
        }).json()
        assert not recipe["problems"], recipe
        client.patch(url, json=body | {"files": recipe["files"]}).raise_for_status()
        body = {"revision": client.get(url).json()["revision"]}
        recipe = client.post(endpoint, json=body | {
            "field": "/settings/optimizer/b1", "edit": {"kind": "text", "text": "0.8, 0.9"},
        }).json()
        assert not recipe["problems"], recipe
        client.patch(url, json=body | {"files": recipe["files"]}).raise_for_status()
        axes = client.get(url + "/open").json()["snapshot"]["project"]["steps"]["train"]["sweep"]["axes"]
        assert set(axes) == {"learning-rate-peak-value", "optimizer-b1"}


def test_a_sweep_with_an_exception_can_still_sweep_the_fields_the_exception_leaves_alone(tmp_path):
    raw = _model_project()
    raw["steps"].pop("baseline")
    rate = "settings.optimizer.learning_rate.value"
    raw["steps"]["train"]["sweep"] = {
        "axes": {"learning-rate": {"values": {
            "slow": {"value": 0.001, "patches": {rate: {"value": 0.001}}},
            "fast": {"value": 0.003, "patches": {rate: {"value": 0.003}}},
        }}},
        "exceptions": {"wide": {
            "where": {"rule": "learning-rate=fast"},
            "patches": {"training_config.batch_size": {"value": 8}},
        }},
    }
    _save_project(ProjectFolderService(tmp_path), "excepted", Project.model_validate(raw))
    url = "/api/projects/excepted"
    endpoint = url + "/authoring/train/train/fields"
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        body = {"revision": client.get(url).json()["revision"]}
        fields = {field["id"]: field for field in client.post(endpoint, json=body).json()["fields"]}
        assert fields["/settings/optimizer/b1"]["can_sweep"]
        assert not fields["/settings/optimizer/learning_rate/value"]["can_sweep"]
        assert not fields["/training_config/batch_size"]["can_sweep"]

        recipe = client.post(endpoint, json=body | {
            "field": "/settings/optimizer/b1", "edit": {"kind": "text", "text": "0.8, 0.9"},
        }).json()
        assert not recipe["problems"], recipe
        client.patch(url, json=body | {"files": recipe["files"]}).raise_for_status()
        revision = client.get(url).json()["revision"]
    plan = ProjectFolderService(tmp_path).read_plan("excepted", revision=revision).plan
    keys = tuple(row.key for row in plan.rows if row.key.step == "train")
    assert len(keys) == 4
    plan = ProjectFolderService(tmp_path).read_plan("excepted", revision=revision, source_runs=keys).plan
    batch_sizes = sorted(
        (definition.payload["settings"]["optimizer"]["learning_rate"]["value"], definition.payload["training_config"]["batch_size"])
        for definition in plan.selected_definitions.values()
    )
    assert [size for value, size in batch_sizes if value == 0.003] == [8, 8]
    assert all(size != 8 for value, size in batch_sizes if value == 0.001)


def _seed_swept_project(tmp_path, name):
    raw = _model_project()
    raw["steps"].pop("baseline")
    raw["steps"]["train"]["base"]["inline"]["seed"] = 0
    raw["steps"]["train"]["sweep"] = {"axes": {"seed": {"values": {
        f"seed-{seed}": {"value": seed, "patches": {"seed": {"value": seed}}} for seed in (1, 2, 3)
    }}}}
    _save_project(ProjectFolderService(tmp_path), name, Project.model_validate(raw))
    return f"/api/projects/{name}"


class _TrainClient:
    """A Train step's edits through the Studio endpoints, each saved as it is made."""

    def __init__(self, client, url, *, accepted_steps=()):
        self.client, self.url = client, url
        self.accepted_steps = set(accepted_steps)

    @property
    def revision(self):
        return self.client.get(self.url).json()["revision"]

    def plan(self, tmp_path, name):
        folders = ProjectFolderService(tmp_path)
        plan = folders.read_plan(name, revision=self.revision).plan
        keys = tuple(row.key for row in plan.rows)
        return folders.read_plan(name, revision=self.revision, source_runs=keys).plan

    def requests(self, tmp_path, name):
        return {run.key: run.request_id for run in self.plan(tmp_path, name).runs}

    def runs(self, tmp_path, name, **coordinates):
        return [row.key for row in self.plan(tmp_path, name).rows if row.key.step == "train" and all(
            row.coordinates[axis].value_id == value for axis, value in coordinates.items()
        )]

    def save(self, response):
        assert response.status_code == 200, response.text
        body = response.json()
        if choices := body.get("store_choices"):
            assert not body["files"] and not body["created"]
            (choice,) = choices
            assert set(choice["steps"]) == self.accepted_steps, choice
            original = json.loads(response.request.content)
            response = self.client.post(response.request.url, json=original | {"store_choice": choice["id"]})
            assert response.status_code == 200, response.text
            body = response.json()
        assert not body.get("problems") and not body.get("refusal"), body
        assert body["files"], body.get("store_choices")
        saved = self.client.patch(self.url, json={"revision": self.revision, "files": body["files"]})
        assert saved.status_code == 200, saved.text
        return body

    def number(self, field, edit, keys=(), axes=None, *, save=True):
        response = self.client.post(self.url + "/authoring/train/train/fields", json=self.scope(keys, axes) | {
            "field": field, "edit": edit,
        })
        return self.save(response) if save else response.json()

    def choose(self, field, value, keys=(), axes=None):
        return self.save(self.client.post(self.url + "/authoring/train/train", json=self.scope(keys, axes) | {
            "change": {"field": field, "value": value},
        }))

    def sweep_class(self, field, edit, *, save=True):
        response = self.client.post(self.url + "/authoring/train/train", json=self.scope() | {
            "class_sweep": {"field": field, "edit": edit},
        })
        return self.save(response) if save else response.json()

    def read(self, keys=(), named=None):
        return self.client.post(self.url + "/authoring/train/train", json=self.scope(keys, named=named)).json()

    def scope(self, keys=(), axes=None, *, named=None):
        run = named or (keys[0] if keys else None)
        return {"revision": self.revision} | ({"run": {
            "variant": run.variant, "step": run.step, "row": run.row,
        }, "rows": [key.row for key in keys], "axes": axes or {}} if run else {})


def test_an_edit_for_a_sweep_value_is_stored_on_that_value(tmp_path):
    url = _seed_swept_project(tmp_path, "by-value")
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        train = _TrainClient(client, url)
        seed_2 = train.runs(tmp_path, "by-value", seed="seed-2")
        before = train.requests(tmp_path, "by-value")
        train.number("/training_config/n_batches", {"kind": "text", "text": "7"}, seed_2, {"seed": ["seed-2"]})
        after = train.requests(tmp_path, "by-value")
        assert {key for key in before if before[key] != after[key]} == set(seed_2)
        step = client.get(url + "/open").json()["snapshot"]["project"]["steps"]["train"]
        assert not step["sweep"].get("exceptions")

        train.number("/settings/optimizer/learning_rate/value", {"kind": "to_sweep"})
        train.number("/settings/optimizer/learning_rate/value", {"kind": "text", "text": "0.001, 0.003"})
        train.number("/seed", {"kind": "text", "text": "1, 2, 3, 4"})
        plan = train.plan(tmp_path, "by-value")
        batches = {
            (row.coordinates["seed"].value_id, row.coordinates["learning-rate"].value_id):
                plan.selected_definitions[row.key].payload["training_config"]["n_batches"]
            for row in plan.rows
        }
        assert len(batches) == 8
        assert {key: count for key, count in batches.items() if count != 2} == {
            ("seed-2", rate): 7 for rate in ("value", "value-2")
        }


def test_an_edit_for_runs_that_are_not_a_sweep_value_is_an_exception_for_those_runs(tmp_path):
    url = _seed_swept_project(tmp_path, "by-run")
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        train = _TrainClient(client, url)
        picked = train.runs(tmp_path, "by-run")[:2]
        before = train.requests(tmp_path, "by-run")
        train.number("/training_config/n_batches", {"kind": "text", "text": "7"}, picked, {"seed": ["seed-1", "seed-2", "seed-3"]})
        after = train.requests(tmp_path, "by-run")
        assert {key for key in before if before[key] != after[key]} == set(picked)
        exceptions = client.get(url + "/open").json()["snapshot"]["project"]["steps"]["train"]["sweep"]["exceptions"]
        assert [exception["where"] for exception in exceptions.values()] == [{"rows": sorted(key.row for key in picked)}]


def test_a_class_chosen_for_one_run_changes_that_run_and_reads_mixed_for_the_step(tmp_path):
    url = _seed_swept_project(tmp_path, "one-run")
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        train = _TrainClient(client, url)
        train.number("/settings/optimizer/learning_rate/value", {"kind": "text", "text": "0.001, 0.003"})
        first, *others = train.runs(tmp_path, "one-run")
        before = train.requests(tmp_path, "one-run")
        train.choose("/settings/optimizer", "adamw", [first])
        after = train.requests(tmp_path, "one-run")
        assert {key for key in before if before[key] != after[key]} == {first}
        plan = train.plan(tmp_path, "one-run")
        assert not plan.diagnostics
        assert plan.selected_definitions[first].payload["settings"]["optimizer"]["type"] == "adamw"
        assert {plan.selected_definitions[key].payload["settings"]["optimizer"]["type"] for key in others} == {"adam"}
        assert {"/settings/optimizer/type", "/settings/optimizer/weight_decay"} <= set(train.read(named=first)["mixed"])
        assert "/settings/optimizer/b1" not in train.read(named=first)["mixed"]
        assert not train.read([others[0]])["mixed"]


def _optimizer_swept_project(tmp_path, name):
    """Adam with b1 0.8 and every parameter stored, as Studio stores a chosen class, and the learning rate swept."""
    raw = _model_project()
    raw["steps"].pop("baseline")
    raw["steps"]["train"]["base"]["inline"]["settings"]["optimizer"] |= {
        "b1": 0.8, "b2": 0.999, "eps": 1e-8, "eps_root": 0.0, "nesterov": False,
    }
    _save_project(ProjectFolderService(tmp_path), name, Project.model_validate(raw))
    url = f"/api/projects/{name}"
    return url


def test_an_optimizer_swept_with_its_parameters_crosses_the_learning_rate(tmp_path):
    url = _optimizer_swept_project(tmp_path, "classes")
    rate, optimizer = "/settings/optimizer/learning_rate/value", "/settings/optimizer"
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        train = _TrainClient(client, url)
        train.number(rate, {"kind": "text", "text": "0.001, 0.003"})
        train.sweep_class(optimizer, {"kind": "to_sweep"})
        train.sweep_class(optimizer, {"kind": "add_member", "key": "sgd"})

        plan = train.plan(tmp_path, "classes")
        assert not plan.diagnostics and all(run.admission_problem is None for run in plan.runs)
        definitions = {row.key: plan.selected_definitions[row.key].payload["settings"]["optimizer"] for row in plan.rows}
        assert sorted((row.coordinates["optimizer"].label, row.coordinates["learning-rate"].value) for row in plan.rows) == [
            ("Adam", 0.001), ("Adam", 0.003), ("SGD", 0.001), ("SGD", 0.003),
        ]
        assert {row.label for row in plan.rows} == {
            f"learning-rate={rate}, optimizer={name}" for rate in (0.001, 0.003) for name in ("Adam", "SGD")
        }
        adam = train.runs(tmp_path, "classes", optimizer="adam")
        assert {definitions[key]["b1"] for key in adam} == {0.8}
        assert all("b1" not in optimizer_ for key, optimizer_ in definitions.items() if key not in adam)

        whole = train.read(named=adam[0])
        [field] = [field for field in whole["classes"] if field["id"] == optimizer]
        assert field["classes"] == ["adam", "sgd"]
        assert [(member["label"], member["value"]) for member in field["sweep"]["members"]] == [("Adam", "adam"), ("SGD", "sgd")]
        numbers = {field["id"]: field for field in client.post(
            url + "/authoring/train/train/fields", json=train.scope(named=adam[0]),
        ).json()["fields"]}
        assert "/settings/optimizer/b1" not in numbers and numbers[rate]["sweep"]["id"] == "learning-rate"

        group = {"optimizer": ["adam"]}
        before = train.requests(tmp_path, "classes")
        drawn = client.post(url + "/authoring/train/train/fields", json=train.scope(adam, group)).json()["fields"]
        assert "/settings/optimizer/b1" in {field["id"] for field in drawn}
        train.number("/settings/optimizer/b1", {"kind": "text", "text": "0.7"}, adam, group)
        after = train.requests(tmp_path, "classes")
        assert {key for key in before if before[key] != after[key]} == set(adam)

        train.sweep_class(optimizer, {"kind": "add_member", "key": "rmsprop"})
        assert len(train.plan(tmp_path, "classes").rows) == 6
        kept = {request for key, request in train.requests(tmp_path, "classes").items() if key in adam}
        body = train.sweep_class(optimizer, {"kind": "to_value", "value_id": "adam"})
        assert body["removed_values"] == {"optimizer": ["sgd", "rmsprop"]}
        assert set(train.requests(tmp_path, "classes").values()) == kept


def test_a_parameter_swept_for_one_optimizer_divides_only_its_runs(tmp_path):
    """The owner's case: Adam and SGD, b1 swept over 0.8 and 0.9 for Adam's runs only."""
    from tests.support.managed_train_fixture import managed_evaluation_step

    raw = _model_project()
    raw["steps"]["train"]["base"]["inline"]["settings"]["optimizer"] |= {
        "b1": 0.8, "b2": 0.999, "eps": 1e-8, "eps_root": 0.0, "nesterov": False,
    }
    raw["steps"]["baseline"]["sweep"] = {"axes": {"subject": {"input": "subject"}}}
    raw["steps"]["picked"] = managed_evaluation_step("train")
    raw["steps"]["train-2"] = {
        "kind": "train", "base": {"step": "train"}, "inputs": {"checkpoint": {"step": "train", "checkpoint": "last"}},
        "deltas": [{"layer_id": "initialize", "patches": [
            {"path": "checkpoint_initialization", "op": "add", "value": "initialize_from"},
        ]}],
    }
    name, optimizer, b1 = "conditional", "/settings/optimizer", "/settings/optimizer/b1"
    _save_project(ProjectFolderService(tmp_path), name, Project.model_validate(raw))
    url = f"/api/projects/{name}"
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        train = _TrainClient(client, url, accepted_steps=("train", "train-2"))
        train.sweep_class(optimizer, {"kind": "to_sweep"})
        train.sweep_class(optimizer, {"kind": "add_member", "key": "sgd"})
        before = train.requests(tmp_path, name)
        adam, group = train.runs(tmp_path, name, optimizer="adam"), {"optimizer": ["adam"]}

        def fields(keys, axes=None):
            body = client.post(url + "/authoring/train/train/fields", json=train.scope(keys, axes)).json()
            return {field["id"]: field for field in body["fields"]}

        assert fields(adam, group)[b1]["can_sweep"]
        train.number(b1, {"kind": "to_sweep"}, adam, group)
        adam = train.runs(tmp_path, name, optimizer="adam")
        train.number(b1, {"kind": "text", "text": "0.8, 0.9"}, adam, group)

        plan = train.plan(tmp_path, name)
        runs = [row for row in plan.rows if row.key.step == "train"]
        axis = next(axis for axis in runs[0].coordinates if axis != "optimizer")
        coordinate = {row.key: (row.coordinates["optimizer"].label, getattr(row.coordinates.get(axis), "value", None))
                      for row in runs}
        assert sorted(coordinate.values(), key=str) == [("Adam", 0.8), ("Adam", 0.9), ("SGD", None)]
        assert {row.label for row in runs} == {
            f"optimizer=Adam, {axis}=0.8", f"optimizer=Adam, {axis}=0.9", "optimizer=SGD",
        }
        (sgd,) = (key for key, value in coordinate.items() if value[0] == "SGD")
        assert train.requests(tmp_path, name)[sgd] == before[sgd]
        assert {key: plan.selected_definitions[key].payload["settings"]["optimizer"].get("b1") for key in coordinate} == {
            key: value for key, (_, value) in coordinate.items()
        }
        assert fields(adam, group)[b1]["sweep"]["id"] == axis
        rate = optimizer + "/learning_rate/value"
        assert fields([sgd], {"optimizer": ["sgd"]})[rate]["can_sweep"] and not fields(adam[:1])[rate]["can_sweep"]

        evaluated = {
            row.key.row: row.coordinates for row in plan.rows if row.key.step == "baseline"
        }
        assert sorted(
            (coordinates["optimizer"].label, getattr(coordinates.get(axis), "value", None)) for coordinates in evaluated.values()
        ) == sorted(coordinate.values(), key=str)
        continued = {run.key.row: run.inputs["checkpoint"].producer for run in plan.runs if run.key.step == "train-2"}
        assert {row: producer.row for row, producer in continued.items()} == {key.row: key.row for key in coordinate}

        train.save(Mock(status_code=200, json=lambda: {"files": [{
            "address": {"kind": "step", "id": "picked"},
            "patches": [{"op": "add", "path": "/inputs/subject/where", "value": {"rule": f"{axis}=0.9"}}],
        }]}))
        picked = [run for run in train.plan(tmp_path, name).runs if run.key.step == "picked"]
        assert [run.inputs["subject"].producer.row for run in picked] == [
            key.row for key, value in coordinate.items() if value == ("Adam", 0.9)
        ]

        train.number(optimizer + "/learning_rate/value", {"kind": "text", "text": "0.001, 0.003"})
        assert len([row for row in train.plan(tmp_path, name).rows if row.key.step == "train"]) == 6


def test_a_rate_swept_for_adam_only_reads_as_the_selected_runs_have_it(tmp_path):
    """The whole step reads Mixed with the runs' values; Adam's runs read the sweep; SGD's runs read their one value."""
    url = _optimizer_swept_project(tmp_path, "adam-rate")
    optimizer, rate = "/settings/optimizer", "/settings/optimizer/learning_rate/value"
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        train = _TrainClient(client, url)
        train.sweep_class(optimizer, {"kind": "to_sweep"})
        train.sweep_class(optimizer, {"kind": "add_member", "key": "sgd"})
        adam, group = train.runs(tmp_path, "adam-rate", optimizer="adam"), {"optimizer": ["adam"]}
        train.number(rate, {"kind": "to_sweep"}, adam, group)
        adam = train.runs(tmp_path, "adam-rate", optimizer="adam")
        train.number(rate, {"kind": "text", "text": "0.002, 0.003"}, adam, group)
        adam, sgd = (train.runs(tmp_path, "adam-rate", optimizer=key) for key in ("adam", "sgd"))
        (held,) = {
            definition.payload["settings"]["optimizer"]["learning_rate"]["value"]
            for key, definition in train.plan(tmp_path, "adam-rate").selected_definitions.items() if key in sgd
        }

        def field(keys=(), axes=None, named=None):
            body = client.post(url + "/authoring/train/train/fields", json=train.scope(keys, axes, named=named)).json()
            return next(item for item in body["fields"] if item["id"] == rate)

        whole = field(named=adam[0])
        assert whole["sweep"] is None and whole["mixed"]
        assert sorted(member["value"] for member in whole["members"]) == sorted([held, 0.002, 0.003])
        swept = field(adam, group)
        assert [value["value"] for value in swept["sweep"]["values"]] == [0.002, 0.003] and swept["can_change_axis"]
        assert field(adam[:1])["sweep"]["values"]
        other = field(sgd, {"optimizer": ["sgd"]})
        assert other["sweep"] is None and not other["mixed"] and other["text"] == str(held)


def test_a_class_swept_for_some_runs_reads_as_the_selected_runs_have_it(tmp_path):
    """The optimizer swept for the fast rate's runs only: Mixed for the step, the sweep for those runs, Adam for the others."""
    url = _optimizer_swept_project(tmp_path, "fast-optimizer")
    optimizer, rate = "/settings/optimizer", "/settings/optimizer/learning_rate/value"
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        train = _TrainClient(client, url)
        train.number(rate, {"kind": "text", "text": "0.001, 0.003"})
        train.sweep_class(optimizer, {"kind": "to_sweep"})
        train.sweep_class(optimizer, {"kind": "add_member", "key": "sgd"})
        rows = train.plan(tmp_path, "fast-optimizer").rows
        slow, fast = sorted({row.coordinates["learning-rate"].value_id for row in rows},
                            key=lambda value_id: next(row.coordinates["learning-rate"].value for row in rows
                                                      if row.coordinates["learning-rate"].value_id == value_id))
        train.save(Mock(status_code=200, json=lambda: {"files": [{
            "address": {"kind": "step", "id": "train"},
            "patches": [{"op": "add", "path": "/sweep/axes/optimizer/where", "value": {"rule": f"learning-rate={fast}"}}],
        }]}))
        assert len(train.plan(tmp_path, "fast-optimizer").rows) == 3
        fast_runs = train.runs(tmp_path, "fast-optimizer", **{"learning-rate": fast})
        slow_runs = train.runs(tmp_path, "fast-optimizer", **{"learning-rate": slow})

        def classes(keys=(), axes=None, named=None):
            body = client.post(url + "/authoring/train/train", json=train.scope(keys, axes, named=named)).json()
            return next(item for item in body["classes"] if item["id"] == optimizer), body["mixed"]

        whole, mixed = classes(named=fast_runs[0])
        assert whole["sweep"] is None and whole["classes"] == ["adam", "sgd"] and "/settings/optimizer/type" in mixed
        swept, _ = classes(fast_runs, {"learning-rate": [fast]})
        assert [member["value"] for member in swept["sweep"]["members"]] == ["adam", "sgd"]
        other, mixed = classes(slow_runs, {"learning-rate": [slow]})
        assert other["sweep"] is None and other["classes"] == ["adam"] and "/settings/optimizer/type" not in mixed


def test_a_model_field_swept_for_some_runs_reads_as_the_selected_runs_have_it(tmp_path):
    """Mass swept for seed 1's runs only: Mixed for the step, the sweep for those runs, the one mass for the others."""
    url = _seed_swept_project(tmp_path, "seed-mass")
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        train = _TrainClient(client, url)
        plant = {"composites": [], "element": {"kind": "component", "id": "plant"}}

        def mass(keys=(), axes=None, named=None, **edit):
            body = train.scope(keys, axes, named=named) | {"document": "reach-model", "owner": plant} | edit
            response = client.post(url + "/model/fields", json=body)
            return response if edit else next(item for item in response.json()["fields"] if item["label"] == "Mass")

        (default,) = train.runs(tmp_path, "seed-mass", seed="seed-1")
        train.save(mass(named=default, address=mass(named=default)["address"], edit={"kind": "text", "text": "2, 3"}))
        axis = next(name for name in client.get(url + "/open").json()["snapshot"]["project"]["steps"]["train"]["sweep"]["axes"]
                    if name != "seed")
        train.save(Mock(status_code=200, json=lambda: {"files": [{
            "address": {"kind": "step", "id": "train"},
            "patches": [{"op": "add", "path": f"/sweep/axes/{axis}/where", "value": {"rule": 'seed="seed-1"'}}],
        }]}))
        first = train.runs(tmp_path, "seed-mass", seed="seed-1")
        (second,) = train.runs(tmp_path, "seed-mass", seed="seed-2")
        assert len(first) == 2

        whole = mass(named=first[0])
        assert whole["sweep"] is None and whole["mixed"]
        assert sorted(member["value"] for member in whole["members"]) == [1.0, 2, 3]
        assert [value["value"] for value in mass(first, {"seed": ["seed-1"]})["sweep"]["values"]] == [2, 3]
        other = mass([second], {"seed": ["seed-2"]})
        assert other["sweep"] is None and not other["mixed"] and other["text"] == "1.0"


def test_a_class_without_a_swept_parameter_keeps_the_authored_axis(tmp_path):
    raw = managed_train_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    raw["steps"]["train"]["base"]["inline"]["settings"]["optimizer"]["learning_rate"] = {
        "type": "warmup_cosine_decay_schedule", "init_value": 0.0, "peak_value": 0.001, "warmup_steps": 10,
        "decay_steps": 100, "end_value": 0.0,
    }
    _save_project(ProjectFolderService(tmp_path), "peak", Project.model_validate(raw))
    url, schedule = "/api/projects/peak", "/settings/optimizer/learning_rate"
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        train = _TrainClient(client, url)
        train.number(schedule + "/peak_value", {"kind": "text", "text": "0.001, 0.002"})
        train.sweep_class(schedule, {"kind": "to_sweep"})
        recipe = train.sweep_class(schedule, {"kind": "add_member", "key": "constant_schedule"})
        assert recipe["files"]
        revision = client.get(url).json()["revision"]
        read = client.post(url + "/authoring/train/train", json={"revision": revision})
        assert read.status_code == 200, read.text
        reopened = read.json()
        (field,) = [field for field in reopened["classes"] if field["id"] == schedule]
        assert "constant_schedule" in {value["value"] for value in field["sweep"]["members"]}
        assert any("peak_value" in problem["path"] for problem in reopened["sweep_problems"])


def test_a_sweep_turned_into_its_value_keeps_the_edits_made_for_that_value(tmp_path):
    url = _seed_swept_project(tmp_path, "kept")
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        train = _TrainClient(client, url)
        seed_2 = train.runs(tmp_path, "kept", seed="seed-2")
        train.number("/training_config/n_batches", {"kind": "text", "text": "7"}, seed_2, {"seed": ["seed-2"]})
        before = train.requests(tmp_path, "kept")[seed_2[0]]
        train.number("/seed", {"kind": "to_value", "value_id": "seed-2"})
        assert list(train.requests(tmp_path, "kept").values()) == [before]


def test_managed_rate_field_keeps_one_authored_geomspace_rule(tmp_path):
    raw = _model_project()
    raw["steps"].pop("baseline")
    _save_project(ProjectFolderService(tmp_path), "managed-rate", Project.model_validate(raw))
    url = "/api/projects/managed-rate"
    field_path = "/settings/optimizer/learning_rate/value"
    endpoint = url + "/authoring/train/train/fields"
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        initial = client.get(url).json()
        body = {"revision": initial["revision"]}
        fields = client.post(endpoint, json=body).json()["fields"]
        rate, = [field for field in fields if field["id"] == field_path]
        assert rate["can_sweep"]
        response = client.post(endpoint, json=body | {
            "field": field_path, "edit": {"kind": "text", "text": "geomspace(1e-4, 1e-2, 5)"},
        })
        assert response.status_code == 200, response.text
        recipe = response.json()
        assert not recipe["problems"], recipe
        updated = client.patch(url, json=body | {"files": recipe["files"]})
        assert updated.status_code == 200, updated.text
        reopened = client.get(url + "/open").json()["snapshot"]
        step = reopened["project"]["steps"]["train"]
        assert "learning_rate" not in step["base"]["inline"]["training_config"]
        assert step["sweep"]["axes"]["learning-rate"] == {
            "path": field_path[1:].replace("/", "."),
            "range": {"start": 1e-4, "stop": 1e-2, "count": 5, "scale": "log"},
        }
        fields = client.post(endpoint, json={"revision": reopened["revision"]}).json()["fields"]
        assert next(field["text"] for field in fields if field["id"] == field_path) == "geomspace(0.0001, 0.01, 5)"
        folders = ProjectFolderService(tmp_path)
        plan = folders.read_plan("managed-rate", revision=reopened["revision"]).plan
        keys = tuple(row.key for row in plan.rows)
        assert len(keys) == 5
        plan = folders.read_plan("managed-rate", revision=reopened["revision"], source_runs=keys).plan
        for key, expected in zip(keys, np.geomspace(1e-4, 1e-2, 5), strict=True):
            resolved = plan.selected_definitions[key]
            assert "learning_rate" not in resolved.payload["training_config"]
            assert resolved.payload["settings"]["optimizer"]["learning_rate"]["value"] == pytest.approx(expected)
            assert resolved.sources["/payload" + field_path].kind == "sweep_value"

        body = {"revision": reopened["revision"]}
        edit = lambda text: client.post(endpoint, json=body | {"field": field_path, "edit": {"kind": "text", "text": text}}).json()
        listed = edit("0.0001, 0.01")
        assert not listed["problems"], listed
        assert {axis: len(values) for axis, values in listed["removed_values"].items()} == {"learning-rate": 3}
        single = edit("0.0001")
        assert not single["problems"], single
        assert {axis: len(values) for axis, values in single["removed_values"].items()} == {"learning-rate": 4}
        updated = client.patch(url, json=body | {"files": single["files"]})
        assert updated.status_code == 200, updated.text
        reduced = client.get(url + "/open").json()["snapshot"]
        assert not any(((reduced["project"]["steps"]["train"].get("sweep") or {}).get("axes") or {}).values())
        assert len(folders.read_plan("managed-rate", revision=reduced["revision"]).plan.rows) == 1


def _group_reaching_pipeline(raw, registry):
    payload = raw["documents"]["reach-model"]["payload"]
    payload["interior"]["components"]["relay"] = {
        "kind": "leaf", "name": "Input relay", "type": "feedbax.Input",
    }
    payload["interior"]["wires"]["task-wire"]["target"] = _end("relay", "input")
    payload["interior"]["wires"]["relay-wire"] = _wire(
        _end("relay", "output"), _end("wrapper", "target"),
    )
    ports = ModelRegistryPorts(registry)
    state = ModelEditState(document=ModelDocument.model_validate(payload))
    command = ModelEditCommand.model_validate({"operation": {
        "op": "group", "name": "Group",
        "selection": [
            {"composites": [], "element": {"kind": "component", "id": name}}
            for name in ("relay", "plant")
        ],
    }})
    result = apply_model_edit(
        state, command, ports=ports.port, component_problem=ports.placement_problem,
    )
    assert isinstance(result, ModelEditResult), result
    raw["documents"]["reach-model"]["payload"] = result.state.document.model_dump(mode="json")
    group = result.changed[-1].element.id
    for link in raw["documents"]["task"]["payload"]["links"]:
        address = ModelAddress.from_text(link["model_field"])
        link["model_field"] = address.model_copy(update={"composites": (group, *address.composites)}).to_text()
    return group


def test_nested_model_plans_with_stable_identity_and_compiles():
    raw = _model_project()
    before = compile_project(Project.model_validate(raw))
    assert not before.diagnostics
    _rename_components(raw["documents"]["reach-model"]["payload"])
    after = compile_project(Project.model_validate(raw))
    assert not after.diagnostics
    assert [run.request_id for run in before.runs] == [run.request_id for run in after.runs]
    assert before.requests == after.requests
    assert before.documents == after.documents

    run = next(run for run in before.runs if run.kind == "train")
    spec = decode_request(before.requests[run.request_id], before.documents.__getitem__).spec
    registry = new_application_registry_bundle()
    graph = _prepare_request(spec, registry).executable.graph
    assert graph.period_seconds == spec.trial_documents.model.step_seconds == graph.nodes["plant"].dt
    assert graph.nodes["wrapper"].nodes["controller"].period_seconds == graph.period_seconds
    assert graph.nodes["wrapper"].nodes["controller"].nodes["cell"].cell.weight_ih.shape[1] == 2

    documents = raw["steps"]["train"]["base"]["inline"]["trial_documents"]
    documents.pop("input_bindings")
    raw["documents"]["reach-model"]["payload"]["ports"]["inputs"][0]["name"] = "unknown"
    refused = compile_project(Project.model_validate(raw))
    (problem,) = [problem for row in refused.rows if row.key.step == "train" for problem in row.problems]
    assert problem.path.startswith("/trial_documents/model/")
    assert problem.detail == "Model input 'unknown' needs a Task signal."
    assert problem.model_address == ModelAddress.model_validate({
        "composites": [],
        "element": {"kind": "port", "end": {"owner": "inputs", "port": "task-input"}},
    })


def _use_initial_evaluation(raw):
    trial = raw["steps"]["train"]["base"]["inline"]["trial_documents"]
    evaluation = raw["steps"]["baseline"]
    evaluation["inputs"] = {
        "subject": {"kind": "initial_model", "source": trial["model"]},
        "task": {"kind": "task", "source": trial["task"]},
        "objective": {"kind": "objective", "source": trial["objective"]},
    }
    evaluation["base"]["inline"]["params"]["input_bindings"] = dict(trial["input_bindings"])


def test_plan_reports_all_authored_graph_problems_for_train_and_initial_evaluate():
    """Catch a dangling wire in an invalid Model masking an independently repairable wire."""
    raw = _model_project()
    _use_initial_evaluation(raw)
    wires = raw["documents"]["reach-model"]["payload"]["interior"]["wires"]
    for key in ("dangling-first", "dangling-second"):
        wires[key] = _wire({"owner": "inputs", "port": key}, {"owner": "outputs", "port": key})
    plan = compile_project(Project.model_validate(raw))
    for step in ("train", "baseline"):
        row, = (row for row in plan.rows if row.key.step == step)
        assert {item.model_address.element.id for item in row.problems} == {"dangling-first", "dangling-second"}


def test_unwired_linear_review_accepts_the_gru_width(tmp_path):
    from feedbax.contracts.model_edit import ModelConnectionReviewRequest
    from tests.support.managed_train_fixture import managed_train_project

    data = managed_train_project().model_dump(mode="json", by_alias=True)
    interior = data["documents"]["point-mass"]["payload"]["interior"]
    for wire in ("target", "decode"):
        del interior["wires"][wire]
    interior["components"]["scalar"] = {"kind": "leaf", "name": "Scalar", "type": "feedbax.Constant", "fields": {"value": 2.0}}
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, "unwired", Project.model_validate(data))
    revision = folders.read("unwired").revision
    registry = new_application_registry_bundle()

    def candidate(component):
        review = folders.review_model_connections("unwired", ModelConnectionReviewRequest.model_validate({
            "revision": revision, "document": "point-mass",
            "graph": {"composites": [], "element": {"kind": "graph"}},
            "source": {"owner": "component", "component": component, "port": "output"},
        }), registry)
        return next(item for item in review.targets if item.target.owner == "component"
                    and item.target.component == "readout" and item.target.port == "input")

    assert candidate("cell").reason is None and candidate("cell").problem is None
    assert "shape" in candidate("scalar").problem


@pytest.mark.parametrize("source", ["gru", "task"])
def test_model_projection_and_compilation_get_input_size_from_the_wire(tmp_path, source):
    """Changing a GRU's hidden size also changes the wired Linear's input width."""
    raw = _model_project()
    interior = raw["documents"]["reach-model"]["payload"]["interior"]
    interior["components"]["receiver"] = {"kind": "leaf", "name": "Receiver", "type": "feedbax.Linear"}
    if source == "gru":
        interior["components"]["sender"] = {
            "kind": "leaf", "name": "Sender", "type": "feedbax.GRU", "fields": {"hidden_size": 4},
        }
        interior["wires"]["sender-input"] = _wire({"owner": "inputs", "port": "task-input"}, _end("sender", "input"))
        interior["wires"]["receiver-input"] = _wire(_end("sender", "output"), _end("receiver", "input"))
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, "nested", Project.model_validate(raw))
    document = "reach-model"
    graph = {"composites": [], "element": {"kind": "graph"}}
    run = {"variant": "default", "step": "train", "row": "default"}
    base = "/api/projects/nested/model"

    with TestClient(create_app(projects_dir=tmp_path)) as client:
        def apply(operation):
            snapshot = folders.read("nested")
            reply = client.post(base + "/edit", json={
                "revision": snapshot.revision, "presentation_revision": snapshot.presentation_revision,
                "document": document, "command": {"operation": operation},
            })
            assert reply.status_code == 200, reply.text
            result = reply.json()["result"]
            assert result["status"] == "applied", result
            return result

        def projection(snapshot):
            reply = client.post(base + "/projection", json={
                "revision": snapshot.revision, "document": document, "run": run,
            })
            assert reply.status_code == 200, reply.text
            return reply.json()

        def projected_type(view, end):
            return next(
                item["type"] for item in view["ports"]
                if not item["address"]["composites"] and item["address"]["element"].get("end") == end
            )

        if source == "task":
            apply({
                "op": "connect_task_signal", "graph": graph, "run": run,
                "signal": "target_pos", "target": _end("receiver", "input"),
            })
            snapshot = folders.read("nested")
            model = snapshot.project.documents[document].payload
            task_port = next(port for port in model["ports"]["inputs"] if port["name"] == "target_pos")
            task_type = {"base": "real", "shape": [2], "unit": "m"}
            assert task_port["type"] == task_type
            assert projected_type(projection(snapshot), {"owner": "inputs", "port": task_port["id"]}) == task_type

        if source == "gru":
            apply({
                "op": "set_field", "address": {"composites": [], "element": {
                    "kind": "field", "parent": {"kind": "component", "id": "sender"}, "path": ["fields", "hidden_size"],
                }}, "value": 7,
            })
        saved = folders.read("nested")
        view = projection(saved)
        assert projected_type(view, _end("receiver", "input") | {"direction": "input"})["shape"] == (
            [7] if source == "gru" else [2]
        )
        model = saved.project.documents[document].payload
        assert "input_size" not in model["interior"]["components"]["receiver"].get("fields", {})
        row, = (row for row in folders.read_plan("nested", revision=saved.revision).plan.rows if row.key.step == "train")
        assert not row.problems, row.problems
        linear = client.get("/api/components/feedbax.Linear").json()["data"]
        assert "input_size" not in linear["default_params"]
        assert "input_size" not in {item["name"] for item in linear["param_schema"]}


def test_unresolved_relative_link_is_reported_at_the_task():
    from tests.test_document_train_pair import authored_documents, document_project

    raw = document_project(authored_documents(), evaluation=True)
    _use_initial_evaluation(raw)
    link = raw["documents"]["task"]["payload"]["links"][0]
    address = ModelAddress.from_text(link["model_field"])
    missing = address.model_copy(update={"composites": ("absent",)}).to_text()
    link["model_field"] = missing
    plan = compile_project(Project.model_validate(raw))

    assert not plan.diagnostics
    assert {(run.key.step, tuple(problem.path for problem in run.problems)) for run in plan.runs} == {
        ("train", (pointer(("trial_documents", "task", "links", missing)),)),
        ("baseline", (pointer(("inputs", "task", "links", missing)),)),
    }


def test_stored_model_plan_uses_the_supplied_component_registry():
    class CustomRelay(Input):
        model_facts = None

    raw = _model_project()
    model = raw["documents"]["reach-model"]["payload"]
    model["interior"]["components"]["custom"] = {
        "kind": "leaf", "name": "Custom relay", "type": "test.CustomRelay",
    }
    model["interior"]["wires"]["task-wire"]["target"] = _end("custom", "input")
    model["interior"]["wires"]["custom-wire"] = _wire(_end("custom", "output"), _end("wrapper", "target"))
    project = Project.model_validate(raw)
    memo = ProjectPlanMemo()
    first = compile_project(project, memo=memo)
    (unregistered,) = [run for run in first.runs if run.kind == "train"]
    assert any("unknown leaf type 'test.CustomRelay'" in problem.detail for problem in unregistered.problems)
    registry = ComponentRegistry()
    register_owners(registry, "test", CustomRelay)
    registry.seal()
    registered = compile_project(project, memo=memo, components=registry)
    (train,) = registered.runs
    assert train.kind == "train"
    problem, blocked = sorted(registered.diagnostics, key=lambda item: item.code == "blocked_input")
    assert problem.step == "baseline" and "has no declared Evaluate layout" in problem.detail
    assert (blocked.step, blocked.row, blocked.code, blocked.blocked_by) == (
        "baseline", problem.row, "blocked_input", train.key,
    )
    assert train.admission_problem is not None
    assert "has no typed model checkpoint owner" in train.admission_problem.detail
    address = ModelAddress.model_validate({
        "composites": [],
        "element": {"kind": "component", "id": "custom"},
    })
    assert problem.model_address == train.admission_problem.model_address == address
    assert compile_project(project, memo=memo) == first


def test_generated_delay_and_input_relay_keep_the_authored_source_addresses():
    raw = _model_project()
    model = ModelDocument.model_validate(raw["documents"]["reach-model"]["payload"])
    lowered = lower_model_document(model, ComponentRegistry())
    (root_relay, _) = lowered.graph.input_bindings["reach_target"]
    port = lowered.source_address(f"/graph/nodes/{root_relay}/input")
    assert port.element.end.port == "task-input"
    controller = lowered.graph.subgraphs["wrapper"].subgraphs["controller"]
    (index,) = (
        index for index, wire in enumerate(controller.wires)
        if wire.target_node.startswith("_model_delay_")
    )
    address = lowered.source_address(f"/graph/subgraphs/wrapper/subgraphs/controller/wires/{index}")
    assert address.composites == ("wrapper", "controller")
    assert (address.element.kind, address.element.id, address.element.parent.id) == (
        "attachment", "carry", "carry",
    )


def test_nested_evaluate_variation_uses_scoped_wire_ids(monkeypatch):
    from tests.test_document_train_pair import authored_documents, document_project

    raw = document_project(authored_documents(), evaluation=True)
    _swept_attachments(raw, "baseline", "decode", [_scale_attachment("scale", 0.5)])
    memo = ProjectPlanMemo()
    train_key = RunKey("default", "train", "default")
    encode = Mock(wraps=request_identity.encode_request)
    monkeypatch.setattr(request_identity, "encode_request", encode)

    def compile_incrementally():
        project = Project.model_validate(raw)
        encode.reset_mock()
        incremental = compile_project(project, memo=memo, source_runs=(train_key,))
        calls = sum(call.kwargs.get("document_memo") is not None for call in encode.call_args_list)
        assert incremental == compile_project(project, source_runs=(train_key,))
        return incremental, calls

    plan, calls = compile_incrementally()
    assert calls == 2
    assert not plan.diagnostics
    train = next(run for run in plan.runs if run.kind == "train")
    request = decode_request(plan.requests[train.request_id], plan.documents.__getitem__).spec
    documents = request.trial_documents
    registry = ComponentRegistry()
    graph = lower_model_document(documents.model, registry).graph
    evaluate = next(run for run in plan.runs if run.kind == "evaluate")
    params = RolloutParameters.model_validate(
        decode_request(plan.requests[evaluate.request_id], plan.documents.__getitem__).spec.params,
    )
    variant = resolve_evaluation_variant(graph, documents.task_source, params.variation, components=registry)
    child = variant.graph.subgraphs["wrapper"].subgraphs["controller"]
    (outgoing,) = (wire for wire in child.wires if wire.id == "decode")
    (incoming,) = (wire for wire in child.wires if wire.target_node == outgoing.source_node)
    assert outgoing.target_node == "readout"
    executable = compile_graph(
        GraphDocument(graph=variant.graph), registry,
        input_prototypes={graph.input_bindings["reach_target"]: jax.ShapeDtypeStruct((2,), jnp.float32)},
    )
    assert executable.graph.nodes["wrapper"].nodes["controller"].nodes[incoming.target_node].scale == 0.5

    raw["documents"]["renamed"] = raw["documents"].pop("reach-model")
    raw["steps"]["train"]["base"]["inline"]["trial_documents"]["model"] = {"document": "renamed"}
    baseline = raw["steps"]["baseline"]
    baseline["base"]["inline"]["trial_documents"]["model"] = {"document": "renamed"}
    patches = baseline["sweep"]["axes"]["edit"]["values"]["on"]["patches"]
    baseline["sweep"]["axes"]["edit"]["values"]["on"]["patches"] = {
        path.replace("documents.reach-model.", "documents.renamed."): patch for path, patch in patches.items()
    }
    renamed, calls = compile_incrementally()
    assert calls == 0
    assert renamed.requests == plan.requests and renamed.documents == plan.documents
    source_path = "/payload/trial_documents/model/step_seconds"
    assert renamed.sources[train_key][source_path].name == "renamed"


def test_train_model_attachment_rebinds_a_pointer_with_the_same_authored_value():
    raw = _model_project()
    del raw["steps"]["baseline"]
    model = raw["documents"]["reach-model"]["payload"]
    model["interior"]["components"]["plant"]["fields"]["mass"] = model["step_seconds"] = 0.01
    train = raw["steps"]["train"]
    optimizer = train["base"]["inline"]["settings"]["optimizer"]["learning_rate"]
    mass = "/interior/components/plant/fields/mass"
    changed = read_changed_model(raw, "train", [{
        "op": "set_field",
        "address": {"composites": [], "element": {
            "kind": "field", "parent": {"kind": "component", "id": "plant"}, "path": ["fields", "mass"],
        }},
        "value": 0.02,
    }])
    alias = {"document": changed, "pointer": mass}
    optimizer["value"] = alias
    memo = ProjectPlanMemo()

    def train_request(expected_rate):
        project = Project.model_validate(raw)
        plan = compile_project(project, memo=memo)
        assert not plan.diagnostics
        assert plan == compile_project(project)
        (run,) = plan.runs
        spec = decode_request(plan.requests[run.request_id], plan.documents.__getitem__).spec
        assert spec.settings["optimizer"]["learning_rate"]["value"] == expected_rate
        return run.request_id

    before = train_request(0.02)
    alias["pointer"] = "/step_seconds"
    assert train_request(0.01) != before


@pytest.mark.parametrize("selection", [
    {"rows": ["seed=control"]},
    {"rows": ["seed=control"], "axes": {"seed": ["control"]}},
])
def test_an_evaluate_attachment_for_selected_runs_changes_only_their_request(tmp_path, selection):
    raw = _model_project()
    raw["steps"]["baseline"]["sweep"] = {"axes": {"seed": {"values": {
        "control": {"patches": {"params.seed": {"op": "replace", "value": 2}}},
        "comparison": {"patches": {"params.seed": {"op": "replace", "value": 3}}},
    }}}}
    selected = RunKey("default", "baseline", "seed=control")
    project = Project.model_validate(raw)
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, "nested", project)
    before = folders.read_plan("nested", revision=None)
    owner = {"composites": ["wrapper", "controller"], "element": {
        "kind": "port", "end": {"owner": "component", "component": "cell", "port": "output", "direction": "output"},
    }}
    attachment = {"kind": "intervention", "id": "condition", "operation": "Scale", "operands": {"factor": {"kind": "value", "value": 0.5}}}

    def edit(client, difference):
        snapshot = folders.read("nested")
        reply = client.post("/api/projects/nested/model/edit", json={
            "revision": snapshot.revision, "presentation_revision": snapshot.presentation_revision,
            "document": "reach-model", "step": "baseline",
            "command": {"operation": {"op": "set_attachment", "difference": difference}},
        } | selection)
        assert reply.status_code == 200, reply.text
        return reply.json()["result"]

    def requests():
        return {run.key: run.request_id for run in folders.read_plan("nested", revision=None).plan.runs}

    prior = {run.key: run.request_id for run in before.plan.runs}
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        missing = edit(client, {"op": "add", "after": None, "owner": owner | {
            "composites": ["wrapper", "controller", "missing"]}, "attachment": attachment})
        assert missing["status"] == "refused"
        added = edit(client, {"op": "add", "after": None, "owner": owner, "attachment": attachment})
        assert added["status"] == "applied", added
        result = requests()
        assert result.keys() == prior.keys()
        assert {key for key in prior if result[key] != prior[key]} == {selected}
        assert folders.read("nested").project.documents == project.documents
        assert "model_attachments" not in (tmp_path / "nested" / "steps/baseline.json").read_text()
        address = added["changed"][0]
        changed = edit(client, {"op": "change", "address": address, "attachment": attachment | {"fields": {"factor": 0.25}}})
        assert changed["status"] == "applied", changed
        updated = requests()
        assert {key for key in updated if updated[key] != result[key]} == {selected}
        removed = edit(client, {"op": "remove", "address": address})
        assert removed["status"] == "applied", removed
    assert requests() == prior


# The ids the variant's evaluation had at a9b3bb7d, where the change was stored in the step's payload.
# The fixture pins requests that bind Task signals by stable Model input ID (8a0009c3).
_VARIANT_WIRE = "evaluate:4f26103959b7ccd1c1f27ba1acd2f8b53bb8f7446751b8c279f7345bdc3f7a4e"
_VARIANT_FIELD = "evaluate:86530315779f1c0f14fc177396bae4f98e4f8ceb8a5600d09bd6b21232ba0374"


@pytest.mark.parametrize(("operation", "expected"), [
    ({"op": "set_attachment", "difference": {
        "op": "add", "after": None,
        "owner": {"composites": ["wrapper", "controller"], "element": {"kind": "wire", "id": "decode"}},
        "attachment": {"kind": "intervention", "id": "condition", "operation": "Scale", "operands": {"factor": {"kind": "value", "value": 0.5}}},
    }}, _VARIANT_WIRE),
    ({"op": "set_field", "address": {"composites": [], "element": {
        "kind": "field", "parent": {"kind": "component", "id": "plant"}, "path": ["fields", "mass"],
    }}, "value": 2.5}, _VARIANT_FIELD),
])
def test_a_variant_edit_of_an_evaluation_is_a_document_the_variant_points_it_at(tmp_path, operation, expected):
    raw = _model_project()
    raw["sweep"] = {"rows": {"control": {}, "treated": {}}}
    project = Project.model_validate(raw)
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, "nested", project)
    before = folders.read_plan("nested", revision=None)
    snapshot = folders.read("nested")
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        request = {
            "revision": snapshot.revision, "presentation_revision": snapshot.presentation_revision,
            "document": "reach-model", "command": {"operation": operation}, "step": "baseline", "variant": "treated",
        }
        reply = client.post("/api/projects/nested/model/edit", json=request)
        assert reply.status_code == 200, reply.text
        decision = reply.json()["result"]
        assert decision["status"] == "asked", decision
        store = next(option for option in decision["store_choices"] if option["kind"] == "selected_readers")
        reply = client.post("/api/projects/nested/model/edit", json=request | {"store_choice": store["id"]})
        assert reply.status_code == 200, reply.text
        result = reply.json()["result"]
    assert result["status"] == "applied", result
    (created,) = result["created"]
    edited = folders.read("nested").project
    assert edited.documents["reach-model"] == project.documents["reach-model"]
    assert edited.documents[created].base.document == "reach-model"
    assert edited.steps == project.steps
    assert {path: write.model_dump(mode="json") for path, write in edited.sweep.rows["treated"].patches.items()} == {
        "steps.baseline.base.inline.trial_documents": {"op": "add", "value": {"model": {"document": created}}},
    }
    after = folders.read_plan("nested", revision=None)
    prior = {run.key: run.request_id for run in before.plan.runs}
    result_ids = {run.key: run.request_id for run in after.plan.runs}
    treated = RunKey("treated", "baseline", "default")
    assert result_ids[treated] == expected
    assert {key: value for key, value in result_ids.items() if key != treated} == {
        key: value for key, value in prior.items() if key != treated
    }


def _swept_attachments(raw, step, wire, attachments):
    """Give ``step`` its own reference to the Model and one sweep value, ``edit=on``, putting ``attachments`` first on
    the controller's ``wire``."""
    authored = _model_controller(raw)["wires"][wire]
    raw["steps"][step]["base"]["inline"]["trial_documents"] = {"model": {"document": "reach-model"}}
    target = "documents.reach-model.payload.interior.components.wrapper.interior.components.controller.interior"
    raw["steps"][step]["sweep"] = {"axes": {"edit": {"values": {"on": {"patches": {
        f"{target}.wires.{wire}.attachments": {
            "op": "replace" if "attachments" in authored else "add",
            "value": [*attachments, *authored.get("attachments", [])],
        },
    }}}}}}


def _attachment_owner(wire):
    return {
        "composites": ["wrapper", "controller"],
        "element": {"kind": "wire", "id": wire},
    }


def _scale_attachment(identity, factor):
    return {"kind": "intervention", "id": identity, "operation": "Scale", "operands": {"factor": {"kind": "value", "value": factor}}}


def _add_attachment(owner, identity, after, factor=0.5):
    return {"op": "add", "owner": owner, "attachment": _scale_attachment(identity, factor), "after": after}


def _attachment_address(owner, identity):
    return owner | {"element": {"kind": "attachment", "parent": owner["element"], "id": identity}}


def _model_controller(raw):
    return raw["documents"]["reach-model"]["payload"]["interior"]["components"]["wrapper"]["interior"][
        "components"
    ]["controller"]["interior"]


def test_evaluate_attachment_order_follows_effective_graph():
    raw = _model_project()
    owner = _attachment_owner("carry")
    records = [
        _add_attachment(owner, "first", None),
        _add_attachment(owner, "second", None, 0.25),
        _add_attachment(owner, "following", "carry"),
    ]
    source = ModelDocument.model_validate(raw["documents"]["reach-model"]["payload"])
    resolved = resolve_effective_model(
        source, ComponentRegistry(), tuple(MODEL_ATTACHMENT_BATCHES.validate_python({"step": records})["step"]),
    )
    controller = resolved.interior.components["wrapper"].interior.components["controller"]
    assert [item.id for item in controller.interior.wires["carry"].attachments] == ["first", "second", "carry", "following"]


@pytest.mark.parametrize("path", ["model_attachments.step", "model_attachments"])
def test_a_sweep_that_writes_model_attachments_is_refused(path):
    raw = _model_project()
    owner = _attachment_owner("carry")
    raw["steps"]["baseline"]["sweep"] = {"axes": {"edit": {"values": {"on": {"patches": {
        path: {"op": "add", "value": [_add_attachment(owner, "condition", None)]},
    }}}}}}
    plan = compile_project(Project.model_validate(raw), memo=ProjectPlanMemo())
    problem, = plan.diagnostics
    assert problem.step == "baseline" and "model_attachments" in problem.detail
    assert [run.kind for run in plan.runs] == ["train"]


def test_group_keeps_delay_on_a_crossing_gru_output():
    raw = _model_project()
    registry = new_application_registry_bundle()
    payload = raw["documents"]["reach-model"]["payload"]
    controller = payload["interior"]["components"]["wrapper"]["interior"]["components"]["controller"]
    controller["interior"]["wires"]["decode"]["attachments"] = [
        {"kind": "delay", "id": "feedback", "delay": {"value": 1, "unit": "st"}},
    ]
    state = ModelEditState(document=ModelDocument.model_validate(payload))
    ports = ModelRegistryPorts(registry.components)
    command = ModelEditCommand.model_validate({"operation": {
        "op": "group", "name": "Cell",
        "selection": [{
            "composites": ["wrapper", "controller"],
            "element": {"kind": "component", "id": "cell"},
        }],
    }})
    result = apply_model_edit(
        state, command, ports=ports.port, component_problem=ports.placement_problem,
    )
    assert isinstance(result, ModelEditResult), result
    raw["documents"]["reach-model"]["payload"] = result.state.document.model_dump(mode="json")
    planned = compile_project(Project.model_validate(raw))
    assert not planned.diagnostics, planned.diagnostics
    train = next(run for run in planned.runs if run.kind == "train")
    spec = decode_request(planned.requests[train.request_id], planned.documents.__getitem__).spec
    graph = _prepare_request(spec, registry).executable.resolved.graph
    resolved = graph.subgraphs["wrapper"].subgraphs["controller"]
    (delay,) = [wire.recurrent_initializer for wire in resolved.wires if wire.is_recurrent]
    assert delay["shape"] == [controller["interior"]["components"]["cell"]["fields"]["hidden_size"]]
    assert delay["units"] == "1"


def test_stored_equations_use_the_scalar_solver_and_free_boundary():
    def leaf(type_, **fields):
        return {"kind": "leaf", "name": type_, "type": f"feedbax.{type_}", "fields": fields}

    model = {
        "step_seconds": 0.01,
        "ports": {
            "inputs": [_port("force", "force", shape=[])],
            "outputs": [_port("position", "position", shape=[])],
        },
        "interior": {
            "components": {"equations": {
                "kind": "composite", "name": "Mass and spring",
                "ports": {
                    "inputs": [_port("input-force", "Force", shape=[])],
                    "outputs": [_port("output-position", "Position", shape=[])],
                    "physical": [{"id": "free", "name": "Free end", "type": "translational"}],
                },
                "interior": {
                    "physical_domain": "translational",
                    "components": {
                        "wall": leaf("Ground"), "mass": leaf("Mass", mass=1.0),
                        "spring": leaf("LinearSpring", stiffness=10.0),
                        "act": leaf("ActuationInput"), "sense": leaf("SensorOutput"),
                    },
                    "connections": {
                        "wall-flange": {"a": _end("wall", "flange"), "b": _end("spring", "flange_a")},
                        "spring-flange": {"a": _end("spring", "flange_b"), "b": _end("mass", "flange")},
                        "act-flange": {"a": _end("act", "flange"), "b": _end("mass", "flange")},
                        "sense-flange": {"a": _end("sense", "flange"), "b": _end("mass", "flange")},
                        "free": {"a": {"owner": "boundary", "port": "free"}, "b": _end("mass", "flange")},
                    },
                    "wires": {
                        "drive": _wire({"owner": "inputs", "port": "input-force"}, _end("act", "u")),
                        "read": _wire(_end("sense", "y"), {"owner": "outputs", "port": "output-position"}),
                    },
                },
            }},
            "wires": {
                "force": _wire({"owner": "inputs", "port": "force"}, _end("equations", "input-force")),
                "position": _wire(_end("equations", "output-position"), {"owner": "outputs", "port": "position"}),
            },
        },
    }
    registry = ComponentRegistry()
    lowered = lower_model_document(ModelDocument.model_validate(model), registry)
    assert lowered.graph.subgraphs["equations"].solver.dt == model["step_seconds"]
    graph = compile_graph(
        GraphDocument(graph=lowered.graph), registry,
        input_prototypes={lowered.graph.input_bindings["force"]: jax.ShapeDtypeStruct((), jnp.float32)},
    ).graph
    outputs, _ = graph(
        {"force": jnp.ones(4)}, graph.init_state(key=jax.random.key(1)),
        key=jax.random.key(1), n_steps=4,
    )
    assert outputs["position"][-1] > 0
    project = _model_project()
    del project["steps"]["baseline"]
    reaching = project["documents"]["reach-model"]["payload"]["interior"]
    reaching["components"].update(model["interior"]["components"])
    reaching["components"]["constant-force"] = leaf("Constant", value=1.0)
    reaching["wires"]["equations-force"] = _wire(
        _end("constant-force", "output"), _end("equations", "input-force"),
    )
    assert not compile_project(Project.model_validate(project)).diagnostics
    interior = model["interior"]["components"]["equations"]["interior"]
    interior["connections"].pop("act-flange")
    with pytest.raises(ModelCompilationProblem) as exc:
        lower_model_document(ModelDocument.model_validate(model), registry)
    assert exc.value.address.composites == ("equations",)


def test_linked_delay_repeats_the_completed_reset_state():
    raw = _model_project()
    model = raw["documents"]["reach-model"]["payload"]
    controller = model["interior"]["components"]["wrapper"]["interior"]["components"]["controller"]
    delay = controller["interior"]["wires"]["carry"]["attachments"][0]
    delay["delay"]["value"] = 3
    delay["initial_source"] = {"kind": "wire", "source": _end("cell", "hidden")}
    plan = compile_project(Project.model_validate(raw))
    assert not plan.diagnostics
    train = next(run for run in plan.runs if run.kind == "train")
    request = decode_request(plan.requests[train.request_id], plan.documents.__getitem__).spec
    registry = new_application_registry_bundle()
    graph = _prepare_request(request, registry).executable.graph
    state = graph.init_state(key=jax.random.key(3))
    cell = graph.nodes["wrapper"].nodes["controller"].nodes["cell"]
    state = cell.replace_state_value(state, ("hidden",), jnp.full((cell.hidden_size,), 2.0, dtype=jnp.float32))
    carry = graph.initial_cycle_port_values(state)[("__nested__", "wrapper")][("__nested__", "controller")]
    (queue,) = carry.values()
    np.testing.assert_array_equal(queue, np.full((3, cell.hidden_size), 2.0))
    delay["initial_source"]["source"] = _end("not-in-this-graph", "hidden")
    refused = compile_project(Project.model_validate(raw))
    (problem, *_) = next(run for run in refused.runs if run.kind == "train").problems
    assert problem.model_address.element.id == "carry"


@pytest.mark.parametrize(("tau", "alpha"), [
    (1, 1),
    ({"kind": "literal", "value": 1, "unit": "st"}, 1),
    ({"kind": "literal", "value": 0.05, "unit": "s"}, 0.2),
])
def test_root_period_resolves_leaky_cell_tau_and_model_interface(tau, alpha):
    payload = {
        "step_seconds": 0.01,
        "ports": {
            "inputs": [{**_port("setpoint", "setpoint", shape=[2]), "default": {"value": [0, 0]}}],
            "outputs": [{"id": "estimate", "name": "estimate", "type": {"shape": [2], "unit": "m/s"}}],
        },
        "interior": {
            "components": {"cell": {
                "kind": "leaf", "name": "Cell", "type": "feedbax.VanillaRNN",
                "fields": {"hidden_size": 2, "tau": tau},
            }},
            "wires": {
                "drive": _wire({"owner": "inputs", "port": "setpoint"}, _end("cell", "input")),
                "read": _wire(_end("cell", "output"), {"owner": "outputs", "port": "estimate"}),
            },
        },
    }
    registry = ComponentRegistry()
    model = ModelDocument.model_validate(payload)
    lowered = lower_model_document(model, registry, StudioTaskBindingSpec())
    interface = lowered.interface(model, registry)
    assert interface.period_seconds == 0.01
    assert not interface.inputs[0].required
    output, = interface.outputs
    assert ModelAddress.from_text(output.id).element.end.port == "estimate"
    assert output.type == ValueType(base=ValueBase.REAL, shape=(2,), dimensions=LENGTH.combine(TIME, -1))
    executable = compile_graph(GraphDocument(graph=lowered.graph), registry)
    assert executable.graph.nodes["cell"].cell.alpha == pytest.approx(alpha)


def _planar_arm():
    spec = two_link_arm_6muscle_template_graph()
    nodes = {
        key: {"kind": "leaf", "name": key, "type": node.type, "fields": dict(node.params)}
        for key, node in spec.nodes.items() if key not in {"effector", "state"}
    }
    nodes["sensor"] = {
        "kind": "leaf", "name": "Position", "type": "feedbax.SensorOutput",
        "fields": {"quantity": "position"},
    }
    return {
        "kind": "composite", "name": "Arm", "substeps": 2,
        "ports": {"inputs": [_port("drive", "Drive", shape=[6])], "outputs": [_port("pos", "Position", shape=[2])]},
        "interior": {
            "physical_domain": "planar_multibody", "components": nodes,
            "connections": {
                f"frame-{i}": {"a": _end(*edge.a), "b": _end(*edge.b)}
                for i, edge in enumerate(spec.connections)
            },
            "wires": {
                "actuate": _wire({"owner": "inputs", "port": "drive"}, _end("excitation", "u")),
                "sense": _wire(_end("sensor", "y"), {"owner": "outputs", "port": "pos"}),
            },
        },
    }


@pytest.mark.parametrize("domain", ["planar_multibody", "spatial_multibody"])
def test_saved_managed_mechanics_refuses_at_model_address(host, tmp_path, domain):
    from tests.test_document_train_pair import authored_documents, document_project

    if domain == "planar_multibody":
        pytest.importorskip("mujoco.mjx")
    raw = document_project(authored_documents(), evaluation=True)
    model = raw["documents"]["reach-model"]["payload"]
    arm = _planar_arm()
    arm["interior"]["physical_domain"] = domain
    model["interior"]["components"].update(arm=arm, readout={
        "kind": "leaf", "name": "Arm Command", "type": "feedbax.Linear",
        "fields": {"output_size": 6, "initialization": "zeros"},
    })
    model["interior"]["wires"].update(
        command_arm=_wire({"owner": "inputs", "port": model["ports"]["inputs"][0]["id"]},
                          _end("readout", "input")),
        drive_arm=_wire(_end("readout", "output"), _end("arm", "drive")),
    )
    folders = ProjectFolderService(tmp_path / "projects")
    project = "mechanics"
    _save_project(folders, project, Project.model_validate(raw))
    jobs = JobService(host, new_application_registry_bundle())
    service = ProjectPlanService(folders, jobs)
    view = service.read(project, revision=None)
    response = service.response(view, include_window_map=True)

    train = next(problem for problem in response.window_map.problems if problem.step == "train")
    assert train.place == StudioModelPlace(part="model", document="reach-model", member=ModelAddress.model_validate({
        "composites": ["arm"] if domain == "spatial_multibody" else [],
        "element": {"kind": "graph"} if domain == "spatial_multibody" else {"kind": "component", "id": "arm"},
    }))
    if domain == "spatial_multibody":
        assert train.path == "/trial_documents/model/interior/components/arm"
        assert "requires an authored spatial asset" in train.detail
        evaluate = next(problem for problem in response.window_map.problems if problem.step == "baseline")
        assert (evaluate.path, evaluate.code) == ("inputs.subject", "blocked_input")
    else:
        assert train.path == "/trial_documents/model/interior/components/arm"
        assert "no typed model checkpoint owner for solver 'euler'" in train.detail
        evaluate = next(
            problem for problem in response.window_map.problems
            if problem.step == "baseline" and problem.code != "blocked_input"
        )
        assert evaluate.place.member == train.place.member
        assert "no qualified Evaluate realization" in evaluate.detail
        (blocked,) = [
            problem for problem in response.window_map.problems
            if problem.step == "baseline" and problem.code == "blocked_input"
        ]
        assert (blocked.row, blocked.blocked_by.step) == (evaluate.row, "train")

    snapshot = view.settled
    train_key = RunKey("default", "train", "default")
    evaluate_key = RunKey("default", "baseline", "default")
    status = {row.key: row.status for row in view.status_rows()}
    assert status[train_key] == "cannot_run"
    if domain == "planar_multibody":
        assert status[evaluate_key] == "cannot_run"
    refused = jobs.admit_plan(
        folders, snapshot, run_keys=(train_key, evaluate_key), token=token("unsupported-mechanics"),
    ).items
    assert len(refused) == 2 and all(item.disposition == "rejected" for item in refused)
    assert not host.jobs(project=snapshot.project_id)


def test_shipped_reaching_matches_the_original_plant_feedback():
    document = load_shipped_projects()["point-mass-reaching"].documents["point-mass"]
    model = ModelDocument.model_validate(document.payload)
    assert not model.interior.wires["sense-hand"].attachments
    registry = ComponentRegistry()
    lowered = lower_model_document(model, registry).graph
    baseline = lowered.model_copy(deep=True)
    sample = next(wire for wire in baseline.wires if (wire.recurrent_initializer or {}).get("model_state_sample"))
    feedback = next(wire for wire in baseline.wires if wire.source_node == sample.target_node)
    del baseline.nodes[sample.target_node]
    baseline.wires.remove(sample)
    feedback.source_node, feedback.source_port = sample.source_node, sample.source_port
    feedback.recurrent_initializer = {"kind": "state_output"}
    target = jnp.linspace(-0.1, 0.1, 128, dtype=jnp.float32).reshape((64, 2))
    inputs = {"target_pos": target}
    initialization_key, rollout_key = jax.random.split(jax.random.key(42))

    def roll_out(spec):
        graph = compile_graph(
            GraphDocument(graph=spec), registry, model_input_prototypes(spec, model, {}),
            initialization_key=initialization_key,
        ).graph
        return graph(inputs, graph.init_state(key=rollout_key), key=rollout_key, n_steps=64)[0]

    before, after = roll_out(baseline), roll_out(lowered)
    for expected, actual in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True):
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()


@pytest.mark.parametrize("kind", ["LinearStateSpace", "StructuralLinearStateSpace"])
def test_discrete_mechanics_feedback_reads_start_state_with_exact_delay(kind):
    initial = 3
    steps = 10
    fields = {
        "A": [[1.0]], "B": [[1.0]], "initial_state": [float(initial)],
        "pos_slice": [0, 1], "vel_slice": [0, 1],
    }
    if kind == "StructuralLinearStateSpace":
        fields["delta_A"] = [[0.0]]
    payload = {
        "step_seconds": 0.01,
        "ports": {"outputs": [_port("after", "after", shape=[1]), _port("read", "read", shape=[1])]},
        "interior": {
            "components": {
                "plant": {"kind": "leaf", "name": "Plant", "type": f"feedbax.{kind}", "fields": fields},
                "gain": {"kind": "leaf", "name": "Gain", "type": "feedbax.Gain", "fields": {"gain": 1.0}},
            },
            "wires": {
                "drive": _wire(_end("gain", "output"), _end("plant", "force")),
                "feedback": _wire(_end("plant", "state"), _end("gain", "input")),
                "observed-feedback": _wire(_end("gain", "output"), {"owner": "outputs", "port": "read"}),
                "observed-state": _wire(_end("plant", "state"), {"owner": "outputs", "port": "after"}),
            },
        },
    }
    registry = ComponentRegistry()
    key = jax.random.key(4)

    def run():
        document = ModelDocument.model_validate(payload)
        spec = lower_model_document(document, registry).graph
        graph = compile_graph(GraphDocument(graph=spec), registry).graph
        return graph({}, graph.init_state(key=key), key=key, n_steps=steps)[0]

    observed = run()
    expected_pre_step = np.array([[initial * 2**index] for index in range(steps)])
    np.testing.assert_array_equal(observed["read"], expected_pre_step)
    np.testing.assert_array_equal(observed["after"], 2 * expected_pre_step)

    payload["interior"]["wires"]["feedback"]["attachments"] = [{
        "kind": "delay", "id": "lag", "delay": {"value": 5, "unit": "st"},
        "initial_source": {"kind": "wire", "source": _end("plant", "state")},
    }]
    delayed = run()
    state = [initial]
    feedback = []
    for index in range(steps):
        sample = initial if index < 5 else state[index - 5]
        feedback.append([sample])
        state.append(state[-1] + sample)
    np.testing.assert_array_equal(delayed["read"], feedback)
    np.testing.assert_array_equal(delayed["after"], [[sample] for sample in state[1:]])


@pytest.mark.parametrize("kind", ["LinearStateSpace", "StructuralLinearStateSpace"])
def test_discrete_mechanics_effector_reads_start_velocity_in_curl_loop(kind):
    period = 0.125
    initial = [0.5, -0.25, 1.0, 2.0]
    fields = {
        "A": [[1, 0, period, 0], [0, 1, 0, period], [0, 0, 1, 0], [0, 0, 0, 1]],
        "B": [[0, 0], [0, 0], [period, 0], [0, period]],
        "initial_state": initial,
    }
    if kind == "StructuralLinearStateSpace":
        fields["delta_A"] = np.zeros((4, 4)).tolist()
    document = ModelDocument.model_validate({
        "step_seconds": period,
        "ports": {"outputs": [_port("state", "state", shape=[4])]},
        "interior": {
            "components": {
                "plant": {"kind": "leaf", "name": "Plant", "type": f"feedbax.{kind}", "fields": fields},
                "curl": {"kind": "leaf", "name": "Curl", "type": "feedbax.CurlField", "fields": {"active": True}},
            },
            "wires": {
                "feedback": _wire(_end("plant", "effector"), _end("curl", "effector")),
                "drive": _wire(_end("curl", "force"), _end("plant", "force")),
                "observation": _wire(_end("plant", "state"), {"owner": "outputs", "port": "state"}),
            },
        },
    })
    registry = ComponentRegistry()
    spec = lower_model_document(document, registry).graph
    graph = compile_graph(GraphDocument(graph=spec), registry).graph
    key = jax.random.key(4)
    outputs, _, traces = graph.rollout_with_trace(
        {}, graph.init_state(key=key), key=key, n_steps=5,
        trace=(GraphTraceRequest("port", "effector", node="plant", port="effector"),),
    )

    expected, forces = [], []
    position, velocity = np.asarray(initial).reshape(2, 2)
    for _ in range(5):
        force = np.array([-velocity[1], velocity[0]])
        position = position + period * velocity
        velocity = velocity + period * force
        forces.append(force)
        expected.append(np.concatenate((position, velocity)))
    expected = np.asarray(expected)
    np.testing.assert_array_equal(outputs["state"], expected)
    np.testing.assert_array_equal(traces["effector"].pos, expected[:, :2])
    np.testing.assert_array_equal(traces["effector"].vel, expected[:, 2:])
    np.testing.assert_array_equal(traces["effector"].force, forces)


def test_serial_mechanics_hold_samples_this_step_and_delay_adds_five_steps():
    pytest.importorskip("mujoco.mjx")
    plant = _planar_arm()
    hold = {"kind": "hold", "id": "hold"}
    linear = {
        "kind": "leaf", "name": "Readout", "type": "feedbax.Linear",
        "fields": {"output_size": 6, "initialization": "zeros"},
    }
    payload = {
        "step_seconds": 0.01,
        "world": {"gravity": [0.0, -9.81, 0.0]},
        "ports": {"outputs": [_port("pos", "position", shape=[2])]},
        "interior": {"components": {"arm": plant, "linear": linear}, "wires": {
            "sample": _wire(_end("arm", "pos"), _end("linear", "input"), (hold,)),
            "drive": _wire(_end("linear", "output"), _end("arm", "drive")),
            "observed": _wire(_end("arm", "pos"), {"owner": "outputs", "port": "pos"}),
        }},
    }
    registry = ComponentRegistry()
    lowered = lower_model_document(ModelDocument.model_validate(payload), registry)
    graph = compile_graph(GraphDocument(graph=lowered.graph), registry).graph
    key = jax.random.key(7)
    state = graph.init_state(key=key)
    initial = graph.nodes["arm"].initial_outputs(graph.nodes["arm"].state_view(state))["pos"]
    (hold_node,) = [name for name in lowered.graph.nodes if name.startswith("_model_hold_")]
    trace = (GraphTraceRequest("port", "sample", node=hold_node, port="output"),)
    outputs, _, traces = graph.rollout_with_trace(
        {}, state, n_steps=10, key=key, trace=trace,
    )
    np.testing.assert_array_equal(traces["sample"][0], initial)
    np.testing.assert_array_equal(traces["sample"][1:], outputs["position"][:-1])
    assert not np.array_equal(outputs["position"][0], initial)
    assert graph.nodes["arm"].backend.n_substeps == 2
    assert graph.nodes["arm"].backend.sub_dt == pytest.approx(0.005)

    lag = {
        "kind": "delay", "id": "lag", "delay": {"value": 0.05, "unit": "s"},
        "initial_source": {"kind": "wire", "source": _end("arm", "pos")},
    }
    payload["interior"]["wires"]["sample"]["attachments"] = [lag, hold]
    linked = lower_model_document(ModelDocument.model_validate(payload), registry)
    delayed = compile_graph(GraphDocument(graph=linked.graph), registry).graph
    carries = delayed.initial_cycle_port_values(delayed.init_state(key=key))
    np.testing.assert_array_equal(carries[hold_node, "input"], np.stack([initial] * 5))
    _, _, delayed_traces = delayed.rollout_with_trace(
        {}, delayed.init_state(key=key), n_steps=10, key=key, trace=trace,
    )
    np.testing.assert_array_equal(delayed_traces["sample"][:5], np.stack([initial] * 5))
    np.testing.assert_array_equal(delayed_traces["sample"][5:], traces["sample"][:-5])

    lag["delay"]["value"] = 0.001
    with pytest.raises(ModelCompilationProblem, match="round to at least one step") as error:
        lower_model_document(ModelDocument.model_validate(payload), registry)
    assert error.value.address.element.path == ("delay",)
    assert error.value.address.element.parent.id == "lag"

    payload["interior"]["wires"]["sample"]["attachments"] = []
    with pytest.raises(ModelCompilationProblem, match="within-step") as error:
        lower_model_document(ModelDocument.model_validate(payload), registry)
    assert error.value.address.element.id == "drive"


@pytest.mark.parametrize("grouped", [False, True])
def test_three_level_model_train_evaluate_and_rename_current(host, tmp_path, capsys, record_property, grouped):
    from tests.test_document_train_pair import authored_documents, document_project

    registry = new_application_registry_bundle()
    jobs = JobService(host, registry)
    folders = ProjectFolderService(tmp_path / "projects")
    raw = document_project(authored_documents(), evaluation=True)
    group = _group_reaching_pipeline(raw, registry.components) if grouped else None
    if grouped:
        model = raw["documents"]["reach-model"]["payload"]
        controller = model["interior"]["components"]["wrapper"]["interior"]["components"]["controller"]
        controller["interior"]["wires"]["carry"]["attachments"][0]["delay"]["value"] = 3
        controller["interior"]["wires"]["decode"]["attachments"] = [
            _scale_attachment("train-scale", 0.95),
            _scale_attachment("train-extra", 0.9),
        ]
    else:
        _use_initial_evaluation(raw)
    project = "nested"
    _save_project(folders, project, Project.model_validate(raw))
    plan = folders.read_plan(project, revision=None)
    if grouped:
        train_before = next(run.request_id for run in plan.plan.runs if run.kind == "train")
        owner = _attachment_owner("decode")
        for difference in (
            _add_attachment(owner, "condition", "train-scale"),
            {
                "op": "change", "address": _attachment_address(owner, "train-scale"),
                "attachment": _scale_attachment("train-scale", 0.75),
            },
            {"op": "remove", "address": _attachment_address(owner, "train-extra")},
        ):
            snapshot = folders.read(project)
            applied = folders.edit_model(project, ModelEditRequest.model_validate({
                "revision": snapshot.revision, "presentation_revision": snapshot.presentation_revision,
                "document": "reach-model", "step": "baseline", "rows": ["default"],
                "command": {"operation": {"op": "set_attachment", "difference": difference}},
            }), registry.components)
            assert applied.result.status == "applied", applied.result
        raw = folders.read(project).project.model_dump(mode="json", by_alias=True, exclude_unset=True)
        plan = folders.read_plan(project, revision=None)
        assert next(run.request_id for run in plan.plan.runs if run.kind == "train") == train_before
    assert main(["plan", str(folders.root / project)]) == 0
    assert not plan.plan.diagnostics
    assert set(line.split()[1] for line in capsys.readouterr().out.splitlines()) == {
        run.request_id for run in plan.plan.runs
    }
    train = RunKey("default", "train", "default")
    evaluate = RunKey("default", "baseline", "default")
    (train_job,) = jobs.admit_plan(folders, plan, run_keys=(train,), token=token("nested-train")).items
    (evaluation_job,) = jobs.admit_plan(folders, plan, run_keys=(evaluate,), token=token("nested-evaluate")).items
    assert [train_job.disposition, evaluation_job.disposition] == ["admitted", "admitted"], (
        train_job, evaluation_job,
    )

    def finished(job):
        return wait_until(lambda: (
            job_row(host, job.job_id, "succeeded") or job_row(host, job.job_id, "failed")
        ))

    trained = finished(train_job)
    assert trained["state"] == "succeeded", trained.get("outcome")
    checkpoint = PublishedTrainingCheckpoint.model_validate(trained["outcome"]["completion"]["checkpoint"])
    slot = read_method_checkpoint(checkpoint.checkpoint_root, checkpoint.pointer, components=new_application_registry_bundle().components).model
    assert slot is not None
    restored = slot.restore()
    assert restored.input_shapes == {"reach_target": (2,)}
    assert slot.input_arrays["reach_target"].dtype == "float32"
    assert restored.nodes["wrapper"].nodes["controller"].nodes["cell"].cell.weight_ih.shape[1] == 2
    evaluated = finished(evaluation_job)
    assert evaluated["state"] == "succeeded", evaluated.get("outcome")
    result = jobs.read_evaluation_result(project=plan.project_id, store=plan.store_path, job_id=evaluation_job.job_id)
    service = ProjectPlanService(folders, jobs)
    accepted = service.read(project, revision=plan.revision)
    train_receipt = accepted.attempts_by_run[train][0].successful_receipt
    assert train_receipt is not None
    store = ImmutableArtifactBlobProvider(plan.store_path)

    def authenticated_receipts():
        view = SuccessView(store)
        receipts = tuple(view.read_attempt(job.job_id) for job in (train_job, evaluation_job))
        assert all(receipt is not None for receipt in receipts)
        return receipts

    original_receipts = authenticated_receipts()
    n_samples = raw["documents"]["task"]["payload"]["fixed_steps"] + 1
    with result.open_states() as dataset:
        assert dataset.data("/v1/model/state/wrapper/controller/cell/hidden").shape[-2] == n_samples
        carry = dataset.data("/v1/model/carry/wrapper/controller/carry/carry")
        assert carry.shape[-(3 if grouped else 2)] == n_samples
        if grouped:
            assert carry.shape[-2] == 3
            np.testing.assert_array_equal(carry[..., 0, :, :], 0)
        assert np.isfinite(dataset.data("objective/total")).all()
        if group is not None:
            assert dataset.data(f"/v1/model/state/{group}/plant/effector~1position").shape[-2] == n_samples

    _rename_components(raw["documents"]["reach-model"]["payload"])
    _save_project(folders, project, Project.model_validate(raw))
    renamed = folders.read_plan(project, revision=None)
    assert [run.request_id for run in renamed.plan.runs] == [run.request_id for run in plan.plan.runs]
    joined = service.read(project, revision=renamed.revision)
    assert joined.attempts_by_run[train][0].job_id == train_job.job_id
    assert joined.attempts_by_run[train][0].successful_receipt == train_receipt
    assert joined.attempts_by_run[evaluate][0].successful_receipt == result.receipt

    model_name = "reach-model"
    if not grouped:
        trial = raw["steps"]["train"]["base"]["inline"]["trial_documents"]
        inputs = raw["steps"]["baseline"]["inputs"]
        for document, role, input_role in (
            ("reach-model", "model", "subject"),
            ("task", "task", "task"),
            ("objective", "objective", "objective"),
        ):
            name = f"{document}-renamed"
            doc_path = folders.root / project / "documents" / f"{document}.json"
            doc_path.rename(doc_path.with_stem(name))
            raw["documents"][name] = raw["documents"].pop(document)
            trial[role] = {"document": name}
            inputs[input_role]["source"] = {"document": name}
            _save_project(folders, project, Project.model_validate(raw))
            settled = service.read(project, revision=None)
            assert {row.key: row.status for row in settled.status_rows()} == {
                train: JobStatus.SUCCEEDED, evaluate: JobStatus.SUCCEEDED,
            }
            assert settled.settled.plan.requests == plan.plan.requests
            for key, job, receipt in (
                (train, train_job, train_receipt), (evaluate, evaluation_job, result.receipt),
            ):
                attempt, = settled.attempts_by_run[key]
                assert (attempt.job_id, attempt.successful_receipt) == (job.job_id, receipt)
        model_name = "reach-model-renamed"

    reopened = ProjectFolderService(folders.root)
    reopened_view = ProjectPlanService(reopened, jobs).read(project, revision=None)
    reopened_model = reopened.read(project).project.documents[model_name].payload
    assert reopened_model == raw["documents"][model_name]["payload"]
    assert reopened_view.settled.plan.requests == plan.plan.requests
    assert reopened_view.attempts_by_run[train][0].job_id == train_job.job_id
    assert reopened_view.attempts_by_run[train][0].successful_receipt == train_receipt
    assert reopened_view.attempts_by_run[evaluate][0].successful_receipt == result.receipt
    assert authenticated_receipts() == original_receipts

    model = raw["documents"][model_name]["payload"]
    model["ports"]["inputs"][0]["name"] = "renamed_reach_target"
    _save_project(folders, project, Project.model_validate(raw))
    changed = reopened.read_plan(project, revision=None)
    assert not changed.plan.diagnostics
    before_ids = {run.key: run.request_id for run in plan.plan.runs}
    after_ids = {run.key: run.request_id for run in changed.plan.runs}
    assert before_ids.keys() == after_ids.keys() == {train, evaluate}
    assert all(before_ids[key] != after_ids[key] for key in before_ids)
    record_property("request_ids_before_root_rename", ",".join(before_ids.values()))
    record_property("request_ids_after_root_rename", ",".join(after_ids.values()))
    record_property("authenticated_receipts", ",".join(receipt.ref.sha256 for receipt in original_receipts))


def test_same_seed_nested_rename_preserves_values_and_batch_loss():
    from tests.test_document_train_pair import authored_documents, document_project

    raw = document_project(authored_documents())
    registry = new_application_registry_bundle()

    def trained():
        plan = compile_project(Project.model_validate(raw))
        row = next(run for run in plan.runs if run.kind == "train")
        request = decode_request(plan.requests[row.request_id], plan.documents.__getitem__).spec
        paired = _prepare_request(request, registry)
        _, method, settings = normalize_train_method(request, registry.training_methods)
        seed = request.seed or 0
        ids = ("0",)
        state = initialize_method(method, settings, models=(paired.executable.graph,), seed=seed, replica_ids=ids)
        result = TrainingLoop(
            method, settings, execution=paired.execution, objective=paired.objective,
            batch_size=request.training_config.batch_size, seed=seed, replica_ids=ids,
        ).run_block(state, completed=0, count=2)
        assert result.completed == 2
        losses = result.metrics["train_loss"]
        model = replica_value(result.state.model, 0).nodes["wrapper"].nodes["controller"]
        return losses, jax.tree.leaves(eqx.filter(model, eqx.is_array))

    before, values = trained()
    _rename_components(raw["documents"]["reach-model"]["payload"])
    after, renamed = trained()
    np.testing.assert_array_equal(before, after)
    assert len(values) == len(renamed) and values
    for a, b in zip(values, renamed, strict=True):
        np.testing.assert_array_equal(a, b)


def _refuse_realization(monkeypatch):
    """Makes realizing a graph, or constructing any registered component but an acausal element descriptor, raise."""
    from feedbax.acausal.base import AcausalElement
    from feedbax.compiler import graph
    from feedbax.execution import trial_resolution as trial_execution
    from feedbax.component_registry.declarations import ComponentParamContract
    from feedbax.host import model_input

    def refuse(*args, **kwargs):
        raise AssertionError("a plan read realized the model")

    for module in (graph, trial_execution, model_input):
        monkeypatch.setattr(module, "realize_graph", refuse)
    for name in ("build", "build_resolved"):
        construct = getattr(ComponentParamContract, name)

        def guarded(contract, *args, construct=construct, **kwargs):
            if not issubclass(contract.runtime_type, AcausalElement):
                raise AssertionError(f"a plan read constructed {contract.runtime_type.__name__}")
            return construct(contract, *args, **kwargs)

        monkeypatch.setattr(ComponentParamContract, name, guarded)


@pytest.mark.parametrize("scope", ["plain", "within-step"])
def test_reading_the_plan_realizes_no_model(monkeypatch, scope):
    from web.e2e.studio.fixtures.stored_model import add_physical_interior

    raw = _model_project()
    if scope != "plain":
        add_physical_interior(raw, scope)
    project = Project.model_validate(raw)
    expected = compile_project(project)
    _refuse_realization(monkeypatch)
    plan = compile_project(project)
    assert plan == expected
    refusals = [
        problem.detail for row in plan.rows for problem in row.problems if "within-step evaluation" in problem.detail
    ]
    assert bool(refusals) == (scope == "within-step")


def test_reading_the_plan_of_a_grounded_network_realizes_no_model(monkeypatch):
    def leaf(name):
        return {"kind": "leaf", "type": f"feedbax.{name}", "name": name}

    raw = _model_project()
    del raw["steps"]["baseline"]
    raw["documents"]["reach-model"]["payload"]["interior"]["components"]["equations"] = {
        "kind": "composite", "name": "Mass spring wall", "ports": {},
        "interior": {
            "physical_domain": "translational",
            "components": {"mass": leaf("Mass"), "spring": leaf("LinearSpring"), "wall": leaf("Ground")},
            "connections": {
                "end1": {"a": _end("wall", "flange"), "b": _end("spring", "flange_a")},
                "end2": {"a": _end("spring", "flange_b"), "b": _end("mass", "flange")},
            },
        },
    }
    _refuse_realization(monkeypatch)
    plan = compile_project(Project.model_validate(raw))
    assert len(plan.requests) == 1
    assert plan.diagnostics == ()


def test_reading_the_plan_refuses_an_acausal_leaf_in_a_causal_model(monkeypatch):
    raw = _model_project()
    raw["documents"]["reach-model"]["payload"]["interior"]["components"]["mass"] = {
        "kind": "leaf", "name": "Mass", "type": "feedbax.Mass", "fields": {"mass": 1},
    }
    _refuse_realization(monkeypatch)
    plan = compile_project(Project.model_validate(raw))
    (train,) = [run for run in plan.runs if run.kind == "train"]
    assert any(
        "'feedbax.Mass' for node 'mass' belongs to domain 'feedbax.domain.acausal'" in problem.detail
        for problem in train.problems
    )


def test_shipped_effector_delay_keeps_the_full_recorded_value(tmp_path):
    from feedbax.runtime.raw_state import bind_raw_state
    from feedbax.tasks.document import TaskParameters
    from feedbax.validity.computation import capture_arguments, pytree_operands

    example = load_shipped_projects()["point-mass-reaching"]
    name = "simple-reaches"
    selected = example.model_copy(update={"steps": {name: example.steps[name]}})
    app = create_app(projects_dir=tmp_path)
    folders = app.state.project_folders
    _save_project(folders, "reach", selected)
    snapshot = folders.read("reach")
    with TestClient(app) as client:
        response = client.post("/api/projects/reach/model/edit", json={
            "revision": snapshot.revision, "presentation_revision": snapshot.presentation_revision,
            "document": "point-mass",
            "step": name,
            "command": {"operation": {"op": "set_attachment", "difference": {
                "op": "add",
                "after": None,
                "owner": {"composites": [], "element": {"kind": "wire", "id": "effector"}},
                "attachment": {"kind": "delay", "id": "observe-lag", "delay": {"value": 2, "unit": "st"}},
            }}},
        })
    assert response.status_code == 200, response.text
    assert response.json()["result"]["status"] == "applied", response.text

    plan = folders.read_plan("reach", revision=None).plan
    assert not plan.diagnostics, plan.diagnostics
    run, = plan.runs
    assert run.admission_problem is None, run.admission_problem
    spec = decode_request(plan.requests[run.request_id], plan.documents.__getitem__).spec
    prepared = _prepare_request(spec, new_application_registry_bundle())
    executable = prepared.executable
    graph = executable.graph
    rebuilt = _prepare_request(spec, new_application_registry_bundle()).executable.graph

    def identity(model):
        arrays, static = pytree_operands(model)
        return capture_arguments(arrays, static=static)

    assert identity(graph) == identity(rebuilt)
    task = prepared.execution.task
    layout = bind_raw_state(graph, executable.resolved.graph)
    delay_wire, = [
        wire for wire in executable.resolved.graph.wires
        if (wire.recurrent_initializer or {}).get("model_delay_id") == "observe-lag"
    ]
    key = jax.random.key(4)
    trial = task.ahead(TaskParameters(), key, key)
    inputs = {port: trial.inputs[signal] for port, signal in prepared.execution.input_bindings}
    reset = prepared.execution.reset.prepare(graph, trial.inits, key=key, inputs=inputs)
    outputs, _, retained, history = graph.rollout_with_trace_and_history(
        inputs, reset.state, key=key, cycle_init=reset.cycle_init, n_steps=8,
        trace=(GraphTraceRequest("port", "before", node=delay_wire.source_node, port=delay_wire.source_port),),
        capture_state=layout.capture,
    )
    leaves = jax.tree.leaves(retained["before"])
    assert any(np.any(np.asarray(leaf) != 0) for leaf in leaves)
    for actual, prior in zip(jax.tree.leaves(outputs["effector"]), leaves, strict=True):
        np.testing.assert_array_equal(actual[:2], 0)
        np.testing.assert_array_equal(actual[2:], prior[:-2])
    for field in ("pos", "vel", "force"):
        assert history[f"/v1/model/carry/effector/observe-lag/{field}"].shape == (9, 2, 2)
