"""Plan and shared graph compilation for a model document with nested owners."""

from copy import deepcopy

import jax
import numpy as np
import equinox as eqx
import jax.numpy as jnp
import pytest
from fastapi.testclient import TestClient

from feedbax.cli import main
from feedbax.compiler import GraphDocument, compile_graph
from feedbax.compiler.model_document import ModelCompilationProblem, lower_model_document
from feedbax.compiler.model_attachment import resolve_effective_model
from feedbax.compiler.mechanics_templates import two_link_arm_6muscle_template_graph
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_attachment import MODEL_ATTACHMENT_BATCHES
from feedbax.contracts.expression_values import LENGTH, TIME, ValueBase, ValueType
from feedbax.contracts.graph import StudioTaskBindingSpec
from feedbax.component_registry import ComponentPack, ComponentRegistry
from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID, ModelDocument
from feedbax.contracts.model_edit import ModelEditCommand, ModelEditResult, ModelEditState
from feedbax.contracts.project import Project
from feedbax.contracts.project_plan import ProjectPlanMemo, RunKey, compile_project
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.rollout import RolloutParameters
from feedbax.jobs import JobService
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.studio.canonical_train import lower_canonical_train
from feedbax.studio.model_edit import apply_model_edit
from feedbax.studio.model_ports import ModelRegistryPorts
from feedbax.runtime.graph import GraphTraceRequest
from feedbax.runtime.components import Input
from feedbax.runtime._signal_schedule import SignalScheduleError, resolve_signal_schedule
from feedbax.training.checkpoint_custody import open_published_model_slot
from feedbax.training.evaluation_variant import resolve_evaluation_variant
from feedbax.training.managed_studio import prepare_managed_training
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.contracts.studio_training import StudioGraphMethodPayload
from feedbax.contracts.training import TrainingSpec
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.evaluate_attachment import prepare_evaluate_attachment
from feedbax.contracts.model_edit import EvaluateAttachmentRequest
from feedbax.web.services.project_plan import ProjectPlanService
from feedbax.web.app import create_app
from feedbax.compiler.prototypes import prototypes_from_task_bindings
from feedbax.training.retained_graph import compile_training_run
from tests.test_canonical_evaluate_admission import _save_project, authored_evaluation_project
from tests.test_local_job_host import host as host_fixture, job_row, wait_until

host = host_fixture


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
        {"kind": "delay", "id": "delay", "delay": delay, "initial": initial_value},
        {"kind": "intervention", "id": "scale", "operation": "Scale", "fields": {"factor": 3.0}},
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
            {"document": "model", "composites": [], "element": {"kind": "component", "id": name}}
            for name in ("a", "c")
        ],
    }})
    grouped = apply_model_edit(
        "model", ModelEditState(document=model), command, ports=ports.port, accepts=ports.accepts,
    )
    assert isinstance(grouped, ModelEditResult), grouped
    group = grouped.changed[-1].element.id
    flat = lower_model_document("model", model, registry).graph
    lowered = lower_model_document("model", grouped.state.document, registry).graph
    assert [scope for scope in resolve_signal_schedule(lowered).order if scope[-1] in {"a", "b", "c"}] == [
        (group, "a"), ("b",), (group, "c"),
    ]
    initial = {flat.input_bindings["signal"]: jax.ShapeDtypeStruct((), jnp.float32)}
    first = compile_graph(GraphDocument(graph=flat), registry, input_prototypes=initial)
    second = compile_graph(GraphDocument(graph=lowered), registry, input_prototypes=initial)
    assert second.record.key_schedule.scoped_node_order == resolve_signal_schedule(lowered).order

    def run(graph, signal):
        outputs, _, trace = graph.rollout_with_trace(
            {"signal": signal}, graph.init_state(key=jax.random.key(5)), n_steps=signal.size,
            key=jax.random.key(5),
            trace=(GraphTraceRequest("graph_output", "result", port="result"),),
        )
        return outputs["result"], trace["result"]

    signal = jnp.arange(1, 5, dtype=jnp.float32)
    for graph in (first.graph, second.graph):
        output, trace = run(graph, signal)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(trace, output)
        derivative = jax.grad(lambda data: run(graph, data)[0].sum())(signal)
        np.testing.assert_array_equal(derivative, gradient)

    command = ModelEditCommand.model_validate({"operation": {
        "op": "ungroup", "address": grouped.changed[-1].model_dump(mode="json"),
    }})
    ungrouped = apply_model_edit(
        "model", grouped.state, command, ports=ports.port, accepts=ports.accepts,
    )
    assert isinstance(ungrouped, ModelEditResult), ungrouped
    assert ungrouped.state.document == model

    cycle = lower_model_document("model", grouped.state.document, registry).graph
    first_cut = next(wire for wire in cycle.subgraphs[group].wires if wire.temporality == "recurrent")
    first_cut.temporality = "instant"
    upstream = next(wire for wire in cycle.wires if wire.source_node == flat.input_bindings["signal"][0])
    last = cycle.output_bindings["result"]
    upstream.source_node, upstream.source_port = last
    with pytest.raises(SignalScheduleError, match="same-step cycle") as exc:
        resolve_signal_schedule(cycle)
    assert exc.value.authored_path.startswith("/graph/")


def _model_project() -> dict:
    project, _ = authored_evaluation_project()
    raw = project.model_dump(mode="json", by_alias=True, exclude_unset=True)
    train = raw["steps"]["train"]["base"]["inline"]
    source = train["graph"]
    leaves = {
        key: {"kind": "leaf", "name": key.capitalize(), "type": value["type"], "fields": value["params"]}
        for key, value in source["nodes"].items()
    }
    period = leaves["plant"]["fields"].pop("dt")
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
    model = {
        "step_seconds": period,
        "ports": {"inputs": [_port("task-input", "reach_target", shape=[2])],
                  "outputs": [_port("effector", "effector")]},
        "interior": {"components": {"wrapper": wrapper, "plant": leaves["plant"]}, "wires": {
            "task-wire": _wire({"owner": "inputs", "port": "task-input"}, _end("wrapper", "target")),
            "plant-wire": _wire(_end("wrapper", "force"), _end("plant", "force")),
            "effector-wire": _wire(_end("plant", "effector"), {"owner": "outputs", "port": "effector"}),
        }},
    }
    train["graph"] = {"document": "reach-model"}
    raw["documents"] = {"reach-model": {"schema": MODEL_DOCUMENT_SCHEMA_ID, "payload": model}}
    raw["steps"] = {key: raw["steps"][key] for key in ("train", "baseline")}
    return raw


def _rename_components(model):
    for component in model["interior"]["components"].values():
        component["name"] += " display name"
        if component["kind"] == "composite":
            for ports in component["ports"].values():
                for port in ports:
                    port["name"] += " display name"
            _rename_components(component)


def test_addressed_field_range_sample_and_row_provenance(tmp_path):
    raw = _model_project()
    raw["steps"].pop("baseline")
    model = raw["documents"]["reach-model"]["payload"]
    raw["steps"]["train"]["base"]["inline"]["alias"] = {"document": "reach-model", "pointer": "/step_seconds"}
    raw["steps"]["train"]["document_overrides"] = {"reach-model": {"/step_seconds": 0.02}}
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, "fields", Project.model_validate(raw))
    url = "/api/projects/fields"
    owner = {"document": "reach-model", "composites": [], "element": {"kind": "graph"}}

    with TestClient(create_app(projects_dir=tmp_path)) as client:
        initial = client.get(url).json()
        body = {"revision": initial["revision"], "owner": owner,
                "run": {"variant": "default", "step": "train", "row": "default"}}
        state = client.post(url + "/model/fields", json=body).json()["fields"][0]
        assert (state["text"], state["effective"]) == (f"{model['step_seconds']} s", "0.02 s")
        assert state["site"]["kinds"] == ["value", "sweep", "formula"]
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
            assert definition.payload["graph"]["step_seconds"] == pytest.approx(period)
            assert definition.payload["alias"] == pytest.approx(period)
            assert definition.document_ids["payload.graph"] == "reach-model"
            source = definition.sources["/payload/graph/step_seconds"]
            assert source.kind == "sweep_value"
            assert definition.sources["/payload/alias"] == source
        assert snapshot.project.documents["reach-model"].payload == model

        component_owner = {"document": "reach-model", "composites": [],
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
        mass_path = "/payload/graph/interior/components/plant/fields/mass"
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
        sampled_axis = folders.read("fields").project.steps["train"].sweep.axes["step"]
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
        "fields": {"tau": 2, "input_size": 4, "hidden_size": 4},
    }
    _save_project(ProjectFolderService(tmp_path), "duration", Project.model_validate(raw))
    url = "/api/projects/duration"
    owner = {"document": "reach-model", "composites": [], "element": {"kind": "component", "id": "rnn"}}

    with TestClient(create_app(projects_dir=tmp_path)) as client:
        def prepare(label, text, row="default", *, files=False):
            snapshot = client.get(url).json()
            body = {"revision": snapshot["revision"], "owner": owner, "lit_steps": ["train"],
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
        assert value == {"kind": "literal", "value": 0.03, "unit": "s"}
        field, value = prepare("Hidden size", "8")
        assert field["text"] == "4"
        assert value == 8
        field, snapshot = prepare("Tau", "linspace(20 ms, 40 ms, 3)", files=True)
        assert field["text"] == "0.03 s"
        axis = snapshot["project"]["steps"]["train"]["sweep"]["axes"]["tau"]
        assert ModelAddress.from_text(axis["path"]).element.path == ("fields", "tau")
        assert axis["range"] == {"start": 0.02, "stop": 0.04, "count": 3}
        folders = ProjectFolderService(tmp_path)
        revision = folders.read("duration").revision
        keys = tuple(row.key for row in folders.read_plan("duration", revision=revision).plan.rows if row.key.step == "train")
        definitions = folders.read_plan("duration", revision=revision, source_runs=keys).plan.selected_definitions
        assert [item.payload["graph"]["interior"]["components"]["rnn"]["fields"]["tau"]
                for item in definitions.values()] == [
            {"kind": "literal", "value": value, "unit": "s"} for value in (0.02, 0.03, 0.04)
        ]
        _, snapshot = prepare("Hidden size", "sample(choice(8, 12), 4)", keys[0].row, files=True)
        sampled = snapshot["project"]["steps"]["train"]["sweep"]["axes"]["hidden-size"]
        assert len(sampled["draws"]) == 4
        assert all(type(value) is int and value in (8, 12) for value in sampled["draws"].values())
        assert sampled["sampler"]["distribution"]["family"] == "choice"


def test_managed_rate_field_keeps_one_authored_geomspace_rule(tmp_path):
    raw = _model_project()
    train = raw["steps"]["train"]["base"]["inline"]
    train["training_config"].pop("learning_rate")
    train["trial_documents"] = {
        "model": train.pop("graph"), "task": train.pop("task"), "objective": train.pop("objective"),
    }
    raw["steps"].pop("baseline")
    _save_project(ProjectFolderService(tmp_path), "managed-rate", Project.model_validate(raw))
    url = "/api/projects/managed-rate"
    field_path = "/method_payload/payload/optimizer/params/learning_rate"
    endpoint = url + "/authoring/train/train/fields"
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        initial = client.get(url).json()
        body = {"revision": initial["revision"]}
        fields = client.post(endpoint, json=body).json()["fields"]
        rate, = [field for field in fields if field["label"] == "Learning Rate"]
        assert rate["id"] == field_path and rate["can_sweep"]
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
            assert resolved.payload["method_payload"]["payload"]["optimizer"]["params"]["learning_rate"] == pytest.approx(expected)
            assert resolved.sources["/payload" + field_path].kind == "sweep_value"


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
            {"document": "reach-model", "composites": [], "element": {"kind": "component", "id": name}}
            for name in ("relay", "plant")
        ],
    }})
    result = apply_model_edit(
        "reach-model", state, command, ports=ports.port, accepts=ports.accepts,
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
    registry = ComponentRegistry()
    graph = compile_graph(
        GraphDocument(graph=spec.graph), registry,
        input_prototypes=prototypes_from_task_bindings(spec.task_binding_spec),
        initialization_key=jax.random.key(42),
    ).graph
    assert graph.period_seconds == spec.graph.step_seconds == graph.nodes["plant"].dt
    assert graph.nodes["wrapper"].nodes["controller"].period_seconds == graph.period_seconds
    assert graph.nodes["wrapper"].nodes["controller"].nodes["cell"].cell.weight_ih.shape[1] == 2
    method = StudioGraphMethodPayload.model_validate(spec.method_payload.payload)
    prepared = compile_training_run(
        graph_spec=spec.graph,
        training_spec=TrainingSpec(
            optimizer=method.optimizer, loss=spec.objective.loss,
            n_batches=spec.training_config.n_batches, batch_size=1,
        ),
        task_spec=spec.task.model_dump(mode="json", exclude_none=True),
        task_binding_spec=spec.task_binding_spec,
        cfg=spec.training_config, component_registry=registry,
    )
    assert prepared.graph.nodes["wrapper"].nodes["controller"].nodes["cell"].cell.weight_ih.shape[1] == 2

    raw["documents"]["reach-model"]["payload"]["ports"]["inputs"][0]["name"] = "unknown"
    refused = compile_project(Project.model_validate(raw))
    problem = next(item for item in refused.diagnostics if item.model_address is not None)
    assert problem.document == "reach-model"
    assert problem.model_address.element.end.port == "task-input"


def test_stored_model_plan_uses_the_supplied_component_registry():
    class CustomRelay(Input):
        pass

    raw = _model_project()
    model = raw["documents"]["reach-model"]["payload"]
    model["interior"]["components"]["custom"] = {
        "kind": "leaf", "name": "Custom relay", "type": "test.CustomRelay",
    }
    model["interior"]["wires"]["task-wire"]["target"] = _end("custom", "input")
    model["interior"]["wires"]["custom-wire"] = _wire(_end("custom", "output"), _end("wrapper", "target"))
    project = Project.model_validate(raw)
    memo = ProjectPlanMemo()
    assert compile_project(project, memo=memo).diagnostics
    registry = ComponentRegistry()
    registry.ingest(ComponentPack(namespace="test", components=(CustomRelay,)))
    registry.seal()
    assert not compile_project(project, memo=memo, components=registry).diagnostics
    assert compile_project(project, memo=memo).diagnostics


def test_generated_delay_and_input_relay_keep_the_authored_source_addresses():
    raw = _model_project()
    model = ModelDocument.model_validate(raw["documents"]["reach-model"]["payload"])
    lowered = lower_model_document("reach-model", model, ComponentRegistry())
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


def _nested_scale_variation():
    prefix = "graph.subgraphs.wrapper.subgraphs.controller"
    return {
        "scale": {"op": "add", "path": f"{prefix}.nodes.scale", "value": {
            "type": "feedbax.Scale", "params": {"scale": 0.5, "fixed": True},
            "input_ports": ["input"], "output_ports": ["output"],
        }},
        "incoming": {"op": "replace", "path": f"{prefix}.wires.decode.target_node", "value": "scale"},
        "outgoing": {"op": "add", "path": f"{prefix}.wires.scaled", "value": {
            "source_node": "scale", "source_port": "output",
            "target_node": "readout", "target_port": "input",
        }},
    }


def test_nested_evaluate_variation_uses_scoped_wire_ids():
    from tests.test_document_train_pair import authored_documents, document_project

    raw = document_project(authored_documents(), evaluation=True)
    settings = raw["steps"]["baseline"]["base"]["inline"]["params"]
    settings["variation"] = _nested_scale_variation()
    plan = compile_project(Project.model_validate(raw))
    assert not plan.diagnostics
    train = next(run for run in plan.runs if run.kind == "train")
    request = decode_request(plan.requests[train.request_id], plan.documents.__getitem__).spec
    documents = request.trial_documents
    registry = ComponentRegistry()
    graph = lower_model_document(documents.model_name, documents.model, registry).graph
    evaluate = next(run for run in plan.runs if run.kind == "evaluate")
    params = RolloutParameters.model_validate(
        decode_request(plan.requests[evaluate.request_id], plan.documents.__getitem__).spec.params,
    )
    variant = resolve_evaluation_variant(graph, documents.task_source, params.variation)
    child = variant.graph.subgraphs["wrapper"].subgraphs["controller"]
    assert next(wire for wire in child.wires if wire.id == "decode").target_node == "scale"
    assert next(wire for wire in child.wires if wire.id == "scaled").target_node == "readout"
    index = next(i for i, wire in enumerate(child.wires) if wire.id == "scaled")
    assert variant.edit_location(
        f"/graph/subgraphs/wrapper/subgraphs/controller/wires/{index}/target_port",
    ) == "/params/variation/outgoing"
    executable = compile_graph(
        GraphDocument(graph=variant.graph), registry,
        input_prototypes={graph.input_bindings["reach_target"]: jax.ShapeDtypeStruct((2,), jnp.float32)},
    )
    assert executable.graph.nodes["wrapper"].nodes["controller"].nodes["scale"].scale == 0.5
    settings["variation"]["outgoing"]["path"] = settings["variation"]["outgoing"]["path"].replace(
        ".subgraphs.controller.", ".subgraphs.missing.",
    )
    invalid = compile_project(Project.model_validate(raw))
    assert not any(run.kind == "evaluate" for run in invalid.runs)
    problem, = invalid.diagnostics
    assert problem.path == "/params/variation/outgoing"
    assert "unknown causal composite" in problem.detail


@pytest.mark.parametrize(("kind", "scope"), [
    ("wire", {"kind": "step"}),
    ("port", {"kind": "rows", "rows": ["seed=control"]}),
    ("port", {"kind": "axis_value", "axis": "seed", "value": "control"}),
    ("wire", {"kind": "variant", "variant": "treated"}),
])
def test_addressed_evaluate_attachment_is_a_step_edit(tmp_path, kind, scope):
    raw = _model_project()
    variant, selected = "default", "default"
    if scope["kind"] == "variant":
        raw["sweep"] = {"rows": {"control": {}, "treated": {}}}
        variant = "treated"
    if scope["kind"] in {"rows", "axis_value"}:
        raw["steps"]["baseline"]["sweep"] = {"axes": {"seed": {"values": {
            "control": {"patches": {"params.seed": {"op": "replace", "value": 2}}},
            "comparison": {"patches": {"params.seed": {"op": "replace", "value": 3}}},
        }}}}
        selected = "seed=control"
    project = Project.model_validate(raw)
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, "nested", project)
    before = folders.read_plan("nested", revision=None)
    element = (
        {"kind": "wire", "id": "decode"} if kind == "wire" else
        {"kind": "port", "end": {"owner": "component", "component": "cell", "port": "output", "direction": "output"}}
    )
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        base = "/api/projects/nested"
        selection = {
            "revision": before.revision,
            "run": {"step": "baseline", "row": selected, "variant": variant},
            "scope": scope,
        }
        record = {
            "op": "add", "after": None,
            "owner": {"document": "reach-model", "composites": ["wrapper", "controller"], "element": element},
            "attachment": {"kind": "intervention", "id": "condition", "operation": "Scale", "fields": {"factor": 0.5}},
        }
        body = selection | {"difference": record}
        reply = client.post(f"{base}/model/evaluate-attachment", json=body)
        assert reply.status_code == 200, reply.text
        edited = reply.json()
        assert edited["changed"]["element"]["id"] == "condition"
        assert folders.read("nested").project == project
        invalid = deepcopy(body)
        invalid["difference"]["owner"]["composites"].append("missing")
        refused = client.post(f"{base}/model/evaluate-attachment", json=invalid)
        assert refused.status_code == 422
        assert refused.json()["detail"]["address"]["element"]["id"] == "condition"
        assert refused.json()["detail"]["address"]["composites"][-1] == "missing"
        committed = client.patch(base, json={"revision": before.revision, **edited["file"]})
        assert committed.status_code == 200, committed.text
        assert client.post(f"{base}/model/evaluate-attachment", json=body).status_code == 409
    after = folders.read_plan("nested", revision=None)
    prior = {run.key: run.request_id for run in before.plan.runs}
    result = {run.key: run.request_id for run in after.plan.runs}
    assert result.keys() == prior.keys()
    assert {key for key in prior if result[key] != prior[key]} == {RunKey(variant, "baseline", selected)}
    assert folders.read("nested").project.documents == project.documents
    stored = tmp_path / "nested" / ("variants/treated.json" if scope["kind"] == "variant" else "steps/baseline.json")
    saved = stored.read_text()
    assert '"after": null' in saved
    assert "_model_attachment_" not in saved and "graph.subgraphs" not in saved
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        selection["revision"] = after.revision
        reopened = client.post(f"{base}/model/evaluate-attachments", json=selection)
        assert reopened.status_code == 200, reopened.text
        assert reopened.json()["records"] == reopened.json()["effective"]
        assert reopened.json()["records"][0]["attachment"]["fields"] == {"factor": 0.5}
        change = {
            "op": "change", "address": edited["changed"],
            "attachment": record["attachment"] | {"fields": {"factor": 0.25}},
        }
        changed = client.post(f"{base}/model/evaluate-attachment", json=selection | {"difference": change})
        assert changed.status_code == 200, changed.text
        committed = client.patch(base, json={"revision": after.revision, **changed.json()["file"]})
        assert committed.status_code == 200, committed.text
        updated = folders.read_plan("nested", revision=None)
        selected_key = RunKey(variant, "baseline", selected)
        assert {run.key for run in updated.plan.runs if run.request_id != result[run.key]} == {selected_key}
        selection["revision"] = updated.revision
        remove = {"op": "remove", "address": edited["changed"]}
        removed = client.post(f"{base}/model/evaluate-attachment", json=selection | {"difference": remove})
        assert removed.status_code == 200, removed.text
        assert client.patch(base, json={"revision": updated.revision, **removed.json()["file"]}).status_code == 200
        restored = folders.read_plan("nested", revision=None)
        assert {run.key: run.request_id for run in restored.plan.runs} == prior


def _attachment_owner(wire):
    return {
        "document": "reach-model", "composites": ["wrapper", "controller"],
        "element": {"kind": "wire", "id": wire},
    }


def _scale_attachment(identity, factor):
    return {"kind": "intervention", "id": identity, "operation": "Scale", "fields": {"factor": factor}}


def _add_attachment(owner, identity, after, factor=0.5):
    return {"op": "add", "owner": owner, "attachment": _scale_attachment(identity, factor), "after": after}


def _attachment_address(owner, identity):
    return owner | {"element": {"kind": "attachment", "parent": owner["element"], "id": identity}}


def _model_controller(raw):
    return raw["documents"]["reach-model"]["payload"]["interior"]["components"]["wrapper"]["interior"][
        "components"
    ]["controller"]["interior"]


def test_evaluate_attachment_order_and_identity_follow_effective_graph():
    raw = _model_project()
    owner = _attachment_owner("carry")
    records = [
        _add_attachment(owner, "first", None),
        _add_attachment(owner, "second", None, 0.25),
        _add_attachment(owner, "following", "carry"),
    ]
    raw["steps"]["baseline"]["model_attachments"] = {"step": records}
    source = ModelDocument.model_validate(raw["documents"]["reach-model"]["payload"])
    components = ComponentRegistry()
    resolved = resolve_effective_model(
        "reach-model", source, components, tuple(MODEL_ATTACHMENT_BATCHES.validate_python({"step": records})["step"]),
    )
    def chain(model):
        wrapper = model.interior.components["wrapper"]
        controller = wrapper.interior.components["controller"]
        return [item.id for item in controller.interior.wires["carry"].attachments]

    assert chain(resolved) == ["first", "second", "carry", "following"]

    memo = ProjectPlanMemo()

    def evaluate_id():
        plan = compile_project(Project.model_validate(raw), memo=memo)
        assert not plan.diagnostics
        return next(run.request_id for run in plan.runs if run.kind == "evaluate")

    expected = evaluate_id()
    records[1]["after"] = "first"
    assert evaluate_id() == expected
    records[1]["after"] = "carry"
    assert evaluate_id() != expected


def test_evaluate_attachment_can_reorder_a_saved_record(tmp_path):
    raw = _model_project()
    owner = _attachment_owner("carry")
    raw["steps"]["baseline"]["model_attachments"] = {"step": [
        _add_attachment(owner, "condition", None),
    ]}
    folders = ProjectFolderService(tmp_path)
    _save_project(folders, "nested", Project.model_validate(raw))
    before = folders.read_plan("nested", revision=None)
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        base = "/api/projects/nested"
        selection = {
            "revision": before.revision, "scope": {"kind": "step"},
            "run": {"step": "baseline", "row": "default", "variant": "default"},
        }
        changed = client.post(f"{base}/model/evaluate-attachment", json=selection | {"difference": {
            "op": "change", "address": _attachment_address(owner, "condition"),
            "attachment": _scale_attachment("condition", 0.5), "after": "carry",
        }})
        assert changed.status_code == 200, changed.text
        assert client.patch(base, json={"revision": before.revision, **changed.json()["file"]}).status_code == 200
        after = folders.read_plan("nested", revision=None)
        reopened = client.post(f"{base}/model/evaluate-attachments", json=selection | {"revision": after.revision})
        assert reopened.json()["records"][0]["after"] == "carry"
    old = {run.kind: run.request_id for run in before.plan.runs}
    new = {run.kind: run.request_id for run in after.plan.runs}
    assert new["train"] == old["train"]
    assert new["evaluate"] != old["evaluate"]


@pytest.mark.parametrize(("invalid", "reason"), [
    ("missing-wire", "wire no longer exists"),
    ("missing-anchor", "anchor 'missing' no longer exists"),
    ("duplicate", "ID repeats"),
    ("cycle", "anchor cycle"),
    ("removed-anchor", "anchor 'carry' no longer exists"),
])
def test_evaluate_attachment_plan_refusals_are_addressed(invalid, reason):
    raw = _model_project()
    owner = _attachment_owner("missing" if invalid == "missing-wire" else "carry")
    identity = "carry" if invalid == "duplicate" else "condition"
    anchor = {
        "missing-anchor": "missing", "removed-anchor": "carry", "cycle": "condition",
    }.get(invalid)
    records = [_add_attachment(owner, identity, anchor)]
    if invalid == "removed-anchor":
        records.append({"op": "remove", "address": _attachment_address(owner, "carry")})
    raw["steps"]["baseline"]["model_attachments"] = {"step": records}
    plan = compile_project(Project.model_validate(raw), memo=ProjectPlanMemo())
    problem, = plan.diagnostics
    assert reason in problem.detail
    assert problem.model_address == ModelAddress.model_validate(_attachment_address(owner, identity))
    assert [run.kind for run in plan.runs] == ["train"]


def test_group_keeps_delay_on_a_crossing_gru_output():
    raw = _model_project()
    registry = ComponentRegistry()
    payload = raw["documents"]["reach-model"]["payload"]
    controller = payload["interior"]["components"]["wrapper"]["interior"]["components"]["controller"]
    controller["interior"]["wires"]["decode"]["attachments"] = [
        {"kind": "delay", "id": "feedback", "delay": {"value": 1, "unit": "st"}},
    ]
    state = ModelEditState(document=ModelDocument.model_validate(payload))
    ports = ModelRegistryPorts(registry)
    command = ModelEditCommand.model_validate({"operation": {
        "op": "group", "name": "Cell",
        "selection": [{
            "document": "reach-model", "composites": ["wrapper", "controller"],
            "element": {"kind": "component", "id": "cell"},
        }],
    }})
    result = apply_model_edit(
        "reach-model", state, command, ports=ports.port, accepts=ports.accepts,
    )
    assert isinstance(result, ModelEditResult), result
    raw["documents"]["reach-model"]["payload"] = result.state.document.model_dump(mode="json")
    planned = compile_project(Project.model_validate(raw))
    assert not planned.diagnostics, planned.diagnostics
    train = next(run for run in planned.runs if run.kind == "train")
    graph = decode_request(planned.requests[train.request_id], planned.documents.__getitem__).spec.graph
    resolved = graph.subgraphs["wrapper"].subgraphs["controller"]
    (delay,) = [wire.recurrent_initializer for wire in resolved.wires if wire.temporality == "recurrent"]
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
    lowered = lower_model_document("model", ModelDocument.model_validate(model), registry)
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
        lower_model_document("model", ModelDocument.model_validate(model), registry)
    assert exc.value.address.composites == ("equations",)


def test_linked_delay_repeats_the_completed_reset_state():
    raw = _model_project()
    model = raw["documents"]["reach-model"]["payload"]
    scope = ("wrapper", "controller")
    controller = model["interior"]["components"]["wrapper"]["interior"]["components"]["controller"]
    delay = controller["interior"]["wires"]["carry"]["attachments"][0]
    delay["delay"]["value"] = 3
    reference = ModelAddress.model_validate({
        "document": "reach-model", "composites": scope,
        "element": {"kind": "port", "end": {**_end("cell", "hidden"), "direction": "output"}},
    })
    delay["initial"] = {"kind": "name", "reference": reference.to_text()}
    plan = compile_project(Project.model_validate(raw))
    assert not plan.diagnostics
    train = next(run for run in plan.runs if run.kind == "train")
    request = decode_request(plan.requests[train.request_id], plan.documents.__getitem__).spec
    graph = compile_graph(
        GraphDocument(graph=request.graph), ComponentRegistry(),
        input_prototypes=prototypes_from_task_bindings(request.task_binding_spec),
    ).graph
    state = graph.init_state(key=jax.random.key(3))
    cell = graph.nodes["wrapper"].nodes["controller"].nodes["cell"]
    state = cell.replace_state_value(state, ("hidden",), jnp.full((cell.hidden_size,), 2.0, dtype=jnp.float32))
    carry = graph.initial_cycle_port_values(state)[("__nested__", "wrapper")][("__nested__", "controller")]
    (queue,) = carry.values()
    np.testing.assert_array_equal(queue, np.full((3, cell.hidden_size), 2.0))
    delay["initial"]["reference"] = reference.model_copy(update={"composites": ()}).to_text()
    refused = compile_project(Project.model_validate(raw))
    assert refused.diagnostics[0].model_address.element.id == "carry"


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
                "fields": {"input_size": 2, "hidden_size": 2, "tau": tau},
            }},
            "wires": {
                "drive": _wire({"owner": "inputs", "port": "setpoint"}, _end("cell", "input")),
                "read": _wire(_end("cell", "output"), {"owner": "outputs", "port": "estimate"}),
            },
        },
    }
    registry = ComponentRegistry()
    model = ModelDocument.model_validate(payload)
    lowered = lower_model_document("model", model, registry, StudioTaskBindingSpec())
    interface = lowered.interface("model", model, registry)
    assert interface.period_seconds == 0.01
    assert not interface.inputs[0].required
    output, = interface.outputs
    assert ModelAddress.from_text(output.id).element.end.port == "estimate"
    assert output.type == ValueType(base=ValueBase.REAL, shape=(2,), dimensions=LENGTH.combine(TIME, -1))
    executable = compile_graph(GraphDocument(graph=lowered.graph), registry)
    assert executable.graph.nodes["cell"].cell.alpha == pytest.approx(alpha)


def test_serial_mechanics_feedback_reads_start_state_and_refuses_within_step():
    pytest.importorskip("mujoco.mjx")
    spec = two_link_arm_6muscle_template_graph()
    nodes = {
        key: {"kind": "leaf", "name": key, "type": node.type, "fields": dict(node.params)}
        for key, node in spec.nodes.items() if key not in {"effector", "state"}
    }
    nodes["sensor"] = {
        "kind": "leaf", "name": "Position", "type": "feedbax.SensorOutput",
        "fields": {"quantity": "position"},
    }
    plant = {
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
    hold = {
        "kind": "leaf", "name": "Hold", "type": "feedbax.ZeroOrderHold",
        "fields": {"hold_steps": 1, "n_dims": 2},
    }
    linear = {
        "kind": "leaf", "name": "Readout", "type": "feedbax.Linear",
        "fields": {"input_size": 2, "output_size": 6, "initialization": "zeros"},
    }
    payload = {
        "step_seconds": 0.01,
        "world": {"gravity": [0.0, -9.81, 0.0]},
        "ports": {"outputs": [_port("pos", "position", shape=[2])]},
        "interior": {"components": {"arm": plant, "hold": hold, "linear": linear}, "wires": {
            "sample": _wire(_end("arm", "pos"), _end("hold", "input")),
            "sampled": _wire(_end("hold", "output"), _end("linear", "input")),
            "drive": _wire(_end("linear", "output"), _end("arm", "drive")),
            "observed": _wire(_end("arm", "pos"), {"owner": "outputs", "port": "pos"}),
        }},
    }
    registry = ComponentRegistry()
    lowered = lower_model_document("model", ModelDocument.model_validate(payload), registry)
    graph = compile_graph(GraphDocument(graph=lowered.graph), registry).graph
    key = jax.random.key(7)
    state = graph.init_state(key=key)
    initial = graph.nodes["arm"].initial_outputs(graph.nodes["arm"].state_view(state))["pos"]
    outputs, _, traces = graph.rollout_with_trace(
        {}, state, n_steps=3, key=key,
        trace=(GraphTraceRequest("port", "sample", node="hold", port="output"),),
    )
    np.testing.assert_array_equal(traces["sample"][0], initial)
    np.testing.assert_array_equal(traces["sample"][1:], outputs["position"][:-1])
    assert not np.array_equal(outputs["position"][0], initial)
    assert graph.nodes["arm"].backend.n_substeps == 2
    assert graph.nodes["arm"].backend.sub_dt == pytest.approx(0.005)

    reference = ModelAddress.model_validate({
        "document": "model", "composites": [],
        "element": {"kind": "port", "end": {**_end("arm", "pos"), "direction": "output"}},
    })
    payload["interior"]["wires"]["sample"]["attachments"] = [{
        "kind": "delay", "id": "lag", "delay": {"value": 2, "unit": "st"},
        "initial": {"kind": "name", "reference": reference.to_text()},
    }]
    linked = lower_model_document("model", ModelDocument.model_validate(payload), registry)
    delayed = compile_graph(GraphDocument(graph=linked.graph), registry).graph
    carries = delayed.initial_cycle_port_values(delayed.init_state(key=key))
    np.testing.assert_array_equal(carries["hold", "input"], np.stack([initial, initial]))
    payload["interior"]["wires"]["sample"]["attachments"] = []
    payload["interior"]["components"].pop("hold")
    payload["interior"]["wires"].pop("sampled")
    payload["interior"]["wires"]["sample"]["target"] = _end("linear", "input")
    with pytest.raises(ModelCompilationProblem, match="within-step") as error:
        lower_model_document("model", ModelDocument.model_validate(payload), registry)
    assert error.value.address.element.id == "drive"


@pytest.mark.parametrize("grouped", [False, True])
def test_three_level_model_train_evaluate_and_rename_current(host, tmp_path, capsys, record_property, grouped):
    from tests.test_document_train_pair import authored_documents, document_project

    registry = new_application_registry_bundle()
    jobs = JobService(host, registry, enable_evaluation_candidate=True)
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
            edit = prepare_evaluate_attachment(folders, project, EvaluateAttachmentRequest.model_validate({
                "revision": plan.revision,
                "run": {"step": "baseline", "row": "default", "variant": "default"},
                "difference": difference, "scope": {"kind": "step"},
            }), registry.components)
            patch, = edit.file.patches
            raw["steps"]["baseline"][patch.path.removeprefix("/")] = patch.value
            _save_project(folders, project, Project.model_validate(raw))
            plan = folders.read_plan(project, revision=None)
        assert next(run.request_id for run in plan.plan.runs if run.kind == "train") == train_before
    assert main(["plan", str(folders.root / project)]) == 0
    assert not plan.plan.diagnostics
    assert set(line.split()[1] for line in capsys.readouterr().out.splitlines()) == {
        run.request_id for run in plan.plan.runs
    }
    train = RunKey("default", "train", "default")
    evaluate = RunKey("default", "baseline", "default")
    (train_job,) = jobs.admit_plan(folders, plan, run_keys=(train,), token="nested-train").items
    (evaluation_job,) = jobs.admit_plan(folders, plan, run_keys=(evaluate,), token="nested-evaluate").items
    assert [train_job.disposition, evaluation_job.disposition] == ["admitted", "admitted"], (
        train_job, evaluation_job,
    )

    def finished(job):
        return wait_until(lambda: (
            job_row(host, job.job_id, "succeeded") or job_row(host, job.job_id, "failed")
        ), timeout=25)

    trained = finished(train_job)
    assert trained["state"] == "succeeded", trained.get("outcome")
    checkpoint = PublishedTrainingCheckpoint.model_validate(trained["outcome"]["completion"]["checkpoint"])
    slot = open_published_model_slot(checkpoint.checkpoint_root, checkpoint.pointer)
    assert slot is not None
    restored = slot.restore()
    assert restored.nodes["wrapper"].nodes["controller"].nodes["cell"].cell.weight_ih.shape[1] == 2
    evaluated = finished(evaluation_job)
    assert evaluated["state"] == "succeeded", evaluated.get("outcome")
    result = jobs.read_evaluation_result(project=project, store=plan.store_path, job_id=evaluation_job.job_id)
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
    with result.open_states() as dataset:
        n_samples = raw["steps"]["train"]["base"]["inline"]["training_config"]["n_reach_steps"] + 1
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

    reopened = ProjectFolderService(folders.root)
    reopened_view = ProjectPlanService(reopened, jobs).read(project, revision=None)
    reopened_model = reopened.read(project).project.documents["reach-model"].payload
    assert reopened_model == raw["documents"]["reach-model"]["payload"]
    assert reopened_view.settled.plan.requests == plan.plan.requests
    assert reopened_view.attempts_by_run[train][0].job_id == train_job.job_id
    assert reopened_view.attempts_by_run[train][0].successful_receipt == train_receipt
    assert reopened_view.attempts_by_run[evaluate][0].successful_receipt == result.receipt
    assert authenticated_receipts() == original_receipts

    model = raw["documents"]["reach-model"]["payload"]
    model["ports"]["inputs"][0]["name"] = "renamed_reach_target"
    overrides = raw["steps"]["train"]["base"]["inline"]["trial_documents"]["input_overrides"]
    overrides["renamed_reach_target"] = overrides.pop("reach_target")
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
        run = lower_canonical_train(request, registry.training_programs, registries=registry)
        prepared = prepare_managed_training(run, registry)
        update = prepared.kernel_context["studio_graph_update"].reported_update
        state, losses = prepared.initial_slots, []
        for _ in range(2):
            state = update(state)
            losses.append((state["train_loss"], state["loss_terms"]))
        model = state["model"].nodes["wrapper"].nodes["controller"]
        return losses, jax.tree.leaves(eqx.filter(model, eqx.is_array))

    before, values = trained()
    _rename_components(raw["documents"]["reach-model"]["payload"])
    after, renamed = trained()
    assert before == after
    assert len(values) == len(renamed) and values
    for a, b in zip(values, renamed, strict=True):
        np.testing.assert_array_equal(a, b)
