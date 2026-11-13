"""Plan and shared graph compilation for a model document with nested owners."""

from copy import deepcopy

import jax
import numpy as np
import equinox as eqx
import jax.numpy as jnp
import pytest

from feedbax.cli import main
from feedbax.compiler import GraphDocument, compile_graph
from feedbax.compiler.model_document import lower_model_document
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID, ModelDocument
from feedbax.contracts.model_edit import ModelEditCommand, ModelEditResult, ModelEditState
from feedbax.contracts.project import Project
from feedbax.contracts.project_plan import RunKey, compile_project
from feedbax.contracts.request_identity import decode_request
from feedbax.jobs import JobService
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.studio.canonical_train import lower_canonical_train
from feedbax.studio.model_edit import apply_model_edit
from feedbax.studio.model_ports import ModelRegistryPorts
from feedbax.runtime.graph import GraphTraceRequest
from feedbax.runtime._signal_schedule import SignalScheduleError, resolve_signal_schedule
from feedbax.training.checkpoint_custody import open_published_model_slot
from feedbax.training.invocation import prepare_training_invocation
from feedbax.training.preparation import ExecutionPreparationResult
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.contracts.studio_training import StudioGraphMethodPayload
from feedbax.contracts.training import TrainingSpec
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanService
from feedbax.web.worker.execution import compile_training_run, prototypes_from_task_bindings
from tests.test_canonical_evaluate_admission import _save_project, authored_evaluation_project
from tests.test_local_job_host import host as host_fixture, job_row, wait_until

host = host_fixture


def _end(component, port):
    return {"owner": "component", "component": component, "port": port}


def _wire(source, target, attachments=()):
    return {"source": source, "target": target, "attachments": list(attachments)}


def _port(id_, name, *, shape=None):
    return {"id": id_, "name": name, "type": {"shape": shape}}


def test_noncontiguous_group_preserves_rollout_and_gradient():
    scale = {
        "kind": "leaf", "name": "Scale", "type": "feedbax.Scale",
        "fields": {"scale": 2.0, "fixed": True},
    }
    boundary = {"owner": "inputs", "port": "signal"}
    source_wire = _wire(boundary, _end("a", "input"), [
        {"kind": "delay", "id": "delay", "delay": {"value": 1, "unit": "st"}},
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
        np.testing.assert_array_equal(output, [0.0, 24.0, 48.0, 72.0])
        np.testing.assert_array_equal(trace, output)
        gradient = jax.grad(lambda data: run(graph, data)[0].sum())(signal)
        np.testing.assert_array_equal(gradient, [24.0, 24.0, 24.0, 0.0])

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
    term = raw["steps"]["train"]["base"]["inline"]["objective"]["loss"]
    term["selector"] = f"path:{group}.plant.effector.pos"
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


@pytest.mark.parametrize("grouped", [False, True])
def test_three_level_model_train_evaluate_and_rename_current(host, tmp_path, capsys, grouped):
    registry = new_application_registry_bundle()
    jobs = JobService(host, registry, enable_evaluation_candidate=True)
    folders = ProjectFolderService(tmp_path / "projects")
    raw = _model_project()
    group = _group_reaching_pipeline(raw, registry.components) if grouped else None
    project = "nested"
    _save_project(folders, project, Project.model_validate(raw))
    plan = folders.read_plan(project, revision=None)
    assert main(["plan", str(folders.root / project)]) == 0
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
    with result.open_states() as dataset:
        assert dataset.data("/v1/model/state/wrapper/controller/cell/hidden").shape[-2] == 101
        assert dataset.data("/v1/model/carry/wrapper/controller/carry/carry").shape[-2] == 101
        assert np.isfinite(dataset.data("objective/total")).all()
        if group is not None:
            assert dataset.data(f"/v1/model/state/{group}/plant/effector~1position").shape[-2] == 101

    _rename_components(raw["documents"]["reach-model"]["payload"])
    _save_project(folders, project, Project.model_validate(raw))
    renamed = folders.read_plan(project, revision=None)
    assert [run.request_id for run in renamed.plan.runs] == [run.request_id for run in plan.plan.runs]
    joined = ProjectPlanService(folders, jobs).read(project, revision=renamed.revision)
    assert joined.attempts_by_run[train][0].job_id == train_job.job_id
    assert joined.attempts_by_run[evaluate][0].successful_receipt == result.receipt


def test_same_seed_nested_rename_preserves_values_and_batch_loss():
    raw = _model_project()
    registry = new_application_registry_bundle()

    def trained():
        plan = compile_project(Project.model_validate(raw))
        row = next(run for run in plan.runs if run.kind == "train")
        request = decode_request(plan.requests[row.request_id], plan.documents.__getitem__).spec
        run = lower_canonical_train(request, registry.training_programs)
        prepared = prepare_training_invocation(run, registry, run_id=None, resume=False)
        assert isinstance(prepared, ExecutionPreparationResult)
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
