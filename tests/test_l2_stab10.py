"""The archived stab10 topology can train and publish native semantic weights."""

import json
from operator import attrgetter
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from feedbax.application.composition import load_application
from feedbax.builtins.training import StandardTrainingSettings
from feedbax.contracts.method_checkpoint import MethodCheckpointClock, MethodCheckpointIdentity
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.model_values import model_owner_problem
from feedbax.contracts.objective_document import ObjectiveDocument
from feedbax.contracts.semantic_values import semantic_key
from feedbax.contracts.task_document import TaskDocument
from feedbax.execution.trial_resolution import PreparedTrialExecution, compile_trial_execution
from feedbax.host.method_checkpoint import read_method_checkpoint, write_method_checkpoint
from feedbax.host.model_checkpoint import SemanticModelWriter
from feedbax.persistence.array_container import decode_array_container
from feedbax.runtime.raw_state import bind_raw_state
from feedbax.tasks.document import TaskParameters
from feedbax.training.method_runtime import TrainingLoop, initialize_method, replica_value


def _leaf(kind, **fields):
    return {"kind": "leaf", "name": kind, "type": "feedbax." + kind, "fields": fields}


def _end(text):
    owner, port = text.split(".")
    if owner in ("inputs", "outputs"):
        return {"owner": owner, "port": port}
    return {"owner": "component", "component": owner, "port": port}


def _wire(source, target, *, delayed=False):
    attachments = [{"kind": "delay", "id": "delay", "delay": {"value": 1, "unit": "st"}}] if delayed else []
    return {"source": _end(source), "target": _end(target), "attachments": attachments}


def _ports(inputs, outputs):
    def ports(items):
        return [{"id": name, "name": name, "type": {"shape": [width]}} for name, width in items.items()]

    return {"inputs": ports(inputs), "outputs": ports(outputs)}


def _model():
    # Matrices from dabbf70 stab10 graph c062f700. Its initial vector and delta are zero.
    matrices = json.loads(Path(__file__).with_name("fixtures").joinpath("stab10_matrices.json").read_bytes())
    width = len(matrices["A"])
    feedback = {
        "kind": "composite", "name": "Feedback", "ports": _ports({"state": width, "target": 2}, {"feedback": 6}),
        "interior": {
            "components": {
                "split": _leaf("Demux", sizes=[30, 2, 2, 2]), "position": _leaf("Subtract"),
                "velocity": _leaf("Gain", gain=-1.0), "mux": _leaf("Mux", n_inputs=3),
            },
            "wires": {
                "state": _wire("inputs.state", "split.input"),
                "target": _wire("inputs.target", "position.a"),
                "select_position": _wire("split.out_1", "position.b"),
                "select_velocity": _wire("split.out_2", "velocity.input"),
                "error": _wire("position.out", "mux.in_0"),
                "rate": _wire("velocity.output", "mux.in_1"),
                "force": _wire("split.out_3", "mux.in_2"),
                "feedback": _wire("mux.output", "outputs.feedback"),
            },
        },
    }
    controller = {
        "kind": "composite", "name": "Controller",
        "ports": _ports({"context": 1, "feedback": 6}, {"command": 2, "hidden": 180}),
        "interior": {
            "components": {
                "carry": _leaf("Gain", gain=1.0),
                "cell": _leaf("GRU", input_size=7, hidden_size=180, dtype="float32"),
                "input_mux": _leaf("Mux", n_inputs=2),
                "readout": _leaf("Linear", input_size=180, output_size=2, activation="identity", use_bias=True, dtype="float32"),
            },
            "wires": {
                "context": _wire("inputs.context", "input_mux.in_1"),
                "feedback": _wire("inputs.feedback", "input_mux.in_0"),
                "encoded": _wire("input_mux.output", "cell.input"),
                "hidden": _wire("cell.hidden", "outputs.hidden"),
                "decode_hidden": _wire("cell.hidden", "readout.input"),
                "encode_carry": _wire("cell.hidden", "carry.input"),
                "memory": _wire("carry.output", "cell.hidden", delayed=True),
                "command": _wire("readout.output", "outputs.command"),
            },
        },
    }
    return ModelDocument.model_validate({
        "step_seconds": 0.01,
        "ports": _ports(
            {"target": 2, "controller_context": 1, "epsilon": 6, "command_disturbance": 2, "feedback_disturbance": 6},
            {"state": width, "command": 2, "feedback": 6, "hidden": 180},
        ),
        "interior": {
            "components": {
                "command_sum": _leaf("Sum"), "controller": controller,
                "feedback": feedback, "feedback_sum": _leaf("Sum"),
                "mechanics": _leaf(
                    "StructuralLinearStateSpace", **matrices, delta_A=[[0.0] * width for _ in range(width)],
                    active=False, scale=1.0, pos_slice=[0, 2], vel_slice=[2, 4],
                ),
                "motor_noise": _leaf(
                    "Channel", delay=0, input_shape=[2], noise_model="signal_dependent_plus_additive",
                    additive_noise_std=0.03162277660168379, signal_dependent_noise_std=0.1,
                    noise_role="motor_command", noise_timing="pre_force_filter",
                ),
                "process_mux": _leaf("Mux", n_inputs=2),
                "process_noise": _leaf("Noise", shape=[2], std=0.03162277660168379),
                "sensory_noise": _leaf(
                    "Channel", delay=0, input_shape=[6], noise_model="additive_gaussian", noise_std=0.004791329788133908,
                    noise_role="sensory_feedback", noise_timing="pre_controller",
                ),
                "target": _leaf("Input"),
            },
            "wires": {
                "target_input": _wire("inputs.target", "target.input"),
                "target_feedback": _wire("target.output", "feedback.target"),
                "state_feedback": _wire("mechanics.state", "feedback.state", delayed=True),
                "add_feedback": _wire("feedback.feedback", "feedback_sum.a"),
                "feedback_disturbance": _wire("inputs.feedback_disturbance", "feedback_sum.b"),
                "noise_feedback": _wire("feedback_sum.output", "sensory_noise.input"),
                "sense": _wire("sensory_noise.output", "controller.feedback"),
                "context": _wire("inputs.controller_context", "controller.context"),
                "noise_command": _wire("controller.command", "motor_noise.input"),
                "add_command": _wire("motor_noise.output", "command_sum.a"),
                "command_disturbance": _wire("inputs.command_disturbance", "command_sum.b"),
                "drive": _wire("command_sum.output", "mechanics.force"),
                "epsilon": _wire("inputs.epsilon", "process_mux.in_0"),
                "sample_noise": _wire("process_noise.output", "process_mux.in_1"),
                "disturbance": _wire("process_mux.output", "mechanics.epsilon"),
                "state": _wire("mechanics.state", "outputs.state"),
                "command": _wire("controller.command", "outputs.command"),
                "observed_feedback": _wire("sensory_noise.output", "outputs.feedback"),
                "hidden": _wire("controller.hidden", "outputs.hidden"),
            },
        },
    })


def _task():
    def literal(value):
        return {"kind": "literal", "value": value}

    signals = {
        "target": [0.3, -0.6], "controller_context": [0.1], "epsilon": [0.0] * 6,
        "command_disturbance": [0.0] * 2, "feedback_disturbance": [0.0] * 6,
    }
    return TaskDocument.model_validate({
        "id": "stab10", "variables": [],
        "epochs": [{"id": "move", "name": "Move", "end": {
            "kind": "length", "length": literal(4) | {"unit": "st"},
        }}],
        "signals": [{
            "id": name, "name": name, "type": {"base": "real", "shape": [len(value)]},
            "cells": [{
                "id": name, "first_epoch": "move", "last_epoch": "move",
                "value": {"kind": "call", "function": "vector", "args": [literal(item) for item in value]},
            }],
        } for name, value in signals.items()],
    })


def test_stab10_trains_and_restores_a_semantic_checkpoint(tmp_path):
    model, task = _model(), _task()
    command = ModelAddress.model_validate({
        "composites": [], "element": {"kind": "port", "end": {"owner": "outputs", "port": "command"}},
    }).to_text()
    objective = ObjectiveDocument.model_validate({"id": "stab10", "terms": [{
        "id": "effort", "reduction": "last",
        "measure": {"kind": "call", "function": "sqnorm", "args": [{"kind": "name", "reference": command}]},
    }]})
    bundle = load_application().bundle

    def realize(seed):
        result = compile_trial_execution(
            model, task, bundle.components, bundle.expression_functions, python_tasks=bundle.python_tasks,
            objective=objective, initialization_key=jax.random.key(seed),
        )
        assert isinstance(result, PreparedTrialExecution), result
        return result

    initial = realize(19)
    runtime, graph = initial.executable.graph, initial.executable.resolved.graph
    assert model_owner_problem(graph, "checkpoint", bundle.components) is None
    assert sum(wire.temporality == "recurrent" for wire in graph.wires) == 1
    assert sum(wire.temporality == "recurrent" for wire in graph.subgraphs["controller"].wires) == 1
    key = jax.random.key(13)
    feedback = runtime.nodes["feedback"]
    output, _ = feedback(
        {"state": jnp.arange(36, dtype=jnp.float32), "target": jnp.array([7.0, 11.0])},
        feedback.init_state(key=key), key=key,
    )
    np.testing.assert_array_equal(output["feedback"], [-23.0, -20.0, -32.0, -33.0, 34.0, 35.0])

    method = bundle.training_methods.resolve("feedbax.supervised_gradient")
    ids, settings = ("stab10",), StandardTrainingSettings()
    state = initialize_method(method, settings, models=(runtime,), seed=19, replica_ids=ids)
    result = TrainingLoop(
        method, settings, execution=initial.execution, objective=initial.objective, batch_size=1, seed=19, replica_ids=ids,
    ).run_block(state, completed=0, count=1)
    assert result.completed == 1 and not result.rejected
    assert np.isfinite(result.metrics["train_loss"]).all()
    trained = replica_value(result.state.model, 0)

    writer = SemanticModelWriter.for_graph(graph, None, bundle.components)
    assert writer is not None
    written = write_method_checkpoint(
        tmp_path, run_id="stab10",
        identity=MethodCheckpointIdentity(request_id="train:" + "1" * 64, method_key=method.key),
        clock=MethodCheckpointClock(completed_updates=1, segment_start=0, schedule_origin=0, seed=19),
        state=result.state, model_writer=writer, model_templates=(runtime,), replica_ids=ids,
    )
    slot = read_method_checkpoint(written.root, written.latest_pointer, components=bundle.components).model
    restored = slot.restore(realize(23).executable.graph)
    (replica,) = slot.replicas
    tensors = decode_array_container(
        slot.provider.get_object(replica.arrays),
        expected_descriptor_sha256=replica.descriptor.array_descriptor_sha256,
    ).tensors

    changed = []
    for owner, symbol, attr in (("cell", "input_kernel", "cell.weight_ih"), ("readout", "kernel", "layer.weight")):
        address = semantic_key("model", "value", ("controller", owner), symbol)
        def weights(graph):
            return attrgetter(attr)(graph.nodes["controller"].nodes[owner])

        changed.append(not np.array_equal(weights(runtime), weights(trained)))
        np.testing.assert_array_equal(weights(trained), tensors[address])
        np.testing.assert_array_equal(weights(restored), tensors[address])
    assert any(changed)

    def rollout(candidate):
        layout = bind_raw_state(candidate, slot.graph)
        trial = initial.execution.evaluate(
            candidate, initial.objective, TaskParameters(), key=key, trial_ids=("0",), capture_state=layout.capture,
        )
        return trial.retained, trial.history

    assert eqx.tree_equal(rollout(trained), rollout(restored), typematch=True)
