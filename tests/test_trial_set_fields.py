"""A Set in trials must switch the authored model without changing off trials."""

from dataclasses import replace

import jax
import jax.random as jr
import numpy as np

from feedbax.builtins.training import StandardTrainingSettings, supervised_gradient
from feedbax.application.composition import application_python_tasks
from feedbax.component_registry import ComponentRegistry
from feedbax.compiler.set_fields import lower_trial_sets
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.project import Project, RootDocument
from feedbax.contracts.project_sweep import TrialSetField
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.set_document import SET_DOCUMENT_SCHEMA_ID
from feedbax.contracts.task_document import TaskDocument
from feedbax.execution.trial_resolution import (
    PreparedTrialExecution, PreviewTrialSource, compile_trial_execution, preview_trial_source,
)
from feedbax.planner.project_plan import compile_project
from feedbax.runtime._graph_key_schedule import model_initialization_key
from feedbax.tasks.document import TaskParameters
from feedbax.training.method_runtime import TrainingLoop, initialize_method
from feedbax.training.methods import TrainingMethodRegistry
from feedbax.plugins.declarations import DistributionSource, callable_declaration
from tests.test_document_train_pair import authored_documents, document_project


def test_curl_set_entry_uses_its_amplitude_and_onset_draws():
    registry = ComponentRegistry()
    functions, *_ = authored_documents()

    def port(owner, id_, component=None):
        return {"owner": owner, "port": id_} | ({"component": component} if component else {})

    def wire(a, b):
        return {"source": a, "target": b}

    def address(kind, **fields):
        return ModelAddress.model_validate({"composites": [], "element": {"kind": kind, **fields}}).to_text()

    def choice(*values, unit=None):
        return {"kind": "draw", "family": "choice", "args": [
            {"kind": "literal", "value": value} | ({"unit": unit} if unit else {}) for value in values
        ]}

    command = port("component", "output", "drive")
    effector = port("component", "effector", "plant")
    model = ModelDocument.model_validate({
        "step_seconds": 0.01,
        "ports": {
            "inputs": [{"id": "drive", "name": "drive", "type": {"shape": [2]}}],
            "outputs": [{"id": "force", "name": "force", "type": {"shape": [2]}},
                        {"id": "effector", "name": "effector", "type": {}}],
        },
        "interior": {
            "components": {
                "drive": {"kind": "leaf", "name": "Drive", "type": "feedbax.Input"},
                "plant": {"kind": "leaf", "name": "Plant", "type": "feedbax.PointMass"},
            },
            "wires": {
                "input": wire(port("inputs", "drive"), port("component", "input", "drive")),
                "force": wire(command, port("component", "force", "plant")),
                "show-force": wire(command, port("outputs", "force")),
                "motion": wire(effector, port("outputs", "effector")),
            },
        },
    })
    task = TaskDocument.model_validate({
        "id": "trial", "variables": [],
        "epochs": [{"id": "move", "name": "Move", "end": {"kind": "length", "length": {
            "kind": "literal", "value": 10, "unit": "st",
        }}}],
        "signals": [{"id": "drive", "name": "drive", "type": {"base": "real", "shape": [2]},
                     "cells": [{"id": "force", "first_epoch": "move", "last_epoch": "move", "value": {
                         "kind": "call", "function": "vector",
                         "args": [{"kind": "literal", "value": n} for n in (1.0, 0.0)],
                     }}]}],
    })
    intervention = {"kind": "attachment", "id": "pulse", "parent": {"kind": "wire", "id": "input"}}
    bank = RootDocument(schema=SET_DOCUMENT_SCHEMA_ID, payload={"entries": {"curl": {
        "task_variables": [
            {"id": "amp", "name": "Amplitude", "type": {"base": "real"}, "value": choice(1.0, 2.0)},
            {"id": "onset", "name": "Onset", "type": {"base": "real", "dimensions": {"time": 1}},
             "value": choice(0.02, 0.04, unit="s")},
        ],
        "task_links": [
            {"variable": "amp", "model_field": address(
                "field", parent={"kind": "component", "id": "curl"}, path=["fields", "scale"],
            )},
            {"variable": "onset", "model_field": address(
                "field", parent=intervention, path=["window", "spans", 0, "start", "offset"],
            )},
        ],
        "differences": {
            "/interior/components/curl": {"value": {
                "kind": "leaf", "name": "Curl", "type": "feedbax.CurlField", "fields": {"active": True},
            }},
            "/interior/wires/curl-input": {"value": wire(effector, port("component", "effector", "curl"))},
            "/interior/wires/input/attachments": {"value": [{
                "kind": "intervention", "id": "pulse", "operation": "add", "fields": {"value": {
                    "kind": "name", "reference": address(
                        "port", end=port("component", "force", "curl") | {"direction": "output"},
                    ),
                }},
                "window": {"spans": [{
                    "kind": "interval", "start": {
                        "anchor": {"kind": "epoch", "epoch": "move", "edge": "start"},
                        "offset": {"value": 0, "unit": "st"},
                    }, "end": {"offset": {"value": 2, "unit": "st"}},
                }]},
            }]},
        },
    }}})
    sampled = TrialSetField.model_validate({"axis": {"set": "bank"}, "sampling": {"kind": "weighted"}})
    model, task = lower_trial_sets(model, task, {"bank": sampled}, {"bank": bank}.get, registry)
    execution = compile_trial_execution(model, task, registry, functions, python_tasks=application_python_tasks())
    assert isinstance(execution, PreparedTrialExecution), execution
    parameters = TaskParameters()
    trials, keys = execution.execution.sample_trials(parameters, run_key=jr.key(1), update_key=jr.key(2), count=16)
    graph = execution.executable.graph

    def rollout(trial, key):
        inputs = {port: trial.inputs[signal] for port, signal in execution.execution.input_bindings}
        reset = execution.execution.reset.prepare(graph, trial.inits, key=key, inputs=inputs)
        return graph(inputs, reset.state, key=key, n_steps=10, cycle_init=reset.cycle_init)[0]

    outputs = jax.vmap(rollout)(trials, keys)
    velocity = np.asarray(outputs["effector"].vel)
    input_velocity = np.concatenate((np.zeros_like(velocity[:, :1]), velocity[:, :-1]), axis=1)
    names = {variable.name: variable.id for variable in task.variables}
    amps = np.asarray(trials.variables[names["curl · Amplitude"]])
    onset = np.rint(np.asarray(trials.variables[names["curl · Onset"]]) / 0.01).astype(int)
    selected = np.asarray(trials.variables["bank"], dtype=bool)
    assert set(onset) == {2, 4} and set(amps) == {1.0, 2.0} and set(selected) == {False, True}
    steps = np.arange(10)
    gate = selected[:, None] & (onset[:, None] <= steps) & (steps < onset[:, None] + 2)
    curl = amps[:, None, None] * input_velocity[..., ::-1] * np.array([-1.0, 1.0])
    expected = np.array([1.0, 0.0]) + np.where(gate[..., None], curl, 0.0)
    np.testing.assert_allclose(outputs["force"], expected, atol=1e-6)


def test_weighted_set_switches_real_training_trials_at_the_window_boundaries():
    functions, model, task, objective = authored = authored_documents()
    raw = document_project(authored)
    window = {"spans": [{
        "kind": "interval",
        "start": {
            "anchor": {"kind": "epoch", "edge": "start", "epoch": "move"},
            "offset": {"kind": "draw", "family": "choice", "args": [
                {"kind": "literal", "value": onset, "unit": "st"} for onset in (15, 30, 45)
            ]},
        },
        "end": {"offset": {"value": 5, "unit": "st"}},
    }]}
    entries = {
        name: {"differences": {"/interior/wires/effector-wire/attachments": {
            "value": [{
                "kind": "intervention", "id": name, "operation": "add", "fields": {"value": vector},
                "window": window,
            }],
        }}}
        for name, vector in {"x": [1.0, 0.0], "y": [0.0, 2.0], "z": [-3.0, -3.0]}.items()
    }
    raw["documents"]["bank"] = {"schema": "feedbax.set_document.v1", "payload": {"entries": entries}}
    raw["steps"]["train"]["set_fields"] = {"bank": {
        "axis": {"set": "bank"},
        "sampling": {"kind": "weighted", "weights": {"x": 2, "y": 3, "z": 4}, "none": 1},
    }}
    plan = compile_project(Project.model_validate(raw))
    assert not plan.diagnostics, plan.diagnostics
    (run,) = plan.runs
    assert run.admission_problem is None, run.problems
    document = decode_request(plan.requests[run.request_id], plan.documents.__getitem__).spec.trial_documents
    registry = ComponentRegistry()
    seed = 4

    def prepare(model, task):
        result = compile_trial_execution(
            model, task, registry, functions, python_tasks=application_python_tasks(),
            objective=objective, input_bindings=document.input_bindings,
            initialization_key=model_initialization_key(seed),
        )
        assert isinstance(result, PreparedTrialExecution), result
        return result

    compiled = prepare(document.model, document.task)
    nominal = prepare(model, task)
    parameters = TaskParameters()
    representative = preview_trial_source(
        document.model, document.task, registry, functions, python_tasks=application_python_tasks(),
        input_bindings=document.input_bindings, nominal=True,
    )
    assert isinstance(representative, PreviewTrialSource), representative
    assert int(representative.task.ahead(parameters, jr.key(0), jr.key(0)).variables["bank"]) == 3
    sampled, _ = compiled.execution.sample_trials(parameters, run_key=jr.key(0), update_key=jr.key(1), count=2000)
    variable = next(variable for variable in document.task.variables if variable.id == "bank")
    assert variable.set_choices.set == "bank"
    assert variable.set_choices.entries == (None, "x", "y", "z")
    assert np.allclose(np.bincount(sampled.variables["bank"], minlength=4) / 2000, [0.1, 0.2, 0.3, 0.4], atol=0.035)
    observed_perturbations = np.asarray(sampled.variables["bank"][:64])
    assert set(observed_perturbations) == {0, 1, 2, 3}
    methods = TrainingMethodRegistry()
    methods.register_declaration(
        supervised_gradient, callable_declaration(supervised_gradient),
        key="feedbax.supervised_gradient", source=DistributionSource("feedbax", "feedbax", "0.2.0"),
    )
    method = methods.resolve("feedbax.supervised_gradient")
    settings = StandardTrainingSettings.model_validate({
        "optimizer": {"type": "adam", "learning_rate": {"type": "constant_schedule", "value": 0.003}},
    })

    def inspect_update(state, ctx):
        updated, metrics = supervised_gradient(state, ctx)
        trials, keys = ctx.sample_trials()
        (union_output,) = ctx.rollout(state.model, trials, keys).values()
        (base_output,) = nominal.execution.rollout_batch(nominal.executable.graph, trials, keys)[0].values()
        return updated, {
            **metrics, "delta": union_output - base_output, "bank": trials.variables["bank"],
            "move": trials.epochs.starts[:, -1],
        }

    graph = compiled.executable.graph
    initial = initialize_method(method, settings, models=(graph,), seed=seed, replica_ids=("0",))
    result = TrainingLoop(
        replace(method, update=inspect_update), settings, execution=compiled.execution,
        objective=compiled.objective, batch_size=64, seed=seed, replica_ids=("0",),
    ).run_block(initial, completed=0, count=1)
    assert result.completed == 1 and not result.rejected
    recorded = np.asarray(result.metrics["bank"][0, 0])
    movements = np.asarray(result.metrics["move"][0, 0])
    deltas = np.asarray(result.metrics["delta"][0, 0])
    assert set(recorded) == {0, 1, 2, 3}
    assert len(set(movements)) > 1
    starts = set()
    deviations = ((0, 0), (1, 0), (0, 2), (-3, -3))
    for entry, movement, delta in zip(recorded, movements, deltas, strict=True):
        changed = np.flatnonzero(np.linalg.norm(delta, axis=1) > 1e-6)
        expected = np.zeros_like(delta)
        if entry:
            onset = int(changed[0])
            starts.add(onset - movement)
            assert onset - movement in (15, 30, 45)
            expected[onset:onset + 5] = deviations[entry]
        np.testing.assert_allclose(
            delta, expected, rtol=1e-5, atol=1e-6,
        )
    assert starts == {15, 30, 45}
