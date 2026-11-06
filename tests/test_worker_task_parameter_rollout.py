from types import SimpleNamespace

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.runtime._graph_key_schedule import derive_key
from feedbax.runtime.mapping import WhereDict
from feedbax.tasks import TaskTrialSpec, prepare_trial
from feedbax.web.worker.execution import compile_training_run, rollout_graph


def test_studio_rollout_matches_trial_preparation_with_constant_parameter() -> None:
    graph_spec = GraphSpec(
        nodes={
            "field": ComponentSpec(
                type="AffineValueComposer",
                params={
                    "schema_version": "feedbax.component.affine_value_composer.v2",
                    "output_block_size": 1,
                    "feature_rules": [{"kind": "identity", "state_slice": [0, 1]}],
                    "gain_init": [[2.0]],
                    "bias_init": [0.0],
                },
                input_ports=["base", "state", "target", "gain", "bias"],
                output_ports=["value"],
            )
        },
        output_ports=["value"],
        output_bindings={"value": ("field", "value")},
    )
    steps = {
        "base": [[1.0], [1.0], [1.0]],
        "state": [[2.0], [2.0], [2.0]],
        "gain": [[[2.0]], [[-1.0]], [[4.0]]],
    }
    bias = [5.0]
    task_data = [
        {
            "id": port,
            "label": port,
            "kind": "signal",
            "role": "model_input",
            "path": f"inputs.{port}",
            "bindable": True,
            "expected_shape": ["time", 1, 1] if port == "gain" else ["time", 1],
            "value_spec": {"mode": "constant", "value": values},
        }
        for port, values in steps.items()
    ]
    task_data.append(
        {
            "id": "trial_bias",
            "label": "Trial bias",
            "kind": "signal",
            "role": "graph_input",
            "path": "inputs.bias",
            "bindable": True,
            "expected_shape": [1],
            "value_spec": {"mode": "constant", "value": bias},
        }
    )
    bindings = [
        {
            "id": f"task:{port}->field:{port}",
            "source_data_id": port,
            "target_node_id": "field",
            "target_port": port,
            "role": "model_input",
        }
        for port in steps
    ]
    bindings.append(
        {
            "id": "task:trial_bias->field:bias",
            "source_data_id": "trial_bias",
            "target_node_id": "field",
            "target_port": "bias",
            "role": "graph_input",
        }
    )
    compiled = compile_training_run(
        graph_spec=graph_spec,
        training_spec={
            "optimizer": {"type": "adam"},
            "loss": {
                "type": "TargetStateLoss",
                "label": "value_zero",
                "selector": "graph_output:value",
                "target_value": [0.0],
                "weight": 1.0,
                "norm": "squared_l2",
            },
            "n_batches": 1,
            "batch_size": 1,
        },
        task_spec={"type": "Generic", "params": {}},
        task_binding_spec={
            "schema_version": "feedbax.spec.studio.task_bindings.v3",
            "exposed_data": task_data,
            "bindings": bindings,
        },
        cfg=SimpleNamespace(n_reach_steps=3),
        component_registry=ComponentRegistry(load_user_components=False),
    )

    graph = compiled.graph
    prepared = prepare_trial(
        graph,
        TaskTrialSpec(
            inits=WhereDict(),
            targets=WhereDict(),
            inputs={
                graph_input: (
                    jnp.asarray([bias] * 3) if port == "bias" else jnp.asarray(steps[port])
                )
                for graph_input, (_, port) in graph.input_bindings.items()
            },
        ),
    )
    key = jr.key(0)
    graph_key = derive_key(key, "studio-trial", "graph")
    expected, _, _ = graph.rollout_with_trace(
        prepared.inputs,
        prepared.init_state,
        key=graph_key,
        n_steps=prepared.n_steps,
        trace=(),
    )
    actual = rollout_graph(graph, compiled, key=key)

    np.testing.assert_allclose(expected["value"], [[10.0], [4.0], [14.0]])
    np.testing.assert_allclose(actual["outputs"]["value"], expected["value"])
