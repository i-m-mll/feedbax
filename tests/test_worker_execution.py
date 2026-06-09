from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import threading

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from feedbax.graph_templates import network_template_graph
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.web.worker.execution import (
    compile_training_run,
    rollout_graph,
    run_training_graph,
)


def _linear_graph_spec(component_type: str = "Linear") -> dict:
    return GraphSpec(
        nodes={
            "readout": ComponentSpec(
                type=component_type,
                params={
                    "input_size": 1,
                    "output_size": 1,
                    "activation": "identity",
                    "trainable": True,
                },
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        output_ports=["output"],
        output_bindings={"output": ("readout", "output")},
    ).model_dump(mode="json", exclude_none=True)


def _task_binding_spec() -> dict:
    return {
        "schema_version": "feedbax.studio.task_bindings.v2",
        "exposed_data": [
            {
                "id": "model_input",
                "label": "Model input",
                "kind": "signal",
                "role": "model_input",
                "path": "inputs.model",
                "bindable": True,
                "expected_shape": ["time", 1],
                "value_spec": {
                    "mode": "constant",
                    "value": [1.0],
                    "dtype": "float32",
                    "shape": ["time", 1],
                },
                "metadata": {},
            }
        ],
        "bindings": [
            {
                "id": "task:model_input->readout:input",
                "source_data_id": "model_input",
                "target_node_id": "readout",
                "target_port": "input",
                "role": "model_input",
                "metadata": {},
            }
        ],
        "metadata": {},
    }


def _network_task_binding_spec() -> dict:
    return {
        "schema_version": "feedbax.studio.task_bindings.v2",
        "exposed_data": [
            {
                "id": "model_input",
                "label": "Model input",
                "kind": "signal",
                "role": "model_input",
                "path": "inputs.model",
                "bindable": True,
                "expected_shape": ["time", 1],
                "value_spec": {
                    "mode": "constant",
                    "value": [1.0],
                    "dtype": "float32",
                    "shape": ["time", 1],
                },
                "metadata": {},
            },
            {
                "id": "feedback",
                "label": "Feedback",
                "kind": "signal",
                "role": "model_input",
                "path": "inputs.feedback",
                "bindable": True,
                "expected_shape": ["time", 1],
                "value_spec": {
                    "mode": "constant",
                    "value": [0.0],
                    "dtype": "float32",
                    "shape": ["time", 1],
                },
                "metadata": {},
            },
        ],
        "bindings": [
            {
                "id": "task:model_input->network:input",
                "source_data_id": "model_input",
                "target_node_id": "network",
                "target_port": "input",
                "role": "model_input",
                "metadata": {},
            },
            {
                "id": "task:feedback->network:feedback",
                "source_data_id": "feedback",
                "target_node_id": "network",
                "target_port": "feedback",
                "role": "model_input",
                "metadata": {},
            },
        ],
        "metadata": {},
    }


def _training_spec(**overrides) -> dict:
    spec = {
        "optimizer": {"type": "adam", "params": {"learning_rate": 0.1}},
        "loss": {
            "type": "TargetStateLoss",
            "label": "output_zero",
            "selector": "graph_output:output",
            "target_value": [0.0],
            "weight": 1.0,
            "norm": "squared_l2",
        },
        "n_batches": 2,
        "batch_size": 1,
    }
    spec.update(overrides)
    return spec


def _cfg(**overrides):
    values = {
        "n_reach_steps": 4,
        "learning_rate": 0.1,
        "grad_clip": 1.0,
        "snapshot_interval": 10,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_compile_training_run_accepts_full_graph_without_bridge_nodes() -> None:
    compiled = compile_training_run(
        graph_spec=_linear_graph_spec(),
        training_spec=_training_spec(),
        task_spec={"type": "Generic", "params": {}},
        task_binding_spec=_task_binding_spec(),
        cfg=_cfg(),
    )

    assert compiled.metadata["execution"] == "generic_graph"
    assert compiled.task_inputs[0].target_node == "readout"
    assert compiled.task_inputs[0].graph_input in compiled.graph.input_ports
    assert compiled.trainable_nodes == ("readout",)


def test_compile_training_run_uses_array_leaf_trainability_for_network_template() -> None:
    graph_spec = network_template_graph(
        {"input_size": 2, "hidden_size": 3, "out_size": 1}
    ).model_dump(mode="json", exclude_none=True)

    compiled = compile_training_run(
        graph_spec=graph_spec,
        training_spec=_training_spec(),
        task_spec={"type": "Generic", "params": {}},
        task_binding_spec=_network_task_binding_spec(),
        cfg=_cfg(),
    )
    trainable, _ = eqx.partition(compiled.graph, compiled.trainable_filter)

    assert compiled.trainable_nodes == ("network",)
    assert all(
        eqx.is_inexact_array(leaf)
        for leaf in jax.tree_util.tree_leaves(trainable)
        if leaf is not None
    )


def test_rollout_graph_threads_network_template_recurrence() -> None:
    graph_spec = network_template_graph(
        {"input_size": 2, "hidden_size": 3, "out_size": 1}
    ).model_dump(mode="json", exclude_none=True)

    compiled = compile_training_run(
        graph_spec=graph_spec,
        training_spec=_training_spec(),
        task_spec={"type": "Generic", "params": {}},
        task_binding_spec=_network_task_binding_spec(),
        cfg=_cfg(n_reach_steps=5),
    )
    rollout = rollout_graph(compiled.graph, compiled, key=jax.random.PRNGKey(3))
    output = rollout["outputs"]["output"]

    assert output.shape == (5, 1)
    assert not jnp.allclose(output[0], output[-1])


def test_run_training_graph_trains_tiny_full_graph() -> None:
    compiled = compile_training_run(
        graph_spec=_linear_graph_spec(),
        training_spec=_training_spec(),
        task_spec={"type": "Generic", "params": {}},
        task_binding_spec=_task_binding_spec(),
        cfg=_cfg(),
    )
    initial_rollout = rollout_graph(
        compiled.graph,
        compiled,
        key=jax.random.PRNGKey(0),
    )
    initial_output = initial_rollout["outputs"]["output"]
    initial_loss = jnp.mean(jnp.square(initial_output))
    events: list[dict] = []

    result = run_training_graph(
        compiled,
        job_id="test-job",
        total_batches=5,
        cfg=_cfg(snapshot_interval=5),
        stop_event=threading.Event(),
        emit=events.append,
    )

    assert result.checkpoint_path is not None
    assert result.final_loss < float(initial_loss)
    assert any(event["type"] == "training_progress" for event in events)
    trajectory = next(event for event in events if event["type"] == "training_trajectory")
    assert trajectory["trajectory"]["outputs"]["output"]
    assert set(result.retained_observables) == {"observables", "outputs", "task_data"}
    assert "graph_output:output" in result.retained_observables["outputs"]
    assert "task_data:model_input" in result.retained_observables["task_data"]
    assert "task_data:inputs.model" in result.retained_observables["task_data"]


def test_compile_training_run_fails_unsupported_display_only_component() -> None:
    graph_spec = _linear_graph_spec(component_type="MomentArmProjection")

    with pytest.raises(ValueError, match="unsupported executable component"):
        compile_training_run(
            graph_spec=graph_spec,
            training_spec=_training_spec(),
            task_spec={"type": "Generic", "params": {}},
            task_binding_spec=_task_binding_spec(),
            cfg=_cfg(),
        )


def test_compile_training_run_rejects_task_binding_to_occupied_port() -> None:
    graph_spec = _linear_graph_spec()
    graph_spec["nodes"]["source"] = {
        "type": "Constant",
        "params": {"value": [1.0]},
        "input_ports": [],
        "output_ports": ["output"],
    }
    graph_spec["wires"] = [
        {
            "source_node": "source",
            "source_port": "output",
            "target_node": "readout",
            "target_port": "input",
        }
    ]

    with pytest.raises(ValueError, match="task_binding_target_occupied"):
        compile_training_run(
            graph_spec=graph_spec,
            training_spec=_training_spec(),
            task_spec={"type": "Generic", "params": {}},
            task_binding_spec=_task_binding_spec(),
            cfg=_cfg(),
        )
def test_worker_infers_channel_prototype_from_task_binding_shape() -> None:
    graph_spec = GraphSpec(
        nodes={
            "delay": ComponentSpec(
                type="Channel",
                params={"delay": 2, "add_noise": False},
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        output_ports=["output"],
        output_bindings={"output": ("delay", "output")},
    ).model_dump(mode="json", exclude_none=True)
    task_binding_spec = {
        "schema_version": "feedbax.studio.task_bindings.v2",
        "exposed_data": [
            {
                "id": "model_input",
                "label": "Model input",
                "kind": "signal",
                "role": "model_input",
                "path": "inputs.model",
                "bindable": True,
                "expected_shape": ["time", 3],
                "value_spec": {
                    "mode": "constant",
                    "value": [1.0, 2.0, 3.0],
                    "dtype": "float32",
                    "shape": ["time", 3],
                },
                "metadata": {},
            }
        ],
        "bindings": [
            {
                "id": "task:model_input->delay:input",
                "source_data_id": "model_input",
                "target_node_id": "delay",
                "target_port": "input",
                "role": "model_input",
                "metadata": {},
            }
        ],
        "metadata": {},
    }

    compiled = compile_training_run(
        graph_spec=graph_spec,
        training_spec=_training_spec(),
        task_spec={"type": "Generic", "params": {}},
        task_binding_spec=task_binding_spec,
        cfg=_cfg(n_reach_steps=5),
    )
    rollout = rollout_graph(compiled.graph, compiled, key=jax.random.PRNGKey(4))

    assert compiled.graph.nodes["delay"].input_proto.shape == (3,)
    assert rollout["outputs"]["output"].shape == (5, 3)
    assert jnp.allclose(rollout["outputs"]["output"][:2], 0.0)


def test_compile_training_run_rejects_batch_size_larger_than_one() -> None:
    with pytest.raises(ValueError, match="supports batch_size=1"):
        compile_training_run(
            graph_spec=_linear_graph_spec(),
            training_spec=_training_spec(batch_size=2),
            task_spec={"type": "Generic", "params": {}},
            task_binding_spec=_task_binding_spec(),
            cfg=_cfg(),
        )


@pytest.mark.parametrize("mode", ["stream", "window"])
def test_compile_training_run_rejects_unsupported_retention_modes(mode: str) -> None:
    graph_spec = _linear_graph_spec()
    retention = {"mode": mode}
    if mode == "window":
        retention["window_size"] = 2
    graph_spec["retained_observables"] = [
        {"selector": "port:readout.output", "retention": retention}
    ]

    with pytest.raises(ValueError, match="not supported by the current graph worker"):
        compile_training_run(
            graph_spec=graph_spec,
            training_spec=_training_spec(),
            task_spec={"type": "Generic", "params": {}},
            task_binding_spec=_task_binding_spec(),
            cfg=_cfg(),
        )


def test_compile_training_run_rejects_window_retention_without_size() -> None:
    graph_spec = _linear_graph_spec()
    graph_spec["retained_observables"] = [
        {"selector": "graph_output:output", "retention": {"mode": "window"}}
    ]

    with pytest.raises(ValueError, match="positive window_size"):
        compile_training_run(
            graph_spec=graph_spec,
            training_spec=_training_spec(),
            task_spec={"type": "Generic", "params": {}},
            task_binding_spec=_task_binding_spec(),
            cfg=_cfg(),
        )


def test_compile_training_run_rejects_unsupported_task_data_value_spec_mode() -> None:
    task_binding_spec = deepcopy(_task_binding_spec())
    task_binding_spec["exposed_data"][0]["value_spec"] = {
        "mode": "reference",
        "reference": {"path": "inputs.model"},
    }

    with pytest.raises(ValueError, match="unsupported value_spec mode='reference'"):
        compile_training_run(
            graph_spec=_linear_graph_spec(),
            training_spec=_training_spec(),
            task_spec={"type": "Generic", "params": {}},
            task_binding_spec=task_binding_spec,
            cfg=_cfg(),
        )


def test_compile_training_run_rejects_unsupported_task_data_function() -> None:
    task_binding_spec = deepcopy(_task_binding_spec())
    task_binding_spec["exposed_data"][0]["value_spec"] = {
        "mode": "function",
        "function_id": "unsupported_function",
    }

    with pytest.raises(
        ValueError,
        match="unsupported value_spec function_id='unsupported_function'",
    ):
        compile_training_run(
            graph_spec=_linear_graph_spec(),
            training_spec=_training_spec(),
            task_spec={"type": "Generic", "params": {}},
            task_binding_spec=task_binding_spec,
            cfg=_cfg(),
        )


def test_compile_training_run_lowers_segment_aggregation_with_task_timeline() -> None:
    training = _training_spec(
        loss={
            "type": "TargetStateLoss",
            "label": "movement_segment",
            "selector": "graph_output:output",
            "target_value": [0.0],
            "time_agg": {"mode": "segment", "segment_name": "movement"},
        }
    )
    task = {
        "type": "DelayedReaches",
        "params": {"n_steps": 4},
        "timeline": {
            "schema_version": "feedbax.studio.task_timeline.v1",
            "epochs": [
                {
                    "id": "epoch:0",
                    "label": "hold",
                    "index": 0,
                    "length": {"mode": "constant", "value": {"steps": 1}, "metadata": {}},
                    "metadata": {},
                },
                {
                    "id": "epoch:1",
                    "label": "movement",
                    "index": 1,
                    "length": {
                        "mode": "constant",
                        "value": None,
                        "metadata": {"inferred_from_remaining_steps": True},
                    },
                    "metadata": {},
                },
            ],
            "metadata": {"n_steps": 4},
        },
    }

    compiled = compile_training_run(
        graph_spec=_linear_graph_spec(),
        training_spec=training,
        task_spec=task,
        task_binding_spec=_task_binding_spec(),
        cfg=_cfg(n_reach_steps=4),
    )

    assert compiled.loss_terms[0].metadata["time_mask"]["epoch_ids"] == ["epoch:1"]


def test_compile_training_run_allows_absent_optional_task_data_value_spec_default() -> None:
    task_binding_spec = deepcopy(_task_binding_spec())
    task_binding_spec["exposed_data"][0].pop("value_spec")

    compiled = compile_training_run(
        graph_spec=_linear_graph_spec(),
        training_spec=_training_spec(),
        task_spec={"type": "Generic", "params": {}},
        task_binding_spec=task_binding_spec,
        cfg=_cfg(),
    )

    assert jnp.allclose(compiled.task_data["model_input"], 0.0)
