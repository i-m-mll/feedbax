from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import threading

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from feedbax.component_registry import ComponentRegistry, register_cde_templates
from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.graphs.templates import network_template_graph
from feedbax.contracts.graphs.normalization import normalize_task_binding_spec_for_studio_authoring
from feedbax.contracts.graph import (
    ComponentSpec,
    GraphSpec,
    ParameterConstraintSpec,
    StudioTaskBindingSpec,
)
from feedbax.web.worker.execution import (
    _build_optimizer,
    _derive_trainable_nodes,
    compile_training_run,
    rollout_graph,
    run_training_graph,
)


def _linear_graph_spec(component_type: str = "Linear", output_size: int = 1) -> dict:
    return GraphSpec(
        nodes={
            "readout": ComponentSpec(
                type=component_type,
                params={
                    "input_size": 1,
                    "output_size": output_size,
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


def _acausal_training_graph_spec() -> dict:
    interior = AcausalGraphSpec(
        physical_domain="translational",
        nodes={
            "wall": ComponentSpec(type="Ground"),
            "mass": ComponentSpec(type="Mass", params={"mass": 1.0}),
            "spring": ComponentSpec(type="LinearSpring", params={"stiffness": 10.0}),
            "damper": ComponentSpec(type="LinearDamper", params={"damping": 0.5}),
            "act": ComponentSpec(
                type="ActuationInput",
                params={"port_name": "u", "source_kind": "force"},
            ),
            "sense": ComponentSpec(
                type="SensorOutput",
                params={"port_name": "output", "quantity": "position"},
            ),
        },
        connections=[
            {"a": ("wall", "flange"), "b": ("spring", "flange_a")},
            {"a": ("wall", "flange"), "b": ("damper", "flange_a")},
            {"a": ("spring", "flange_b"), "b": ("mass", "flange")},
            {"a": ("damper", "flange_b"), "b": ("mass", "flange")},
            {"a": ("act", "flange"), "b": ("mass", "flange")},
            {"a": ("sense", "flange"), "b": ("mass", "flange")},
        ],
        solver={"solver_type": "euler", "dt": 0.001},
    )
    return GraphSpec(
        nodes={
            "plant": ComponentSpec(
                type="AcausalSystem",
                input_ports=["u"],
                output_ports=["output"],
            )
        },
        output_ports=["output"],
        output_bindings={"output": ("plant", "output")},
        subgraphs={"plant": interior},
    ).model_dump(mode="json", exclude_none=True)


def _acausal_task_binding_spec() -> dict:
    spec = _task_binding_spec()
    spec["bindings"][0] = {
        **spec["bindings"][0],
        "id": "task:model_input->plant:u",
        "target_node_id": "plant",
        "target_port": "u",
    }
    return spec


def _task_binding_spec() -> dict:
    return {
        "schema_version": "feedbax.spec.studio.task_bindings.v2",
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
        "schema_version": "feedbax.spec.studio.task_bindings.v2",
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
                "id": "task:model_input->input_mux:in_0",
                "source_data_id": "model_input",
                "target_node_id": "input_mux",
                "target_port": "in_0",
                "role": "model_input",
                "metadata": {},
            },
            {
                "id": "task:feedback->input_mux:in_1",
                "source_data_id": "feedback",
                "target_node_id": "input_mux",
                "target_port": "in_1",
                "role": "model_input",
                "metadata": {},
            },
        ],
        "metadata": {},
    }


def _cde_task_binding_spec() -> dict:
    exposed_data = []
    bindings = []
    target_nodes = {
        "obs": "obs_in",
        "obs_prev": "obs_prev_in",
        "h_prev": "h_prev_in",
    }
    values = {
        "obs": [0.2],
        "obs_prev": [0.1],
        "h_prev": [0.3],
    }
    for port, target_node in target_nodes.items():
        exposed_data.append(
            {
                "id": port,
                "label": port,
                "kind": "signal",
                "role": "model_input",
                "path": f"inputs.{port}",
                "bindable": True,
                "expected_shape": ["time", 1],
                "value_spec": {
                    "mode": "constant",
                    "value": values[port],
                    "dtype": "float32",
                    "shape": ["time", 1],
                },
                "metadata": {},
            }
        )
        bindings.append(
            {
                "id": f"task:{port}->{target_node}:input",
                "source_data_id": port,
                "target_node_id": target_node,
                "target_port": "input",
                "role": "model_input",
                "metadata": {},
            }
        )
    return {
        "schema_version": "feedbax.spec.studio.task_bindings.v2",
        "exposed_data": exposed_data,
        "bindings": bindings,
        "metadata": {},
    }


def _mux_graph_spec() -> dict:
    return GraphSpec(
        nodes={
            "mux": ComponentSpec(
                type="Mux",
                params={"n_inputs": 2},
                input_ports=["in_0", "in_1"],
                output_ports=["output"],
            )
        },
        output_ports=["output"],
        output_bindings={"output": ("mux", "output")},
    ).model_dump(mode="json", exclude_none=True)


def _mux_task_binding_spec() -> dict:
    return {
        "schema_version": "feedbax.spec.studio.task_bindings.v2",
        "exposed_data": [
            {
                "id": "position",
                "label": "Position",
                "kind": "signal",
                "role": "model_input",
                "path": "inputs.position",
                "bindable": True,
                "expected_shape": ["time", 2],
                "value_spec": {
                    "mode": "constant",
                    "value": [1.0, 2.0],
                    "dtype": "float32",
                    "shape": ["time", 2],
                },
                "metadata": {},
            },
            {
                "id": "cue",
                "label": "Cue",
                "kind": "signal",
                "role": "model_input",
                "path": "inputs.cue",
                "bindable": True,
                "expected_shape": ["time", 1],
                "value_spec": {
                    "mode": "constant",
                    "value": [0.5],
                    "dtype": "float32",
                    "shape": ["time", 1],
                },
                "metadata": {},
            },
        ],
        "bindings": [
            {
                "id": "task:position->mux:in_0",
                "source_data_id": "position",
                "target_node_id": "mux",
                "target_port": "in_0",
                "role": "model_input",
                "metadata": {},
            },
            {
                "id": "task:cue->mux:in_1",
                "source_data_id": "cue",
                "target_node_id": "mux",
                "target_port": "in_1",
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


def test_build_optimizer_omits_clip_when_grad_clip_is_none() -> None:
    params = {"weight": jnp.array([0.0])}

    clipped_state = _build_optimizer(0.1, 1.0).init(params)
    unclipped_state = _build_optimizer(0.1, None).init(params)

    assert len(clipped_state) == 2
    assert len(unclipped_state) == 1


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


def test_compile_training_run_dry_runs_acausal_graph_node() -> None:
    compiled = compile_training_run(
        graph_spec=_acausal_training_graph_spec(),
        training_spec=_training_spec(n_batches=1),
        task_spec={"type": "Generic", "params": {}},
        task_binding_spec=_acausal_task_binding_spec(),
        cfg=_cfg(n_reach_steps=3),
    )

    assert compiled.metadata["execution"] == "generic_graph"
    assert compiled.task_inputs[0].target_node == "plant"
    assert compiled.graph.nodes["plant"].input_ports == ("u",)
    assert compiled.graph.nodes["plant"].output_ports == ("output",)


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

    assert compiled.trainable_nodes == ("cell", "readout")
    assert all(
        eqx.is_inexact_array(leaf)
        for leaf in jax.tree_util.tree_leaves(trainable)
        if leaf is not None
    )


def test_trainable_nodes_come_from_registry_metadata_and_explicit_overrides() -> None:
    graph_spec = GraphSpec(
        nodes={
            "default_readout": ComponentSpec(
                type="Linear",
                params={"input_size": 1, "output_size": 1},
                input_ports=["input"],
                output_ports=["output"],
            ),
            "disabled_readout": ComponentSpec(
                type="Linear",
                params={"input_size": 1, "output_size": 1, "trainable": False},
                input_ports=["input"],
                output_ports=["output"],
            ),
            "explicit_gain": ComponentSpec(
                type="Gain",
                params={"gain": 1.0, "trainable": True},
                input_ports=["input"],
                output_ports=["output"],
            ),
        }
    )

    assert _derive_trainable_nodes(graph_spec) == ("default_readout", "explicit_gain")


def test_default_trainable_nodes_include_neural_and_executable_template_components() -> None:
    graph_spec = GraphSpec(
        nodes={
            "linear": ComponentSpec(type="Linear", params={}),
            "mlp": ComponentSpec(type="MLP", params={}),
            "gru": ComponentSpec(type="GRU", params={}),
            "lstm": ComponentSpec(type="LSTM", params={}),
            "recurrent_controller": ComponentSpec(type="Recurrent Controller", params={}),
            "simple_feedback_loop": ComponentSpec(type="Simple Feedback Loop", params={}),
            "gain": ComponentSpec(type="Gain", params={}),
        }
    )

    assert _derive_trainable_nodes(graph_spec) == (
        "linear",
        "mlp",
        "gru",
        "lstm",
        "recurrent_controller",
        "simple_feedback_loop",
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


def test_compile_training_run_dry_runs_cde_templates() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    registry.register_template_pack(register_cde_templates, provenance="test-cde-pack")
    training_spec = _training_spec(
        loss={
            "type": "TargetStateLoss",
            "label": "action_zero",
            "selector": "graph_output:action",
            "target_value": [0.0],
            "weight": 1.0,
            "norm": "squared_l2",
        }
    )

    for template_name in ("CDE Standard", "CDE + Decay", "CDE + Anti-NF", "CDE Hybrid v9b"):
        meta = registry.get(template_name)
        assert meta is not None
        compiled = compile_training_run(
            graph_spec=meta.template_graph.model_dump(mode="json", exclude_none=True),
            training_spec=training_spec,
            task_spec={"type": "Generic", "params": {}},
            task_binding_spec=_cde_task_binding_spec(),
            cfg=_cfg(n_reach_steps=2),
        )

        assert compiled.metadata["execution"] == "generic_graph"


def test_run_training_graph_trains_tiny_full_graph(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path / "runs"))
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
    assert result.manifest_path is not None
    assert Path(result.manifest_path).is_file()
    assert result.manifest_payload is not None
    assert result.manifest_payload["kind"] == "TrainingRunManifest"
    assert result.manifest_payload["checkpoint_custody"]
    assert result.final_loss < float(initial_loss)
    assert any(event["type"] == "training_progress" for event in events)
    trajectory = next(event for event in events if event["type"] == "training_trajectory")
    assert trajectory["trajectory"]["schema_id"] == "feedbax.event.studio.training_trajectory"
    assert trajectory["trajectory"]["schema_version"] == (
        "feedbax.event.studio.training_trajectory.v1"
    )
    assert trajectory["trajectory"]["fidelity"] == "lower_fidelity_live_snapshot"
    assert trajectory["trajectory"]["time"]["length"] == compiled.n_steps
    assert trajectory["trajectory"]["tracks"] == {}
    assert "effector" not in trajectory["trajectory"]
    assert "target" not in trajectory["trajectory"]
    assert "t" not in trajectory["trajectory"]
    assert trajectory["trajectory"]["outputs"]["output"]
    assert set(result.retained_observables) == {"observables", "outputs", "task_data"}
    assert "graph_output:output" in result.retained_observables["outputs"]
    assert "task_data:model_input" in result.retained_observables["task_data"]
    assert "task_data:inputs.model" in result.retained_observables["task_data"]


def test_worker_checkpoint_cleanup_removes_managed_tempdir(tmp_path: Path) -> None:
    from feedbax.web.worker.app import _cleanup_checkpoint_path

    checkpoint_dir = tmp_path / "feedbax_ckpt_demo"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "job.eqx"
    checkpoint_path.write_bytes(b"checkpoint")

    _cleanup_checkpoint_path(str(checkpoint_path))

    assert not checkpoint_dir.exists()


def test_run_training_graph_emits_executor_progress_each_batch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path / "runs"))
    compiled = compile_training_run(
        graph_spec=_linear_graph_spec(),
        training_spec=_training_spec(),
        task_spec={"type": "Generic", "params": {}},
        task_binding_spec=_task_binding_spec(),
        cfg=_cfg(),
    )
    events: list[dict] = []

    run_training_graph(
        compiled,
        job_id="test-job",
        total_batches=5,
        cfg=_cfg(snapshot_interval=3),
        stop_event=threading.Event(),
        emit=events.append,
    )

    progress_batches = [event["batch"] for event in events if event["type"] == "training_progress"]
    log_batches = [event["batch"] for event in events if event["type"] == "training_log"]
    trajectory_batches = [
        event["batch"] for event in events if event["type"] == "training_trajectory"
    ]

    assert progress_batches == [1, 2, 3, 4, 5]
    assert log_batches == [1, 2, 3, 4, 5]
    assert trajectory_batches == [3, 5]


def test_run_training_graph_emits_selector_keyed_live_trajectory_tracks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path / "runs"))
    compiled = compile_training_run(
        graph_spec=_linear_graph_spec(output_size=2),
        training_spec=_training_spec(),
        task_spec={"type": "Generic", "params": {}},
        task_binding_spec=_task_binding_spec(),
        cfg=_cfg(snapshot_interval=5),
    )
    events: list[dict] = []

    run_training_graph(
        compiled,
        job_id="test-job",
        total_batches=5,
        cfg=_cfg(snapshot_interval=5),
        stop_event=threading.Event(),
        emit=events.append,
    )

    trajectory = next(event for event in events if event["type"] == "training_trajectory")
    payload = trajectory["trajectory"]
    track = payload["tracks"]["graph_output:output"]

    assert payload["schema_id"] == "feedbax.event.studio.training_trajectory"
    assert payload["time"]["length"] == len(track["samples"])
    assert track["selector"]["compact"] == "graph_output:output"
    assert track["selector"]["role"] == "observed"
    assert track["dim"] == 2
    assert all(len(sample) == 2 for sample in track["samples"])


def test_run_training_graph_stopped_run_returns_latest_batch_loss(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path / "runs"))

    class StopAfterTwoChecks:
        def __init__(self) -> None:
            self.calls = 0

        def is_set(self) -> bool:
            self.calls += 1
            return self.calls > 2

    compiled = compile_training_run(
        graph_spec=_linear_graph_spec(),
        training_spec=_training_spec(),
        task_spec={"type": "Generic", "params": {}},
        task_binding_spec=_task_binding_spec(),
        cfg=_cfg(),
    )
    events: list[dict] = []

    result = run_training_graph(
        compiled,
        job_id="test-job",
        total_batches=5,
        cfg=_cfg(snapshot_interval=5),
        stop_event=StopAfterTwoChecks(),
        emit=events.append,
    )

    progress_losses = [event["loss"] for event in events if event["type"] == "training_progress"]
    assert progress_losses
    assert result.final_loss == pytest.approx(progress_losses[-1])
    assert result.final_batch == len(progress_losses)


def test_run_training_graph_projects_parameter_constraints_after_update(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path / "runs"))
    graph_spec = GraphSpec.model_validate(_linear_graph_spec())
    graph_spec.parameter_constraints = [
        ParameterConstraintSpec(node="readout", role="weight", mask=[[0]], value=0.0)
    ]
    compiled = compile_training_run(
        graph_spec=graph_spec,
        training_spec=_training_spec(),
        task_spec={"type": "Generic", "params": {}},
        task_binding_spec=_task_binding_spec(),
        cfg=_cfg(),
    )

    assert compiled.graph.nodes["readout"].layer.weight[0, 0] == 0.0

    run_training_graph(
        compiled,
        job_id="test-job",
        total_batches=3,
        cfg=_cfg(snapshot_interval=3),
        stop_event=threading.Event(),
        emit=lambda event: None,
    )

    assert compiled.graph.nodes["readout"].layer.weight[0, 0] == 0.0


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


def test_compile_training_run_rejects_network_without_subgraph_during_validation() -> None:
    graph_spec = GraphSpec(
        nodes={
            "network": ComponentSpec(
                type="Network",
                params={"input_size": 1, "hidden_size": 3, "out_size": 1},
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        output_ports=["output"],
        output_bindings={"output": ("network", "output")},
    ).model_dump(mode="json", exclude_none=True)

    with pytest.raises(ValueError, match="missing_subgraph"):
        compile_training_run(
            graph_spec=graph_spec,
            training_spec=_training_spec(),
            task_spec={"type": "Generic", "params": {}},
            task_binding_spec={
                **_task_binding_spec(),
                "bindings": [
                    {
                        **_task_binding_spec()["bindings"][0],
                        "id": "task:model_input->network:input",
                        "target_node_id": "network",
                        "target_port": "input",
                    }
                ],
            },
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
        "schema_version": "feedbax.spec.studio.task_bindings.v2",
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


def test_worker_materializes_task_binding_fed_mux_prototypes_after_normalization() -> None:
    graph_spec = GraphSpec.model_validate(_mux_graph_spec())
    task_binding_spec = normalize_task_binding_spec_for_studio_authoring(
        StudioTaskBindingSpec.model_validate(_mux_task_binding_spec()),
        graph_spec,
    )

    compiled = compile_training_run(
        graph_spec=graph_spec,
        training_spec=_training_spec(),
        task_spec={"type": "Generic", "params": {}},
        task_binding_spec=task_binding_spec,
        cfg=_cfg(n_reach_steps=5),
    )
    rollout = rollout_graph(compiled.graph, compiled, key=jax.random.PRNGKey(5))

    assert compiled.graph.nodes["mux"].input_ports == ("in_0", "in_1")
    assert rollout["outputs"]["output"].shape == (5, 3)
    assert jnp.allclose(rollout["outputs"]["output"][0], jnp.array([1.0, 2.0, 0.5]))


def test_worker_rejects_degenerate_single_input_mux_before_materialization() -> None:
    task_binding_spec = _mux_task_binding_spec()
    task_binding_spec["exposed_data"] = task_binding_spec["exposed_data"][:1]
    task_binding_spec["bindings"] = task_binding_spec["bindings"][:1]

    with pytest.raises(ValueError, match=r"Mux 'mux' needs at least two connected inputs"):
        compile_training_run(
            graph_spec=_mux_graph_spec(),
            training_spec=_training_spec(),
            task_spec={"type": "Generic", "params": {}},
            task_binding_spec=task_binding_spec,
            cfg=_cfg(),
        )


def test_compile_training_run_rejects_batch_size_larger_than_one() -> None:
    with pytest.raises(ValueError, match="supports batch_size=1") as excinfo:
        compile_training_run(
            graph_spec=_linear_graph_spec(),
            training_spec=_training_spec(batch_size=2),
            task_spec={"type": "Generic", "params": {}},
            task_binding_spec=_task_binding_spec(),
            cfg=_cfg(),
        )
    assert "got batch_size=2" in str(excinfo.value)


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
            "schema_version": "feedbax.spec.studio.task_timeline.v1",
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
