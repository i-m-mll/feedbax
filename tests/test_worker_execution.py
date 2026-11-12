from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import threading

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
import numpy as np
import optax
import pytest
from pydantic import TypeAdapter

import feedbax.web.worker.execution as worker_execution
import feedbax.web.worker.checkpoint as worker_checkpoint
from feedbax.compiler import GraphDocument, compile_graph
from feedbax.component_registry import ComponentRegistry
from feedbax.component_registry.cde_templates import cde_template_pack
from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.compiler.templates import network_template_graph, recurrent_node_output_initializer
from feedbax.compiler.serialization import graph_to_spec, prototypes_from_task_bindings
from feedbax.contracts.base import ParentRef
from feedbax.contracts.graph import (
    ComponentSpec,
    GraphSpec,
    ParameterConstraintSpec,
    STUDIO_TASK_TIMELINE_SCHEMA_VERSION,
    StudioTaskBindingSpec,
    WireSpec,
)
from feedbax.runtime._graph_key_schedule import derive_key
from feedbax.runtime.graph import init_state_from_component
from feedbax.web.worker.execution import (
    _build_optimizer,
    compile_training_run,
    rollout_graph,
    run_training_graph,
)
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.plugins.bootstrap import BootstrapState
from feedbax.training.checkpoint_custody import (
    authenticate_checkpoint_custody_ref,
    structural_abi_fingerprint,
)
from feedbax.web.worker.app import _Job, _require_worker_specs
from feedbax.web.worker.checkpoint import CheckpointCleanupError
from feedbax.web.worker.diagnostics import GraphCompilationError
from feedbax.contracts.studio_api import TrainingWebSocketEvent
from feedbax.contracts.training import TaskSpec, TrainingRunSpec
from feedbax.orchestration.events import run_event_from_legacy_worker_event
from feedbax.web.services.training_service import TrainingService


def _linear_graph_spec(component_type: str = "feedbax.Linear", output_size: int = 1) -> dict:
    return GraphSpec(
        nodes={
            "readout": ComponentSpec(
                type=component_type,
                params={
                    "input_size": 1,
                    "output_size": output_size,
                    "activation": "identity",
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
            "wall": ComponentSpec(type="feedbax.Ground"),
            "mass": ComponentSpec(type="feedbax.Mass", params={"mass": 1.0}),
            "spring": ComponentSpec(type="feedbax.LinearSpring", params={"stiffness": 10.0}),
            "damper": ComponentSpec(type="feedbax.LinearDamper", params={"damping": 0.5}),
            "act": ComponentSpec(
                type="feedbax.ActuationInput",
                params={"port_name": "u", "source_kind": "force"},
            ),
            "sense": ComponentSpec(
                type="feedbax.SensorOutput",
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
                type="feedbax.AcausalSystem",
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
        "schema_version": "feedbax.spec.studio.task_bindings.v3",
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
        "schema_version": "feedbax.spec.studio.task_bindings.v3",
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
        "schema_version": "feedbax.spec.studio.task_bindings.v3",
        "exposed_data": exposed_data,
        "bindings": bindings,
        "metadata": {},
    }


def _mux_graph_spec() -> dict:
    return GraphSpec(
        nodes={
            "mux": ComponentSpec(
                type="feedbax.Mux",
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
        "schema_version": "feedbax.spec.studio.task_bindings.v3",
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
        component_registry=ComponentRegistry(),
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


def _studio_worker_run_spec() -> TrainingRunSpec:
    from feedbax.contracts.studio_training import (
        STUDIO_GRAPH_METHOD,
        STUDIO_GRAPH_METHOD_PAYLOAD_ID,
        STUDIO_GRAPH_METHOD_PAYLOAD_VERSION,
        STUDIO_INDEPENDENT_KEY_SCHEDULE,
        StudioGraphMethodPayload,
    )
    from feedbax.contracts.worker import derive_consistency_predicate
    from feedbax.training.studio_executor import studio_graph_method_contract

    legacy = _training_spec()
    payload = StudioGraphMethodPayload(
        optimizer=legacy["optimizer"], target_total=2, key_schedule=STUDIO_INDEPENDENT_KEY_SCHEDULE
    )
    contract = studio_graph_method_contract(payload)
    return TrainingRunSpec(
        graph={"inline": _linear_graph_spec()},
        task={"type": "Generic", "params": {}},
        task_binding_spec=_task_binding_spec(),
        training_config={"n_batches": 2, "batch_size": 1, "n_reach_steps": 4},
        objective={"loss": legacy["loss"]},
        method_ref=STUDIO_GRAPH_METHOD,
        method_payload={
            "schema_id": STUDIO_GRAPH_METHOD_PAYLOAD_ID,
            "schema_version": STUDIO_GRAPH_METHOD_PAYLOAD_VERSION,
            "payload": payload.model_dump(mode="json"),
        },
        worker_execution={
            "method_contract": contract,
            "effective_phase": {
                "method_ref": contract.method_ref, "axes": contract.axes,
                "state_slots": contract.state_slots, "phase_program": contract.phase_program,
                "consistency_predicate": derive_consistency_predicate(contract.phase_program),
            },
        },
        checkpoint_progress={"checkpoint_interval": 1, "progress_interval": 1},
    )


def test_studio_worker_cli_accepts_serialized_graph(tmp_path: Path) -> None:
    from feedbax.__main__ import main
    from feedbax.training.checkpoint_custody import authenticate_published_checkpoint_custody

    spec_path = tmp_path / "spec.json"
    spec_path.write_text(_studio_worker_run_spec().model_dump_json(exclude_none=True))
    checkpoint_root = tmp_path / "checkpoints"
    assert main([
        "execute-training-run-spec", str(spec_path),
        "--manifest-root", str(tmp_path / "runs"),
        "--checkpoint-root", str(checkpoint_root),
        "--run-id", "studio-cli",
        "--update-budget", "1", "--no-progress",
    ]) == 0
    checkpoint = authenticate_published_checkpoint_custody(checkpoint_root)
    assert checkpoint.manifest.completed_training_batches == 1


def test_builtin_studio_graph_resumes_the_numerical_checkpoint(tmp_path: Path) -> None:
    from feedbax.contracts.checkpoints import CheckpointContinuationRequest
    from feedbax.contracts.training import resolve_training_run_spec
    from feedbax.training.checkpoint_custody import authenticate_published_checkpoint_custody
    from feedbax.training.executor import execute_training_run_spec
    from feedbax.training.preparation import ExecutionPreparationRequest

    spec = _studio_worker_run_spec()
    registries = new_application_registry_bundle()

    def prepare(run_spec, *, resume=False):
        method = resolve_training_run_spec(run_spec, registries.training_programs)
        graph = compile_graph(
            GraphDocument(graph=run_spec.graph.inline), registries.components,
            input_prototypes=prototypes_from_task_bindings(run_spec.task_binding_spec),
        )
        return registries.execution_preparations.prepare(ExecutionPreparationRequest(
            run_spec=run_spec, method_payload=method.payload, method_contract=method.contract,
            effective_phase=method.effective_phase, executable_graph=graph, resume=resume,
        ))

    first = execute_training_run_spec(
        spec, run_id="studio-graph-first", preparation=prepare(spec),
        registry=registries.training_programs,
        manifest_root=tmp_path, checkpoint_root=tmp_path / "first",
        stop_after_barrier="after_train_batch",
    )
    assert first.end_completed_batches == 1
    source = authenticate_published_checkpoint_custody(tmp_path / "first")
    continuation = spec.model_copy(deep=True)
    continuation.training_config.n_batches = 1
    continuation.checkpoint_progress.continuation = CheckpointContinuationRequest(
        source_completed_batches=1, additional_batches=1,
    )
    prepared = prepare(continuation, resume=True)
    expected = prepared.kernel_context["studio_graph_update"](first.final_slots)
    resumed = execute_training_run_spec(
        continuation, run_id="studio-graph-resumed", preparation=prepared,
        registry=registries.training_programs,
        manifest_root=tmp_path, checkpoint_root=tmp_path / "resumed",
        checkpoint_source_root=tmp_path / "first", resume=True,
    )
    assert resumed.status == "completed"
    assert (resumed.start_completed_batches, resumed.end_completed_batches) == (1, 2)
    terminal = authenticate_published_checkpoint_custody(tmp_path / "resumed")
    assert terminal.manifest.parent_lineage[0].transaction_id == source.manifest.transaction_id
    for slot in ("model", "optimizer", "prng", "batch_counter"):
        actual = resumed.final_slots[slot]
        assert jt.structure(actual) == jt.structure(expected[slot])
        for actual_leaf, expected_leaf in zip(jt.leaves(actual), jt.leaves(expected[slot]), strict=True):
            np.testing.assert_array_equal(actual_leaf, expected_leaf, err_msg=slot)


def test_compile_training_run_dry_runs_acausal_graph_node() -> None:
    compiled = compile_training_run(
        component_registry=ComponentRegistry(),
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
        component_registry=ComponentRegistry(),
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


def test_rollout_graph_threads_network_template_recurrence() -> None:
    graph_spec = network_template_graph(
        {"input_size": 2, "hidden_size": 3, "out_size": 1}
    ).model_dump(mode="json", exclude_none=True)

    compiled = compile_training_run(
        component_registry=ComponentRegistry(),
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


def test_worker_rollout_matches_runtime_for_task_seeded_noisy_cycle() -> None:
    # A noisy encoder seeds the accumulator once. A noisy delay reads its
    # output on every frame and gives the rollout nontrivial state to retain.
    graph_spec = GraphSpec(
        nodes={
            "encoder": ComponentSpec(
                type="feedbax.Channel",
                params={"delay": 0, "noise_std": 0.25},
                input_ports=["input"],
                output_ports=["output"],
            ),
            "total": ComponentSpec(
                type="feedbax.Sum",
                input_ports=["a", "b"],
                output_ports=["output"],
            ),
            "delay": ComponentSpec(
                type="feedbax.Channel",
                params={"delay": 1, "noise_std": 0.25},
                input_ports=["input"],
                output_ports=["output"],
            ),
        },
        wires=[
            WireSpec(
                source_node="total",
                source_port="output",
                target_node="total",
                target_port="a",
                temporality="recurrent",
                recurrent_initializer=recurrent_node_output_initializer(
                    "encoder", "output", state_slot="a"
                ),
            ),
            WireSpec(
                source_node="total",
                source_port="output",
                target_node="delay",
                target_port="input",
            ),
        ],
        input_ports=["seed", "drive"],
        input_bindings={"seed": ("encoder", "input"), "drive": ("total", "b")},
        output_ports=["output"],
        output_bindings={"output": ("delay", "output")},
        retained_observables=[
            {"selector": "recurrent_carry:total.output->total.a"},
            {"selector": "state_path:delay.output"},
        ],
    )
    task_binding_spec = _task_binding_spec()
    seed = [7.0]
    drive = [[1.0], [3.0], [-2.0], [4.0]]
    task_binding_spec["exposed_data"][0]["value_spec"]["value"] = drive
    task_binding_spec["bindings"][0].update(
        id="task:model_input->total:b", target_node_id="total", target_port="b"
    )
    task_binding_spec["exposed_data"].append(
        {
            "id": "seed",
            "label": "Seed",
            "kind": "signal",
            "role": "model_input",
            "path": "inputs.seed",
            "bindable": True,
            "expected_shape": [1],
            "value_spec": {"mode": "constant", "value": seed, "shape": [1]},
        }
    )
    task_binding_spec["bindings"].append(
        {
            "id": "task:seed->encoder:input",
            "source_data_id": "seed",
            "target_node_id": "encoder",
            "target_port": "input",
            "role": "model_input",
        }
    )
    compiled = compile_training_run(
        component_registry=ComponentRegistry(),
        graph_spec=graph_spec,
        training_spec=_training_spec(),
        task_spec={"type": "Generic", "params": {}},
        task_binding_spec=task_binding_spec,
        cfg=_cfg(n_reach_steps=len(drive)),
    )
    graph = compiled.graph
    key = jax.random.PRNGKey(17)
    rollout = rollout_graph(graph, compiled, key=key)
    graph_key = derive_key(key, "studio-trial", "graph")
    outputs, final_state, history = graph(
        {"seed": jnp.array(seed), "drive": jnp.array(drive)},
        init_state_from_component(graph),
        key=graph_key,
        n_steps=len(drive),
        return_state_history=True,
    )

    assert jnp.array_equal(rollout["task_data"]["model_input"], jnp.array(drive))
    assert jnp.array_equal(rollout["outputs"]["output"], outputs["output"])
    assert eqx.tree_equal(rollout["final_state"], final_state)
    assert jnp.array_equal(graph.state_view(final_state).encoder.output, jnp.zeros(1))
    assert jnp.array_equal(rollout["trace"]["state_path:delay.output"], history.delay.output[1:])
    carry_selector = "recurrent_carry:total.output->total.a"
    assert jnp.array_equal(rollout["trace"][carry_selector], history.delay.queue[0][1:])

    other_rollout = rollout_graph(graph, compiled, key=jax.random.PRNGKey(18))
    assert not jnp.array_equal(
        rollout["trace"][carry_selector][0], other_rollout["trace"][carry_selector][0]
    )


def test_compile_training_run_dry_runs_cde_templates() -> None:
    registry = ComponentRegistry()
    registry.ingest(cde_template_pack(), provenance="test-cde-pack")
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

    for template_name in ("CDEStandard", "CDEDecay", "CDEAntiNF", "CDEHybridV9b"):
        meta = registry.get(f"feedbax.{template_name}")
        assert meta is not None
        compiled = compile_training_run(
            component_registry=ComponentRegistry(),
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
        component_registry=ComponentRegistry(),
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


def test_worker_checkpoint_uses_provider_owned_filename(tmp_path: Path, monkeypatch) -> None:
    checkpoint_dir = tmp_path / "feedbax_ckpt_test"
    checkpoint_dir.mkdir()
    written_paths: list[str] = []

    monkeypatch.setattr(
        worker_execution.tempfile,
        "mkdtemp",
        lambda **_kwargs: str(checkpoint_dir),
    )
    monkeypatch.setattr(worker_execution.jax, "block_until_ready", lambda graph: graph)
    monkeypatch.setattr(
        worker_execution.eqx,
        "tree_serialise_leaves",
        lambda path, _graph: written_paths.append(path),
    )

    checkpoint_path = worker_execution._write_checkpoint(object())

    assert checkpoint_path == str(checkpoint_dir / "checkpoint.eqx")
    assert written_paths == [checkpoint_path]


@pytest.mark.parametrize("failure_stage", ["block_until_ready", "serialize"])
def test_worker_checkpoint_removes_partial_directory_on_write_failure(
    failure_stage: str, tmp_path: Path, monkeypatch
) -> None:
    checkpoint_dir = tmp_path / "feedbax_ckpt_partial"
    checkpoint_dir.mkdir()

    monkeypatch.setattr(
        worker_execution.tempfile,
        "mkdtemp",
        lambda **_kwargs: str(checkpoint_dir),
    )

    if failure_stage == "block_until_ready":
        monkeypatch.setattr(
            worker_execution.jax,
            "block_until_ready",
            lambda _graph: (_ for _ in ()).throw(RuntimeError("block failed")),
        )
    else:
        monkeypatch.setattr(worker_execution.jax, "block_until_ready", lambda graph: graph)

        def fail_after_partial_write(path: str, _graph: object) -> None:
            Path(path).write_bytes(b"partial")
            raise RuntimeError("serialize failed")

        monkeypatch.setattr(
            worker_execution.eqx,
            "tree_serialise_leaves",
            fail_after_partial_write,
        )

    with pytest.raises(RuntimeError, match="failed"):
        worker_execution._write_checkpoint(object())

    assert not checkpoint_dir.exists()


def test_worker_partial_serialization_cleanup_failure_reports_residual_path(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint_dir = tmp_path / "feedbax_ckpt_partial_cleanup_failure"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "checkpoint.eqx"

    monkeypatch.setattr(
        worker_execution.tempfile,
        "mkdtemp",
        lambda **_kwargs: str(checkpoint_dir),
    )
    monkeypatch.setattr(worker_execution.jax, "block_until_ready", lambda graph: graph)

    def fail_after_partial_write(path: str, _graph: object) -> None:
        Path(path).write_bytes(b"partial")
        raise RuntimeError("serialize failed")

    monkeypatch.setattr(
        worker_execution.eqx,
        "tree_serialise_leaves",
        fail_after_partial_write,
    )
    monkeypatch.setattr(
        worker_checkpoint.shutil,
        "rmtree",
        lambda _path: (_ for _ in ()).throw(PermissionError("cleanup denied")),
    )

    with pytest.raises(CheckpointCleanupError, match="cleanup denied") as caught:
        worker_execution._write_checkpoint(object())

    assert caught.value.checkpoint_path == str(checkpoint_path)
    assert caught.value.context == "worker checkpoint serialization failed"
    assert checkpoint_path.read_bytes() == b"partial"


def test_worker_checkpoint_calls_use_distinct_directories(tmp_path: Path, monkeypatch) -> None:
    checkpoint_dirs = [
        tmp_path / "feedbax_ckpt_first",
        tmp_path / "feedbax_ckpt_second",
    ]
    pending_dirs = iter(checkpoint_dirs)

    def make_checkpoint_dir(**_kwargs) -> str:
        path = next(pending_dirs)
        path.mkdir()
        return str(path)

    monkeypatch.setattr(worker_execution.tempfile, "mkdtemp", make_checkpoint_dir)
    monkeypatch.setattr(worker_execution.jax, "block_until_ready", lambda graph: graph)
    monkeypatch.setattr(
        worker_execution.eqx,
        "tree_serialise_leaves",
        lambda path, _graph: Path(path).write_bytes(b"checkpoint"),
    )

    first = worker_execution._write_checkpoint(object())
    second = worker_execution._write_checkpoint(object())

    assert Path(first).parent != Path(second).parent
    assert Path(first).read_bytes() == b"checkpoint"
    assert Path(second).read_bytes() == b"checkpoint"


def test_run_training_graph_removes_checkpoint_after_post_serialization_failure(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path / "runs"))
    compiled = compile_training_run(
        component_registry=ComponentRegistry(),
        graph_spec=_linear_graph_spec(),
        training_spec=_training_spec(),
        task_spec={"type": "Generic", "params": {}},
        task_binding_spec=_task_binding_spec(),
        cfg=_cfg(),
    )
    checkpoint_dir = tmp_path / "feedbax_ckpt_post_serialization"

    def write_checkpoint(_graph: object) -> str:
        checkpoint_dir.mkdir()
        checkpoint_path = checkpoint_dir / "checkpoint.eqx"
        checkpoint_path.write_bytes(b"serialized")
        return str(checkpoint_path)

    monkeypatch.setattr(worker_execution, "_write_checkpoint", write_checkpoint)
    monkeypatch.setattr(
        worker_execution,
        "_retained_observables_payload",
        lambda _rollout: (_ for _ in ()).throw(RuntimeError("post-serialization failed")),
    )

    with pytest.raises(RuntimeError, match="post-serialization failed"):
        run_training_graph(
            compiled,
            job_id="test-job",
            total_batches=1,
            cfg=_cfg(snapshot_interval=1),
            stop_event=threading.Event(),
            emit=lambda _event: None,
        )

    assert not checkpoint_dir.exists()


def test_post_serialization_cleanup_failure_reports_residual_path(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path / "runs"))
    compiled = compile_training_run(
        component_registry=ComponentRegistry(),
        graph_spec=_linear_graph_spec(),
        training_spec=_training_spec(),
        task_spec={"type": "Generic", "params": {}},
        task_binding_spec=_task_binding_spec(),
        cfg=_cfg(),
    )
    checkpoint_dir = tmp_path / "feedbax_ckpt_post_cleanup_failure"
    checkpoint_path = checkpoint_dir / "checkpoint.eqx"

    def write_checkpoint(_graph: object) -> str:
        checkpoint_dir.mkdir()
        checkpoint_path.write_bytes(b"serialized")
        return str(checkpoint_path)

    monkeypatch.setattr(worker_execution, "_write_checkpoint", write_checkpoint)
    monkeypatch.setattr(
        worker_execution,
        "_retained_observables_payload",
        lambda _rollout: (_ for _ in ()).throw(RuntimeError("post-serialization failed")),
    )

    def deny_checkpoint_cleanup(_path: Path) -> None:
        raise PermissionError("cleanup denied")

    monkeypatch.setattr(
        worker_checkpoint, "shutil", SimpleNamespace(rmtree=deny_checkpoint_cleanup)
    )

    with pytest.raises(CheckpointCleanupError, match="cleanup denied") as caught:
        run_training_graph(
            compiled,
            job_id="test-job-cleanup-failure",
            total_batches=1,
            cfg=_cfg(snapshot_interval=1),
            stop_event=threading.Event(),
            emit=lambda _event: None,
        )

    assert caught.value.checkpoint_path == str(checkpoint_path)
    assert caught.value.context == "worker result finalization failed"
    assert checkpoint_path.read_bytes() == b"serialized"


@pytest.mark.parametrize(
    "fixture, expected_losses, expected_output",
    [
        ("linear", [0.4799930155, 0.2428681999, 0.0857427195], [-0.0948367119] * 4),
        ("constrained", [0.0715106800, 0.0280279387, 0.0050593941], [-0.0155360159] * 4),
        (
            "recurrent",
            [0.0006390046, 0.1656780392, 0.0214153193],
            [-0.0984299108, -0.1582289338, -0.1803956926, -0.1881809831],
        ),
    ],
)
def test_studio_update_reproduces_v4_seeded_training(fixture, expected_losses, expected_output):
    """Pin v4 seeded updates with reset templates excluded from optimizer ownership."""
    from feedbax.training.studio_executor import StudioUpdateKernel

    graph_spec = _linear_graph_spec()
    binding = _task_binding_spec()
    if fixture == "constrained":
        graph_spec["parameter_constraints"] = [
            {"node": "readout", "role": "weight", "mask": [[0]], "value": 0.0}
        ]
    elif fixture == "recurrent":
        graph_spec = network_template_graph(
            {"input_size": 2, "hidden_size": 3, "out_size": 1}
        ).model_dump(mode="json", exclude_none=True)
        binding = _network_task_binding_spec()
    compiled = compile_training_run(
        component_registry=ComponentRegistry(),
        graph_spec=graph_spec,
        training_spec=_training_spec(),
        task_spec={"type": "Generic", "params": {}},
        task_binding_spec=binding,
        cfg=_cfg(),
    )
    initial_graph = compiled.graph
    optimizer = _build_optimizer(0.1, 1.0)
    kernel = StudioUpdateKernel(
        compiled=compiled,
        optimizer=optimizer,
        rollout_fn=lambda graph, plan, key: rollout_graph(graph, plan, key=key),
        evaluate_loss_fn=worker_execution._evaluate_loss,
    )
    slots = {
        "model": compiled.graph,
        "optimizer": optimizer.init(eqx.partition(compiled.graph, compiled.trainable_filter)[0]),
        "prng": jax.random.PRNGKey(0),
        "batch_counter": jnp.array(0, dtype=jnp.int32),
    }
    for expected_loss in expected_losses:
        slots = kernel(slots)
        assert isinstance(slots["train_loss"], jax.Array)
        assert isinstance(slots["grad_norm"], jax.Array)
        assert float(slots["train_loss"]) == pytest.approx(expected_loss, rel=2e-5, abs=2e-6)
    output = rollout_graph(slots["model"], compiled, key=slots["prng"])["outputs"]["output"]
    assert output[:, 0].tolist() == pytest.approx(expected_output, rel=2e-5, abs=2e-6)
    assert int(slots["batch_counter"]) == 3
    assert compiled.graph is initial_graph
    with pytest.raises((TypeError, ValueError)):
        kernel({**slots, "batch_counter": jnp.zeros((2,), dtype=jnp.int32)})


def _checkpoint_steps(manifest_payload: dict, root: Path) -> list[int]:
    return [
        authenticate_checkpoint_custody_ref(
            ParentRef.model_validate(receipt), allowed_root=root
        ).manifest.completed_coordinate.program_step
        for receipt in manifest_payload["checkpoint_custody"]
    ]


def test_run_training_graph_emits_executor_progress_each_batch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path / "runs"))
    compiled = compile_training_run(
        component_registry=ComponentRegistry(),
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
        cfg=_cfg(snapshot_interval=3),
        stop_event=threading.Event(),
        emit=events.append,
    )

    progress_steps = [
        event["program_step"] for event in events if event["type"] == "training_progress"
    ]
    log_steps = [event["program_step"] for event in events if event["type"] == "training_log"]
    trajectory_steps = [
        event["program_step"] for event in events if event["type"] == "training_trajectory"
    ]

    assert progress_steps == [1, 2, 3, 4, 5]
    assert log_steps == [1, 2, 3, 4, 5]
    assert trajectory_steps == [3, 5]
    assert result.manifest_payload is not None
    assert _checkpoint_steps(
        result.manifest_payload, tmp_path / "runs" / "checkpoints" / "test-job"
    ) == [3, 5]
    service = TrainingService()
    validator = TypeAdapter(TrainingWebSocketEvent)
    for seq, event in enumerate(events):
        worker_event = run_event_from_legacy_worker_event(
            event, run_set_id="test-run", row_id="test-job", seq=seq
        )
        payload = service._normalize_training_event(
            "test-job", worker_event.model_dump(mode="json")
        )
        studio_event = validator.validate_python(payload)
        assert studio_event.worker_seq == seq
        assert studio_event.batch == event["batch"]
    assert [event["batch"] for event in events if event["type"] == "training_progress"] == [
        1, 2, 3, 4, 5
    ]
    assert result.final_batch == 5

    assert compiled.graph is result.graph
    restored = eqx.tree_deserialise_leaves(result.checkpoint_path, result.graph)
    for expected, actual in zip(jt.leaves(result.graph), jt.leaves(restored), strict=True):
        if eqx.is_array(expected):
            assert jnp.array_equal(expected, actual)


def test_run_training_graph_emits_selector_keyed_live_trajectory_tracks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path / "runs"))
    compiled = compile_training_run(
        component_registry=ComponentRegistry(),
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


@pytest.mark.parametrize(
    ("stop_at", "terminal_batch", "outcome"),
    [(0, 0, "cancelled"), (3, 3, "cancelled"), (5, 5, "completed")],
)
def test_run_training_graph_stop_preserves_committed_checkpoint(
    tmp_path: Path,
    monkeypatch,
    stop_at: int,
    terminal_batch: int,
    outcome: str,
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path / "runs"))

    compiled = compile_training_run(
        component_registry=ComponentRegistry(),
        graph_spec=_linear_graph_spec(),
        training_spec=_training_spec(),
        task_spec={"type": "Generic", "params": {}},
        task_binding_spec=_task_binding_spec(),
        cfg=_cfg(),
    )
    progress: list[dict] = []
    stop_event = threading.Event()
    if stop_at == 0:
        stop_event.set()

    def emit(event: dict) -> None:
        if event["type"] == "training_progress":
            progress.append(event)
            if event["batch"] == stop_at:
                stop_event.set()

    result = run_training_graph(
        compiled,
        job_id="test-job",
        total_batches=5,
        cfg=_cfg(snapshot_interval=5),
        stop_event=stop_event,
        emit=emit,
    )

    assert [event["batch"] for event in progress] == list(range(1, terminal_batch + 1))
    assert result.final_batch == result.final_program_step == terminal_batch
    if progress:
        assert result.final_loss == pytest.approx(progress[-1]["loss"])
    manifest = result.manifest_payload
    assert manifest is not None
    assert manifest["status"] == manifest["terminal_certification"]["termination_reason"] == outcome
    assert manifest["stopped"] is (outcome == "cancelled")
    assert manifest["completed_batches"] == terminal_batch
    assert _checkpoint_steps(manifest, tmp_path / "runs" / "checkpoints" / "test-job") == (
        [terminal_batch] if terminal_batch else []
    )


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
        component_registry=ComponentRegistry(),
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


@pytest.mark.parametrize(
    ("wrapper_type", "freeze_wrapper", "freeze_child"),
    [
        pytest.param("feedbax.Subgraph", False, False, id="generic-wrapper"),
        pytest.param("feedbax.Subgraph", True, False, id="frozen-wrapper"),
        pytest.param("feedbax.RecurrentController", False, True, id="named-wrapper"),
    ],
)
def test_nested_trainability_matches_flat_training(
    wrapper_type: str, freeze_wrapper: bool, freeze_child: bool, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path / "runs"))
    # Use SGD to make the one-step updates analytically checkable.
    monkeypatch.setattr(
        worker_execution,
        "_build_optimizer",
        lambda learning_rate, _grad_clip: optax.sgd(learning_rate),
    )

    def linear(**params) -> ComponentSpec:
        return ComponentSpec(
            type="feedbax.Linear",
            params={"input_size": 1, "output_size": 1, "use_bias": False, **params},
            input_ports=["input"],
            output_ports=["output"],
        )

    def wire(source: str, target: str) -> WireSpec:
        return WireSpec(
            source_node=source, source_port="output", target_node=target, target_port="input"
        )

    first = linear(trainable=False) if freeze_child else linear()
    second = linear(trainable=True) if freeze_wrapper else linear()
    flat_spec = GraphSpec(
        nodes={
            "first": linear(trainable=False) if freeze_wrapper else first,
            "second": linear(trainable=False) if freeze_wrapper else second,
            "readout": linear(),
        },
        wires=[wire("first", "second"), wire("second", "readout")],
        output_ports=["output"],
        output_bindings={"output": ("readout", "output")},
    )
    nested_spec = GraphSpec(
        nodes={
            "readout": linear(),
            "wrapper": ComponentSpec(
                type=wrapper_type,
                params={"trainable": False} if freeze_wrapper else {},
                input_ports=["input"],
                output_ports=["output"],
            ),
        },
        wires=[wire("wrapper", "readout")],
        output_ports=flat_spec.output_ports,
        output_bindings=flat_spec.output_bindings,
        subgraphs={
            "wrapper": GraphSpec(
                nodes={"first": first, "second": second},
                wires=[wire("first", "second")],
                input_ports=["input"],
                input_bindings={"input": ("first", "input")},
                output_ports=["output"],
                output_bindings={"output": ("second", "output")},
            )
        },
    )

    def task_bindings(target: str) -> dict:
        bindings = _task_binding_spec()
        bindings["bindings"][0].update(
            id=f"task:model_input->{target}:input",
            target_node_id=target,
        )
        return bindings

    registry = ComponentRegistry()
    nested_bindings = task_bindings("wrapper")
    authored = compile_graph(
        GraphDocument(graph=nested_spec),
        registry,
        input_prototypes=prototypes_from_task_bindings(nested_bindings),
    )
    serialized = graph_to_spec(authored.graph, registry)
    cfg = _cfg(grad_clip=None, snapshot_interval=1)
    nested_fingerprints = []
    for variant, spec, bindings in (
        ("flat", flat_spec, task_bindings("first")),
        ("nested", nested_spec, nested_bindings),
        ("serialized", serialized, nested_bindings),
    ):
        nested = variant != "flat"
        compiled = compile_training_run(
            component_registry=registry,
            graph_spec=spec,
            training_spec=_training_spec(
                optimizer={"type": "sgd", "params": {"learning_rate": cfg.learning_rate}},
                n_batches=1,
            ),
            task_spec={"type": "Generic", "params": {}},
            task_binding_spec=bindings,
            cfg=cfg,
        )

        prefix = "wrapper." if nested else ""
        expected_paths = {"readout"}
        if not freeze_wrapper:
            expected_paths.add(f"{prefix}second")
            if not freeze_child:
                expected_paths.add(f"{prefix}first")
        if nested:
            nested_fingerprints.append(
                structural_abi_fingerprint(compiled.graph).fingerprint_sha256
            )

        def weights(graph):
            interior = graph.nodes["wrapper"] if nested else graph
            return (
                interior.nodes["first"].layer.weight,
                interior.nodes["second"].layer.weight,
                graph.nodes["readout"].layer.weight,
            )

        key = jax.random.PRNGKey(0)
        # Match starting weights across depths and give each leaf a distinct update.
        compiled.graph = eqx.tree_at(
            weights, compiled.graph, tuple(jnp.array([[weight]]) for weight in (1.0, 2.0, 0.5))
        )
        initial = rollout_graph(compiled.graph, compiled, key=key)["outputs"]["output"]
        assert jnp.all(initial == 1.0)

        result = run_training_graph(
            compiled,
            job_id=f"trainability-{variant}",
            total_batches=1,
            cfg=cfg,
            stop_event=threading.Event(),
            emit=lambda event: None,
        )
        expected_weights = (
            1.0 if freeze_wrapper or freeze_child else 0.8,
            2.0 if freeze_wrapper else 1.9,
            0.1,
        )
        assert tuple(weight.item() for weight in weights(result.graph)) == pytest.approx(
            expected_weights
        )
        final = rollout_graph(result.graph, compiled, key=key)["outputs"]["output"]
        assert jnp.allclose(final, expected_weights[0] * expected_weights[1] * expected_weights[2])

        assert set(result.execution_metadata["trainable_nodes"]) == expected_paths

    authored_fingerprint, restored_fingerprint = nested_fingerprints
    assert restored_fingerprint == authored_fingerprint

    for original, restored in (
        (nested_spec.nodes, serialized.nodes),
        (nested_spec.subgraphs["wrapper"].nodes, serialized.subgraphs["wrapper"].nodes),
    ):
        assert {
            name: node.params["trainable"]
            for name, node in restored.items()
            if "trainable" in node.params
        } == {
            name: node.params["trainable"]
            for name, node in original.items()
            if "trainable" in node.params
        }


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
            component_registry=ComponentRegistry(),
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
            component_registry=ComponentRegistry(),
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
                type="feedbax.Channel",
                params={"delay": 2, "add_noise": False},
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        output_ports=["output"],
        output_bindings={"output": ("delay", "output")},
    ).model_dump(mode="json", exclude_none=True)
    task_binding_spec = {
        "schema_version": "feedbax.spec.studio.task_bindings.v3",
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
        component_registry=ComponentRegistry(),
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
    task_binding_spec = StudioTaskBindingSpec.model_validate(_mux_task_binding_spec())

    compiled = compile_training_run(
        component_registry=ComponentRegistry(),
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


def test_worker_request_materializes_omitted_dynamic_ports_from_bootstrap_registry() -> None:
    graph_spec = GraphSpec.model_validate(_mux_graph_spec())
    graph_spec.nodes["mux"].input_ports = []
    graph_spec.nodes["mux"].output_ports = []
    job = _Job(
        job_id="dynamic-ports",
        run_set_id="run-set",
        total_batches=1,
        stop_event=threading.Event(),
        training_spec=_training_spec(),
        task_spec={"type": "Generic", "params": {}},
        task_binding_spec=_mux_task_binding_spec(),
        graph_spec=graph_spec.model_dump(mode="json", exclude_none=True),
    )
    state = BootstrapState(
        bundle=new_application_registry_bundle(),
        provenance=(),
    )

    _require_worker_specs(job, state)

    normalized = GraphSpec.model_validate(job.graph_spec)
    assert normalized.nodes["mux"].input_ports == ["in_0", "in_1"]
    assert normalized.nodes["mux"].output_ports == ["output"]


def test_worker_rejects_degenerate_single_input_mux_before_materialization() -> None:
    task_binding_spec = _mux_task_binding_spec()
    task_binding_spec["exposed_data"] = task_binding_spec["exposed_data"][:1]
    task_binding_spec["bindings"] = task_binding_spec["bindings"][:1]

    with pytest.raises(ValueError, match=r"Mux 'mux' needs at least two connected inputs"):
        compile_training_run(
            component_registry=ComponentRegistry(),
            graph_spec=_mux_graph_spec(),
            training_spec=_training_spec(),
            task_spec={"type": "Generic", "params": {}},
            task_binding_spec=task_binding_spec,
            cfg=_cfg(),
        )


def test_compile_training_run_rejects_batch_size_larger_than_one() -> None:
    with pytest.raises(ValueError, match="supports batch_size=1") as excinfo:
        compile_training_run(
            component_registry=ComponentRegistry(),
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
            component_registry=ComponentRegistry(),
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
            component_registry=ComponentRegistry(),
            graph_spec=graph_spec,
            training_spec=_training_spec(),
            task_spec={"type": "Generic", "params": {}},
            task_binding_spec=_task_binding_spec(),
            cfg=_cfg(),
        )


@pytest.mark.parametrize(
    "reference",
    [
        {"namespace": "task_data", "compact": "task_data:other_input", "path": "inputs.other"},
        None,
    ],
    ids=["reference", "bound-catalog-entry"],
)
def test_compile_training_run_rejects_unsupported_task_data_value_spec_mode(
    reference: dict[str, str] | None,
) -> None:
    task_binding_spec = deepcopy(_task_binding_spec())
    task_binding_spec["exposed_data"][0]["value_spec"] = {
        "mode": "reference",
        "reference": reference,
        "metadata": {"value_schema": {"type": "array"}},
    }

    with pytest.raises(ValueError, match="unsupported value_spec mode='reference'"):
        compile_training_run(
            component_registry=ComponentRegistry(),
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
            component_registry=ComponentRegistry(),
            graph_spec=_linear_graph_spec(),
            training_spec=_training_spec(),
            task_spec={"type": "Generic", "params": {}},
            task_binding_spec=task_binding_spec,
            cfg=_cfg(),
        )


def test_worker_trains_on_named_timeline_segment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FEEDBAX_RUNS_DIR", str(tmp_path / "runs"))
    cfg = _cfg(n_reach_steps=3, grad_clip=None)
    graph_spec = _linear_graph_spec()
    graph_spec["nodes"]["readout"]["params"].update(input_size=3, use_bias=False)
    binding = _task_binding_spec()
    model_input = binding["exposed_data"][0]
    model_input["expected_shape"] = ["time", 3]
    # Each epoch reads one weight, giving a squared error of 1, 4, or 9.
    model_input["value_spec"].update(
        value=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        shape=["time", 3],
    )
    compiled = compile_training_run(
        component_registry=ComponentRegistry(),
        graph_spec=graph_spec,
        training_spec=_training_spec(
            n_batches=1,
            loss={
                **_training_spec()["loss"],
                "time_agg": {"mode": "segment", "segment_name": "movement"},
            },
        ),
        task_spec={
            "type": "DelayedReaches",
            "params": {"n_steps": 3},
            "timeline": {
                "epochs": [
                    {
                        "id": epoch,
                        "label": epoch,
                        "index": index,
                        "length": {"mode": "constant", "value": length},
                    }
                    for index, (epoch, length) in enumerate(
                        (("hold", {"steps": 1}), ("reach", {"steps": 1}), ("settle", None))
                    )
                ],
                "segments": [
                    {"id": "movement", "label": "Movement", "epoch_ids": ["reach"]},
                ],
            },
        },
        task_binding_spec=binding,
        cfg=cfg,
    )
    compiled.graph = eqx.tree_at(
        lambda graph: graph.nodes["readout"].layer.weight,
        compiled.graph,
        jnp.array([[1.0, 2.0, 3.0]]),
    )
    events: list[dict] = []
    result = run_training_graph(
        compiled,
        job_id="segment-loss",
        total_batches=1,
        cfg=cfg,
        stop_event=threading.Event(),
        emit=events.append,
    )

    (progress,) = [event for event in events if event["type"] == "training_progress"]
    assert progress["loss"] == pytest.approx(4.0)
    assert progress["grad_norm"] == pytest.approx(4.0)
    weights = result.graph.nodes["readout"].layer.weight[0]
    assert weights[0] == 1.0
    assert weights[1] == pytest.approx(1.9)
    assert weights[2] == 3.0


def test_studio_sampling_follows_task_data_and_epoch_ids() -> None:
    key = jax.random.key(40)
    data = _task_binding_spec()["exposed_data"][0]
    other_data = {**data, "id": "other_input", "path": "inputs.other"}
    sampled_length = {
        "mode": "distribution",
        "distribution": {"family": "uniform", "parameters": {"min": 2, "max": 5}},
    }
    sampled_value = {
        "mode": "distribution",
        "distribution": {"family": "normal", "parameters": {"mean": 0, "std": 1}},
    }
    final_epoch = {"id": "finish", "label": "Finish", "length": {"mode": "constant"}}
    sample_epoch = {"id": "sample", "label": "Sample", "length": sampled_length}
    unrelated_epoch = {"id": "prefix", "label": "Prefix", "length": sampled_length}
    value_entry = {
        "target_id": data["id"],
        "epoch_id": "sample",
        "value_spec": sampled_value,
    }

    def sample(epochs, entries, exposed):
        epoch_order = {epoch["id"]: i for i, epoch in enumerate(epochs)}
        task = TaskSpec.model_validate(
            {
                "type": "Generic",
                "timeline": {
                    "schema_version": STUDIO_TASK_TIMELINE_SCHEMA_VERSION,
                    "epochs": [{**epoch, "index": i} for i, epoch in enumerate(epochs)],
                    "signals": [
                        {key: item[key] for key in ("id", "label", "kind", "path")}
                        for item in exposed
                    ],
                    "epoch_value_specs": sorted(
                        entries,
                        key=lambda entry: (entry["target_id"], epoch_order[entry["epoch_id"]]),
                    ),
                },
            }
        )
        binding = StudioTaskBindingSpec(exposed_data=exposed)
        values = worker_execution._materialize_task_data(binding, task, 12, key=key)
        bounds = worker_execution._timeline_epoch_bounds(task.timeline, 12, key)
        return values, bounds

    before, original_bounds = sample(
        [sample_epoch, final_epoch],
        [value_entry],
        [data],
    )
    after, edited_bounds = sample(
        [unrelated_epoch, {**sample_epoch, "label": "Relabeled"}, final_epoch],
        [
            {**value_entry, "epoch_id": "prefix"},
            {**value_entry, "target_id": other_data["id"]},
            value_entry,
        ],
        [other_data, {**data, "label": "Relabeled input"}],
    )

    start, end = edited_bounds[1:3]
    assert end - start == original_bounds[1]
    kept = after[data["id"]][start:end]
    assert jnp.array_equal(before[data["id"]][: original_bounds[1]], kept)
    assert not jnp.array_equal(after[other_data["id"]][start:end], kept)


def test_compile_training_run_rejects_a_path_that_shadows_task_data_identity() -> None:
    binding = _task_binding_spec()
    original = binding["exposed_data"][0]
    binding["exposed_data"].append({**original, "id": "alias", "path": original["id"]})

    with pytest.raises(GraphCompilationError) as caught:
        compile_training_run(
            component_registry=ComponentRegistry(),
            graph_spec=_linear_graph_spec(),
            training_spec=_training_spec(),
            task_spec={"type": "Generic", "params": {}},
            task_binding_spec=binding,
            cfg=_cfg(),
        )

    (diagnostic,) = caught.value.diagnostics
    assert diagnostic.code == "worker.ambiguous_task_data_path"
    assert diagnostic.location == {"path": "/task_binding_spec/exposed_data/1/path"}


def test_compile_training_run_rejects_bound_task_data_without_value_spec() -> None:
    task_binding_spec = deepcopy(_task_binding_spec())
    task_binding_spec["exposed_data"][0].pop("value_spec")

    with pytest.raises(GraphCompilationError) as exc_info:
        compile_training_run(
            component_registry=ComponentRegistry(),
            graph_spec=_linear_graph_spec(),
            training_spec=_training_spec(),
            task_spec={"type": "Generic", "params": {}},
            task_binding_spec=task_binding_spec,
            cfg=_cfg(),
        )

    assert exc_info.value.diagnostics[0].code == "worker.missing_task_data_value_spec"
