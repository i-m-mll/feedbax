"""Registered nonlinear models survive checkpoint publication and fresh-process resume."""

import subprocess
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from feedbax.compiler import GraphDocument, compile_graph
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.graph import ComponentSpec, GraphSpec, WireSpec
from feedbax.contracts.training import GraphTopologySourceSpec
from feedbax.contracts.worker import ProgressCoordinate
from feedbax.models.cde import CDENetwork
from feedbax.runtime.iteration import iterate_component
from feedbax.training.checkpoint_custody import load_latest_checkpoint, write_checkpoint_transaction
from tests.test_checkpoint_custody import _run_spec
from tests.test_solver_checkpoint_continuation import _assert_same_arrays


def _nonlinear_graph():
    nodes = {
        name: ComponentSpec(
            type=type_,
            params=params,
            input_ports=["input", "hidden"] if name == "rnn" else ["input"],
            output_ports=["output", "hidden"] if name == "rnn" else ["output"],
        )
        for name, type_, params in (
            (
                "mlp",
                "MLP",
                {
                    "input_size": 2,
                    "output_size": 2,
                    "hidden_sizes": [3],
                    "activation": "relu",
                    "final_activation": "tanh",
                },
            ),
            ("rnn", "VanillaRNN", {"input_size": 2, "hidden_size": 2, "activation": "tanh"}),
            ("linear", "Linear", {"input_size": 2, "output_size": 2, "activation": "sigmoid"}),
            ("output", "Activation", {"activation": "tanh"}),
        )
    }
    return GraphSpec(
        nodes=nodes,
        wires=[
            WireSpec(
                source_node=source, source_port="output", target_node=target, target_port="input"
            )
            for source, target in (("mlp", "rnn"), ("rnn", "linear"), ("linear", "output"))
        ],
        input_ports=["input"],
        output_ports=["output"],
        input_bindings={"input": ("mlp", "input")},
        output_bindings={"output": ("output", "output")},
    )


def _initial_slots(kind, *, seed):
    key = jax.random.PRNGKey(seed)
    if kind == "graph":
        graph = compile_graph(
            GraphDocument(graph=_nonlinear_graph()), ComponentRegistry(load_user_components=False)
        ).graph
        model = graph.prepare()
    else:
        model = CDENetwork(
            obs_dim=2, hidden_dim=3, out_size=2, vf_width=4, vf_depth=1, key=key
        ).prepare()
    return {
        "model": (model, model.init_state(key=key)),
        "optimizer": optax.adam(0.05).init(eqx.filter(model, eqx.is_inexact_array)),
        "prng": key,
        "batch_counter": jnp.asarray(0, dtype=jnp.int32),
    }


@eqx.filter_jit
def _train_step(slots, observations):
    model, carry = slots["model"]
    rollout_key, next_key = jax.random.split(slots["prng"])

    def loss(component):
        outputs, final_carry, history = iterate_component(
            component, {"input": observations}, carry, len(observations), rollout_key
        )
        return jnp.sum(outputs["output"][:, 0]), (outputs, final_carry, history)

    (_, trajectory), grads = eqx.filter_value_and_grad(loss, has_aux=True)(model)
    optimizer = optax.adam(0.05)
    updates, optimizer_state = optimizer.update(
        grads, slots["optimizer"], eqx.filter(model, eqx.is_inexact_array)
    )
    return {
        "model": (eqx.apply_updates(model, updates), trajectory[1]),
        "optimizer": optimizer_state,
        "prng": next_key,
        "batch_counter": slots["batch_counter"] + 1,
    }, (trajectory, grads)


def _run_contract(kind):
    spec = _run_spec()
    if kind == "graph":
        spec = spec.model_copy(
            update={
                "graph": GraphTopologySourceSpec(inline=_nonlinear_graph().model_dump(mode="json"))
            }
        )
    return spec, spec.worker_execution.method_contract.phase_program


def _interrupted_training(kind):
    fresh = _initial_slots(kind, seed=4)
    prefix = jnp.array([[1.25, -0.75], [-0.5, 0.9]], dtype=jnp.float32)
    interrupted, ((_, _, history), grads) = _train_step(fresh, prefix)
    if kind == "cde":
        model, _ = fresh["model"]
        learned, _ = interrupted["model"]
        _assert_same_arrays(history.hidden[0], model.h0)
        assert jnp.any(grads.h0 != 0)
        assert jnp.any(learned.h0 != model.h0)
    else:
        model, carry = interrupted["model"]
        assert jnp.any(carry.get(model.nodes["rnn"].state_index).hidden != 0)
    return interrupted


def _assert_slots(actual, expected):
    _assert_same_arrays(eqx.filter(actual, eqx.is_array), eqx.filter(expected, eqx.is_array))


def _resume(root, kind):
    spec, program = _run_contract(kind)
    expected = _interrupted_training(kind)
    fresh = _initial_slots(kind, seed=17)
    loaded = load_latest_checkpoint(
        root,
        expected_run_spec=spec,
        expected_phase_program=program,
        expected_slots=fresh,
    ).slots
    _assert_slots(loaded, expected)

    observations = jnp.array([[-0.2, -0.8], [0.9, 0.45]], dtype=jnp.float32)
    continued, (trajectory, grads) = _train_step(loaded, observations)
    expected_continued, expected_step = _train_step(expected, observations)
    _assert_slots(continued, expected_continued)
    _assert_slots((trajectory, grads), expected_step)

    if kind == "cde":
        model, carry = loaded["model"]
        stored_hidden = carry.get(model.state_index).hidden
        _assert_same_arrays(trajectory[2].hidden[0], stored_hidden)
        assert jnp.any(stored_hidden != model.h0)
        _assert_same_arrays(grads.h0, jnp.zeros_like(model.h0))
        assert jnp.any(grads.vector_field.layers[0].weight != 0)


@pytest.mark.parametrize("kind", ["graph", "cde"])
def test_nonlinear_checkpoint_resumes_training_in_fresh_process(tmp_path, kind):
    spec, program = _run_contract(kind)
    write_checkpoint_transaction(
        tmp_path,
        run_spec=spec,
        phase_program=program,
        barrier_name="after_train_batch",
        coordinate=ProgressCoordinate(
            run_id="run-2",
            phase="train_batch",
            program_step=1,
            completed_barrier="after_train_batch",
        ),
        slots=_interrupted_training(kind),
    )
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from pathlib import Path
import sys
from scripts.full_suite import configure_native_threads
configure_native_threads()
from tests.test_activation_checkpoint_continuation import _resume
_resume(Path(sys.argv[1]), sys.argv[2])
""",
            str(tmp_path),
            kind,
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert process.returncode == 0, process.stdout + process.stderr
