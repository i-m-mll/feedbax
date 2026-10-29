"""Solver checkpoints resume their stored model and persistent runtime state."""

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
import numpy as np
import pytest

from feedbax.contracts.worker import ProgressCoordinate
from feedbax.mechanics import Mechanics
from feedbax.mechanics.plant import DirectForceInput
from feedbax.mechanics.skeleton.pointmass import PointMass
from feedbax.mechanics.skeleton.pointmass_dae import PointMassDAE
from feedbax.runtime.filters import FirstOrderFilter
from feedbax.runtime.graph import Graph
from feedbax.runtime.state import CartesianState
from feedbax.training.checkpoint_custody import (
    authenticate_published_checkpoint_custody,
    load_latest_checkpoint,
    write_checkpoint_transaction,
)
from tests.test_checkpoint_custody import _run_spec


pytestmark = pytest.mark.usefixtures("enable_jax_x64")


def _mixed_cartesian_state():
    return CartesianState(
        pos=jnp.zeros(2, dtype=jnp.float32),
        vel=jnp.zeros(2, dtype=jnp.float64),
        force=jnp.zeros(2, dtype=jnp.float32),
    )


class MixedCheckpointDAE(PointMassDAE):
    def init_system_state(self, *, key):
        return _mixed_cartesian_state()


class MixedCheckpointPointMass(PointMass):
    def init(self, *, key):
        return _mixed_cartesian_state()


def _build(kind, solver_type, parameter):
    if kind == "nested":
        return Graph(
            nodes={name: _build("filter", solver_type, parameter) for name in ("left", "right")},
            wires=(),
            input_ports=("left", "right"),
            output_ports=("left", "right"),
            input_bindings={name: (name, "input") for name in ("left", "right")},
            output_bindings={name: (name, "output") for name in ("left", "right")},
        ).prepare()
    parameter = jnp.asarray(parameter, dtype=jnp.float64)
    if kind == "filter":
        return FirstOrderFilter(
            tau_rise=parameter,
            tau_decay=parameter,
            solver_type=solver_type,
            input_proto={
                "x": jnp.zeros(2, dtype=jnp.float32),
                "y": jnp.zeros(3, dtype=jnp.float64),
            },
        ).prepare()
    if kind == "dae":
        return MixedCheckpointDAE(mass=parameter, solver_type=solver_type).prepare()
    return Mechanics(
        DirectForceInput(MixedCheckpointPointMass(mass=parameter), clip_states=False),
        dt=0.01,
        solver_type=solver_type,
    ).prepare()


def _parameter(kind, model):
    if kind == "nested":
        return {name: node.tau_rise for name, node in model.nodes.items()}
    if kind == "filter":
        return model.tau_rise
    if kind == "dae":
        return model.params.mass
    return model.plant.skeleton.mass


def _indices(kind, model):
    if kind == "nested":
        return {name: node.state_index for name, node in model.nodes.items()}
    return {kind: model.state_index}


def _state_values(kind, model, state):
    return {name: state.get(index) for name, index in _indices(kind, model).items()}


def _assert_same_arrays(actual, expected):
    assert jt.structure(actual) == jt.structure(expected)
    for actual_leaf, expected_leaf in zip(jt.leaves(actual), jt.leaves(expected), strict=True):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)
        if eqx.is_array(expected_leaf):
            assert actual_leaf.dtype == expected_leaf.dtype


@pytest.mark.parametrize("kind", ["filter", "dae", "mechanics", "nested"])
@pytest.mark.parametrize("solver_type", [dfx.Euler, dfx.Tsit5])
def test_solver_checkpoint_executes_returned_model_and_runtime_state(tmp_path, kind, solver_type):
    key = jax.random.PRNGKey(0)
    model = _build(kind, solver_type, 0.2)
    fresh = _build(kind, solver_type, 0.6)
    first_input = (
        {"input": jt.map(jnp.ones_like, model.input_proto)}
        if kind == "filter"
        else {"force": jnp.ones(2)}
    )
    if kind == "nested":
        first_input = {
            name: jt.map(lambda value: jnp.full_like(value, scale), node.input_proto)
            for (name, node), scale in zip(model.nodes.items(), (1, -2), strict=True)
        }
    next_input = jt.map(lambda value: -value, first_input)

    def warmed_state():
        return model(first_input, model.init_state(key=key), key=key)[1]

    state = warmed_state()
    saved_value = _state_values(kind, model, state)
    initial_value = _state_values(kind, model, model.init_state(key=key))
    assert not eqx.tree_equal(saved_value, initial_value)
    expected_steps = [
        step(next_input, warmed_state(), key=key) for step in (model, eqx.filter_jit(model))
    ]

    def slots(component, runtime_state):
        return {
            "model": (component, runtime_state),
            "optimizer": (),
            "prng": key,
            "batch_counter": jnp.array(1, dtype=jnp.int32),
        }

    spec = _run_spec()
    program = spec.worker_execution.method_contract.phase_program
    written = write_checkpoint_transaction(
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
        slots=slots(model, state),
    )
    authenticated = authenticate_published_checkpoint_custody(tmp_path)
    assert authenticated.manifest.transaction_id == written.manifest.transaction_id
    loaded = load_latest_checkpoint(
        tmp_path,
        expected_run_spec=spec,
        expected_phase_program=program,
        expected_slots=slots(fresh, fresh.init_state(key=key)),
    )
    restored, restored_state = loaded.slots["model"]
    _assert_same_arrays(_parameter(kind, restored), _parameter(kind, model))
    assert not eqx.tree_equal(_parameter(kind, restored), _parameter(kind, fresh))
    markers = [index.marker for index in _indices(kind, restored).values()]
    assert markers == [index.marker for index in _indices(kind, model).values()]
    assert len(set(markers)) == len(markers)
    _assert_same_arrays(
        _state_values(kind, restored, jt.map(lambda value: value, restored_state)), saved_value
    )
    _assert_same_arrays(_state_values(kind, restored, restored.init_state(key=key)), initial_value)

    for step, (expected_output, expected_state) in zip(
        (restored, eqx.filter_jit(restored)), expected_steps, strict=True
    ):
        actual_output, actual_state = step(
            next_input, jt.map(lambda value: value, restored_state), key=key
        )
        _assert_same_arrays(actual_output, expected_output)
        _assert_same_arrays(
            _state_values(kind, restored, actual_state), _state_values(kind, model, expected_state)
        )
