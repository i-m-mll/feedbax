"""Solver checkpoints resume their stored model and persistent runtime state."""

import json
import shutil
import subprocess
import sys
from textwrap import dedent

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
import numpy as np
import optimistix as optx
import pytest

from feedbax.contracts.worker import ProgressCoordinate
from feedbax.mechanics import DiffraxBackend, Mechanics
from feedbax.mechanics.plant import DirectForceInput
from feedbax.mechanics.skeleton.pointmass import PointMass
from feedbax.mechanics.skeleton.pointmass_dae import PointMassDAE
from feedbax.runtime.filters import FirstOrderFilter
from feedbax.runtime.graph import Graph
from feedbax.runtime.state import CartesianState
from feedbax.training.checkpoint_custody import (
    CheckpointCompatibilityError,
    authenticate_published_checkpoint_custody,
    load_latest_checkpoint,
    write_checkpoint_transaction,
)
from tests.test_checkpoint_custody import _run_spec


pytestmark = pytest.mark.usefixtures("enable_jax_x64")


def _mixed_cartesian_state():
    return CartesianState(
        pos=jnp.zeros(2, dtype=jnp.float32),
        vel=jnp.full(2, 1 + 1e-10, dtype=jnp.float64),
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
            init_value=jnp.asarray(1 + 1e-10, dtype=jnp.float64),
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
        backend=DiffraxBackend(0.01, solver=solver_type()) if kind == "backend" else None,
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


def _exercise_checkpoint_continuation(tmp_path, kind, solver_type, *, publish=True):
    key = jax.random.PRNGKey(0)
    model = _build(kind, solver_type, 0.2)
    if kind == "filter":
        model = _build(kind, solver_type, 0.4)
        _ = model._term
        model = eqx.tree_at(
            lambda component: (component.tau_rise, component.tau_decay),
            model,
            (jnp.asarray(0.2, dtype=jnp.float64), jnp.asarray(0.2, dtype=jnp.float64)),
        )
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
    if publish:
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
    if publish:
        assert authenticated.manifest.transaction_id == written.manifest.transaction_id
    loaded = load_latest_checkpoint(
        tmp_path,
        expected_run_spec=spec,
        expected_phase_program=program,
        expected_slots=slots(fresh, fresh.init_state(key=key)),
    )
    restored, restored_state = loaded.slots["model"]
    if solver_type is dfx.ImplicitEuler:
        assert restored.root_finder is restored.solver.root_finder
        _assert_same_arrays(restored.root_finder, model.root_finder)
        replacement = optx.Newton(rtol=1e-6, atol=1e-9)
        updated = eqx.tree_at(
            lambda component: component.solver, restored, dfx.ImplicitEuler(root_finder=replacement)
        )
        assert updated.root_finder is updated.solver.root_finder
        _assert_same_arrays(updated.root_finder, replacement)
        assert restored.root_finder.rtol == 1e-8
        different_norm = eqx.tree_at(
            lambda component: component.solver.root_finder.norm, fresh, optx.rms_norm
        )
        with pytest.raises(CheckpointCompatibilityError, match="norm field=static_repr_sha256"):
            load_latest_checkpoint(
                tmp_path,
                expected_run_spec=spec,
                expected_phase_program=program,
                expected_slots=slots(different_norm, different_norm.init_state(key=key)),
            )
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
        if kind == "filter":
            decay = 1 - model.dt / 0.2 if solver_type is dfx.Euler else np.exp(-model.dt / 0.2)
            analytic = 2 * decay - 1 + 1e-10 * decay**2
            np.testing.assert_allclose(actual_output["output"]["y"], analytic, atol=1e-13, rtol=0)
            assert abs(float(actual_output["output"]["y"][0]) - (2 * decay - 1)) > 1e-11
        elif kind != "nested":
            physical = (
                actual_output["state"] if kind == "dae" else actual_output["state"].plant.skeleton
            )
            np.testing.assert_allclose(physical.vel, 1 + 1e-10, atol=1e-13, rtol=0)


@pytest.mark.parametrize(
    ("kind", "solver_type"),
    [
        (kind, solver)
        for kind in ("filter", "dae", "mechanics", "nested")
        for solver in (dfx.Euler, dfx.Tsit5)
    ]
    + [("dae", dfx.ImplicitEuler), ("backend", dfx.Tsit5)],
)
def test_solver_checkpoint_executes_returned_model_and_runtime_state(tmp_path, kind, solver_type):
    _exercise_checkpoint_continuation(tmp_path, kind, solver_type)


@pytest.mark.parametrize(("kind", "solver"), [("filter", "Tsit5"), ("dae", "ImplicitEuler")])
def test_solver_checkpoint_continuation_survives_cold_and_warm_processes(tmp_path, kind, solver):
    probe = dedent(
        """
        import json
        import logging
        import sys
        from pathlib import Path
        from scripts.full_suite import configure_native_threads
        configure_native_threads()
        import diffrax as dfx
        import jax
        from tests.test_solver_checkpoint_continuation import (
            _exercise_checkpoint_continuation,
        )

        class SolverCacheHits(logging.Handler):
            def __init__(self):
                super().__init__()
                self.keys = []
            def emit(self, record):
                if 'Persistent compilation cache hit' in record.getMessage():
                    module, key = record.args
                    if module == 'jit_ode_step':
                        self.keys.append(key)

        hits = SolverCacheHits()
        logger = logging.getLogger('jax._src.compiler')
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.handlers = [hits]
        jax.config.update('jax_compilation_cache_dir', sys.argv[1])
        jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)
        jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)
        with jax.experimental.enable_x64():
            _exercise_checkpoint_continuation(
                Path(sys.argv[2]), sys.argv[3], getattr(dfx, sys.argv[4]),
                publish=sys.argv[5] == 'cold',
            )
        print(json.dumps(hits.keys))
        """
    )
    cache = tmp_path / "compilation_cache"
    results = []
    try:
        for temperature in ("cold", "warm"):
            process = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    probe,
                    str(cache),
                    str(tmp_path / "checkpoint"),
                    kind,
                    solver,
                    temperature,
                ],
                capture_output=True,
                text=True,
                timeout=25,
            )
            assert process.returncode == 0, process.stdout + process.stderr
            results.append(json.loads(process.stdout))
        assert results[1], "warm continuation did not reuse a persistent jit_ode_step entry"
    finally:
        if cache.exists():
            shutil.rmtree(cache)
