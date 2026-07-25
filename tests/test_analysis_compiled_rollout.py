"""Contract tests for the compile-once batched trial-rollout facility."""

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.tree as jt
import numpy as np
import pytest

from feedbax.analysis import (
    TrialStructureError,
    compiled_trial_rollout,
    stack_trials,
)


class _Context(NamedTuple):
    """Trial-invariant operands: a linear plant plus a static feedback gain."""

    transition: jax.Array
    command: jax.Array
    gain: jax.Array


class _Trial(NamedTuple):
    """Per-trial operands: an initial state and a per-step disturbance."""

    initial_state: jax.Array
    disturbance: jax.Array


def _rollout_trial(context: _Context, trial: _Trial) -> dict[str, jax.Array]:
    """Roll one closed-loop linear trial forward with `jax.lax.scan`."""

    def step(
        state: jax.Array, disturbance: jax.Array
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        command = -context.gain @ state
        next_state = context.transition @ state + context.command @ command + disturbance
        return next_state, (next_state, command)

    final_state, (states, commands) = jax.lax.scan(step, trial.initial_state, trial.disturbance)
    return {
        "states": jnp.concatenate([trial.initial_state[None, :], states], axis=0),
        "commands": commands,
        "final_state": final_state,
    }


def _counting_rollout_trial() -> tuple[Callable[[_Context, _Trial], Any], dict[str, int]]:
    """Return a fresh per-trial function plus its trace counter.

    Each call builds a new function object, so each test gets its own `jax.jit`
    cache entries and the tests stay order-independent under `pytest-xdist`.
    """
    counter = {"traces": 0}

    def per_trial(context: _Context, trial: _Trial) -> dict[str, jax.Array]:
        counter["traces"] += 1
        return _rollout_trial(context, trial)

    return per_trial, counter


def _context(state_dim: int = 3, command_dim: int = 2, dtype: Any = None) -> _Context:
    generator = np.random.default_rng(0)
    return _Context(
        transition=jnp.asarray(generator.normal(size=(state_dim, state_dim)) * 0.1, dtype=dtype),
        command=jnp.asarray(generator.normal(size=(state_dim, command_dim)) * 0.1, dtype=dtype),
        gain=jnp.asarray(generator.normal(size=(command_dim, state_dim)) * 0.1, dtype=dtype),
    )


def _trials(n_trials: int = 4, horizon: int = 5, state_dim: int = 3, seed: int = 1) -> list[_Trial]:
    generator = np.random.default_rng(seed)
    return [
        _Trial(
            initial_state=jnp.asarray(generator.normal(size=(state_dim,))),
            disturbance=jnp.asarray(generator.normal(size=(horizon, state_dim)) * 0.01),
        )
        for _ in range(n_trials)
    ]


class _GatedTrial(NamedTuple):
    """Per-trial operands plus a gate selecting a `lax.cond` branch."""

    initial_state: jax.Array
    disturbance: jax.Array
    gate: jax.Array


def _gated_rollout_trial(context: _Context, trial: _GatedTrial) -> dict[str, jax.Array]:
    """Roll one trial forward, branching on a per-trial gate with `lax.cond`.

    Mirrors the numeric regime of the downstream analytical kernels: a float64
    scan carry with a `lax.cond` on a trial-level scalar, so a batch containing
    both zero and non-zero gates exercises both branch outcomes in one compiled
    executable.
    """

    def step(
        state: jax.Array, disturbance: jax.Array
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        raw = -context.gain @ state

        def without_gate() -> jax.Array:
            return raw

        def with_gate() -> jax.Array:
            return raw + trial.gate * jnp.tanh(raw)

        command = jax.lax.cond(trial.gate == 0, without_gate, with_gate)
        next_state = context.transition @ state + context.command @ command + disturbance
        return next_state, (next_state, command)

    final_state, (states, commands) = jax.lax.scan(step, trial.initial_state, trial.disturbance)
    return {
        "states": jnp.concatenate([trial.initial_state[None, :], states], axis=0),
        "commands": commands,
        "final_state": final_state,
    }


def _gated_trials(
    gates: Sequence[float],
    horizon: int = 5,
    state_dim: int = 3,
    seed: int = 2,
    dtype: Any = jnp.float64,
) -> list[_GatedTrial]:
    generator = np.random.default_rng(seed)
    return [
        _GatedTrial(
            initial_state=jnp.asarray(generator.normal(size=(state_dim,)), dtype=dtype),
            disturbance=jnp.asarray(
                generator.normal(size=(horizon, state_dim)) * 0.01, dtype=dtype
            ),
            gate=jnp.asarray(gate, dtype=dtype),
        )
        for gate in gates
    ]


def _assert_bit_identical(actual: Any, expected: Any) -> None:
    actual_leaves, actual_treedef = jt.flatten(actual)
    expected_leaves, expected_treedef = jt.flatten(expected)
    assert actual_treedef == expected_treedef
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
        actual_array = np.asarray(actual_leaf)
        expected_array = np.asarray(expected_leaf)
        assert actual_array.dtype == expected_array.dtype
        assert actual_array.shape == expected_array.shape
        assert actual_array.tobytes() == expected_array.tobytes()


def test_compiled_rollout_is_bit_identical_to_python_loop() -> None:
    context = _context()
    trials = _trials()
    rollout = compiled_trial_rollout(_rollout_trial)

    compiled = rollout(context, stack_trials(trials))
    scalar = stack_trials([_rollout_trial(context, trial) for trial in trials])

    _assert_bit_identical(compiled, scalar)


def test_float64_cond_rollout_is_bit_identical_to_python_loop(enable_jax_x64: None) -> None:
    gates = [0.0, 0.75, 0.0, -1.25]
    context = _context(dtype=jnp.float64)
    trials = _gated_trials(gates)
    rollout = compiled_trial_rollout(_gated_rollout_trial)

    compiled = rollout(context, stack_trials(trials))
    scalar = stack_trials([_gated_rollout_trial(context, trial) for trial in trials])

    _assert_bit_identical(compiled, scalar)
    for leaf in jt.leaves(compiled):
        assert np.asarray(leaf).dtype == np.float64

    # Both branch outcomes are live: zeroing the gates leaves the already
    # ungated trials untouched and changes exactly the gated ones.
    ungated = rollout(context, stack_trials(_gated_trials([0.0] * len(gates))))
    for index, gate in enumerate(gates):
        same = np.array_equal(
            np.asarray(compiled["commands"][index]), np.asarray(ungated["commands"][index])
        )
        assert same == (gate == 0.0)


def test_repeated_calls_with_one_shape_compile_once() -> None:
    per_trial, counter = _counting_rollout_trial()
    rollout = compiled_trial_rollout(per_trial)
    context = _context()
    stacked = stack_trials(_trials())

    first = rollout(context, stacked)
    for _ in range(3):
        _assert_bit_identical(rollout(context, stacked), first)

    assert counter["traces"] == 1


def test_rebuilt_wrapper_reuses_the_jit_cache() -> None:
    per_trial, counter = _counting_rollout_trial()
    context = _context()
    stacked = stack_trials(_trials())

    for _ in range(3):
        compiled_trial_rollout(per_trial)(context, stacked)

    assert counter["traces"] == 1


def test_second_shape_structure_triggers_exactly_one_recompile() -> None:
    per_trial, counter = _counting_rollout_trial()
    rollout = compiled_trial_rollout(per_trial)
    context = _context()

    rollout(context, stack_trials(_trials(n_trials=4, horizon=5)))
    assert counter["traces"] == 1

    rollout(context, stack_trials(_trials(n_trials=4, horizon=7)))
    assert counter["traces"] == 2

    rollout(context, stack_trials(_trials(n_trials=4, horizon=5)))
    rollout(context, stack_trials(_trials(n_trials=4, horizon=7)))
    assert counter["traces"] == 2


def test_fresh_per_trial_identity_retraces_as_documented() -> None:
    counter = {"traces": 0}
    context = _context()
    stacked = stack_trials(_trials())

    for _ in range(2):

        def per_trial(context: _Context, trial: _Trial) -> dict[str, jax.Array]:
            counter["traces"] += 1
            return _rollout_trial(context, trial)

        compiled_trial_rollout(per_trial)(context, stacked)

    assert counter["traces"] == 2


def test_declared_trial_structure_accepts_matching_trials() -> None:
    context = _context()
    trials = _trials(horizon=5)
    structure = jt.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), trials[0])
    rollout = compiled_trial_rollout(_rollout_trial, trial_structure=structure)

    _assert_bit_identical(
        rollout(context, stack_trials(trials)),
        stack_trials([_rollout_trial(context, trial) for trial in trials]),
    )


def test_declared_trial_structure_rejects_shape_mismatch() -> None:
    context = _context()
    structure = jt.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), _trials(horizon=5)[0])
    rollout = compiled_trial_rollout(_rollout_trial, trial_structure=structure)

    with pytest.raises(TrialStructureError, match="per-trial shape"):
        rollout(context, stack_trials(_trials(horizon=7)))


def test_declared_trial_structure_rejects_dtype_mismatch() -> None:
    context = _context()
    trials = _trials()
    structure = jt.map(lambda x: jax.ShapeDtypeStruct(x.shape, jnp.int32), trials[0])
    rollout = compiled_trial_rollout(_rollout_trial, trial_structure=structure)

    with pytest.raises(TrialStructureError, match="dtype"):
        rollout(context, stack_trials(trials))


def test_declared_trial_structure_rejects_structure_mismatch() -> None:
    context = _context()
    trials = _trials()
    structure = {
        "initial_state": jax.ShapeDtypeStruct((3,), jnp.float32),
        "disturbance": jax.ShapeDtypeStruct((5, 3), jnp.float32),
    }
    rollout = compiled_trial_rollout(_rollout_trial, trial_structure=structure)

    with pytest.raises(TrialStructureError, match="does not match the declared"):
        rollout(context, stack_trials(trials))


def test_ragged_leading_trial_axis_is_rejected() -> None:
    context = _context()
    stacked = stack_trials(_trials(n_trials=4))
    ragged = _Trial(
        initial_state=stacked.initial_state[:3],
        disturbance=stacked.disturbance,
    )
    rollout = compiled_trial_rollout(_rollout_trial)

    with pytest.raises(TrialStructureError, match="disagree on the leading trial axis"):
        rollout(context, ragged)


def test_scalar_trial_leaf_is_rejected() -> None:
    context = _context()
    stacked = stack_trials(_trials())
    scalar_leaf = _Trial(
        initial_state=stacked.initial_state,
        disturbance=jnp.asarray(1.0),
    )
    rollout = compiled_trial_rollout(_rollout_trial)

    with pytest.raises(TrialStructureError, match="is a scalar"):
        rollout(context, scalar_leaf)


def test_empty_trial_axis_is_rejected() -> None:
    context = _context()
    stacked = stack_trials(_trials())
    empty = jt.map(lambda x: x[:0], stacked)
    rollout = compiled_trial_rollout(_rollout_trial)

    with pytest.raises(TrialStructureError, match="empty leading trial axis"):
        rollout(context, empty)


def test_leafless_trials_pytree_is_rejected() -> None:
    context = _context()
    rollout = compiled_trial_rollout(_rollout_trial)

    with pytest.raises(TrialStructureError, match="no array leaves"):
        rollout(context, {})


def test_non_callable_per_trial_is_rejected() -> None:
    with pytest.raises(TypeError, match="per_trial must be callable"):
        compiled_trial_rollout(object())  # type: ignore[arg-type]


def test_stack_trials_matches_leafwise_stacking() -> None:
    trials = _trials()
    stacked = stack_trials(trials)

    _assert_bit_identical(
        stacked,
        _Trial(
            initial_state=jnp.stack([trial.initial_state for trial in trials]),
            disturbance=jnp.stack([trial.disturbance for trial in trials]),
        ),
    )


def test_stack_trials_rejects_an_empty_sequence() -> None:
    with pytest.raises(TrialStructureError, match="at least one trial"):
        stack_trials([])


def test_stack_trials_rejects_shape_mismatch() -> None:
    trials = _trials(n_trials=2, horizon=5)
    mismatched = [trials[0], _trials(n_trials=1, horizon=7)[0]]

    with pytest.raises(TrialStructureError, match="has shape"):
        stack_trials(mismatched)


def test_stack_trials_rejects_dtype_mismatch() -> None:
    trials = _trials(n_trials=2)
    mismatched = [
        trials[0],
        _Trial(
            initial_state=trials[1].initial_state.astype(jnp.float16),
            disturbance=trials[1].disturbance,
        ),
    ]

    with pytest.raises(TrialStructureError, match="has dtype"):
        stack_trials(mismatched)


def test_stack_trials_rejects_structure_mismatch() -> None:
    trials = _trials(n_trials=2)
    mismatched = [
        trials[0],
        {
            "initial_state": trials[1].initial_state,
            "disturbance": trials[1].disturbance,
        },
    ]

    with pytest.raises(TrialStructureError, match="pytree structure"):
        stack_trials(mismatched)
