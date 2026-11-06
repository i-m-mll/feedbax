"""Eager perturbations for feedback signals."""

from __future__ import annotations

import equinox as eqx
from equinox.nn import StateIndex
import jax
import jax.numpy as jnp
import jax.random as jr

from feedbax.runtime.signal_transforms import ConstantInput, ConstantInputParams
from feedbax.runtime.graph import Component, Graph


def random_unit_vector(key, dim):
    v = jr.normal(key, (dim,))
    return v / jnp.linalg.norm(v)


class _FeedbackOffset(ConstantInput):
    input_ports = ("input", "scheduled_active")

    feedback_var: int = eqx.field(static=True)
    feedback_dim: int | None = eqx.field(static=True)
    direction_index: StateIndex | None

    def __init__(self, amplitude, feedback_var, feedback_dim, *, sample_direction):
        axis = 0 if feedback_dim is None else feedback_dim
        basis = jax.nn.one_hot(axis, 2, dtype=jnp.float32)
        super().__init__(ConstantInputParams(scale=amplitude, arrays=basis))
        self.feedback_var = feedback_var
        self.feedback_dim = feedback_dim
        self.direction_index = StateIndex((jnp.asarray(False), basis)) if sample_direction else None

    def __call__(self, inputs, state, *, key):
        params: ConstantInputParams = state.get(self.params_index)
        direction = params.arrays
        if self.direction_index is not None:
            initialized, direction = state.get(self.direction_index)

            def sample():
                angle = jr.uniform(key, (), dtype=direction.dtype, maxval=2 * jnp.pi)
                return jnp.stack((jnp.cos(angle), jnp.sin(angle)))

            direction = jax.lax.cond(initialized, lambda: direction, sample)
            state = state.set(self.direction_index, (jnp.asarray(True), direction))

        feedback = inputs["input"]
        signal = feedback[self.feedback_var]
        if signal.shape[-1:] != (2,):
            raise ValueError("feedback_impulse expects a two-dimensional position or velocity")

        active = params.active & inputs["scheduled_active"]
        if self.feedback_dim is None:
            updated = self._apply(signal, ConstantInputParams(params.scale, active, direction))
        else:
            dim = self.feedback_dim
            coord_params = ConstantInputParams(params.scale, active, direction[dim])
            updated = signal.at[..., dim].set(self._apply(signal[..., dim], coord_params))
        output = eqx.tree_at(lambda value: value[self.feedback_var], feedback, updated)
        return {"output": output}, state


class _FeedbackImpulse(Graph):
    active_schedule: jax.Array

    def __init__(self, offset: _FeedbackOffset, active_schedule: jax.Array):
        super().__init__(
            nodes={"offset": offset},
            input_ports=("input",),
            output_ports=("output",),
            input_bindings={"input": ("offset", "input")},
            output_bindings={"output": ("offset", "output")},
        )
        self.active_schedule = active_schedule

    def _execute_step(self, port_values, state, *, t, **kwargs):
        # The final entry is an inactive sentinel for times outside the schedule.
        end = self.active_schedule.shape[0] - 1
        schedule_step = jnp.where((0 <= t) & (t < end), t, end)
        port_values[("offset", "scheduled_active")] = self.active_schedule[schedule_step]
        return super()._execute_step(port_values, state, t=t, **kwargs)

    def state_view(self, state):
        return None


def feedback_impulse(
    n_steps: int,
    amplitude: float,
    duration: int,
    feedback_var: int,
    start_timestep: int,
    feedback_dim: int | None = None,
) -> Component:
    """Add an offset to a point mass's ``(position, velocity)`` feedback tuple.

    ``feedback_var`` selects position (0) or velocity (1). With ``feedback_dim``
    set, the signed amplitude is added along x (0) or y (1). Otherwise a
    seeded unit direction is carried throughout the trial. Use fresh
    component state for each trial.

    The constructor's ``n_steps`` counts state samples, so the schedule applies
    over at most ``n_steps - 1`` transitions. A rollout's ``n_steps`` counts
    actual transitions and its clock starts at zero. Manual stepping uses
    the supplied ``t``. The impulse is active on the intersection of the
    schedule horizon with ``[start_timestep, start_timestep + duration)``.
    """
    if n_steps < 1:
        raise ValueError("n_steps must count at least the initial state")
    if feedback_var not in (0, 1):
        raise ValueError("feedback_var must select position (0) or velocity (1)")
    if feedback_dim not in (None, 0, 1):
        raise ValueError("feedback_dim must select x (0), y (1), or a random direction (None)")

    end = min(start_timestep + duration, n_steps - 1)
    steps = jnp.arange(n_steps)
    active = (start_timestep <= steps) & (steps < end)
    offset = _FeedbackOffset(
        amplitude,
        feedback_var,
        feedback_dim,
        sample_direction=feedback_dim is None and max(start_timestep, 0) < end,
    )
    return _FeedbackImpulse(offset, active)
