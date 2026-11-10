"""Pure stage-local laws for coupling muscle state to an attachment."""

from __future__ import annotations

from typing import NamedTuple, Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array


class MuscleStage(NamedTuple):
    """Differential state per second and scalar shortening tension in newtons."""

    derivative: Array
    tension: Array


@runtime_checkable
class MuscleLaw(Protocol):
    """A fixed-width, pure JAX law built by the graph's component registry.

    State is a floating vector. Excitation is dimensionless, length is in
    metres, and velocity is in metres per second, positive during lengthening.
    The attachment consumes positive tension as a shortening force.
    """

    @property
    def state_size(self) -> int: ...

    def init_state(self, length: Array, velocity: Array) -> Array: ...

    def evaluate(
        self, state: Array, excitation: Array, length: Array, velocity: Array,
    ) -> MuscleStage: ...


class MuscleLawBank(eqx.Module):
    """Evaluate heterogeneous laws in attachment order at one dynamics stage."""

    laws: tuple[MuscleLaw, ...]

    def init_state(self, lengths: Array, velocities: Array) -> tuple[Array, ...]:
        states = tuple(
            law.init_state(length, velocity)
            for law, length, velocity in zip(self.laws, lengths, velocities, strict=True)
        )
        for law, state in zip(self.laws, states, strict=True):
            if state.shape != (law.state_size,) or not jnp.issubdtype(state.dtype, jnp.floating):
                raise ValueError(f"{type(law).__name__} must initialize a floating state vector")
        return states

    def evaluate(
        self, states: tuple[Array, ...], excitations: Array, lengths: Array, velocities: Array,
    ) -> tuple[tuple[Array, ...], Array]:
        evaluated = tuple(
            law.evaluate(state, excitation, length, velocity)
            for law, state, excitation, length, velocity in zip(
                self.laws, states, jnp.clip(excitations, 0.0, 1.0), lengths, velocities,
                strict=True,
            )
        )
        for law, current, result in zip(self.laws, states, evaluated, strict=True):
            if result.derivative.shape != current.shape or result.derivative.dtype != current.dtype:
                raise ValueError(f"{type(law).__name__} derivative must match its state shape and dtype")
            if result.tension.shape != () or not jnp.issubdtype(result.tension.dtype, jnp.floating):
                raise ValueError(f"{type(law).__name__} must return a floating scalar tension")
        return tuple(result.derivative for result in evaluated), jnp.stack(
            [result.tension for result in evaluated],
        )
