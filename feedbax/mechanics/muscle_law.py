"""Pure stage-local laws for coupling muscle state to an attachment."""

from __future__ import annotations

from typing import NamedTuple, Protocol, runtime_checkable

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
