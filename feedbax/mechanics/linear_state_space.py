"""Discrete linear state-space mechanics component."""

from __future__ import annotations

from equinox import Module, field
from equinox.nn import State, StateIndex
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree

from feedbax.runtime.graph import Component
from feedbax.runtime.state import CartesianState


class LinearStateSpaceState(Module):
    """State for a discrete linear state-space mechanics component.

    Attributes:
        vector: Raw state vector, shape ``[n_state]``.
    """

    vector: Array


class LinearStateSpace(Component):
    """Discrete linear state-space mechanics.

    Steps

    ``x_next = A @ x + B @ u + B_w @ epsilon``

    once per call. The disturbance input ``epsilon`` is optional at execution
    time and defaults to zeros with the width of ``B_w``.

    Args:
        A: Discrete state matrix, shape ``[n_state, n_state]``.
        B: Input matrix, shape ``[n_state, n_input]``.
        B_w: Optional disturbance matrix, shape ``[n_state, n_epsilon]``.
        dt: Control timestep metadata. The stepping equation is already
            discrete and does not integrate over ``dt``.
        initial_state: Optional initial state vector, shape ``[n_state]``.
        pos_slice: Half-open slice selecting effector position from the state.
        vel_slice: Half-open slice selecting effector velocity from the state.
    """

    input_ports = ("force", "epsilon")
    output_ports = ("effector", "state")

    A: Array
    B: Array
    B_w: Array
    dt: float
    pos_slice: tuple[int, int] = field(static=True)
    vel_slice: tuple[int, int] = field(static=True)
    initial_state: tuple[float, ...] = field(static=True)
    state_index: StateIndex

    def __init__(
        self,
        A: Array,
        B: Array,
        B_w: Array | None = None,
        dt: float = 1.0,
        initial_state: Array | None = None,
        pos_slice: tuple[int, int] = (0, 2),
        vel_slice: tuple[int, int] = (2, 4),
    ):
        self.A = jnp.asarray(A, dtype=float)
        self.B = jnp.asarray(B, dtype=float)
        self.dt = float(dt)
        self.pos_slice = tuple(int(x) for x in pos_slice)
        self.vel_slice = tuple(int(x) for x in vel_slice)

        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must be a square matrix.")
        if self.B.ndim != 2 or self.B.shape[0] != self.A.shape[0]:
            raise ValueError("B must have shape [n_state, n_input].")

        n_state = self.A.shape[0]
        if B_w is None:
            self.B_w = jnp.zeros((n_state, 0), dtype=self.A.dtype)
        else:
            self.B_w = jnp.asarray(B_w, dtype=float)
            if self.B_w.ndim != 2 or self.B_w.shape[0] != n_state:
                raise ValueError("B_w must have shape [n_state, n_epsilon].")

        if initial_state is None:
            vector = jnp.zeros(n_state, dtype=self.A.dtype)
        else:
            vector = jnp.asarray(initial_state, dtype=float)
            if vector.shape != (n_state,):
                raise ValueError("initial_state must have shape [n_state].")

        self.initial_state = tuple(float(x) for x in vector.tolist())
        self.state_index = StateIndex(LinearStateSpaceState(vector=vector))

    def _effector(self, vector: Array, force: Array) -> CartesianState:
        pos_start, pos_stop = self.pos_slice
        vel_start, vel_stop = self.vel_slice
        return CartesianState(
            pos=vector[pos_start:pos_stop],
            vel=vector[vel_start:vel_stop],
            force=force,
        )

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        current: LinearStateSpaceState = state.get(self.state_index)
        force = jnp.asarray(inputs["force"], dtype=self.A.dtype)
        epsilon = jnp.asarray(
            inputs.get("epsilon", jnp.zeros((self.B_w.shape[1],), dtype=self.A.dtype)),
            dtype=self.A.dtype,
        )

        vector = self.A @ current.vector + self.B @ force + self.B_w @ epsilon
        next_state = LinearStateSpaceState(vector=vector)
        state = state.set(self.state_index, next_state)
        return {
            "effector": self._effector(vector, force),
            "state": vector,
        }, state

    def state_view(self, state: State) -> LinearStateSpaceState:
        return state.get(self.state_index)

    def initial_outputs(self, state_value: PyTree | None) -> dict[str, PyTree]:
        if state_value is None:
            return {}
        return {
            "effector": self._effector(
                state_value.vector,
                jnp.zeros((self.B.shape[1],), dtype=self.A.dtype),
            ),
            "state": state_value.vector,
        }
