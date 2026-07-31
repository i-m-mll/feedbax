"""Discrete linear state-space mechanics component."""

from __future__ import annotations

from collections.abc import Sequence

from equinox import Module, field
from equinox.nn import State, StateIndex
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree

from feedbax.runtime.graph import Component
from feedbax.runtime.state import CartesianState
from feedbax.contracts.array_values import (
    ARRAY_VALUE_SCHEMA_ID,
    ARRAY_VALUE_SCHEMA_VERSION,
    ArrayValueSpec,
    SparseCooArrayValueSpec,
    SparseCooEntrySpec,
    materialize_array_value,
)


STRUCTURAL_LINEAR_STATE_SPACE_PARAM_SCHEMA_VERSION = (
    "feedbax.component.structural_linear_state_space.v1"
)


class StructuralLinearDynamicsPerturbation(Module):
    """Trial-constant structural change to a discrete linear transition.

    ``delta_A`` is expressed in the same discrete-time domain as the nominal
    transition matrix. When active, the effective transition is
    ``A + scale * delta_A``. This contract does not project the change through
    an input or disturbance matrix.
    """

    delta_A: Array
    scale: Array
    active: Array

    def __init__(
        self,
        delta_A: Array,
        *,
        scale: float | Array = 1.0,
        active: bool | Array = True,
    ):
        delta_A = jnp.asarray(delta_A)
        if not jnp.issubdtype(delta_A.dtype, jnp.floating):
            delta_A = delta_A.astype(float)
        if delta_A.ndim != 2 or delta_A.shape[0] != delta_A.shape[1]:
            raise ValueError("delta_A must be a square matrix.")
        self.delta_A = delta_A
        self.scale = jnp.asarray(scale, dtype=delta_A.dtype)
        self.active = jnp.asarray(active, dtype=bool)
        if self.scale.ndim != 0:
            raise ValueError("scale must be scalar.")
        if self.active.ndim != 0:
            raise ValueError("active must be scalar.")

    @classmethod
    def from_entries(
        cls,
        shape: Sequence[int],
        entries: Sequence[tuple[int, int, float]],
        *,
        scale: float | Array = 1.0,
        active: bool | Array = True,
    ) -> StructuralLinearDynamicsPerturbation:
        """Construct a dense runtime perturbation from sparse authored entries."""

        declaration = SparseCooArrayValueSpec(
            schema_id=ARRAY_VALUE_SCHEMA_ID,
            schema_version=ARRAY_VALUE_SCHEMA_VERSION,
            encoding="sparse_coo",
            shape=tuple(shape),
            dtype="float32",
            nonfinite="forbid",
            fill=0.0,
            entries=tuple(
                SparseCooEntrySpec(coordinate=(row, column), value=value)
                for row, column, value in entries
            ),
        )
        return cls(materialize_array_value(declaration), scale=scale, active=active)

    def effective_transition(self, transition: Array) -> Array:
        """Return the transition matrix with this structural change applied."""
        transition = jnp.asarray(transition, dtype=self.delta_A.dtype)
        if transition.shape != self.delta_A.shape:
            raise ValueError("delta_A must have the same shape as the transition matrix.")
        delta_A = jax.lax.cond(
            self.active,
            lambda: self.scale * self.delta_A,
            lambda: jnp.zeros_like(self.delta_A),
        )
        return transition + delta_A


def structural_linear_transition(
    transition: Array,
    state: Array,
    perturbation: StructuralLinearDynamicsPerturbation,
) -> Array:
    """Apply a structural linear transition without an additive input channel."""
    return perturbation.effective_transition(transition) @ state


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

    def _next_vector(
        self,
        current: LinearStateSpaceState,
        force: Array,
        epsilon: Array,
        *,
        transition: Array,
    ) -> Array:
        return transition @ current.vector + self.B @ force + self.B_w @ epsilon

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

        vector = self._next_vector(current, force, epsilon, transition=self.A)
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


class StructuralLinearStateSpace(LinearStateSpace):
    """Discrete linear mechanics with a task-selectable structural ``delta_A``.

    Unlike force-port interventions, the perturbation changes the matrix that
    advances the state. Parameters live in an Equinox ``StateIndex`` so an
    existing task can select one constant signed, nominal, or zero variant per
    trial through ``TaskTrialSpec.intervene``.
    """

    structural_params_index: StateIndex
    initial_delta_A: tuple[tuple[float, ...], ...] = field(static=True)
    initial_delta_A_value_spec: ArrayValueSpec | None = field(static=True)
    initial_scale: float = field(static=True)
    initial_active: bool = field(static=True)
    label: str = field(static=True)

    def __init__(
        self,
        A: Array,
        B: Array,
        *,
        delta_A: Array,
        authored_delta_A_value_spec: ArrayValueSpec | None = None,
        B_w: Array | None = None,
        dt: float = 1.0,
        initial_state: Array | None = None,
        pos_slice: tuple[int, int] = (0, 2),
        vel_slice: tuple[int, int] = (2, 4),
        scale: float | Array = 1.0,
        active: bool | Array = False,
        label: str = "structural_linear_dynamics",
    ):
        super().__init__(
            A=A,
            B=B,
            B_w=B_w,
            dt=dt,
            initial_state=initial_state,
            pos_slice=pos_slice,
            vel_slice=vel_slice,
        )
        params = StructuralLinearDynamicsPerturbation(
            delta_A=jnp.asarray(delta_A, dtype=self.A.dtype),
            scale=scale,
            active=active,
        )
        if params.delta_A.shape != self.A.shape:
            raise ValueError("delta_A must have the same shape as A.")
        self.initial_delta_A = tuple(
            tuple(float(value) for value in row)
            for row in params.delta_A.tolist()
        )
        self.initial_delta_A_value_spec = authored_delta_A_value_spec
        self.initial_scale = float(params.scale)
        self.initial_active = bool(params.active)
        self.structural_params_index = StateIndex(params)
        self.label = str(label)

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        del key
        current: LinearStateSpaceState = state.get(self.state_index)
        params: StructuralLinearDynamicsPerturbation = state.get(
            self.structural_params_index
        )
        force = jnp.asarray(inputs["force"], dtype=self.A.dtype)
        epsilon = jnp.asarray(
            inputs.get("epsilon", jnp.zeros((self.B_w.shape[1],), dtype=self.A.dtype)),
            dtype=self.A.dtype,
        )
        vector = (
            structural_linear_transition(self.A, current.vector, params)
            + self.B @ force
            + self.B_w @ epsilon
        )
        next_state = LinearStateSpaceState(vector=vector)
        state = state.set(self.state_index, next_state)
        return {
            "effector": self._effector(vector, force),
            "state": vector,
        }, state

    def task_parameter_state_indices(self) -> dict[str, StateIndex]:
        """Expose the trial-constant structural parameters under ``label``."""
        return {self.label: self.structural_params_index}
