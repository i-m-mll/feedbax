"""Discrete linear state-space mechanics component."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from equinox import Module, field
from equinox.nn import State, StateIndex
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree

from feedbax.component_registry import ParameterField, component
from feedbax.contracts.component import PortType, PortTypeSpec
from feedbax.contracts.migrations import ComponentMigration
from feedbax.runtime.graph import Component
from feedbax.runtime.state import CartesianState
from feedbax.contracts.array_values import (
    ARRAY_VALUE_SCHEMA_ID,
    ARRAY_VALUE_SCHEMA_VERSION,
    SparseCooArrayValueSpec,
    SparseCooEntrySpec,
    materialize_array_value,
)


STRUCTURAL_LINEAR_STATE_SPACE_PARAM_SCHEMA_VERSION_V1 = (
    "feedbax.component.structural_linear_state_space.v1"
)
STRUCTURAL_LINEAR_STATE_SPACE_PARAM_SCHEMA_VERSION = (
    "feedbax.component.structural_linear_state_space.v2"
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
        """Apply the change in the nominal floating-point matrix's dtype.

        When enabled, cast the stored scale and delta to the nominal dtype
        before multiplying and adding. When disabled, return the nominal
        matrix directly.
        """
        transition = jnp.asarray(transition)
        if transition.shape != self.delta_A.shape:
            raise ValueError("delta_A must have the same shape as the transition matrix.")
        if not jnp.issubdtype(transition.dtype, jnp.floating):
            raise TypeError("The nominal transition matrix must have a floating-point dtype.")

        def apply_change() -> Array:
            scale = self.scale.astype(transition.dtype)
            delta_A = self.delta_A.astype(transition.dtype)
            return transition + scale * delta_A

        return jax.lax.cond(self.active, apply_change, lambda: transition)


def structural_linear_transition(
    transition: Array,
    state: Array,
    perturbation: StructuralLinearDynamicsPerturbation,
) -> Array:
    """Apply a floating-point structural transition without an additive input channel."""
    return perturbation.effective_transition(transition) @ state


class LinearStateSpaceState(Module):
    """State for a discrete linear state-space mechanics component.

    Attributes:
        vector: Raw state vector, shape ``[n_state]``.
    """

    vector: Array


def linear_state_space_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return LinearStateSpace effector/state prototypes from matrices and slices."""

    a_matrix = jnp.asarray(params["A"])
    b_matrix = jnp.asarray(params["B"])
    state_dim = int(a_matrix.shape[0])
    pos_start, pos_stop = (int(value) for value in params.get("pos_slice", [0, 2]))
    vel_start, vel_stop = (int(value) for value in params.get("vel_slice", [2, 4]))
    force_dim = int(b_matrix.shape[1]) if b_matrix.ndim == 2 else 0
    return {
        "effector": CartesianState(
            pos=jnp.zeros((pos_stop - pos_start,), dtype=a_matrix.dtype),
            vel=jnp.zeros((vel_stop - vel_start,), dtype=a_matrix.dtype),
            force=jnp.zeros((force_dim,), dtype=a_matrix.dtype),
        ),
        "state": jnp.zeros((state_dim,), dtype=a_matrix.dtype),
    }


_LINEAR_MATRIX_FIELDS = [
    ParameterField(
        name="A",
        type="array",
        default=[
            [1.0, 0.0, 0.01, 0.0],
            [0.0, 1.0, 0.0, 0.01],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        required=True,
    ),
    ParameterField(
        name="B",
        type="array",
        default=[
            [0.0, 0.0],
            [0.0, 0.0],
            [0.01, 0.0],
            [0.0, 0.01],
        ],
        required=True,
    ),
    ParameterField(name="B_w", type="array", default=None, required=False),
]
_LINEAR_INITIALIZATION_FIELDS = [
    ParameterField(name="dt", type="float", default=0.01, min=0.0, required=False),
    ParameterField(
        name="initial_state",
        type="array",
        default=[0.0, 0.0, 0.0, 0.0],
        required=False,
    ),
    ParameterField(name="pos_slice", type="array", default=[0, 2], required=False),
    ParameterField(name="vel_slice", type="array", default=[2, 4], required=False),
]
_LINEAR_STATE_SPACE_PORT_TYPES = PortTypeSpec(
    inputs={
        "force": PortType(dtype="vector"),
        "epsilon": PortType(dtype="vector"),
    },
    outputs={
        "effector": PortType(dtype="state"),
        "state": PortType(dtype="vector"),
    },
)


@component(
    "LinearStateSpace",
    category="Mechanics",
    description="Discrete linear state-space mechanics.",
    parameter_fields=[*_LINEAR_MATRIX_FIELDS, *_LINEAR_INITIALIZATION_FIELDS],
    icon="Grid3x3",
    port_types=_LINEAR_STATE_SPACE_PORT_TYPES,
    output_prototype_fn=linear_state_space_output_prototype,
)
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


_DEFAULT_STRUCTURAL_DELTA_A = SparseCooArrayValueSpec(
    schema_id=ARRAY_VALUE_SCHEMA_ID,
    schema_version=ARRAY_VALUE_SCHEMA_VERSION,
    encoding="sparse_coo",
    shape=(4, 4),
    dtype="float32",
    nonfinite="forbid",
    fill=0.0,
    entries=(),
).model_dump(mode="json")


def _drop_structural_label(params: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in params.items() if name != "label"}


@component(
    "StructuralLinearStateSpace",
    category="Mechanics",
    description="Discrete linear mechanics with a trial-selectable structural delta_A.",
    parameter_fields=[
        *_LINEAR_MATRIX_FIELDS,
        ParameterField(
            name="delta_A",
            type="object",
            default=_DEFAULT_STRUCTURAL_DELTA_A,
            description="Versioned sparse COO or constant array declaration; dense square arrays remain accepted.",
            required=True,
        ),
        ParameterField(name="scale", type="float", default=1.0, required=False),
        ParameterField(name="active", type="bool", default=False, required=False),
        *_LINEAR_INITIALIZATION_FIELDS,
    ],
    icon="Grid3x3",
    port_types=_LINEAR_STATE_SPACE_PORT_TYPES,
    output_prototype_fn=linear_state_space_output_prototype,
    param_schema_version=STRUCTURAL_LINEAR_STATE_SPACE_PARAM_SCHEMA_VERSION,
    attribute_paths={
        "delta_A": "structural_params_index.init.delta_A",
        "scale": "structural_params_index.init.scale",
        "active": "structural_params_index.init.active",
    },
    migrations=(
        ComponentMigration(
            source_type="StructuralLinearStateSpace",
            target_type="StructuralLinearStateSpace",
            owner="feedbax",
            source_param_schema_version=STRUCTURAL_LINEAR_STATE_SPACE_PARAM_SCHEMA_VERSION_V1,
            target_param_schema_version=STRUCTURAL_LINEAR_STATE_SPACE_PARAM_SCHEMA_VERSION,
            migration_id="feedbax.component.StructuralLinearStateSpace.drop-label",
            migrate_params=_drop_structural_label,
            description="Use the graph node identity in place of the legacy parameter label.",
        ),
    ),
)
class StructuralLinearStateSpace(LinearStateSpace):
    """Discrete linear mechanics with a task-selectable structural ``delta_A``.

    Unlike force-port interventions, the perturbation changes the matrix that
    advances the state. Parameters live in an Equinox ``StateIndex`` so an
    task can select one constant signed, nominal, or zero variant in its
    trial initial state.
    """

    structural_params_index: StateIndex

    def __init__(
        self,
        A: Array,
        B: Array,
        *,
        delta_A: Array,
        B_w: Array | None = None,
        dt: float = 1.0,
        initial_state: Array | None = None,
        pos_slice: tuple[int, int] = (0, 2),
        vel_slice: tuple[int, int] = (2, 4),
        scale: float | Array = 1.0,
        active: bool | Array = False,
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
        self.structural_params_index = StateIndex(params)

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        del key
        current: LinearStateSpaceState = state.get(self.state_index)
        params: StructuralLinearDynamicsPerturbation = state.get(self.structural_params_index)
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
