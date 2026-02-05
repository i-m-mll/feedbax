"""Differential-Algebraic Equation (DAE) components for physics simulation.

DAE systems arise in mechanics when dealing with:
- Constrained dynamics (e.g., muscle tendon constraints)
- Stiff systems (e.g., compliant tendons with fast dynamics)
- Implicit force-balance equations

This module provides base classes for DAE-based mechanical components
using diffrax implicit solvers and optimistix root finders.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
import logging
from typing import ClassVar, Optional, Type, TypeVar, Generic

import diffrax as dfx
import equinox as eqx
from equinox import Module, field
from equinox.nn import State, StateIndex
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree, Scalar
import optimistix as optx

from feedbax.graph import Component


logger = logging.getLogger(__name__)


StateT = TypeVar("StateT", bound=Module)


class DAEState(Module, Generic[StateT]):
    """State container for DAE component integration.

    Attributes:
        system: The underlying system state (e.g., skeleton, muscle state).
        solver: The diffrax solver state for implicit integration.
    """

    system: StateT
    solver: PyTree


class DAEComponent(Component, Generic[StateT]):
    """Base class for components that use implicit/DAE integration.

    DAE components are used when the dynamics are stiff or involve
    algebraic constraints. They use diffrax implicit solvers with
    optimistix root finders for numerical stability.

    Subclasses must implement:
        - `vector_field`: The ODE/DAE right-hand side
        - `init_system_state`: Initialize the system-specific state
        - `extract_outputs`: Extract output dict from system state

    Attributes:
        dt: Integration timestep.
        solver: The implicit diffrax solver instance.
        root_finder: The optimistix root finder for implicit steps.
    """

    input_ports: ClassVar[tuple[str, ...]] = ("input",)
    output_ports: ClassVar[tuple[str, ...]] = ("state",)

    dt: float
    solver: dfx.AbstractSolver = field(static=True)
    root_finder: Optional[optx.AbstractRootFinder] = field(static=True)
    state_index: StateIndex
    _initial_state: DAEState[StateT] = field(static=True)

    def __init__(
        self,
        dt: float,
        solver_type: Type[dfx.AbstractSolver] = dfx.Euler,
        root_finder: Optional[optx.AbstractRootFinder] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        """Initialize DAE component.

        Args:
            dt: Integration timestep.
            solver_type: Type of solver to use. Can be explicit (Euler, etc.)
                or implicit (ImplicitEuler, Kvaerno3, etc.).
            root_finder: Root finder for implicit steps. Only used for implicit solvers.
            key: PRNG key for initialization.
        """
        self.dt = dt

        # Check if this is an implicit solver that needs a root finder
        is_implicit = issubclass(solver_type, dfx.AbstractImplicitSolver)

        if is_implicit:
            if root_finder is None:
                root_finder = optx.Newton(rtol=1e-8, atol=1e-8)
            self.root_finder = root_finder
            # Type narrowing doesn't work with issubclass, but we've verified it's implicit
            self.solver = solver_type(root_finder=self.root_finder)  # type: ignore[call-arg]
        else:
            self.root_finder = None
            self.solver = solver_type()

        if key is None:
            key = jax.random.PRNGKey(0)

        # Initialize system state (subclass-specific)
        system_state = self.init_system_state(key=key)

        # Force strong dtypes to avoid weak dtype issues with StateIndex
        import jax.tree as jt
        def make_strong_dtype(x):
            if isinstance(x, jax.Array):
                # Convert to numpy and back to remove weak type
                return jnp.asarray(x.astype(x.dtype))
            return x
        system_state = jt.map(make_strong_dtype, system_state)

        # Initialize solver state
        init_input = self._get_zero_input()
        # Also canonicalize input dtypes
        init_input = jt.map(make_strong_dtype, init_input)

        solver_state = self.solver.init(
            self._term, 0.0, self.dt, system_state, init_input
        )

        self._initial_state = DAEState(system=system_state, solver=solver_state)
        self.state_index = StateIndex(self._initial_state)

    @cached_property
    def _term(self) -> dfx.ODETerm:
        """Create the diffrax ODE term from the vector field."""
        # Type mismatch: diffrax uses RealScalarLike, we use jaxtyping.Scalar
        return dfx.ODETerm(self.vector_field)  # type: ignore[arg-type]

    @abstractmethod
    def vector_field(
        self,
        t: Scalar,
        state: StateT,
        input: PyTree[Array],
    ) -> StateT:
        """Compute the time derivative of the system state.

        This defines the dynamics of the DAE system. For pure ODE systems,
        this is just f(t, y, u). For DAE systems, the implicit solver
        handles the algebraic constraints.

        Args:
            t: Current time.
            state: Current system state.
            input: External input (e.g., forces, muscle activations).

        Returns:
            Time derivative of the state (same structure as state).
        """
        ...

    @abstractmethod
    def init_system_state(self, *, key: PRNGKeyArray) -> StateT:
        """Initialize the system-specific state.

        Args:
            key: PRNG key for any random initialization.

        Returns:
            Initial system state.
        """
        ...

    @abstractmethod
    def extract_outputs(self, state: StateT) -> dict[str, PyTree]:
        """Extract output dictionary from system state.

        Args:
            state: Current system state.

        Returns:
            Dictionary mapping output port names to values.
        """
        ...

    def _get_zero_input(self) -> PyTree[Array]:
        """Get a zero-valued input for solver initialization.

        Override if input structure is complex.
        """
        return jnp.zeros((self.input_size,))

    @property
    @abstractmethod
    def input_size(self) -> int:
        """Number of scalar input values."""
        ...

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        """Execute one DAE integration step.

        Args:
            inputs: Dictionary of input port values.
            state: Current equinox State container.
            key: PRNG key (unused, for API compatibility).

        Returns:
            Tuple of (output dict, updated State).
        """
        dae_state: DAEState[StateT] = state.get(self.state_index)
        input_value = inputs.get("input", self._get_zero_input())

        # Perform implicit integration step
        new_system_state, _, _, new_solver_state, _ = self.solver.step(
            self._term,
            0.0,
            self.dt,
            dae_state.system,
            input_value,
            dae_state.solver,
            made_jump=False,
        )

        # Update state
        new_dae_state = DAEState(system=new_system_state, solver=new_solver_state)
        state = state.set(self.state_index, new_dae_state)

        # Extract outputs
        outputs = self.extract_outputs(new_system_state)
        outputs["state"] = new_system_state

        return outputs, state

    def state_view(self, state: State) -> DAEState[StateT]:
        """Return the DAE state from the State container."""
        return state.get(self.state_index)


class DAEParams(Module):
    """Base class for DAE component parameters.

    Subclasses should define fields for physical parameters
    (masses, lengths, stiffnesses, etc.).
    """

    pass
