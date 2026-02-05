"""AcausalSystem: bridge between acausal elements and the feedbax DAE framework.

An ``AcausalSystem`` is a ``DAEComponent`` whose vector field is produced by
the assembly algorithm at ``__init__`` time.  At runtime it behaves exactly
like any other ``Component`` in a feedbax ``Graph``.

:copyright: Copyright 2024-2025 by MLL <mll@mll.bio>.
:license: Apache 2.0.  See LICENSE for details.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Type

import diffrax as dfx
import equinox as eqx
from equinox import Module, field
from equinox.nn import State, StateIndex
import jax
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Array, Float, PRNGKeyArray, PyTree, Scalar
import optimistix as optx

from feedbax.graph import Component
from feedbax.mechanics.dae import DAEComponent, DAEState
from feedbax.acausal.assembly import assemble_system
from feedbax.acausal.base import (
    AcausalConnection,
    AcausalElement,
    StateLayout,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State / Params
# ---------------------------------------------------------------------------

class AcausalSystemState(Module):
    """Flat state vector for an assembled acausal system."""
    y: Float[Array, " n"]


class AcausalParams(Module):
    """Named parameter array for an acausal system.

    Attributes:
        values: 1-D array of parameter scalars.
        _names: Parameter names in the same order as ``values``.
    """
    values: Float[Array, " n_params"]
    _names: tuple[str, ...] = field(static=True)

    def get(self, name: str) -> Scalar:
        """Look up a parameter value by name."""
        idx = self._names.index(name)
        return self.values[idx]


# ---------------------------------------------------------------------------
# AcausalSystem
# ---------------------------------------------------------------------------

class AcausalSystem(DAEComponent[AcausalSystemState]):
    """Assembled acausal system as a standard DAE component.

    Construction-time:
        Takes element descriptors and connections, runs the assembly
        algorithm, and captures the resulting vector field and layout.

    Runtime:
        Standard ``DAEComponent`` execution through ``diffrax``.

    Attributes:
        params: Trainable physical parameters (mass, stiffness, ...).
    """

    # Override ClassVar with instance fields so each instance can differ.
    input_ports: tuple[str, ...]
    output_ports: tuple[str, ...]

    params: AcausalParams
    _layout: StateLayout = field(static=True)
    _compiled_vf: Callable = field(static=True)
    _output_indices: dict[str, int] = field(static=True)
    _input_size_val: int = field(static=True)

    def __init__(
        self,
        elements: dict[str, AcausalElement],
        connections: list[AcausalConnection],
        dt: float = 0.01,
        solver_type: Type[dfx.AbstractSolver] = dfx.Euler,
        root_finder: Optional[optx.AbstractRootFinder] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        """Assemble acausal elements into a DAE component.

        Args:
            elements: Named acausal element descriptors.
            connections: Port-level connections.
            dt: Integration time step.
            solver_type: ``diffrax`` solver class.
            root_finder: Optional ``optimistix`` root finder for implicit
                solvers.
            key: PRNG key (passed through to ``DAEComponent``).
        """

        # ---- Assembly ----------------------------------------------------
        layout, vf_fn, params_dict = assemble_system(elements, connections)

        self._layout = layout
        self._compiled_vf = vf_fn

        # ---- Parameters --------------------------------------------------
        param_names = tuple(sorted(params_dict.keys()))
        param_values = jnp.array([params_dict[k] for k in param_names])
        self.params = AcausalParams(values=param_values, _names=param_names)

        # ---- Ports -------------------------------------------------------
        # Input ports: one per causal-input variable (ForceSource, etc.)
        input_port_names: list[str] = []
        for vname in sorted(layout._inputs.keys()):
            # Use the element name as the port name for readability
            parts = vname.split(".")
            label = parts[0]  # element name
            if label not in input_port_names:
                input_port_names.append(label)
        self.input_ports = tuple(input_port_names) if input_port_names else ("input",)

        # Output ports: one per sensor + always "state"
        output_port_names = sorted(layout._outputs.keys())
        self.output_ports = tuple(output_port_names) + ("state",)

        # Output index map: sensor_label -> state-vector index
        out_idx: dict[str, int] = {}
        for label, var_fqn in layout._outputs.items():
            canon = layout.resolve(var_fqn)
            if canon in layout._differential:
                out_idx[label] = layout.var_index(canon)
        self._output_indices = out_idx

        self._input_size_val = max(len(layout._inputs), 1)

        # ---- DAEComponent base init -------------------------------------
        super().__init__(
            dt=dt,
            solver_type=solver_type,
            root_finder=root_finder,
            key=key,
        )

    # -- DAEComponent abstract method implementations ---------------------

    def vector_field(
        self,
        t: Scalar,
        state: AcausalSystemState,
        input_val: PyTree[Array],
    ) -> AcausalSystemState:
        """Compute dy/dt for the assembled system."""
        dy = self._compiled_vf(t, state.y, (input_val, self.params.values))
        return AcausalSystemState(y=dy)

    def init_system_state(self, *, key: PRNGKeyArray) -> AcausalSystemState:
        """Return a zero-initialised state vector."""
        y0 = jnp.zeros(self._layout.total_size)
        return AcausalSystemState(y=y0)

    def extract_outputs(self, state: AcausalSystemState) -> dict[str, PyTree]:
        """Read sensor values from the state vector."""
        outputs: dict[str, PyTree] = {}
        for label, idx in self._output_indices.items():
            outputs[label] = state.y[idx]
        return outputs

    @property
    def input_size(self) -> int:
        """Number of scalar causal inputs."""
        return self._input_size_val

    # -- Override __call__ for input routing --------------------------------

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        """Execute one integration step.

        Collects causal inputs into a flat array and delegates to the base
        ``DAEComponent.__call__``.
        """
        # Build a flat input array from the causal input dict
        n_inputs = len(self._layout._inputs)
        if n_inputs > 0:
            parts: list[Array] = []
            for vname in sorted(self._layout._inputs.keys()):
                elem_name = vname.split(".")[0]
                val = inputs.get(elem_name, None)
                if val is None:
                    val = inputs.get("input", jnp.zeros(1))
                parts.append(jnp.atleast_1d(jnp.asarray(val)).ravel())
            input_val = jnp.concatenate(parts)
        else:
            input_val = jnp.zeros(1)

        modified_inputs = {"input": input_val}
        return super().__call__(modified_inputs, state, key=key)

    def _get_zero_input(self) -> Array:
        """Zero-valued input for solver initialisation."""
        return jnp.zeros(max(len(self._layout._inputs), 1))
