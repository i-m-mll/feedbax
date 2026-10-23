"""AcausalSystem: bridge between acausal elements and the feedbax DAE framework.

An ``AcausalSystem`` is a ``DAEComponent`` whose vector field is produced by
the assembly algorithm at ``__init__`` time.  At runtime it behaves exactly
like any other ``Component`` in a feedbax ``Graph``.

:copyright: Copyright 2024-2025 by MLL <mll@mll.bio>.
:license: Apache 2.0.  See LICENSE for details.
"""

from __future__ import annotations

from collections.abc import Mapping
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

from feedbax.runtime.graph import Component
from feedbax.mechanics.dae import DAEComponent, DAEState
from feedbax.acausal.assembly import assemble_system
from feedbax.acausal.base import (
    AcausalConnection,
    AcausalElement,
    StateLayout,
)




# ---------------------------------------------------------------------------
# State / Params
# ---------------------------------------------------------------------------

class AcausalSystemState(Module):
    """Flat state vector for an assembled acausal system."""
    y: Float[Array, " n"]


class AcausalParams(Module):
    """Named parameter array for an acausal system.

    Attributes:
        values: Mapping of parameter names to scalar arrays.
    """
    values: dict[str, Array]

    def get(self, name: str) -> Scalar:
        """Look up a parameter value by name."""
        return self.values[name]


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
    _compiled_sensor_fn: Callable = field(static=True)
    _output_indices: dict[str, int] = field(static=True)
    _input_bindings: dict[str, str] = field(static=True)
    _output_bindings: dict[str, str] = field(static=True)
    _input_size_val: int = field(static=True)

    def __init__(
        self,
        elements: dict[str, AcausalElement],
        connections: list[AcausalConnection],
        dt: float = 0.01,
        solver_type: Type[dfx.AbstractSolver] = dfx.Euler,
        root_finder: Optional[optx.AbstractRootFinder] = None,
        *,
        input_bindings: Mapping[str, str] | None = None,
        output_bindings: Mapping[str, str] | None = None,
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
            input_bindings: Optional mapping from public causal input port
                names to source element names in the assembled acausal graph.
            output_bindings: Optional mapping from public causal output port
                names to sensor element names in the assembled acausal graph.
            key: PRNG key (passed through to ``DAEComponent``).
        """

        # ---- Assembly ----------------------------------------------------
        layout, vf_fn, params_dict, sensor_fn = assemble_system(
            elements, connections
        )

        self._layout = layout
        self._compiled_vf = vf_fn
        self._compiled_sensor_fn = sensor_fn

        # ---- Parameters --------------------------------------------------
        param_values = {
            name: jnp.zeros(()) + value for name, value in params_dict.items()
        }
        self.params = AcausalParams(values=param_values)

        # ---- Ports -------------------------------------------------------
        element_input_names: list[str] = []
        for vname, _idx in sorted(layout._inputs.items(), key=lambda item: item[1]):
            label = vname.split(".")[0]
            if label not in element_input_names:
                element_input_names.append(label)

        if input_bindings is None:
            public_input_bindings = {name: name for name in element_input_names}
            include_default_input_port = not public_input_bindings
        else:
            public_input_bindings = dict(input_bindings)
            include_default_input_port = False
            unknown_inputs = set(public_input_bindings.values()) - set(element_input_names)
            if unknown_inputs:
                raise ValueError(
                    "AcausalSystem input bindings reference unknown source elements: "
                    f"{sorted(unknown_inputs)!r}"
                )
        if output_bindings is None:
            public_output_bindings = {name: name for name in sorted(layout._outputs)}
            include_state_port = True
        else:
            public_output_bindings = dict(output_bindings)
            unknown_outputs = set(public_output_bindings.values()) - set(layout._outputs)
            if unknown_outputs:
                raise ValueError(
                    "AcausalSystem output bindings reference unknown sensor elements: "
                    f"{sorted(unknown_outputs)!r}"
                )
            include_state_port = False

        self._input_bindings = public_input_bindings
        self._output_bindings = public_output_bindings
        self.input_ports = (
            tuple(public_input_bindings)
            if public_input_bindings or not include_default_input_port
            else ("input",)
        )

        output_port_names = tuple(public_output_bindings)
        self.output_ports = (
            output_port_names + ("state",) if include_state_port else output_port_names
        )

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
        element_to_public_input = {
            element_name: port_name
            for port_name, element_name in self._input_bindings.items()
        }
        if n_inputs > 0:
            input_val = jnp.zeros((n_inputs,))
            for vname, idx in sorted(
                self._layout._inputs.items(), key=lambda item: item[1]
            ):
                elem_name, slot_idx = self._layout._input_specs[vname]
                public_port = element_to_public_input.get(elem_name, elem_name)
                val = inputs.get(public_port, None)
                if val is None:
                    val = inputs.get("input", jnp.zeros(1))
                arr = jnp.atleast_1d(jnp.asarray(val)).ravel()
                input_val = input_val.at[idx].set(arr[slot_idx])
        else:
            input_val = jnp.zeros(1)

        modified_inputs = {"input": input_val}
        outputs, state = super().__call__(modified_inputs, state, key=key)
        if "state" not in self.output_ports:
            outputs = {key: value for key, value in outputs.items() if key != "state"}

        dae_state = self.state_view(state)
        sensor_outputs = self._compiled_sensor_fn(
            dae_state.system.y, input_val, self.params.values
        )
        internal_outputs = {**outputs, **sensor_outputs}
        for public_port, sensor_element in self._output_bindings.items():
            if sensor_element in internal_outputs:
                outputs[public_port] = internal_outputs[sensor_element]
        if "state" not in self.output_ports:
            outputs = {
                port_name: outputs[port_name]
                for port_name in self.output_ports
                if port_name in outputs
            }
        else:
            outputs.update(sensor_outputs)
        return outputs, state

    def _get_zero_input(self) -> Array:
        """Zero-valued input for solver initialisation."""
        return jnp.zeros(max(len(self._layout._inputs), 1))
