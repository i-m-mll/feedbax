"""Intervention components for graph-based models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import equinox as eqx
from equinox import field
from equinox.nn import State, StateIndex
import jax
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Array, ArrayLike, PRNGKeyArray, PyTree

from feedbax.graph import Component
from feedbax.noise import Normal


class InterventionParams(eqx.Module):
    scale: float = 1.0
    active: bool = True


class CurlFieldParams(InterventionParams):
    amplitude: float = 1.0


class FixedFieldParams(InterventionParams):
    amplitude: float = 1.0
    field: Array = field(default_factory=lambda: jnp.array([0.0, 0.0]))


class AddNoiseParams(InterventionParams):
    ...


class NetworkIntervenorParams(InterventionParams):
    unit_spec: Optional[PyTree] = None


class ConstantInputParams(InterventionParams):
    arrays: Optional[PyTree] = None


class CopyParams(InterventionParams):
    ...


class CurlField(Component):
    """Velocity-dependent curl field added to a force signal."""

    input_ports = ("effector", "force")
    output_ports = ("force",)

    params_index: StateIndex
    _initial_state: CurlFieldParams = field(static=True)
    label: str = field(default="curl_field", static=True)

    def __init__(self, params: Optional[CurlFieldParams] = None, label: str = "curl_field"):
        if params is None:
            params = CurlFieldParams(active=False)
        self._initial_state = params
        self.params_index = StateIndex(params)
        self.label = label

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        params: CurlFieldParams = state.get(self.params_index)
        effector = inputs["effector"]
        force = inputs["force"]

        def apply_field():
            vel = effector.vel
            curl = params.amplitude * jnp.array([-1.0, 1.0]) * vel[..., ::-1]
            return force + params.scale * curl

        new_force = jax.lax.cond(params.active, apply_field, lambda: force)
        return {"force": new_force}, state

    def intervention_state_indices(self):
        return {self.label: self.params_index}


class FixedField(Component):
    """Adds a fixed force vector to a force signal."""

    input_ports = ("force",)
    output_ports = ("force",)

    params_index: StateIndex
    _initial_state: FixedFieldParams = field(static=True)
    label: str = field(default="fixed_field", static=True)

    def __init__(self, params: Optional[FixedFieldParams] = None, label: str = "fixed_field"):
        if params is None:
            params = FixedFieldParams(active=False)
        self._initial_state = params
        self.params_index = StateIndex(params)
        self.label = label

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        params: FixedFieldParams = state.get(self.params_index)
        force = inputs["force"]

        def apply_field():
            return force + params.scale * params.amplitude * params.field

        new_force = jax.lax.cond(params.active, apply_field, lambda: force)
        return {"force": new_force}, state

    def intervention_state_indices(self):
        return {self.label: self.params_index}


class AddNoise(Component):
    """Adds noise to a signal."""

    input_ports = ("input",)
    output_ports = ("output",)

    noise_func: Callable[[PRNGKeyArray, Array], Array] = Normal()
    params_index: StateIndex
    _initial_state: AddNoiseParams = field(static=True)
    label: str = field(default="add_noise", static=True)

    def __init__(
        self,
        noise_func: Callable[[PRNGKeyArray, Array], Array] = Normal(),
        params: Optional[AddNoiseParams] = None,
        label: str = "add_noise",
    ):
        if params is None:
            params = AddNoiseParams(active=False)
        self.noise_func = noise_func
        self._initial_state = params
        self.params_index = StateIndex(params)
        self.label = label

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        params: AddNoiseParams = state.get(self.params_index)
        signal = inputs["input"]

        def apply_noise():
            noise = jt.map(lambda x: self.noise_func(key, x), signal)
            return jt.map(lambda x, n: x + params.scale * n, signal, noise)

        output = jax.lax.cond(params.active, apply_noise, lambda: signal)
        return {"output": output}, state

    def intervention_state_indices(self):
        return {self.label: self.params_index}


class NetworkClamp(Component):
    """Clamps units to specified values (NaN means unchanged)."""

    input_ports = ("input",)
    output_ports = ("output",)

    params_index: StateIndex
    _initial_state: NetworkIntervenorParams = field(static=True)
    label: str = field(default="network_clamp", static=True)

    def __init__(
        self,
        params: Optional[NetworkIntervenorParams] = None,
        label: str = "network_clamp",
    ):
        if params is None:
            params = NetworkIntervenorParams(active=False)
        self._initial_state = params
        self.params_index = StateIndex(params)
        self.label = label

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        params: NetworkIntervenorParams = state.get(self.params_index)
        signal = inputs["input"]

        def apply():
            if params.unit_spec is None:
                return signal
            return jt.map(
                lambda x, y: jnp.where(jnp.isnan(y), x, y),
                signal,
                params.unit_spec,
            )

        output = jax.lax.cond(params.active, apply, lambda: signal)
        return {"output": output}, state

    def intervention_state_indices(self):
        return {self.label: self.params_index}


class NetworkConstantInput(Component):
    """Adds a constant input to a signal."""

    input_ports = ("input",)
    output_ports = ("output",)

    params_index: StateIndex
    _initial_state: NetworkIntervenorParams = field(static=True)
    label: str = field(default="network_constant_input", static=True)

    def __init__(
        self,
        params: Optional[NetworkIntervenorParams] = None,
        label: str = "network_constant_input",
    ):
        if params is None:
            params = NetworkIntervenorParams(active=False)
        self._initial_state = params
        self.params_index = StateIndex(params)
        self.label = label

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        params: NetworkIntervenorParams = state.get(self.params_index)
        signal = inputs["input"]

        def apply():
            if params.unit_spec is None:
                return signal
            return jt.map(
                lambda x, y: x + params.scale * jnp.nan_to_num(y),
                signal,
                params.unit_spec,
            )

        output = jax.lax.cond(params.active, apply, lambda: signal)
        return {"output": output}, state

    def intervention_state_indices(self):
        return {self.label: self.params_index}


class ConstantInput(Component):
    """Adds a constant array to a signal."""

    input_ports = ("input",)
    output_ports = ("output",)

    params_index: StateIndex
    _initial_state: ConstantInputParams = field(static=True)
    label: str = field(default="constant_input", static=True)

    def __init__(
        self,
        params: Optional[ConstantInputParams] = None,
        label: str = "constant_input",
    ):
        if params is None:
            params = ConstantInputParams(active=False)
        self._initial_state = params
        self.params_index = StateIndex(params)
        self.label = label

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        params: ConstantInputParams = state.get(self.params_index)
        signal = inputs["input"]

        def apply():
            if params.arrays is None:
                return signal
            return jt.map(lambda x, y: x + params.scale * y, signal, params.arrays)

        output = jax.lax.cond(params.active, apply, lambda: signal)
        return {"output": output}, state

    def intervention_state_indices(self):
        return {self.label: self.params_index}


class Copy(Component):
    """Pass-through component."""

    input_ports = ("input",)
    output_ports = ("output",)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        return {"output": inputs["input"]}, state


def is_intervenor(element) -> bool:
    return isinstance(element, Component) and bool(element.intervention_state_indices())
