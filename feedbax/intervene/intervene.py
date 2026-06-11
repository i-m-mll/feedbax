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
from jaxtyping import Array, PRNGKeyArray, PyTree

from feedbax.graph import Component
from feedbax.noise import Normal


def _float_param(value):
    return jnp.asarray(value, dtype=jnp.float32)


def _bool_param(value):
    return jnp.asarray(value, dtype=bool)


def _merge_params_override(
    base_params: "InterventionParams",
    override: "InterventionParams",
) -> "InterventionParams":
    """Merge override params into base, keeping base values where override has None.

    Override leaves take precedence. This allows per-step time-varying params
    to override the default State-stored params.
    """
    return jt.map(
        lambda o, b: o if o is not None else b,
        override,
        base_params,
        is_leaf=lambda x: x is None,
    )


class InterventionParams(eqx.Module):
    scale: Array = field(default_factory=lambda: _float_param(1.0), converter=_float_param)
    active: Array = field(default_factory=lambda: _bool_param(True), converter=_bool_param)


def _strong_typed(params: InterventionParams) -> InterventionParams:
    """Convert params to strong-typed JAX arrays for StateIndex storage.

    State.set() enforces exact dtype/weak-type matching between old and new
    values.  Trial-specific params from JAX random functions are strong-typed.
    To ensure compatibility, the StateIndex initial values must also be
    strong-typed.
    """
    def _convert(x):
        if isinstance(x, jnp.ndarray):
            return jnp.asarray(x, dtype=jnp.result_type(x))
        if isinstance(x, bool | int | float):
            return jnp.asarray(x, dtype=jnp.result_type(x))
        return x
    return jt.map(_convert, params)


class CurlFieldParams(InterventionParams):
    amplitude: Array = field(default_factory=lambda: _float_param(1.0), converter=_float_param)


class FixedFieldParams(InterventionParams):
    amplitude: Array = field(default_factory=lambda: _float_param(1.0), converter=_float_param)
    field: Array = field(default_factory=lambda: jnp.array([0.0, 0.0], dtype=jnp.float32))


class DynamicsMatrixPerturbParams(InterventionParams):
    """Parameters for ``DynamicsMatrixPerturb``.

    The perturbation acts on the **velocity row** of the skeleton dynamics:
    ``dot v += delta_A @ [pos, vel]``. This is the natural force-channel
    embedding of a model-class ``ΔA`` perturbation, matching the disturbance
    structure used by ``feedbax.intervene.FixedField`` / ``CurlField``
    (see ``rlrmp.analysis.hinf_riccati.linearize_pointmass``'s ``Bw_c``).

    Attributes:
        delta_A: ``(n_dim, 2 * n_dim)`` matrix mapping ``[pos, vel]`` to the
            velocity row's added derivative. Defaults to a 2D zero matrix.
    """

    delta_A: Array = field(
        default_factory=lambda: jnp.zeros((2, 4), dtype=jnp.float32)
    )


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

    input_ports = ("effector", "force", "params_override")
    output_ports = ("force",)

    params_index: StateIndex
    _initial_state: CurlFieldParams = field(static=True)
    label: str = field(default="curl_field", static=True)

    def __init__(self, params: Optional[CurlFieldParams] = None, label: str = "curl_field"):
        if params is None:
            params = CurlFieldParams(active=False)
        self._initial_state = params
        self.params_index = StateIndex(_strong_typed(params))
        self.label = label

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        params: CurlFieldParams = state.get(self.params_index)
        if "params_override" in inputs:
            params = _merge_params_override(params, inputs["params_override"])
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

    input_ports = ("force", "params_override")
    output_ports = ("force",)

    params_index: StateIndex
    _initial_state: FixedFieldParams = field(static=True)
    label: str = field(default="fixed_field", static=True)

    def __init__(self, params: Optional[FixedFieldParams] = None, label: str = "fixed_field"):
        if params is None:
            params = FixedFieldParams(active=False)
        self._initial_state = params
        self.params_index = StateIndex(_strong_typed(params))
        self.label = label

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        params: FixedFieldParams = state.get(self.params_index)
        if "params_override" in inputs:
            params = _merge_params_override(params, inputs["params_override"])
        force = inputs["force"]

        def apply_field():
            return force + params.scale * params.amplitude * params.field

        new_force = jax.lax.cond(params.active, apply_field, lambda: force)
        return {"force": new_force}, state

    def intervention_state_indices(self):
        return {self.label: self.params_index}


class DynamicsMatrixPerturb(Component):
    """Adds a state-feedback dynamics-matrix perturbation as a force.

    Implements the force-channel embedding of a model-class ``ΔA``
    perturbation: at each timestep, contributes an additional force
    ``f = mass * (delta_A @ [pos, vel])`` so that the velocity-row
    derivative becomes ``dot v += delta_A @ [pos, vel]`` (since the
    pointmass control matrix is ``B = [0; I/mass]``). This is the
    minimal lift that puts a ``ΔA`` perturbation in the same disturbance
    channel as ``FixedField`` / ``CurlField``, without modifying the
    feedbax dynamics scan body.

    Inserts at the same hook point as ``CurlField`` (between the
    force-filter or efferent and ``Mechanics``), reading the current
    effector state and writing to the force port. Bug: c723082 (rlrmp).

    Attributes:
        mass: Effector mass, used to convert the desired ``dx/dt``
            perturbation into a force. Should match the plant's mass
            (typically ``1.0``).
        label: Component label for parameter scheduling.
    """

    input_ports = ("effector", "force", "params_override")
    output_ports = ("force",)

    params_index: StateIndex
    _initial_state: DynamicsMatrixPerturbParams = field(static=True)
    label: str = field(default="dynamics_matrix_perturb", static=True)
    mass: float = field(default=1.0, static=True)

    def __init__(
        self,
        params: Optional[DynamicsMatrixPerturbParams] = None,
        label: str = "dynamics_matrix_perturb",
        mass: float = 1.0,
    ):
        if params is None:
            params = DynamicsMatrixPerturbParams(active=False)
        self._initial_state = params
        self.params_index = StateIndex(_strong_typed(params))
        self.label = label
        self.mass = mass

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        params: DynamicsMatrixPerturbParams = state.get(self.params_index)
        if "params_override" in inputs:
            params = _merge_params_override(params, inputs["params_override"])
        effector = inputs["effector"]
        force = inputs["force"]

        def apply_perturbation():
            # Concatenate [pos, vel] to form the kinematic state vector.
            x = jnp.concatenate([effector.pos, effector.vel], axis=-1)
            # f = mass * (delta_A @ x); dividing by mass via plant's B
            # converts this back into a velocity-row perturbation
            # dot v += delta_A @ x.
            df = self.mass * (params.delta_A @ x)
            return force + params.scale * df

        new_force = jax.lax.cond(params.active, apply_perturbation, lambda: force)
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
        self.params_index = StateIndex(_strong_typed(params))
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
        self.params_index = StateIndex(_strong_typed(params))
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
        self.params_index = StateIndex(_strong_typed(params))
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
        self.params_index = StateIndex(_strong_typed(params))
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
