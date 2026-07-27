"""Intervention components for graph-based models."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Optional

import equinox as eqx
from equinox import field
from equinox.nn import State, StateIndex
import jax
import jax.numpy as jnp
import jax.tree as jt
import jax_cookbook.tree as jtree
from jaxtyping import Array, PRNGKeyArray, PyTree

from feedbax.runtime.graph import Component
from feedbax.mechanics.units import require_positive_finite
from feedbax.runtime.noise import Normal


THRESHOLD_LATCHED_FORCE_SCHEMA_VERSION_V1 = "feedbax.component.threshold_latched_force.v1"
THRESHOLD_LATCHED_FORCE_SCHEMA_VERSION = "feedbax.component.threshold_latched_force.v2"


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


class ThresholdLatchedForceParams(InterventionParams):
    """Trial-specific parameters for ``ThresholdLatchedForce``."""

    threshold: Array = field(default_factory=lambda: _float_param(0.0), converter=_float_param)
    force: Array = field(default_factory=lambda: jnp.array([0.0, 0.0], dtype=jnp.float32))
    lateral_force: Array = field(
        default_factory=lambda: _float_param(0.0),
        converter=_float_param,
    )
    ramp_duration: Array = field(
        default_factory=lambda: _float_param(0.0),
        converter=_float_param,
    )


@dataclass(frozen=True)
class StateSelector:
    """A typed fixed-coordinate path selecting one scalar from runtime state."""

    path: tuple[str | int, ...]

    def __init__(self, path: Sequence[str | int]):
        normalized = tuple(path)
        if not normalized:
            raise ValueError("StateSelector.path must contain at least one segment")
        if any(not isinstance(segment, str | int) for segment in normalized):
            raise TypeError("StateSelector.path segments must be strings or integers")
        object.__setattr__(self, "path", normalized)

    @classmethod
    def from_param(cls, value: "StateSelector | Mapping[str, Any]") -> "StateSelector":
        """Parse a selector from its typed object or GraphSpec representation."""

        if isinstance(value, cls):
            return value
        if (
            not isinstance(value, Mapping)
            or value.get("kind", "fixed") != "fixed"
            or "path" not in value
        ):
            raise TypeError("state_selector must be a StateSelector or {'path': [...]} mapping")
        path = value["path"]
        if not isinstance(path, Sequence) or isinstance(path, str | bytes):
            raise TypeError("state_selector.path must be a sequence")
        return cls(path)

    def select(self, state: PyTree) -> Array:
        """Select and validate one scalar array from ``state``."""

        value = jnp.asarray(_select_path(state, self.path))
        if value.ndim != 0:
            raise ValueError(
                f"StateSelector path {self.path!r} must select a scalar; got shape {value.shape}"
            )
        return value

    def to_param(self) -> dict[str, Any]:
        """Return the stable GraphSpec representation."""

        return {"kind": "fixed", "path": list(self.path)}


@dataclass(frozen=True)
class PlanarTargetRelativeSelector:
    """Planar frame selected from runtime state and target.

    Positive lateral force points 90 degrees counter-clockwise from forward.
    """

    position_path: tuple[str | int, ...]
    target_path: tuple[str | int, ...]

    def __init__(
        self,
        position_path: Sequence[str | int],
        target_path: Sequence[str | int],
    ):
        object.__setattr__(self, "position_path", StateSelector(position_path).path)
        object.__setattr__(self, "target_path", StateSelector(target_path).path)

    @classmethod
    def from_param(
        cls,
        value: "PlanarTargetRelativeSelector | Mapping[str, Any]",
    ) -> "PlanarTargetRelativeSelector":
        """Parse the stable GraphSpec representation."""

        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping) or value.get("kind") != "planar_target_relative":
            raise TypeError(
                "state_selector must be a PlanarTargetRelativeSelector or "
                "a planar_target_relative mapping"
            )
        return cls(
            position_path=_selector_path_param(value, "position_path"),
            target_path=_selector_path_param(value, "target_path"),
        )

    def position(self, state: PyTree) -> Array:
        """Select a planar position from runtime state."""

        return _select_planar_path(state, self.position_path, name="position")

    def target(self, target: PyTree) -> Array:
        """Select a planar per-trial target from runtime reference data."""

        return _select_planar_path(target, self.target_path, name="target")

    def to_param(self) -> dict[str, Any]:
        """Return the stable GraphSpec representation."""

        return {
            "kind": "planar_target_relative",
            "position_path": list(self.position_path),
            "target_path": list(self.target_path),
        }


ThresholdStateSelector = StateSelector | PlanarTargetRelativeSelector


def _selector_path_param(value: Mapping[str, Any], name: str) -> Sequence[str | int]:
    path = value.get(name)
    if not isinstance(path, Sequence) or isinstance(path, str | bytes):
        raise TypeError(f"state_selector.{name} must be a sequence")
    return path


def _select_path(tree: PyTree, path: Sequence[str | int]) -> PyTree:
    value = tree
    for segment in path:
        if isinstance(segment, int):
            value = value[segment]
        elif isinstance(value, Mapping):
            value = value[segment]
        else:
            value = getattr(value, segment)
    return value


def _select_planar_path(
    tree: PyTree,
    path: Sequence[str | int],
    *,
    name: str,
) -> Array:
    value = jnp.asarray(_select_path(tree, path))
    if value.shape != (2,):
        raise ValueError(
            f"PlanarTargetRelativeSelector {name}_path {tuple(path)!r} must select "
            f"shape (2,); got {value.shape}"
        )
    return value


def _parse_threshold_state_selector(
    value: ThresholdStateSelector | Mapping[str, Any],
) -> ThresholdStateSelector:
    if isinstance(value, StateSelector | PlanarTargetRelativeSelector):
        return value
    if isinstance(value, Mapping) and value.get("kind", "fixed") == "fixed":
        return StateSelector.from_param(value)
    return PlanarTargetRelativeSelector.from_param(value)


class ThresholdLatchState(eqx.Module):
    """Runtime state for one threshold latch."""

    previous: Array
    initialized: Array
    latched: Array
    elapsed: Array
    origin: Array
    forward: Array


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

    delta_A: Array = field(default_factory=lambda: jnp.zeros((2, 4), dtype=jnp.float32))


class AddNoiseParams(InterventionParams): ...


class NetworkIntervenorParams(InterventionParams):
    unit_spec: Optional[PyTree] = None


class ConstantInputParams(InterventionParams):
    arrays: Optional[PyTree] = None


class CopyParams(InterventionParams): ...


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

    def task_parameter_state_indices(self):
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

    def task_parameter_state_indices(self):
        return {self.label: self.params_index}


class ThresholdLatchedForce(Component):
    """Add force after a selected runtime state crosses a threshold.

    The crossing is evaluated from consecutive runtime samples. Once crossed,
    the intervention remains latched for the rest of that component state
    lifetime. A positive ramp duration linearly raises the force from zero at
    the crossing sample to full scale.
    """

    input_ports = ("state", "target", "force", "params_override")
    output_ports = ("force",)

    params_index: StateIndex
    latch_index: StateIndex
    _initial_state: ThresholdLatchedForceParams = field(static=True)
    state_selector: ThresholdStateSelector = field(static=True)
    direction: Literal["increasing", "decreasing"] = field(static=True)
    dt: float = field(static=True)
    label: str = field(default="threshold_latched_force", static=True)

    def __init__(
        self,
        *,
        state_selector: ThresholdStateSelector | Mapping[str, Any],
        direction: Literal["increasing", "decreasing"],
        dt: float,
        params: Optional[ThresholdLatchedForceParams] = None,
        label: str = "threshold_latched_force",
    ):
        if direction not in ("increasing", "decreasing"):
            raise ValueError("direction must be 'increasing' or 'decreasing'")
        if params is None:
            params = ThresholdLatchedForceParams(active=False)
        if float(jnp.asarray(params.ramp_duration)) < 0:
            raise ValueError("ramp_duration must be non-negative")
        self._initial_state = params
        self.params_index = StateIndex(_strong_typed(params))
        self.latch_index = StateIndex(
            ThresholdLatchState(
                previous=_float_param(0.0),
                initialized=_bool_param(False),
                latched=_bool_param(False),
                elapsed=_float_param(0.0),
                origin=jnp.zeros((2,), dtype=jnp.float32),
                forward=jnp.zeros((2,), dtype=jnp.float32),
            )
        )
        self.state_selector = _parse_threshold_state_selector(state_selector)
        self.direction = direction
        self.dt = require_positive_finite("ThresholdLatchedForce.dt", dt)
        self.label = label

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        del key
        params: ThresholdLatchedForceParams = state.get(self.params_index)
        if "params_override" in inputs:
            params = _merge_params_override(params, inputs["params_override"])
        latch: ThresholdLatchState = state.get(self.latch_index)
        if isinstance(self.state_selector, PlanarTargetRelativeSelector):
            position = jnp.asarray(
                self.state_selector.position(inputs["state"]),
                dtype=latch.origin.dtype,
            )
            target = jnp.asarray(
                self.state_selector.target(inputs["target"]),
                dtype=latch.origin.dtype,
            )
            origin = jnp.where(latch.initialized, latch.origin, position)
            displacement = target - origin
            distance = jnp.linalg.norm(displacement)
            displacement = eqx.error_if(
                displacement,
                ~latch.initialized & (distance <= 0),
                "PlanarTargetRelativeSelector requires distinct trial start and target",
            )
            candidate_forward = displacement / jnp.where(distance > 0, distance, 1.0)
            forward = jnp.where(latch.initialized, latch.forward, candidate_forward)
            observed = jnp.dot(position - origin, forward)
            lateral = jnp.stack([-forward[1], forward[0]])
            applied_force = params.lateral_force * lateral
        else:
            observed = self.state_selector.select(inputs["state"])
            origin = latch.origin
            forward = latch.forward
            applied_force = params.force
        observed = jnp.asarray(observed, dtype=latch.previous.dtype)

        if self.direction == "increasing":
            crossed = latch.initialized & (latch.previous < params.threshold) & (
                observed >= params.threshold
            )
        else:
            crossed = latch.initialized & (latch.previous > params.threshold) & (
                observed <= params.threshold
            )
        newly_latched = params.active & ~latch.latched & crossed
        latched = latch.latched | newly_latched
        elapsed = jnp.where(
            newly_latched,
            _float_param(0.0),
            jnp.where(latched, latch.elapsed + self.dt, latch.elapsed),
        )
        ramp = jnp.where(
            params.ramp_duration > 0,
            jnp.minimum(elapsed / jnp.maximum(params.ramp_duration, self.dt), 1.0),
            _float_param(1.0),
        )
        applied = params.active & latched
        new_force = jnp.where(
            applied,
            inputs["force"] + params.scale * ramp * applied_force,
            inputs["force"],
        )
        state = state.set(
            self.latch_index,
            ThresholdLatchState(
                previous=observed,
                initialized=_bool_param(True),
                latched=latched,
                elapsed=elapsed,
                origin=origin,
                forward=forward,
            ),
        )
        return {"force": new_force}, state

    def task_parameter_state_indices(self):
        return {self.label: self.params_index}

    def to_params(self) -> dict[str, Any]:
        """Return the versioned GraphSpec parameters for this component."""

        return {
            "state_selector": self.state_selector.to_param(),
            "direction": self.direction,
            "threshold": float(self._initial_state.threshold),
            "force": jnp.asarray(self._initial_state.force).tolist(),
            "lateral_force": float(self._initial_state.lateral_force),
            "ramp_duration": float(self._initial_state.ramp_duration),
            "scale": float(self._initial_state.scale),
            "active": bool(self._initial_state.active),
            "dt": self.dt,
            "label": self.label,
        }


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
    mass: float = field(static=True)

    def __init__(
        self,
        params: Optional[DynamicsMatrixPerturbParams] = None,
        label: str = "dynamics_matrix_perturb",
        mass: float | None = None,
    ):
        if mass is None:
            raise ValueError(
                "DynamicsMatrixPerturb requires an explicit positive mass matching "
                "the wired plant; pass mass=<plant mass> at graph construction."
            )
        if params is None:
            params = DynamicsMatrixPerturbParams(active=False)
        self._initial_state = params
        self.params_index = StateIndex(_strong_typed(params))
        self.label = label
        self.mass = require_positive_finite("DynamicsMatrixPerturb.mass", mass)

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

    def task_parameter_state_indices(self):
        return {self.label: self.params_index}


class AddNoise(Component):
    """Adds noise to a signal."""

    input_ports = ("input",)
    output_ports = ("output",)

    noise_func: Callable[[PRNGKeyArray, Array], Array] = field(default_factory=Normal)
    params_index: StateIndex
    _initial_state: AddNoiseParams = field(static=True)
    label: str = field(default="add_noise", static=True)

    def __init__(
        self,
        noise_func: Optional[Callable[[PRNGKeyArray, Array], Array]] = None,
        params: Optional[AddNoiseParams] = None,
        label: str = "add_noise",
    ):
        if params is None:
            params = AddNoiseParams(active=False)
        if noise_func is None:
            noise_func = Normal()
        self.noise_func = noise_func
        self._initial_state = params
        self.params_index = StateIndex(_strong_typed(params))
        self.label = label

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        params: AddNoiseParams = state.get(self.params_index)
        signal = inputs["input"]

        def apply_noise():
            keys = jtree.random_split_like_tree(key, signal)
            noise = jt.map(lambda x, leaf_key: self.noise_func(leaf_key, x), signal, keys)
            return jt.map(lambda x, n: x + params.scale * n, signal, noise)

        output = jax.lax.cond(params.active, apply_noise, lambda: signal)
        return {"output": output}, state

    def task_parameter_state_indices(self):
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

    def task_parameter_state_indices(self):
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

    def task_parameter_state_indices(self):
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

    def task_parameter_state_indices(self):
        return {self.label: self.params_index}


class Copy(Component):
    """Pass-through component."""

    input_ports = ("input",)
    output_ports = ("output",)

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        return {"output": inputs["input"]}, state


def is_intervenor(element) -> bool:
    return isinstance(element, Component) and bool(element.task_parameter_state_indices())
