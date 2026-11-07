"""Components that add forces to a mechanics graph."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Optional, ClassVar, Annotated

from pydantic import BaseModel, ConfigDict, Field

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import field
from equinox.nn import State, StateIndex
from jaxtyping import Array, PRNGKeyArray, PyTree

from feedbax.component_registry.declarations import (
    ComponentConstructorPlan,
    MissingPrototypeInput,
    ParameterField,
)
from feedbax.component_registry.discovery import component
from feedbax.contracts.component import PortType, PortTypeSpec
from feedbax.contracts.migrations import ComponentMigration
from feedbax.mechanics.units import require_positive_finite
from feedbax.runtime.graph import Component
from feedbax.contracts.portable import ConfigBinding, Ports, ComponentOptions, default_configuration
from feedbax.runtime.signal_transforms import (
    TransformParams,
    _bool_param,
    _float_param,
    _match_signal_dtype,
    _merge_params_override,
    _strong_typed,
)


class CurlFieldParams(TransformParams):
    amplitude: Array = field(default_factory=lambda: _float_param(1.0), converter=_float_param)


class FixedFieldParams(TransformParams):
    amplitude: Array = field(default_factory=lambda: _float_param(1.0), converter=_float_param)
    field: Array = field(default_factory=lambda: jnp.array([0.0, 0.0], dtype=jnp.float32))


class ThresholdLatchedForceParams(TransformParams):
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


class DynamicsMatrixPerturbParams(TransformParams):
    """Parameters for ``DynamicsMatrixPerturb``.

    The perturbation acts on the **velocity row** of the skeleton dynamics:
    ``dot v += delta_A @ [pos, vel]``. This is the natural force-channel
    embedding of a model-class ``ΔA`` perturbation, matching the disturbance
    structure used by ``FixedField`` / ``CurlField``
    (see ``rlrmp.analysis.hinf_riccati.linearize_pointmass``'s ``Bw_c``).

    Attributes:
        delta_A: ``(n_dim, 2 * n_dim)`` matrix mapping ``[pos, vel]`` to the
            velocity row's added derivative. Defaults to a 2D zero matrix.
    """

    delta_A: Array = field(default_factory=lambda: jnp.zeros((2, 4), dtype=jnp.float32))


def force_passthrough_output_prototype(
    _params: Mapping[str, Any], inputs: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the incoming force prototype for force-passthrough interventions."""
    if "force" not in inputs:
        raise MissingPrototypeInput(
            "force passthrough output prototype requires input prototype 'force'"
        )
    return {"force": inputs["force"]}


def _has_legacy_label(params: Mapping[str, Any]) -> bool:
    return "label" in params


def _drop_transform_label(params: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in params.items() if name != "label"}


def _drop_label_migration(component_id: str) -> tuple[ComponentMigration, ...]:
    return (
        ComponentMigration(
            source_type=component_id,
            target_type=component_id,
            owner="feedbax",
            source_param_schema_version="1",
            target_param_schema_version="2",
            migration_id=f"feedbax.component.{component_id}.drop-label",
            migrate_params=_drop_transform_label,
            description="Use the graph node identity in place of the legacy parameter label.",
        ),
    )


class CurlFieldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    scale: float = 1.0
    amplitude: float = 1.0
    active: bool = False


class CurlField(Component):
    """Velocity-dependent curl field added to a force signal."""

    config: ClassVar = CurlFieldConfig
    config_bindings: ClassVar = (
        ConfigBinding("params_index.init", group=CurlFieldParams, argument="params"),
    )
    ports: ClassVar = Ports(
        inputs={
            "effector": PortType(dtype="state"),
            "force": PortType(dtype="vector"),
            "params_override": PortType(dtype="object"),
        },
        outputs={"force": PortType(dtype="vector")},
    )
    component_options: ClassVar = ComponentOptions(
        output_prototype_fn=force_passthrough_output_prototype
    )

    params_index: StateIndex

    def __init__(self, params: Optional[CurlFieldParams] = None):
        if params is None:
            params = self.config_bindings[0].materialize(default_configuration(type(self)))
        self.params_index = StateIndex(_strong_typed(params))

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        params: CurlFieldParams = state.get(self.params_index)
        if "params_override" in inputs:
            params = _merge_params_override(params, inputs["params_override"])
        effector = inputs["effector"]
        force = inputs["force"]

        def apply_field():
            vel = effector.vel
            curl = params.amplitude * jnp.array([-1.0, 1.0]) * vel[..., ::-1]
            return _match_signal_dtype(force, force + params.scale * curl)

        new_force = jax.lax.cond(params.active, apply_field, lambda: force)
        return {"force": new_force}, state


@component(
    "FixedField",
    param_schema_version="2",
    migrations=_drop_label_migration("FixedField"),
    constructor=ComponentConstructorPlan(
        groups={
            "params": (
                FixedFieldParams,
                ("scale", "amplitude", "field", "active"),
            )
        },
        array_conversions={"field": None},
    ),
    attribute_paths={
        "scale": "params_index.init.scale",
        "amplitude": "params_index.init.amplitude",
        "field": "params_index.init.field",
        "active": "params_index.init.active",
    },
    category="Mechanics",
    description="Fixed force field.",
    parameter_fields=[
        ParameterField(name="scale", type="float", default=1.0, required=True),
        ParameterField(name="amplitude", type="float", default=1.0, required=True),
        ParameterField(name="field", type="array", default=[0.0, 0.0], required=True),
        ParameterField(name="active", type="bool", default=False, required=False),
    ],
    icon="Flag",
    port_types=PortTypeSpec(
        inputs={
            "force": PortType(dtype="vector"),
            "params_override": PortType(dtype="object"),
        },
        outputs={"force": PortType(dtype="vector")},
    ),
    output_prototype_fn=force_passthrough_output_prototype,
)
class FixedField(Component):
    """Adds a fixed force vector to a force signal."""

    input_ports = ("force", "params_override")
    output_ports = ("force",)

    params_index: StateIndex

    def __init__(self, params: Optional[FixedFieldParams] = None):
        if params is None:
            params = FixedFieldParams(active=False)
        self.params_index = StateIndex(_strong_typed(params))

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        params: FixedFieldParams = state.get(self.params_index)
        if "params_override" in inputs:
            params = _merge_params_override(params, inputs["params_override"])
        force = inputs["force"]

        def apply_field():
            return _match_signal_dtype(
                force, force + params.scale * params.amplitude * params.field
            )

        new_force = jax.lax.cond(params.active, apply_field, lambda: force)
        return {"force": new_force}, state


def _extract_declared_to_params(component: Any, model: Any) -> Any:
    return type(model).model_validate(component.to_params())


class ThresholdLatchedForceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    state_selector: dict[str, Any] = {"kind": "fixed", "path": ["pos", 0]}
    direction: Literal["increasing", "decreasing"] = "increasing"
    dt: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 0.01
    scale: float = 1.0
    active: bool = False
    threshold: float = 0.0
    force: list[float] = [0.0, 0.0]
    lateral_force: float = 0.0
    ramp_duration: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0


class ThresholdLatchedForce(Component):
    """Add force after a selected runtime state crosses a threshold.

    The crossing is evaluated from consecutive runtime samples. Once crossed,
    the intervention remains latched for the rest of that component state
    lifetime. A positive ramp duration linearly raises the force from zero at
    the crossing sample to full scale.

    A target-relative frame uses the first position and target, including
    when the intervention starts inactive. Activation requires a usable frame.
    """

    config: ClassVar = ThresholdLatchedForceConfig
    config_bindings: ClassVar = (
        ConfigBinding("params_index.init", group=ThresholdLatchedForceParams, argument="params"),
    )
    ports: ClassVar = Ports(
        inputs={
            "state": PortType(dtype="state"),
            "target": PortType(dtype="state"),
            "force": PortType(dtype="vector"),
            "params_override": PortType(dtype="object"),
        },
        outputs={"force": PortType(dtype="vector")},
    )
    component_options: ClassVar = ComponentOptions(
        output_prototype_fn=force_passthrough_output_prototype
    )

    params_index: StateIndex
    latch_index: StateIndex
    state_selector: ThresholdStateSelector = field(static=True)
    direction: Literal["increasing", "decreasing"] = field(static=True)
    dt: float = field(static=True)

    def __init__(
        self,
        *,
        state_selector: ThresholdStateSelector | Mapping[str, Any] | None = None,
        direction: Literal["increasing", "decreasing"] | None = None,
        dt: float | None = None,
        params: Optional[ThresholdLatchedForceParams] = None,
    ):
        authored = {}
        if state_selector is not None:
            authored["state_selector"] = _parse_threshold_state_selector(state_selector).to_param()
        if direction is not None:
            authored["direction"] = direction
        if dt is not None:
            authored["dt"] = dt
        config = default_configuration(type(self), **authored)
        state_selector, direction, dt = config.state_selector, config.direction, config.dt
        if params is None:
            params = self.config_bindings[0].materialize(config)
        params = eqx.tree_at(
            lambda item: item.ramp_duration,
            params,
            eqx.error_if(
                params.ramp_duration, params.ramp_duration < 0, "ramp_duration must be non-negative"
            ),
        )
        params = eqx.tree_at(lambda item: item.force, params, jnp.asarray(params.force))
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
            # Mask before the dot product so invalid frames have finite gradients.
            displacement = jnp.where(jnp.all(jnp.isfinite(displacement)), displacement, 0.0)
            distance_squared = jnp.dot(displacement, displacement)
            valid_distance = (distance_squared > 0) & jnp.isfinite(distance_squared)
            safe_distance = jnp.sqrt(jnp.where(valid_distance, distance_squared, 1.0))
            candidate_forward = jnp.where(valid_distance, displacement / safe_distance, 0.0)
            forward = jnp.where(latch.initialized, latch.forward, candidate_forward)
            forward = eqx.error_if(
                forward,
                params.active & ~jnp.any(forward),
                "PlanarTargetRelativeSelector requires distinct trial start and target "
                "with a finite distance",
            )
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
            crossed = (
                latch.initialized
                & (latch.previous < params.threshold)
                & (observed >= params.threshold)
            )
        else:
            crossed = (
                latch.initialized
                & (latch.previous > params.threshold)
                & (observed <= params.threshold)
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


@component(
    "DynamicsMatrixPerturb",
    param_schema_version="2",
    migrations=_drop_label_migration("DynamicsMatrixPerturb"),
    constructor=ComponentConstructorPlan(
        groups={
            "params": (
                DynamicsMatrixPerturbParams,
                ("scale", "delta_A", "active"),
            )
        },
        array_conversions={"delta_A": (2, 4)},
    ),
    attribute_paths={
        "scale": "params_index.init.scale",
        "delta_A": "params_index.init.delta_A",
        "active": "params_index.init.active",
    },
    category="Mechanics",
    description="State-feedback dynamics-matrix perturbation in the force channel.",
    parameter_fields=[
        ParameterField(name="scale", type="float", default=1.0, required=True),
        ParameterField(
            name="delta_A",
            type="array",
            default=[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            required=True,
        ),
        ParameterField(name="active", type="bool", default=False, required=False),
        ParameterField(
            name="mass",
            type="float",
            default=None,
            min=0.0,
            required=True,
            description="Effector mass in kg; must match the wired plant.",
        ),
    ],
    icon="Activity",
    port_types=PortTypeSpec(
        inputs={
            "effector": PortType(dtype="state"),
            "force": PortType(dtype="vector"),
            "params_override": PortType(dtype="object"),
        },
        outputs={"force": PortType(dtype="vector")},
    ),
    output_prototype_fn=force_passthrough_output_prototype,
)
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
    """

    input_ports = ("effector", "force", "params_override")
    output_ports = ("force",)

    params_index: StateIndex
    mass: float = field(static=True)

    def __init__(
        self,
        params: Optional[DynamicsMatrixPerturbParams] = None,
        mass: float | None = None,
    ):
        if mass is None:
            raise ValueError(
                "DynamicsMatrixPerturb requires an explicit positive mass matching "
                "the wired plant; pass mass=<plant mass> at graph construction."
            )
        if params is None:
            params = DynamicsMatrixPerturbParams(active=False)
        self.params_index = StateIndex(_strong_typed(params))
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
            return _match_signal_dtype(force, force + params.scale * df)

        new_force = jax.lax.cond(params.active, apply_perturbation, lambda: force)
        return {"force": new_force}, state
