"""Components that transform signal values."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from math import isnan
from typing import Annotated, Any, ClassVar, Literal, Optional, Self, cast, get_args

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
from equinox import field
from equinox.nn import State, StateIndex
from jaxtyping import Array, PRNGKeyArray, PyTree
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator
from pydantic import Field as ConfigField

import feedbax._vendor.jax_cookbook.tree as jtree
from feedbax.contracts.component import PortType, PortTypeSpec
from feedbax.contracts.portable import (
    ComponentOptions,
    ConfigBinding,
    MissingPrototypeInput,
    Ports,
    default_configuration,
)
from feedbax.runtime.graph import Component
from feedbax.runtime.noise import Normal
from feedbax.runtime.prototypes import input_passthrough_output_prototype


def _float_param(value):
    return jnp.asarray(value, dtype=jnp.float32)


def _bool_param(value):
    return jnp.asarray(value, dtype=bool)


def _merge_params_override(
    base_params: "TransformParams",
    override: "TransformParams",
) -> "TransformParams":
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


class TransformParams(eqx.Module):
    scale: Array = field(default_factory=lambda: _float_param(1.0), converter=_float_param)
    active: Array = field(default_factory=lambda: _bool_param(True), converter=_bool_param)


def _strong_typed(params: TransformParams) -> TransformParams:
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


def _match_signal_dtype(signal, result):
    """Match the unchanged branch dtype for real floating signals."""
    dtype = getattr(signal, "dtype", None)
    if dtype is not None and dtype != result.dtype and jnp.issubdtype(dtype, jnp.floating):
        return result.astype(dtype)
    return result


class AddNoiseParams(TransformParams): ...


class NetworkValueParams(TransformParams):
    unit_spec: Optional[PyTree] = None


class ConstantInputParams(TransformParams):
    arrays: Optional[PyTree] = None


class _AddNoiseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    scale: float = ConfigField(default=1.0)
    active: bool = ConfigField(default=False)


class AddNoise(Component):
    """Adds noise to a signal."""

    config: ClassVar[type[BaseModel]] = _AddNoiseConfig
    config_bindings: ClassVar = (
        ConfigBinding("params_index.init", group=AddNoiseParams, argument="params"),
    )

    ports: ClassVar[Ports] = Ports(
        inputs={"input": PortType(dtype="any")}, outputs={"output": PortType(dtype="any")}
    )
    component_options: ClassVar[ComponentOptions] = ComponentOptions(
        output_prototype_fn=input_passthrough_output_prototype
    )

    noise_func: Callable[[PRNGKeyArray, Array], Array] = field(default_factory=Normal)
    params_index: StateIndex

    def __init__(
        self,
        noise_func: Optional[Callable[[PRNGKeyArray, Array], Array]] = None,
        params: Optional[AddNoiseParams] = None,
    ):
        if params is None:
            params = self.config_bindings[0].materialize(default_configuration(type(self)))
        if noise_func is None:
            noise_func = Normal()
        self.noise_func = noise_func
        self.params_index = StateIndex(_strong_typed(params))

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        params: AddNoiseParams = state.get(self.params_index)
        signal = inputs["input"]

        def apply_noise():
            keys = jtree.random_split_like_tree(key, signal)
            noise = jt.map(lambda x, leaf_key: self.noise_func(leaf_key, x), signal, keys)
            return jt.map(
                lambda x, n: _match_signal_dtype(x, x + params.scale * n),
                signal,
                noise,
            )

        output = jax.lax.cond(params.active, apply_noise, lambda: signal)
        return {"output": output}, state


_PayloadDtype = Literal["float16", "bfloat16", "float32", "float64"]


_FLOAT32_MAX = float(jnp.finfo(jnp.float32).max)


_InterventionScale = Annotated[FiniteFloat, Field(ge=-_FLOAT32_MAX, le=_FLOAT32_MAX)]


class _VectorTransformParams(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    payload_field: ClassVar[str]
    active: bool = False
    payload_dtype: _PayloadDtype = Field(
        default="float32", description="Floating-point dtype of the stored transform vector."
    )

    @model_validator(mode="after")
    def validate_payload(self) -> Self:
        payload = getattr(self, self.payload_field)
        if payload is None:
            if self.active:
                raise ValueError(f"{self.payload_field} is required when active is true")
        else:
            _validate_transform_vector_range(payload, jnp.dtype(self.payload_dtype))
        return self


class _NetworkClampParams(_VectorTransformParams):
    payload_field = "unit_spec"
    unit_spec: list[FiniteFloat | None] | None = Field(
        default=None, description="Vector of unit values. Null entries leave units unchanged."
    )


class _NetworkConstantInputParams(_NetworkClampParams):
    scale: _InterventionScale = 1.0


class _ConstantInputParams(_VectorTransformParams):
    payload_field = "arrays"
    scale: _InterventionScale = 1.0
    arrays: list[FiniteFloat] | None = Field(
        default=None, description="Vector to add to the signal."
    )


def _transform_payload_dtype(name: _PayloadDtype) -> jnp.dtype:
    dtype = jnp.dtype(name)
    if jax.dtypes.canonicalize_dtype(dtype) != dtype:
        raise ValueError(
            f"Intervention payload_dtype {name!r} is unavailable in this JAX configuration"
        )
    return dtype


def _validate_transform_vector_range(values: list[float | None], dtype: Any) -> None:
    limit = float(jnp.finfo(dtype).max)
    if any(value is not None and abs(value) > limit for value in values):
        raise ValueError(f"Intervention vector contains values outside the range of {dtype}")


def _vector_runtime_params(values: Mapping[str, Any], payload_field: str):
    params = dict(values)
    payload_dtype = params.pop("payload_dtype")
    payload = params[payload_field]
    if payload is not None:
        dtype = _transform_payload_dtype(payload_dtype)
        params[payload_field] = jnp.asarray(
            [float("nan") if value is None else value for value in payload], dtype=dtype
        )
    parameter_type = ConstantInputParams if payload_field == "arrays" else NetworkValueParams
    return parameter_type(**params)


def _default_vector_params(cls):
    return _vector_runtime_params(
        default_configuration(cls).model_dump(mode="python"), cls.config.payload_field
    )


def _materialize_vector_intervention(values, *, runtime_type, payload_field):
    return runtime_type(params=_vector_runtime_params(values, payload_field))


def _vector_intervention_configuration(
    component: Component,
    model: BaseModel,
    *,
    payload_field: str,
) -> BaseModel:
    index = cast(_VectorTransform, component).params_index
    if not isinstance(index.init, (ConstantInputParams, NetworkValueParams)):
        raise ValueError(
            f"{type(component).__name__} reset payload was stripped by make_with_state; preserve initial payloads for export"
        )
    params = cast(ConstantInputParams | NetworkValueParams, index.init)
    payload = getattr(params, payload_field)
    values: dict[str, Any] = {
        "active": params.active,
        payload_field: None,
        "payload_dtype": type(model).model_fields["payload_dtype"].default,
    }
    if "scale" in type(model).model_fields:
        values["scale"] = params.scale
    if payload is not None:
        if len(getattr(payload, "shape", ())) != 1:
            raise ValueError(f"{type(component).__name__} {payload_field} must be a single vector")
        payload_dtype = payload.dtype.name
        if payload_dtype not in get_args(_PayloadDtype):
            raise ValueError(
                f"{type(component).__name__} {payload_field} has unsupported GraphSpec dtype "
                f"{payload_dtype!r}. Convert the payload to a supported real floating dtype."
            )
        _transform_payload_dtype(payload_dtype)
        values["payload_dtype"] = payload_dtype
        vector = payload if isinstance(payload, jax.core.Tracer) else payload.tolist()
        if payload_field == "unit_spec" and not isinstance(payload, jax.core.Tracer):
            vector = [
                None if isinstance(value, float) and isnan(value) else value for value in vector
            ]
        values[payload_field] = vector
    return values


def _vector_transform_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
    *,
    component: str,
    payload_field: str,
) -> dict[str, Any]:
    if "input" not in inputs:
        raise MissingPrototypeInput(
            f"{component} output prototype requires input prototype 'input'"
        )
    signal = inputs["input"]
    payload = params.get(payload_field)
    if payload is not None:
        shape = getattr(signal, "shape", ())
        if len(shape) != 1 or not jnp.issubdtype(signal.dtype, jnp.floating):
            raise ValueError(f"{component} {payload_field} requires a real floating input vector")
        if len(payload) not in (1, shape[0]):
            raise ValueError(
                f"{component} {payload_field} length {len(payload)} cannot broadcast "
                f"to input vector shape {shape}"
            )
        if component == "NetworkClamp":
            _validate_transform_vector_range(payload, signal.dtype)
    return {"output": signal}


_VECTOR_PORT_TYPES = PortTypeSpec(
    inputs={"input": PortType(dtype="vector")},
    outputs={"output": PortType(dtype="vector")},
)


class NetworkClamp(Component):
    """Clamps units to specified values (NaN means unchanged)."""

    ports: ClassVar[Ports] = Ports(
        inputs=_VECTOR_PORT_TYPES.inputs, outputs=_VECTOR_PORT_TYPES.outputs
    )
    component_options: ClassVar[ComponentOptions] = ComponentOptions(
        output_prototype_fn=partial(
            _vector_transform_output_prototype,
            component="NetworkClamp",
            payload_field=_NetworkClampParams.payload_field,
        )
    )
    config: ClassVar[type[BaseModel]] = _NetworkClampParams

    params_index: StateIndex

    def __init__(
        self,
        params: Optional[NetworkValueParams] = None,
    ):
        if params is None:
            params = _default_vector_params(type(self))
        self.params_index = StateIndex(_strong_typed(params))

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        params: NetworkValueParams = state.get(self.params_index)
        signal = inputs["input"]

        def apply():
            if params.unit_spec is None:
                return signal
            return jt.map(
                lambda x, y: _match_signal_dtype(x, jnp.where(jnp.isnan(y), x, y)),
                signal,
                params.unit_spec,
            )

        output = jax.lax.cond(params.active, apply, lambda: signal)
        return {"output": output}, state

    @classmethod
    def from_config(cls, config, **context):
        return _materialize_vector_intervention(
            config.model_dump(mode="python"),
            runtime_type=cls,
            payload_field=cls.config.payload_field,
        )

    def configuration_values(self):
        return _vector_intervention_configuration(
            self, self.config.model_construct(), payload_field=self.config.payload_field
        )


class NetworkConstantInput(Component):
    """Adds a constant input to a signal."""

    ports: ClassVar[Ports] = Ports(
        inputs=_VECTOR_PORT_TYPES.inputs, outputs=_VECTOR_PORT_TYPES.outputs
    )
    component_options: ClassVar[ComponentOptions] = ComponentOptions(
        output_prototype_fn=partial(
            _vector_transform_output_prototype,
            component="NetworkConstantInput",
            payload_field=_NetworkConstantInputParams.payload_field,
        )
    )
    config: ClassVar[type[BaseModel]] = _NetworkConstantInputParams

    params_index: StateIndex

    def __init__(
        self,
        params: Optional[NetworkValueParams] = None,
    ):
        if params is None:
            params = _default_vector_params(type(self))
        self.params_index = StateIndex(_strong_typed(params))

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        params: NetworkValueParams = state.get(self.params_index)
        signal = inputs["input"]

        def apply():
            if params.unit_spec is None:
                return signal
            return jt.map(
                lambda x, y: _match_signal_dtype(x, x + params.scale * jnp.nan_to_num(y)),
                signal,
                params.unit_spec,
            )

        output = jax.lax.cond(params.active, apply, lambda: signal)
        return {"output": output}, state

    @classmethod
    def from_config(cls, config, **context):
        return _materialize_vector_intervention(
            config.model_dump(mode="python"),
            runtime_type=cls,
            payload_field=cls.config.payload_field,
        )

    def configuration_values(self):
        return _vector_intervention_configuration(
            self, self.config.model_construct(), payload_field=self.config.payload_field
        )


class ConstantInput(Component):
    """Adds a constant array to a signal."""

    ports: ClassVar[Ports] = Ports(
        inputs=_VECTOR_PORT_TYPES.inputs, outputs=_VECTOR_PORT_TYPES.outputs
    )
    component_options: ClassVar[ComponentOptions] = ComponentOptions(
        output_prototype_fn=partial(
            _vector_transform_output_prototype,
            component="ConstantInput",
            payload_field=_ConstantInputParams.payload_field,
        )
    )
    config: ClassVar[type[BaseModel]] = _ConstantInputParams

    params_index: StateIndex

    def __init__(
        self,
        params: Optional[ConstantInputParams] = None,
    ):
        if params is None:
            params = _default_vector_params(type(self))
        self.params_index = StateIndex(_strong_typed(params))

    def __call__(self, inputs: dict[str, PyTree], state: State, *, key: PRNGKeyArray):
        params: ConstantInputParams = state.get(self.params_index)
        return {"output": self._apply(inputs["input"], params)}, state

    @staticmethod
    def _apply(signal: PyTree, params: ConstantInputParams) -> PyTree:
        def apply():
            if params.arrays is None:
                return signal
            return jt.map(
                lambda x, y: _match_signal_dtype(x, x + params.scale * y),
                signal,
                params.arrays,
            )

        return jax.lax.cond(params.active, apply, lambda: signal)

    @classmethod
    def from_config(cls, config, **context):
        return _materialize_vector_intervention(
            config.model_dump(mode="python"),
            runtime_type=cls,
            payload_field=cls.config.payload_field,
        )

    def configuration_values(self):
        return _vector_intervention_configuration(
            self, self.config.model_construct(), payload_field=self.config.payload_field
        )


_VectorTransform = NetworkClamp | NetworkConstantInput | ConstantInput
