"""GraphSpec declarations for vector transform components."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from math import isnan
from typing import Annotated, Any, ClassVar, Literal, Protocol, Self, cast, get_args

import jax
import jax.numpy as jnp
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from feedbax.contracts.component import PortType, PortTypeSpec
from feedbax.contracts.migrations import ComponentMigration
from feedbax.runtime.signal_transforms import (
    ConstantInput,
    ConstantInputParams,
    NetworkClamp,
    NetworkConstantInput,
    NetworkValueParams,
)
from feedbax.runtime.graph import Component

from .declarations import DeclaredComponent, MissingPrototypeInput, declare_component


class _Registry(Protocol):
    def register(self, meta: DeclaredComponent) -> None: ...

    def register_migration(self, migration: ComponentMigration) -> None: ...


_VectorTransform = NetworkClamp | NetworkConstantInput | ConstantInput
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


def _without_legacy_clamp_scale(params: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the unused scale field from v1 and unversioned Studio clamp nodes."""
    return {name: value for name, value in params.items() if name != "scale"}


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


def _build_vector_intervention(
    values: Mapping[str, Any],
    *,
    runtime_type: type[_VectorTransform],
    payload_field: str,
) -> _VectorTransform:
    params = dict(values)
    payload_dtype = params.pop("payload_dtype")
    payload = params[payload_field]
    if payload is not None:
        dtype = _transform_payload_dtype(payload_dtype)
        params[payload_field] = jnp.asarray(
            [float("nan") if value is None else value for value in payload], dtype=dtype
        )
    if issubclass(runtime_type, ConstantInput):
        return runtime_type(params=ConstantInputParams(**params))
    return runtime_type(params=NetworkValueParams(**params))


def _extract_vector_intervention(
    component: Component,
    model: BaseModel,
    *,
    payload_field: str,
) -> BaseModel:
    params = cast(
        ConstantInputParams | NetworkValueParams,
        cast(_VectorTransform, component).params_index.init,
    )
    payload = getattr(params, payload_field)
    values: dict[str, Any] = {"active": params.active.tolist(), payload_field: None}
    if "scale" in type(model).model_fields:
        values["scale"] = params.scale.tolist()
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
        vector = payload.tolist()
        if payload_field == "unit_spec":
            vector = [
                None if isinstance(value, float) and isnan(value) else value for value in vector
            ]
        values[payload_field] = vector
    return type(model).model_validate(values)


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


def register_vector_transforms(registry: _Registry) -> None:
    for runtime_type, model, description, icon in (
        (NetworkClamp, _NetworkClampParams, "Clamp network unit activity.", "Pin"),
        (
            NetworkConstantInput,
            _NetworkConstantInputParams,
            "Add constant input to network units.",
            "Asterisk",
        ),
        (ConstantInput, _ConstantInputParams, "Add a constant input to a signal.", "Minus"),
    ):
        name = runtime_type.__name__
        payload_field = model.payload_field
        legacy_projection = _without_legacy_clamp_scale if runtime_type is NetworkClamp else None
        registry.register(
            declare_component(
                name=name,
                runtime_type=runtime_type,
                param_model=model,
                param_projection=legacy_projection,
                builder=partial(
                    _build_vector_intervention,
                    runtime_type=runtime_type,
                    payload_field=payload_field,
                ),
                extractor=partial(_extract_vector_intervention, payload_field=payload_field),
                override_reason="Translate JSON vectors to and from transform state arrays.",
                category="Signal Processing",
                description=description,
                input_ports=["input"],
                output_ports=["output"],
                icon=icon,
                port_types=PortTypeSpec(
                    inputs={"input": PortType(dtype="vector")},
                    outputs={"output": PortType(dtype="vector")},
                ),
                output_prototype_fn=partial(
                    _vector_transform_output_prototype,
                    component=name,
                    payload_field=payload_field,
                ),
                param_schema_version="2",
            )
        )
        registry.register_migration(
            ComponentMigration(
                source_type=name,
                target_type=name,
                owner="feedbax",
                source_param_schema_version="1",
                target_param_schema_version="2",
                migration_id=f"feedbax.component.{name}.params.v1-to-v2",
                migrate_params=legacy_projection or dict,
                description="Preserve disabled transforms and require a payload when enabled.",
            )
        )
