"""Single-array stateful prototype parameter encoding."""

from typing import Any, Mapping

import jax.numpy as jnp

from feedbax.component_registry.declarations import MissingPrototypeInput


def array_prototype(params: Mapping[str, Any]) -> Any | None:
    shape = params.get("input_shape")
    if shape is None:
        return None
    if not isinstance(shape, (list, tuple)) or any(
        isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape
    ):
        raise ValueError("input_shape must be a list of nonnegative integer dimensions")
    return jnp.zeros(tuple(shape), dtype=params.get("input_dtype"))


def prototype_params(proto: Any) -> dict[str, Any]:
    if not hasattr(proto, "shape") or not hasattr(proto, "dtype"):
        raise ValueError("Only single-array prototypes are supported by input_shape/input_dtype")
    return {"input_shape": [int(dim) for dim in proto.shape], "input_dtype": str(proto.dtype)}


def migrate_stateful_v1(params: dict[str, Any]) -> dict[str, Any]:
    """Preserve declarations, leaving omitted dtype available for inference."""
    return dict(params)


def _missing_input(port: str, *, component: str) -> MissingPrototypeInput:
    return MissingPrototypeInput(f"{component} output prototype requires input prototype {port!r}")


def _required_input(inputs: Mapping[str, Any], port: str, *, component: str) -> Any:
    if port not in inputs:
        raise _missing_input(port, component=component)
    return inputs[port]


def input_passthrough_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the incoming ``input`` prototype as ``output``."""

    return {"output": _required_input(inputs, "input", component="input passthrough")}


def stateful_passthrough_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return input-shaped output, falling back to configured ``input_shape``."""

    proto = inputs.get("input")
    if proto is None:
        proto = array_prototype(params)
    if proto is None:
        raise _missing_input("input", component="stateful passthrough")
    return {"output": proto}
