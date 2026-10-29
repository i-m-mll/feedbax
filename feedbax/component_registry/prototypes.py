"""Single-array stateful prototype parameter encoding."""

from typing import Any, Mapping

import jax.numpy as jnp


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
