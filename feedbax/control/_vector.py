"""Shape and dtype contract shared by stateful vector control blocks."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from operator import index
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from feedbax.contracts.portable import InputPrototypeParameter

_Vector = TypeVar("_Vector")
_CONTROL_DTYPES = frozenset(
    jnp.dtype(dtype) for dtype in ("bfloat16", "float16", "float32", "float64")
)


def control_vector_prototype(
    n_dims: int, input_dtype: DTypeLike | None = None
) -> jax.ShapeDtypeStruct:
    """Describe the exact input and carry vector for a control block."""
    try:
        size = index(n_dims)
    except TypeError as exc:
        raise ValueError(f"n_dims must be a positive integer, got {n_dims!r}") from exc
    if isinstance(n_dims, bool) or size < 1:
        raise ValueError(f"n_dims must be a positive integer, got {n_dims!r}")

    if input_dtype is None:
        dtype = jnp.result_type(float)
    else:
        try:
            dtype = jnp.dtype(input_dtype)
        except TypeError as exc:
            raise ValueError(
                f"input_dtype must be a real floating dtype, got {input_dtype!r}"
            ) from exc
        if dtype not in _CONTROL_DTYPES:
            raise ValueError(
                f"input_dtype must be bfloat16, float16, float32, or float64, got {dtype}"
            )
        if not dtype.isnative or dtype != jax.dtypes.canonicalize_dtype(dtype):
            raise ValueError(
                f"input_dtype {dtype} is unavailable with the current JAX configuration"
            )

    return jax.ShapeDtypeStruct((size,), dtype)


def initial_control_vector(
    n_dims: int, input_dtype: DTypeLike | None, initial_value: float = 0.0
) -> jax.Array:
    """Materialize the initial carry using the control vector contract."""
    prototype = control_vector_prototype(n_dims, input_dtype)
    return jnp.full(prototype.shape, initial_value, dtype=prototype.dtype)


def require_control_vector(
    value: _Vector, expected: jax.Array | jax.ShapeDtypeStruct, *, port: str
) -> _Vector:
    """Check a control port without converting or inspecting its array values."""
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape != expected.shape or dtype != expected.dtype:
        observed = f"shape {shape} and dtype {dtype}" if shape is not None else type(value).__name__
        raise ValueError(
            f"{port} requires shape {expected.shape} and dtype {expected.dtype}, got {observed}"
        )
    return value


def _control_vector(params: Mapping[str, Any], proto: Any, *, port: str) -> jax.ShapeDtypeStruct:
    dtype = params.get("input_dtype")
    if dtype is None and proto is not None:
        dtype = getattr(proto, "dtype", None)
    expected = control_vector_prototype(params.get("n_dims", 1), dtype)
    if proto is not None:
        require_control_vector(proto, expected, port=port)
    return expected


def _control_input_dtype(params: Mapping[str, Any], proto: Any, *, port: str) -> str:
    return _control_vector(params, proto, port=port).dtype.name


def _control_output_prototype(
    params: Mapping[str, Any],
    inputs: Mapping[str, Any],
    *,
    port: str,
) -> dict[str, Any]:
    proto = _control_vector(params, inputs.get(port), port=port)
    return {"output": jnp.zeros(proto.shape, dtype=proto.dtype)}


_CONTROL_INPUT_PROTOTYPE = InputPrototypeParameter(
    "input", "input_dtype", partial(_control_input_dtype, port="input")
)


_CONTROL_ERROR_PROTOTYPE = InputPrototypeParameter(
    "error", "input_dtype", partial(_control_input_dtype, port="error")
)
