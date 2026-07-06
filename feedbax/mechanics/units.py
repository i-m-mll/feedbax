"""Shared mechanics units, frames, and numerical convention checks.

Feedbax mechanics use SI units unless a dimensionless normalized quantity is
explicitly named:

- length: metres
- mass: kilograms
- time: seconds
- force: newtons
- torque: newton-metres
- angles: radians

Cartesian states are expressed in a right-handed XY frame.  Two-link arm
joint coordinates follow the skeleton convention: positive joint angles rotate
counter-clockwise and angular velocities are radians per second.  Muscle
velocities are positive during lengthening and negative during shortening.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import jax.numpy as jnp
from jaxtyping import Array


DEFAULT_MUSCLE_VMAX: float = 10.0
"""Default maximum shortening velocity in optimal fiber lengths per second."""

ECCENTRIC_VELOCITY_LIMIT_FRACTION: float = 0.1
"""Maximum lengthening speed as a fraction of ``vmax`` used by Hill models."""


def require_positive_finite(name: str, value: float) -> float:
    """Return ``value`` as ``float`` after a constructor-time finite/positive check."""
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite positive scalar") from exc
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return scalar


def require_array_shape(name: str, value, shape: Sequence[int]) -> Array:
    """Return ``value`` as an array with an exact constructor-time shape check."""
    arr = jnp.asarray(value)
    expected = tuple(shape)
    if arr.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {arr.shape}")
    return arr


def require_finite_array(name: str, value, shape: Sequence[int] | None = None) -> Array:
    """Return ``value`` as an array after constructor-time finite checks."""
    arr = require_array_shape(name, value, shape) if shape is not None else jnp.asarray(value)
    if bool(jnp.any(~jnp.isfinite(arr))):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def require_positive_array(name: str, value, shape: Sequence[int] | None = None) -> Array:
    """Return ``value`` as an array after finite and strictly-positive checks."""
    arr = require_finite_array(name, value, shape)
    if bool(jnp.any(arr <= 0.0)):
        raise ValueError(f"{name} must contain only positive values")
    return arr
