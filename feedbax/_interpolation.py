"""Interpolation kernels for sampled trajectories."""

from jax import Array


def catmull_rom(p0: Array, p1: Array, p2: Array, p3: Array, t: Array) -> Array:
    """Catmull-Rom spline interpolation."""
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2 * p1)
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
    )
