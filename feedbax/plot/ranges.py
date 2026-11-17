"""Dependency-light numeric range primitives for scientific figures."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Real

import numpy as np


def calculate_array_minmax(
    arrays: Iterable[object],
    *,
    padding: float,
) -> tuple[float, float]:
    """Return the finite union range with symmetric proportional padding.

    When the finite union has zero span, its padding is based on
    ``max(abs(value), 1)``. This keeps the fallback deterministic at every
    scale while retaining the authored padding fraction.

    Args:
        arrays: The coordinate samples rendered on the grouped axes.
        padding: Non-negative finite fraction of the union span added at both
            ends.

    Returns:
        An increasing ``(minimum, maximum)`` pair of Python floats.

    Raises:
        ValueError: If padding is invalid, no finite values exist, an array is
            not numeric, or zero-span input is paired with zero padding.
    """
    if isinstance(padding, bool) or not isinstance(padding, Real):
        raise ValueError("range padding must be a finite non-negative number")
    padding_value = float(padding)
    if not np.isfinite(padding_value) or padding_value < 0.0:
        raise ValueError("range padding must be a finite non-negative number")

    finite_parts: list[np.ndarray] = []
    for position, value in enumerate(arrays):
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"range input {position} is not numeric") from exc
        finite = array.reshape(-1)
        finite_parts.append(finite[np.isfinite(finite)])

    finite_parts = [part for part in finite_parts if part.size]
    if not finite_parts:
        raise ValueError("a data-derived range requires at least one finite value")
    finite = np.concatenate(finite_parts)
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    span = maximum - minimum
    if span == 0.0:
        if padding_value == 0.0:
            raise ValueError("a zero-span data-derived range requires positive padding")
        span = max(abs(minimum), 1.0)
    margin = padding_value * span
    return minimum - margin, maximum + margin
