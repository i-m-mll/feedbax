"""Choose a placement from measured capacity and the resolved Evaluate workload."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from feedbax.checkpoint_coordinates import finite_coordinates


@dataclass(frozen=True)
class ValidationBudget:
    durable_point_p95_ms: float
    steady_update_ms: float
    max_point_ms: float
    max_update_fraction: float

    def __post_init__(self) -> None:
        if any(not isfinite(value) or value <= 0 for value in (
            self.durable_point_p95_ms, self.steady_update_ms, self.max_point_ms, self.max_update_fraction,
        )):
            raise ValueError("Placement needs finite positive timing and budget inputs.")


@dataclass(frozen=True)
class ValidationPlacement:
    location: Literal["live", "ordinary"]
    reason: Literal["within_budget", "late", "unqualified", "unmeasured", "over_budget"]
    predicted_update_fraction: float | None = None


def place_validation(
    updates: tuple[int, ...],
    *,
    producer_started: bool,
    live_products_supported: bool,
    budget: ValidationBudget | None,
) -> ValidationPlacement:
    """Resolve placement for a finite series without changing its requested updates."""
    coordinates = sorted(finite_coordinates(updates))
    if producer_started:
        return ValidationPlacement("ordinary", "late")
    if not live_products_supported:
        return ValidationPlacement("ordinary", "unqualified")
    if budget is None:
        return ValidationPlacement("ordinary", "unmeasured")
    interval = min(right - left for left, right in zip((0, *coordinates), coordinates))
    fraction = budget.durable_point_p95_ms / (budget.steady_update_ms * interval)
    if budget.durable_point_p95_ms > budget.max_point_ms or fraction > budget.max_update_fraction:
        return ValidationPlacement("ordinary", "over_budget", fraction)
    return ValidationPlacement("live", "within_budget", fraction)
