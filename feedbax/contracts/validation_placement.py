"""Placement facts recorded by the Host for a checkpoint evaluation attempt."""

from typing import Literal

from pydantic import Field

from feedbax.contracts.base import StrictModel


class ValidationPlacement(StrictModel):
    location: Literal["live", "ordinary"]
    reason: Literal["within_budget", "late", "unqualified", "unmeasured", "over_budget", "driver_occupied"]
    predicted_update_fraction: float | None = Field(
        default=None, allow_inf_nan=False, exclude_if=lambda value: value is None,
    )
