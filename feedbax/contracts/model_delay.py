"""The carry projection of a stored delay attachment."""

from typing import Annotated, Literal

from pydantic import Field

from feedbax.contracts.base import StrictModel
from feedbax.contracts.model_address import ModelElementId


class ModelDelayCarry(StrictModel):
    kind: Literal["zeros"] = "zeros"
    scope: Literal["trial"] = "trial"
    shape: tuple[Annotated[int, Field(strict=True, ge=0)], ...]
    dtype: Literal["float32"] = "float32"
    units: str | None = Field(default=None, exclude_if=lambda value: value is None)
    steps: Annotated[int, Field(strict=True, ge=1)] = Field(default=1, exclude_if=lambda value: value == 1)
    initial: Annotated[float, Field(allow_inf_nan=False)] = Field(default=0, exclude_if=lambda value: value == 0)
    model_delay_id: ModelElementId
    model_wire_id: ModelElementId

    @property
    def carry_shape(self) -> tuple[int, ...]:
        return (self.steps, *self.shape) if self.steps > 1 else self.shape
