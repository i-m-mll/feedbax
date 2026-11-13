"""The qualified one-step carry projection of a stored delay attachment."""

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
    model_delay_id: ModelElementId
    model_wire_id: ModelElementId
