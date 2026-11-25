"""The carry projection of a stored delay attachment."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.model_address import ComponentPortEnd, ModelElementId


class ModelDelayCarry(StrictModel):
    kind: Literal["zeros"] = "zeros"
    scope: Literal["trial"] = "trial"
    shape: tuple[Annotated[int, Field(strict=True, ge=0)], ...] | None = Field(
        default=None, exclude_if=lambda value: value is None,
    )
    """An optional array seed for prototype resolution on recurrent circuits."""
    dtype: Literal["float32", "float64"] = "float32"
    units: str | None = Field(default=None, exclude_if=lambda value: value is None)
    steps: Annotated[int, Field(strict=True, ge=1)] = Field(default=1, exclude_if=lambda value: value == 1)
    initial: Annotated[float, Field(allow_inf_nan=False)] = Field(default=0, exclude_if=lambda value: value == 0)
    reset_output: ComponentPortEnd | None = Field(default=None, exclude_if=lambda value: value is None)
    reset_evaluate: ComponentPortEnd | None = Field(default=None, exclude_if=lambda value: value is None)
    reset_value: str | None = Field(default=None, exclude_if=lambda value: value is None)
    model_delay_id: ModelElementId
    model_wire_id: ModelElementId

    @model_validator(mode="after")
    def _single_reset_source(self) -> "ModelDelayCarry":
        sources = self.reset_output, self.reset_evaluate, self.reset_value
        if sum(source is not None for source in sources) > 1:
            raise ValueError("Delay Initial names one reset source.")
        return self
