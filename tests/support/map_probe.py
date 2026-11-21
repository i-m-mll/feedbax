"""Operations whose ports take one member, or the dimensions a parameter lists, of what is bound."""

from typing import Annotated

from pydantic import Field

from feedbax.analysis.operations import DatasetOutput, operation
from feedbax.builtins.operations import (
    _STATES, _SUMMARY, _VELOCITY, AlignedVelocityParameters, VelocitySummaryParameters, aligned_velocity,
    summarize_velocity,
)
from feedbax.contracts.operation import DatasetInputPort, WholeDimensions
from feedbax.dataset import Dataset


class WholeParameters(AlignedVelocityParameters):
    whole: list[str] = Field(
        default_factory=list, description="Label dimensions each request takes whole.",
        json_schema_extra={"feedbax": {"static": True}},
    )


@operation(settings=AlignedVelocityParameters, label="Align each")
def align_each(
    states: Annotated[Dataset, DatasetInputPort(**_STATES.model_dump(), takes="one")],
    params: AlignedVelocityParameters,
) -> Annotated[Dataset, DatasetOutput("velocity", _VELOCITY)]:
    return aligned_velocity(states, params)


@operation(settings=WholeParameters, label="Align over")
def align_over(
    states: Annotated[Dataset, DatasetInputPort(**_STATES.model_dump(), takes=WholeDimensions(parameter="whole"))],
    params: WholeParameters,
) -> Annotated[Dataset, DatasetOutput("velocity", _VELOCITY)]:
    return aligned_velocity(states, AlignedVelocityParameters.model_validate(params.model_dump(exclude={"whole"})))


@operation(settings=VelocitySummaryParameters, label="Summarize each")
def summarize_each(
    velocity: Annotated[Dataset, DatasetInputPort(**_VELOCITY.model_dump(), takes="one")],
    params: VelocitySummaryParameters,
) -> Annotated[Dataset, DatasetOutput("summary", _SUMMARY)]:
    return summarize_velocity(velocity, params)


@operation(settings=VelocitySummaryParameters, label="Pair")
def pair(
    left: Annotated[Dataset, DatasetInputPort(**_STATES.model_dump(exclude={"selected_variables"}))],
    right: Annotated[Dataset, DatasetInputPort(**_VELOCITY.model_dump())],
    params: VelocitySummaryParameters,
) -> Annotated[Dataset, DatasetOutput("summary", _SUMMARY)]:
    return summarize_velocity(right, params)


@operation(settings=VelocitySummaryParameters, label="Pair by left")
def pair_left(
    left: Annotated[Dataset, DatasetInputPort(**_STATES.model_dump(exclude={"selected_variables"}))],
    right: Annotated[Dataset, DatasetInputPort(**_VELOCITY.model_dump())],
    params: VelocitySummaryParameters,
) -> Annotated[Dataset, DatasetOutput("summary", _SUMMARY, carries="left")]:
    return summarize_velocity(right, params)
