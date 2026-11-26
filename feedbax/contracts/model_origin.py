"""Identify which source supplied a consumed model."""

from typing import Annotated, Literal

from pydantic import Field

from feedbax.contracts.base import StrictModel
from feedbax.contracts.request_refs import BatchCoordinate, Name, RequestId


class ModelProductKey(StrictModel):
    variant: str
    step: str
    row: str


class InitialModelOrigin(StrictModel):
    kind: Literal["initial"]


class LoadedModelOrigin(StrictModel):
    kind: Literal["loaded"]


class ModelProducerOrigin(StrictModel):
    key: ModelProductKey
    job_id: Name
    request_id: RequestId


class TrainedModelOrigin(ModelProducerOrigin):
    kind: Literal["trained"]
    completed_updates: BatchCoordinate


class SynthesizedModelOrigin(ModelProducerOrigin):
    kind: Literal["synthesized"]


type ModelOrigin = Annotated[
    InitialModelOrigin | LoadedModelOrigin | TrainedModelOrigin | SynthesizedModelOrigin,
    Field(discriminator="kind"),
]
