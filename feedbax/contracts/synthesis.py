"""The authored Synthesize payload and the provenance of a model product."""

from typing import Annotated, Literal

from pydantic import Field, JsonValue, field_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.operation import DatasetPort, JaxOperation
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.objective_document import ObjectiveDocument
from feedbax.contracts.project import DocumentReference
from feedbax.contracts.request_refs import BatchCoordinate, Name, RequestId
from feedbax.contracts.task_document import PythonTaskSelection, TaskDocument
from feedbax.plugins.declarations import validate_declaration_key


class SynthesisStepPayload(StrictModel):
    model: DocumentReference | ModelDocument
    task: DocumentReference | TaskDocument | PythonTaskSelection
    objective: DocumentReference | ObjectiveDocument
    method: str
    settings: dict[str, JsonValue] = Field(default_factory=dict)
    target: ModelAddress = ModelAddress.at_stored_path(())

    _method_key = field_validator("method")(validate_declaration_key)


class SynthesisMethodExecution(StrictModel):
    key: str
    numerical: JaxOperation
    outputs: dict[str, DatasetPort]


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
