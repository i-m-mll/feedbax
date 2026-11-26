"""The authored Synthesize payload and method execution metadata."""

from pydantic import Field, JsonValue, field_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.operation import DatasetPort, JaxOperation
from feedbax.contracts.stored_model import StoredModelDocument
from feedbax.contracts.objective_document import StoredObjectiveDocument
from feedbax.contracts.project import DocumentReference
from feedbax.contracts.task_document import PythonTaskSelection, StoredTaskDocument
from feedbax.plugins.declarations import validate_declaration_key


class SynthesisStepPayload(StrictModel):
    model: DocumentReference | StoredModelDocument
    task: DocumentReference | StoredTaskDocument | PythonTaskSelection
    objective: DocumentReference | StoredObjectiveDocument
    input_bindings: dict[str, str | None] = Field(default_factory=dict)
    method: str
    settings: dict[str, JsonValue] = Field(default_factory=dict)
    target: ModelAddress = ModelAddress.at_stored_path(())

    _method_key = field_validator("method")(validate_declaration_key)


class SynthesisMethodExecution(StrictModel):
    key: str
    numerical: JaxOperation
    outputs: dict[str, DatasetPort]
