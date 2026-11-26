"""An authored Analyze run's operations, figures, and dataset wires."""

from __future__ import annotations

from typing import Annotated, Generic, Literal

from pydantic import Field, JsonValue, TypeAdapter, ValidationError, model_validator
from typing_extensions import TypeVar

from feedbax.contracts.base import StrictModel
from feedbax.contracts.figure_document import FigureSource
from feedbax.contracts.project_inputs import DocumentReference, NonemptyText, ProjectName


ANALYSIS_SCHEMA = "feedbax.analysis_document.v1"
ELEMENT_FAMILIES = ("operations", "views")
_LABEL = TypeAdapter(NonemptyText)


def check_analysis_labels(payload: JsonValue) -> None:
    """Check stored element names even when the rest of the Analyze document is still a draft."""
    if not isinstance(payload, dict):
        return
    for family in ELEMENT_FAMILIES:
        elements = payload.get(family)
        if not isinstance(elements, dict):
            continue
        for key, element in elements.items():
            try:
                _LABEL.validate_python(element.get("label") if isinstance(element, dict) else None)
            except ValidationError as error:
                raise ValueError(f"Analysis {family} member {key!r} needs a nonempty label.") from error


class StepInput(StrictModel):
    input: ProjectName


class OperationDataset(StrictModel):
    element: ProjectName
    product: ProjectName
    unresolved: Literal[True] | None = Field(default=None, exclude_if=lambda value: value is None)


DatasetSource = StepInput | OperationDataset


class AnalyzeDatasetInput(StrictModel):
    kind: Literal["dataset"]
    cardinality: Literal["one", "many"] = "many"


class AnalyzeModelInput(StrictModel):
    kind: Literal["model"]


class AnalyzeTaskInput(StrictModel):
    kind: Literal["task"]


AnalyzeInput = Annotated[AnalyzeDatasetInput | AnalyzeModelInput | AnalyzeTaskInput, Field(discriminator="kind")]


class OperationElement(StrictModel):
    """An operation; its key is held by one family of Analysis documents, and ``label`` is only shown."""

    operation: NonemptyText
    label: NonemptyText
    inputs: dict[ProjectName, DatasetSource]
    params: dict[str, JsonValue] = Field(default_factory=dict)


_Figure = TypeVar("_Figure", default=FigureSource | DocumentReference)


class FigureElement(StrictModel, Generic[_Figure]):
    """A view's figure source and its dataset wiring; ``label`` is only shown."""

    figure: _Figure
    label: NonemptyText
    inputs: dict[ProjectName, DatasetSource]


class AnalysisPayload(StrictModel, Generic[_Figure]):
    schema_version: Literal["feedbax.analysis_document.v1"] = "feedbax.analysis_document.v1"
    inputs: dict[ProjectName, AnalyzeInput] = Field(default_factory=dict)
    operations: dict[ProjectName, OperationElement] = Field(default_factory=dict)
    views: dict[ProjectName, FigureElement[_Figure]] = Field(default_factory=dict)


class AnalysisDocument(AnalysisPayload[_Figure], Generic[_Figure]):
    @model_validator(mode="after")
    def _unique_elements(self) -> AnalysisDocument:
        if not self.operations and not self.views:
            raise ValueError("An analysis document requires an operation or figure.")
        if duplicate := self.operations.keys() & self.views.keys():
            raise ValueError(f"Analysis element names are repeated: {sorted(duplicate)}.")
        return self
