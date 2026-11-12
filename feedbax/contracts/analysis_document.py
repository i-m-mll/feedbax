"""An authored Analyze run's operations, figures, and dataset wires."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.figure_document import FigureDocument
from feedbax.contracts.project_inputs import NonemptyText, ProjectName


class StepDataset(StrictModel):
    input: ProjectName


class OperationDataset(StrictModel):
    element: ProjectName
    product: ProjectName


DatasetSource = StepDataset | OperationDataset


class OperationElement(StrictModel):
    operation: NonemptyText
    inputs: dict[ProjectName, DatasetSource]
    params: dict[str, JsonValue] = Field(default_factory=dict)


class FigureElement(StrictModel):
    figure: FigureDocument
    inputs: dict[ProjectName, DatasetSource]


class AnalysisDocument(StrictModel):
    schema_version: Literal["feedbax.analysis_document.v1"] = "feedbax.analysis_document.v1"
    operations: dict[ProjectName, OperationElement] = Field(default_factory=dict)
    views: dict[ProjectName, FigureElement] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_elements(self) -> AnalysisDocument:
        if not self.operations and not self.views:
            raise ValueError("An analysis document requires an operation or figure.")
        if duplicate := self.operations.keys() & self.views.keys():
            raise ValueError(f"Analysis element names are repeated: {sorted(duplicate)}.")
        return self
