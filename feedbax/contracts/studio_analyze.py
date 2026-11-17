"""Managed Analyze authoring projections and reviewed wire edits."""

from pydantic import Field

from feedbax.contracts.analysis_document import DatasetSource
from feedbax.contracts.base import StrictModel
from feedbax.contracts.figure_document import FigureDocument
from feedbax.contracts.jobs import StudioSavedTrainKey
from feedbax.contracts.operation import DatasetPort, OperationDefinition
from feedbax.contracts.project import Sha256
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.studio_project import StudioProjectFilePatch


class StudioAnalyzeWire(StrictModel):
    element: ProjectName
    role: ProjectName
    source: DatasetSource


class StudioAnalyzeRequest(StrictModel):
    revision: Sha256
    key: StudioSavedTrainKey
    wire: StudioAnalyzeWire | None = None


class StudioAnalyzeAuthoring(StrictModel):
    revision: Sha256
    document: ProjectName
    operations: list[OperationDefinition]
    states: DatasetPort
    variables: dict[str, dict[str, list[str]]]
    figure: FigureDocument
    files: list[StudioProjectFilePatch] = Field(default_factory=list)
