"""Studio archive exchange and the located problems of model import."""

from typing import Annotated, Literal

from pydantic import Field, RootModel

from feedbax.contracts.base import StrictModel
from feedbax.contracts.synthesis import ModelProductKey
from feedbax.contracts.loaded_model import ModelFitReport
from feedbax.contracts.project_inputs import NonemptyText, ProjectName
from feedbax.contracts.request_refs import BatchCoordinate, Name, Sha256
from feedbax.contracts.studio_project import StudioProjectSnapshot


class StudioLoadedModelImportQuery(StrictModel):
    revision: Sha256
    name: ProjectName
    model: ProjectName


class StudioLoadedModelExportRequest(StrictModel):
    revision: Sha256
    key: ModelProductKey
    job_id: NonemptyText
    completed_updates: BatchCoordinate | None = None


class StudioLoadedModelImported(StrictModel):
    status: Literal["imported"]
    document: ProjectName
    snapshot: StudioProjectSnapshot


class StudioLoadedModelDoesNotFit(StrictModel):
    status: Literal["does_not_fit"]
    report: ModelFitReport


class StudioLoadedModelUnreadableArchive(StrictModel):
    status: Literal["unreadable_archive"]
    path: Name
    detail: NonemptyText


class StudioLoadedModelImportResult(RootModel[Annotated[
    StudioLoadedModelImported | StudioLoadedModelDoesNotFit | StudioLoadedModelUnreadableArchive,
    Field(discriminator="status"),
]]):
    pass
