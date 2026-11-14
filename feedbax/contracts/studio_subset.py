"""Revision-bound freezing of a saved measured subject."""

from pydantic import Field

from feedbax.contracts.base import StrictModel
from feedbax.contracts.project_inputs import ProjectName, RowKey
from feedbax.contracts.request_refs import Sha256
from feedbax.contracts.studio_project import StudioProjectFilePatch


class StudioSubsetToRowsRequest(StrictModel):
    revision: Sha256
    variant: RowKey
    step: ProjectName
    attempts: list[str] = Field(min_length=1)


class StudioSubsetToRowsEdit(StrictModel):
    revision: Sha256
    files: list[StudioProjectFilePatch]
