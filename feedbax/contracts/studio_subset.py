"""Revision-bound freezing of a saved measured subject."""

from typing import Annotated

from pydantic import Field, StringConstraints

from feedbax.contracts.base import StrictModel
from feedbax.contracts.project_inputs import ProjectName, RowKey
from feedbax.contracts.request_refs import Sha256
from feedbax.contracts.studio_project import StudioProjectFilePatch


class StudioSubsetToRowsRequest(StrictModel):
    revision: Sha256
    variant: RowKey
    step: ProjectName
    input: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_-]*(?:\.(?:0|[1-9][0-9]*))?$")]
    """The selection to freeze: its role, or its role and its index in a list."""
    attempts: list[str] = Field(min_length=1)


class StudioSubsetToRowsEdit(StrictModel):
    revision: Sha256
    files: list[StudioProjectFilePatch]
