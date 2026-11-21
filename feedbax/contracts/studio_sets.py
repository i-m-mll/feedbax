"""Revision-bound edits that make runs from a selection and move a step's axis between inline values and a Set."""

from typing import Annotated, Literal

from pydantic import Field

from feedbax.contracts.base import StrictModel
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.request_refs import Sha256
from feedbax.contracts.studio_authoring import StudioRunSelection
from feedbax.contracts.studio_project import CreateNamedDocument, StudioProjectFilePatch


class DuplicateRun(StudioRunSelection):
    """Give each selected run a copy with its differences.

    The runs of one value of a named axis give that axis, or the set it reads, a value; any other selection gives the
    step an axis of the run and its copy over the selected runs, every run of the step when no rows are given.
    """

    kind: Literal["duplicate_run"]


class SaveSet(StrictModel):
    """Make the step's inline axis a new Set document the axis reads."""

    kind: Literal["save_set"]
    step: ProjectName
    axis: ProjectName


class InlineSet(StrictModel):
    """Give the step's axis the values of the set it reads."""

    kind: Literal["inline_set"]
    step: ProjectName
    axis: ProjectName


class ApplySet(StrictModel):
    """Add an axis to the step's sweep reading a set offered to it."""

    kind: Literal["apply_set"]
    step: ProjectName
    set: ProjectName


class RemoveAxisValue(StrictModel):
    """Remove a value of a named axis or an entry of the set it reads; an inline axis left with one value that makes
    no difference is removed."""

    kind: Literal["remove_value"]
    step: ProjectName
    axis: ProjectName
    value: ProjectName


class StudioSetEditRequest(StrictModel):
    revision: Sha256
    command: Annotated[
        DuplicateRun | SaveSet | InlineSet | ApplySet | RemoveAxisValue, Field(discriminator="kind"),
    ]


class StudioAxisValue(StrictModel):
    axis: ProjectName
    value: ProjectName


class StudioSetEdit(StrictModel):
    """The files that store the edit, after the documents it creates; ``selected`` is the value Duplicate Run made."""

    revision: Sha256
    files: list[StudioProjectFilePatch]
    created: list[CreateNamedDocument] = Field(default_factory=list)
    selected: StudioAxisValue | None = None


class StudioSetsOfferedRequest(StrictModel):
    revision: Sha256
    step: ProjectName


class StudioSetsOffered(StrictModel):
    sets: list[ProjectName]
