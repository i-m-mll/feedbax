"""Revision-bound field authoring for the stored Model inspector."""

from typing import Annotated, Literal

from pydantic import Field

from feedbax.contracts.base import StrictModel
from feedbax.contracts.expression_values import Problem
from feedbax.contracts.jobs import StudioSavedTrainKey
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_edit import ModelEditCommand
from feedbax.contracts.project import Sha256
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.studio_authoring import StudioSweepList, StudioSweepRange, StudioSweepSampler, StudioSweepValue
from feedbax.contracts.studio_project import StudioProjectFilePatch
from feedbax.expression_site import SiteProjection


class FieldTextEdit(StrictModel):
    kind: Literal["text"]
    text: Annotated[str, Field(max_length=8192)]
    value_id: ProjectName | None = None
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] | None = None


class FieldKindEdit(StrictModel):
    kind: Literal["to_sweep", "to_value", "to_list"]
    value_id: ProjectName | None = None


class ModelFieldRequest(StrictModel):
    revision: Sha256
    owner: ModelAddress
    run: StudioSavedTrainKey | None = None
    address: ModelAddress | None = None
    lit_steps: list[ProjectName] = Field(default_factory=list)
    rows: list[str] = Field(default_factory=list)
    edit: Annotated[FieldTextEdit | FieldKindEdit, Field(discriminator="kind")] | None = None


class StudioFieldView(StrictModel):
    address: ModelAddress
    label: str
    text: str
    effective: str | None = None
    site: SiteProjection
    sweep: StudioSweepList | StudioSweepRange | StudioSweepSampler | None = None
    selected_value: ProjectName | None = None
    can_sweep: bool = False
    members: list[StudioSweepValue] = Field(default_factory=list)


class ModelFieldResponse(StrictModel):
    fields: list[StudioFieldView] = Field(default_factory=list)
    files: list[StudioProjectFilePatch] = Field(default_factory=list)
    problems: list[Problem] = Field(default_factory=list)
    command: ModelEditCommand | None = None
