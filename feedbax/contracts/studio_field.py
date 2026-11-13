"""Revision-bound Studio field controls and prepared edits."""

from typing import Annotated, Literal

from pydantic import Field

from feedbax.contracts.base import StrictModel
from feedbax.contracts.expression_values import Problem
from feedbax.contracts.jobs import StudioSavedTrainKey
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_edit import EvaluateAttachmentScope, ModelEditCommand
from feedbax.contracts.project import Sha256
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.project_sweep import RangeValues, SampleValues
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


class FieldNameEdit(StrictModel):
    kind: Literal["name_value"]
    value_id: ProjectName
    label: str | None


class ModelFieldRequest(StrictModel):
    revision: Sha256
    owner: ModelAddress
    run: StudioSavedTrainKey | None = None
    evaluate_scope: EvaluateAttachmentScope | None = None
    address: ModelAddress | None = None
    lit_steps: list[ProjectName] = Field(default_factory=list)
    rows: list[str] = Field(default_factory=list)
    edit: Annotated[FieldTextEdit | FieldKindEdit | FieldNameEdit, Field(discriminator="kind")] | None = None


class StudioSweepValue(StrictModel):
    id: ProjectName
    label: str | None
    value: int | float


class StudioSweepList(StrictModel):
    kind: Literal["list"]
    id: ProjectName
    values: list[StudioSweepValue]
    editable: bool


class StudioSweepRange(StrictModel):
    kind: Literal["range"]
    id: ProjectName
    rule: RangeValues
    can_list: bool


class StudioSweepSampler(StrictModel):
    kind: Literal["sampler"]
    id: ProjectName
    rule: SampleValues
    can_list: bool


class StudioFieldView(StrictModel):
    label: str
    text: str
    effective: str | None = None
    unit: str | None = None
    site: SiteProjection
    sweep: StudioSweepList | StudioSweepRange | StudioSweepSampler | None = None
    selected_value: ProjectName | None = None
    can_sweep: bool = False
    can_change_axis: bool = False
    members: list[StudioSweepValue] = Field(default_factory=list)


class ModelFieldView(StudioFieldView):
    address: ModelAddress


class TrainFieldView(StudioFieldView):
    id: str


class TrainFieldRequest(StrictModel):
    revision: Sha256
    run: StudioSavedTrainKey | None = None
    field: str | None = None
    rows: list[str] = Field(default_factory=list)
    edit: Annotated[FieldTextEdit | FieldKindEdit | FieldNameEdit, Field(discriminator="kind")] | None = None


class TrainFieldResponse(StrictModel):
    fields: list[TrainFieldView] = Field(default_factory=list)
    files: list[StudioProjectFilePatch] = Field(default_factory=list)
    dependencies: list[ProjectName] = Field(default_factory=list)
    problems: list[Problem] = Field(default_factory=list)


class ModelFieldResponse(StrictModel):
    fields: list[ModelFieldView] = Field(default_factory=list)
    files: list[StudioProjectFilePatch] = Field(default_factory=list)
    problems: list[Problem] = Field(default_factory=list)
    command: ModelEditCommand | None = None
