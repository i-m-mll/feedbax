"""Revision-bound Task, Objective and Model pairing projections for Studio."""

from typing import Annotated, Literal

from pydantic import Field, JsonValue

from feedbax.contracts.base import StrictModel
from feedbax.contracts.expression_values import Level, Problem, ScopedValue, ValueType
from feedbax.contracts.jobs import StudioSavedTrainKey
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_edit import (
    EvaluateAttachmentAxis, EvaluateAttachmentRows, EvaluateAttachmentVariant, StudioEvaluateValueSource,
)
from feedbax.contracts.objective_document import ObjectiveDocument
from feedbax.contracts.project import ProjectValueEdit, Sha256
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.studio_project import StudioProjectFilePatch
from feedbax.contracts.task_document import Epoch, PythonTaskSelection, TaskDocument
from feedbax.expression_site import SiteProjection


class StudioTrialField(StrictModel):
    path: str
    projection: SiteProjection
    text: str | None = None


class StudioTaskType(StrictModel):
    type: str
    label: str
    parameter_schema: dict[str, JsonValue]


class StudioTrialInput(StrictModel):
    address: ModelAddress
    name: str
    type: ValueType | None
    required: bool
    signal: str | None
    explicit: bool


class StudioTrialOutput(StrictModel):
    address: ModelAddress
    value: ScopedValue


class StudioTrialReset(StrictModel):
    address: ModelAddress
    type: ValueType
    finest: Level
    variable: str | None


class StudioTrialDocumentEdit(StrictModel):
    kind: Literal["task", "objective"]
    edits: list[ProjectValueEdit] = Field(min_length=1)


class StudioTrialSelectTask(StrictModel):
    kind: Literal["select_task"]
    document: ProjectName


class StudioTrialBindingEdit(StrictModel):
    kind: Literal["binding"]
    input: ModelAddress
    signal: str | None


class StudioTrialExpressionEdit(StrictModel):
    kind: Literal["expression"]
    term: str
    field: Literal["measure", "weight"]
    text: Annotated[str, Field(max_length=8192)]


class StudioTrialStepScope(StrictModel):
    kind: Literal["step"]
    lit_steps: list[ProjectName] = Field(default_factory=list)


class StudioTrialAuthoringRequest(StrictModel):
    revision: Sha256
    run: StudioSavedTrainKey
    change: Annotated[
        StudioTrialDocumentEdit | StudioTrialBindingEdit | StudioTrialSelectTask | StudioTrialExpressionEdit,
        Field(discriminator="kind"),
    ] | None = None
    scope: Annotated[
        StudioTrialStepScope | EvaluateAttachmentAxis | EvaluateAttachmentRows | EvaluateAttachmentVariant,
        Field(discriminator="kind"),
    ] | None = None


class StudioTaskChoice(StrictModel):
    document: ProjectName
    problems: list[Problem]


class StudioTrialAuthoring(StrictModel):
    revision: Sha256
    run: StudioSavedTrainKey
    model: ProjectName
    task_document: ProjectName | None
    objective_document: ProjectName | None
    task: TaskDocument | PythonTaskSelection
    task_source: StudioEvaluateValueSource | None = None
    task_field_sources: dict[str, StudioEvaluateValueSource] = Field(default_factory=dict)
    objective: ObjectiveDocument | None
    task_types: list[StudioTaskType]
    tasks: list[StudioTaskChoice]
    variables: list[ScopedValue]
    signals: list[ScopedValue]
    epochs: list[Epoch]
    period_seconds: float
    n_steps: int | None
    inputs: list[StudioTrialInput]
    outputs: list[StudioTrialOutput]
    resets: list[StudioTrialReset]
    fields: list[StudioTrialField]
    problems: list[Problem]
    files: list[StudioProjectFilePatch]
