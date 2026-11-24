"""Revision-bound Task, Objective and Model pairing projections for Studio."""

from typing import Annotated, Literal

from pydantic import Field, JsonValue

from feedbax.contracts.base import StrictModel
from feedbax.contracts.expression_values import Level, Problem, ScopedValue, ValueType
from feedbax.contracts.jobs import StudioSavedTrainKey
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_edit import (
    AxisValueRuns, ListedRuns, StudioEvaluateValueSource,
)
from feedbax.contracts.objective_document import ObjectiveDocument
from feedbax.contracts.project import Pointer, ProjectValueEdit, Sha256
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.studio_authoring import StudioSweptParameterMissing
from feedbax.contracts.studio_field import FieldKindEdit, FieldNameEdit, FieldTextEdit, StudioFieldView
from feedbax.contracts.studio_project import CreateNamedDocument, StudioProjectFilePatch
from feedbax.contracts.task_document import EpochSteps, PythonTaskSelection, TaskDocument
from feedbax.contracts.task_structure import TaskStructureChange


class StudioTrialField(StudioFieldView):
    """A field of the pairing at ``path``; a number the step can sweep carries the step's sweep of it."""

    path: str
    problems: list[Problem] = Field(default_factory=list)
    """The stored value's problems; one about a term of the text carries the term's ``span`` in ``text``."""


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
    """A value of the paired Model that a variable of the Task may initialize at the start of each trial."""

    address: ModelAddress
    name: str
    """The component's name and the field's path."""
    type: ValueType
    finest: Level
    required: bool


class StudioTrialDocumentEdit(StrictModel):
    kind: Literal["task", "objective"]
    edits: list[ProjectValueEdit] = Field(min_length=1)


class StudioTrialDocumentReset(StrictModel):
    """Give each of ``paths`` in the step's ``document`` the value of that document's base."""

    kind: Literal["reset"]
    document: Literal["task", "objective"]
    paths: list[Pointer] = Field(min_length=1)


class StudioTrialSelectDocument(StrictModel):
    kind: Literal["select_document"]
    role: Literal["model", "task", "objective", "training"]
    document: ProjectName


class StudioTrialBindingEdit(StrictModel):
    kind: Literal["binding"]
    input: ModelAddress
    signal: str | None


class StudioTrialExpressionEdit(StrictModel):
    """An edit of the field at ``path``, one of the view's fields."""

    kind: Literal["expression"]
    path: Pointer
    edit: Annotated[FieldTextEdit | FieldKindEdit | FieldNameEdit, Field(discriminator="kind")]


class StudioTrialStructureEdit(StrictModel):
    """Changes to the structure of the step's task document, made in order as one edit: all of them, or none when one
    is refused."""

    kind: Literal["structure"]
    changes: list[TaskStructureChange] = Field(min_length=1)


class StudioTrialStepScope(StrictModel):
    kind: Literal["step"]
    chosen_steps: list[ProjectName] = Field(default_factory=list)


class StudioTrialAuthoringRequest(StrictModel):
    revision: Sha256
    run: StudioSavedTrainKey
    change: Annotated[
        StudioTrialDocumentEdit | StudioTrialDocumentReset | StudioTrialBindingEdit | StudioTrialSelectDocument
        | StudioTrialExpressionEdit | StudioTrialStructureEdit,
        Field(discriminator="kind"),
    ] | None = None
    scope: Annotated[
        StudioTrialStepScope | AxisValueRuns | ListedRuns,
        Field(discriminator="kind"),
    ] | None = None


class StudioTrialEpoch(StrictModel):
    """An epoch of the paired task; ``steps`` once the Task pairs with the Model."""

    id: str
    name: str
    steps: EpochSteps | None


class StudioTaskChoice(StrictModel):
    document: ProjectName
    problems: list[Problem]


class StudioTrialObjectiveSource(StrictModel):
    kind: Literal["train", "own"]
    step: ProjectName


class StudioTrialAuthoring(StrictModel):
    revision: Sha256
    run: StudioSavedTrainKey
    model: ProjectName | None
    task_document: ProjectName | None
    objective_document: ProjectName | None
    task: TaskDocument | PythonTaskSelection
    task_source: StudioEvaluateValueSource | None = None
    task_field_sources: dict[str, StudioEvaluateValueSource] = Field(default_factory=dict)
    task_changed: list[Pointer] = Field(default_factory=list)
    """Pointers, at every depth, whose value in the Task document the step reads differs from its base's."""
    objective_changed: list[Pointer] = Field(default_factory=list)
    """The same, in the Objective document."""
    objective: ObjectiveDocument | None
    objective_source: StudioTrialObjectiveSource | None = None
    input_overrides: dict[str, str] = Field(default_factory=dict)
    task_types: list[StudioTaskType]
    tasks: list[StudioTaskChoice]
    variables: list[ScopedValue]
    signals: list[ScopedValue]
    epochs: list[StudioTrialEpoch]
    period_seconds: float
    n_steps: int | None
    batch_size: int | None
    """How many trials the step runs at once (a Train's batch size, an Evaluate's sampled Trials), which bounds how many
    the clock samples; null for an Evaluate of a fixed cohort's conditions."""
    reads_model: bool
    """Whether the paired task reads the model's outputs, so that no trial can be drawn from it without a model."""
    inputs: list[StudioTrialInput]
    outputs: list[StudioTrialOutput]
    resets: list[StudioTrialReset]
    fields: list[StudioTrialField]
    problems: list[Problem]
    files: list[StudioProjectFilePatch]
    created: list[CreateNamedDocument] = Field(default_factory=list)
    """Documents the edit creates, written before ``files``, which may name them."""
    refusal: StudioSweptParameterMissing | None = None
    """Why the chosen document was not taken: it lacks a parameter the step sweeps."""
