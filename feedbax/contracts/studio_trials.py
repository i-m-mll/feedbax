"""Revision-bound Task, Objective and Model pairing projections for Studio."""

from typing import Annotated, Literal

from pydantic import Field, JsonValue, field_serializer

from feedbax.contracts.project_sweep import SetEntryParts, SetTrialSampling
from feedbax.contracts.base import StrictModel
from feedbax.contracts.expression_values import Level, Problem, ScopedValue, ValueType, WeightPart
from feedbax.contracts.jobs import StudioSavedTrainKey
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_edit import (
    AxisValueRuns, ListedRuns, StudioEvaluateValueSource,
)
from feedbax.contracts.objective_document import ObjectiveDocument
from feedbax.contracts.project import Pointer, ProjectValueEdit, Sha256
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.studio_field import FieldKindEdit, FieldNameEdit, FieldTextEdit, StudioFieldView
from feedbax.contracts.studio_project import CreateNamedDocument, StudioProjectFilePatch
from feedbax.contracts.studio_scene import ScenePoint
from feedbax.contracts.studio_scope import StudioStoreChoice
from feedbax.contracts.task_document import EpochSteps, PythonTaskSelection, TaskDocument
from feedbax.contracts.task_structure import TaskStructureChange
from feedbax.contracts.trial_record import TrialAnchorPlace, TrialSignalPlace, TrialVariablePlace


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
    id: str
    address: ModelAddress
    name: str
    type: ValueType | None
    required: bool
    signal_id: str | None
    explicit: bool


class StudioTrialOutput(StrictModel):
    address: ModelAddress
    value: ScopedValue


class StudioTrialReset(StrictModel):
    """A value of the paired Model that a variable of the Task may initialize at the start of each trial."""

    address: ModelAddress
    path: tuple[str, ...]
    """Display segments for enclosing composites, the component, and the state or parameter."""
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
    input_id: str
    signal_id: str | None


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


class StudioRelatePort(StrictModel):
    kind: Literal["port"]
    address: ModelAddress


class StudioRelatePoint(StrictModel):
    kind: Literal["point"]
    position: ScenePoint


class StudioTrialRelate(StrictModel):
    kind: Literal["relate"]
    first: Annotated[
        TrialAnchorPlace | TrialVariablePlace | TrialSignalPlace | StudioRelatePort, Field(discriminator="kind")
    ]
    second: Annotated[
        TrialAnchorPlace | TrialVariablePlace | TrialSignalPlace | StudioRelatePort | StudioRelatePoint,
        Field(discriminator="kind"),
    ] | None = None


class StudioTrialStepScope(StrictModel):
    kind: Literal["step"]
    chosen_steps: list[ProjectName] = Field(default_factory=list)


class StudioTrialAuthoringRequest(StrictModel):
    revision: Sha256
    run: StudioSavedTrainKey
    change: Annotated[
        StudioTrialDocumentEdit | StudioTrialDocumentReset | StudioTrialBindingEdit | StudioTrialSelectDocument
        | StudioTrialExpressionEdit | StudioTrialStructureEdit | StudioTrialRelate,
        Field(discriminator="kind"),
    ] | None = None
    store_choice: Sha256 | None = Field(default=None, exclude_if=lambda value: value is None)
    scope: Annotated[
        StudioTrialStepScope | AxisValueRuns | ListedRuns,
        Field(discriminator="kind"),
    ] | None = None


class StudioTrialEpoch(StrictModel):
    """An epoch of the paired task; ``steps`` once the Task pairs with the Model."""

    id: str
    name: str
    steps: EpochSteps | None


class StudioTrialEpochInstance(StudioTrialEpoch):
    """One compiled occurrence of an authored epoch in trial order."""

    occurrence: int = Field(ge=0)
    """Zero-based slot in the trial, unique within this list."""
    steps: EpochSteps


class StudioTaskChoice(StrictModel):
    document: ProjectName
    problems: list[Problem]


class StudioTrialObjectiveSource(StrictModel):
    kind: Literal["train", "own"]
    step: ProjectName


class StudioSetFieldEntry(StrictModel):
    entry: ProjectName | None
    label: str | None = None
    weight: WeightPart | SetEntryParts | None = None


class StudioSetField(StrictModel):
    axis: ProjectName
    set: ProjectName
    sampling: SetTrialSampling
    entries: list[StudioSetFieldEntry]


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
    set_fields: list[StudioSetField] = Field(default_factory=list)
    task_types: list[StudioTaskType]
    tasks: list[StudioTaskChoice]
    variables: list[ScopedValue]
    signals: list[ScopedValue]
    epochs: list[StudioTrialEpoch]
    """Authored epochs in editor column order."""
    epoch_instances: list[StudioTrialEpochInstance] = Field(default_factory=list)
    """Compiled epoch occurrences in trial order, with each repeat already expanded."""
    period_seconds: float | None
    n_steps: int | None
    batch_size: int | None
    """How many trials the step runs at once (a Train's batch size, an Evaluate's sampled Trials), which bounds how many
    the clock samples; null for an Evaluate of a fixed cohort's conditions."""
    reads_model: bool
    """Whether the paired task reads the model's outputs, so that no trial can be drawn from it without a model."""
    input_bindings: dict[str, str | None]
    inputs: list[StudioTrialInput]
    outputs: list[StudioTrialOutput]
    resets: list[StudioTrialReset]
    fields: list[StudioTrialField]
    problems: list[Problem]
    related_term: str | None = Field(default=None, exclude_if=lambda value: value is None)
    files: list[StudioProjectFilePatch]
    created: list[CreateNamedDocument] = Field(default_factory=list)
    """Documents the edit creates, written before ``files``, which may name them."""
    store_choices: list[StudioStoreChoice] | None = Field(
        default=None, min_length=1, exclude_if=lambda value: value is None,
    )
    """An unanswered edit. No files or documents are proposed until the caller chooses a store."""

    @field_serializer("task", "objective")
    def _authored_document(self, document: TaskDocument | PythonTaskSelection | ObjectiveDocument | None):
        # The editor uses field presence to choose between add and replace.
        return document.model_dump(mode="json", exclude_unset=True) if document is not None else None
