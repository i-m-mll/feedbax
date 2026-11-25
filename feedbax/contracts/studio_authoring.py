"""Wire contracts for authoring discovery and revision-bound field editing."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue, StrictFloat, StrictInt

from feedbax.contracts.base import StrictModel
from feedbax.contracts.component import ComponentDefinition
from feedbax.contracts.expression_registry import FunctionInfo
from feedbax.contracts.operation import DatasetPort, OperationDefinition
from feedbax.contracts.expression_values import ValueType
from feedbax.contracts.project import RootDocument, Sha256, StepKind
from feedbax.contracts.task_document import TaskDocument
from feedbax.contracts.project_inputs import LoadedModelReference, NonemptyText, ProjectName, StepReference
from feedbax.contracts.request_refs import Name
from feedbax.contracts.jobs import StudioProjectWindowMapProblem, StudioSavedTrainKey
from feedbax.contracts.studio_project import CreateNamedDocument, StudioProjectFilePatch
from feedbax.contracts.studio_scope import StudioStoreChoice


class InputProducer(StrictModel):
    """The step whose rows an input's selections read, and the product they read of each row."""

    kind: StepKind
    product: str
    operation_outputs: bool
    """Whether a selection may instead name an Analyze operation's dataset product, read from the card that holds it."""


class InputRole(StrictModel):
    """A step input as the planner resolves it: whether it is required, whether it takes a list of selections, the
    producer its selections read, the documents it may read instead, and whether its checkpoint may be a finite set."""

    required: bool
    gathers: bool
    producers: list[InputProducer]
    documents: list[Literal["model", "task", "loaded_model"]]
    checkpoint_sets: bool


class StudioAuthoringDeclaration(StrictModel):
    key: str
    label: str
    renamed_from: tuple[str, ...]
    settings_schema: dict[str, JsonValue]
    outputs: dict[str, DatasetPort] = Field(default_factory=dict, exclude_if=lambda value: not value)


class StudioAuthoringSurface(StrictModel):
    project_grammar: dict[str, JsonValue]
    step_payloads: dict[str, dict[str, JsonValue]]
    document_schemas: dict[str, dict[str, JsonValue]]
    training_methods: list[StudioAuthoringDeclaration]
    synthesis_methods: list[StudioAuthoringDeclaration]


class StudioAuthoredDocumentFamily(StrictModel):
    schema_id: str | None
    payload_schema: dict[str, JsonValue]


class StudioAuthoringOrigin(StrictModel):
    package: str
    distribution: str
    version: str


class StudioAuthoringRegistry(StrictModel):
    entries: list[
        ComponentDefinition | OperationDefinition | StudioAuthoringDeclaration | FunctionInfo
    ]
    origins: dict[str, StudioAuthoringOrigin]


class StudioNamedValueType(StrictModel):
    name: str
    type: ValueType


class StudioLibraryDocument(StrictModel):
    """A document of the library ``library`` that ``distribution`` ships, taken into a project whole."""

    distribution: str
    library: str
    name: ProjectName
    document: RootDocument


class StudioTaskCatalog(StrictModel):
    """What New offers for a task (Blank, every library's task documents; the installed Python task types are the
    ``python_tasks`` registry's), and the types a variable or a signal can have."""

    blank: TaskDocument
    documents: list[StudioLibraryDocument]
    value_types: list[StudioNamedValueType]


class StudioAuthoringCatalog(StrictModel):
    """Authored schemas and the registry declarations those schemas can name."""

    project_grammar: dict[str, JsonValue]
    step_grammar: dict[str, JsonValue]
    step_payloads: dict[str, dict[str, JsonValue]]
    document_families: dict[str, StudioAuthoredDocumentFamily]
    registries: dict[str, StudioAuthoringRegistry]
    step_inputs: dict[str, dict[str, InputRole]]
    """Each step kind's inputs by name."""
    analyze_inputs: dict[str, InputRole]
    """The input an Analyze document's declaration of each kind gives its step."""
    tasks: StudioTaskCatalog


class StudioAuthoringOption(StrictModel):
    value: str
    label: str


class StudioAuthoringChoice(StrictModel):
    kind: Literal["choice"]
    options: list[StudioAuthoringOption]


class StudioAuthoringNumber(StrictModel):
    kind: Literal["number"]
    min: float | None = None
    max: float | None = None
    step: int | None = None


class StudioAuthoringSource(StrictModel):
    kind: Literal["document", "step", "variant"]
    name: str
    path: str


class StudioAuthoringField(StrictModel):
    id: str
    label: str
    value: str | int | float | None
    source: StudioAuthoringSource | None
    control: (
        Annotated[StudioAuthoringChoice | StudioAuthoringNumber, Field(discriminator="kind")] | None
    )


class StudioAuthoringChange(StrictModel):
    """Sets a declared field, or any value under ``/settings`` that the method's settings model accepts.

    At ``/method`` or a class-valued field the value is the chosen method's or class's key; the server keeps each
    current setting the chosen one declares and accepts, and takes its defaults for the rest.
    """

    field: str
    value: JsonValue


class StudioToSweep(StrictModel):
    kind: Literal["toSweep"]


class StudioSweepEdit(StrictModel):
    kind: Literal["addValue", "toList"]
    sweep_id: ProjectName


class StudioSetSweepValue(StrictModel):
    kind: Literal["setSweepValue"]
    sweep_id: ProjectName
    value_id: ProjectName
    value: StrictInt | StrictFloat


class StudioNameSweepValue(StrictModel):
    kind: Literal["nameSweepValue"]
    sweep_id: ProjectName
    value_id: ProjectName
    label: NonemptyText | None


class StudioAuthoringSweepChange(StrictModel):
    field: str
    edit: Annotated[
        StudioToSweep | StudioSweepEdit | StudioSetSweepValue | StudioNameSweepValue,
        Field(discriminator="kind"),
    ]


class StudioRunSelection(StrictModel):
    """The runs an edit is made for: a named run and, when they are not the whole step, the selected rows.

    Rows that are exactly the runs of a set of values per sweep, named in ``axes``, are stored by those values;
    other rows are stored by row. ``chosen_steps`` are the steps the user named for an edit of a document, none being
    the edited step alone; the edit also reaches the steps based on them.
    """

    run: StudioSavedTrainKey | None = None
    rows: list[str] = Field(default_factory=list)
    axes: dict[ProjectName, Annotated[list[Name], Field(min_length=1)]] = Field(default_factory=dict)
    chosen_steps: list[ProjectName] = Field(default_factory=list)


class StudioClassToSweep(StrictModel):
    kind: Literal["to_sweep"]


class StudioClassToValue(StrictModel):
    """Turns the sweep into its member ``value_id``: the step takes that member's class, parameters and edits."""

    kind: Literal["to_value"]
    value_id: ProjectName


class StudioAddClassMember(StrictModel):
    """Adds the class ``key`` to the sweep; a parameter the step holds that the class lacks moves to the members that
    have it."""

    kind: Literal["add_member"]
    key: str


class StudioRemoveClassMember(StrictModel):
    kind: Literal["remove_member"]
    value_id: ProjectName


class StudioClassSweepChange(StrictModel):
    """An edit of the sweep over the class-valued field ``field``, a pointer in the Train definition."""

    field: str
    edit: Annotated[
        StudioClassToSweep | StudioClassToValue | StudioAddClassMember | StudioRemoveClassMember,
        Field(discriminator="kind"),
    ]


class StudioTrainAuthoringRequest(StudioRunSelection):
    revision: Sha256
    store_choice: Sha256 | None = None
    change: StudioAuthoringChange | None = None
    class_sweep: StudioClassSweepChange | None = None
    reset: str | None = None
    """A pointer in the Train definition to give the value of the base of the document it is read from."""


class StudioTrainStartRequest(StrictModel):
    revision: Sha256
    source: StepReference | LoadedModelReference | None = None
    """The starting checkpoint or Loaded Model. With no source, initialize from the Train's Model."""


class StudioTrainStartEdit(StrictModel):
    revision: Sha256
    files: list[StudioProjectFilePatch]
    problems: list[StudioProjectWindowMapProblem] = Field(default_factory=list)


class StudioClassMember(StrictModel):
    id: ProjectName
    label: str
    value: str
    """The member's class key."""


class StudioClassSweep(StrictModel):
    id: ProjectName
    members: list[StudioClassMember]


class StudioClassField(StrictModel):
    """A class-valued field of the Train definition as the selected runs have it.

    ``classes`` are the keys of the selected runs' classes, in the sweep's order when it is swept. The step's fields
    below it are the parameters every one of those classes has.
    """

    id: str
    label: str
    options: list[StudioAuthoringOption]
    classes: list[str]
    sweep: StudioClassSweep | None = None
    selected_value: ProjectName | None = None
    """The member whose runs include the named run."""
    can_sweep: bool = False


class StudioTrainAuthoring(StrictModel):
    project_id: ProjectName
    step_id: ProjectName
    revision: Sha256
    fields: list[StudioAuthoringField]
    settings: dict[str, JsonValue] | None
    """The selected runs' settings: the named run's."""
    mixed: list[str] = Field(default_factory=list)
    """Pointers in the definition where the selected runs hold different values."""
    files: list[StudioProjectFilePatch]
    created: list[CreateNamedDocument] = Field(default_factory=list)
    """Documents the edit creates, written before ``files``, which may name them."""
    documents: list[ProjectName]
    store_choices: list[StudioStoreChoice] | None = None
    sweep_problems: list[StudioProjectWindowMapProblem] = Field(default_factory=list)
    classes: list[StudioClassField] = Field(default_factory=list)
    removed_values: dict[str, list[str]] = Field(default_factory=dict)
    """Per sweep, the values whose runs the edit removes from the project."""
    changed: list[str] = Field(default_factory=list)
    """Pointers in the definition whose value is a difference of the based document it is read from."""
