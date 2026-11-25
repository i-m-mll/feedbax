"""Managed Analyze authoring projections and the edits of a step's analysis document."""

from typing import Annotated, Literal

from pydantic import Field, JsonValue

from feedbax.contracts.analysis_document import DatasetSource
from feedbax.contracts.base import StrictModel
from feedbax.contracts.dataset import DimensionSize, Name
from feedbax.contracts.figure_document import FigureDocument
from feedbax.contracts.jobs import StudioProjectWindowMapProblem, StudioSavedTrainKey
from feedbax.contracts.operation import DatasetPort, OperationDefinition
from feedbax.contracts.project import DocumentEdit, Pointer, Sha256
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.studio_project import CreateNamedDocument, StudioProjectFilePatch


class StudioAnalyzeWire(StrictModel):
    """Bind ``role`` of the operation or figure ``element`` to ``source``, reviewed against the plan's ports."""

    kind: Literal["wire"]
    element: ProjectName
    role: ProjectName
    source: DatasetSource


class StudioAnalyzeEdit(StrictModel):
    """Set or remove values of the composed analysis document: an element added, a parameter, a caption."""

    kind: Literal["edit"]
    edits: list[DocumentEdit] = Field(min_length=1)


class StudioAnalyzeRemove(StrictModel):
    """Remove the operation or figure ``element`` and every input bound to it."""

    kind: Literal["remove"]
    element: ProjectName


class StudioAnalyzeReset(StrictModel):
    """Give each of ``paths`` the value the document's base has there."""

    kind: Literal["reset"]
    paths: list[Pointer] = Field(min_length=1)


type StudioAnalyzeChange = Annotated[
    StudioAnalyzeWire | StudioAnalyzeEdit | StudioAnalyzeRemove | StudioAnalyzeReset, Field(discriminator="kind")
]


class StudioAnalyzeRequest(StrictModel):
    """Read the step's authoring catalog, and store ``change`` for the ``chosen_steps`` (the step alone when none)."""

    revision: Sha256
    key: StudioSavedTrainKey
    change: StudioAnalyzeChange | None = None
    chosen_steps: list[ProjectName] = Field(default_factory=list)


class StudioAnalyzeAuthoring(StrictModel):
    revision: Sha256
    document: ProjectName
    operations: list[OperationDefinition]
    states: DatasetPort | None
    """The port common to the saved states the step's ``states`` input reads; none when it has no such input."""
    sources: dict[ProjectName, dict[ProjectName, list[DatasetSource]]]
    """Compatible inputs and outputs of plannable operations, by receiving element and role."""
    variables: dict[str, dict[str, list[str]]]
    """Each operation's parameter choices: the matching variables of a dataset selector, and the bound label
    dimensions of a parameter that lists the dimensions a port takes whole."""
    variable_names: dict[str, str] = Field(default_factory=dict)
    """Display names for the dataset field IDs offered in ``variables``."""
    figure: FigureDocument
    changed: list[Pointer] = Field(default_factory=list)
    """The pointers at which the document's composed content differs from its base's."""
    files: list[StudioProjectFilePatch] = Field(default_factory=list)
    created: list[CreateNamedDocument] = Field(default_factory=list)
    problems: list[StudioProjectWindowMapProblem] = Field(default_factory=list)
    """Problems in this run's authoring catalog, with field pointers relative to their analysis elements."""


MAX_OPERATION_DATASET_PAGE = 256


class StudioOperationDatasetPage(StrictModel):
    """A bounded page of one field, flattened in the descriptor's dimension order."""

    field: Name
    offset: DimensionSize
    total: DimensionSize
    values: list[JsonValue] = Field(max_length=MAX_OPERATION_DATASET_PAGE)
    """Nonfinite numbers and wide integers use strings; complex numbers give real and imaginary parts."""
