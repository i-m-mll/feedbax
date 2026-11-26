"""Revision-bound Studio field controls and prepared edits."""

from typing import Annotated, Literal

from pydantic import Field

from feedbax.contracts.base import StrictModel
from feedbax.contracts.expression_registry import FunctionClass, FunctionParameter
from feedbax.contracts.expression_site import ExpressionPartSite, FieldKind
from feedbax.contracts.expression_values import (
    FieldValue, Level, Problem, SetVariableOrigin, SignalSpace, UnresolvedValue, ValueBase, ValueType,
)
from feedbax.contracts.jobs import StudioPlanReviewSource
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_edit import SelectedRuns, ModelEditCommand
from feedbax.contracts.project import Sha256
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.project_sweep import RangeValues, SampleValues
from feedbax.contracts.studio_authoring import StudioRunSelection
from feedbax.contracts.studio_project import CreateNamedDocument, StudioProjectFilePatch
from feedbax.contracts.studio_scope import StudioStoreChoice
from feedbax.contracts.time_units import TimeValue


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


class FieldResetEdit(StrictModel):
    """Give the field the value it has in the document the shown one changes it from."""

    kind: Literal["reset"]


class ModelFieldRequest(StudioRunSelection):
    revision: Sha256
    store_choice: Sha256 | None = None
    document: ProjectName
    owner: ModelAddress
    evaluate_scope: SelectedRuns | None = None
    address: ModelAddress | None = None
    edit: Annotated[
        FieldTextEdit | FieldKindEdit | FieldNameEdit | FieldResetEdit, Field(discriminator="kind"),
    ] | None = None


class StudioSweepValue(StrictModel):
    id: ProjectName
    label: str | None
    value: int | float | UnresolvedValue


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


class StudioDensity(StrictModel):
    """A draw's density at evenly spaced points from ``low`` to ``high``, scaled to a largest value of 1."""

    low: float
    high: float
    values: list[float]


class StudioValueStructure(StrictModel):
    """The value a field's text reads as, with the text of each positional argument of its root."""

    value: FieldValue
    arguments: list[str] = Field(default_factory=list)
    density: StudioDensity | None = None


class StudioFieldFunction(StrictModel):
    """The expression function information the editor presents."""

    id: str
    signature: str
    label: str | None = Field(default=None, exclude_if=lambda value: value is None)
    parameters: tuple[FunctionParameter, ...] = ()
    display_class: FunctionClass
    min_args: int
    max_args: int | None
    keywords: tuple[str, ...] = ()


class StudioFieldNameType(StrictModel):
    base: ValueBase
    shape: tuple[int, ...] = ()
    space: SignalSpace | None = Field(default=None, exclude_if=lambda value: value is None)


class StudioFieldName(StrictModel):
    id: str
    name: str
    type: StudioFieldNameType
    list_count: int | None = None
    set_entry: SetVariableOrigin | None = Field(default=None, exclude_if=lambda value: value is None)


class StudioFieldSite(StrictModel):
    expected: ValueType
    finest: Level
    names: tuple[StudioFieldName, ...] = ()
    structural: bool = False
    list_count: int | None = None
    repeat_count: int | None = None
    kinds: tuple[FieldKind, ...]
    functions: str
    """Key in this response's function lists."""
    parts: tuple[ExpressionPartSite, ...]
    name_paths: dict[str, tuple[str, ...]] = Field(default_factory=dict)


class StudioFieldSiteTables(StrictModel):
    """Field editor data shared by the views in one response."""

    sites: dict[str, StudioFieldSite] = Field(default_factory=dict)
    function_lists: dict[str, tuple[StudioFieldFunction, ...]] = Field(default_factory=dict)


class StudioFieldView(StrictModel):
    label: str
    text: str
    structure: StudioValueStructure | None = None
    effective: str | None = None
    """The value the selected runs share where it differs from the document's."""
    mixed: bool = False
    """The selected runs hold different values."""
    unit: str | None = None
    conversion: TimeValue | None = None
    """The equivalent in the other time unit under the selected Model's Step."""
    site: str
    """Key in this response's sites."""
    sweep: StudioSweepList | StudioSweepRange | StudioSweepSampler | None = None
    selected_value: ProjectName | None = None
    can_sweep: bool = False
    can_change_axis: bool = False
    members: list[StudioSweepValue] = Field(default_factory=list)
    changed: bool = False
    """The value is set by a difference of the shown document: from its base, or in an evaluation's Model from the
    nearest document it shares with the subject's."""


class ModelFieldView(StudioFieldView):
    address: ModelAddress
    address_text: str
    problems: list[Problem] = Field(default_factory=list)
    sources: list[StudioPlanReviewSource] = Field(default_factory=list)


class TrainFieldView(StudioFieldView):
    id: str


class TrainFieldRequest(StudioRunSelection):
    revision: Sha256
    store_choice: Sha256 | None = None
    field: str | None = None
    edit: Annotated[
        FieldTextEdit | FieldKindEdit | FieldNameEdit | FieldResetEdit, Field(discriminator="kind"),
    ] | None = None


class TrainFieldResponse(StudioFieldSiteTables):
    fields: list[TrainFieldView] = Field(default_factory=list)
    files: list[StudioProjectFilePatch] = Field(default_factory=list)
    created: list[CreateNamedDocument] = Field(default_factory=list)
    """Documents the edit creates, written before ``files``, which may name them."""
    dependencies: list[ProjectName] = Field(default_factory=list)
    problems: list[Problem] = Field(default_factory=list)
    store_choices: list[StudioStoreChoice] | None = None
    removed_values: dict[str, list[str]] = Field(default_factory=dict)
    """Per sweep, the values whose runs the edit removes from the project."""


class ModelFieldResponse(StudioFieldSiteTables):
    fields: list[ModelFieldView] = Field(default_factory=list)
    files: list[StudioProjectFilePatch] = Field(default_factory=list)
    created: list[CreateNamedDocument] = Field(default_factory=list)
    problems: list[Problem] = Field(default_factory=list)
    store_choices: list[StudioStoreChoice] | None = None
    removed_values: dict[str, list[str]] = Field(default_factory=dict)
    """Per sweep, the values whose runs the edit removes from the project."""
    command: ModelEditCommand | None = None
