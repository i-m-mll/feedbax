"""Addressed model commands and their transaction results."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.model_address import ModelAddress, ModelElementId, SignalSource, SignalTarget
from feedbax.contracts.model_attachment import ModelAttachmentDifference, SetField as SetField
from feedbax.contracts.model_document import (
    CompositeComponent, LeafComponent, ModelName, ModelPhysicalConnection,
    ModelDocument, ModelWire, PhysicalPort, PhysicalPortType, SignalInput, SignalOutput, SignalType, WireAttachment,
)
from feedbax.contracts.jobs import StudioProjectWindowMapProblem, StudioSavedTrainKey
from feedbax.contracts.project import Sha256
from feedbax.contracts.project_inputs import AxisValueId, ProjectName, RowKey
from feedbax.contracts.studio_scope import StudioStoreChoice
from feedbax.contracts.request_refs import Name
from feedbax.contracts.studio_project import StudioProjectSnapshot
from feedbax.contracts.studio_project import StudioProjectFilePatch


class ModelPosition(StrictModel):
    x: Annotated[float, Field(allow_inf_nan=False)]
    y: Annotated[float, Field(allow_inf_nan=False)]


class RegisteredComponent(StrictModel):
    kind: Literal["definition"]
    definition: Name


type ComponentToPlace = LeafComponent | CompositeComponent | RegisteredComponent


class AddComponent(StrictModel):
    op: Literal["add_component"]
    graph: ModelAddress
    component: ComponentToPlace
    position: ModelPosition
    id: ModelElementId | None = None
    connect_from: SignalSource | None = None
    connect_to_port: ModelElementId | None = None


class ModelSelection(StrictModel):
    selection: Annotated[tuple[ModelAddress, ...], Field(min_length=1)]


class PlaceCard(StrictModel):
    address: ModelAddress
    position: ModelPosition


class MoveCards(StrictModel):
    op: Literal["move_cards"]
    placements: Annotated[tuple[PlaceCard, ...], Field(min_length=1)]


class FlipComponent(StrictModel):
    op: Literal["flip_component"]
    address: ModelAddress


class CollapseGraph(StrictModel):
    op: Literal["collapse_graph"]
    graph: ModelAddress
    collapsed: bool


class RouteWire(StrictModel):
    op: Literal["route_wire"]
    address: ModelAddress
    waypoints: tuple[ModelPosition, ...]


class DeleteElements(ModelSelection):
    op: Literal["delete"]


class ConnectSignal(StrictModel):
    op: Literal["connect"]
    graph: ModelAddress
    source: SignalSource
    target: SignalTarget
    grow_source: bool = False
    grow_target: bool = False


class ConnectTaskSignal(StrictModel):
    op: Literal["connect_task_signal"]
    graph: ModelAddress
    run: StudioSavedTrainKey
    signal: str
    target: SignalTarget
    grow_target: bool = False


class ConnectPhysical(StrictModel):
    op: Literal["connect_physical"]
    graph: ModelAddress
    connection: ModelPhysicalConnection


class AddPort(StrictModel):
    op: Literal["add_port"]
    graph: ModelAddress
    direction: Literal["input", "output", "physical"]
    port: SignalInput | SignalOutput | PhysicalPort


class RemovePort(StrictModel):
    op: Literal["remove_port"]
    address: ModelAddress


class RenameElement(StrictModel):
    op: Literal["rename"]
    address: ModelAddress
    name: ModelName


class MovePort(StrictModel):
    op: Literal["move_port"]
    address: ModelAddress
    index: Annotated[int, Field(strict=True, ge=0)]


class SetModelAttachment(StrictModel):
    op: Literal["set_attachment"]
    difference: ModelAttachmentDifference


class MoveWireAttachment(StrictModel):
    op: Literal["move_wire_attachment"]
    address: ModelAddress
    destination: ModelAddress
    after: ModelElementId | None = None


class InsertComponentOnWire(StrictModel):
    op: Literal["insert_component"]
    address: ModelAddress
    component: ComponentToPlace
    input_port: ModelElementId
    output_port: ModelElementId
    position: ModelPosition
    flipped: bool = False
    id: ModelElementId | None = None


class InsertComponentOnPhysicalConnection(StrictModel):
    op: Literal["insert_physical_component"]
    address: ModelAddress
    component: LeafComponent | CompositeComponent
    a_port: ModelElementId
    b_port: ModelElementId
    position: ModelPosition
    flipped: bool = False
    id: ModelElementId | None = None


class BringSignalIn(StrictModel):
    op: Literal["bring_signal_in"]
    source_graph: ModelAddress
    source: SignalSource
    destination: ModelAddress


class ResetDifference(StrictModel):
    """Undo the structural difference at ``address``: remove what was added, restore what was removed with its
    wiring and place, or restore a connection's ends."""

    op: Literal["reset_difference"]
    address: ModelAddress


class ModelLayout(StrictModel):
    positions: dict[str, ModelPosition] = Field(default_factory=dict)
    flipped: dict[str, bool] = Field(default_factory=dict)
    collapsed: dict[str, bool] = Field(default_factory=dict)
    waypoints: dict[str, tuple[ModelPosition, ...]] = Field(default_factory=dict)


class ModelClipboard(ModelLayout):
    components: dict[ModelElementId, LeafComponent | CompositeComponent]
    wires: dict[ModelElementId, ModelWire]
    connections: dict[ModelElementId, ModelPhysicalConnection]


class CopyComponents(ModelSelection):
    op: Literal["copy"]


class CutComponents(ModelSelection):
    op: Literal["cut"]


class DuplicateComponents(ModelSelection):
    op: Literal["duplicate"]


class PasteComponents(StrictModel):
    op: Literal["paste"]
    graph: ModelAddress
    clipboard: ModelClipboard


class GroupComponents(ModelSelection):
    op: Literal["group"]
    name: ModelName


class UngroupComposite(StrictModel):
    op: Literal["ungroup"]
    address: ModelAddress


class ModelEditCommand(StrictModel):
    operation: Annotated[
        AddComponent | MoveCards | FlipComponent | CollapseGraph | RouteWire
        | DeleteElements | ConnectSignal | ConnectTaskSignal | ConnectPhysical | AddPort | RemovePort
        | RenameElement | MovePort | SetField | SetModelAttachment | MoveWireAttachment
        | InsertComponentOnWire | InsertComponentOnPhysicalConnection
        | BringSignalIn | CopyComponents | CutComponents
        | DuplicateComponents | PasteComponents | GroupComponents | UngroupComposite | ResetDifference,
        Field(discriminator="op"),
    ]


class ModelEditProblem(StrictModel):
    address: ModelAddress
    detail: str


class ModelGraphProblem(StrictModel):
    address: ModelAddress
    reason: str


class ModelUngroupEligibility(StrictModel):
    address: ModelAddress
    reason: str | None


class TaskSignalSource(StrictModel):
    owner: Literal["task"]
    run: StudioSavedTrainKey
    signal: str


class ModelWireReplacement(StrictModel):
    address: ModelAddress
    attachments: tuple[ModelAddress, ...]


class ModelConnectionVerdict(StrictModel):
    target: SignalTarget
    reason: str | None
    problem: str | None = None
    replaced: tuple[ModelWireReplacement, ...] = ()
    growing: bool = False


class ModelConnectionReview(StrictModel):
    targets: tuple[ModelConnectionVerdict, ...]
    library: tuple[ModelLibraryConnection, ...] = ()


class ModelLibraryConnection(StrictModel):
    component: str
    target_port: ModelElementId


class ModelProjectionRequest(StrictModel):
    """``rows`` and ``axes`` select runs of ``run``'s step as a Model edit does; when they are one sweep value or set
    entry, the projection gives that value's differences from the step's Model."""

    revision: Sha256
    run: StudioSavedTrainKey | None = None
    document: ProjectName
    rows: list[RowKey] = Field(default_factory=list)
    axes: dict[ProjectName, list[Name]] = Field(default_factory=dict)
    evaluate_scope: SelectedRuns | None = None


class ModelConnectionReviewRequest(ModelProjectionRequest):
    graph: ModelAddress
    source: Annotated[SignalSource | TaskSignalSource, Field(discriminator="owner")]
    grow_source: bool = False


class ModelPortProjection(StrictModel):
    address: ModelAddress
    type: SignalType
    readers: tuple[str, ...] = ()


class ModelPhysicalPortProjection(StrictModel):
    address: ModelAddress
    type: PhysicalPortType


class ModelTaskSignal(StrictModel):
    id: str
    label: str
    type: SignalType


class ModelTaskInput(ModelAddress):
    """The root Model input and the current Task signal it reads, by stable signal ID."""

    signal_id: str


class ModelAddedElement(StrictModel):
    """A component, wire, physical connection or attachment the shown Model has and the one it differs from lacks.
    The wiring of an added component is part of it."""

    change: Literal["added"]
    address: ModelAddress


class ModelRemovedElement(StrictModel):
    """An element the shown Model lacks, as the one it differs from has it.

    A removed component carries the ``wires`` and ``connections`` that reached it there, which are not reported apart;
    ``position`` is where it was drawn.
    """

    change: Literal["removed"]
    address: ModelAddress
    definition: LeafComponent | CompositeComponent | ModelWire | ModelPhysicalConnection | WireAttachment
    position: ModelPosition | None = None
    wires: dict[ModelElementId, ModelWire] = Field(default_factory=dict)
    connections: dict[ModelElementId, ModelPhysicalConnection] = Field(default_factory=dict)


class ModelRewiredElement(StrictModel):
    """A wire or physical connection both Models have whose ends differ; ``previous`` is the one differed from."""

    change: Literal["rewired"]
    address: ModelAddress
    previous: ModelWire | ModelPhysicalConnection


ModelStructuralDifference = Annotated[
    ModelAddedElement | ModelRemovedElement | ModelRewiredElement, Field(discriminator="change"),
]


class ModelProjection(StrictModel):
    """``base_differences`` are the shown document's structural differences from its base, when it has one;
    ``value_differences`` are the selected sweep value's or set entry's from the step's Model."""

    ports: tuple[ModelPortProjection, ...] = ()
    physical_ports: tuple[ModelPhysicalPortProjection, ...] = ()
    delay_steps: dict[str, int | None] = Field(default_factory=dict)
    task_signals: tuple[ModelTaskSignal, ...] = ()
    task_inputs: tuple[ModelTaskInput, ...] = ()
    problems: tuple[ModelGraphProblem, ...] = ()
    resolved_model: ModelDocument | None = None
    base_differences: tuple[ModelStructuralDifference, ...] = ()
    value_differences: tuple[ModelStructuralDifference, ...] = ()
    ungroup: tuple[ModelUngroupEligibility, ...] = ()


class ModelEditState(ModelLayout):
    document: ModelDocument


class ModelEditResult(StrictModel):
    state: ModelEditState
    relocations: dict[str, ModelAddress] = Field(default_factory=dict)
    changed: tuple[ModelAddress, ...] = ()
    clipboard: ModelClipboard | None = None
    problems: tuple[ModelGraphProblem, ...] = ()


class ModelEditRequest(StrictModel):
    """A command applied to the composed content of ``document``, stored by the document rule.

    Without ``step`` the edit is stored for every step that reads the document or one based on it. With it, the edit is
    made for ``step`` and the ``chosen_steps``, in ``variant`` when it names one other than the default, whose row then
    holds it. ``files`` are step-file patches written in the same transaction, ahead of the command, such as the removal
    of the sweep whose kept value the command sets.

    ``rows`` selects runs of ``step`` in the default variant, ``axes`` the sweep values they are exactly the runs of:
    the edit applies to the first row's Model and is stored for the selected runs, on the value when they are one value
    of a named axis or a set's entry, and otherwise on the sweep's exception for them.
    """

    revision: Sha256
    presentation_revision: Sha256
    document: ProjectName
    command: ModelEditCommand
    step: ProjectName | None = None
    chosen_steps: list[ProjectName] = Field(default_factory=list)
    variant: ProjectName | None = None
    rows: list[RowKey] = Field(default_factory=list)
    axes: dict[ProjectName, list[Name]] = Field(default_factory=dict)
    files: list[StudioProjectFilePatch] = Field(default_factory=list)
    store_choice: Sha256 | None = None

    @model_validator(mode="after")
    def _variant_has_step(self) -> ModelEditRequest:
        if self.variant is not None and self.step is None:
            raise ValueError("A Model edit made in a variant is made for a step")
        if (self.rows or self.axes) and (self.step is None or self.variant is not None):
            raise ValueError("A Model edit for selected runs is made for a step's runs in the default variant")
        if self.axes and not self.rows:
            raise ValueError("Sweep values are given with their runs")
        return self


type ModelUndoToken = Annotated[str, StringConstraints(min_length=1)]


class ModelUndoRequest(StrictModel):
    revision: Sha256
    presentation_revision: Sha256
    token: ModelUndoToken


class StepRuns(StrictModel):
    kind: Literal["step"]


class AxisValueRuns(StrictModel):
    kind: Literal["axis_value"]
    axis: ProjectName
    value: AxisValueId


class ListedRuns(StrictModel):
    kind: Literal["rows"]
    rows: Annotated[tuple[RowKey, ...], Field(min_length=1)]


class VariantRuns(StrictModel):
    kind: Literal["variant"]
    variant: ProjectName


SelectedRuns = Annotated[
    StepRuns | AxisValueRuns | ListedRuns | VariantRuns,
    Field(discriminator="kind"),
]


class EvaluateModelRequest(StrictModel):
    revision: Sha256
    run: StudioSavedTrainKey
    scope: SelectedRuns


class StudioEvaluateValueSource(StrictModel):
    kind: Literal["trained", "evaluate"]
    step: ProjectName


class EvaluateModelState(StrictModel):
    revision: Sha256
    editable: bool
    effective: tuple[ModelAttachmentDifference, ...]
    resolved_model: ModelDocument | None = None
    source: StudioEvaluateValueSource | None = None
    field_sources: dict[str, StudioEvaluateValueSource] = Field(default_factory=dict)
    changed_from: ProjectName | None = None
    """The nearest document the read Model document shares with the subject's, when they differ: what the
    evaluation changes is the read document's differences from it."""
    problems: list[StudioProjectWindowMapProblem] = Field(default_factory=list)
    ungroup: tuple[ModelUngroupEligibility, ...] = ()


class ModelEditApplied(StrictModel):
    """``undo_token`` is given when the edit wrote the shown document's file and layout alone; ``created`` names the
    documents the edit made."""

    status: Literal["applied"]
    snapshot: StudioProjectSnapshot
    changed: tuple[ModelAddress, ...]
    clipboard: ModelClipboard | None = None
    undo_token: ModelUndoToken | None = None
    created: tuple[ProjectName, ...] = ()
    problems: tuple[ModelGraphProblem, ...] = ()
    sweep_problems: tuple[StudioProjectWindowMapProblem, ...] = ()


class ModelEditRefused(StrictModel):
    status: Literal["refused"]
    problem: ModelEditProblem


class ModelEditAsked(StrictModel):
    status: Literal["asked"]
    store_choices: list[StudioStoreChoice] = Field(min_length=1)


class ModelEditResponse(StrictModel):
    result: Annotated[ModelEditApplied | ModelEditRefused | ModelEditAsked, Field(discriminator="status")]
