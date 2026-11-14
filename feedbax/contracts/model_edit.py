"""Addressed model commands and their transaction results."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from feedbax.contracts.base import StrictModel
from feedbax.contracts.model_address import ModelAddress, ModelElementId, SignalSource, SignalTarget
from feedbax.contracts.model_attachment import EditableModelDifference, ModelAttachmentDifference, SetField as SetField
from feedbax.contracts.model_document import (
    CompositeComponent, LeafComponent, ModelName, ModelPhysicalConnection,
    ModelDocument, ModelWire, PhysicalPort, SignalInput, SignalOutput, SignalType,
)
from feedbax.contracts.jobs import StudioSavedTrainKey
from feedbax.contracts.project import Sha256
from feedbax.contracts.project_inputs import ProjectName, RowKey
from feedbax.contracts.studio_project import StudioProjectSnapshot
from feedbax.contracts.studio_project import StudioProjectFilePatch


class ModelPosition(StrictModel):
    x: Annotated[float, Field(allow_inf_nan=False)]
    y: Annotated[float, Field(allow_inf_nan=False)]


class AddComponent(StrictModel):
    op: Literal["add_component"]
    graph: ModelAddress
    component: LeafComponent | CompositeComponent
    position: ModelPosition
    connect_from: SignalSource | None = None
    connect_to_port: ModelElementId | None = None


class ModelSelection(StrictModel):
    selection: Annotated[tuple[ModelAddress, ...], Field(min_length=1)]


class PlaceComponent(StrictModel):
    address: ModelAddress
    position: ModelPosition


class MoveComponents(StrictModel):
    op: Literal["move_components"]
    placements: Annotated[tuple[PlaceComponent, ...], Field(min_length=1)]


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
    grow_target: bool = False


class ConnectTaskSignal(StrictModel):
    op: Literal["connect_task_signal"]
    graph: ModelAddress
    run: StudioSavedTrainKey
    signal: str
    target: SignalTarget


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
    component: LeafComponent | CompositeComponent
    input_port: ModelElementId
    output_port: ModelElementId
    position: ModelPosition
    flipped: bool = False


class BringSignalIn(StrictModel):
    op: Literal["bring_signal_in"]
    source_graph: ModelAddress
    source: SignalSource
    destination: ModelAddress


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
        AddComponent | MoveComponents | FlipComponent | CollapseGraph | RouteWire
        | DeleteElements | ConnectSignal | ConnectTaskSignal | ConnectPhysical | AddPort | RemovePort
        | RenameElement | MovePort | SetField | SetModelAttachment | MoveWireAttachment | InsertComponentOnWire
        | BringSignalIn | CopyComponents | CutComponents
        | DuplicateComponents | PasteComponents | GroupComponents | UngroupComposite,
        Field(discriminator="op"),
    ]


class ModelEditProblem(StrictModel):
    address: ModelAddress
    detail: str


class TaskSignalSource(StrictModel):
    owner: Literal["task"]
    run: StudioSavedTrainKey
    signal: str


class ModelConnectionReviewRequest(StrictModel):
    revision: Sha256
    graph: ModelAddress
    source: Annotated[SignalSource | TaskSignalSource, Field(discriminator="owner")]


class ModelConnectionVerdict(StrictModel):
    target: SignalTarget
    reason: str | None


class ModelConnectionReview(StrictModel):
    targets: tuple[ModelConnectionVerdict, ...]
    library: tuple[ModelLibraryConnection, ...] = ()


class ModelLibraryConnection(StrictModel):
    component: str
    target_port: ModelElementId


class ModelProjectionRequest(StrictModel):
    revision: Sha256
    run: StudioSavedTrainKey
    document: ProjectName


class ModelPortProjection(StrictModel):
    address: ModelAddress
    type: SignalType


class ModelTaskSignal(StrictModel):
    id: str
    label: str
    type: SignalType


class ModelProjection(StrictModel):
    ports: tuple[ModelPortProjection, ...] = ()
    task_signals: tuple[ModelTaskSignal, ...] = ()
    task_inputs: tuple[ModelAddress, ...] = ()
    objective_ports: tuple[ModelAddress, ...] = ()
    resolved_model: ModelDocument | None = None


class ModelEditState(ModelLayout):
    document: ModelDocument


class ModelEditResult(StrictModel):
    state: ModelEditState
    changed: tuple[ModelAddress, ...] = ()
    clipboard: ModelClipboard | None = None


class ModelEditRequest(StrictModel):
    revision: Sha256
    presentation_revision: Sha256
    document: ProjectName
    command: ModelEditCommand


type ModelUndoToken = Annotated[str, StringConstraints(min_length=1)]


class ModelUndoRequest(StrictModel):
    revision: Sha256
    presentation_revision: Sha256
    token: ModelUndoToken


class EvaluateAttachmentStep(StrictModel):
    kind: Literal["step"]


class EvaluateAttachmentAxis(StrictModel):
    kind: Literal["axis_value"]
    axis: ProjectName
    value: ProjectName


class EvaluateAttachmentRows(StrictModel):
    kind: Literal["rows"]
    rows: Annotated[tuple[RowKey, ...], Field(min_length=1)]


class EvaluateAttachmentVariant(StrictModel):
    kind: Literal["variant"]
    variant: ProjectName


EvaluateAttachmentScope = Annotated[
    EvaluateAttachmentStep | EvaluateAttachmentAxis | EvaluateAttachmentRows | EvaluateAttachmentVariant,
    Field(discriminator="kind"),
]


class EvaluateAttachmentReadRequest(StrictModel):
    revision: Sha256
    run: StudioSavedTrainKey
    scope: EvaluateAttachmentScope


class EvaluateAttachmentRequest(EvaluateAttachmentReadRequest):
    difference: EditableModelDifference


class StudioEvaluateValueSource(StrictModel):
    kind: Literal["trained", "evaluate"]
    step: ProjectName


class EvaluateAttachmentState(StrictModel):
    revision: Sha256
    records: tuple[ModelAttachmentDifference, ...]
    effective: tuple[ModelAttachmentDifference, ...]
    resolved_model: ModelDocument | None = None
    source: StudioEvaluateValueSource | None = None
    field_sources: dict[str, StudioEvaluateValueSource] = Field(default_factory=dict)


class EvaluateAttachmentEdit(StrictModel):
    revision: Sha256
    changed: ModelAddress
    file: StudioProjectFilePatch


class ModelEditApplied(StrictModel):
    status: Literal["applied"]
    snapshot: StudioProjectSnapshot
    changed: tuple[ModelAddress, ...]
    clipboard: ModelClipboard | None = None
    undo_token: ModelUndoToken | None = None


class ModelEditRefused(StrictModel):
    status: Literal["refused"]
    problem: ModelEditProblem


class ModelEditResponse(StrictModel):
    result: Annotated[ModelEditApplied | ModelEditRefused, Field(discriminator="status")]
