"""Addressed model commands and their transaction results."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from feedbax.contracts.base import StrictModel
from feedbax.contracts.model_address import ModelAddress, ModelElementId, SignalSource, SignalTarget
from feedbax.contracts.model_attachment import ModelAttachmentDifference, SetField as SetField
from feedbax.contracts.model_document import (
    CompositeComponent, LeafComponent, ModelName, ModelPhysicalConnection,
    ModelDocument, ModelWire, PhysicalPort, SignalInput, SignalOutput, SignalType,
)
from feedbax.contracts.jobs import StudioSavedTrainKey
from feedbax.contracts.project import Sha256
from feedbax.contracts.project_inputs import AxisValueId, ProjectName, RowKey
from feedbax.contracts.request_refs import Name
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
        AddComponent | MoveCards | FlipComponent | CollapseGraph | RouteWire
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
    document: ProjectName
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


class ModelEditApplied(StrictModel):
    """``undo_token`` is given when the edit wrote the shown document's file and layout alone; ``created`` names the
    documents the edit made."""

    status: Literal["applied"]
    snapshot: StudioProjectSnapshot
    changed: tuple[ModelAddress, ...]
    clipboard: ModelClipboard | None = None
    undo_token: ModelUndoToken | None = None
    created: tuple[ProjectName, ...] = ()


class ModelEditRefused(StrictModel):
    status: Literal["refused"]
    problem: ModelEditProblem


class ModelEditResponse(StrictModel):
    result: Annotated[ModelEditApplied | ModelEditRefused, Field(discriminator="status")]
