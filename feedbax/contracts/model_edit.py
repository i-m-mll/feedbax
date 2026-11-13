"""Addressed model commands and their transaction results."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, JsonValue, StringConstraints

from feedbax.contracts.base import StrictModel
from feedbax.contracts.model_address import ModelAddress, ModelElementId, SignalSource, SignalTarget
from feedbax.contracts.model_document import (
    CompositeComponent, LeafComponent, ModelName, ModelPhysicalConnection,
    ModelDocument, ModelWire, PhysicalPort, SignalInput, SignalOutput,
)
from feedbax.contracts.project import Sha256
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.studio_project import StudioProjectSnapshot


class ModelPosition(StrictModel):
    x: Annotated[float, Field(allow_inf_nan=False)]
    y: Annotated[float, Field(allow_inf_nan=False)]


class AddComponent(StrictModel):
    op: Literal["add_component"]
    graph: ModelAddress
    component: LeafComponent | CompositeComponent
    position: ModelPosition


class ModelSelection(StrictModel):
    selection: Annotated[tuple[ModelAddress, ...], Field(min_length=1)]


class DeleteElements(ModelSelection):
    op: Literal["delete"]


class ConnectSignal(StrictModel):
    op: Literal["connect"]
    graph: ModelAddress
    source: SignalSource
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


class SetField(StrictModel):
    op: Literal["set_field"]
    address: ModelAddress
    value: JsonValue


class BringSignalIn(StrictModel):
    op: Literal["bring_signal_in"]
    source_graph: ModelAddress
    source: SignalSource
    destination: ModelAddress


class ModelClipboard(StrictModel):
    components: dict[ModelElementId, LeafComponent | CompositeComponent]
    wires: dict[ModelElementId, ModelWire]
    connections: dict[ModelElementId, ModelPhysicalConnection]
    positions: dict[str, ModelPosition] = Field(default_factory=dict)


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
        AddComponent | DeleteElements | ConnectSignal | ConnectPhysical | AddPort | RemovePort
        | RenameElement | MovePort | SetField | BringSignalIn | CopyComponents | CutComponents
        | DuplicateComponents | PasteComponents | GroupComponents | UngroupComposite,
        Field(discriminator="op"),
    ]


class ModelEditProblem(StrictModel):
    address: ModelAddress
    detail: str


class ModelEditState(StrictModel):
    document: ModelDocument
    positions: dict[str, ModelPosition] = Field(default_factory=dict)


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
