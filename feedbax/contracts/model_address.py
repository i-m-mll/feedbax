"""Addresses relative to an authored Model's stored tree."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, StrictStr, StringConstraints

from feedbax.contracts.base import StrictModel
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.strict_json import strict_model_validate_json

ModelElementId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]+$")]
FieldIndex = Annotated[int, Field(strict=True, ge=0, le=2**53 - 1)]
FieldPath = Annotated[tuple[StrictStr | FieldIndex, ...], Field(min_length=1)]


class ComponentPortEnd(StrictModel):
    owner: Literal["component"]
    component: ModelElementId
    port: ModelElementId


class InputBoundaryEnd(StrictModel):
    owner: Literal["inputs"]
    port: ModelElementId


class OutputBoundaryEnd(StrictModel):
    owner: Literal["outputs"]
    port: ModelElementId


class PhysicalBoundaryEnd(StrictModel):
    owner: Literal["boundary"]
    port: ModelElementId


SignalSource = Annotated[ComponentPortEnd | InputBoundaryEnd, Field(discriminator="owner")]
SignalTarget = Annotated[ComponentPortEnd | OutputBoundaryEnd, Field(discriminator="owner")]
PhysicalEnd = Annotated[ComponentPortEnd | PhysicalBoundaryEnd, Field(discriminator="owner")]
class ComponentPortAddress(ComponentPortEnd):
    direction: Literal["input", "output", "physical"]


PortLocation = Annotated[
    ComponentPortAddress | InputBoundaryEnd | OutputBoundaryEnd | PhysicalBoundaryEnd,
    Field(discriminator="owner"),
]


class GraphElement(StrictModel):
    kind: Literal["graph"]


class InputsElement(StrictModel):
    kind: Literal["inputs"]


class OutputsElement(StrictModel):
    kind: Literal["outputs"]


class ComponentElement(StrictModel):
    kind: Literal["component"]
    id: ModelElementId


class WireElement(StrictModel):
    kind: Literal["wire"]
    id: ModelElementId


class PhysicalConnectionElement(StrictModel):
    kind: Literal["physical_connection"]
    id: ModelElementId


class PortElement(StrictModel):
    kind: Literal["port"]
    end: PortLocation


AttachmentOwner = Annotated[WireElement | PortElement, Field(discriminator="kind")]


class AttachmentElement(StrictModel):
    kind: Literal["attachment"]
    parent: AttachmentOwner
    id: ModelElementId


ModelElement = Annotated[
    GraphElement | ComponentElement | WireElement | PhysicalConnectionElement
    | PortElement | AttachmentElement,
    Field(discriminator="kind"),
]


class FieldElement(StrictModel):
    kind: Literal["field"]
    parent: ModelElement
    path: FieldPath


AddressedElement = Annotated[
    GraphElement | InputsElement | OutputsElement | ComponentElement | WireElement | PhysicalConnectionElement
    | PortElement | AttachmentElement | FieldElement,
    Field(discriminator="kind"),
]


class ModelAddress(StrictModel):
    model_config = ConfigDict(frozen=True)

    composites: tuple[ModelElementId, ...]
    element: AddressedElement

    @classmethod
    def at_state(cls, composites: tuple[str, ...], component: str, symbol: str) -> ModelAddress:
        """Address a component's declared state by its semantic symbol."""
        return cls(
            composites=composites,
            element=FieldElement(
                kind="field", parent=ComponentElement(kind="component", id=component), path=("state", symbol),
            ),
        )

    def stored_path(self) -> tuple[str, ...]:
        """Locate the addressed member in the stored tree when it has a keyed path."""
        scope = tuple(token for owner in self.composites for token in ("interior", "components", owner))

        def element_path(element: AddressedElement) -> tuple[str, ...]:
            if isinstance(element, FieldElement):
                return (*element_path(element.parent), *map(str, element.path))
            if isinstance(element, ComponentElement):
                return "interior", "components", element.id
            if isinstance(element, WireElement):
                return "interior", "wires", element.id
            if isinstance(element, PhysicalConnectionElement):
                return "interior", "connections", element.id
            if isinstance(element, AttachmentElement):
                return (*element_path(element.parent), "attachments")
            return ()

        return (*scope, *element_path(self.element))

    @classmethod
    def at_stored_path(cls, path: Sequence[str]) -> ModelAddress:
        """Address the member at ``path`` in the stored tree: the inverse of ``stored_path`` for a member that is
        not an attachment."""
        parts = tuple(path)
        composites: list[str] = []
        while len(parts) > 5 and parts[:2] == ("interior", "components") and parts[3:5] == ("interior", "components"):
            composites.append(parts[2])
            parts = parts[3:]
        element: ModelElement = GraphElement(kind="graph")
        if len(parts) >= 3 and parts[0] == "interior":
            member = {
                "components": lambda id: ComponentElement(kind="component", id=id),
                "wires": lambda id: WireElement(kind="wire", id=id),
                "connections": lambda id: PhysicalConnectionElement(kind="physical_connection", id=id),
            }.get(parts[1])
            if member is not None:
                element, parts = member(parts[2]), parts[3:]
        return cls(
            composites=tuple(composites),
            element=FieldElement(kind="field", parent=element, path=parts) if parts else element,
        )

    def label(self) -> str:
        """Name the addressed member as a dotted path inside its document, for reasons."""

        def element_label(element: AddressedElement) -> tuple[str, ...]:
            if isinstance(element, FieldElement):
                return (*element_label(element.parent), *map(str, element.path))
            if isinstance(element, PortElement):
                end = element.end
                return (end.component if isinstance(end, ComponentPortEnd) else end.owner, end.port)
            if isinstance(element, AttachmentElement):
                return (*element_label(element.parent), element.id)
            if isinstance(element, GraphElement):
                return ()
            if isinstance(element, InputsElement | OutputsElement):
                return (element.kind,)
            return (element.id,)

        return ".".join((*self.composites, *element_label(self.element)))

    def relocated(self, moves: Mapping[str, ModelAddress]) -> ModelAddress:
        """Follow the identity of this element through a graph move."""
        if (direct := moves.get(self.to_text())) is not None:
            return direct
        element = self.element
        if isinstance(element, FieldElement | AttachmentElement):
            owner = ModelAddress(composites=self.composites, element=element.parent).relocated(moves)
            return ModelAddress(composites=owner.composites, element=element.model_copy(update={"parent": owner.element}))
        if isinstance(element, PortElement) and isinstance(element.end, ComponentPortEnd):
            owner = ModelAddress(composites=self.composites, element=ComponentElement(
                kind="component", id=element.end.component,
            )).relocated(moves)
            assert isinstance(owner.element, ComponentElement)
            return ModelAddress(composites=owner.composites, element=element.model_copy(update={
                "end": element.end.model_copy(update={"component": owner.element.id}),
            }))
        for i, name in enumerate(self.composites):
            part = ModelAddress(composites=self.composites[:i], element=ComponentElement(kind="component", id=name))
            if (owner := moves.get(part.to_text())) is not None:
                assert isinstance(owner.element, ComponentElement)
                return self.model_copy(update={"composites": (*owner.composites, owner.element.id, *self.composites[i + 1:])})
        return self

    def to_text(self) -> str:
        """Encode the relative address with the shared canonical JSON codec."""
        return canonical_json_v2_bytes(self.model_dump(mode="json")).decode("utf-8")

    @classmethod
    def from_text(cls, text: str) -> ModelAddress:
        address = strict_model_validate_json(cls, text)
        if address.to_text() != text:
            raise ValueError("Model address text must be canonical JSON")
        return address
