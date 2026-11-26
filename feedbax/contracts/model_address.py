"""Addresses relative to an authored Model's stored tree."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, StrictStr, StringConstraints

from feedbax.contracts.base import StrictModel
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.strict_json import strict_json_loads, strict_model_validate_json

_ELEMENT_ID = re.compile(r"[A-Za-z0-9_-]+")
_FIELD_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")
_AUTHORED_TOKEN = re.compile(rf'"(?:[^"\\\x00-\x1f]|\\.)*"|{_ELEMENT_ID.pattern}|[./#]')
_PORT_HEADS: dict[Literal["input", "output", "physical"], str] = {
    "input": "in", "output": "out", "physical": "physical",
}
_PORT_DIRECTIONS = {head: direction for direction, head in _PORT_HEADS.items()}

ModelElementId = Annotated[str, StringConstraints(pattern=f"^{_ELEMENT_ID.pattern}$")]
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
            if isinstance(element, PortElement):
                end = element.end
                if isinstance(end, ComponentPortAddress):
                    group = {"input": "inputs", "output": "outputs", "physical": "physical"}[end.direction]
                    return "interior", "components", end.component, "ports", group, end.port
                return "ports", "physical" if end.owner == "boundary" else end.owner, end.port
            if isinstance(element, AttachmentElement):
                owner = element.parent
                if isinstance(owner, WireElement):
                    return (*element_path(owner), "attachments", element.id)
                end = owner.end
                if isinstance(end, ComponentPortAddress):
                    side = "sources" if end.direction == "output" else "sinks"
                    return "interior", "components", end.component, "attachments", side, end.port, element.id
                side = "sources" if end.owner == "inputs" else "sinks"
                return "interior", "attachments", side, end.port, element.id
            return ()

        return (*scope, *element_path(self.element))

    @classmethod
    def at_stored_path(cls, path: Sequence[str]) -> ModelAddress:
        """Address the member at a keyed Model path."""
        parts = tuple(path)
        composites: list[str] = []
        while len(parts) >= 5 and parts[:2] == ("interior", "components") and parts[3] == "interior":
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
        if isinstance(element, WireElement) and len(parts) >= 2 and parts[0] == "attachments":
            element, parts = AttachmentElement(kind="attachment", parent=element, id=parts[1]), parts[2:]
        elif isinstance(element, ComponentElement) and len(parts) >= 4 and parts[0] == "attachments" and parts[1] in {"sources", "sinks"}:
            end = ComponentPortAddress(
                owner="component", component=element.id, port=parts[2],
                direction="output" if parts[1] == "sources" else "input",
            )
            element, parts = AttachmentElement(kind="attachment", parent=PortElement(kind="port", end=end), id=parts[3]), parts[4:]
        elif isinstance(element, GraphElement) and len(parts) >= 5 and parts[:2] == ("interior", "attachments") and parts[2] in {"sources", "sinks"}:
            end = InputBoundaryEnd(owner="inputs", port=parts[3]) if parts[2] == "sources" else OutputBoundaryEnd(
                owner="outputs", port=parts[3],
            )
            element, parts = AttachmentElement(kind="attachment", parent=PortElement(kind="port", end=end), id=parts[4]), parts[5:]
        elif isinstance(element, GraphElement | ComponentElement) and len(parts) >= 3 and parts[0] == "ports" and parts[1] in {"inputs", "outputs", "physical"}:
            group, port = parts[1:3]
            if isinstance(element, ComponentElement):
                direction = {"inputs": "input", "outputs": "output", "physical": "physical"}[group]
                end = ComponentPortAddress(owner="component", component=element.id, port=port, direction=direction)
            elif group == "inputs":
                end = InputBoundaryEnd(owner="inputs", port=port)
            elif group == "outputs":
                end = OutputBoundaryEnd(owner="outputs", port=port)
            else:
                end = PhysicalBoundaryEnd(owner="boundary", port=port)
            element, parts = PortElement(kind="port", end=end), parts[3:]
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

    def to_authored_text(self) -> str:
        """Encode the relative address for authored documents and expressions."""
        return "/".join((*self.composites, _authored_element(self.element)))

    @classmethod
    def from_authored_text(cls, text: str) -> ModelAddress:
        """Read an authored address in its one canonical spelling."""
        tokens = _authored_tokens(text)

        def take(expected: str | None = None) -> str:
            if not tokens or expected is not None and tokens[0] != expected:
                raise ValueError("Invalid authored Model address")
            return tokens.popleft()

        def id_after(separator: str | None = None) -> str:
            if separator is not None:
                take(separator)
            value = take()
            if not _ELEMENT_ID.fullmatch(value):
                raise ValueError("Invalid element ID in authored Model address")
            return value

        composites = []
        while len(tokens) > 1 and tokens[1] == "/":
            composites.append(id_after())
            take("/")

        head = take()
        element: AddressedElement
        match head:
            case "graph":
                element = GraphElement(kind="graph")
            case "inputs" if not tokens:
                element = InputsElement(kind="inputs")
            case "outputs" if not tokens:
                element = OutputsElement(kind="outputs")
            case "inputs":
                element = PortElement(kind="port", end=InputBoundaryEnd(owner="inputs", port=id_after(".")))
            case "outputs":
                element = PortElement(kind="port", end=OutputBoundaryEnd(owner="outputs", port=id_after(".")))
            case "boundary":
                element = PortElement(kind="port", end=PhysicalBoundaryEnd(owner="boundary", port=id_after(".")))
            case "component":
                element = ComponentElement(kind="component", id=id_after("."))
            case "wire":
                element = WireElement(kind="wire", id=id_after("."))
            case "connection":
                element = PhysicalConnectionElement(kind="physical_connection", id=id_after("."))
            case _ if head in _PORT_DIRECTIONS:
                element = PortElement(kind="port", end=ComponentPortAddress(
                    owner="component", direction=_PORT_DIRECTIONS[head], component=id_after("."), port=id_after("."),
                ))
            case _:
                raise ValueError("Invalid element in authored Model address")

        if tokens and tokens[0] == "#":
            if not isinstance(element, PortElement | WireElement):
                raise ValueError("An attachment address requires a port or wire owner")
            element = AttachmentElement(kind="attachment", parent=element, id=id_after("#"))

        if tokens:
            assert not isinstance(element, InputsElement | OutputsElement)
            path = []
            while tokens:
                take(".")
                path.append(_authored_field_part(take()))
            element = FieldElement(kind="field", parent=element, path=tuple(path))

        address = cls(composites=tuple(composites), element=element)
        if address.to_authored_text() != text:
            raise ValueError("Model address must use canonical authored text")
        return address


def _authored_tokens(text: str) -> deque[str]:
    tokens: deque[str] = deque()
    start = 0
    while start < len(text):
        found = _AUTHORED_TOKEN.match(text, start)
        if found is None:
            raise ValueError("Invalid token in authored Model address")
        tokens.append(found.group())
        start = found.end()
    return tokens


def _authored_field_part(token: str) -> str | int:
    if token.startswith('"'):
        value = strict_json_loads(token)
        assert isinstance(value, str)
        return value
    if _FIELD_WORD.fullmatch(token):
        return token
    if token.isdigit():
        return int(token)
    raise ValueError("Invalid field segment in authored Model address")


def _authored_element(element: AddressedElement) -> str:
    if isinstance(element, FieldElement):
        path = ".".join(
            str(part) if isinstance(part, int) or _FIELD_WORD.fullmatch(part)
            else canonical_json_v2_bytes(part).decode("utf-8")
            for part in element.path
        )
        return f"{_authored_element(element.parent)}.{path}"
    if isinstance(element, AttachmentElement):
        return f"{_authored_element(element.parent)}#{element.id}"
    if isinstance(element, PortElement):
        end = element.end
        if isinstance(end, ComponentPortAddress):
            return f"{_PORT_HEADS[end.direction]}.{end.component}.{end.port}"
        return f"{end.owner}.{end.port}"
    if isinstance(element, GraphElement | InputsElement | OutputsElement):
        return element.kind
    if isinstance(element, PhysicalConnectionElement):
        return f"connection.{element.id}"
    return f"{element.kind}.{element.id}"


@dataclass(frozen=True)
class ModelFieldTarget:
    address: ModelAddress
    scope: tuple[Literal["steps", "documents"], str] | None = None

    @property
    def step(self) -> str | None:
        return self.scope[1] if self.scope and self.scope[0] == "steps" else None

    @property
    def document(self) -> str | None:
        return self.scope[1] if self.scope and self.scope[0] == "documents" else None


@lru_cache(maxsize=4096)
def authored_model_field_target(path: str) -> ModelFieldTarget | None:
    """Read a Model field in braces and any step or document scope that precedes it."""
    scope = None
    if path.startswith(("steps.", "documents.")) and ".{" in path:
        prefix, _, path = path.partition(".{")
        match = re.fullmatch(r"(steps|documents)\.([A-Za-z0-9_][A-Za-z0-9_-]*)", prefix)
        if match is None:
            raise ValueError("A scoped Model field target must name one step or document")
        group, name = match.groups()
        scope = ("steps" if group == "steps" else "documents", name)
        path = "{" + path
    elif not path.startswith("{"):
        return None
    if not path.endswith("}"):
        raise ValueError("A Model sweep target must close its address braces")
    address = ModelAddress.from_authored_text(path[1:-1])
    if not isinstance(address.element, FieldElement):
        raise ValueError("A Model sweep target must name a field")
    return ModelFieldTarget(address, scope)
