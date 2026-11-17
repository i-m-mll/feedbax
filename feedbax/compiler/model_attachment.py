"""Apply authored Model differences and derive executable Evaluate patches."""

from __future__ import annotations

from dataclasses import dataclass, field

from feedbax.component_registry import ComponentRegistry
from feedbax.compiler.model_document import LoweredModelDocument, ModelCompilationProblem, lower_model_document
from feedbax.contracts.graph import GraphSpec, StudioTaskBindingSpec
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.model_address import AttachmentElement, FieldElement, ModelAddress, WireElement
from feedbax.contracts.model_attachment import (
    AddModelAttachment, AddModelComponent, ChangeModelAttachment, ModelAttachmentDifference, ModelDifference,
    ModelWireDifference, SetField, is_generated_attachment_node,
)
from feedbax.contracts.model_document import ModelDocument, ModelWire
from feedbax.contracts.model_edit import ModelEditState
from feedbax.studio.model_edit import ModelEditDraft
from feedbax.studio.model_ports import ModelRegistryPorts


def _chain(draft: dict, address: ModelAddress, ports: ModelRegistryPorts) -> list[dict]:
    graph = draft
    for name in address.composites:
        child = graph["interior"]["components"].get(name)
        if child is None or child["kind"] != "composite":
            raise ValueError("Attachment owner composite no longer exists")
        graph = child
    element = address.element
    if isinstance(element, WireElement):
        wire = graph["interior"]["wires"].get(element.id)
        if wire is None:
            raise ValueError("Attachment owner wire no longer exists")
        return wire["attachments"]
    end = element.end
    interior = graph["interior"]
    if end.owner in {"inputs", "outputs"}:
        if not any(port["id"] == end.port for port in graph["ports"][end.owner]):
            raise ValueError("Attachment owner port no longer exists")
        direction = "sources" if end.owner == "inputs" else "sinks"
        return interior["attachments"][direction].setdefault(end.port, [])
    if end.owner == "component":
        component = interior["components"].get(end.component)
        if component is not None and end.direction in {"input", "output"}:
            present = (
                ports.port(component, end.direction, end.port) is not None if component["kind"] == "leaf"
                else any(port["id"] == end.port for port in component["ports"][end.direction + "s"])
            )
            if present:
                direction = "sources" if end.direction == "output" else "sinks"
                return component["attachments"][direction].setdefault(end.port, [])
    raise ValueError("Attachment owner port no longer exists")


def _check_attachment_site(owner: ModelAddress, difference: ModelAttachmentDifference) -> None:
    if isinstance(owner.element, WireElement) or not isinstance(difference, (AddModelAttachment, ChangeModelAttachment)):
        return
    kind = difference.attachment.kind
    end = owner.element.end
    if kind == "delay":
        raise ValueError("A delay attaches to a wire")
    if kind == "intervention" and not (end.owner == "inputs" or end.owner == "component" and end.direction == "output"):
        raise ValueError("An intervention attaches to a source port")


@dataclass
class _Chain:
    original: list[dict]
    items: dict[str, dict] = field(init=False)
    positions: dict[str, tuple[str | None, ModelAddress]] = field(default_factory=dict)
    known: set[str] = field(init=False)

    def __post_init__(self):
        self.items = {item["id"]: item for item in self.original}
        self.known = set(self.items)

    def apply(self, difference: ModelAttachmentDifference) -> None:
        address = difference.address
        identity = address.element.id
        if isinstance(difference, AddModelAttachment):
            if identity in self.known:
                raise ValueError("Attachment ID repeats on this wire or port")
            self.known.add(identity)
        elif identity not in self.items:
            raise ValueError("The addressed attachment no longer exists")
        elif not isinstance(difference, ChangeModelAttachment):
            del self.items[identity]
            self.positions.pop(identity, None)
            return
        elif self.items[identity]["kind"] != difference.attachment.kind:
            raise ValueError("Changing an attachment must preserve its kind")
        self.items[identity] = difference.attachment.model_dump(mode="json")
        if "after" in difference.model_fields_set:
            self.positions.pop(identity, None)
            self.positions[identity] = (difference.after, address)

    def finish(self) -> None:
        children = {}
        for identity, (anchor, address) in self.positions.items():
            if anchor is not None and anchor not in self.items:
                raise ModelCompilationProblem(address, f"Attachment anchor {anchor!r} no longer exists on this wire or port")
            children.setdefault(anchor, []).append(identity)
        roots = [item["id"] for item in self.original if item["id"] in self.items and item["id"] not in self.positions]
        ordered = []
        pending = [*reversed(roots), *reversed(children.get(None, ()))]
        while pending:
            identity = pending.pop()
            ordered.append(identity)
            pending.extend(reversed(children.get(identity, ())))
        if unresolved := self.items.keys() - set(ordered):
            identity = next(name for name in self.positions if name in unresolved)
            raise ModelCompilationProblem(self.positions[identity][1], "Attachment positions form an anchor cycle")
        self.original[:] = [self.items[identity] for identity in ordered]


def resolve_effective_model(
    document: str, model: ModelDocument, components: ComponentRegistry,
    differences: tuple[ModelDifference, ...], *, functions=None,
) -> ModelDocument:
    if not differences:
        return model
    ports = ModelRegistryPorts(components)
    edit = ModelEditDraft(document, ModelEditState(document=model), ports.port, ports.accepts)
    draft = edit.document
    chains: dict[str, _Chain] = {}

    def chain_for(element: AttachmentElement, address: ModelAddress) -> _Chain:
        owner = address.model_copy(update={"element": element.parent})
        key = owner.to_text()
        if key not in chains:
            chains[key] = _Chain(_chain(draft, owner, ports))
        return chains[key]

    for difference in differences:
        address = difference.address
        element = address.element
        try:
            if document != address.document:
                raise ValueError("This step does not use the addressed Model document")
            if isinstance(difference, AddModelComponent):
                _, graph = edit.context(address)
                if element.id in graph["components"]:
                    raise ValueError("The addressed component already exists")
                graph["components"][element.id] = difference.component.model_dump(mode="json")
                continue
            if isinstance(difference, ModelWireDifference):
                _, graph = edit.context(address)
                wires = graph["wires"]
                if difference.op == "add_wire":
                    if element.id in wires:
                        raise ValueError("The addressed wire already exists")
                    wires[element.id] = ModelWire(
                        source=difference.source, target=difference.target,
                    ).model_dump(mode="json")
                else:
                    if element.id not in wires:
                        raise ValueError("The addressed wire no longer exists")
                    wires[element.id].update(
                        source=difference.source.model_dump(mode="json"),
                        target=difference.target.model_dump(mode="json"),
                    )
                continue
            if isinstance(difference, SetField):
                if not isinstance(element, FieldElement) or any(not isinstance(part, str) for part in element.path):
                    raise ValueError("A Model difference must name a field using named parts")
                attachment = None
                if isinstance(element.parent, AttachmentElement):
                    attachment = chain_for(element.parent, address).items.get(element.parent.id)
                    if attachment is None:
                        raise ValueError("The addressed attachment no longer exists")
                elif element.parent.kind == "component":
                    _, graph = edit.context(address)
                    component = graph["components"].get(element.parent.id)
                    if component is not None and component["kind"] == "leaf" and len(element.path) > 1:
                        name = element.path[1]
                        declaration = components.get(component["type"])
                        declared = declaration is not None and name in declaration.params.model.model_fields
                        if name not in component["fields"] and not declared:
                            raise ValueError("The addressed component field no longer exists")
                edit.set_field(address, difference.value, functions=functions, attachment=attachment)
                continue
            owner = address.model_copy(update={"element": element.parent})
            _check_attachment_site(owner, difference)
            chain_for(element, address).apply(difference)
        except ValueError as error:
            raise ModelCompilationProblem(address, str(error)) from error
    for chain in chains.values():
        chain.finish()
    try:
        return ModelDocument.model_validate(draft)
    except ValueError as error:
        raise ModelCompilationProblem(differences[-1].address, str(error)) from error


def _graph_edits(before: GraphSpec, after: GraphSpec, path=("graph",)):
    projected = {"nodes", "wires", "output_bindings", "subgraphs"}
    if any(
        getattr(before, field) != getattr(after, field) for field in type(before).model_fields.keys() - projected
    ):
        raise ValueError("This Model field changes the trained graph interface or clock")
    for member in ("nodes", "wires", "output_bindings"):
        old = getattr(before, member)
        new = getattr(after, member)
        if member == "wires":
            old = {wire.authored_id(): wire for wire in old}
            new = {wire.authored_id(): wire for wire in new}
        for name in sorted(old.keys() | new.keys()):
            previous, current = old.get(name), new.get(name)
            if previous == current:
                continue
            target = ".".join((*path, member, name))
            if (
                member == "nodes" and previous is not None and current is not None
                and not is_generated_attachment_node(name, previous.type)
                and previous.model_copy(update={"params": current.params}) == current
                and previous.params.keys() <= current.params.keys()
            ):
                for parameter, value in sorted(current.params.items()):
                    if parameter not in previous.params or previous.params[parameter] != value:
                        yield OverridePatch(
                            path=f"{target}.params.{parameter}", value=value,
                            op="replace" if parameter in previous.params else "add",
                        )
                continue
            if current is None:
                yield OverridePatch(path=target, op="remove")
            else:
                value = list(current) if isinstance(current, tuple) else current.model_dump(mode="json")
                yield OverridePatch(path=target, op="add" if previous is None else "replace", value=value)
    old_children, new_children = before.subgraphs or {}, after.subgraphs or {}
    if old_children.keys() != new_children.keys():
        raise ValueError("Evaluate must retain the trained composite owners")
    for name, child in old_children.items():
        new_child = new_children[name]
        if child == new_child:
            continue
        if not isinstance(child, GraphSpec) or not isinstance(new_child, GraphSpec):
            raise ValueError("Evaluate must retain the trained physical composites")
        yield from _graph_edits(child, new_child, (*path, "subgraphs", name))


def lower_evaluate_attachments(
    document: str, model: ModelDocument, binding: StudioTaskBindingSpec | None, components: ComponentRegistry,
    differences: tuple[ModelDifference, ...], *, functions=None,
) -> tuple[LoweredModelDocument, LoweredModelDocument, tuple[OverridePatch, ...]]:
    """Derive the scoped wire/port variation from the resolved authored model."""
    before = lower_model_document(document, model, components, binding)
    resolved = resolve_effective_model(document, model, components, differences, functions=functions)
    after = lower_model_document(document, resolved, components, binding)
    try:
        derived = tuple(sorted(_graph_edits(before.graph, after.graph), key=lambda edit: edit.path))
    except ValueError as error:
        raise ModelCompilationProblem(differences[-1].address, str(error)) from error
    return before, after, derived
