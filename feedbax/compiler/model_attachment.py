"""Apply authored attachment differences and derive executable Evaluate patches."""

from __future__ import annotations

from dataclasses import dataclass, field

from feedbax.component_registry import ComponentRegistry
from feedbax.compiler.model_document import LoweredModelDocument, ModelCompilationProblem, lower_model_document
from feedbax.contracts.graph import GraphSpec, StudioTaskBindingSpec
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.model_address import ModelAddress, WireElement
from feedbax.contracts.model_attachment import AddModelAttachment, ChangeModelAttachment, ModelAttachmentDifference
from feedbax.contracts.model_document import ModelDocument


def _chain(draft: dict, address: ModelAddress, lowered: GraphSpec) -> list[dict]:
    graph = draft
    scoped = lowered
    for name in address.composites:
        child = graph["interior"]["components"].get(name)
        if child is None or child["kind"] != "composite":
            raise ValueError("Attachment owner composite no longer exists")
        graph = child
        scoped = (scoped.subgraphs or {}).get(name) if isinstance(scoped, GraphSpec) else None
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
        node = scoped.nodes.get(end.component) if scoped is not None else None
        if component is not None and node is not None:
            if end.direction in {"input", "output"} and end.port in getattr(node, end.direction + "_ports"):
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


def resolve_model_attachments(
    document: str, model: ModelDocument, lowered: GraphSpec,
    differences: tuple[ModelAttachmentDifference, ...],
) -> ModelDocument:
    if not differences:
        return model
    draft = model.model_dump(mode="json")
    chains: dict[str, _Chain] = {}
    for difference in differences:
        address = difference.address
        element = address.element
        owner = address.model_copy(update={"element": element.parent})
        try:
            if document != address.document:
                raise ValueError("Evaluate subject does not use the addressed Model document")
            _check_attachment_site(owner, difference)
            owner_key = owner.to_text()
            if owner_key not in chains:
                chains[owner_key] = _Chain(_chain(draft, owner, lowered))
            chains[owner_key].apply(difference)
        except ValueError as error:
            raise ModelCompilationProblem(address, str(error)) from error
    for chain in chains.values():
        chain.finish()
    try:
        return ModelDocument.model_validate(draft)
    except ValueError as error:
        raise ModelCompilationProblem(differences[-1].address, str(error)) from error


def _graph_edits(before: GraphSpec, after: GraphSpec, path=("graph",)):
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
            if current is None:
                yield OverridePatch(path=target, op="remove")
            else:
                value = list(current) if isinstance(current, tuple) else current.model_dump(mode="json")
                yield OverridePatch(path=target, op="add" if previous is None else "replace", value=value)
    for name, child in (before.subgraphs or {}).items():
        if isinstance(child, GraphSpec):
            yield from _graph_edits(child, after.subgraphs[name], (*path, "subgraphs", name))


def lower_evaluate_attachments(
    document: str, model: ModelDocument, binding: StudioTaskBindingSpec | None, components: ComponentRegistry,
    differences: tuple[ModelAttachmentDifference, ...],
) -> tuple[LoweredModelDocument, tuple[OverridePatch, ...]]:
    """Derive the scoped wire/port variation from the resolved authored model."""
    before = lower_model_document(document, model, components, binding)
    resolved = resolve_model_attachments(document, model, before.graph, differences)
    after = lower_model_document(document, resolved, components, binding)
    return after, tuple(sorted(_graph_edits(before.graph, after.graph), key=lambda edit: edit.path))
