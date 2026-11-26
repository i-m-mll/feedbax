"""Name the members of a pinned Model view from its authored document."""

from collections import defaultdict
from collections.abc import Mapping

from feedbax.compiler.model_document import ModelCompilationProblem
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_document import (
    CompositeComponent, ModelDocument, ModelPortAttachments, ModelWire, ProbeAttachment, WireAttachment,
)
from feedbax.contracts.stored_model import (
    StoredCompositeComponent, StoredModelDocument, StoredModelPortAttachments, StoredModelWire,
    StoredProbeAttachment, StoredWireAttachment,
)


def _unmatched(path: tuple[str, ...], kind: str) -> ModelCompilationProblem:
    return ModelCompilationProblem(
        ModelAddress.at_stored_path(path),
        f"The authored Model has no matching {kind} for this member of the pinned revision.",
    )


def _matched[T](members: Mapping[str, T], path: tuple[str, ...], kind: str) -> T:
    if (member := members.get(path[-1])) is None:
        raise _unmatched(path, kind)
    return member


def _attachment_labels[T: WireAttachment](
    pinned: tuple[T, ...], authored: Mapping[str, StoredWireAttachment], path: tuple[str, ...],
) -> tuple[T, ...]:
    shown = []
    for item in pinned:
        if isinstance(item, ProbeAttachment):
            place = (*path, item.id)
            match = _matched(authored, place, "probe")
            if not isinstance(match, StoredProbeAttachment):
                raise _unmatched(place, "probe")
            item = item.model_copy(update={"name": match.label})
        shown.append(item)
    return tuple(shown)


def _port_attachment_labels(
    pinned: ModelPortAttachments, authored: StoredModelPortAttachments, path: tuple[str, ...],
) -> ModelPortAttachments:
    return pinned.model_copy(update={
        side: {
            port: _attachment_labels(chain, getattr(authored, side).get(port, {}), (*path, side, port))
            for port, chain in getattr(pinned, side).items()
        }
        for side in ("sources", "sinks")
    })


def _ends(wire: ModelWire | StoredModelWire) -> tuple[str, str]:
    return wire.source.model_dump_json(), wire.target.model_dump_json()


def pinned_model_labels[T: (ModelDocument, CompositeComponent)](
    pinned: T, authored: StoredModelDocument | StoredCompositeComponent, path: tuple[str, ...] = (),
) -> T:
    """Copy presentation labels across fitted members while retaining pinned structure and addresses."""
    ports = pinned.ports.model_copy(update={
        direction: tuple(
            port.model_copy(update={"name": _matched(
                getattr(authored.ports, direction), (*path, "ports", direction, port.id), "port",
            ).label})
            for port in getattr(pinned.ports, direction)
        )
        for direction in ("inputs", "outputs", "physical")
    })
    interior = (*path, "interior")
    components = {}
    for key, component in pinned.interior.components.items():
        at = (*interior, "components", key)
        match = _matched(authored.interior.components, at, "component")
        if component.kind != match.kind:
            raise _unmatched(at, "component")
        if isinstance(component, CompositeComponent) and isinstance(match, StoredCompositeComponent):
            component = pinned_model_labels(component, match, at)
        components[key] = component.model_copy(update={
            "name": match.label,
            "attachments": _port_attachment_labels(component.attachments, match.attachments, (*at, "attachments")),
        })

    authored_wires = authored.interior.wires
    by_ends: dict[tuple[str, str], list[StoredModelWire]] = defaultdict(list)
    for wire in authored_wires.values():
        by_ends[_ends(wire)].append(wire)
    wires = {}
    for key, wire in pinned.interior.wires.items():
        at = (*interior, "wires", key)
        match = authored_wires.get(key)
        if match is None:
            matches = by_ends.get(_ends(wire), ())
            if len(matches) != 1:
                raise _unmatched(at, "wire")
            match = matches[0]
        wires[key] = wire.model_copy(update={
            "attachments": _attachment_labels(wire.attachments, match.attachments, (*at, "attachments")),
        })
    return pinned.model_copy(update={
        "ports": ports,
        "interior": pinned.interior.model_copy(update={
            "components": components,
            "wires": wires,
            "attachments": _port_attachment_labels(
                pinned.interior.attachments, authored.interior.attachments, (*interior, "attachments"),
            ),
        }),
    })
