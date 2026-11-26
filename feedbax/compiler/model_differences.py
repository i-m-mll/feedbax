"""The structural differences between two stored Models: what one adds, removes or rewires against the other."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_edit import (
    ModelAddedElement, ModelRemovedElement, ModelRewiredElement, ModelStructuralDifference,
)

Record = dict[str, Any]

_EDGES = (("wires", "wire", ("source", "target")), ("connections", "physical_connection", ("a", "b")))


def _address(scope: tuple[str, ...], element: Record) -> ModelAddress:
    return ModelAddress.model_validate({"composites": list(scope), "element": element})


def _attachment(scope: tuple[str, ...], parent: Record, id_: str) -> ModelAddress:
    return _address(scope, {"kind": "attachment", "parent": parent, "id": id_})


def _touches(edge: Record, ends: tuple[str, ...], components: set[str]) -> bool:
    return any(edge[end].get("component") in components for end in ends)


def _chain_differences(
    scope: tuple[str, ...], parent: Record, reference: Record, shown: Record,
) -> Iterator[ModelStructuralDifference]:
    for key, definition in reference.items():
        if key not in shown:
            yield ModelRemovedElement(change="removed", address=_attachment(scope, parent, key), definition=definition)
    for key in shown:
        if key not in reference:
            yield ModelAddedElement(change="added", address=_attachment(scope, parent, key))


def _port_chains(
    scope: tuple[str, ...], reference: Record, shown: Record, port_end,
) -> Iterator[ModelStructuralDifference]:
    for placement in ("sources", "sinks"):
        earlier, later = reference[placement], shown[placement]
        for port in dict.fromkeys([*earlier, *later]):
            yield from _chain_differences(
                scope, {"kind": "port", "end": port_end(placement, port)}, earlier.get(port, {}), later.get(port, {}),
            )


def _boundary_end(placement: str, port: str) -> Record:
    return {"owner": "inputs" if placement == "sources" else "outputs", "port": port}


def _component_end(component: str):
    def end(placement: str, port: str) -> Record:
        return {
            "owner": "component", "component": component, "port": port,
            "direction": "output" if placement == "sources" else "input",
        }
    return end


def _graph_differences(
    scope: tuple[str, ...], reference: Record, shown: Record, positions: Mapping[str, Any],
) -> Iterator[ModelStructuralDifference]:
    before, after = reference["interior"], shown["interior"]
    removed = set(before["components"]) - set(after["components"])
    added = set(after["components"]) - set(before["components"])
    for id_, component in before["components"].items():
        if id_ not in removed:
            continue
        address = _address(scope, {"kind": "component", "id": id_})
        wiring = {
            collection: {
                key: edge for key, edge in before.get(collection, {}).items()
                if key not in after.get(collection, {}) and _touches(edge, ends, {id_})
            } for collection, _, ends in _EDGES
        }
        yield ModelRemovedElement(
            change="removed", address=address, definition=component, position=positions.get(address.to_text()),
            wires=wiring["wires"], connections=wiring["connections"],
        )
    for id_ in after["components"]:
        if id_ in added:
            yield ModelAddedElement(change="added", address=_address(scope, {"kind": "component", "id": id_}))
    for collection, kind, ends in _EDGES:
        earlier, later = before.get(collection, {}), after.get(collection, {})
        for id_, edge in earlier.items():
            address = _address(scope, {"kind": kind, "id": id_})
            if id_ not in later:
                if not _touches(edge, ends, removed):
                    yield ModelRemovedElement(change="removed", address=address, definition=edge)
            elif any(edge[end] != later[id_][end] for end in ends):
                yield ModelRewiredElement(change="rewired", address=address, previous=edge)
        for id_, edge in later.items():
            if id_ not in earlier and not _touches(edge, ends, added):
                yield ModelAddedElement(change="added", address=_address(scope, {"kind": kind, "id": id_}))
    for id_, wire in before["wires"].items():
        if id_ in after["wires"]:
            yield from _chain_differences(
                scope, {"kind": "wire", "id": id_}, wire["attachments"], after["wires"][id_]["attachments"],
            )
    yield from _port_chains(scope, before["attachments"], after["attachments"], _boundary_end)
    for id_, component in before["components"].items():
        other = after["components"].get(id_)
        if other is None:
            continue
        yield from _port_chains(scope, component["attachments"], other["attachments"], _component_end(id_))
        if component["kind"] == other["kind"] == "composite":
            yield from _graph_differences((*scope, id_), component, other, positions)


def structural_differences(
    reference: Record, shown: Record, positions: Mapping[str, Any],
) -> tuple[ModelStructuralDifference, ...]:
    """What the stored Model ``shown`` adds, removes or rewires against ``reference``, both as dumped, at every depth
    of the composites both have; ``positions`` are ``reference``'s drawn places by address text."""
    return tuple(_graph_differences((), reference, shown, positions))
