"""Apply a model edit atomically to the stored document and its presentation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from pydantic import ValidationError

from feedbax.contracts import model_edit as edits
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_attachment import AddModelAttachment, ChangeModelAttachment, ModelAttachmentDifference
from feedbax.contracts.model_document import (
    CompositeComponent,
    GraphInterior,
    ModelDocument,
    ModelPorts,
    PhysicalPort,
    SignalInput,
    SignalOutput,
    SignalType,
)

type Record = dict[str, Any]
type Direction = Literal["input", "output", "physical"]


@dataclass(frozen=True)
class Port:
    type: SignalType | str
    reads_state_only: bool = False


type LeafPorts = Callable[[Record, Direction, str], Port | None]
type AcceptComponent = Callable[[Record, str | None], bool]


class _Refusal(ValueError):
    def __init__(self, address: ModelAddress, detail: str):
        super().__init__(detail)
        self.address = address


def _address(document: str, scope: tuple[str, ...], kind: str, **element: Any) -> ModelAddress:
    return ModelAddress.model_validate({
        "document": document, "composites": scope, "element": {"kind": kind, **element},
    })


def _id(graph: Record, reserved: set[str] | None = None) -> str:
    used = graph["components"].keys() | graph["wires"].keys() | graph.get("connections", {}).keys()
    used |= reserved or set()
    while (key := uuid4().hex) in used:
        pass
    return key


def _card(owner: str, port: str) -> Record:
    return {"owner": owner, "port": port}


def _component(component: str, port: str) -> Record:
    return {"owner": "component", "component": component, "port": port}


def _signal(source: Record, target: Record, attachments: list | None = None) -> Record:
    return {"source": source, "target": target, "attachments": attachments or []}


def _same_port(left: SignalType | str, right: SignalType | str) -> bool:
    if isinstance(left, str) or isinstance(right, str):
        return left == right
    if any(a is not None and b is not None and a != b for a, b in (
        (left.base, right.base), (left.unit, right.unit),
    )):
        return False
    if left.shape is None or right.shape is None:
        return True
    return len(left.shape) == len(right.shape) and all(
        not (isinstance(a, int) and isinstance(b, int) and a != b)
        for a, b in zip(left.shape, right.shape)
    )


def _port_list(owner: Record, direction: Direction) -> list[Record]:
    return owner["ports"][{"input": "inputs", "output": "outputs", "physical": "physical"}[direction]]


def _listed_port(owner: Record, direction: Direction, port: str) -> Record | None:
    return next((item for item in _port_list(owner, direction) if item["id"] == port), None)


def _port_name(owner: Record, direction: Direction, port: str) -> str:
    item = _listed_port(owner, direction, port)
    return item["name"] if item else port


def _new_port(owner: Record, direction: Direction, name: str, port: Port) -> Record:
    names = {item["name"] for item in _port_list(owner, direction)}
    base = name
    suffix = 1
    while name in names:
        suffix += 1
        name = f"{base} {suffix}"
    value = {"id": uuid4().hex, "name": name, "type": (
        port.type if isinstance(port.type, str) else port.type.model_dump(mode="json")
    )}
    _port_list(owner, direction).append(value)
    return value


def _copy_endpoint(end: Record, ids: dict[str, str]) -> Record:
    return end | {"component": ids[end["component"]]} if "component" in end else end


class _Edit:
    def __init__(self, name: str, state: edits.ModelEditState, ports: LeafPorts, accepts: AcceptComponent):
        self.name = name
        self.document = state.document.model_dump(mode="json")
        self.positions = state.positions.copy()
        self.changed: list[ModelAddress] = []
        self.leaf_ports = ports
        self.accepts_component = accepts

    def context(self, address: ModelAddress) -> tuple[Record, Record]:
        if address.document != self.name:
            raise _Refusal(address, "address names another model document")
        owner = self.document
        for component in address.composites:
            child = owner["interior"]["components"].get(component)
            if not child or child["kind"] != "composite":
                raise _Refusal(address, f"composite {component!r} does not exist at this level")
            owner = child
        return owner, owner["interior"]

    def graph(self, address: ModelAddress) -> tuple[Record, Record]:
        if address.element.kind != "graph":
            raise _Refusal(address, "expected a graph address")
        return self.context(address)

    def port(self, owner: Record, graph: Record, end: Record, direction: Direction) -> Port | None:
        if end["owner"] != "component":
            expected = {"inputs": "output", "outputs": "input", "boundary": "physical"}
            if expected[end["owner"]] != direction:
                return None
            direction = {"inputs": "input", "outputs": "output", "boundary": "physical"}[end["owner"]]
            component = owner
        else:
            component = graph["components"].get(end["component"])
            if component is None:
                return None
            if component["kind"] == "leaf":
                return self.leaf_ports(component, direction, end["port"])
        port = _listed_port(component, direction, end["port"])
        if port is None:
            return None
        state_only = False
        if direction == "output" and end["owner"] == "component":
            child = component["interior"]
            source = next((
                wire["source"] for wire in child["wires"].values()
                if wire["target"] == _card("outputs", end["port"]) and not wire["attachments"]
            ), None)
            if source is not None and source["owner"] == "component":
                leaf = child["components"].get(source["component"])
                if child["physical_domain"] == "planar_multibody" and leaf is not None:
                    from feedbax.contracts.acausal_interface import PLANAR_STATE_ONLY_QUANTITIES, is_acausal_type

                    state_only = leaf["kind"] == "leaf" and is_acausal_type(leaf["type"], "SensorOutput") and (
                        leaf["fields"].get("quantity", "position") in PLANAR_STATE_ONLY_QUANTITIES
                    )
                else:
                    upstream = self.port(component, child, source, "output")
                    state_only = upstream is not None and upstream.reads_state_only
        return Port(
            port["type"] if direction == "physical" else SignalType.model_validate(port["type"]),
            reads_state_only=state_only,
        )

    def component(self, address: ModelAddress) -> tuple[Record, Record]:
        _, graph = self.context(address)
        if address.element.kind != "component" or address.element.id not in graph["components"]:
            raise _Refusal(address, "component does not exist")
        return graph, graph["components"][address.element.id]

    def select(self, selection: tuple[ModelAddress, ...]) -> tuple[Record, set[str]]:
        first = selection[0]
        _, graph = self.context(first)
        ids = set()
        for address in selection:
            if address.document != self.name or address.composites != first.composites:
                raise _Refusal(address, "select elements of one graph")
            if address.element.kind != "component" or address.element.id not in graph["components"]:
                raise _Refusal(address, "component does not exist")
            ids.add(address.element.id)
        return graph, ids

    def connect(self, address: ModelAddress, source: Record, target: Record) -> str:
        owner, graph = self.graph(address)
        output = self.port(owner, graph, source, "output")
        reason = self.connection_reason(owner, graph, output, target)
        if reason is not None:
            raise _Refusal(address, reason)
        assert output is not None
        key = _id(graph)
        delayed = self.closes_loop(owner, graph, source, target, output)
        attachments = [{"kind": "delay", "id": uuid4().hex, "delay": {"value": 1, "unit": "st"},
                        "initial": 0}] if delayed else []
        graph["wires"][key] = _signal(source, target, attachments)
        self.changed.append(_address(self.name, address.composites, "wire", id=key))
        return key

    def connection_reason(self, owner: Record, graph: Record, output: Port | None, target: Record) -> str | None:
        input_port = self.port(owner, graph, target, "input")
        if output is None:
            return "Source must be a local output or a port of the inputs card"
        if input_port is None:
            return "Target must be a local input or a port of the outputs card"
        if not _same_port(output.type, input_port.type):
            def describe(value: SignalType | str) -> str:
                if isinstance(value, str):
                    return value
                return " ".join(part for part in (
                    value.base, str(list(value.shape)) if value.shape is not None else None, value.unit,
                ) if part)

            left, right = describe(output.type), describe(input_port.type)
            return f"{left} ≠ {right}" if left and right else "Signal types are incompatible"
        if any(wire["target"] == target for wire in graph["wires"].values()):
            return "Target already has a wire"
        return None

    def closes_loop(self, owner: Record, graph: Record, source: Record, target: Record, port: Port) -> bool:
        if port.reads_state_only or "component" not in source or "component" not in target:
            return False
        pending = [target["component"]]
        seen = set()
        while pending:
            current = pending.pop()
            if current == source["component"]:
                return True
            if current in seen:
                continue
            seen.add(current)
            for wire in graph["wires"].values():
                a, b = wire["source"], wire["target"]
                if a.get("component") != current or "component" not in b:
                    continue
                declaration = self.port(owner, graph, a, "output")
                if declaration and not declaration.reads_state_only and not any(
                    item["kind"] == "delay" for item in wire["attachments"]
                ):
                    pending.append(b["component"])
        return False

    def physical(self, address: ModelAddress, connection: Record) -> None:
        owner, graph = self.graph(address)
        if "connections" not in graph:
            raise _Refusal(address, "physical connections require an Equations or Mechanics interior")
        a = self.port(owner, graph, connection["a"], "physical")
        b = self.port(owner, graph, connection["b"], "physical")
        if not a or not b or a.type != b.type or connection["a"] == connection["b"]:
            raise _Refusal(address, "physical endpoints must be distinct local ports of the same type")
        if any(
            (item["a"], item["b"]) in (
                (connection["a"], connection["b"]), (connection["b"], connection["a"]),
            )
            for item in graph["connections"].values()
        ):
            raise _Refusal(address, "physical connection already exists")
        key = _id(graph)
        graph["connections"][key] = connection
        self.changed.append(_address(self.name, address.composites, "physical_connection", id=key))

    def remove(self, address: ModelAddress) -> None:
        _, graph = self.context(address)
        element = address.element
        collections = {"component": "components", "wire": "wires", "physical_connection": "connections"}
        if element.kind == "attachment":
            chain = self.attachment_chain(address, element.parent.model_dump(mode="json"))
            index = next((i for i, item in enumerate(chain) if item["id"] == element.id), None)
            if index is None:
                raise _Refusal(address, "attachment does not exist")
            del chain[index]
        elif element.kind == "port":
            self.change_port(address, remove=True)
            return
        elif element.kind in collections and element.id in graph.get(collections[element.kind], {}):
            del graph[collections[element.kind]][element.id]
            if element.kind == "component":
                self.remove_endpoints(graph, lambda end: end.get("component") == element.id)
                for key in tuple(self.positions):
                    saved = ModelAddress.from_text(key)
                    if saved == address or (
                        saved.document == self.name
                        and saved.composites[:len(address.composites) + 1]
                        == (*address.composites, element.id)
                    ):
                        del self.positions[key]
        else:
            raise _Refusal(address, "element does not exist or cannot be deleted")
        self.changed.append(address)

    @staticmethod
    def remove_endpoints(graph: Record, matches: Callable[[Record], bool]) -> None:
        for collection, ends in (("wires", ("source", "target")), ("connections", ("a", "b"))):
            for key, edge in tuple(graph.get(collection, {}).items()):
                if any(matches(edge[end]) for end in ends):
                    del graph[collection][key]

    def port_owner(self, address: ModelAddress) -> tuple[Record, Record, Direction, str]:
        owner, graph = self.context(address)
        if address.element.kind != "port":
            raise _Refusal(address, "expected a port address")
        end = address.element.end
        if end.owner == "component":
            component = graph["components"].get(end.component)
            if not component or component["kind"] != "composite":
                raise _Refusal(address, "only a composite declares editable ports")
            return component, graph, end.direction, end.port
        direction = {"inputs": "input", "outputs": "output", "boundary": "physical"}[end.owner]
        return owner, graph, direction, end.port

    def change_port(self, address: ModelAddress, *, remove=False, name=None, index=None) -> None:
        owner, graph, direction, port_id = self.port_owner(address)
        port = _listed_port(owner, direction, port_id)
        if port is None:
            raise _Refusal(address, "port does not exist")
        ports = _port_list(owner, direction)
        if name is not None:
            if any(other is not port and other["name"] == name for other in ports):
                raise _Refusal(address, "port name is already in use")
            port["name"] = name
        elif index is not None:
            if index >= len(ports):
                raise _Refusal(address, "port position is out of range")
            ports.remove(port)
            ports.insert(index, port)
        elif remove:
            ports.remove(port)
            end = address.element.end
            boundary = {"input": "inputs", "output": "outputs", "physical": "boundary"}[direction]
            self.remove_endpoints(owner["interior"], lambda item: item == _card(boundary, port_id))
            if end.owner == "component":
                self.remove_endpoints(graph, lambda item: item == _component(end.component, port_id))
            elif address.composites:
                parent = _address(self.name, address.composites[:-1], "graph")
                _, parent_graph = self.graph(parent)
                self.remove_endpoints(parent_graph, lambda item: item == _component(address.composites[-1], port_id))
            for placement in (owner.get("attachments", {}), owner["interior"]["attachments"]):
                for chain in placement.values():
                    chain.pop(port_id, None)
        self.changed.append(address)

    def attachment_chain(self, address: ModelAddress, parent: Record, *, create: bool = False) -> list[Record]:
        owner, graph = self.context(address)
        if parent["kind"] == "wire":
            if parent["id"] not in graph["wires"]:
                raise _Refusal(address, "wire does not exist")
            return graph["wires"][parent["id"]]["attachments"]
        end = parent["end"]
        if end["owner"] == "component":
            component = graph["components"].get(end["component"])
            direction = end["direction"]
            if direction == "physical":
                raise _Refusal(address, "Physical ports do not carry signal attachments")
            key = "sources" if direction == "output" else "sinks"
            if component is None or self.port(owner, graph, end, direction) is None:
                raise _Refusal(address, "port does not exist")
            placement = component["attachments"]
        else:
            key = "sources" if end["owner"] == "inputs" else "sinks"
            direction = "output" if key == "sources" else "input"
            if self.port(owner, graph, end, direction) is None:
                raise _Refusal(address, "port does not exist")
            placement = graph["attachments"]
        if create:
            return placement[key].setdefault(end["port"], [])
        return placement[key].get(end["port"], [])

    def set_attachment(self, difference: ModelAttachmentDifference) -> None:
        address = difference.address
        element = address.element
        assert element.kind == "attachment"
        chain = self.attachment_chain(address, element.parent.model_dump(mode="json"), create=True)
        existing = next((i for i, item in enumerate(chain) if item["id"] == element.id), None)
        if isinstance(difference, AddModelAttachment):
            if existing is not None:
                raise _Refusal(address, "Attachment already exists on this owner")
        elif existing is None:
            raise _Refusal(address, "Attachment does not exist")
        else:
            if isinstance(difference, ChangeModelAttachment) and chain[existing]["kind"] != difference.attachment.kind:
                raise _Refusal(address, "Changing an attachment must preserve its kind")
            chain.pop(existing)
        if isinstance(difference, (AddModelAttachment, ChangeModelAttachment)):
            if isinstance(difference, ChangeModelAttachment) and "after" not in difference.model_fields_set:
                assert existing is not None
                position = existing
            elif difference.after is None:
                position = 0
            else:
                after = next((i for i, item in enumerate(chain) if item["id"] == difference.after), None)
                if after is None:
                    raise _Refusal(address, "The preceding attachment does not exist")
                position = after + 1
            chain.insert(position, difference.attachment.model_dump(mode="json"))
        self.changed.append(address)

    def set_field(self, address: ModelAddress, value: Any) -> None:
        owner, graph = self.context(address)
        field = address.element
        if field.kind != "field":
            raise _Refusal(address, "expected a field address")
        parent = field.parent
        path = field.path
        if parent.kind == "component":
            item = graph["components"].get(parent.id)
            allowed = {"fields"} if item and item["kind"] == "leaf" else {"substeps"}
        elif parent.kind == "attachment":
            chain = self.attachment_chain(address, parent.parent.model_dump(mode="json"))
            item = next((entry for entry in chain if entry["id"] == parent.id), None)
            allowed = {"fields", "window", "operation", "initial", "delay", "name"}
        elif parent.kind == "port":
            port_address = _address(self.name, address.composites, "port", end=parent.end.model_dump(mode="json"))
            port_owner, _, direction, port = self.port_owner(port_address)
            item, allowed = _listed_port(port_owner, direction, port), {"type", "default"}
        elif parent.kind == "graph" and not address.composites:
            item, allowed = owner, {"step_seconds", "world"}
        else:
            item, allowed = None, set()
        if item is None or path[0] not in allowed:
            raise _Refusal(address, "field does not exist or is not editable")
        try:
            for segment in path[:-1]:
                item = item[segment]
            item[path[-1]] = value
        except (KeyError, IndexError, TypeError) as error:
            raise _Refusal(address, "field path does not resolve") from error
        self.changed.append(address)

    def add_component(self, command: edits.AddComponent) -> None:
        _, graph = self.graph(command.graph)
        component = command.component.model_dump(mode="json")
        if not self.accepts_component(component, graph["physical_domain"]):
            raise _Refusal(command.graph, "the destination graph cannot hold this component")
        key = _id(graph)
        graph["components"][key] = component
        address = _address(self.name, command.graph.composites, "component", id=key)
        self.positions[address.to_text()] = command.position
        self.changed.append(address)
        if command.connect_from is not None and command.connect_to_port is not None:
            self.connect(command.graph, command.connect_from.model_dump(mode="json"), _component(key, command.connect_to_port))
        elif command.connect_from is not None or command.connect_to_port is not None:
            raise _Refusal(command.graph, "Choose both ports for the new connection")

    def add_port(self, command: edits.AddPort) -> None:
        owner, graph = self.graph(command.graph)
        expected = {"input": SignalInput, "output": SignalOutput, "physical": PhysicalPort}
        if not isinstance(command.port, expected[command.direction]):
            raise _Refusal(command.graph, "port does not match the requested direction")
        if command.direction == "physical" and graph["physical_domain"] is None:
            raise _Refusal(command.graph, "physical ports require an Equations or Mechanics interior")
        _port_list(owner, command.direction).append(command.port.model_dump(mode="json"))
        boundary = {"input": "inputs", "output": "outputs", "physical": "boundary"}[command.direction]
        self.changed.append(_address(
            self.name, command.graph.composites, "port", end=_card(boundary, command.port.id),
        ))

    def bring_in(self, command: edits.BringSignalIn) -> None:
        owner, graph = self.graph(command.source_graph)
        self.graph(command.destination)
        prefix, destination = command.source_graph.composites, command.destination.composites
        if len(destination) <= len(prefix) or destination[:len(prefix)] != prefix:
            raise _Refusal(command.destination, "destination must be inside the source graph")
        source = command.source.model_dump(mode="json")
        port = self.port(owner, graph, source, "output")
        if port is None:
            raise _Refusal(command.source_graph, "source port does not exist")
        if "component" in source:
            component = graph["components"][source["component"]]
            name = _port_name(component, "output", source["port"]) if "ports" in component else source["port"]
        else:
            name = _port_name(owner, "input", source["port"])
        for index in range(len(prefix), len(destination)):
            scope = destination[:index]
            current = _address(self.name, scope, "graph")
            _, graph = self.graph(current)
            child_id = destination[index]
            child = graph["components"][child_id]
            added = _new_port(child, "input", name, port)
            self.connect(current, source, _component(child_id, added["id"]))
            source = _card("inputs", added["id"])
            self.changed.append(_address(self.name, destination[:index + 1], "port", end=source))

    def copy(self, selection: tuple[ModelAddress, ...]) -> edits.ModelClipboard:
        graph, ids = self.select(selection)
        scope = selection[0].composites
        wires, connections = (
            {key: item for key, item in graph.get(collection, {}).items()
             if all(item[end].get("component") in ids for end in ends)}
            for collection, ends in (("wires", ("source", "target")), ("connections", ("a", "b")))
        )
        positions = {}
        for key, position in self.positions.items():
            address = ModelAddress.from_text(key)
            if address.document == self.name and (
                (address.composites == scope and address.element.kind == "component" and address.element.id in ids)
                or (address.composites[:len(scope)] == scope and len(address.composites) > len(scope)
                    and address.composites[len(scope)] in ids)
            ):
                relative = address.model_copy(update={"composites": address.composites[len(scope):]})
                positions[relative.to_text()] = position
        return edits.ModelClipboard.model_validate({
            "components": {key: graph["components"][key] for key in ids},
            "wires": wires, "connections": connections, "positions": positions,
        })

    def paste(self, address: ModelAddress, clipboard: edits.ModelClipboard) -> None:
        _, graph = self.graph(address)
        data = clipboard.model_dump(mode="json")
        if data["connections"] and "connections" not in graph:
            raise _Refusal(address, "destination graph cannot hold physical connections")
        if any(
            not self.accepts_component(component, graph["physical_domain"])
            for component in data["components"].values()
        ):
            raise _Refusal(address, "destination graph cannot hold the selected components")
        ids: dict[str, str] = {}
        for old, component in data["components"].items():
            key = _id(graph)
            graph["components"][key] = component
            ids[old] = key
            self.changed.append(_address(self.name, address.composites, "component", id=key))
        for collection, ends in (("wires", ("source", "target")), ("connections", ("a", "b"))):
            for edge in data[collection].values():
                if any(end.get("component") not in ids for end in (edge[field] for field in ends)):
                    raise _Refusal(address, "clipboard contains an endpoint outside the copied selection")
                key = _id(graph)
                graph[collection][key] = edge | {end: _copy_endpoint(edge[end], ids) for end in ends}
                self.changed.append(_address(
                    self.name, address.composites, "wire" if collection == "wires" else "physical_connection",
                    id=key,
                ))
        for key, position in clipboard.positions.items():
            old = ModelAddress.from_text(key)
            if old.composites:
                first, *rest = old.composites
                if first not in ids:
                    raise _Refusal(address, "clipboard position is outside the copied selection")
                new = old.model_copy(update={"document": self.name, "composites": (*address.composites, ids[first], *rest)})
            else:
                if old.element.kind != "component" or old.element.id not in ids:
                    raise _Refusal(address, "clipboard position is outside the copied selection")
                new = _address(self.name, address.composites, "component", id=ids[old.element.id])
            self.positions[new.to_text()] = position

    def relocate_positions(
        self, source: tuple[str, ...], destination: tuple[str, ...], ids: dict[str, str],
    ) -> None:
        for key, position in tuple(self.positions.items()):
            old = ModelAddress.from_text(key)
            if old.document != self.name or old.composites[:len(source)] != source:
                continue
            rest = old.composites[len(source):]
            if rest and rest[0] in ids:
                new = old.model_copy(update={"composites": (*destination, ids[rest[0]], *rest[1:])})
            elif not rest and old.element.kind == "component" and old.element.id in ids:
                new = _address(self.name, destination, "component", id=ids[old.element.id])
            else:
                continue
            del self.positions[key]
            self.positions[new.to_text()] = position

    def group(self, command: edits.GroupComponents) -> None:
        graph, ids = self.select(command.selection)
        scope = command.selection[0].composites
        owner, _ = self.context(command.selection[0])
        if graph["physical_domain"] is not None:
            raise _Refusal(command.selection[0], "Group requires a Graph interior")
        group_id = _id(graph)
        child = CompositeComponent(
            kind="composite", name=command.name, ports=ModelPorts(), interior=GraphInterior(),
        ).model_dump(mode="json")
        interior = child["interior"]
        feeders: dict[tuple[Direction, str | None, str], str] = {}
        for key, wire in tuple(graph["wires"].items()):
            source, target = wire["source"], wire["target"]
            src_inside = source.get("component") in ids
            dst_inside = target.get("component") in ids
            if src_inside == dst_inside:
                if src_inside:
                    interior["wires"][key] = graph["wires"].pop(key)
                continue
            direction: Direction = "output" if src_inside else "input"
            source_component = source.get("component")
            signature = (direction, source_component, source["port"])
            port_id = feeders.get(signature)
            if port_id is None:
                port = self.port(owner, graph, source, "output")
                if port is None:
                    raise _Refusal(command.selection[0], "crossing wire source does not resolve")
                source_owner = graph["components"][source_component] if source_component else owner
                name = _port_name(
                    source_owner, "output" if source_component else "input", source["port"],
                ) if "ports" in source_owner else source["port"]
                port_id = _new_port(child, direction, name, port)["id"]
                feeders[signature] = port_id
                if src_inside:
                    interior["wires"][_id(interior, set(graph["wires"]))] = _signal(source, _card("outputs", port_id))
                else:
                    graph["wires"][_id(graph)] = _signal(source, _component(group_id, port_id))
            if src_inside:
                wire["source"] = _component(group_id, port_id)
            else:
                graph["wires"].pop(key)
                wire["source"] = _card("inputs", port_id)
                interior["wires"][key] = wire
        for key in ids:
            interior["components"][key] = graph["components"].pop(key)
        graph["components"][group_id] = child
        self.relocate_positions(scope, (*scope, group_id), {key: key for key in ids})
        self.changed.extend(command.selection)
        self.changed.append(_address(self.name, scope, "component", id=group_id))

    def ungroup(self, address: ModelAddress) -> None:
        graph, component = self.component(address)
        element = address.element
        assert element.kind == "component"
        key, scope = element.id, address.composites
        inner_scope = (*scope, key)
        if component["kind"] != "composite" or component["interior"]["physical_domain"] is not None:
            raise _Refusal(address, "Ungroup requires a composite with a Graph interior")
        inner = component["interior"]
        for location, attachments in ((scope, component["attachments"]), (inner_scope, inner["attachments"])):
            for placement, ports in attachments.items():
                for port, chain in ports.items():
                    if chain:
                        if location == scope:
                            end = _component(key, port) | {"direction": "output" if placement == "sources" else "input"}
                        else:
                            end = _card("inputs" if placement == "sources" else "outputs", port)
                        raise _Refusal(
                            _address(self.name, location, "port", end=end),
                            "Ungroup would remove a port that has attachments",
                        )
        parent_wires, child_wires = graph["wires"], inner["wires"]
        outside_inputs = {wire["target"]["port"]: (id_, wire) for id_, wire in parent_wires.items()
                          if wire["target"].get("component") == key}
        inside_outputs = {wire["target"]["port"]: (id_, wire) for id_, wire in child_wires.items()
                          if wire["target"]["owner"] == "outputs"}
        for port in _port_list(component, "input"):
            if port.get("default") is not None and port["id"] not in outside_inputs:
                raise _Refusal(
                    _address(self.name, scope, "port", end=_component(key, port["id"]) | {"direction": "input"}),
                    "Ungroup would remove an unconnected defaulted input",
                )

        consumed: dict[tuple[tuple[str, ...], str], int] = {}

        def trace(end: Record, local_scope: tuple[str, ...], seen: frozenset) -> tuple[Record | None, list[Record]]:
            if local_scope == scope and end.get("component") == key:
                feed = inside_outputs.get(end["port"])
                feed_scope = inner_scope
            elif local_scope == inner_scope and end["owner"] == "inputs":
                feed = outside_inputs.get(end["port"])
                feed_scope = scope
            else:
                return (end if local_scope == scope else _copy_endpoint(end, ids)), []
            if feed is None:
                return None, []
            id_, feeder = feed
            path = (feed_scope, id_)
            if path in seen:
                raise _Refusal(_address(self.name, feed_scope, "wire", id=id_), "boundary feeds itself")
            source, chain = trace(feeder["source"], feed_scope, seen | {path})
            if feeder["attachments"] and source is not None:
                consumed[path] = consumed.get(path, 0) + 1
                if consumed[path] > 1:
                    raise _Refusal(
                        _address(self.name, feed_scope, "wire", id=id_),
                        "Ungroup would copy an attachment onto multiple wires",
                    )
            return source, [*chain, *feeder["attachments"]]

        ids: dict[str, str] = {}
        del graph["components"][key]
        reserved = set(inner["components"]) | set(child_wires)
        for id_, item in inner["components"].items():
            assigned = id_ if id_ not in (graph["components"].keys() | parent_wires.keys() | graph.get("connections", {}).keys()) else _id(graph, reserved)
            ids[id_] = assigned
            graph["components"][assigned] = item
        for id_, wire in tuple(parent_wires.items()):
            if wire["target"].get("component") == key:
                del parent_wires[id_]
                continue
            if wire["source"].get("component") != key:
                continue
            source, chain = trace(wire["source"], scope, frozenset())
            if source is None:
                del parent_wires[id_]
            else:
                wire["source"] = source
                wire["attachments"] = [*wire["attachments"], *chain]
        for id_, wire in child_wires.items():
            if wire["target"]["owner"] == "outputs":
                continue
            source, chain = trace(wire["source"], inner_scope, frozenset())
            if source is None:
                continue
            assigned = id_ if id_ not in (graph["components"].keys() | parent_wires.keys() | graph.get("connections", {}).keys()) else _id(graph, reserved)
            parent_wires[assigned] = _signal(
                source, _copy_endpoint(wire["target"], ids), [*chain, *wire["attachments"]],
            )
        self.relocate_positions(inner_scope, scope, ids)
        self.positions.pop(address.to_text(), None)
        self.changed.append(address)
        self.changed.extend(_address(self.name, scope, "component", id=id_) for id_ in ids.values())

    def validate(self, address: ModelAddress) -> None:
        def check(owner: Record, graph: Record, scope: tuple[str, ...]) -> None:
            targets = set()
            for id_, wire in graph["wires"].items():
                source, target = wire["source"], wire["target"]
                output = self.port(owner, graph, source, "output")
                input_port = self.port(owner, graph, target, "input")
                at = _address(self.name, scope, "wire", id=id_)
                if output is None or input_port is None:
                    raise _Refusal(at, "wire does not name a local source and target port")
                if not _same_port(output.type, input_port.type):
                    raise _Refusal(at, "wire endpoints have incompatible types")
                target_key = (target["owner"], target.get("component"), target["port"])
                if target_key in targets:
                    raise _Refusal(at, "target already has a wire")
                targets.add(target_key)
                self.validate_chain(at, wire["attachments"])
            for id_, connection in graph.get("connections", {}).items():
                left, right = (
                    self.port(owner, graph, connection[end], "physical") for end in ("a", "b")
                )
                at = _address(self.name, scope, "physical_connection", id=id_)
                if not left or not right or left.type != right.type or connection["a"] == connection["b"]:
                    raise _Refusal(at, "physical endpoints must be distinct local ports of the same type")
            for placement, ports in graph["attachments"].items():
                for port, chain in ports.items():
                    end = _card("inputs" if placement == "sources" else "outputs", port)
                    at = _address(self.name, scope, "port", end=end)
                    direction: Direction = "output" if placement == "sources" else "input"
                    if self.port(owner, graph, end, direction) is None:
                        raise _Refusal(at, "attachment owner port does not exist")
                    self.validate_chain(at, chain)
            for id_, component in graph["components"].items():
                for placement, ports in component["attachments"].items():
                    for port, chain in ports.items():
                        direction = "output" if placement == "sources" else "input"
                        end = _component(id_, port)
                        at = _address(self.name, scope, "port", end=end | {"direction": direction})
                        if self.port(owner, graph, end, direction) is None:
                            raise _Refusal(at, "attachment owner port does not exist")
                        self.validate_chain(at, chain)
                if component["kind"] == "composite":
                    check(component, component["interior"], (*scope, id_))

        owner, graph = self.context(address)
        check(owner, graph, address.composites)

    @staticmethod
    def validate_chain(address: ModelAddress, chain: list[Record]) -> None:
        if len({item["id"] for item in chain}) != len(chain):
            raise _Refusal(address, "attachment IDs collide on this owner")


def review_model_connections(
    document: ModelDocument,
    review: edits.ModelConnectionReviewRequest,
    *,
    layout: Callable[[Record], dict[tuple[Direction, str], Port]],
) -> edits.ModelConnectionReview:
    address = review.graph
    edit = _Edit(
        address.document, edits.ModelEditState(document=document),
        lambda component, direction, port: layout(component).get((direction, port)),
        lambda *_: False,
    )
    owner, graph = edit.graph(address)
    source = review.source.model_dump(mode="json")
    output = edit.port(owner, graph, source, "output")
    targets = [_card("outputs", port["id"]) for port in owner["ports"]["outputs"]]
    for id_, component in graph["components"].items():
        if component["kind"] == "leaf":
            inputs = (port for direction, port in layout(component) if direction == "input")
        else:
            inputs = (port["id"] for port in component["ports"]["inputs"])
        targets.extend(_component(id_, port) for port in inputs)
    return edits.ModelConnectionReview(targets=tuple(
        edits.ModelConnectionVerdict(target=target, reason=edit.connection_reason(owner, graph, output, target))
        for target in targets
    ))


def apply_model_edit(
    document: str,
    state: edits.ModelEditState,
    command: edits.ModelEditCommand,
    *,
    ports: LeafPorts,
    accepts: AcceptComponent,
) -> edits.ModelEditResult | edits.ModelEditProblem:
    """Apply one command to a private draft and return its complete transaction."""
    operation = command.operation
    edit = _Edit(document, state, ports, accepts)
    clipboard = None
    primary = _address(document, (), "graph")
    try:
        match operation:
            case edits.AddComponent(graph=primary):
                edit.add_component(operation)
            case edits.MoveComponents(placements=placements):
                primary = placements[0].address
                for placement in placements:
                    address = placement.address
                    if address.composites != primary.composites:
                        raise _Refusal(address, "select components of one graph")
                    edit.component(address)
                    edit.positions[address.to_text()] = placement.position
                    edit.changed.append(address)
            case edits.DeleteElements(selection=selection):
                primary = selection[0]
                for address in selection:
                    if address.composites != primary.composites:
                        raise _Refusal(address, "select elements of one graph")
                    edit.remove(address)
            case edits.ConnectSignal(graph=primary, source=source, target=target):
                edit.connect(primary, source.model_dump(mode="json"), target.model_dump(mode="json"))
            case edits.ConnectPhysical(graph=primary, connection=connection):
                edit.physical(primary, connection.model_dump(mode="json"))
            case edits.AddPort(graph=primary):
                edit.add_port(operation)
            case edits.RemovePort(address=primary):
                edit.change_port(primary, remove=True)
            case edits.RenameElement(address=primary, name=name):
                if primary.element.kind == "component":
                    _, component = edit.component(primary)
                    component["name"] = name
                    edit.changed.append(primary)
                else:
                    edit.change_port(primary, name=name)
            case edits.MovePort(address=primary, index=index):
                edit.change_port(primary, index=index)
            case edits.SetField(address=primary, value=value):
                edit.set_field(primary, value)
            case edits.SetModelAttachment(difference=difference):
                primary = difference.address
                edit.set_attachment(difference)
            case edits.BringSignalIn(destination=primary):
                edit.bring_in(operation)
            case edits.CopyComponents(selection=selection):
                return edits.ModelEditResult(state=state, clipboard=edit.copy(selection))
            case edits.CutComponents(selection=selection):
                primary = selection[0]
                clipboard = edit.copy(selection)
                for address in selection:
                    edit.remove(address)
            case edits.DuplicateComponents(selection=selection):
                primary = selection[0]
                edit.paste(_address(document, primary.composites, "graph"), edit.copy(selection))
            case edits.PasteComponents(graph=primary, clipboard=payload):
                edit.paste(primary, payload)
            case edits.GroupComponents(selection=selection):
                primary = selection[0]
                edit.group(operation)
            case edits.UngroupComposite(address=primary):
                edit.ungroup(primary)
        result = ModelDocument.model_validate(edit.document)
        edit.validate(_address(document, (), "graph"))
        return edits.ModelEditResult(
            state=edits.ModelEditState(document=result, positions=edit.positions),
            changed=tuple(edit.changed), clipboard=clipboard,
        )
    except _Refusal as error:
        return edits.ModelEditProblem(address=error.address, detail=str(error))
    except (ValidationError, ValueError) as error:
        return edits.ModelEditProblem(address=primary, detail=str(error))
