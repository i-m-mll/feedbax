"""Prepare structural patches for the bounded saved reaching model."""

from collections.abc import Set
from dataclasses import dataclass
from typing import Any

from pydantic import JsonValue

from feedbax.compiler.templates import recurrent_zero_initializer
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.graph import ComponentSpec, GraphSpec, StudioTaskBindingSpec, WireSpec
from feedbax.contracts.project import (
    DocumentEdit,
    DocumentOverrides,
    EnsureDocumentCollection,
    SetProjectValue,
    document_edit_ops,
    json_pointer_parts,
)
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.studio_authoring import (
    StudioModelAddComponent,
    StudioModelAuthoringChange,
    StudioModelConnectPorts,
)
from feedbax.studio.schema import validate_graph_connection_schema, validate_task_binding_schema
from feedbax.web.services.project_folder import ProjectFolderInvalid


@dataclass(frozen=True)
class ModelAdditions:
    entries: tuple[SetProjectValue, ...]
    component: str | None = None

    def collections(self) -> dict[str, EnsureDocumentCollection]:
        return {
            name: EnsureDocumentCollection(
                op="ensure", path=pointer((name,)),
                collection="array" if name in {"wires", "output_ports"} else "object",
            )
            for entry in self.entries
            for name, _ in (json_pointer_parts(entry.path),)
        }

    def document_edits(self, payload: dict[str, Any]) -> list[SetProjectValue]:
        return [
            *(
                SetProjectValue(op="add", path="/payload" + edit.path, value=[] if edit.collection == "array" else {})
                for name, edit in self.collections().items() if name not in payload
            ),
            *(SetProjectValue(op="add", path="/payload" + entry.path, value=entry.value) for entry in self.entries),
        ]

    def step_edits(self) -> list[DocumentEdit]:
        return [*self.collections().values(), *self.entries]

    def supersede(self, previous: DocumentOverrides, step: str) -> DocumentOverrides:
        """Drop only declarations that the shared edit will now supply."""
        if self.component is not None or not previous:
            return previous
        entries = {entry.path: entry.value for entry in self.entries}
        collections = self.collections()
        operations = document_edit_ops(previous)
        matched: set[str] = set()
        remaining: list[DocumentEdit] = []
        for edit in operations:
            parts = json_pointer_parts(edit.path)
            collection = collections.get(parts[0])
            if collection is None or isinstance(edit, EnsureDocumentCollection):
                remaining.append(edit)
                continue
            if collection.collection == "array" and not (
                parts[1:] == ("-",) and isinstance(edit, SetProjectValue) and edit.op == "add"
            ):
                raise ProjectFolderInvalid(f"{step} edits /{parts[0]} by position; a shared edit could move its entries.")
            if parts == ("output_bindings",):
                raise ProjectFolderInvalid(f"{step} replaces the output bindings as a collection.")

            overlaps = False
            if edit.path == "/wires/-":
                assert isinstance(edit, SetProjectValue)
                wire = WireSpec.model_validate(edit.value)
                shared_wire = WireSpec.model_validate(entries[edit.path])
                overlaps = _wire_address(wire) == _wire_address(shared_wire)
                if overlaps and wire != shared_wire:
                    raise ProjectFolderInvalid(
                        f"{step} has a different wire declaration for {wire.target_node}.{wire.target_port}."
                    )
            elif edit.path == "/output_ports/-":
                assert isinstance(edit, SetProjectValue)
                overlaps = edit.value == entries[edit.path]
            else:
                binding_path = pointer(parts[:2])
                if binding_path in entries:
                    if not (
                        edit.path == binding_path and isinstance(edit, SetProjectValue)
                        and edit.op == "add" and edit.value == entries[binding_path]
                    ):
                        raise ProjectFolderInvalid(f"{step} has a different binding for public output {parts[1]!r}.")
                    overlaps = True

            if not overlaps:
                remaining.append(edit)
            elif edit.path in matched:
                raise ProjectFolderInvalid(f"{step} declares the same topology more than once.")
            else:
                matched.add(edit.path)
        if not matched:
            return previous
        if matched != entries.keys():
            raise ProjectFolderInvalid(f"{step} has an incomplete local public output declaration.")
        used_collections = {
            name for edit in remaining for name in (json_pointer_parts(edit.path)[0],)
            if edit != collections.get(name)
        }
        return [
            edit for edit in remaining
            if edit != collections.get(name := json_pointer_parts(edit.path)[0]) or name in used_collections
        ]


def _wire_address(wire: WireSpec) -> tuple[str, str, str, str]:
    return wire.source_node, wire.source_port, wire.target_node, wire.target_port


def _add(collection: str, key: str, value: JsonValue) -> SetProjectValue:
    return SetProjectValue(op="add", path=pointer((collection, key)), value=value)


def validate_model_connections(
    graph: GraphSpec, binding: StudioTaskBindingSpec, components: ComponentRegistry,
) -> None:
    issues = validate_graph_connection_schema(graph, task_binding_spec=binding, component_registry=components)
    issues.extend(validate_task_binding_schema(binding, graph, "/task_binding_spec", component_registry=components))
    error = next((issue for issue in issues if issue.severity == "error"), None)
    if error is not None:
        raise ProjectFolderInvalid(error.message)


def prepare_model_edit(
    graph: GraphSpec,
    change: StudioModelAuthoringChange,
    components: ComponentRegistry,
    reserved_nodes: Set[str],
) -> ModelAdditions:
    operation = change.operation
    if isinstance(operation, StudioModelAddComponent):
        definition = next((
            item for item in components.list_all()
            if item.name == operation.component and item.revision == operation.revision
        ), None)
        if definition is None:
            raise ProjectFolderInvalid("The component revision changed. Choose it again.")
        name = definition.name.rsplit(".", 1)[-1]
        stem = name.lower() if name.isupper() else name[:1].lower() + name[1:]
        node_id = stem
        index = 2
        while node_id in reserved_nodes:
            node_id = f"{stem}{index}"
            index += 1
        node = ComponentSpec(
            type=definition.name, revision=definition.revision, params=definition.default_params,
            input_ports=definition.input_ports, output_ports=definition.output_ports,
        )
        return ModelAdditions((_add("nodes", node_id, node.model_dump(mode="json", exclude_none=True)),), node_id)

    if isinstance(operation, StudioModelConnectPorts):
        wire = operation.wire
        if wire.temporality != "instant" or wire.recurrent_initializer is not None:
            raise ProjectFolderInvalid("Choose the source and target ports. Studio supplies the initializer.")
        if wire.source_node == wire.target_node:
            node = graph.nodes.get(wire.source_node)
            if node is None or (node.type, wire.source_port, wire.target_port) != ("feedbax.GRU", "hidden", "hidden"):
                raise ProjectFolderInvalid("Only the GRU hidden state can connect to itself.")
            width = node.params.get("hidden_size")
            dtype = node.params.get("dtype", "float32")
            if type(width) is not int or width <= 0 or not isinstance(dtype, str):
                raise ProjectFolderInvalid("Set the GRU hidden size and dtype before connecting its carry.")
            wire = WireSpec(
                id=wire.id,
                source_node=wire.source_node, source_port="hidden",
                target_node=wire.target_node, target_port="hidden", temporality="recurrent",
                recurrent_initializer=recurrent_zero_initializer(width, state_slot="hidden", dtype=dtype),
            )
        wire = wire.model_copy(update={"id": wire.authored_id()})
        return ModelAdditions((_add("wires", "-", wire.model_dump(mode="json", exclude_none=True)),))

    name = operation.source_port
    node = graph.nodes.get(operation.source_node)
    if node is None or name not in node.output_ports:
        raise ProjectFolderInvalid("Choose an existing output port.")
    if name in graph.output_ports or name in graph.output_bindings:
        raise ProjectFolderInvalid("An output already uses that name.")
    return ModelAdditions((
        _add("output_ports", "-", name),
        _add("output_bindings", name, [operation.source_node, name]),
    ))
