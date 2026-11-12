"""Prepare structural patches for the bounded saved reaching model."""

from collections.abc import Iterable
from typing import Any

from feedbax.compiler.templates import recurrent_zero_initializer
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.graph import ComponentSpec, GraphSpec, StudioTaskBindingSpec, WireSpec
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.studio_authoring import (
    StudioModelAddComponent,
    StudioModelAuthoringChange,
    StudioModelConnectPorts,
)
from feedbax.contracts.studio_project import SetProjectValue
from feedbax.studio.schema import validate_graph_connection_schema, validate_task_binding_schema
from feedbax.web.services.project_folder import ProjectFolderInvalid


def _add(payload: dict[str, Any], collection: str, key: str, value: Any) -> SetProjectValue:
    if collection not in payload:
        path = ("payload", collection)
        value = [value] if collection in {"wires", "output_ports"} else {key: value}
    else:
        path = ("payload", collection, key)
    return SetProjectValue(op="add", path=pointer(path), value=value)


def prepare_model_edit(
    payload: dict[str, Any],
    graph: GraphSpec,
    change: StudioModelAuthoringChange,
    components: ComponentRegistry,
    task_bindings: Iterable[StudioTaskBindingSpec],
) -> tuple[list[SetProjectValue], str | None]:
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
        while node_id in graph.nodes:
            node_id = f"{stem}{index}"
            index += 1
        node = ComponentSpec(
            type=definition.name, revision=definition.revision, params=definition.default_params,
            input_ports=definition.input_ports, output_ports=definition.output_ports,
        )
        return [_add(payload, "nodes", node_id, node.model_dump(mode="json", exclude_none=True))], node_id

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
                source_node=wire.source_node, source_port="hidden",
                target_node=wire.target_node, target_port="hidden", temporality="recurrent",
                recurrent_initializer=recurrent_zero_initializer(width, state_slot="hidden", dtype=dtype),
            )
        updated = graph.model_copy(update={"wires": [*graph.wires, wire]})
        for bindings in task_bindings:
            issues = validate_graph_connection_schema(
                updated, task_binding_spec=bindings, component_registry=components,
            )
            issues.extend(validate_task_binding_schema(bindings, updated, "/task_binding_spec", component_registry=components))
            error = next((issue for issue in issues if issue.severity == "error"), None)
            if error is not None:
                raise ProjectFolderInvalid(error.message)
        return [_add(payload, "wires", str(len(graph.wires)), wire.model_dump(mode="json", exclude_none=True))], None

    name = operation.source_port
    node = graph.nodes.get(operation.source_node)
    if node is None or name not in node.output_ports:
        raise ProjectFolderInvalid("Choose an existing output port.")
    if name in graph.output_ports or name in graph.output_bindings:
        raise ProjectFolderInvalid("An output already uses that name.")
    return [
        _add(payload, "output_ports", str(len(graph.output_ports)), name),
        _add(payload, "output_bindings", name, [operation.source_node, name]),
    ], None
