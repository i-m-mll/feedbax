"""Declarative GraphSpec channel-adapter materialization."""

from __future__ import annotations

import re

from feedbax.contracts.graph import (
    AdditiveGraphChannelAdapterSpec,
    ComponentSpec,
    GraphSpec,
    WireSpec,
)


def materialize_additive_channel_adapters(spec: GraphSpec) -> GraphSpec:
    """Return a graph with additive channel adapters lowered to nodes/bindings."""

    subgraphs = (
        {
            node_id: (
                materialize_additive_channel_adapters(subgraph)
                if isinstance(subgraph, GraphSpec)
                else subgraph
            )
            for node_id, subgraph in spec.subgraphs.items()
        }
        if spec.subgraphs
        else None
    )
    if not spec.additive_channel_adapters:
        return spec.model_copy(update={"subgraphs": subgraphs})

    nodes = dict(spec.nodes)
    wires = list(spec.wires)
    input_ports = list(spec.input_ports)
    input_bindings = dict(spec.input_bindings)

    for adapter in spec.additive_channel_adapters:
        if adapter.input_key not in input_ports:
            input_ports.append(adapter.input_key)
        if adapter.target.kind == "edge":
            _materialize_edge_adapter(
                adapter,
                nodes=nodes,
                wires=wires,
                input_bindings=input_bindings,
            )
            continue
        _materialize_input_adapter(
            adapter,
            nodes=nodes,
            wires=wires,
            input_bindings=input_bindings,
        )

    return spec.model_copy(
        update={
            "nodes": nodes,
            "wires": wires,
            "additive_channel_adapters": [],
            "input_ports": input_ports,
            "input_bindings": input_bindings,
            "subgraphs": subgraphs,
        }
    )


def _materialize_edge_adapter(
    adapter: AdditiveGraphChannelAdapterSpec,
    *,
    nodes: dict[str, ComponentSpec],
    wires: list[WireSpec],
    input_bindings: dict[str, tuple[str, str]],
) -> None:
    target = adapter.target
    matches = [
        index
        for index, wire in enumerate(wires)
        if wire.source_node == target.source_node
        and wire.source_port == target.source_port
        and wire.target_node == target.target_node
        and wire.target_port == target.target_port
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one wire for additive channel adapter {adapter.label!r}; "
            f"found {len(matches)}"
        )
    if adapter.input_key in input_bindings:
        raise ValueError(f"Additive channel adapter input {adapter.input_key!r} is already bound")

    wire = wires.pop(matches[0])
    node_name = _adapter_node_name(adapter, nodes)
    nodes[node_name] = _sum_node_spec(adapter)
    wires.extend(
        [
            WireSpec(
                source_node=wire.source_node,
                source_port=wire.source_port,
                target_node=node_name,
                target_port="a",
                temporality=wire.temporality,
                recurrent_initializer=wire.recurrent_initializer,
            ),
            WireSpec(
                source_node=node_name,
                source_port="output",
                target_node=wire.target_node,
                target_port=wire.target_port,
            ),
        ]
    )
    input_bindings[adapter.input_key] = (node_name, "b")


def _materialize_input_adapter(
    adapter: AdditiveGraphChannelAdapterSpec,
    *,
    nodes: dict[str, ComponentSpec],
    wires: list[WireSpec],
    input_bindings: dict[str, tuple[str, str]],
) -> None:
    target = adapter.target
    target_key = (target.target_node, target.target_port)
    if adapter.input_key in input_bindings:
        raise ValueError(f"Additive channel adapter input {adapter.input_key!r} is already bound")
    incoming = [wire for wire in wires if (wire.target_node, wire.target_port) == target_key]
    if incoming:
        raise ValueError(
            f"Input additive channel adapter {adapter.label!r} targets "
            f"{target.target_node}.{target.target_port}, which is already wired; "
            "use an edge adapter for wired targets"
        )

    base_bindings = [key for key, binding in input_bindings.items() if tuple(binding) == target_key]
    if not base_bindings:
        input_bindings[adapter.input_key] = target_key
        return
    if len(base_bindings) != 1:
        raise ValueError(
            f"Input additive channel adapter {adapter.label!r} found "
            f"{len(base_bindings)} base bindings for {target.target_node}.{target.target_port}"
        )

    base_key = base_bindings[0]
    node_name = _adapter_node_name(adapter, nodes)
    nodes[node_name] = _sum_node_spec(adapter)
    input_bindings[base_key] = (node_name, "a")
    input_bindings[adapter.input_key] = (node_name, "b")
    wires.append(
        WireSpec(
            source_node=node_name,
            source_port="output",
            target_node=target.target_node,
            target_port=target.target_port,
        )
    )


def _adapter_node_name(
    adapter: AdditiveGraphChannelAdapterSpec,
    nodes: dict[str, ComponentSpec],
) -> str:
    base = adapter.adapter_node or f"{adapter.label}_additive"
    candidate = re.sub(r"[^0-9A-Za-z_]+", "_", base).strip("_") or "additive_channel"
    if candidate not in nodes:
        return candidate
    suffix = 2
    while f"{candidate}_{suffix}" in nodes:
        suffix += 1
    return f"{candidate}_{suffix}"


def _sum_node_spec(adapter: AdditiveGraphChannelAdapterSpec) -> ComponentSpec:
    params = {
        "channel_adapter": {
            "label": adapter.label,
            "input_key": adapter.input_key,
            "target": adapter.target.model_dump(mode="json", exclude_none=True),
            "payload_shape": adapter.payload_shape,
            "payload_dtype": adapter.payload_dtype,
            "provenance_role": adapter.provenance_role,
            "metadata": adapter.metadata,
        }
    }
    return ComponentSpec(
        type="Sum",
        params=params,
        input_ports=["a", "b"],
        output_ports=["output"],
    )
