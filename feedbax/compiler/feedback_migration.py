"""Lower durable feedback wrapper specs to ordinary selector and channel nodes."""

from typing import Any

from feedbax.contracts.graph import ComponentSpec, GraphSpec, WireSpec


_LEGACY_PATHS = {
    "point_mass_pos_vel": ["plant.skeleton.pos", "plant.skeleton.vel"],
    "point_mass_state": ["plant.skeleton.pos", "plant.skeleton.vel"],
    "mechanics.plant.skeleton.pos_vel": ["plant.skeleton.pos", "plant.skeleton.vel"],
    "effector_pos_vel": ["effector.pos", "effector.vel"],
    "mechanics.effector.pos_vel": ["effector.pos", "effector.vel"],
    "plant_skeleton": ["plant.skeleton"],
    "mechanics.plant.skeleton": ["plant.skeleton"],
}


def lower_legacy_feedback(
    graph: GraphSpec,
    input_prototypes: dict[tuple[str, str], Any] | None = None,
    *,
    recurse: bool = True,
) -> GraphSpec:
    """Expand the registered single-channel wrapper with a recoverable selector."""
    legacy = {name: node for name, node in graph.nodes.items() if node.type == "FeedbackChannels"}
    subgraphs = (
        {
            name: lower_legacy_feedback(interior) if isinstance(interior, GraphSpec) else interior
            for name, interior in (graph.subgraphs or {}).items()
        }
        if recurse
        else graph.subgraphs
    )
    if not legacy:
        return (
            graph.model_copy(update={"subgraphs": subgraphs or None}) if graph.subgraphs else graph
        )

    nodes = dict(graph.nodes)
    selectors = {}
    for name, node in legacy.items():
        if (
            node.input_ports != ["mechanics"]
            or node.output_ports != ["feedback"]
            or "channels" in node.params
            or node.param_schema_version not in (None, "1")
        ):
            raise ValueError(
                f"FeedbackChannels node {name!r} must declare its mechanics input and "
                "feedback output with a supported selector to migrate"
            )
        selector = node.params.get("selector", "point_mass_pos_vel")
        if not isinstance(selector, str):
            raise ValueError(f"FeedbackChannels node {name!r} requires a selector name")
        paths = node.params.get("paths") if selector == "paths" else _LEGACY_PATHS.get(selector)
        if not isinstance(paths, (list, tuple)) or not paths:
            raise ValueError(
                f"FeedbackChannels node {name!r} has unsupported selector {selector!r}"
            )
        selector_name = f"{name}_select"
        suffix = 1
        while selector_name in nodes:
            selector_name = f"{name}_select_{suffix}"
            suffix += 1
        selectors[name] = selector_name
        nodes[selector_name] = ComponentSpec(
            type="SelectState",
            params={"paths": list(paths)},
            input_ports=["state"],
            output_ports=["output"],
        )
        noise_and_delay = {
            "delay": 0,
            "noise_std": 0.0,
            "add_noise": False,
            "noise_role": "sensory_feedback",
            "noise_timing": "pre_controller",
            **{
                key: value
                for key, value in node.params.items()
                if key not in {"selector", "paths", "input_shape"}
            },
        }
        nodes[name] = ComponentSpec(
            type="Channel", params=noise_and_delay, input_ports=["input"], output_ports=["output"]
        )
        if input_prototypes is not None and (name, "mechanics") in input_prototypes:
            input_prototypes[(selector_name, "state")] = input_prototypes.pop((name, "mechanics"))

    def remap_input(node: str, port: str) -> tuple[str, str]:
        return (
            (selectors[node], "state")
            if node in selectors and port == "mechanics"
            else (node, port)
        )

    def remap_output(node: str, port: str) -> tuple[str, str]:
        return (node, "output") if node in selectors and port == "feedback" else (node, port)

    wires = []
    for wire in graph.wires:
        source, out = remap_output(wire.source_node, wire.source_port)
        target, inp = remap_input(wire.target_node, wire.target_port)
        initializer = wire.recurrent_initializer
        if initializer is not None and initializer.get("kind") == "node-output":
            source_node = initializer.get("source_node")
            source_port = initializer.get("source_port")
            if source_node in selectors and source_port == "feedback":
                initializer = {**initializer, "source_port": "output"}
        wires.append(
            wire.model_copy(
                update={
                    "source_node": source,
                    "source_port": out,
                    "target_node": target,
                    "target_port": inp,
                    "recurrent_initializer": initializer,
                }
            )
        )
    wires.extend(
        WireSpec(source_node=selector, source_port="output", target_node=name, target_port="input")
        for name, selector in selectors.items()
    )
    return graph.model_copy(
        update={
            "nodes": nodes,
            "wires": wires,
            "input_bindings": {
                name: remap_input(*binding) for name, binding in graph.input_bindings.items()
            },
            "output_bindings": {
                name: remap_output(*binding) for name, binding in graph.output_bindings.items()
            },
            "subgraphs": subgraphs or None,
        }
    )
