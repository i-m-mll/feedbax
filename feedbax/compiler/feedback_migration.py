"""Lower durable feedback wrapper specs to ordinary selector and channel nodes."""

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


def lower_legacy_feedback(graph: GraphSpec, *, recurse: bool = True) -> GraphSpec:
    """Expand each recoverable wrapper while preserving its authored boundary."""
    nodes = dict(graph.nodes)
    subgraphs = {
        name: lower_legacy_feedback(interior) if recurse and isinstance(interior, GraphSpec) else interior
        for name, interior in (graph.subgraphs or {}).items()
    }
    for name, node in graph.nodes.items():
        if node.type != "FeedbackChannels":
            continue
        if (
            node.input_ports != ["mechanics"]
            or node.output_ports != ["feedback"]
            or "channels" in node.params
            or node.revision not in (None, 1)
            or name in subgraphs
        ):
            raise ValueError(
                f"FeedbackChannels node {name!r} must declare its mechanics input and "
                "feedback output with a supported selector to migrate"
            )
        selector = node.params.get("selector", "point_mass_pos_vel")
        if not isinstance(selector, str):
            raise ValueError(f"FeedbackChannels node {name!r} requires a selector name")
        paths = (
            node.params.get("paths", _LEGACY_PATHS["point_mass_pos_vel"])
            if selector == "paths"
            else _LEGACY_PATHS.get(selector)
        )
        if not isinstance(paths, (list, tuple)) or not paths:
            raise ValueError(
                f"FeedbackChannels node {name!r} has unsupported selector {selector!r}"
            )
        channel_params = {
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
        nodes[name] = node.model_copy(
            update={"type": "feedbax.Subgraph", "params": {}, "revision": None}
        )
        subgraphs[name] = GraphSpec(
            nodes={
                "select": ComponentSpec(
                    type="feedbax.SelectState",
                    params={"paths": list(paths)},
                    input_ports=["state"],
                    output_ports=["output"],
                ),
                "channel": ComponentSpec(
                    type="feedbax.Channel",
                    params=channel_params,
                    revision=2,
                    input_ports=["input"],
                    output_ports=["output"],
                ),
            },
            wires=[
                WireSpec(
                    source_node="select",
                    source_port="output",
                    target_node="channel",
                    target_port="input",
                ),
            ],
            input_ports=node.input_ports,
            output_ports=node.output_ports,
            input_bindings={"mechanics": ("select", "state")},
            output_bindings={"feedback": ("channel", "output")},
        )
    return graph.model_copy(update={"nodes": nodes, "subgraphs": subgraphs or None})
