"""Lower typed Studio intervention taps into ordinary causal components and wires."""

from __future__ import annotations

import math

from feedbax.contracts.graph import (
    ComponentSpec,
    GraphSpec,
    StudioInterventionTransformSpec,
    TapSpec,
    WireSpec,
)


class TapLoweringError(ValueError):
    def __init__(self, authored_path: str, message: str):
        super().__init__(message)
        self.authored_path = authored_path


def _number(value: object) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or abs(value) > 3.4028234663852886e38
    ):
        raise ValueError("expected a finite scalar number representable in float32")
    return float(value)


def _component(tap: TapSpec) -> tuple[ComponentSpec, StudioInterventionTransformSpec]:
    transform = tap.transform
    if transform is None or transform.type != "intervention" or transform.intervention is None:
        raise ValueError("requires a typed intervention transform")
    if transform.params or tap.paths:
        raise ValueError("custom transform parameters and tap paths cannot be executed")
    intervention = transform.intervention
    if intervention.operation == "scale":
        value = intervention.value
        if (
            intervention.bounds is not None
            or value is None
            or value.value_form != "literal"
            or value.variation.scope != "fixed"
        ):
            raise ValueError("scale requires a fixed scalar literal")
        kind, params = "Scale", {"scale": _number(value.value), "fixed": True}
    elif intervention.operation == "clamp":
        bounds = intervention.bounds
        if (
            intervention.value is not None
            or bounds is None
            or bounds.min is None
            or bounds.max is None
        ):
            raise ValueError("clamp requires both scalar bounds")
        lower, upper = _number(bounds.min), _number(bounds.max)
        if lower > upper:
            raise ValueError("clamp minimum must be at most its maximum")
        kind, params = "Saturation", {"min_val": lower, "max_val": upper}
    else:
        raise ValueError(
            f"unsupported executable intervention operation {intervention.operation!r}"
        )
    return (
        ComponentSpec(type=kind, params=params, input_ports=["input"], output_ports=["output"]),
        intervention,
    )


def _source(wire: WireSpec) -> tuple[str, str]:
    return wire.source_node, wire.source_port


def _compact(wire: WireSpec) -> str:
    return f"edge:{wire.source_node}.{wire.source_port}->{wire.target_node}.{wire.target_port}"


def _write_reset(wire: WireSpec, *, producer: bool) -> dict:
    initializer = wire.recurrent_initializer or {}
    if initializer.get("seed_channel_history"):
        raise ValueError("cannot write a carry that seeds channel history")
    if initializer.get("state_slot") and not producer:
        raise ValueError("edge-local carry write cannot preserve a graph state_slot alias")
    if initializer.get("kind") != "state_output":
        raise ValueError(
            "carry write requires a state_output initializer to preserve the reset sample"
        )
    if "source_node" in initializer or "source_port" in initializer:
        return initializer
    return {**initializer, "source_node": wire.source_node, "source_port": wire.source_port}


def _split(wire: WireSpec, node: str, phase: str | None) -> tuple[WireSpec, WireSpec]:
    incoming = WireSpec(
        source_node=wire.source_node,
        source_port=wire.source_port,
        target_node=node,
        target_port="input",
    )
    outgoing = WireSpec(
        source_node=node,
        source_port="output",
        target_node=wire.target_node,
        target_port=wire.target_port,
    )
    if phase == "read":
        initializer = wire.recurrent_initializer
        if (initializer or {}).get("seed_channel_history"):
            raise ValueError("cannot insert a carry read on a wire that seeds channel history")
        incoming = incoming.model_copy(
            update={
                "temporality": "recurrent",
                "recurrent_initializer": initializer,
            }
        )
    elif phase == "write":
        outgoing = outgoing.model_copy(
            update={
                "temporality": "recurrent",
                "recurrent_initializer": _write_reset(wire, producer=False),
            }
        )
    return incoming, outgoing


def _pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def lower_studio_taps(
    graph: GraphSpec, path: str = "/graph", *, nested: bool = False
) -> tuple[GraphSpec, dict[tuple[str, str], str]]:
    """Return a literal graph and the authored tap paths of its generated nodes."""
    for node, barnacles in (graph.barnacles or {}).items():
        for index, barnacle in enumerate(barnacles):
            if barnacle.kind == "intervention":
                raise TapLoweringError(
                    f"{path}/barnacles/{_pointer_token(node)}/{index}",
                    "Legacy barnacle intervention cannot execute. Reauthor it as a typed tap or graph component.",
                )
    origins: dict[tuple[str, str], str] = {}
    children = dict(graph.subgraphs or {})
    for node, subgraph in children.items():
        if isinstance(subgraph, GraphSpec):
            token = _pointer_token(node)
            children[node], child_origins = lower_studio_taps(
                subgraph, f"{path}/subgraphs/{token}", nested=True
            )
            origins.update(child_origins)

    nodes = dict(graph.nodes)
    wires = list(graph.wires)
    outputs = dict(graph.output_bindings)
    remaining = []
    claimed_wires: set[int] = set()
    claimed_producers: set[tuple[str, str]] = set()
    for index, tap in enumerate(graph.taps or []):
        if tap.type == "probe":
            if tap.transform and tap.transform.intervention is not None:
                raise TapLoweringError(
                    f"{path}/taps/{index}", "probe cannot execute an intervention"
                )
            remaining.append(tap)
            continue
        anchor = f"{path}/taps/{index}"
        try:
            component, intervention = _component(tap)
            selector = intervention.target_selector
            if selector is None:
                raise ValueError("requires a typed target selector")
            params = intervention.parameters or {}
            generated = f"__studio_tap__{tap.id}__transform"
            if not tap.id or generated in nodes or generated in children:
                raise ValueError(f"tap ID is empty or generated node {generated!r} already exists")

            if selector.namespace == "graph_port":
                if params:
                    raise ValueError("producer taps do not accept intervention parameters")
                if selector.metadata.get("direction") not in (None, "output"):
                    raise ValueError("producer taps require a graph output port")
                matches = [
                    (node, port)
                    for node, spec in graph.nodes.items()
                    for port in spec.output_ports
                    if selector.compact == f"port:{node}.{port}"
                    and selector.target_id in (None, node)
                    and selector.path in (None, port)
                ]
                if len(matches) != 1:
                    raise ValueError(f"expected one producer output for {selector.compact!r}")
                source = matches[0]
                affected = [i for i, wire in enumerate(graph.wires) if _source(wire) == source]
                exposes_source = source in outputs.values()
                if nested and exposes_source:
                    raise ValueError(
                        "cannot preserve reset provenance across a tapped subgraph output"
                    )
                if not affected and not exposes_source:
                    raise ValueError("producer output has no consumers or graph output bindings")
                if source in claimed_producers or claimed_wires.intersection(affected):
                    raise ValueError("multiple intervention taps target the same graph connection")
                claimed_producers.add(source)
                claimed_wires.update(affected)
                for i in affected:
                    wire = wires[i]
                    initializer = (
                        _write_reset(wire, producer=True)
                        if wire.temporality == "recurrent"
                        else wire.recurrent_initializer
                    )
                    wires[i] = wire.model_copy(
                        update={
                            "source_node": generated,
                            "source_port": "output",
                            "recurrent_initializer": initializer,
                        }
                    )
                outputs = {
                    name: (generated, "output") if binding == source else binding
                    for name, binding in outputs.items()
                }
                wires.append(
                    WireSpec(
                        source_node=source[0],
                        source_port=source[1],
                        target_node=generated,
                        target_port="input",
                    )
                )
            elif selector.namespace in ("graph_edge", "recurrent_carry"):
                recurrent = selector.namespace == "recurrent_carry"
                matches = [
                    i
                    for i, wire in enumerate(graph.wires)
                    if selector.compact
                    in (
                        _compact(wire),
                        _compact(wire).replace("edge:", "recurrent_carry:", 1) if recurrent else "",
                    )
                    and (wire.temporality == "recurrent") == recurrent
                ]
                if len(matches) != 1:
                    raise ValueError(f"expected one matching wire for {selector.compact!r}")
                i = matches[0]
                if i in claimed_wires:
                    raise ValueError("multiple intervention taps target the same graph connection")
                phase = params.get("phase")
                if recurrent and (phase not in ("read", "write") or set(params) != {"phase"}):
                    raise ValueError("recurrent taps require parameters.phase of 'read' or 'write'")
                if not recurrent and params:
                    raise ValueError("instantaneous taps do not accept intervention parameters")
                wires[i], outgoing = _split(wires[i], generated, phase)
                wires.append(outgoing)
                claimed_wires.add(i)
            else:
                raise ValueError(
                    f"unsupported writable tap target {selector.namespace!r}; "
                    "select a producer output, a wire, or a recurrent carry"
                )
            nodes[generated] = component
            origins[path, generated] = anchor
        except (ValueError, OverflowError) as exc:
            raise TapLoweringError(anchor, f"Intervention tap {tap.id!r} {exc}") from exc

    if not origins:
        return graph, origins
    return graph.model_copy(
        update={
            "nodes": nodes,
            "wires": wires,
            "output_bindings": outputs,
            "subgraphs": children if graph.subgraphs is not None else None,
            "taps": (remaining or None) if len(remaining) != len(graph.taps or []) else graph.taps,
        }
    ), origins
