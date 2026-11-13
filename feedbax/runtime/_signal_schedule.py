"""Resolve causal boundaries into a schedule of scoped leaf occurrences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.graph import GraphSpec
from feedbax.runtime._graph import detect_cycles_and_sort

Scope = tuple[str, ...]


def _pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _graph_path(owner: Scope) -> str:
    segments = (token for name in owner for token in ("subgraphs", _pointer_token(name)))
    return "/".join(("", "graph", *segments))


@dataclass(frozen=True)
class ScopedPort:
    node: Scope
    port: str


@dataclass(frozen=True)
class ScopedWire:
    owner: Scope
    index: int

    @property
    def authored_path(self) -> str:
        return f"{_graph_path(self.owner)}/wires/{self.index}"


@dataclass(frozen=True)
class SignalRoute:
    source: ScopedPort
    target: ScopedPort
    wire: ScopedWire
    temporality: Literal["instant", "recurrent"]


@dataclass(frozen=True)
class SignalSchedule:
    order: tuple[Scope, ...]
    routes: tuple[SignalRoute, ...]
    inputs: tuple[tuple[str, ScopedPort], ...]
    outputs: tuple[tuple[str, ScopedPort], ...]
    output_aliases: tuple[tuple[ScopedPort, ScopedPort], ...]


class SignalScheduleError(ValueError):
    def __init__(
        self, message: str, authored_path: str, *, issue_type: str = "invalid_signal_route",
    ):
        super().__init__(message)
        self.authored_path = authored_path
        self.issue_type = issue_type


def needs_signal_schedule(graph: GraphSpec) -> bool:
    """Identify a topology that cannot be stepped in composite order."""
    children = tuple(
        child for child in (graph.subgraphs or {}).values() if isinstance(child, GraphSpec)
    )
    if not children:
        return False
    adjacency: dict[str, set[str]] = {node: set() for node in graph.nodes}
    for wire in graph.wires:
        if wire.temporality == "instant" and wire.source_node in adjacency:
            adjacency[wire.source_node].add(wire.target_node)
    _, back_edges = detect_cycles_and_sort(adjacency)
    return bool(back_edges) or any(needs_signal_schedule(child) for child in children)


def resolve_signal_schedule(graph: GraphSpec) -> SignalSchedule:
    """Schedule all causal leaves, preserving wire identity across port aliases."""
    interiors = {(): graph}
    adjacency: dict[Scope, set[Scope]] = {}
    routes: list[SignalRoute] = []
    dependencies: dict[tuple[Scope, Scope], ScopedWire] = {}

    def collect(owner: Scope, interior: GraphSpec) -> None:
        for node in interior.nodes:
            scope = (*owner, node)
            child = (interior.subgraphs or {}).get(node)
            if isinstance(child, GraphSpec):
                interiors[scope] = child
                collect(scope, child)
            else:
                adjacency[scope] = set()

    def resolve(
        scope: Scope, port: str, direction: Literal["input", "output"], authored_path: str,
    ) -> ScopedPort:
        while (interior := interiors.get(scope)) is not None:
            bindings = interior.input_bindings if direction == "input" else interior.output_bindings
            if port not in bindings:
                raise SignalScheduleError(
                    f"Composite {scope!r} has no binding for {direction} port {port!r}",
                    authored_path,
                )
            node, port = bindings[port]
            scope = (*scope, node)
        if scope not in adjacency:
            raise SignalScheduleError(f"Signal route names unknown occurrence {scope!r}", authored_path)
        node = interiors[scope[:-1]].nodes[scope[-1]]
        physical = (interiors[scope[:-1]].subgraphs or {}).get(scope[-1])
        ports = node.input_ports if direction == "input" else node.output_ports
        if port not in ports and not isinstance(physical, AcausalGraphSpec):
            raise SignalScheduleError(
                f"Occurrence {scope!r} has no {direction} port {port!r}", authored_path,
            )
        return ScopedPort(scope, port)

    collect((), graph)
    inputs = tuple(
        (port, resolve((), port, "input", f"/graph/input_bindings/{_pointer_token(port)}"))
        for port in sorted(graph.input_bindings)
    )
    outputs = tuple(
        (port, resolve((), port, "output", f"/graph/output_bindings/{_pointer_token(port)}"))
        for port in sorted(graph.output_bindings)
    )
    drivers: set[ScopedPort] = set()
    for port, target in inputs:
        if target in drivers:
            raise SignalScheduleError(
                f"Multiple drivers for input {target!r}",
                f"/graph/input_bindings/{_pointer_token(port)}",
            )
        drivers.add(target)
    for owner, interior in sorted(interiors.items()):
        for index, wire in enumerate(interior.wires):
            reference = ScopedWire(owner, index)
            source = resolve((*owner, wire.source_node), wire.source_port, "output", reference.authored_path)
            target = resolve((*owner, wire.target_node), wire.target_port, "input", reference.authored_path)
            if target in drivers:
                raise SignalScheduleError(
                    f"Multiple drivers for input {target!r}", reference.authored_path,
                )
            drivers.add(target)
            if wire.temporality == "recurrent" and (
                (wire.recurrent_initializer or {}).get("kind") in {"graph-input", "node-output"}
            ):
                raise SignalScheduleError(
                    "Trial-input and node-output resets require a contiguous composite schedule",
                    reference.authored_path,
                )
            routes.append(SignalRoute(source, target, reference, wire.temporality))
            if wire.temporality == "instant":
                adjacency[source.node].add(target.node)
                dependencies.setdefault((source.node, target.node), reference)

    order, back_edges = detect_cycles_and_sort(adjacency)
    if back_edges:
        source, target = back_edges[0]
        raise SignalScheduleError(
            f"Instant signal wires contain a same-step cycle at {source!r} → {target!r}",
            dependencies[source, target].authored_path,
            issue_type="instant_cycle",
        )
    output_aliases = tuple(
        (ScopedPort(owner, port), resolve(
            owner, port, "output", f"{_graph_path(owner)}/output_bindings/{_pointer_token(port)}",
        ))
        for owner, interior in sorted(interiors.items()) if owner
        for port in sorted(interior.output_bindings)
    )
    return SignalSchedule(tuple(order), tuple(routes), inputs, outputs, output_aliases)
