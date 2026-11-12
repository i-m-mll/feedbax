"""Bind stable raw-state declarations to the exact reset and scan carry."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from operator import attrgetter

from equinox.nn import State, StateIndex
import jax
from jax.tree_util import GetAttrKey
import numpy as np

from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.semantic_values import semantic_key
from feedbax.dataset import Variable
from feedbax.runtime.graph import Graph, init_state_from_component
from feedbax.runtime.model_values import MODEL_OWNERS, StateValue


class StateLayoutError(ValueError):
    def __init__(self, path: str, reason: str):
        self.path = path
        super().__init__(reason)


@dataclass(frozen=True)
class BoundStateValue:
    key: str
    dimension: str | None
    units: str
    frame: str | None
    read: Callable[[State, dict | None], jax.Array]


@dataclass(frozen=True)
class RawStateLayout:
    values: tuple[BoundStateValue, ...]

    def capture(self, state: State, carry: dict | None) -> dict[str, jax.Array]:
        return {value.key: value.read(state, carry) for value in self.values}

    def dataset_fields(
        self, trials: Sequence[Mapping[str, jax.Array]], *, axes: tuple[str, ...],
    ) -> tuple[dict[str, Variable], dict[str, Variable]]:
        """Stack each independent state part on the shared reset-inclusive axis."""
        variables, coordinates = {}, {}
        for value in self.values:
            samples = np.stack([np.asarray(trial[value.key]) for trial in trials])
            if not np.isfinite(samples).all():
                raise FloatingPointError(f"Nonfinite state samples at {value.key}")
            dims = (*axes, "state_time")
            if value.dimension is not None:
                dims = (*dims, value.dimension)
                labels = ["x", "y"] if value.dimension == "xy" else np.arange(samples.shape[-1])
                coordinates[value.dimension] = Variable(np.asarray(labels), (value.dimension,))
            variables[value.key] = Variable(
                samples[None, None], dims, units=value.units, frame=value.frame,
            )
        return variables, coordinates


def _leaf_reader(index: StateIndex, value: StateValue) -> Callable[[State, dict | None], jax.Array]:
    if not value.fields:
        return lambda state, carry: state.get(index)
    get = attrgetter(".".join(value.fields))
    return lambda state, carry: get(state.get(index))


def _bind_owner(
    name: str, graph: Graph, authored: GraphSpec, initial: State, *, selected: frozenset[str] | None,
) -> tuple[BoundStateValue, ...]:
    component = graph.nodes[name]
    binding = graph.component_bindings[name]
    path = pointer(("inputs", "subject", "graph", "nodes", name))
    owner = MODEL_OWNERS.get(binding.type_id)
    node = authored.nodes[name]
    if (
        owner is None or owner.states is None or binding.revision != owner.revision
        or node.type != binding.type_id or node.revision not in {None, owner.revision}
    ):
        raise StateLayoutError(path, "owner revision has no declared raw-state layout")
    index = getattr(component, "state_index", None)
    slots = jax.tree.leaves(component, is_leaf=lambda item: isinstance(item, StateIndex))
    indices = [slot for slot in slots if isinstance(slot, StateIndex)]
    for name in owner.configuration_slots:
        configured = getattr(component, name, None)
        if not any(slot is configured for slot in indices):
            raise StateLayoutError(path, "runtime configuration slots differ from the owner declaration")
        indices = [slot for slot in indices if slot is not configured]
    declared = owner.states
    if not indices and (not declared or binding.type_id == "feedbax.GRU" and index is None):
        return ()
    if len(indices) != 1 or indices[0] is not index or not declared:
        raise StateLayoutError(path, "runtime state slots differ from the owner declaration")
    leaves = jax.tree.leaves_with_path(initial.get(index))
    if any(
        not isinstance(leaf, jax.Array) or not all(isinstance(key, GetAttrKey) for key in fields)
        for fields, leaf in leaves
    ) or {tuple(key.name for key in fields) for fields, _ in leaves} != {
        leaf.fields for leaf in declared
    }:
        raise StateLayoutError(path, "runtime state leaves differ from the owner declaration")

    values = []
    for leaf in declared:
        key = semantic_key("model", "state", (name,), leaf.symbol)
        if selected is not None and key not in selected:
            continue
        read = _leaf_reader(index, leaf)
        shape = read(initial, None).shape
        dimension = f"{name}:{leaf.axis}"
        if leaf.axis == "xy":
            if shape != (2,):
                raise StateLayoutError(path, f"{leaf.symbol} requires two Cartesian coordinates")
            dimension = "xy"
        elif leaf.axis == "muscle" and shape == ():
            dimension = None
        elif len(shape) != 1:
            raise StateLayoutError(path, f"{leaf.symbol} requires a one-dimensional owner axis")
        values.append(BoundStateValue(key, dimension, leaf.units, leaf.frame, read))
    return tuple(values)


def _carry_reader(target: tuple[str, str]) -> Callable[[State, dict | None], jax.Array]:
    def read(state: State, carry: dict | None) -> jax.Array:
        assert carry is not None
        return carry[target]
    return read


def _bind_carries(graph: Graph, authored: GraphSpec) -> tuple[BoundStateValue, ...]:
    """Migrate the unique mounted legacy GRU hidden self-loop to a wire ID."""
    path = "/inputs/subject/graph/wires"
    expected = {}
    for wire in authored.wires:
        if wire.temporality != "recurrent":
            continue
        node = wire.target_node
        target = (node, wire.target_port)
        initializer = wire.recurrent_initializer or {}
        if (
            (wire.source_node, wire.source_port) != (node, "hidden")
            or wire.target_port != "hidden"
            or initializer.get("state_slot") != "hidden"
            or initializer.get("kind") != "zeros"
            or initializer.get("scope") != "trial"
            or authored.nodes[node].type != "feedbax.GRU"
            or target in expected
        ):
            raise StateLayoutError(
                f"{path}/{wire.authored_id()}", "recurrent wire has no qualified raw carry declaration",
            )
        expected[target] = wire
    actual = {
        (wire.target_node, wire.target_port): (wire.source_node, wire.source_port)
        for wire in graph.wires if wire.temporality == "recurrent"
    }
    if actual != {
        target: (wire.source_node, wire.source_port) for target, wire in expected.items()
    }:
        raise StateLayoutError(path, "runtime carries differ from the authored declarations")
    return tuple(
        BoundStateValue(
            semantic_key("model", "carry", (), "hidden", wire=f"{node}:hidden"),
            f"{node}:unit", "1", None, _carry_reader((node, port)),
        )
        for node, port in expected
    )


def bind_raw_state(
    graph: Graph, authored: GraphSpec, *, selected: frozenset[str] | None = None,
) -> RawStateLayout:
    """Require one declaration for each qualified runtime state leaf and carry."""
    if authored.subgraphs:
        raise StateLayoutError("/inputs/subject/graph/subgraphs", "nested state owners are unavailable")
    if graph.nodes.keys() != authored.nodes.keys() or graph.nodes.keys() != graph.component_bindings.keys():
        raise StateLayoutError("/inputs/subject/graph/nodes", "runtime occurrences differ from declared owners")
    initial = init_state_from_component(graph)
    values = tuple(
        value for name in graph.nodes for value in _bind_owner(name, graph, authored, initial, selected=selected)
    )
    if selected is None:
        return RawStateLayout((*values, *_bind_carries(graph, authored)))
    if selected != {value.key for value in values}:
        raise StateLayoutError("/params/retention", "selected observations are unavailable")
    return RawStateLayout(values)
