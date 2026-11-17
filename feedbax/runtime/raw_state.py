"""Bind stable raw-state declarations to the exact reset and scan carry."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from operator import attrgetter

from equinox.nn import State, StateIndex
import jax
from jax.tree_util import GetAttrKey
import numpy as np
from pydantic import ValidationError

from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.model_delay import ModelDelayCarry
from feedbax.contracts.model_occurrence import graph_leaf_specs, graph_output_sources
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.semantic_values import semantic_key
from feedbax.dataset import Variable
from feedbax.runtime.graph import Graph, _nested_cycle_key, init_state_from_component
from feedbax.contracts.model_values import MODEL_OWNERS, StateValue


class StateLayoutError(ValueError):
    def __init__(self, path: str, reason: str):
        self.path = path
        super().__init__(reason)


def scene_position_key(graph: GraphSpec) -> str | None:
    """Resolve the declared Scene effector from the effective model."""
    leaves = dict(graph_leaf_specs(graph))
    candidates = {
        scope: owner for scope, node in leaves.items()
        if (owner := MODEL_OWNERS.get(node.type)) is not None and owner.scene_position is not None
    }
    if len(candidates) == 1:
        (scope, owner), = candidates.items()
        return semantic_key("model", "state", scope, owner.scene_position)
    positions = {
        (scope, owner.scene_position)
        for scope, port in graph_output_sources(graph)
        if (owner := candidates.get(scope)) is not None and owner.scene_effector_port == port
    }
    if len(positions) != 1:
        return None
    (scope, symbol), = positions
    return semantic_key("model", "state", scope, symbol)


@dataclass(frozen=True)
class BoundStateValue:
    key: str
    dimension: str | tuple[str, ...] | None
    units: str | None
    frame: str | None
    read: Callable[[State, dict | None], jax.Array]
    legacy_carry: bool = False


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
                extra = (value.dimension,) if isinstance(value.dimension, str) else value.dimension
                dims = (*dims, *extra)
                for axis, size in zip(extra, samples.shape[-len(extra):], strict=True):
                    labels = ["x", "y"] if axis == "xy" else np.arange(size)
                    coordinates[axis] = Variable(np.asarray(labels), (axis,))
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
    name: str, graph: Graph, authored: GraphSpec, initial: State, *,
    selected: frozenset[str] | None, scope: tuple[str, ...] = (),
) -> tuple[BoundStateValue, ...]:
    component = graph.nodes[name]
    binding = graph.component_bindings[name]
    path = pointer(("inputs", "subject", "graph", *(
        token for parent in scope for token in ("subgraphs", parent)
    ), "nodes", name))
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
    for slot_name in owner.configuration_slots:
        configured = getattr(component, slot_name, None)
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
        key = semantic_key("model", "state", (*scope, name), leaf.symbol)
        if selected is not None and key not in selected:
            continue
        read = _leaf_reader(index, leaf)
        shape = read(initial, None).shape
        dimension = f"{'/'.join((*scope, name))}:{leaf.axis}"
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


def _carry_reader(
    target: tuple[str, str], scope: tuple[str, ...], declaration: ModelDelayCarry | None = None,
) -> Callable[[State, dict | None], jax.Array]:
    def read(state: State, carry: dict | None) -> jax.Array:
        assert carry is not None
        for parent in scope:
            carry = carry[_nested_cycle_key(parent)]
        value = carry[target]
        if declaration is not None and (
            value.shape != declaration.carry_shape or value.dtype.name != declaration.dtype
        ):
            raise StateLayoutError(
                pointer(("inputs", "subject", "graph", *scope, "wires", declaration.model_wire_id)),
                "runtime carry differs from the declared delay layout",
            )
        return value
    return read


def _delay_dimensions(delay: ModelDelayCarry, scope: tuple[str, ...]) -> tuple[str, ...] | None:
    name = "/".join((*scope, delay.model_wire_id, delay.model_delay_id))
    history = (f"{name}:history",) if delay.steps > 1 else ()
    signal = (name,) if len(delay.shape) == 1 else tuple(
        f"{name}:axis{axis}" for axis in range(len(delay.shape))
    )
    return (*history, *signal) or None


def _bind_carries(graph: Graph, authored: GraphSpec, scope: tuple[str, ...]) -> tuple[BoundStateValue, ...]:
    """Migrate the unique mounted legacy GRU hidden self-loop to a wire ID."""
    path = "/inputs/subject/graph/wires"
    expected = {}
    model_delays = {}
    delay_ids = set()
    for wire in authored.wires:
        if wire.temporality != "recurrent":
            continue
        node = wire.target_node
        target = (node, wire.target_port)
        initializer = wire.recurrent_initializer or {}
        if initializer.get("model_state_sample") is True and initializer.get("kind") == "state_output":
            expected[target] = wire
            continue
        if "model_delay_id" in initializer:
            try:
                declaration = ModelDelayCarry.model_validate(initializer)
            except ValidationError as error:
                raise StateLayoutError(
                    f"{path}/{wire.authored_id()}", "delay has no qualified raw carry layout",
                ) from error
            identity = declaration.model_wire_id, declaration.model_delay_id
            if identity in delay_ids or target in expected:
                raise StateLayoutError(f"{path}/{wire.authored_id()}", "delay carry address is ambiguous")
            delay_ids.add(identity)
            model_delays[target] = declaration
            expected[target] = wire
            continue
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
            semantic_key("model", "carry", scope, delay.model_delay_id, wire=delay.model_wire_id),
            _delay_dimensions(delay, scope),
            delay.units, None, _carry_reader(target, scope, delay),
        )
        for target, delay in model_delays.items()
    ) + tuple(
        BoundStateValue(
            semantic_key("model", "carry", scope, "hidden", wire=f"{node}:hidden"),
            f"{'/'.join((*scope, node))}:unit", "1", None, _carry_reader((node, port), scope),
            legacy_carry=True,
        )
        for (node, port), wire in expected.items()
        if (node, port) not in model_delays and not (wire.recurrent_initializer or {}).get("model_state_sample")
    )


def bind_raw_state(
    graph: Graph, authored: GraphSpec, *, selected: frozenset[str] | None = None,
) -> RawStateLayout:
    """Require one declaration for each qualified runtime state leaf and carry."""
    initial = init_state_from_component(graph)

    def bind(current: Graph, spec: GraphSpec, scope: tuple[str, ...]):
        if current.nodes.keys() != spec.nodes.keys() or current.nodes.keys() != current.component_bindings.keys():
            raise StateLayoutError("/inputs/subject/graph/nodes", "runtime occurrences differ from declared owners")
        for name in current.nodes:
            child = (spec.subgraphs or {}).get(name)
            if isinstance(child, GraphSpec):
                runtime = current.nodes[name]
                if not isinstance(runtime, Graph):
                    raise StateLayoutError("/inputs/subject/graph/nodes", "runtime composite is missing")
                yield from bind(runtime, child, (*scope, name))
            else:
                yield from _bind_owner(name, current, spec, initial, selected=selected, scope=scope)
        if selected is None:
            yield from _bind_carries(current, spec, scope)

    values = tuple(bind(graph, authored, ()))
    if selected is not None and selected != {value.key for value in values}:
        raise StateLayoutError("/params/retention", "selected observations are unavailable")
    return RawStateLayout(values)
