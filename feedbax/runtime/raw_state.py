"""Bind stable raw-state declarations to the exact reset and scan carry."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from operator import attrgetter

from equinox.nn import State, StateIndex
import jax
from jax.tree_util import DictKey, GetAttrKey, KeyPath, SequenceKey, keystr
import numpy as np
from pydantic import ValidationError

from feedbax.contracts.expression_units import UNITS
from feedbax.contracts.expression_values import ANGLE, LENGTH, TIME, SignalSpace
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.model_delay import ModelDelayCarry
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.semantic_values import semantic_key
from feedbax.dataset import Variable
from feedbax.runtime.graph import Graph, _nested_cycle_key, init_state_from_component


class StateLayoutError(ValueError):
    def __init__(self, path: str, reason: str):
        self.path = path
        super().__init__(reason)


@dataclass(frozen=True)
class BoundStateValue:
    key: str
    dimension: str | tuple[str, ...] | None
    units: str | None
    read: Callable[[State, dict | None], jax.Array]
    space: SignalSpace | None = None
    position: bool = False
    configuration: bool = False

    @property
    def quantity(self) -> str | None:
        unit = UNITS.get(self.units)
        if unit is None:
            return None
        if self.space == "point":
            return "position"
        if self.space == "free" and unit.dimensions == LENGTH:
            return "displacement"
        return _STATE_QUANTITIES.get(unit.dimensions)


_STATE_QUANTITIES = {
    LENGTH.combine(TIME, -1): "velocity",
    ANGLE: "angle",
    ANGLE.combine(TIME, -1): "angular_velocity",
    UNITS["N"].dimensions: "force",
    UNITS["N*m"].dimensions: "torque",
}


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
                samples.reshape((1,) * (len(axes) - 1) + samples.shape),
                dims, units=value.units, quantity=value.quantity,
                frame="world" if value.space is not None else None,
            )
        return variables, coordinates


def _leaf_reader(index: StateIndex, fields: KeyPath) -> Callable[[State, dict | None], jax.Array]:
    def read(state: State, carry: dict | None) -> jax.Array:
        value = state.get(index)
        for key in fields:
            if isinstance(key, GetAttrKey):
                value = getattr(value, key.name)
            elif isinstance(key, DictKey):
                value = value[key.key]
            elif isinstance(key, SequenceKey):
                value = value[key.idx]
            else:
                raise ValueError(f"unsupported state key {key!r}")
        return value

    return read


def _bind_owner(
    name: str, graph: Graph, authored: GraphSpec, initial: State, *,
    selected: frozenset[str] | None, scope: tuple[str, ...] = (),
) -> tuple[BoundStateValue, ...]:
    component = graph.nodes[name]
    binding = graph.component_bindings[name]
    path = pointer(("inputs", "subject", "graph", *(
        token for parent in scope for token in ("subgraphs", parent)
    ), "nodes", name))
    owner = binding.model_facts
    node = authored.nodes[name]
    if (
        owner is None or owner.states is None or not isinstance(component, binding.owner_class)
        or node.type != binding.type_id or node.revision not in (None, binding.revision)
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
    if not indices and (not declared or not owner.has_state(node)):
        return ()
    if len(indices) != 1 or indices[0] is not index or not declared:
        raise StateLayoutError(path, "runtime state slots differ from the owner declaration")
    storage = initial.get(index)
    leaves = dict(jax.tree.leaves_with_path(storage))
    if not all(
        isinstance(array, jax.Array) and all(isinstance(key, (GetAttrKey, DictKey, SequenceKey)) for key in fields)
        for fields, array in leaves.items()
    ):
        raise StateLayoutError(path, "runtime state leaves differ from the owner declaration")

    values = []
    covered = set()
    for leaf in declared:
        root = tuple(GetAttrKey(field) for field in leaf.fields)
        if leaf.subtree:
            if leaf.fields and not _has_path(storage, leaf.fields):
                raise StateLayoutError(path, "runtime state subtree differs from the owner declaration")
            paths = tuple(fields for fields in leaves if fields[:len(root)] == root)
        else:
            paths = (root,)
        for fields in paths:
            if fields not in leaves or fields in covered:
                raise StateLayoutError(path, "runtime state leaves differ from the owner declaration")
            covered.add(fields)
            relative = fields[len(root):]
            symbol = leaf.symbol + (f"/{keystr(relative, separator='/')}" if relative else "")
            key = semantic_key("model", "state", (*scope, name), symbol)
            if selected is not None and key not in selected:
                continue
            shape = leaves[fields].shape
            dimension = f"{'/'.join((*scope, name))}:{leaf.axis}"
            if leaf.subtree:
                prefix = f"{'/'.join((*scope, name))}:{symbol}"
                dimension = tuple(f"{prefix}:axis{axis}" for axis in range(len(shape))) or None
            elif leaf.axis == "xy":
                if shape != (2,):
                    raise StateLayoutError(path, f"{leaf.symbol} requires two Cartesian coordinates")
                dimension = "xy"
            elif leaf.axis == "muscle" and shape == ():
                dimension = None
            elif len(shape) != 1:
                raise StateLayoutError(path, f"{leaf.symbol} requires a one-dimensional owner axis")
            values.append(BoundStateValue(
                key, dimension, leaf.units, _leaf_reader(index, fields), space=leaf.space,
                position=leaf.space == "point",
                configuration=leaf.configuration,
            ))
    if covered != leaves.keys():
        raise StateLayoutError(path, "runtime state leaves differ from the owner declaration")
    return tuple(values)


def _has_path(value, fields: tuple[str, ...]) -> bool:
    try:
        attrgetter(".".join(fields))(value)
    except AttributeError:
        return False
    return True


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
    """Bind explicit model delays and recurrent reads of declared reset state."""
    path = "/inputs/subject/graph/wires"
    expected = {}
    model_delays = {}
    state_loops = {}
    delay_ids = set()
    for wire in authored.wires:
        if not wire.is_recurrent:
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
        binding = graph.component_bindings.get(node)
        facts = binding.model_facts if binding is not None else None
        state = next((
            state for state in facts.states or ()
            if state.reset is not None and state.symbol == initializer.get("state_slot")
            and state.prototype[0] == wire.source_port
            and state.reset.override_port == wire.target_port
        ), None) if facts is not None else None
        if (
            wire.source_node != node
            or initializer.get("kind") != "zeros"
            or initializer.get("scope") != "trial"
            or state is None or target in expected
        ):
            raise StateLayoutError(
                f"{path}/{wire.authored_id()}", "recurrent wire has no qualified raw carry declaration",
            )
        expected[target] = wire
        state_loops[target] = state
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
            delay.units, _carry_reader(target, scope, delay),
        )
        for target, delay in model_delays.items()
    ) + tuple(
        BoundStateValue(
            semantic_key("model", "carry", scope, state.symbol, wire=f"{node}:{port}"),
            f"{'/'.join((*scope, node))}:{state.axis}", state.units,
            _carry_reader((node, port), scope), space=state.space,
        )
        for (node, port), state in state_loops.items()
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
