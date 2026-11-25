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

from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.model_delay import ModelDelayCarry
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.retained_state import RetainedStateField, owner_state_field
from feedbax.contracts.semantic_values import semantic_key
from feedbax.dataset import Variable
from feedbax.runtime.delay import DelayLeaf
from feedbax.runtime.graph import Graph, _nested_cycle_key, init_state_from_component


class StateLayoutError(ValueError):
    def __init__(self, path: str, reason: str):
        self.path = path
        super().__init__(reason)


@dataclass(frozen=True)
class BoundStateValue:
    field: RetainedStateField
    read: Callable[[State, dict | None], jax.Array]


@dataclass(frozen=True)
class RawStateLayout:
    values: tuple[BoundStateValue, ...]

    def capture(self, state: State, carry: dict | None) -> dict[str, jax.Array]:
        return {value.field.key: value.read(state, carry) for value in self.values}

    def dataset_fields(
        self, trials: Sequence[Mapping[str, jax.Array]], *, axes: tuple[str, ...],
    ) -> tuple[dict[str, Variable], dict[str, Variable]]:
        """Stack each independent state part on the shared reset-inclusive axis."""
        variables, coordinates = {}, {}
        for value in self.values:
            field = value.field
            samples = np.stack([np.asarray(trial[field.key]) for trial in trials])
            if not np.isfinite(samples).all():
                raise FloatingPointError(f"Nonfinite state samples at {field.key}")
            if extra := field.axes:
                for axis, size in zip(extra, samples.shape[-len(extra):], strict=True):
                    labels = ["x", "y"] if axis == "xy" else np.arange(size)
                    coordinates[axis] = Variable(np.asarray(labels), (axis,))
            descriptor = field.descriptor(axes)
            variables[field.key] = Variable.from_field(
                samples.reshape((1,) * (len(axes) - 1) + samples.shape), descriptor, descriptor.dims,
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
            suffix = f"/{keystr(relative, separator='/')}" if relative else ""
            try:
                field = owner_state_field((*scope, name), leaf, leaves[fields].shape, suffix=suffix)
            except ValueError as error:
                raise StateLayoutError(path, str(error)) from error
            if selected is None or field.key in selected:
                values.append(BoundStateValue(field, _leaf_reader(index, fields)))
    if covered != leaves.keys():
        raise StateLayoutError(path, "runtime state leaves differ from the owner declaration")
    return tuple(values)


def _has_path(value, fields: tuple[str, ...]) -> bool:
    try:
        attrgetter(".".join(fields))(value)
    except AttributeError:
        return False
    return True


def _carry_reader(target: tuple[str, str], scope: tuple[str, ...]):
    def read(state: State, carry: dict | None):
        assert carry is not None
        for parent in scope:
            carry = carry[_nested_cycle_key(parent)]
        return carry[target]
    return read


def _delay_reader(target, scope, wire_id: str, leaf: DelayLeaf, steps: int, index: int):
    carry_at = _carry_reader(target, scope)

    def read(state: State, carry: dict | None) -> jax.Array:
        value = jax.tree.leaves(carry_at(state, carry))[index]
        if value.shape != leaf.carry_shape(steps) or value.dtype != leaf.dtype:
            raise StateLayoutError(
                pointer(("inputs", "subject", "graph", *scope, "wires", wire_id)),
                "runtime carry differs from the resolved delay leaf layout",
            )
        return value
    return read


def _delay_dimensions(
    delay: ModelDelayCarry, scope: tuple[str, ...], leaf: DelayLeaf, steps: int,
) -> tuple[str, ...] | None:
    name = "/".join((*scope, delay.model_wire_id, delay.model_delay_id))
    history = (f"{name}:history",) if steps > 1 else ()
    field = name + pointer(leaf.path) if leaf.path else name
    signal = (field,) if len(leaf.shape) == 1 else tuple(
        f"{field}:axis{axis}" for axis in range(len(leaf.shape))
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
        (wire.target_node, wire.target_port): wire
        for wire in graph.wires if wire.temporality == "recurrent"
    }
    if {
        target: (wire.source_node, wire.source_port) for target, wire in actual.items()
    } != {
        target: (wire.source_node, wire.source_port) for target, wire in expected.items()
    }:
        raise StateLayoutError(path, "runtime carries differ from the authored declarations")
    delays = []
    for target, delay in model_delays.items():
        layout = actual[target].delay_layout
        if layout is None:
            raise StateLayoutError(f"{path}/{delay.model_wire_id}", "delay has no resolved raw carry layout")
        base = semantic_key("model", "carry", scope, delay.model_delay_id, wire=delay.model_wire_id)
        for index, leaf in enumerate(layout.leaves):
            delays.append(BoundStateValue(
                RetainedStateField(
                    base + pointer(leaf.path) if leaf.path else base,
                    _delay_dimensions(delay, scope, leaf, layout.steps) or (), None if leaf.path else delay.units,
                ),
                _delay_reader(target, scope, delay.model_wire_id, leaf, layout.steps, index),
            ))
    return tuple(delays) + tuple(
        BoundStateValue(
            RetainedStateField(
                semantic_key("model", "carry", scope, state.symbol, wire=f"{node}:{port}"),
                (f"{'/'.join((*scope, node))}:{state.axis}",), state.units, space=state.space,
            ),
            _carry_reader((node, port), scope),
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
    if selected is not None and selected != {value.field.key for value in values}:
        raise StateLayoutError("/params/retention", "selected observations are unavailable")
    return RawStateLayout(values)
