"""Declared model fields that a task may initialize for a single trial."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from operator import attrgetter
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple

import jax.numpy as jnp
from equinox.nn import State, StateIndex
from jaxtyping import PRNGKeyArray, PyTree

from feedbax.contracts.expression_units import UNITS
from feedbax.contracts.expression_values import Level, ValueBase, ValueType
from feedbax.contracts.model_address import ComponentElement, FieldElement, ModelAddress
from feedbax.runtime._graph_key_schedule import reset_key
from feedbax.runtime._signal_schedule import ScopedPort, resolve_signal_schedule
from feedbax.runtime.graph import Graph, GraphNode
from feedbax.contracts.model_values import MODEL_OWNERS

if TYPE_CHECKING:
    from feedbax.compiler.resolution import GraphParameterResolution
    from feedbax.component_registry import ComponentRegistry


class ModelResetError(ValueError):
    """An unavailable or incompatible trial initializer."""

    def __init__(self, address: str, detail: str):
        super().__init__(detail)
        self.address = address


@dataclass(frozen=True)
class ModelResetField:
    """A semantic reset destination and its resolved sample contract."""

    address: ModelAddress
    type: ValueType
    dtype: str
    finest: Level = Level.TRIAL


@dataclass(frozen=True)
class _ResetBinding:
    field: ModelResetField
    type_id: str
    revision: int
    owner_class: type[GraphNode]
    paths: tuple[tuple[str, ...], ...]
    override_port: str | None

    def component(self, graph: Graph) -> GraphNode:
        address = self.field.address
        current = graph
        for name in address.composites:
            child = current.nodes.get(name)
            if not isinstance(child, Graph):
                raise ModelResetError(address.to_text(), "The declared reset composite is unavailable")
            current = child
        element = address.element
        assert isinstance(element, FieldElement) and isinstance(element.parent, ComponentElement)
        name = element.parent.id
        component = current.nodes.get(name)
        binding = current.component_bindings.get(name)
        if (
            type(component) is not self.owner_class
            or binding is None
            or (binding.type_id, binding.revision) != (self.type_id, self.revision)
        ):
            raise ModelResetError(address.to_text(), "The runtime reset owner differs from the resolved declaration")
        port = self.override_port
        if port is not None:
            target = (name, port)
            if target in current.input_bindings.values() or any(
                (wire.target_node, wire.target_port) == target for wire in current.wires
            ):
                raise ModelResetError(address.to_text(), "A wire or input overrides the trial field")
        return component


def _state_path(path: str) -> tuple[str, ...]:
    slot, marker, *parts = path.split(".")
    if marker != "init":
        raise ValueError("Trial parameter configuration must use a StateIndex")
    return slot, *parts


def _read_path(value, path):
    return attrgetter(".".join(path))(value) if path else value


def _storage(component, state, path):
    slot = getattr(component, path[0], None)
    if isinstance(slot, StateIndex):
        return _read_path(state.get(slot), path[1:])
    return _read_path(component.state_view(state), path)


def _base(dtype) -> ValueBase | None:
    if jnp.issubdtype(dtype, jnp.bool_):
        return ValueBase.BOOLEAN
    if jnp.issubdtype(dtype, jnp.integer):
        return ValueBase.INTEGER
    if jnp.issubdtype(dtype, jnp.floating):
        return ValueBase.REAL
    return None


class PreparedModelReset(NamedTuple):
    """Fresh trial state and the reads made after all trial writes."""

    state: State
    outputs: dict[str, PyTree]
    cycle_init: dict[tuple[str, str], PyTree]


@dataclass(frozen=True)
class ModelResetPlan:
    """Resolve writable addresses once and apply trial values before reset reads."""

    _bindings: tuple[_ResetBinding, ...] = ()

    @cached_property
    def fields(self) -> tuple[ModelResetField, ...]:
        return tuple(binding.field for binding in self._bindings)

    @cached_property
    def _by_address(self) -> Mapping[str, _ResetBinding]:
        return MappingProxyType({item.field.address.to_text(): item for item in self._bindings})

    def prepare(
        self, graph: Graph, inits: Mapping[str, Any], *,
        key: PRNGKeyArray, inputs: dict[str, PyTree] | None = None,
    ) -> PreparedModelReset:
        """Apply a single trial's values and seed reset outputs and cycle carry."""
        writes = []
        for address in sorted(inits):
            value = inits[address]
            binding = self._by_address.get(address)
            if binding is None:
                raise ModelResetError(address, "Model address is not a declared writable trial field")
            component = binding.component(graph)
            try:
                value = jnp.asarray(value)
            except (TypeError, ValueError) as error:
                raise ModelResetError(address, "Trial initializer must be a numeric or boolean sample") from error
            actual = _base(value.dtype)
            expected = binding.field.type
            compatible = actual == expected.base or (
                expected.base == ValueBase.REAL and actual == ValueBase.INTEGER
            )
            if not compatible or value.shape != expected.shape:
                raise ModelResetError(address, f"Expected {expected.base} sample of shape {expected.shape}")
            writes.append((address, binding, component, value))

        state = graph.init_state(key=key)
        for address, binding, component, value in writes:
            for path in binding.paths:
                try:
                    current = _storage(component, state, path)
                    if (
                        current.shape != binding.field.type.shape
                        or current.dtype.name != binding.field.dtype
                    ):
                        raise ValueError("The runtime reset slot differs from the resolved shape or dtype")
                    state = component.replace_state_value(
                        state, path, jnp.asarray(value, dtype=current.dtype),
                    )
                except (AttributeError, TypeError, ValueError) as error:
                    raise ModelResetError(address, str(error)) from error
        touched = {id(component): component for _, _, component, _ in writes}
        for component in touched.values():
            state = component.state_consistency_update(state)
        state = graph.state_consistency_update(state)
        return PreparedModelReset(
            state, graph.initial_outputs(graph.state_view(state)),
            graph.initial_cycle_port_values(state, inputs=inputs, key=reset_key(key)),
        )


def resolve_model_reset(
    document: str, resolution: GraphParameterResolution, components: ComponentRegistry,
) -> ModelResetPlan:
    """Project owner declarations against the resolved occurrences and prototypes."""
    bindings = []
    schedule = resolve_signal_schedule(resolution.graph)
    driven = {port for _, port in schedule.inputs} | {route.target for route in schedule.routes}

    def project(scope: tuple[str, ...], resolved: GraphParameterResolution) -> None:
        for name, node in resolved.graph.nodes.items():
            if name in resolved.subgraphs:
                project((*scope, name), resolved.subgraphs[name])
                continue
            owner = MODEL_OWNERS.get(node.type)
            meta = components.get(node.type)
            if (
                owner is None or meta is None or owner.revision != node.revision
                or meta.owner_class is None or not issubclass(meta.owner_class, GraphNode)
            ):
                continue
            owner_class = meta.owner_class

            def add(
                path: tuple[str, ...], dtype, shape: tuple[int, ...], unit: str,
                paths: tuple[tuple[str, ...], ...], override_port: str | None,
            ) -> None:
                if override_port is not None and ScopedPort((*scope, name), override_port) in driven:
                    return
                address = ModelAddress(
                    document=document, composites=scope,
                    element=FieldElement(kind="field", parent=ComponentElement(kind="component", id=name), path=path),
                )
                base = _base(dtype)
                if base is None:
                    raise ModelResetError(address.to_text(), "Reset storage needs a real, integer or boolean dtype")
                field = ModelResetField(address, ValueType(
                    base=base, shape=shape, dimensions=UNITS[unit].dimensions,
                ), dtype.name)
                bindings.append(_ResetBinding(
                    field, node.type, owner.revision, owner_class, paths, override_port,
                ))

            for state in owner.states or ():
                reset = state.reset
                if reset is None or node.params.get("stateful") is False:
                    continue
                prototype = resolved.node_outputs[(name, reset.output_port)]
                sample = _read_path(prototype, reset.output_path)
                add(
                    ("reset", *state.symbol.split("/")), sample.dtype, sample.shape, state.units,
                    (state.fields, *reset.aliases), reset.override_port,
                )

            config_paths = {
                key: value for binding in meta.owner_class.config_bindings
                for key, value in binding.projections.items()
            } if owner.trial_parameters else {}
            for parameter in owner.trial_parameters:
                path = _state_path(config_paths[parameter.name])
                if path[0] not in owner.configuration_slots:
                    raise ValueError("Trial parameter must belong to a declared configuration slot")
                dtype = jnp.dtype(jnp.bool_ if parameter.kind == "boolean" else jnp.float32)
                add(
                    ("fields", parameter.name), dtype, (), parameter.units,
                    (path,), owner.trial_parameter_override_port,
                )

    project((), resolution)
    return ModelResetPlan(tuple(bindings))
