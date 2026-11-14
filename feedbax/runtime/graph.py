"""Eager graph components and execution.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

from abc import abstractmethod
import dataclasses
from operator import attrgetter
from collections.abc import Callable
from copy import copy
from typing import ClassVar, Literal, Optional, Self

import equinox as eqx
from equinox import Module, field
from equinox.nn import State, StateIndex
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import PRNGKeyArray, PyTree
from pydantic import BaseModel, ConfigDict

from feedbax.contracts.array_values import ArrayValueSpec
from feedbax.contracts.portable import ComponentOptions, ComponentOwner, GraphTemplate
from feedbax.contracts._causal_inputs import input_driver_conflicts
from feedbax.contracts.model_delay import ModelDelayCarry
from feedbax.runtime._graph import instant_execution_order
from feedbax.runtime._graph_key_schedule import (
    derive_key,
    keyed_nodes,
    reset_key,
    step_key,
    step_keys,
)
from feedbax.runtime._rollout import scan_rollout, select_step_inputs
from feedbax.runtime._signal_schedule import ScopedPort, SignalSchedule
from feedbax.runtime.selectors import Selection, select

_NESTED_CYCLE_NODE = "__nested__"
_MISSING_RESET_INPUT = object()
_ResetInputLookup = Callable[[str], PyTree]


def _nested_cycle_key(node_name: str) -> tuple[str, str]:
    return (_NESTED_CYCLE_NODE, node_name)


def _is_nested_cycle_key(key: tuple[str, str]) -> bool:
    return key[0] == _NESTED_CYCLE_NODE


def _cycle_sample(wire: Wire, carry: PyTree) -> PyTree:
    initializer = wire.recurrent_initializer or {}
    return carry[0] if "model_delay_id" in initializer and initializer.get("steps", 1) > 1 else carry


def _advance_cycle(wire: Wire, carry: PyTree, sample: PyTree) -> PyTree:
    initializer = wire.recurrent_initializer or {}
    if "model_delay_id" in initializer and initializer.get("steps", 1) > 1:
        return jnp.concatenate((carry[1:], jnp.expand_dims(sample, 0)), axis=0)
    return sample


def init_state_from_component(component: "Component") -> State:
    """Collect initial state from all StateIndex instances in a component tree."""

    return State(component)


class GraphNode(Module):
    """Executable graph node, without a promise of portable reconstruction."""

    input_ports: ClassVar[tuple[str, ...]] = ()
    output_ports: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        """Execute the component."""
        ...

    def prepare(self) -> "Self":
        """Name state slots in this complete model before creating state or optimizers.

        Reprepare after composing children into a larger model. Parameter updates retain
        this namespace. Equinox make_with_state remains available for explicit State use,
        but its deleted reset payloads cannot initialize or serialize reset values.
        """
        from feedbax.runtime.state_indices import prepare_component

        return prepare_component(self)

    def init_state(self, *, key: PRNGKeyArray) -> State:
        """Return initial state for this component."""
        # StateIndex initial values are set at construction time.
        # For custom initialization, override this method.
        return init_state_from_component(self)

    def state_consistency_update(self, state: State) -> State:
        """Return a state made self-consistent."""
        return state

    def state_view(self, state: State) -> PyTree | None:
        """Return the state view for this component, if any."""
        idx = getattr(self, "state_index", None)
        if isinstance(idx, StateIndex):
            return state.get(idx)
        return None

    def replace_state_value(self, state: State, path: tuple[str, ...], value: PyTree) -> State:
        """Replace an owned StateIndex value or an attribute within it.

        An explicit StateIndex attribute selects that slot. Other paths are
        relative to the component's primary ``state_index``.
        """
        from feedbax.runtime.state_indices import cast_to_state_dtypes

        idx = getattr(self, path[0], None) if path else None
        explicit_slot = isinstance(idx, StateIndex)
        if explicit_slot:
            path = path[1:]
        else:
            idx = getattr(self, "state_index", None)
        if not isinstance(idx, StateIndex):
            raise ValueError(f"{type(self).__name__} has no StateIndex for path {path!r}")
        current = state.get(idx)
        if path:
            where = attrgetter(".".join(path))
            if explicit_slot:
                value = cast_to_state_dtypes(value, where(current))
            value = eqx.tree_at(where, current, value)
        elif explicit_slot:
            value = cast_to_state_dtypes(value, current)
        return state.set(idx, value)

    def initial_outputs(self, state_value: PyTree | None) -> dict[str, PyTree]:
        """Return outputs inferred from current state value, if possible."""
        if state_value is None:
            return {}
        outputs: dict[str, PyTree] = {}
        for port in self.output_ports:
            if hasattr(state_value, port):
                outputs[port] = attrgetter(port)(state_value)
        return outputs


class Component(GraphNode, ComponentOwner):
    """An executable node with a reconstructible typed configuration."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        from feedbax.contracts.portable import install_ports

        install_ports(cls)

    def __check_init__(self):
        from feedbax.contracts.component import derive_dynamic_port_layout
        from feedbax.contracts.portable import configuration_paths, validate_configuration

        ports = type(self).ports
        if ports is not None:
            validate_configuration(self)
        if ports is not None and ports.dynamic is not None:
            names = (
                (*ports.dynamic.input_name_params, *ports.dynamic.output_name_params)
                if ports.dynamic.count_mode == "names"
                else (ports.dynamic.count_param,)
            )
            values = {}
            paths = configuration_paths(type(self))
            for name in names:
                value = self
                for part in paths.get(name, name).split("."):
                    value = getattr(value, part)
                values[name] = value
            layout = derive_dynamic_port_layout(ports.dynamic, values)
            object.__setattr__(self, "input_ports", layout.input_ports)
            object.__setattr__(self, "output_ports", layout.output_ports)



class RolloutStepContext(Module):
    """Context passed to an optional graph rollout-step hook."""

    graph: "Graph"
    component: Component
    t: PyTree
    key: PRNGKeyArray
    state: State
    step_inputs: dict[str, PyTree]
    port_values: dict[tuple[str, str], PyTree]
    node_name: str = field(static=True)
    node_inputs: dict[str, PyTree]

    @property
    def period_seconds(self) -> float | None:
        return self.graph.period_seconds


class RolloutStepHookResult(Module):
    """Replacement values returned by a graph rollout-step hook.

    ``port_values`` is the complete replacement mapping for the current graph
    step. Hooks that only need to update one port should copy
    ``context.port_values`` first and mutate the copy, so other live port values
    remain available to downstream nodes.
    """

    port_values: Optional[dict[tuple[str, str], PyTree]] = None
    state: Optional[State] = None


RolloutStepHook = Callable[
    [RolloutStepContext],
    RolloutStepHookResult | tuple[Optional[dict[tuple[str, str], PyTree]], Optional[State]] | None,
]


class Wire(Module):
    """A connection between an output port and an input port."""

    source_node: str
    source_port: str
    target_node: str
    target_port: str
    temporality: str = field(default="instant", static=True)
    recurrent_initializer: Optional[dict] = field(default=None, static=True)
    id: str | None = field(default=None, static=True)

    def __repr__(self) -> str:
        arrow = "-[recurrent]->" if self.temporality == "recurrent" else "->"
        return (
            f"Wire({self.source_node}.{self.source_port} {arrow} "
            f"{self.target_node}.{self.target_port})"
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.source_node,
                self.source_port,
                self.target_node,
                self.target_port,
                self.temporality,
            )
        )


class GraphState(Module):
    """Attribute-access view of node states."""

    nodes: dict[str, PyTree]

    def __getattr__(self, name: str):
        if name == "nodes":
            return super().__getattribute__(name)
        if name in self.nodes:
            return self.nodes[name]
        raise AttributeError(name)


def _select_state_path(state_view: PyTree, path: str) -> PyTree:
    """Select a dotted path from a graph state view."""
    current = state_view
    for part in path.split("."):
        if part == "":
            continue
        if isinstance(current, dict):
            current = current[part]
            continue
        if hasattr(current, part):
            current = getattr(current, part)
            continue
        if part == "nodes" and isinstance(current, GraphState):
            current = current.nodes
            continue
        raise ValueError(f"State path {path!r} could not resolve segment {part!r}")
    return current


@dataclasses.dataclass(frozen=True)
class GraphTraceRequest:
    """A single graph-boundary value to retain during execution."""

    kind: Literal["port", "edge", "graph_output", "recurrent_carry", "state_path"]
    selector: str
    node: str | None = None
    port: str | None = None
    source_node: str | None = None
    source_port: str | None = None
    target_node: str | None = None
    target_port: str | None = None
    path: str | None = None
    timing: Literal["input", "output", "step", "initial", "final"] | None = None


@dataclasses.dataclass(frozen=True)
class AuthoredArrayEncoding:
    """An authored array encoding paired with its serialized construction value."""

    path: tuple[str, ...]
    declaration: ArrayValueSpec
    baseline_sha256: bytes


@dataclasses.dataclass(frozen=True)
class ComponentBinding:
    """Registry identity and explicit graph-owned controls for one node."""

    type_id: str
    revision: int
    # Checkpoint slot fingerprints include the binding's repr via the PyTree definition.
    trainable_override: bool | None = dataclasses.field(default=None, repr=False)
    authored_arrays: tuple[AuthoredArrayEncoding, ...] = dataclasses.field(default=(), repr=False)


class Graph(GraphNode):
    """A computational graph of components."""

    # All fields must have defaults to avoid ordering issues with ClassVar
    # inheritance from Component (input_ports, output_ports)
    nodes: dict[str, Component] = field(default_factory=dict)
    wires: tuple[Wire, ...] = field(default_factory=tuple)
    input_bindings: dict[str, tuple[str, str]] = field(default_factory=dict)
    output_bindings: dict[str, tuple[str, str]] = field(default_factory=dict)

    # Override ClassVars from Component with instance fields
    input_ports: tuple[str, ...] = ()
    output_ports: tuple[str, ...] = ()

    state_view_fn: Optional[Callable[[dict[str, PyTree]], PyTree]] = field(
        default=None, static=True
    )
    state_consistency_fn: Optional[Callable[[Graph, State], State]] = field(
        default=None, static=True
    )
    checkpoint: bool = eqx.field(default=False, static=True)
    step_seconds: float | None = field(default=None, static=True)
    inherited_period: float | None = field(default=None, static=True, repr=False)
    parameter_constraints: tuple[object, ...] = field(default_factory=tuple, static=True)
    component_bindings: dict[str, ComponentBinding] = field(default_factory=dict, static=True)

    @property
    def period_seconds(self) -> float | None:
        return self.step_seconds if self.step_seconds is not None else self.inherited_period

    @property
    def _scoped_schedule(self) -> SignalSchedule | None:
        return None

    def __check_init__(
        self,
    ):  # Bug: 4e75416 — ensures validation runs even when subclass overrides __init__
        self._validate_graph()

    def _validate_graph(self) -> None:
        for name in self.nodes:
            if not isinstance(self.nodes[name], GraphNode):
                raise TypeError(f"Node '{name}' is not a GraphNode")

        for wire in self.wires:
            if wire.source_node not in self.nodes:
                raise ValueError(f"Source node '{wire.source_node}' does not exist")
            if wire.target_node not in self.nodes:
                raise ValueError(f"Target node '{wire.target_node}' does not exist")
            source = self.nodes[wire.source_node]
            if wire.source_port not in source.output_ports:
                raise ValueError(
                    f"Wire source port '{wire.source_node}.{wire.source_port}' does not exist"
                )
            target = self.nodes[wire.target_node]
            if wire.target_port not in target.input_ports:
                raise ValueError(
                    f"Wire target port '{wire.target_node}.{wire.target_port}' does not exist"
                )

        for ext_port, (node_name, node_port) in self.input_bindings.items():
            if ext_port not in self.input_ports:
                raise ValueError(f"Input binding '{ext_port}' not in input_ports")
            if node_name not in self.nodes:
                raise ValueError(f"Input binding node '{node_name}' does not exist")
            if node_port not in self.nodes[node_name].input_ports:
                raise ValueError(f"Input binding port '{node_name}.{node_port}' does not exist")

        for ext_port, (node_name, node_port) in self.output_bindings.items():
            if ext_port not in self.output_ports:
                raise ValueError(f"Output binding '{ext_port}' not in output_ports")
            if node_name not in self.nodes:
                raise ValueError(f"Output binding node '{node_name}' does not exist")
            if node_port not in self.nodes[node_name].output_ports:
                raise ValueError(f"Output binding port '{node_name}.{node_port}' does not exist")

        if conflict := next(input_driver_conflicts(self.wires, self.input_bindings), None):
            raise conflict
        self._validate_node_output_recurrent_initializers()

    @property
    def _execution_order(self) -> tuple[str, ...]:
        return self._analyze_execution_order()

    @property
    def _cycle_wires(self) -> tuple[Wire, ...]:
        return tuple(wire for wire in self.wires if wire.temporality == "recurrent")

    @property
    def _needs_iteration(self) -> bool:
        return len(self._cycle_wires) > 0 or any(
            isinstance(node, Graph) and node._needs_iteration for node in self.nodes.values()
        )

    def _node_output_initializer_source(self, wire: Wire) -> tuple[str, str] | None:
        initializer = wire.recurrent_initializer
        if initializer is None or initializer.get("kind") != "node-output":
            return None
        source_node = initializer.get("source_node")
        source_port = initializer.get("source_port")
        if not isinstance(source_node, str) or not source_node:
            raise ValueError(
                f"Wire {wire.source_node}.{wire.source_port} -> "
                f"{wire.target_node}.{wire.target_port} recurrent_initializer "
                "kind 'node-output' requires non-empty string 'source_node'"
            )
        if not isinstance(source_port, str) or not source_port:
            raise ValueError(
                f"Wire {wire.source_node}.{wire.source_port} -> "
                f"{wire.target_node}.{wire.target_port} recurrent_initializer "
                "kind 'node-output' requires non-empty string 'source_port'"
            )
        if source_node not in self.nodes:
            raise ValueError(
                f"Wire {wire.source_node}.{wire.source_port} -> "
                f"{wire.target_node}.{wire.target_port} recurrent_initializer "
                f"kind 'node-output' references missing source node {source_node!r}"
            )
        if source_port not in self.nodes[source_node].output_ports:
            raise ValueError(
                f"Wire {wire.source_node}.{wire.source_port} -> "
                f"{wire.target_node}.{wire.target_port} recurrent_initializer "
                f"kind 'node-output' references missing source port "
                f"{source_node}.{source_port}"
            )
        return source_node, source_port

    def _incoming_wires_by_target_node(self) -> dict[str, list[Wire]]:
        incoming: dict[str, list[Wire]] = {name: [] for name in self.nodes}
        for wire in self.wires:
            incoming.setdefault(wire.target_node, []).append(wire)
        return incoming

    def _node_output_initializer_upstream_nodes(self, wire: Wire) -> set[str]:
        source = self._node_output_initializer_source(wire)
        if source is None:
            return set()
        return self._initializer_upstream_nodes(
            source,
            wire,
            description=(
                f"Wire {wire.source_node}.{wire.source_port} -> "
                f"{wire.target_node}.{wire.target_port} recurrent_initializer kind 'node-output'"
            ),
        )

    def _initializer_upstream_nodes(
        self,
        source: tuple[str, str],
        target: Wire,
        *,
        description: str,
    ) -> set[str]:
        source_node, source_port = source
        incoming = self._incoming_wires_by_target_node()
        upstream: set[str] = set()
        visiting: set[str] = set()

        def visit(node_name: str) -> None:
            if node_name in visiting:
                raise ValueError(f"{description} source sub-DAG contains a cycle at {node_name!r}")
            if node_name in upstream:
                return
            if node_name == target.target_node:
                raise ValueError(
                    f"{description} source {source_node}.{source_port} depends on recurrent "
                    f"target {target.target_node}.{target.target_port}"
                )
            visiting.add(node_name)
            for incoming_wire in incoming.get(node_name, []):
                if incoming_wire.temporality == "recurrent":
                    raise ValueError(
                        f"{description} source {source_node}.{source_port} depends on recurrent edge "
                        f"{incoming_wire.source_node}.{incoming_wire.source_port} -> "
                        f"{incoming_wire.target_node}.{incoming_wire.target_port}"
                    )
                visit(incoming_wire.source_node)
            visiting.remove(node_name)
            upstream.add(node_name)

        visit(source_node)
        return upstream

    def _nested_trial_inputs(self) -> dict[str, set[str]]:
        return {
            name: node.trial_initializer_input_names
            for name, node in self.nodes.items()
            if isinstance(node, Graph)
        }

    def _nested_initializer_wires(
        self, nested_inputs: dict[str, set[str]]
    ) -> dict[tuple[str, str], Wire]:
        return {
            (wire.target_node, wire.target_port): wire
            for wire in self.wires
            if wire.target_port in nested_inputs.get(wire.target_node, ())
        }

    def _nested_initializer_upstream_nodes(self, wire: Wire, description: str) -> set[str]:
        if wire.temporality != "instant":
            raise ValueError(
                f"{description} requires an instant wire from {wire.source_node}.{wire.source_port}"
            )
        nodes = self._initializer_upstream_nodes(
            (wire.source_node, wire.source_port), wire, description=description
        )
        for name in sorted(nodes):
            node = self.nodes[name]
            if isinstance(node, Graph) and node._needs_iteration:
                raise ValueError(f"{description} depends on recurrent subgraph {name!r}")
        return nodes

    def _trial_initializer_dependencies(self) -> tuple[set[str], set[str]]:
        """Return trial-scope input names and nodes that provide reset values."""
        nested_inputs = self._nested_trial_inputs()
        upstream: set[str] = set()
        for (node, port), wire in self._nested_initializer_wires(nested_inputs).items():
            description = f"Nested recurrent initializer input {node}.{port}"
            upstream.update(self._nested_initializer_upstream_nodes(wire, description))
        sources: set[str] = set()
        for wire in self._cycle_wires:
            initializer = wire.recurrent_initializer
            if initializer is None:
                continue
            if initializer.get("kind") == "graph-input":
                source = initializer.get("source")
                if isinstance(source, str):
                    sources.add(source)
            elif initializer.get("kind") == "node-output":
                upstream.update(self._node_output_initializer_upstream_nodes(wire))
        for name, (node, port) in self.input_bindings.items():
            if node in upstream or (node in nested_inputs and port in nested_inputs[node]):
                sources.add(name)
        return sources, upstream

    @property
    def trial_initializer_input_names(self) -> set[str]:
        """Graph inputs consumed at reset, without a leading time axis.

        A wire to a child's trial initializer input consumes its upstream
        graph inputs at reset too. Its source must be computable without
        recurrent edges or stepping another recurrent graph. A consumer of
        time-varying data needs a separate step input.
        """
        return self._trial_initializer_dependencies()[0]

    def _validate_node_output_recurrent_initializers(self) -> None:
        for wire in self.wires:
            initializer = wire.recurrent_initializer
            if (
                wire.temporality != "recurrent"
                or initializer is None
                or initializer.get("kind") != "node-output"
            ):
                continue
            self._node_output_initializer_upstream_nodes(wire)

    def get_node(self, path: str) -> Component:
        """Return a named graph node by dotted graph-node path.

        Args:
            path: Node name or dotted path through nested graph nodes.

        Returns:
            The selected graph component.

        Raises:
            ValueError: If any path segment does not name a graph node.
        """
        current: Component = self
        for part in path.split("."):
            if not part:
                continue
            if not isinstance(current, Graph):
                raise ValueError(f"Node path {path!r} enters non-graph node at {part!r}")
            if part not in current.nodes:
                raise ValueError(f"Unknown graph node {part!r} in node path {path!r}")
            current = current.nodes[part]
        return current

    def get_node_attrs(self, node_path: str, *attr_paths: str) -> tuple[PyTree, ...]:
        """Return one or more attributes from a named graph node.

        This gives downstream training code a public selector target for
        trainable submodules such as ``graph.get_node_attrs("net", "hidden",
        "readout")`` without reaching into ``graph.nodes`` directly.
        """
        node = self.get_node(node_path)
        values = []
        for attr_path in attr_paths:
            try:
                values.append(attrgetter(attr_path)(node))
            except AttributeError as exc:
                raise ValueError(f"Node {node_path!r} has no attribute path {attr_path!r}") from exc
        return tuple(values)

    def _with_runtime_callback(self, field_name: str, fn: Callable[..., PyTree]) -> "Graph":
        updated = copy(self)
        object.__setattr__(updated, field_name, fn)
        return updated

    def with_state_view(self, state_view_fn: Callable[[dict[str, PyTree]], PyTree]) -> "Graph":
        """Return a copy with a runtime-only graph state-view function attached."""

        if not callable(state_view_fn):
            raise TypeError("state_view_fn must be callable")
        return self._with_runtime_callback("state_view_fn", state_view_fn)

    def with_state_consistency(
        self, state_consistency_fn: Callable[[Graph, State], State]
    ) -> "Graph":
        """Return a copy with a callback receiving the current graph and State."""

        if not callable(state_consistency_fn):
            raise TypeError("state_consistency_fn must be callable")
        return self._with_runtime_callback("state_consistency_fn", state_consistency_fn)

    @property
    def _outgoing_wires(self) -> dict[tuple[str, str], list[Wire]]:
        outgoing: dict[tuple[str, str], list[Wire]] = {}
        for wire in self.wires:
            outgoing.setdefault((wire.source_node, wire.source_port), []).append(wire)
        return outgoing

    def _analyze_execution_order(self) -> tuple[str, ...]:
        return instant_execution_order(self.nodes, (
            (wire.source_node, wire.target_node) for wire in self.wires if wire.temporality != "recurrent"
        ))

    def state_view(self, state: State) -> PyTree:
        return self._view_from_state_and_carry(state, {})

    def _rollout_state_view(self, state: State, carry: dict | None) -> PyTree:
        if not carry:
            return self.state_view(state)
        return self._view_from_state_and_carry(state, carry)

    def _view_from_state_and_carry(self, state: State, carry: dict) -> PyTree:
        """Project stored state and named recurrent outputs from the live carry."""
        node_states: dict[str, PyTree] = {}
        for name, node in self.nodes.items():
            child_carry = carry.get(_nested_cycle_key(name))
            node_state = (
                node._rollout_state_view(state, child_carry)
                if isinstance(node, Graph) and child_carry
                else node.state_view(state)
            )
            if node_state is not None:
                node_states[name] = node_state
        if carry:
            for wire in self._cycle_wires:
                alias = (wire.recurrent_initializer or {}).get("state_slot")
                if (
                    isinstance(alias, str)
                    and alias not in node_states
                    and self.output_bindings.get(alias) == (wire.source_node, wire.source_port)
                ):
                    node_states[alias] = carry[(wire.target_node, wire.target_port)]
        if self.state_view_fn is None:
            return GraphState(node_states)
        return self.state_view_fn(node_states)

    def state_consistency_update(self, state: State) -> State:
        if self.state_consistency_fn is None:
            return state
        return self.state_consistency_fn(self, state)

    def initial_outputs(self, state_value: PyTree | None) -> dict[str, PyTree]:
        if state_value is None:
            return {}
        outputs: dict[str, PyTree] = {}
        for ext_port, (node_name, node_port) in self.output_bindings.items():
            if not hasattr(state_value, node_name):
                continue
            node_state = getattr(state_value, node_name)
            node = self.nodes[node_name]
            node_outputs = node.initial_outputs(node_state)
            if node_port in node_outputs:
                outputs[ext_port] = node_outputs[node_port]
        return outputs

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
        n_steps: int | None = None,
        return_state_history: bool = False,
        state_filter: PyTree[bool] = True,
        cycle_init: Optional[dict[tuple[str, str], PyTree]] = None,
        streaming_loss_fn: Optional[Callable] = None,
        rollout_step_hook: Optional[RolloutStepHook] = None,
        t: PyTree = None,
    ) -> tuple[dict[str, PyTree], State] | tuple[dict[str, PyTree], State, PyTree | None]:
        """Execute one step or scan a time-major rollout.

        An explicit ``n_steps`` scans even an acyclic graph. A recurrent graph
        infers the rollout length from its inputs when ``n_steps`` is omitted.
        An acyclic graph with no length returns a single step's outputs and,
        if requested, its unstacked state view.

        A rollout returns an extra value for streaming loss or requested,
        enabled state history. History includes the initial state view.
        Streaming loss and filtered state history require a rollout. Cycle
        overrides require a recurrent graph. For a single step, ``t`` supplies
        the timestep for random streams and hooks and defaults to zero.
        Rollouts start at zero, and an acyclic rollout rejects a supplied ``t``.
        """
        recurrent = self._needs_iteration
        if cycle_init and not recurrent:
            raise ValueError("cycle_init requires a recurrent graph")
        if n_steps is not None or recurrent:
            if t is not None and not recurrent:
                raise ValueError("t is only supported for a single step")
            save_history = return_state_history and state_filter is not False
            outputs, state, aux, _ = self._scan_rollout(
                inputs,
                state,
                key=key,
                n_steps=n_steps,
                save_history=save_history,
                state_filter=state_filter,
                cycle_init=cycle_init,
                streaming_loss_fn=streaming_loss_fn,
                rollout_step_hook=rollout_step_hook,
            )
            if streaming_loss_fn is not None or save_history:
                return outputs, state, aux
            return outputs, state
        if streaming_loss_fn is not None:
            raise ValueError("streaming_loss_fn requires n_steps on an acyclic graph")
        if return_state_history and state_filter is not True:
            raise ValueError("state_filter requires n_steps on an acyclic graph")
        outputs, state = self._call_single_step(
            inputs,
            state,
            key=key,
            t=0 if t is None else t,
            rollout_step_hook=rollout_step_hook,
        )
        if return_state_history:
            state_view = self.state_view(state)
            return outputs, state, state_view
        return outputs, state

    def _call_single_step(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
        rollout_step_hook: Optional[RolloutStepHook] = None,
        t: PyTree = 0,
    ) -> tuple[dict[str, PyTree], State]:
        outputs, state, _, _ = self._transition(
            inputs,
            state,
            key=step_key(key, t),
            t=t,
            rollout_step_hook=rollout_step_hook,
        )
        return outputs, state

    def step(
        self,
        inputs: dict[str, PyTree],
        state: State,
        cycle_port_values: Optional[dict[tuple[str, str], PyTree]] = None,
        *,
        key: PRNGKeyArray,
        t: PyTree = 0,
        rollout_step_hook: Optional[RolloutStepHook] = None,
    ) -> tuple[dict[str, PyTree], State, dict[tuple[str, str], PyTree]]:
        """Advance the graph by one timestep.

        Public single-step API that, unlike ``_call_single_step``, threads
        cycle-wire values across calls so callers can drive a cyclic graph
        one step at a time without reinventing the cycle-wiring logic.

        For graphs without local or nested recurrence, ``step`` reduces to
        ``_call_single_step`` and the returned ``cycle_port_values`` is an
        empty dict. For cyclic graphs, the caller should pass the
        ``cycle_port_values`` returned from the previous call as the
        ``cycle_port_values`` argument of the next call. On the first call,
        pass ``None`` to use defaults derived from ``state`` (mirrors the
        behaviour of ``Graph.__call__(n_steps=...)``).

        Args:
            inputs: External (non-cycle) input port values, keyed by external
                input port name (matches ``input_bindings`` keys).
            state: Current ``equinox.nn.State``.
            cycle_port_values: Cycle-wire port values from the previous step,
                keyed by ``(target_node, target_port)`` tuples. ``None`` on
                the first step uses defaults derived from ``state``.
            key: Trial PRNG key. Reuse it across calls and pass the timestep
                as ``t`` to match a scanned rollout.
            t: Scalar integer timestep in the unsigned 32-bit range.

        Returns:
            ``(outputs, new_state, cycle_port_values)`` where:
              * ``outputs`` is the dict of external output port values.
              * ``new_state`` is the updated ``equinox.nn.State``.
              * ``cycle_port_values`` is the dict of cycle-wire port values
                to pass into the next ``step`` call. Empty for graphs
                without cycles.
        """
        outputs, state, cycle_port_values, _ = self._step(
            inputs,
            state,
            cycle_port_values,
            key=key,
            t=t,
            rollout_step_hook=rollout_step_hook,
        )
        return outputs, state, cycle_port_values

    def step_with_trace(
        self,
        inputs: dict[str, PyTree],
        state: State,
        cycle_port_values: Optional[dict[tuple[str, str], PyTree]] = None,
        *,
        key: PRNGKeyArray,
        trace: tuple[GraphTraceRequest, ...] = (),
        t: PyTree = 0,
        rollout_step_hook: Optional[RolloutStepHook] = None,
    ) -> tuple[
        dict[str, PyTree],
        State,
        dict[tuple[str, str], PyTree],
        dict[str, PyTree],
    ]:
        """Advance the graph one timestep and return selected boundary values.

        This API records values that are already visible at the graph boundary:
        node ports, wires/edges, graph outputs, recurrent carries, and state
        paths. Leaf component internals remain opaque; acausal/equational
        components only need executable ports and state views.
        """
        return self._step(
            inputs,
            state,
            cycle_port_values,
            key=key,
            t=t,
            rollout_step_hook=rollout_step_hook,
            trace=trace,
        )

    def _step(
        self,
        inputs: dict[str, PyTree],
        state: State,
        cycle_port_values: dict[tuple[str, str], PyTree] | None,
        *,
        key: PRNGKeyArray,
        t: PyTree,
        rollout_step_hook: Optional[RolloutStepHook],
        trace: tuple[GraphTraceRequest, ...] = (),
    ) -> tuple[dict[str, PyTree], State, dict[tuple[str, str], PyTree], dict[str, PyTree]]:
        transition_key = step_key(key, t)
        if not self._needs_iteration:
            cycle_port_values = None
        elif cycle_port_values is None:
            cycle_port_values = self._seed_cycle_carry(inputs, state, key)
        return self._transition(
            inputs,
            state,
            cycle_port_values,
            key=transition_key,
            t=t,
            rollout_step_hook=rollout_step_hook,
            trace=trace,
        )

    def _transition(
        self,
        inputs: dict[str, PyTree],
        state: State,
        cycle_port_values: dict[tuple[str, str], PyTree] | None = None,
        *,
        key: PRNGKeyArray,
        t: PyTree,
        rollout_step_hook: Optional[RolloutStepHook],
        trace: tuple[GraphTraceRequest, ...] = (),
    ) -> tuple[dict[str, PyTree], State, dict[tuple[str, str], PyTree], dict[str, PyTree]]:
        port_values = {} if cycle_port_values is None else dict(cycle_port_values)
        if self._scoped_schedule is None and cycle_port_values is not None:
            for wire in self._cycle_wires:
                target = (wire.target_node, wire.target_port)
                port_values[target] = _cycle_sample(wire, cycle_port_values[target])
        trial_scope_sources, initializer_upstream_nodes = (
            (set(), set()) if cycle_port_values is None else self._trial_initializer_dependencies()
        )
        for ext_port, (node_name, node_port) in self.input_bindings.items():
            if ext_port in inputs and ext_port not in trial_scope_sources:
                port_values[(node_name, node_port)] = inputs[ext_port]

        if self._scoped_schedule is None:
            port_values, new_state = self._execute_step(
                port_values,
                state,
                key=key,
                step_inputs=inputs,
                t=t,
                initializer_upstream_nodes=initializer_upstream_nodes,
                rollout_step_hook=rollout_step_hook,
            )
        else:
            if rollout_step_hook is not None:
                raise ValueError("A composite rollout hook requires a contiguous composite schedule")
            port_values, new_state = self._execute_scoped_step(
                self._scoped_schedule, port_values, state, key=key,
            )

        new_cycle_port_values: dict[tuple[str, str], PyTree] = {}
        if cycle_port_values is not None:
            for wire in self._cycle_wires:
                source_key = (wire.source_node, wire.source_port)
                target_key = (wire.target_node, wire.target_port)
                new_cycle_port_values[target_key] = _advance_cycle(
                    wire, cycle_port_values[target_key], port_values[source_key],
                )
            for port_key, value in port_values.items():
                if _is_nested_cycle_key(port_key):
                    new_cycle_port_values[port_key] = value

        outputs = {
            port: port_values[self.output_bindings[port]]
            for port in self.output_ports
            if port in self.output_bindings
        }
        trace_values = {}
        if trace:
            trace_values = self._collect_trace_values(
                trace,
                outputs=outputs,
                port_values=port_values,
                cycle_port_values=new_cycle_port_values,
                state=new_state,
            )
        return outputs, new_state, new_cycle_port_values, trace_values

    def _execute_scoped_step(
        self,
        schedule: SignalSchedule,
        root_values: dict[tuple[str, str], PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[tuple[str, str], PyTree], State]:
        """Interleave leaves while retaining hierarchical state and carry addresses."""
        owners = {(): self}
        keys = {(): key}
        for scope in schedule.order:
            for depth in range(1, len(scope) + 1):
                path = scope[:depth]
                if path in keys:
                    continue
                keys[path] = derive_key(keys[path[:-1]], "occurrence", path[-1])
                node = owners[path[:-1]].nodes[path[-1]]
                if isinstance(node, Graph):
                    owners[path] = node

        previous = {(): root_values}
        for owner in sorted(owners):
            if owner:
                previous[owner] = previous[owner[:-1]].get(_nested_cycle_key(owner[-1]), {})
        values: dict[ScopedPort, PyTree] = {}
        for name, target in schedule.inputs:
            binding = self.input_bindings[name]
            if binding in root_values:
                values[target] = root_values[binding]

        outgoing: dict[ScopedPort, list[ScopedPort]] = {}
        for route in schedule.routes:
            if route.temporality == "instant":
                outgoing.setdefault(route.source, []).append(route.target)
            else:
                wire = owners[route.wire.owner].wires[route.wire.index]
                carry = previous[route.wire.owner][wire.target_node, wire.target_port]
                values[route.target] = _cycle_sample(wire, carry)

        for scope in schedule.order:
            node = owners[scope[:-1]].nodes[scope[-1]]
            inputs = {
                port: values[endpoint]
                for port in node.input_ports if (endpoint := ScopedPort(scope, port)) in values
            }
            outputs, state = node(inputs, state, key=keys[scope])
            for port, value in outputs.items():
                endpoint = ScopedPort(scope, port)
                values[endpoint] = value
                for target in outgoing.get(endpoint, ()):
                    values[target] = value

        for alias, source in schedule.output_aliases:
            if source in values:
                values[alias] = values[source]
        for endpoint, value in values.items():
            if len(endpoint.node) == 1:
                root_values[endpoint.node[0], endpoint.port] = value
        for name, source in schedule.outputs:
            root_values[self.output_bindings[name]] = values[source]

        next_carry: dict[tuple[str, ...], dict[tuple[str, str], PyTree]] = {}
        for route in schedule.routes:
            wire_owner = route.wire.owner
            wire = owners[wire_owner].wires[route.wire.index]
            if route.temporality == "recurrent" and wire_owner:
                target = (wire.target_node, wire.target_port)
                next_carry.setdefault(wire_owner, {})[target] = _advance_cycle(
                    wire, previous[wire_owner][target], values[route.source],
                )
            if not wire_owner and route.target in values:
                root_values.setdefault((wire.target_node, wire.target_port), values[route.target])
        for owner in sorted(owners, reverse=True):
            if owner and owner in next_carry:
                next_carry.setdefault(owner[:-1], {})[_nested_cycle_key(owner[-1])] = next_carry[owner]
        for name in self.nodes:
            root_values.pop(_nested_cycle_key(name), None)
        root_values.update(next_carry.get((), {}))
        return root_values, state

    def _execute_step(
        self,
        port_values: dict[tuple[str, str], PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
        step_inputs: dict[str, PyTree],
        t: PyTree,
        initializer_upstream_nodes: set[str],
        rollout_step_hook: Optional[RolloutStepHook] = None,
    ) -> tuple[dict[tuple[str, str], PyTree], State]:
        execution_order = self._execution_order
        cycle_wire_set = set(self._cycle_wires)
        outgoing_wires = self._outgoing_wires
        nested_cycle_values: dict[str, dict[tuple[str, str], PyTree]] = {}
        for port_key in tuple(port_values):
            if _is_nested_cycle_key(port_key):
                nested_cycle_values[port_key[1]] = port_values.pop(port_key)
        next_nested_cycle_values: dict[str, dict[tuple[str, str], PyTree]] = {}

        for node_name, node_key in keyed_nodes(execution_order, key):
            node = self.nodes[node_name]
            node_inputs = {
                port_name: port_values[(node_name, port_name)]
                for port_name in node.input_ports
                if (node_name, port_name) in port_values
            }
            if node_name in initializer_upstream_nodes and len(node_inputs) < len(node.input_ports):
                continue
            if rollout_step_hook is not None:
                context = RolloutStepContext(
                    graph=self,
                    component=node,
                    t=t,
                    key=derive_key(node_key, "hook"),
                    state=state,
                    step_inputs=step_inputs,
                    port_values=port_values,
                    node_name=node_name,
                    node_inputs=node_inputs,
                )
                port_values, state = self._apply_rollout_step_hook(
                    rollout_step_hook,
                    context,
                    port_values,
                    state,
                )
                node_inputs = {
                    port_name: port_values[(node_name, port_name)]
                    for port_name in node.input_ports
                    if (node_name, port_name) in port_values
                }
            if isinstance(node, Graph):
                child_carry = nested_cycle_values.get(node_name)
                if child_carry is None and node._needs_iteration:
                    raise ValueError(
                        f"Missing recurrent carry for subgraph {node_name!r}. "
                        "Pass the previous step's full carry, or None to seed a trial."
                    )
                node_outputs, state, node_cycle_values, _ = node._transition(
                    node_inputs,
                    state,
                    child_carry,
                    key=node_key,
                    t=t,
                    rollout_step_hook=rollout_step_hook,
                )
                if node_cycle_values:
                    next_nested_cycle_values[node_name] = node_cycle_values
            else:
                node_outputs, state = node(node_inputs, state, key=node_key)

            for port_name, value in node_outputs.items():
                port_values[(node_name, port_name)] = value
                for wire in outgoing_wires.get((node_name, port_name), []):
                    if wire in cycle_wire_set:
                        continue
                    port_values[(wire.target_node, wire.target_port)] = value

        for node_name, node_cycle_values in next_nested_cycle_values.items():
            port_values[_nested_cycle_key(node_name)] = node_cycle_values

        return port_values, state

    def _apply_rollout_step_hook(
        self,
        rollout_step_hook: RolloutStepHook,
        context: RolloutStepContext,
        port_values: dict[tuple[str, str], PyTree],
        state: State,
    ) -> tuple[dict[tuple[str, str], PyTree], State]:
        result = rollout_step_hook(context)
        if result is None:
            return port_values, state
        if isinstance(result, RolloutStepHookResult):
            new_port_values = result.port_values
            new_state = result.state
        elif isinstance(result, tuple) and len(result) == 2:
            new_port_values, new_state = result
        else:
            raise TypeError(
                "rollout_step_hook must return None, RolloutStepHookResult, or (port_values, state)"
            )
        if new_port_values is not None:
            self._validate_rollout_hook_port_keys(new_port_values)
            port_values = new_port_values
        if new_state is not None:
            state = new_state
        return port_values, state

    def _validate_rollout_hook_port_keys(
        self,
        port_values: dict[tuple[str, str], PyTree],
    ) -> None:
        for node_name, port_name in port_values:
            if _is_nested_cycle_key((node_name, port_name)):
                continue
            if node_name not in self.nodes:
                raise ValueError(f"rollout_step_hook returned unknown node {node_name!r}")
            node = self.nodes[node_name]
            known_ports = set(node.input_ports) | set(node.output_ports)
            if port_name not in known_ports:
                raise ValueError(f"rollout_step_hook returned unknown port {node_name}.{port_name}")

    def _collect_trace_values(
        self,
        trace: tuple[GraphTraceRequest, ...],
        *,
        outputs: dict[str, PyTree],
        port_values: dict[tuple[str, str], PyTree],
        cycle_port_values: dict[tuple[str, str], PyTree],
        state: State,
    ) -> dict[str, PyTree]:
        values: dict[str, PyTree] = {}
        for request in trace:
            if request.kind == "port":
                if request.node is None or request.port is None:
                    raise ValueError(f"Trace selector {request.selector!r} is missing node/port")
                key = (request.node, request.port)
                if key not in port_values:
                    raise ValueError(
                        f"Trace selector {request.selector!r} did not produce "
                        f"port {request.node}.{request.port}"
                    )
                values[request.selector] = port_values[key]
                continue

            if request.kind == "edge":
                source_node = request.source_node or request.node
                source_port = request.source_port or request.port
                if source_node is None or source_port is None:
                    raise ValueError(f"Trace selector {request.selector!r} is missing edge source")
                key = (source_node, source_port)
                if key not in port_values:
                    raise ValueError(
                        f"Trace selector {request.selector!r} did not produce "
                        f"edge source {source_node}.{source_port}"
                    )
                if request.target_node is not None and request.target_port is not None:
                    matching = [
                        wire
                        for wire in self.wires
                        if wire.source_node == source_node
                        and wire.source_port == source_port
                        and wire.target_node == request.target_node
                        and wire.target_port == request.target_port
                    ]
                    if not matching:
                        raise ValueError(
                            f"Trace selector {request.selector!r} references missing "
                            f"edge {source_node}.{source_port} -> "
                            f"{request.target_node}.{request.target_port}"
                        )
                values[request.selector] = port_values[key]
                continue

            if request.kind == "graph_output":
                output_name = (
                    request.port or request.path or request.selector.removeprefix("graph_output:")
                )
                if output_name not in outputs:
                    raise ValueError(
                        f"Trace selector {request.selector!r} references missing "
                        f"graph output {output_name!r}"
                    )
                values[request.selector] = outputs[output_name]
                continue

            if request.kind == "recurrent_carry":
                if request.node is None or request.port is None:
                    raise ValueError(
                        f"Trace selector {request.selector!r} is missing recurrent carry node/port"
                    )
                key = (request.node, request.port)
                if key not in cycle_port_values:
                    raise ValueError(
                        f"Trace selector {request.selector!r} references missing "
                        f"recurrent carry {request.node}.{request.port}"
                    )
                values[request.selector] = cycle_port_values[key]
                continue

            if request.kind == "state_path":
                path = request.path or request.selector.removeprefix("state_path:")
                values[request.selector] = _select_state_path(
                    self._rollout_state_view(state, cycle_port_values), path
                )
                continue

            raise ValueError(f"Unsupported trace request kind {request.kind!r}")
        return values

    def initial_cycle_port_values(
        self,
        state: State,
        cycle_init: Optional[dict[tuple[str, str], PyTree]] = None,
        inputs: Optional[dict[str, PyTree]] = None,
        key: PRNGKeyArray | None = None,
    ) -> dict[tuple[str, str], PyTree]:
        """Return the cycle-wire port-value dict to seed the first ``step`` call.

        Equivalent to ``cycle_init`` augmented with values derived from
        ``state`` (via each cycle source node's ``initial_outputs``) and
        recurrent-initializer metadata. For graphs without cycles, returns an
        empty dict.

        Args:
            state: Current ``equinox.nn.State`` to derive defaults from.
            cycle_init: Optional explicit overrides keyed by
                ``(target_node, target_port)``. Takes precedence over
                state-derived defaults.
            inputs: Optional external graph inputs used by trial-scope
                recurrent initializers. Keys match this graph's
                ``input_ports``.
            key: Reset-phase PRNG key. When omitted, uses a deterministic seed.

        Returns:
            Dict keyed by ``(target_node, target_port)`` suitable as the
            ``cycle_port_values`` argument to ``step``.

        Raises:
            ValueError: If a cycle-wire target has neither a ``cycle_init``
                override nor a state-derivable default.
        """
        initial_inputs = inputs or {}
        return self._get_initial_cycle_values(
            state,
            cycle_init,
            input_value=lambda name: initial_inputs.get(name, _MISSING_RESET_INPUT),
            key=key,
        )

    def _seed_cycle_carry(
        self,
        inputs: dict[str, PyTree],
        state: State,
        key: PRNGKeyArray,
        cycle_init: dict[tuple[str, str], PyTree] | None = None,
    ) -> dict[tuple[str, str], PyTree]:
        return self.initial_cycle_port_values(
            state,
            cycle_init,
            inputs=inputs,
            key=reset_key(key),
        )

    def _get_initial_cycle_values(
        self,
        state: State,
        cycle_init: Optional[dict[tuple[str, str], PyTree]] = None,
        *,
        input_value: _ResetInputLookup,
        key: PRNGKeyArray | None = None,
    ) -> dict[tuple[str, str], PyTree]:
        init_values: dict[tuple[str, str], PyTree] = {}
        missing_reasons: dict[tuple[str, str], str] = {}
        cycle_wires = self._cycle_wires

        if cycle_init is not None:
            init_values.update(cycle_init)

        node_states = {
            name: node.state_view(state)
            for name, node in self.nodes.items()
            if node.state_view(state) is not None
        }

        for wire in cycle_wires:
            target_key = (wire.target_node, wire.target_port)
            if target_key in init_values:
                continue
            initializer = wire.recurrent_initializer or {}
            if "model_delay_id" in initializer:
                delay = ModelDelayCarry.model_validate(initializer)
                if delay.reset_output is None:
                    value = jnp.full(delay.shape, delay.initial, dtype=delay.dtype)
                else:
                    end = delay.reset_output
                    outputs = self.nodes[end.component].initial_outputs(node_states.get(end.component))
                    if end.port not in outputs:
                        raise ValueError(f"{wire} reset output {end.component}.{end.port} is unavailable")
                    value = jnp.asarray(outputs[end.port], dtype=delay.dtype)
                    if value.shape != delay.shape:
                        raise ValueError(f"{wire} reset output shape {value.shape} differs from {delay.shape}")
                init_values[target_key] = jnp.broadcast_to(value, delay.carry_shape)
                continue
            if initializer.get("kind") == "state_output" and (
                "source_node" in initializer or "source_port" in initializer
            ):
                reset_node = initializer.get("source_node")
                reset_port = initializer.get("source_port")
                if not isinstance(reset_node, str) or reset_node not in self.nodes:
                    raise ValueError(f"{wire} state_output initializer has an unknown source node")
                reset_outputs = self.nodes[reset_node].initial_outputs(node_states.get(reset_node))
                if not isinstance(reset_port, str) or reset_port not in reset_outputs:
                    raise ValueError(f"{wire} state_output initializer has no initial source port")
                init_values[target_key] = reset_outputs[reset_port]
                continue
            source_state = node_states.get(wire.source_node, None)
            if source_state is None:
                metadata_value = self._initial_value_from_recurrent_initializer(
                    wire,
                    input_value=input_value,
                    state=state,
                    key=key,
                )
                if metadata_value is not None:
                    init_values[target_key] = metadata_value
                    missing_reasons.pop(target_key, None)
                else:
                    missing_reasons[target_key] = (
                        f"{wire.source_node}.{wire.source_port} -> "
                        f"{wire.target_node}.{wire.target_port}: source node "
                        f"{wire.source_node!r} has no initial state and "
                        "recurrent_initializer is absent"
                    )
                continue
            source_node = self.nodes[wire.source_node]
            node_outputs = source_node.initial_outputs(source_state)
            if wire.source_port in node_outputs:
                init_values[target_key] = node_outputs[wire.source_port]
                missing_reasons.pop(target_key, None)
                continue
            metadata_value = self._initial_value_from_recurrent_initializer(
                wire,
                input_value=input_value,
                state=state,
                key=key,
            )
            if metadata_value is not None:
                init_values[target_key] = metadata_value
                missing_reasons.pop(target_key, None)
            else:
                missing_reasons[target_key] = (
                    f"{wire.source_node}.{wire.source_port} -> "
                    f"{wire.target_node}.{wire.target_port}: source initial_outputs "
                    f"does not expose port {wire.source_port!r} and "
                    "recurrent_initializer is absent"
                )

        nested_nodes = {
            name: node
            for name, node in self.nodes.items()
            if isinstance(node, Graph)
            and node._needs_iteration
            and _nested_cycle_key(name) not in init_values
        }
        nested_wires = self._nested_initializer_wires(
            {name: node.trial_initializer_input_names for name, node in nested_nodes.items()}
        )
        reset_outputs: dict[tuple[str, str], PyTree] = {}

        def nested_input_value(target: tuple[str, str]) -> PyTree:
            if target not in nested_wires:
                for name, binding in self.input_bindings.items():
                    if binding == target:
                        return input_value(name)
                return _MISSING_RESET_INPUT

            wire = nested_wires[target]
            source = (wire.source_node, wire.source_port)
            if source not in reset_outputs:
                description = f"Nested recurrent initializer input {target[0]}.{target[1]}"
                outputs = self._evaluate_initializer_outputs(
                    self._nested_initializer_upstream_nodes(wire, description),
                    input_value,
                    state,
                    key=key,
                    description=description,
                )
                if source not in outputs:
                    raise ValueError(
                        f"{description} did not receive output {source[0]}.{source[1]}"
                    )
                reset_outputs[source] = outputs[source]
            return reset_outputs[source]

        for node_name, node_key in keyed_nodes(tuple(nested_nodes), key):
            node = nested_nodes[node_name]
            init_values[_nested_cycle_key(node_name)] = node._get_initial_cycle_values(
                state,
                input_value=lambda port, name=node_name: nested_input_value((name, port)),
                key=reset_key(node_key),
            )

        missing = [
            (wire.target_node, wire.target_port)
            for wire in cycle_wires
            if (wire.target_node, wire.target_port) not in init_values
        ]
        if missing:
            raise ValueError(
                "Missing initial values for recurrent cycle wires: "
                + "; ".join(
                    missing_reasons.get((n, p), f"{n}.{p}: no value available") for n, p in missing
                )
            )

        return init_values

    def _initial_value_from_recurrent_initializer(
        self,
        wire: Wire,
        *,
        input_value: _ResetInputLookup | None = None,
        state: State | None = None,
        key: PRNGKeyArray | None = None,
    ) -> PyTree | None:
        initializer = wire.recurrent_initializer
        if initializer is None:
            return None
        kind = initializer.get("kind")
        if kind == "zeros":
            shape = initializer.get("shape")
            if shape is None:
                raise ValueError(
                    f"Wire {wire.source_node}.{wire.source_port} -> "
                    f"{wire.target_node}.{wire.target_port} recurrent_initializer "
                    "kind 'zeros' requires 'shape'"
                )
            try:
                return jnp.zeros(tuple(int(dim) for dim in shape), dtype=initializer.get("dtype"))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Wire {wire.source_node}.{wire.source_port} -> "
                    f"{wire.target_node}.{wire.target_port} recurrent_initializer "
                    f"has invalid shape {shape!r} or dtype {initializer.get('dtype')!r}"
                ) from exc
        if kind == "constant":
            if "value" not in initializer:
                raise ValueError(
                    f"Wire {wire.source_node}.{wire.source_port} -> "
                    f"{wire.target_node}.{wire.target_port} recurrent_initializer "
                    "kind 'constant' requires 'value'"
                )
            dtype = initializer.get("dtype")
            if dtype is not None:
                try:
                    dtype = jnp.dtype(dtype)
                except TypeError as exc:
                    raise ValueError(
                        f"Wire {wire.source_node}.{wire.source_port} -> "
                        f"{wire.target_node}.{wire.target_port} recurrent_initializer "
                        f"kind 'constant' has invalid dtype {dtype!r}"
                    ) from exc
            return jnp.asarray(initializer["value"], dtype=dtype)
        if kind == "graph-input":
            source = initializer.get("source")
            if not isinstance(source, str) or not source:
                raise ValueError(
                    f"Wire {wire.source_node}.{wire.source_port} -> "
                    f"{wire.target_node}.{wire.target_port} recurrent_initializer "
                    "kind 'graph-input' requires non-empty string 'source'"
                )
            value = _MISSING_RESET_INPUT if input_value is None else input_value(source)
            if value is _MISSING_RESET_INPUT:
                raise ValueError(
                    f"Wire {wire.source_node}.{wire.source_port} -> "
                    f"{wire.target_node}.{wire.target_port} recurrent_initializer "
                    f"kind 'graph-input' requires input {source!r}"
                )
            return value
        if kind == "node-output":
            if state is None:
                raise ValueError(
                    f"Wire {wire.source_node}.{wire.source_port} -> "
                    f"{wire.target_node}.{wire.target_port} recurrent_initializer "
                    "kind 'node-output' requires graph state"
                )
            return self._evaluate_node_output_initializer(
                wire,
                input_value=input_value or (lambda name: _MISSING_RESET_INPUT),
                state=state,
                key=key,
            )
        raise ValueError(
            f"Wire {wire.source_node}.{wire.source_port} -> "
            f"{wire.target_node}.{wire.target_port} has unsupported "
            f"recurrent_initializer kind {kind!r}"
        )

    def _evaluate_node_output_initializer(
        self,
        wire: Wire,
        *,
        input_value: _ResetInputLookup,
        state: State,
        key: PRNGKeyArray | None,
    ) -> PyTree:
        source = self._node_output_initializer_source(wire)
        if source is None:
            raise ValueError("node-output initializer source is absent")
        source_node, source_port = source
        description = (
            f"Wire {wire.source_node}.{wire.source_port} -> "
            f"{wire.target_node}.{wire.target_port} recurrent_initializer kind 'node-output'"
        )
        outputs = self._evaluate_initializer_outputs(
            self._node_output_initializer_upstream_nodes(wire),
            input_value,
            state,
            key=key,
            description=f"{description} could not evaluate source {source_node}.{source_port}",
        )
        if source not in outputs:
            raise ValueError(f"{description} did not produce source {source_node}.{source_port}")
        return outputs[source]

    def _evaluate_initializer_outputs(
        self,
        upstream_nodes: set[str],
        input_value: _ResetInputLookup,
        state: State,
        *,
        key: PRNGKeyArray | None,
        description: str,
    ) -> dict[tuple[str, str], PyTree]:
        port_values: dict[tuple[str, str], PyTree] = {}
        for name, (node, port) in self.input_bindings.items():
            if node in upstream_nodes:
                value = input_value(name)
                if value is not _MISSING_RESET_INPUT:
                    port_values[(node, port)] = value

        # State.set invalidates its input object. Initializer evaluation must
        # leave the reset state available to the live rollout.
        eval_state = jt.map(lambda value: value, state)
        outputs: dict[tuple[str, str], PyTree] = {}
        outgoing_wires = self._outgoing_wires
        producer_order = (name for name in self._execution_order if name in upstream_nodes)
        for node_name, node_key in keyed_nodes(producer_order, derive_key(key, "initializer")):
            node = self.nodes[node_name]
            missing_inputs = [
                port_name
                for port_name in node.input_ports
                if (node_name, port_name) not in port_values
            ]
            if missing_inputs:
                missing_text = ", ".join(f"{node_name}.{port}" for port in missing_inputs)
                raise ValueError(f"{description}; missing input(s): {missing_text}")
            node_inputs = {
                port_name: port_values[(node_name, port_name)] for port_name in node.input_ports
            }
            if isinstance(node, Graph):
                node_outputs, eval_state, _, _ = node._transition(
                    node_inputs,
                    eval_state,
                    key=node_key,
                    t=0,
                    rollout_step_hook=None,
                )
            else:
                node_outputs, eval_state = node(node_inputs, eval_state, key=node_key)

            for port_name, value in node_outputs.items():
                outputs[(node_name, port_name)] = value
                for outgoing_wire in outgoing_wires.get((node_name, port_name), []):
                    if outgoing_wire.temporality != "instant":
                        continue
                    port_values[(outgoing_wire.target_node, outgoing_wire.target_port)] = value

        return outputs

    def rollout_with_trace(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
        n_steps: int,
        trace: tuple[GraphTraceRequest, ...],
        cycle_init: Optional[dict[tuple[str, str], PyTree]] = None,
        rollout_step_hook: Optional[RolloutStepHook] = None,
    ) -> tuple[dict[str, PyTree], State, dict[str, PyTree]]:
        """Roll out a graph with selected post-step observations.

        Trial initializer inputs are passed without a time axis. Other inputs
        are time-major. Each trace entry aligns with the output of that step.
        """
        outputs, final_state, traces, _ = self.rollout_with_trace_and_history(
            inputs, state, key=key, n_steps=n_steps, trace=trace,
            cycle_init=cycle_init, rollout_step_hook=rollout_step_hook,
        )
        return outputs, final_state, traces

    def rollout_with_trace_and_history(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
        n_steps: int,
        trace: tuple[GraphTraceRequest, ...],
        capture_state: Callable[[State, dict | None], PyTree] | None = None,
        cycle_init: Optional[dict[tuple[str, str], PyTree]] = None,
        rollout_step_hook: Optional[RolloutStepHook] = None,
    ) -> tuple[dict[str, PyTree], State, dict[str, PyTree], PyTree | None]:
        """Capture declared reset-inclusive state in the same scan as post-step traces."""
        outputs, final_state, history, traces = self._scan_rollout(
            inputs, state, key=key, n_steps=n_steps, trace=trace,
            cycle_init=cycle_init, rollout_step_hook=rollout_step_hook,
            save_history=capture_state is not None, capture_state=capture_state,
        )
        return outputs, final_state, traces, history

    def _scan_rollout(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
        n_steps: int | None = None,
        save_history: bool = False,
        state_filter: PyTree[bool] = True,
        cycle_init: Optional[dict[tuple[str, str], PyTree]] = None,
        streaming_loss_fn: Optional[Callable] = None,
        rollout_step_hook: Optional[RolloutStepHook] = None,
        trace: tuple[GraphTraceRequest, ...] = (),
        checkpoint: bool | None = None,
        capture_state: Callable[[State, dict | None], PyTree] | None = None,
    ) -> tuple[dict[str, PyTree], State, PyTree | None, dict[str, PyTree]]:
        trial_scope_sources = self.trial_initializer_input_names
        scan_inputs = {
            name: value for name, value in inputs.items() if name not in trial_scope_sources
        }
        if n_steps is None:
            if not scan_inputs:
                raise ValueError("n_steps is required when inputs are empty")
            first_input = next(iter(scan_inputs.values()))
            first_leaf = jt.leaves(first_input)[0]
            n_steps = int(first_leaf.shape[0])

        keys = step_keys(key, n_steps)

        init_cycle_values = (
            self._seed_cycle_carry(inputs, state, key, cycle_init)
            if self._needs_iteration
            else None
        )

        # Broadcast scalar input leaves to (n_steps, ...) so they can be
        # indexed by step.  Non-scalar leaves pass through as-is.
        def _broadcast_scalar(x):
            if hasattr(x, "ndim") and x.ndim == 0:
                return jnp.broadcast_to(x, (n_steps,))
            return x

        step_inputs_seq = select_step_inputs(jt.map(_broadcast_scalar, scan_inputs), n_steps)

        def step_fn(step_inputs, step_state, prev_cycle_values, step_key, t):
            outputs, next_state, next_cycle_values, observations = self._transition(
                step_inputs,
                step_state,
                prev_cycle_values,
                key=step_key,
                trace=trace,
                t=t,
                rollout_step_hook=rollout_step_hook,
            )
            step_outputs = (outputs, observations) if trace else outputs
            step_carry = next_cycle_values if prev_cycle_values is not None else None
            return step_outputs, next_state, step_carry

        output_seq, final_state, aux = scan_rollout(
            step_fn,
            state,
            step_inputs_seq,
            keys,
            n_steps,
            state_view_fn=self._rollout_state_view if capture_state is None else capture_state,
            init_step_carry=init_cycle_values,
            checkpoint=self.checkpoint if checkpoint is None else checkpoint,
            save_history=save_history,
            state_filter=state_filter,
            streaming_loss_fn=streaming_loss_fn,
        )

        outputs, traces = output_seq if trace else (output_seq, {})
        return outputs, final_state, aux, traces

    # ========== Graph Surgery API ==========

    def add_node(self, name: str, component: Component) -> "Graph":
        self._check_topology_edit()
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists")
        return eqx.tree_at(
            lambda g: g.nodes,
            self,
            {**self.nodes, name: component},
        )

    def remove_node(self, name: str) -> "Graph":
        """Remove a local node, its incident wires, bindings, and parameter constraints."""
        from feedbax.runtime.parameter_constraints import normalize_parameter_constraints

        self._check_topology_edit()
        if name not in self.nodes:
            raise ValueError(f"Node '{name}' does not exist")

        new_nodes = {k: v for k, v in self.nodes.items() if k != name}
        new_wires = tuple(w for w in self.wires if w.source_node != name and w.target_node != name)
        new_input_bindings = {k: v for k, v in self.input_bindings.items() if v[0] != name}
        new_output_bindings = {k: v for k, v in self.output_bindings.items() if v[0] != name}

        graph = eqx.tree_at(
            lambda g: (g.nodes, g.wires, g.input_bindings, g.output_bindings),
            self,
            (new_nodes, new_wires, new_input_bindings, new_output_bindings),
        )
        object.__setattr__(
            graph,
            "component_bindings",
            {k: v for k, v in self.component_bindings.items() if k != name},
        )
        object.__setattr__(
            graph,
            "parameter_constraints",
            tuple(
                constraint
                for constraint in normalize_parameter_constraints(self.parameter_constraints)
                if constraint.node != name
            ),
        )
        return graph

    def _with_wires(self, wires: tuple[Wire, ...]) -> "Graph":
        self._check_topology_edit()
        graph = eqx.tree_at(lambda g: g.wires, self, wires)
        graph._validate_graph()
        return graph

    def add_wire(self, wire: Wire) -> "Graph":
        return self._with_wires(self.wires + (wire,))

    def remove_wire(self, wire: Wire) -> "Graph":
        self._check_topology_edit()
        new_wires = tuple(w for w in self.wires if w != wire)
        return eqx.tree_at(
            lambda g: g.wires,
            self,
            new_wires,
        )

    def _check_topology_edit(self) -> None:
        if self._scoped_schedule is not None:
            raise ValueError("Edit the authored graph and recompile its interleaved schedule")

    def insert_between(
        self,
        node_name: str,
        component: Component,
        source_node: str,
        source_port: str,
        target_node: str,
        target_port: str,
        *,
        input_port: str = "input",
        output_port: str = "output",
    ) -> "Graph":
        old_wire = next(
            (
                wire
                for wire in self.wires
                if (wire.source_node, wire.source_port, wire.target_node, wire.target_port)
                == (source_node, source_port, target_node, target_port)
            ),
            None,
        )
        if old_wire is None:
            raise ValueError(f"No wire {source_node}.{source_port} -> {target_node}.{target_port}")

        from feedbax.contracts.graph import derived_wire_id

        recurrent = old_wire.temporality == "recurrent"
        # The new component receives the value previously delivered to the target,
        # including any recurrent delay and reset value on that connection.
        incoming = Wire(
            source_node,
            source_port,
            node_name,
            input_port,
            old_wire.temporality,
            old_wire.recurrent_initializer,
            id=old_wire.id if recurrent else derived_wire_id(old_wire.id, node_name, "incoming"),
        )
        outgoing = Wire(
            node_name, output_port, target_node, target_port,
            id=derived_wire_id(old_wire.id, node_name, "outgoing") if recurrent else old_wire.id,
        )
        wires = tuple(wire for wire in self.wires if wire is not old_wire)
        return self.add_node(node_name, component)._with_wires((*wires, incoming, outgoing))

    # ========== Selection API ==========

    def select(self) -> Selection["Graph"]:
        """Create a Selection over this Graph.

        Returns:
            A Selection object for fluent selection and modification.

        Example:
            >>> graph.select().at_instances_of(jnp.ndarray).apply(jnp.zeros_like)
        """
        return select(self)

    def select_node(self, name: str) -> Selection["Graph"]:
        """Create a Selection targeting a specific node by name.

        Args:
            name: The name of the node to select.

        Returns:
            A Selection targeting the specified node.

        Raises:
            KeyError: If the node name does not exist.

        Example:
            >>> graph.select_node("encoder").apply(lambda n: modified_encoder)
        """
        if name not in self.nodes:
            raise KeyError(f"Node '{name}' does not exist in graph")
        return select(self).at(lambda g: g.nodes[name])

    def select_nodes_of_type(self, *types: type) -> Selection["Graph"]:
        """Create a Selection targeting all nodes of the given types.

        Args:
            *types: One or more Component types to match.

        Returns:
            A Selection targeting nodes that are instances of any of the types.

        Example:
            >>> graph.select_nodes_of_type(LinearLayer, MLPLayer).apply(reinit_fn)
        """

        # Build a filter spec for the nodes dict
        def type_predicate(x: Component) -> bool:
            return isinstance(x, types)

        return select(self).at(lambda g: g.nodes).at_instances_of(*types)


class _InterleavedGraph(Graph):
    """A causal graph whose leaf dependencies cross its composite boundaries."""

    schedule: SignalSchedule = field(kw_only=True, static=True, repr=False)

    @property
    def _scoped_schedule(self) -> SignalSchedule:
        return self.schedule


class _SubgraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Subgraph(GraphTemplate):
    """Editable nested graph container."""

    config = _SubgraphConfig
    component_options = ComponentOptions(is_composite=True)

    @classmethod
    def expand(cls, config):
        from feedbax.contracts.graph import GraphSpec

        return GraphSpec()
