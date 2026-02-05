"""Eager graph components and execution.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

from abc import abstractmethod
import dataclasses
from functools import cached_property
from operator import attrgetter
from typing import ClassVar, Optional

import equinox as eqx
from equinox import Module, field
from equinox.nn import State, StateIndex
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import PRNGKeyArray, PyTree

from feedbax._graph import detect_cycles_and_sort
from feedbax._selectors import Selection, select


def init_state_from_component(component: "Component") -> State:
    """Collect initial state from all StateIndex instances in a component tree."""
    def _state_index_init(idx: StateIndex):
        for name in ("value", "init", "initial_value", "_value", "_init"):
            if hasattr(idx, name):
                return getattr(idx, name)
        if dataclasses.is_dataclass(idx):
            fields = dataclasses.fields(idx)
            if len(fields) == 1:
                return getattr(idx, fields[0].name)
        raise ValueError("StateIndex initial value not found")

    # State() constructor requires a model argument in newer equinox versions
    # We pass the component to extract StateIndex values
    state = State(component)
    seen_ids: set[int] = set()

    def _set_index(idx: StateIndex, init_value) -> None:
        nonlocal state
        idx_id = id(idx)
        if idx_id in seen_ids:
            return
        seen_ids.add(idx_id)
        state = state.set(idx, init_value)

    def _iter_state_indices(obj) -> list[StateIndex]:
        indices: list[StateIndex] = []
        if dataclasses.is_dataclass(obj):
            for field_obj in dataclasses.fields(obj):
                try:
                    value = getattr(obj, field_obj.name)
                except Exception:
                    continue
                if isinstance(value, StateIndex):
                    indices.append(value)
        else:
            for value in getattr(obj, "__dict__", {}).values():
                if isinstance(value, StateIndex):
                    indices.append(value)
        return indices

    def _walk(x):
        nonlocal state
        if isinstance(x, StateIndex):
            _set_index(x, _state_index_init(x))
            return

        indices = _iter_state_indices(x)
        if indices:
            for idx in indices:
                _set_index(idx, _state_index_init(idx))

        children, _ = jax.tree_util.tree_flatten(
            x, is_leaf=lambda y: isinstance(y, StateIndex)
        )
        if len(children) == 1 and children[0] is x:
            return
        for child in children:
            _walk(child)

    _walk(component)
    return state


class Component(Module):
    """Base class for all graph nodes."""

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

    def initial_outputs(self, state_value: PyTree | None) -> dict[str, PyTree]:
        """Return outputs inferred from current state value, if possible."""
        if state_value is None:
            return {}
        outputs: dict[str, PyTree] = {}
        for port in self.output_ports:
            if hasattr(state_value, port):
                outputs[port] = attrgetter(port)(state_value)
        return outputs

    def intervention_state_indices(self) -> dict[str, StateIndex]:
        """Return labels mapped to StateIndex for intervention params."""
        return {}


class Wire(Module):
    """A connection between an output port and an input port."""

    source_node: str
    source_port: str
    target_node: str
    target_port: str

    def __repr__(self) -> str:
        return (
            f"Wire({self.source_node}.{self.source_port} -> "
            f"{self.target_node}.{self.target_port})"
        )

    def __hash__(self) -> int:
        return hash((self.source_node, self.source_port, self.target_node, self.target_port))


class GraphState(Module):
    """Attribute-access view of node states."""

    nodes: dict[str, PyTree]

    def __getattr__(self, name: str):
        if name == "nodes":
            return super().__getattribute__(name)
        if name in self.nodes:
            return self.nodes[name]
        raise AttributeError(name)


class Graph(Component):
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

    state_view_fn: Optional[callable] = field(default=None, static=True)
    state_consistency_fn: Optional[callable] = field(default=None, static=True)

    def __post_init__(self):
        self._validate_graph()

    def _validate_graph(self) -> None:
        for name in self.nodes:
            if not isinstance(self.nodes[name], Component):
                raise TypeError(f"Node '{name}' is not a Component")

        for wire in self.wires:
            if wire.source_node not in self.nodes:
                raise ValueError(f"Source node '{wire.source_node}' does not exist")
            if wire.target_node not in self.nodes:
                raise ValueError(f"Target node '{wire.target_node}' does not exist")

        for ext_port, (node_name, node_port) in self.input_bindings.items():
            if ext_port not in self.input_ports:
                raise ValueError(f"Input binding '{ext_port}' not in input_ports")
            if node_name not in self.nodes:
                raise ValueError(f"Input binding node '{node_name}' does not exist")
            if node_port not in self.nodes[node_name].input_ports:
                raise ValueError(
                    f"Input binding port '{node_name}.{node_port}' does not exist"
                )

        for ext_port, (node_name, node_port) in self.output_bindings.items():
            if ext_port not in self.output_ports:
                raise ValueError(f"Output binding '{ext_port}' not in output_ports")
            if node_name not in self.nodes:
                raise ValueError(f"Output binding node '{node_name}' does not exist")
            if node_port not in self.nodes[node_name].output_ports:
                raise ValueError(
                    f"Output binding port '{node_name}.{node_port}' does not exist"
                )

    @cached_property
    def _cycle_analysis(self) -> tuple[tuple[str, ...], tuple[Wire, ...]]:
        return self._analyze_cycles()

    @property
    def _execution_order(self) -> tuple[str, ...]:
        return self._cycle_analysis[0]

    @property
    def _cycle_wires(self) -> tuple[Wire, ...]:
        return self._cycle_analysis[1]

    @cached_property
    def _cycle_wire_set(self) -> set[Wire]:
        return set(self._cycle_wires)

    @property
    def _needs_iteration(self) -> bool:
        return len(self._cycle_wires) > 0

    @cached_property
    def _outgoing_wires(self) -> dict[tuple[str, str], list[Wire]]:
        outgoing: dict[tuple[str, str], list[Wire]] = {}
        for wire in self.wires:
            outgoing.setdefault((wire.source_node, wire.source_port), []).append(wire)
        return outgoing

    def _analyze_cycles(self) -> tuple[tuple[str, ...], tuple[Wire, ...]]:
        adjacency = {name: set() for name in self.nodes}
        wire_lookup: dict[tuple[str, str], list[Wire]] = {}
        for wire in self.wires:
            adjacency[wire.source_node].add(wire.target_node)
            wire_lookup.setdefault((wire.source_node, wire.target_node), []).append(wire)

        execution_order, back_edges = detect_cycles_and_sort(adjacency)

        cycle_wires: list[Wire] = []
        for src, tgt in back_edges:
            cycle_wires.extend(wire_lookup.get((src, tgt), []))

        return tuple(execution_order), tuple(cycle_wires)

    def state_view(self, state: State) -> PyTree:
        node_states = {
            name: node.state_view(state)
            for name, node in self.nodes.items()
            if node.state_view(state) is not None
        }
        if self.state_view_fn is None:
            return GraphState(node_states)
        return self.state_view_fn(node_states)

    def intervention_state_indices(self) -> dict[str, StateIndex]:
        indices: dict[str, StateIndex] = {}
        for name, node in self.nodes.items():
            node_indices = node.intervention_state_indices()
            for label, idx in node_indices.items():
                if label in indices:
                    raise ValueError(
                        f"Duplicate intervention label '{label}' in graph nodes "
                        f"('{name}' conflicts with another node)."
                    )
                indices[label] = idx
        return indices

    def state_consistency_update(self, state: State) -> State:
        if self.state_consistency_fn is None:
            return state
        return self.state_consistency_fn(state)

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
    ) -> tuple[dict[str, PyTree], State] | tuple[dict[str, PyTree], State, PyTree | None]:
        if self._needs_iteration:
            return self._call_with_iteration(
                inputs,
                state,
                key=key,
                n_steps=n_steps,
                return_state_history=return_state_history,
                state_filter=state_filter,
                cycle_init=cycle_init,
            )
        outputs, state = self._call_single_step(inputs, state, key=key)
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
    ) -> tuple[dict[str, PyTree], State]:
        keys = (
            jax.random.split(key, len(self._execution_order))
            if self._execution_order
            else ()
        )

        port_values: dict[tuple[str, str], PyTree] = {}

        for ext_port, (node_name, node_port) in self.input_bindings.items():
            port_values[(node_name, node_port)] = inputs[ext_port]

        for node_name, node_key in zip(self._execution_order, keys):
            node = self.nodes[node_name]
            node_inputs = {
                port_name: port_values[(node_name, port_name)]
                for port_name in node.input_ports
                if (node_name, port_name) in port_values
            }
            node_outputs, state = node(node_inputs, state, key=node_key)

            for port_name, value in node_outputs.items():
                port_values[(node_name, port_name)] = value
                for wire in self._outgoing_wires.get((node_name, port_name), []):
                    if wire in self._cycle_wire_set:
                        continue
                    port_values[(wire.target_node, wire.target_port)] = value

        outputs = {
            ext_port: port_values[(node_name, node_port)]
            for ext_port, (node_name, node_port) in self.output_bindings.items()
        }

        return outputs, state

    def _execute_step(
        self,
        port_values: dict[tuple[str, str], PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[tuple[str, str], PyTree], State]:
        keys = (
            jax.random.split(key, len(self._execution_order))
            if self._execution_order
            else ()
        )

        for node_name, node_key in zip(self._execution_order, keys):
            node = self.nodes[node_name]
            node_inputs = {
                port_name: port_values[(node_name, port_name)]
                for port_name in node.input_ports
                if (node_name, port_name) in port_values
            }
            node_outputs, state = node(node_inputs, state, key=node_key)

            for port_name, value in node_outputs.items():
                port_values[(node_name, port_name)] = value
                for wire in self._outgoing_wires.get((node_name, port_name), []):
                    if wire in self._cycle_wire_set:
                        continue
                    port_values[(wire.target_node, wire.target_port)] = value

        return port_values, state

    def _get_initial_cycle_values(
        self,
        state: State,
        cycle_init: Optional[dict[tuple[str, str], PyTree]] = None,
    ) -> dict[tuple[str, str], PyTree]:
        init_values: dict[tuple[str, str], PyTree] = {}

        if cycle_init is not None:
            init_values.update(cycle_init)

        node_states = {
            name: node.state_view(state)
            for name, node in self.nodes.items()
            if node.state_view(state) is not None
        }

        for wire in self._cycle_wires:
            target_key = (wire.target_node, wire.target_port)
            if target_key in init_values:
                continue
            source_state = node_states.get(wire.source_node, None)
            if source_state is None:
                continue
            source_node = self.nodes[wire.source_node]
            node_outputs = source_node.initial_outputs(source_state)
            if wire.source_port in node_outputs:
                init_values[target_key] = node_outputs[wire.source_port]

        missing = [
            (wire.target_node, wire.target_port)
            for wire in self._cycle_wires
            if (wire.target_node, wire.target_port) not in init_values
        ]
        if missing:
            raise ValueError(
                "Missing initial values for cycle targets: "
                + ", ".join(f"{n}.{p}" for n, p in missing)
            )

        return init_values

    def _call_with_iteration(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
        n_steps: int | None = None,
        return_state_history: bool = False,
        state_filter: PyTree[bool] = True,
        cycle_init: Optional[dict[tuple[str, str], PyTree]] = None,
    ) -> tuple[dict[str, PyTree], State] | tuple[dict[str, PyTree], State, PyTree | None]:
        if n_steps is None:
            if not inputs:
                raise ValueError("n_steps is required when inputs are empty")
            first_input = next(iter(inputs.values()))
            first_leaf = jt.leaves(first_input)[0]
            n_steps = int(first_leaf.shape[0])

        keys = jax.random.split(key, n_steps)

        init_cycle_values = self._get_initial_cycle_values(state, cycle_init)

        step_inputs_seq = jt.map(lambda x: x, inputs)

        def _step_inputs_at(i):
            return jt.map(lambda x: x[i], step_inputs_seq)

        step_inputs_seq = jax.vmap(_step_inputs_at)(jnp.arange(n_steps))

        save_history = return_state_history and state_filter is not False

        def step(carry, args):
            state, prev_cycle_values = carry
            step_inputs, step_key = args

            port_values = dict(prev_cycle_values)

            for ext_port, (node_name, node_port) in self.input_bindings.items():
                port_values[(node_name, node_port)] = step_inputs[ext_port]

            port_values, state = self._execute_step(port_values, state, key=step_key)

            new_cycle_values = {}
            for wire in self._cycle_wires:
                source_key = (wire.source_node, wire.source_port)
                target_key = (wire.target_node, wire.target_port)
                new_cycle_values[target_key] = port_values[source_key]

            outputs = {
                ext_port: port_values[(node_name, node_port)]
                for ext_port, (node_name, node_port) in self.output_bindings.items()
            }

            if save_history:
                state_view = self.state_view(state)
                state_view = eqx.filter(state_view, state_filter)
                return (state, new_cycle_values), (outputs, state_view)

            return (state, new_cycle_values), outputs

        if save_history:
            (final_state, _), (outputs_seq, state_history) = lax.scan(
                step,
                (state, init_cycle_values),
                (step_inputs_seq, keys),
            )

            # Prepend initial state to history
            init_state_view = self.state_view(state)
            init_state_view = eqx.filter(init_state_view, state_filter)
            def _prepend(x0, x):
                if x0 is None or x is None:
                    return None
                return jnp.concatenate([x0[None], x], axis=0)
            state_history = jt.map(_prepend, init_state_view, state_history)

            return outputs_seq, final_state, state_history

        (final_state, _), outputs_seq = lax.scan(
            step,
            (state, init_cycle_values),
            (step_inputs_seq, keys),
        )

        return outputs_seq, final_state

    # ========== Graph Surgery API ==========

    def add_node(self, name: str, component: Component) -> "Graph":
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists")
        return eqx.tree_at(
            lambda g: g.nodes,
            self,
            {**self.nodes, name: component},
        )

    def remove_node(self, name: str) -> "Graph":
        if name not in self.nodes:
            raise ValueError(f"Node '{name}' does not exist")

        new_nodes = {k: v for k, v in self.nodes.items() if k != name}
        new_wires = tuple(
            w for w in self.wires if w.source_node != name and w.target_node != name
        )
        new_input_bindings = {
            k: v for k, v in self.input_bindings.items() if v[0] != name
        }
        new_output_bindings = {
            k: v for k, v in self.output_bindings.items() if v[0] != name
        }

        return eqx.tree_at(
            lambda g: (g.nodes, g.wires, g.input_bindings, g.output_bindings),
            self,
            (new_nodes, new_wires, new_input_bindings, new_output_bindings),
        )

    def add_wire(self, wire: Wire) -> "Graph":
        if wire.source_node not in self.nodes:
            raise ValueError(f"Source node '{wire.source_node}' does not exist")
        if wire.target_node not in self.nodes:
            raise ValueError(f"Target node '{wire.target_node}' does not exist")

        return eqx.tree_at(
            lambda g: g.wires,
            self,
            self.wires + (wire,),
        )

    def remove_wire(self, wire: Wire) -> "Graph":
        new_wires = tuple(w for w in self.wires if w != wire)
        return eqx.tree_at(
            lambda g: g.wires,
            self,
            new_wires,
        )

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
        old_wire = Wire(source_node, source_port, target_node, target_port)
        graph = self.remove_wire(old_wire)
        graph = graph.add_node(node_name, component)
        graph = graph.add_wire(Wire(source_node, source_port, node_name, input_port))
        graph = graph.add_wire(Wire(node_name, output_port, target_node, target_port))
        return graph

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
