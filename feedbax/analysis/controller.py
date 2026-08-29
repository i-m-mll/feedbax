"""Analysis controller adapters for executable Feedbax graphs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Array, PRNGKeyArray, PyTree

from feedbax.contracts.graph import GraphSpec
from feedbax.runtime.graph import Graph, GraphTraceRequest

if TYPE_CHECKING:
    from feedbax.component_registry import ComponentRegistry

GraphInputBuilder = Callable[[PyTree, int], Mapping[str, PyTree]]
GraphOutputSelector = Callable[[Mapping[str, PyTree]], PyTree]
GraphKeyPolicy = Callable[[PRNGKeyArray, int], PRNGKeyArray]


@dataclass(frozen=True)
class _FlatTreeSpec:
    treedef: Any
    leaf_shapes: tuple[tuple[int, ...], ...]
    leaf_sizes: tuple[int, ...]

    @property
    def size(self) -> int:
        return sum(self.leaf_sizes)


def _flatten_pytree_to_array(tree: PyTree, *, dtype: Any) -> tuple[Array, _FlatTreeSpec]:
    leaves = jt.leaves(tree)
    leaf_shapes = tuple(tuple(jnp.asarray(leaf).shape) for leaf in leaves)
    leaf_sizes = tuple(int(jnp.asarray(leaf).size) for leaf in leaves)
    spec = _FlatTreeSpec(jt.structure(tree), leaf_shapes, leaf_sizes)
    if not leaves:
        return jnp.zeros((0,), dtype=dtype), spec
    return (
        jnp.concatenate(
            [jnp.asarray(leaf, dtype=dtype).reshape(-1) for leaf in leaves],
            axis=0,
        ),
        spec,
    )


def _unflatten_array_to_pytree(flat: Array, spec: _FlatTreeSpec) -> PyTree:
    leaves = []
    offset = 0
    for shape, size in zip(spec.leaf_shapes, spec.leaf_sizes):
        leaves.append(flat[offset : offset + size].reshape(shape))
        offset += size
    return jt.unflatten(spec.treedef, leaves)


def _default_key_policy(key: PRNGKeyArray, t: int) -> PRNGKeyArray:
    return jax.random.fold_in(key, t)


@dataclass(frozen=True)
class GraphControllerStep:
    """Result of a traced graph-controller step."""

    state: Array
    output: PyTree
    trace: dict[str, PyTree]
    graph_outputs: dict[str, PyTree]


@dataclass(frozen=True)
class GraphControllerAdapter:
    """Expose a Feedbax graph as a deterministic analysis controller.

    The adapter packs the executable graph state and recurrent cycle carry into
    a single flat controller state:

    ``h = concat(flatten(equinox.nn.State), flatten(cycle_port_values))``.

    Each call to :meth:`step` unpacks that state, executes the graph through the
    public :meth:`feedbax.runtime.graph.Graph.step` API, and repacks the new graph state
    and recurrent carry. Cyclic graphs therefore use the same recurrent/cycle
    semantics as normal graph execution.
    """

    graph: Graph
    h0_flat: Array
    state_spec: _FlatTreeSpec
    cycle_spec: _FlatTreeSpec
    key: PRNGKeyArray
    input_port: str = "input"
    output_port: str | None = "output"
    static_inputs: Mapping[str, PyTree] | None = None
    input_builder: GraphInputBuilder | None = None
    output_selector: GraphOutputSelector | None = None
    trace: tuple[GraphTraceRequest, ...] = ()
    key_policy: GraphKeyPolicy = _default_key_policy
    dtype: Any = jnp.float64

    def initial_state(self) -> Array:
        """Return the packed initial controller state."""
        return self.h0_flat

    def unpack_state(self, h: Array) -> tuple[PyTree, dict[tuple[str, str], PyTree]]:
        """Unpack a flat controller state into graph state and recurrent carry."""
        state_flat = h[: self.state_spec.size]
        cycle_flat = h[self.state_spec.size :]
        state = _unflatten_array_to_pytree(state_flat, self.state_spec)
        cycle_port_values = _unflatten_array_to_pytree(cycle_flat, self.cycle_spec)
        return state, cycle_port_values

    def pack_state(self, state: PyTree, cycle_port_values: Mapping[tuple[str, str], PyTree]) -> Array:
        """Pack graph state and recurrent carry into a flat controller state."""
        state_flat, _ = _flatten_pytree_to_array(state, dtype=self.dtype)
        cycle_flat, _ = _flatten_pytree_to_array(cycle_port_values, dtype=self.dtype)
        return jnp.concatenate([state_flat, cycle_flat], axis=0)

    def _inputs_for_step(self, controller_input: PyTree, t: int) -> dict[str, PyTree]:
        if self.input_builder is not None:
            inputs = dict(self.input_builder(controller_input, t))
        elif isinstance(controller_input, Mapping):
            inputs = dict(controller_input)
        else:
            inputs = {self.input_port: controller_input}

        if self.static_inputs:
            return dict(self.static_inputs) | inputs
        return inputs

    def _select_output(self, outputs: Mapping[str, PyTree]) -> PyTree:
        if self.output_selector is not None:
            return self.output_selector(outputs)
        if self.output_port is None:
            return dict(outputs)
        return outputs[self.output_port]

    def step(self, h: Array, controller_input: PyTree, t: int) -> tuple[Array, PyTree]:
        """Advance the graph controller by one timestep.

        Args:
            h: Packed controller state returned by :meth:`initial_state` or the
                previous :meth:`step`.
            controller_input: Either the value for ``input_port`` or a mapping
                of graph external input names to values. ``input_builder`` may
                replace this conversion for task-specific analysis call shapes.
            t: Discrete timestep used by the deterministic key policy.

        Returns:
            ``(h_next, output)`` where ``output`` is selected from the graph
            outputs by ``output_port`` or ``output_selector``.
        """
        result = self.step_with_trace(h, controller_input, t)
        return result.state, result.output

    def step_with_trace(self, h: Array, controller_input: PyTree, t: int) -> GraphControllerStep:
        """Advance one timestep and return retained graph-boundary values."""
        state, cycle_port_values = self.unpack_state(h)
        key_t = self.key_policy(self.key, t)
        outputs, state_next, cycle_next, trace_values = self.graph.step_with_trace(
            self._inputs_for_step(controller_input, t),
            state,
            cycle_port_values,
            key=key_t,
            trace=self.trace,
        )
        h_next = self.pack_state(state_next, cycle_next)
        return GraphControllerStep(
            state=h_next,
            output=self._select_output(outputs),
            trace=trace_values,
            graph_outputs=outputs,
        )


def graph_controller(
    graph: Graph | GraphSpec,
    *,
    key: PRNGKeyArray,
    component_registry: ComponentRegistry | None = None,
    input_port: str = "input",
    output_port: str | None = "output",
    static_inputs: Mapping[str, PyTree] | None = None,
    input_builder: GraphInputBuilder | None = None,
    output_selector: GraphOutputSelector | None = None,
    trace: tuple[GraphTraceRequest, ...] = (),
    cycle_init: Mapping[tuple[str, str], PyTree] | None = None,
    key_policy: GraphKeyPolicy = _default_key_policy,
    dtype: Any = jnp.float64,
) -> GraphControllerAdapter:
    """Wrap an executable graph or ``GraphSpec`` as an analysis controller.

    ``static_inputs`` and ``input_builder`` cover task-input handling without
    baking a task schema into the adapter. ``trace`` exposes retained graph
    boundary values through :meth:`GraphControllerAdapter.step_with_trace`.
    """
    if isinstance(graph, GraphSpec):
        if component_registry is None:
            raise ValueError(
                "component_registry is required when graph_controller receives a GraphSpec; "
                "pass an executable Graph to use the registry-free boundary"
            )
        from feedbax.compiler import GraphDocument, compile_graph

        executable = compile_graph(GraphDocument(graph=graph), component_registry).graph
    else:
        executable = graph
    if input_builder is None and input_port not in executable.input_ports:
        raise ValueError(f"Graph has no external input port {input_port!r}.")
    if output_selector is None and output_port is not None and output_port not in executable.output_ports:
        raise ValueError(f"Graph has no external output port {output_port!r}.")

    init_state = executable.init_state(key=key)
    cycle_port_values = executable.initial_cycle_port_values(init_state, dict(cycle_init or {}))
    state_flat, state_spec = _flatten_pytree_to_array(init_state, dtype=dtype)
    cycle_flat, cycle_spec = _flatten_pytree_to_array(cycle_port_values, dtype=dtype)
    h0_flat = jnp.concatenate([state_flat, cycle_flat], axis=0)

    return GraphControllerAdapter(
        graph=executable,
        h0_flat=h0_flat,
        state_spec=state_spec,
        cycle_spec=cycle_spec,
        key=key,
        input_port=input_port,
        output_port=output_port,
        static_inputs=static_inputs,
        input_builder=input_builder,
        output_selector=output_selector,
        trace=trace,
        key_policy=key_policy,
        dtype=dtype,
    )


def feedbax_graph_controller(
    graph: Graph | GraphSpec,
    *,
    key: PRNGKeyArray,
    component_registry: ComponentRegistry | None = None,
    input_port: str = "input",
    output_port: str | None = "output",
    static_inputs: Mapping[str, PyTree] | None = None,
    input_builder: GraphInputBuilder | None = None,
    output_selector: GraphOutputSelector | None = None,
    trace: tuple[GraphTraceRequest, ...] = (),
    cycle_init: Mapping[tuple[str, str], PyTree] | None = None,
    key_policy: GraphKeyPolicy = _default_key_policy,
    dtype: Any = jnp.float64,
) -> GraphControllerAdapter:
    """Compatibility alias for :func:`graph_controller`."""
    return graph_controller(
        graph,
        key=key,
        component_registry=component_registry,
        input_port=input_port,
        output_port=output_port,
        static_inputs=static_inputs,
        input_builder=input_builder,
        output_selector=output_selector,
        trace=trace,
        cycle_init=cycle_init,
        key_policy=key_policy,
        dtype=dtype,
    )


__all__ = [
    "GraphControllerAdapter",
    "GraphControllerStep",
    "GraphInputBuilder",
    "GraphKeyPolicy",
    "GraphOutputSelector",
    "feedbax_graph_controller",
    "graph_controller",
]
