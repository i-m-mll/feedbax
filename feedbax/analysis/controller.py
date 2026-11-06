"""Analysis controller adapters for executable Feedbax graphs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import equinox as eqx
from equinox.nn import State
import jax
import jax.numpy as jnp
import jax.tree as jt
from jaxtyping import Array, PRNGKeyArray, PyTree

from feedbax.contracts.graph import GraphSpec
from feedbax.models.cde import CDENetwork
from feedbax.runtime._graph_key_schedule import reset_key
from feedbax.runtime.graph import Graph, GraphTraceRequest

if TYPE_CHECKING:
    from feedbax.component_registry import ComponentRegistry

GraphInputBuilder = Callable[[PyTree, int], Mapping[str, PyTree]]
GraphOutputSelector = Callable[[Mapping[str, PyTree]], PyTree]
GraphKeyPolicy = Callable[[PRNGKeyArray, int], PRNGKeyArray]


@dataclass(frozen=True)
class _FlatTreeSpec:
    treedef: Any
    leaves: tuple[jax.ShapeDtypeStruct, ...]

    @property
    def size(self) -> int:
        return sum(leaf.size for leaf in self.leaves)


def _flatten_pytree_to_array(
    tree: PyTree, *, dtype: Any, source: str, expected: _FlatTreeSpec | None = None
) -> tuple[Array, _FlatTreeSpec]:
    path_leaves, treedef = jt.flatten_with_path(tree)
    if expected is not None and treedef != expected.treedef:
        raise ValueError(f"Graph controller {source} must keep its initial PyTree structure.")

    leaves = []
    for index, (path, leaf) in enumerate(path_leaves):
        array = jnp.asarray(leaf)
        if not jnp.issubdtype(array.dtype, jnp.floating):
            raise ValueError(
                f"Graph controller {source} leaf {jax.tree_util.keystr(path)} must be real "
                f"floating for flat analysis, got {array.dtype}. Use Graph.step for native stepping."
            )
        if expected is not None:
            template = expected.leaves[index]
            if array.shape != template.shape or array.dtype != template.dtype:
                raise ValueError(
                    f"Graph controller {source} leaf {jax.tree_util.keystr(path)} must keep "
                    f"its initial shape {template.shape} and dtype {template.dtype}, "
                    f"got shape {array.shape} and dtype {array.dtype}. "
                    "Use Graph.step for native stepping."
                )
        leaves.append(array)

    spec = expected or _FlatTreeSpec(
        treedef, tuple(jax.ShapeDtypeStruct(leaf.shape, leaf.dtype) for leaf in leaves)
    )
    if not leaves:
        return jnp.zeros((0,), dtype=dtype), spec
    return jnp.concatenate([leaf.astype(dtype).reshape(-1) for leaf in leaves]), spec


def _unflatten_array_to_pytree(flat: Array, spec: _FlatTreeSpec) -> PyTree:
    leaves = []
    offset = 0
    for leaf in spec.leaves:
        end = offset + leaf.size
        leaves.append(flat[offset:end].reshape(leaf.shape).astype(leaf.dtype))
        offset = end
    return jt.unflatten(spec.treedef, leaves)


def _cde_analysis_nodes(graph: Graph) -> tuple[CDENetwork, ...]:
    """Find native CDE owners, including those inside nested graphs."""

    def is_native_cde(node):
        return type(node) is CDENetwork

    owners: dict[object, CDENetwork] = {}
    for node in jt.leaves(graph, is_leaf=is_native_cde):
        if not is_native_cde(node):
            continue
        if owners.setdefault(node.state_index.marker, node) is not node:
            raise ValueError(
                "Graph controller cannot resolve learned h0 for distinct CDENetwork nodes "
                "sharing one state slot. Use graph.prepare() to give the nodes distinct "
                "state slots, or Graph.step to retain native shared-state stepping."
            )
    return tuple(owners.values())


def _project_graph_state(state: State, cde_nodes: tuple[CDENetwork, ...]) -> PyTree:
    if not cde_nodes:
        return state

    def coordinates(native_state):
        slots = (native_state.get(node.state_index) for node in cde_nodes)
        return tuple(leaf for slot in slots for leaf in (slot.hidden, slot._use_h0))

    resolved = tuple(leaf for node in cde_nodes for leaf in (node.state_view(state).hidden, None))
    return eqx.tree_at(coordinates, state, resolved)


def _restore_graph_state(projected: PyTree, cde_nodes: tuple[CDENetwork, ...]) -> State:
    if not cde_nodes:
        return projected

    return eqx.tree_at(
        lambda state: tuple(state.get(node.state_index)._use_h0 for node in cde_nodes),
        projected,
        tuple(jnp.asarray(False) for _ in cde_nodes),
        is_leaf=lambda leaf: leaf is None,
    )


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

    ``h = concat(flatten(graph_coordinates), flatten(cycle_port_values))``.

    Each call to :meth:`step` unpacks that state, executes the graph through the
    public graph step API, and repacks the new graph state and recurrent carry.
    The vector represents continuous coordinates only. Each coordinate and carry
    leaf must have a fixed shape and real floating dtype, which is restored before
    graph execution. When packing fresh `CDENetwork` state, the adapter resolves
    the graph's learned `h0` into the hidden coordinates. Unpacking reconstructs
    native continuation state with initialization already resolved.
    For exact stepping with discrete state, use
    :meth:`feedbax.runtime.graph.Graph.step` with the native state and carry.
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
    _cde_nodes: tuple[CDENetwork, ...] = ()

    def initial_state(self) -> Array:
        """Return the packed initial controller state."""
        return self.h0_flat

    def unpack_state(self, h: Array) -> tuple[State, dict[tuple[str, str], PyTree]]:
        """Unpack a flat controller state into graph state and recurrent carry."""
        state_size = self.state_spec.size
        expected_shape = (state_size + self.cycle_spec.size,)
        if h.shape != expected_shape:
            raise ValueError(
                f"Expected flat graph controller state shape {expected_shape}, got {h.shape}."
            )
        state_flat = h[:state_size]
        cycle_flat = h[state_size:]
        state = _restore_graph_state(
            _unflatten_array_to_pytree(state_flat, self.state_spec), self._cde_nodes
        )
        cycle_port_values = _unflatten_array_to_pytree(cycle_flat, self.cycle_spec)
        return state, cycle_port_values

    def pack_state(
        self, state: State, cycle_port_values: Mapping[tuple[str, str], PyTree]
    ) -> Array:
        """Pack graph state and recurrent carry, resolving fresh CDE hidden state."""
        state_flat, _ = _flatten_pytree_to_array(
            _project_graph_state(state, self._cde_nodes),
            dtype=self.dtype,
            source="graph state",
            expected=self.state_spec,
        )
        cycle_flat, _ = _flatten_pytree_to_array(
            cycle_port_values, dtype=self.dtype, source="recurrent carry", expected=self.cycle_spec
        )
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

    Supply trial-scope recurrent initializer inputs through ``static_inputs``.
    These inputs also provide external input prototypes when compiling a
    ``GraphSpec``. Explicit ``cycle_init`` values take precedence over inferred
    carry. Reset uses the native reset stream derived from ``key_policy(key, 0)``,
    the key of the first controller transition. ``input_builder`` supplies step
    inputs. ``trace`` exposes retained graph boundary values through
    :meth:`GraphControllerAdapter.step_with_trace`.
    ``dtype`` is the dtype of the flat coordinates. Graph execution uses the
    native leaf dtypes. Fresh `CDENetwork` state resolves learned `h0` at this
    boundary, before the first flat step. Other discrete state and discrete carry
    are unsupported in flat analysis.
    """
    if not jnp.issubdtype(dtype, jnp.floating):
        raise ValueError(
            f"Graph controller coordinates require a real floating dtype, got {dtype}."
        )

    reset_inputs = dict(static_inputs) if static_inputs is not None else {}
    if isinstance(graph, GraphSpec):
        if component_registry is None:
            raise ValueError(
                "component_registry is required when graph_controller receives a GraphSpec; "
                "pass an executable Graph to use the registry-free boundary"
            )
        from feedbax.compiler import GraphDocument, compile_graph

        executable = compile_graph(
            GraphDocument(graph=graph),
            component_registry,
            input_prototypes={("__graph__", name): value for name, value in reset_inputs.items()},
        ).graph
    else:
        executable = graph
    if input_builder is None and input_port not in executable.input_ports:
        raise ValueError(f"Graph has no external input port {input_port!r}.")
    if (
        output_selector is None
        and output_port is not None
        and output_port not in executable.output_ports
    ):
        raise ValueError(f"Graph has no external output port {output_port!r}.")

    init_state = executable.init_state(key=key)
    cde_nodes = _cde_analysis_nodes(executable)
    state_flat, state_spec = _flatten_pytree_to_array(
        _project_graph_state(init_state, cde_nodes), dtype=dtype, source="graph state"
    )
    cycle_port_values = executable.initial_cycle_port_values(
        init_state,
        dict(cycle_init) if cycle_init is not None else None,
        inputs=reset_inputs,
        key=reset_key(key_policy(key, 0)),
    )
    cycle_flat, cycle_spec = _flatten_pytree_to_array(
        cycle_port_values, dtype=dtype, source="recurrent carry"
    )
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
        _cde_nodes=cde_nodes,
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
