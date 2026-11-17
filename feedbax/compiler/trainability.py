"""Resolve which graph parameters the optimizer may update."""

from typing import Any

import equinox as eqx
import jax.tree as jt
from equinox.nn import StateIndex

from feedbax.compiler.serialization import _component_owns_trainability
from feedbax.contracts.graph import GraphSpec
from feedbax.runtime.graph import Graph


def compile_trainability(
    graph_spec: GraphSpec, graph: Graph, registry: Any
) -> tuple[tuple[str, ...], Any]:
    """Select inexact leaves of authored components outside frozen subtrees.

    Causal interiors have their own parameter owners. Opaque runtimes are
    selected as a single component, and compiler-generated nodes stay frozen.
    """
    trainable_paths: list[str] = []

    def graph_filter(spec: GraphSpec, runtime: Graph, path: tuple[str, ...]) -> Any:
        node_filters = dict.fromkeys(runtime.nodes, False)
        subgraphs = spec.subgraphs or {}
        for node_id, node in spec.nodes.items():
            binding = runtime.component_bindings.get(node_id)
            override = binding.trainable_override if binding is not None else None
            declaration = registry.get(node.type)
            if (
                override is None
                and isinstance(node.params.get("trainable"), bool)
                and declaration is not None
                and _component_owns_trainability(declaration)
            ):
                override = node.params["trainable"]
            if override is False:
                continue
            component = runtime.nodes[node_id]
            node_path = (*path, node_id)
            interior = subgraphs.get(node_id)
            if isinstance(interior, GraphSpec):
                if not isinstance(component, Graph):
                    raise TypeError(f"Expected a runtime graph at {'.'.join(node_path)!r}")
                node_filters[node_id] = graph_filter(interior, component, node_path)
                continue
            if override is not True and not getattr(declaration, "trainable_by_default", False):
                continue
            leaves, structure = jt.flatten(
                component, is_leaf=lambda value: isinstance(value, StateIndex)
            )
            mask = [eqx.is_inexact_array(leaf) for leaf in leaves]
            if any(mask):
                node_filters[node_id] = structure.unflatten(mask)
                trainable_paths.append(".".join(node_path))

        # Equinox accepts mask prefixes, so unselected node subtrees can share one False.
        return jt.map(
            lambda value: node_filters if value is runtime.nodes else False,
            runtime,
            is_leaf=lambda value: value is runtime.nodes,
        )

    filter_spec = graph_filter(graph_spec, graph, ())
    return tuple(trainable_paths), filter_spec
