"""Whether a checkpoint's model values fit another resolved graph, read without constructing either model.

Owners are compared from the resolved specs. Only a learned owner whose resolved parameters differ is
traced: its construction runs under `jax.eval_shape`, so no array is computed and nothing compiles.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from operator import attrgetter
from typing import TYPE_CHECKING

from feedbax.contracts.model_attachment import is_generated_attachment_node
from feedbax.contracts.model_occurrence import graph_leaf_specs
from feedbax.contracts.model_values import component_model_facts

if TYPE_CHECKING:
    from feedbax.component_registry import ComponentRegistry
    from feedbax.contracts.graph import ComponentSpec, GraphSpec


@dataclass(frozen=True)
class ValueShape:
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class BuildRefusal:
    reason: str


type LearnedShapes = Mapping[str, ValueShape]


@dataclass(frozen=True)
class Misfit:
    """The leaf path the graphs share, or () when the misfit concerns the whole Model."""

    path: tuple[str, ...]
    reason: str


_traced: dict[tuple[str, str, int | None, str], LearnedShapes | BuildRefusal] = {}


def learned_shapes(name: str, node: ComponentSpec, components: ComponentRegistry) -> LearnedShapes | BuildRefusal:
    """The shape of each learned value of a resolved leaf, memoised for the life of the process."""
    key = (name, node.type, node.revision, json.dumps(node.params, sort_keys=True, separators=(",", ":")))
    if (known := _traced.get(key)) is None:
        known = _traced[key] = _trace(name, node, components)
    return known


def _trace(name: str, node: ComponentSpec, components: ComponentRegistry) -> LearnedShapes | BuildRefusal:
    import jax
    import jax.random as jr

    from feedbax.compiler.builders import build_component
    from feedbax.compiler.serialization import _materialize_component_param_value, _split_lowering_metadata

    meta = components.get(node.type)
    facts = component_model_facts(node, components)
    assert meta is not None and facts is not None

    def build():
        params, _ = _split_lowering_metadata(node, components)
        params = {
            field: _materialize_component_param_value(value, declarations={}, path=(field,))
            for field, value in params.items()
        }
        if "key" in meta.params.build_context_fields:
            params["key"] = jr.PRNGKey(0)
        return build_component(name, node.type, params, component_registry=components, params_resolved=True)

    try:
        built = jax.eval_shape(build)
    except Exception as error:  # Component construction is foreign code; its refusal is the run's reason.
        return BuildRefusal(str(error))
    return {
        value.symbol: ValueShape(tuple(leaf.shape), str(leaf.dtype))
        for value in facts.values
        if value.kind == "learned" and (leaf := attrgetter(value.field)(built)) is not None
    }


def checkpoint_misfit(source: GraphSpec, destination: GraphSpec, components: ComponentRegistry) -> Misfit | None:
    """The first owner of the source graph whose checkpointed values the destination graph cannot take."""
    leaves = dict(graph_leaf_specs(destination))
    for path, node in graph_leaf_specs(source):
        facts = component_model_facts(node, components)
        learned = facts is not None and any(value.kind == "learned" for value in facts.values)
        target = leaves.get(path)
        if target is None:
            if is_generated_attachment_node(path[-1], node.type) and not learned:
                continue
            return Misfit((), f"{'.'.join(path)} is in the checkpoint and not in this Model.")
        if (target.type, target.revision) != (node.type, node.revision):
            return Misfit(path, f"{node.type} in the checkpoint, {target.type} here.")
        if not learned or target.params == node.params or facts.is_fixed_configuration(target):
            continue
        after = learned_shapes(path[-1], target, components)
        if isinstance(after, BuildRefusal):
            return Misfit(path, after.reason)
        before = learned_shapes(path[-1], node, components)
        if isinstance(before, BuildRefusal):
            return Misfit(path, before.reason)
        if before != after:
            changed = sorted(
                field for field in node.params.keys() | target.params.keys()
                if node.params.get(field) != target.params.get(field)
            )
            return Misfit(path, "; ".join(
                f"{field} is {json.dumps(node.params.get(field))} in the checkpoint and "
                f"{json.dumps(target.params.get(field))} here"
                for field in changed
            ) + ".")
    return None
