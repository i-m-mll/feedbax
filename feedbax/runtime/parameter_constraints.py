"""Parameter constraint projection for executable graphs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

import equinox as eqx
import jax.numpy as jnp

from feedbax.runtime.components import GRU, LSTM, Linear
from feedbax.contracts.graph import ParameterConstraintSpec
from feedbax.runtime.graph import Component, Graph


ConstraintLike = ParameterConstraintSpec | Mapping[str, Any]

#: Resolve ``(array, getter)`` for one constrained parameter role of a component.
ParameterRoleResolver = Callable[[Component, str], tuple[Any, Callable[[Any], Any]]]

_PARAMETER_ROLE_RESOLVERS: dict[type, ParameterRoleResolver] = {}


def register_parameter_role_resolver(
    component_type: type,
    resolver: ParameterRoleResolver,
) -> None:
    """Declare how a component type resolves constrained parameter roles.

    Component types declare their own constrainable parameters next to their
    definition, so the runtime never needs to name a concrete component class.
    Resolution walks the component MRO, so a subclass inherits its base type's
    resolver unless it registers its own.
    """

    _PARAMETER_ROLE_RESOLVERS[component_type] = resolver


def _normalize_constraint(constraint: ConstraintLike) -> ParameterConstraintSpec:
    if isinstance(constraint, ParameterConstraintSpec):
        return constraint
    return ParameterConstraintSpec.model_validate(constraint)


def normalize_parameter_constraints(
    constraints: Sequence[ConstraintLike] | None,
) -> tuple[ParameterConstraintSpec, ...]:
    """Return validated parameter constraint records."""

    return tuple(_normalize_constraint(constraint) for constraint in constraints or ())


def linear_parameter_role(component: Component, role: str) -> tuple[Any, Callable[[Any], Any]]:
    """Resolve a constrained parameter role on a ``Linear``-shaped component."""

    if role in {"weight", "linear.weight", "readout.weight"}:
        return component.layer.weight, lambda node: node.layer.weight
    if role in {"bias", "linear.bias", "readout.bias"}:
        if component.layer.bias is None:
            raise ValueError("Linear bias constraint requested for a bias-free component")
        return component.layer.bias, lambda node: node.layer.bias
    raise ValueError(
        f"Unsupported Linear parameter role {role!r}; expected 'weight' or 'bias'"
    )


def recurrent_parameter_role(component: Component, role: str) -> tuple[Any, Callable[[Any], Any]]:
    """Resolve a constrained parameter role on a cell-carrying recurrent component."""

    if role in {"input_kernel", "weight_ih"}:
        return component.cell.weight_ih, lambda node: node.cell.weight_ih
    if role in {"hidden_kernel", "weight_hh"}:
        return component.cell.weight_hh, lambda node: node.cell.weight_hh
    if role == "bias":
        bias = getattr(component.cell, "bias", None)
        if bias is None:
            raise ValueError(f"{type(component).__name__} bias constraint requested without bias")
        return bias, lambda node: node.cell.bias
    expected = "'input_kernel', 'hidden_kernel', or 'bias'"
    raise ValueError(f"Unsupported {type(component).__name__} parameter role {role!r}; expected {expected}")


register_parameter_role_resolver(Linear, linear_parameter_role)
register_parameter_role_resolver(GRU, recurrent_parameter_role)
register_parameter_role_resolver(LSTM, recurrent_parameter_role)


def _resolve_role(component: Component, role: str) -> tuple[Any, Callable[[Any], Any]]:
    for component_type in type(component).__mro__:
        resolver = _PARAMETER_ROLE_RESOLVERS.get(component_type)
        if resolver is not None:
            return resolver(component, role)
    raise ValueError(
        f"Parameter constraints are not supported for {type(component).__name__} components"
    )


def _project_array(array, mask, value):
    mask_array = jnp.asarray(mask, dtype=bool)
    value_array = jnp.asarray(value, dtype=array.dtype)
    if mask_array.shape not in {(), array.shape}:
        raise ValueError(
            f"Parameter mask shape {mask_array.shape} is incompatible with target shape {array.shape}"
        )
    if value_array.shape not in {(), array.shape}:
        raise ValueError(
            f"Parameter constraint value shape {value_array.shape} is incompatible with "
            f"target shape {array.shape}"
        )
    return jnp.where(mask_array, array, value_array)


def apply_parameter_constraints(
    graph: Graph,
    constraints: Sequence[ConstraintLike] | None = None,
) -> Graph:
    """Project a graph and its nested graphs using their local node names.

    An explicit ``constraints`` sequence overrides declarations on this graph only.
    """

    replacements: dict[str, Component] = {}
    for name, node in graph.nodes.items():
        if isinstance(node, Graph):
            projected = apply_parameter_constraints(node)
            if projected is not node:
                replacements[name] = projected

    normalized = normalize_parameter_constraints(
        constraints if constraints is not None else getattr(graph, "parameter_constraints", ())
    )
    for constraint in normalized:
        name = constraint.node
        if name not in graph.nodes:
            raise ValueError(f"Parameter constraint references missing node {name!r}")
        component = replacements.get(name, graph.nodes[name])
        array, role_getter = _resolve_role(component, constraint.role)
        value = _project_array(array, constraint.mask, constraint.value)
        replacements[name] = eqx.tree_at(role_getter, component, value)

    if replacements:
        return eqx.tree_at(lambda g: g.nodes, graph, graph.nodes | replacements)
    return graph


def project_component_parameters(model: Component) -> Component:
    """Project parameters of a graph component, including its nested graphs."""

    if isinstance(model, Graph):
        return apply_parameter_constraints(model)
    return model
