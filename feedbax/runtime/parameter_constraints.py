"""Parameter constraint projection for executable graphs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

import equinox as eqx
import jax.numpy as jnp

from feedbax.runtime.components import GRU, LSTM, Linear
from feedbax.contracts.graph import ParameterConstraintSpec
from feedbax.models.networks import VanillaRNN
from feedbax.runtime.graph import Component, Graph


ConstraintLike = ParameterConstraintSpec | Mapping[str, Any]


def _normalize_constraint(constraint: ConstraintLike) -> ParameterConstraintSpec:
    if isinstance(constraint, ParameterConstraintSpec):
        return constraint
    return ParameterConstraintSpec.model_validate(constraint)


def _as_constraint_tuple(
    constraints: Sequence[ConstraintLike] | None,
) -> tuple[ParameterConstraintSpec, ...]:
    return tuple(_normalize_constraint(constraint) for constraint in constraints or ())


def normalize_parameter_constraints(
    constraints: Sequence[ConstraintLike] | None,
) -> tuple[ParameterConstraintSpec, ...]:
    """Return validated parameter constraint records."""

    return _as_constraint_tuple(constraints)


def _linear_role(component: Linear, role: str) -> tuple[Any, Callable[[Any], Any]]:
    if role in {"weight", "linear.weight", "readout.weight"}:
        return component.layer.weight, lambda node: node.layer.weight
    if role in {"bias", "linear.bias", "readout.bias"}:
        if component.layer.bias is None:
            raise ValueError("Linear bias constraint requested for a bias-free component")
        return component.layer.bias, lambda node: node.layer.bias
    raise ValueError(
        f"Unsupported Linear parameter role {role!r}; expected 'weight' or 'bias'"
    )


def _recurrent_role(
    component: GRU | LSTM | VanillaRNN,
    role: str,
) -> tuple[Any, Callable[[Any], Any]]:
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


def _resolve_role(component: Component, role: str) -> tuple[Any, Callable[[Any], Any]]:
    if isinstance(component, Linear):
        return _linear_role(component, role)
    if isinstance(component, (GRU, LSTM, VanillaRNN)):
        return _recurrent_role(component, role)
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
    """Project graph parameters to satisfy structural constraints."""

    normalized = _as_constraint_tuple(
        constraints if constraints is not None else getattr(graph, "parameter_constraints", ())
    )
    constrained = graph
    for constraint in normalized:
        if constraint.node not in constrained.nodes:
            raise ValueError(f"Parameter constraint references missing node {constraint.node!r}")
        component = constrained.nodes[constraint.node]
        array, role_getter = _resolve_role(component, constraint.role)
        projected = _project_array(array, constraint.mask, constraint.value)
        constrained = eqx.tree_at(
            lambda g, node_name=constraint.node, get_role=role_getter: get_role(g.nodes[node_name]),
            constrained,
            projected,
        )
    return constrained


def project_component_parameters(model: Component) -> Component:
    """Project model parameters when it carries graph parameter constraints."""

    if isinstance(model, Graph) and getattr(model, "parameter_constraints", ()):
        return apply_parameter_constraints(model)
    return model
