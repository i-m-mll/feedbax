from __future__ import annotations

from typing import Any, Mapping

from feedbax.contracts.domain import CAUSAL_DOMAIN_ID
from feedbax.mechanics.analytical_component import (
    _build_analytical_musculoskeletal_plant as _build_analytical_musculoskeletal_plant,
)
from feedbax.mechanics.direct_force_components import (
    _build_point_mass as _build_point_mass,
)
from feedbax.mechanics.direct_force_components import (
    _build_two_link_arm as _build_two_link_arm,
)
from feedbax.runtime.channel import Channel
from feedbax.runtime.graph import Component
from feedbax.runtime.noise import Multiplicative, Normal
from feedbax.tasks.reach_components import (
    _build_delayed_reaches as _build_delayed_reaches,
)
from feedbax.tasks.reach_components import (
    _build_simple_reaches as _build_simple_reaches,
)
from feedbax.tasks.reach_components import (
    _build_stabilization as _build_stabilization,
)
from feedbax.tasks.reach_components import (
    _project_delayed_reaches_params as _project_delayed_reaches_params,
)


def _noise_params(params: Mapping[str, Any]) -> dict[str, Any]:
    noncanonical = sorted(
        set(params)
        & {
            "noise",
            "std",
            "additive_std",
            "signal_dependent_std",
        }
    )
    if noncanonical:
        raise ValueError(
            "Use canonical Channel noise parameters noise_model, noise_std, "
            "additive_noise_std, and signal_dependent_noise_std; unsupported: "
            + ", ".join(noncanonical)
        )
    model = str(params.get("noise_model", "additive_gaussian"))
    noise_std = params.get("noise_std", 0.0)
    additive_std = params.get("additive_noise_std", noise_std)
    if model == "additive_gaussian" and noise_std not in (None, 0, 0.0):
        additive_std = noise_std
    signal_dependent_std = params.get("signal_dependent_noise_std", 0.0)
    add_noise_default = model != "none"
    add_noise = bool(params.get("add_noise", add_noise_default))
    return {
        "model": model,
        "add_noise": add_noise,
        "additive_std": float(additive_std or 0.0),
        "include_zero_terms": bool(params.get("include_zero_noise_terms", False)),
        "signal_dependent_std": float(signal_dependent_std or 0.0),
        "role": params.get("noise_role"),
        "timing": params.get("noise_timing"),
    }


def _build_noise_func(noise: Mapping[str, Any]):
    model = str(noise["model"])
    additive_std = float(noise["additive_std"])
    signal_dependent_std = float(noise["signal_dependent_std"])
    supported = {
        "none",
        "additive_gaussian",
        "signal_dependent_gaussian",
        "signal_dependent_plus_additive",
    }
    if model not in supported:
        raise ValueError(
            f"Unsupported Channel noise_model {model!r}; expected one of "
            "'none', 'additive_gaussian', 'signal_dependent_gaussian', or "
            "'signal_dependent_plus_additive'"
        )
    if model == "none":
        return None
    include_zero = noise["include_zero_terms"]
    if model == "additive_gaussian":
        return Normal(std=additive_std) if additive_std or include_zero else None
    if model == "signal_dependent_gaussian":
        return (
            Multiplicative(Normal(std=signal_dependent_std))
            if signal_dependent_std or include_zero
            else None
        )
    if model == "signal_dependent_plus_additive":
        signal = (
            Multiplicative(Normal(std=signal_dependent_std))
            if signal_dependent_std or include_zero
            else None
        )
        additive = Normal(std=additive_std) if additive_std or include_zero else None
        if signal is not None and additive is not None:
            return signal + additive
        return signal if signal is not None else additive
    raise AssertionError(f"Unhandled Channel noise_model {model!r}")


def _build_channel(params: Mapping[str, Any]) -> Channel:
    delay = int(params.get("delay", 0))
    noise = _noise_params(params)
    noise_func = _build_noise_func(noise)
    from feedbax.component_registry.prototypes import array_prototype

    input_proto = params.get("resolved_input_proto")
    if input_proto is None:
        input_proto = array_prototype(params)
    return Channel(
        delay=delay,
        noise_func=noise_func,
        add_noise=bool(noise["add_noise"]),
        input_proto=input_proto,
        init_value=float(params.get("init_value", 0.0)),
        noise_model=str(noise["model"]),
        noise_role=None if noise["role"] is None else str(noise["role"]),
        noise_timing=None if noise["timing"] is None else str(noise["timing"]),
    )


def _template_builder_error(meta: Any, component_registry: Any) -> str | None:
    if getattr(meta, "template_graph", None) is None:
        return None
    template_builder_issues = getattr(component_registry, "template_builder_issues", None)
    if not callable(template_builder_issues):
        return None
    issues = template_builder_issues(meta)
    if not issues:
        return None
    details = "; ".join(issue.summary for issue in issues[:6])
    if len(issues) > 6:
        details += f"; and {len(issues) - 6} more"
    message = (
        f"Component template {meta.name!r} is not executable because its template graph "
        f"contains node types without registered builders: {details}"
    )
    return message


def _unsupported_component_message(
    node_name: str,
    node_type: str,
    component_registry: Any,
) -> str | None:
    meta = component_registry.get(node_type)
    if meta is None:
        return None
    unsupported_message = (
        None
        if meta.builder is None
        else getattr(meta.builder, "_feedbax_unsupported_builder_message", None)
    )
    if meta.builder is not None and unsupported_message is None:
        return None
    message = (
        _template_builder_error(meta, component_registry)
        or unsupported_message
        or (
            f"Component type {node_type!r} for node {node_name!r} is registered "
            "for metadata but has no executable builder."
        )
    )
    return message.format(node_name=node_name)


def build_component(
    node_name: str,
    node_type: str,
    params: Mapping[str, Any],
    *,
    component_registry: Any,
    params_resolved: bool = False,
) -> Component:
    """Build a leaf component from a GraphSpec node."""
    meta = component_registry.get(node_type)
    if meta is None:
        known = ", ".join(component_registry.names())
        raise ValueError(
            f"Unsupported component type {node_type!r} for node {node_name!r}. "
            f"Known component types: {known}"
        )
    if meta.domain != CAUSAL_DOMAIN_ID:
        raise ValueError(
            f"Component type {node_type!r} for node {node_name!r} belongs to "
            f"domain {meta.domain!r} and cannot be built by the causal component builder."
        )
    unsupported_message = _unsupported_component_message(
        node_name,
        node_type,
        component_registry,
    )
    if unsupported_message is not None:
        raise NotImplementedError(unsupported_message)
    try:
        return meta.params.build_resolved(params) if params_resolved else meta.builder(params)
    except ValueError as exc:
        if node_type == "PenzaiAdapter" and "builder_name" in str(exc):
            raise ValueError(
                f"PenzaiAdapter node {node_name!r} requires 'builder_name' parameter"
            ) from exc
        raise
