from __future__ import annotations

from typing import Any, Mapping

from feedbax.contracts.domain import CAUSAL_DOMAIN_ID
from feedbax.runtime.graph import GraphNode


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
) -> GraphNode:
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
        if node_type == "feedbax.PenzaiAdapter" and "builder_name" in str(exc):
            raise ValueError(
                f"PenzaiAdapter node {node_name!r} requires 'builder_name' parameter"
            ) from exc
        raise
