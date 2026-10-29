from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp
import jax.tree as jt

from feedbax.contracts.graph import (
    ComponentSpec,
    DerivedDimensionRuleSpec,
    GraphSpec,
    GraphSubgraphSpec,
    require_causal_subgraph,
    validate_subgraph_domain,
)
from feedbax.component_registry import (
    ComponentRegistry,
    format_missing_interior_message,
    required_interior_domain,
)
from feedbax.component_registry.declarations import MissingPrototypeInput
from feedbax.contracts.domain import CAUSAL_DOMAIN_ID
from feedbax.component_registry.prototypes import array_prototype, prototype_params


STATEFUL_PROTOTYPE_TYPES = {"Channel", "DelayLine", "FirstOrderFilter"}


class DerivedDimensionError(ValueError):
    """Raised when a GraphSpec derived-dimension rule cannot be resolved."""

    def __init__(
        self,
        message: str,
        *,
        rule: DerivedDimensionRuleSpec,
        declared: Any = None,
        derived: int | None = None,
    ) -> None:
        super().__init__(message)
        self.rule = rule
        self.declared = declared
        self.derived = derived


class DerivedDimensionConflict(DerivedDimensionError):
    """Raised when a declared builder dimension contradicts the derived value."""


def array_proto_from_shape(shape: Any) -> jax.Array | None:
    if not isinstance(shape, (list, tuple)):
        return None
    try:
        return jnp.zeros(tuple(int(dim) for dim in shape))
    except (TypeError, ValueError):
        return None


def shape_from_proto(proto: Any) -> list[int] | None:
    if not hasattr(proto, "shape"):
        return None
    return [int(dim) for dim in proto.shape]


def proto_from_value(value: Any) -> Any:
    return jt.map(lambda x: jnp.zeros_like(jnp.asarray(x)), value)


def _task_sample_shape(shape: Any) -> list[int] | None:
    if not isinstance(shape, (list, tuple)):
        return None
    dims = list(shape)
    if dims and dims[0] == "time":
        dims = dims[1:]
    if not dims:
        return []
    sample: list[int] = []
    for dim in dims:
        if isinstance(dim, int):
            sample.append(dim)
        elif isinstance(dim, float) and dim.is_integer():
            sample.append(int(dim))
        else:
            return None
    return sample


def prototypes_from_task_bindings(task_binding_spec: Any) -> dict[tuple[str, str], Any]:
    """Return node-port input prototypes derivable from Studio task bindings."""

    exposed = getattr(task_binding_spec, "exposed_data", None)
    bindings = getattr(task_binding_spec, "bindings", None)
    if exposed is None and isinstance(task_binding_spec, Mapping):
        exposed = task_binding_spec.get("exposed_data", [])
    if bindings is None and isinstance(task_binding_spec, Mapping):
        bindings = task_binding_spec.get("bindings", [])

    data_by_id: dict[str, Any] = {}
    for item in exposed or []:
        item_id = getattr(item, "id", None) if not isinstance(item, Mapping) else item.get("id")
        if item_id is not None:
            data_by_id[str(item_id)] = item

    prototypes: dict[tuple[str, str], Any] = {}
    for binding in bindings or []:
        source_id = (
            getattr(binding, "source_data_id", None)
            if not isinstance(binding, Mapping)
            else binding.get("source_data_id")
        )
        target_node = (
            getattr(binding, "target_node_id", None)
            if not isinstance(binding, Mapping)
            else binding.get("target_node_id")
        )
        target_port = (
            getattr(binding, "target_port", None)
            if not isinstance(binding, Mapping)
            else binding.get("target_port")
        )
        item = data_by_id.get(str(source_id))
        if item is None or target_node is None or target_port is None:
            continue
        role = _field(binding, "role") or _field(item, "role")
        if role == "component_parameter" and _task_data_temporality(item) == "constant":
            continue
        expected_shape = (
            getattr(item, "expected_shape", None)
            if not isinstance(item, Mapping)
            else item.get("expected_shape")
        )
        sample_shape = _task_sample_shape(expected_shape)
        if sample_shape is not None:
            prototypes[(str(target_node), str(target_port))] = jnp.zeros(tuple(sample_shape))
    return prototypes


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _metadata(value: Any) -> dict[str, Any]:
    metadata = _field(value, "metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _task_data_temporality(data: Any) -> str:
    metadata = _metadata(data)
    explicit = metadata.get("temporality") or metadata.get("temporal_support")
    if isinstance(explicit, str):
        if explicit in {"trajectory", "materialized_trajectory", "epoch_masked_signal"}:
            return "time_varying"
        if explicit in {"constant", "trial_constant", "initial", "per_trial"}:
            return "constant"
    shape = _field(data, "expected_shape")
    value_spec = _field(data, "value_spec")
    if shape is None:
        shape = _field(value_spec, "shape")
    if isinstance(shape, (list, tuple)) and shape and shape[0] == "time":
        return "time_varying"
    return "constant"


def explicit_proto(
    params: Mapping[str, Any],
    *,
    node_name: str,
    node_type: str,
) -> Any | None:
    try:
        return array_prototype(params)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{node_type} node {node_name!r}: {exc}") from exc


def validate_or_add_input_shape(
    params: Mapping[str, Any],
    inferred_proto: Any | None,
    *,
    node_name: str,
    node_type: str,
) -> dict[str, Any]:
    next_params = dict(params)
    explicit = explicit_proto(next_params, node_name=node_name, node_type=node_type)
    proto = inferred_proto if inferred_proto is not None else explicit
    if proto is None:
        raise ValueError(
            f"{node_type} node {node_name!r} port 'input' requires an input prototype. "
            "Connect it to a source with a known output shape or provide input_shape."
        )
    try:
        inferred = prototype_params(proto)
    except ValueError as exc:
        raise ValueError(f"{node_type} node {node_name!r}: {exc}") from exc
    for name in ("input_shape", "input_dtype"):
        declared = next_params.get(name)
        if declared is None:
            continue
        declared = list(declared) if name == "input_shape" else str(jnp.dtype(declared))
        if declared != inferred[name]:
            raise ValueError(
                f"{node_type} node {node_name!r} {name} {declared!r} does not match "
                f"inferred input prototype {name} {inferred[name]!r} for port 'input'"
            )
    next_params.update(inferred)
    return next_params


def _defer_or_raise(reason: str, *, strict: bool) -> dict[str, Any]:
    if strict:
        raise ValueError(reason)
    return {}


def _lookup_registry_meta(component_registry: Any, name: str) -> Any | None:
    if component_registry is None:
        return None
    if isinstance(component_registry, Mapping):
        meta = component_registry.get(name)
    elif hasattr(component_registry, "get"):
        meta = component_registry.get(name)
    elif isinstance(component_registry, (list, tuple)):
        meta = next(
            (item for item in component_registry if getattr(item, "name", None) == name), None
        )
    else:
        meta = None
    return meta


def _lookup_output_prototype_fn(component_registry: Any, name: str) -> Any | None:
    meta = _lookup_registry_meta(component_registry, name)
    if meta is None:
        return None
    if isinstance(meta, Mapping):
        return meta.get("output_prototype_fn")
    return getattr(meta, "output_prototype_fn", None)


def _registered_output_prototypes(
    node_name: str,
    node_spec: ComponentSpec,
    input_prototypes: Mapping[tuple[str, str], Any],
    component_registry: Any,
    *,
    strict: bool,
) -> dict[str, Any] | None:
    output_prototype_fn = _lookup_output_prototype_fn(component_registry, node_spec.type)
    if output_prototype_fn is None:
        return None
    node_inputs = {
        port: input_prototypes[(node_name, port)]
        for port in node_spec.input_ports
        if (node_name, port) in input_prototypes
    }
    params = node_spec.params
    try:
        outputs = output_prototype_fn(params, node_inputs)
    except MissingPrototypeInput as exc:
        return _defer_or_raise(f"{node_spec.type} node {node_name!r}: {exc}", strict=strict)
    if not isinstance(outputs, Mapping):
        raise TypeError(
            f"Output prototype function for {node_spec.type!r} node {node_name!r} "
            "must return a mapping of output port names to prototypes"
        )
    return dict(outputs)


def output_prototypes_for_node(
    node_name: str,
    node_spec: ComponentSpec,
    input_prototypes: Mapping[tuple[str, str], Any],
    subgraphs: Mapping[str, GraphSubgraphSpec],
    component_registry: Any = None,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    if component_registry is None:
        component_registry = ComponentRegistry(load_user_components=False)
    node_type = node_spec.type

    required_domain = required_interior_domain(node_type, component_registry)
    if required_domain is not None or node_name in subgraphs:
        subgraph = subgraphs.get(node_name)
        if subgraph is None:
            raise ValueError(
                format_missing_interior_message(
                    node_name=node_name,
                    node_type=node_type,
                    domain_id=required_domain or "",
                )
            )
        validate_subgraph_domain(
            subgraph,
            expected_domain=required_domain,
            node_name=node_name,
            node_type=node_type,
            consumer="output_prototypes_for_node",
        )
        if required_domain is not None and required_domain != CAUSAL_DOMAIN_ID:
            return {port: jnp.zeros(()) for port in node_spec.output_ports}
        subgraph = require_causal_subgraph(
            subgraph,
            node_name=node_name,
            node_type=node_type,
            consumer="output_prototypes_for_node",
        )
        nested_inputs = {
            (node, port): input_prototypes[(node_name, graph_port)]
            for graph_port, (node, port) in subgraph.input_bindings.items()
            if (node_name, graph_port) in input_prototypes
        }
        from feedbax.compiler.resolution import resolve_graph_parameters

        return resolve_graph_parameters(
            subgraph,
            nested_inputs,
            component_registry,
        ).graph_outputs

    registered_outputs = _registered_output_prototypes(
        node_name,
        node_spec,
        input_prototypes,
        component_registry,
        strict=strict,
    )
    if registered_outputs is not None:
        return registered_outputs

    raise ValueError(
        f"Node {node_name!r} of component type {node_type!r} has no registered "
        "output prototype rule; register ComponentCompilerFacet.output_prototype_fn"
    )


def infer_node_input_prototypes(
    spec: GraphSpec,
    external_input_prototypes: Mapping[tuple[str, str], Any],
    subgraphs: Mapping[str, GraphSubgraphSpec],
    component_registry: Any = None,
) -> dict[tuple[str, str], Any]:
    """Infer inputs through the same parameter completion authority as compilation."""
    from feedbax.compiler.resolution import resolve_graph_parameters

    return resolve_graph_parameters(
        spec.model_copy(update={"subgraphs": dict(subgraphs) or None}),
        external_input_prototypes,
        component_registry or ComponentRegistry(load_user_components=False),
    ).node_inputs
