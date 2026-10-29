from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp
import jax.tree as jt

from feedbax.contracts.acausal_interface import normalize_acausal_interfaces_for_graph
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


def _lookup_default_params(component_registry: Any, name: str) -> dict[str, Any]:
    meta = _lookup_registry_meta(component_registry, name)
    if meta is None:
        return {}
    if isinstance(meta, Mapping):
        return dict(meta.get("default_params", {}))
    if hasattr(meta, "default_params"):
        return dict(getattr(meta, "default_params"))
    return {}


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
    params = _lookup_default_params(component_registry, node_spec.type)
    params.update(node_spec.params)
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
        normalized = normalize_stateful_prototypes(
            subgraph,
            nested_inputs,
            component_registry=component_registry,
        )
        return bound_output_prototypes(
            normalized,
            subgraphs=dict(normalized.subgraphs or {}),
            input_prototypes=nested_inputs,
            component_registry=component_registry,
        )

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


def bound_output_prototypes(
    spec: GraphSpec,
    *,
    subgraphs: Mapping[str, GraphSubgraphSpec],
    input_prototypes: Mapping[tuple[str, str], Any],
    component_registry: Any = None,
) -> dict[str, Any]:
    node_inputs = infer_node_input_prototypes(
        spec,
        input_prototypes,
        subgraphs,
        component_registry=component_registry,
    )
    outputs: dict[str, Any] = {}
    for graph_port, (node_name, node_port) in spec.output_bindings.items():
        node_spec = spec.nodes.get(node_name)
        if node_spec is None:
            raise ValueError(
                f"Graph output binding {graph_port!r} references missing node {node_name!r}"
            )
        node_outputs = output_prototypes_for_node(
            node_name,
            node_spec,
            node_inputs,
            subgraphs,
            component_registry=component_registry,
        )
        if node_port not in node_outputs:
            raise ValueError(
                f"Graph output binding {graph_port!r} references missing output port "
                f"{node_name}.{node_port}"
            )
        outputs[graph_port] = node_outputs[node_port]
    return outputs


def infer_node_input_prototypes(
    spec: GraphSpec,
    external_input_prototypes: Mapping[tuple[str, str], Any],
    subgraphs: Mapping[str, GraphSubgraphSpec],
    component_registry: Any = None,
) -> dict[tuple[str, str], Any]:
    if component_registry is None:
        component_registry = ComponentRegistry(load_user_components=False)
    input_prototypes: dict[tuple[str, str], Any] = dict(external_input_prototypes)
    for graph_port, (node_name, node_port) in spec.input_bindings.items():
        graph_proto = external_input_prototypes.get(("__graph__", graph_port))
        if graph_proto is not None:
            input_prototypes[(node_name, node_port)] = graph_proto

    def initializer_prototype(wire: Any) -> Any | None:
        initializer = wire.recurrent_initializer
        if initializer is None:
            return None
        if initializer.get("kind") == "zeros":
            return array_proto_from_shape(initializer.get("shape"))
        if initializer.get("kind") == "constant" and "value" in initializer:
            return proto_from_value(initializer["value"])
        if initializer.get("kind") == "graph-input":
            source = initializer.get("source")
            if isinstance(source, str):
                return external_input_prototypes.get(("__graph__", source))
            return None
        if initializer.get("kind") == "node-output":
            source_node = initializer.get("source_node")
            source_port = initializer.get("source_port")
            if not isinstance(source_node, str) or not isinstance(source_port, str):
                return None
            source_spec = spec.nodes.get(source_node)
            if source_spec is None:
                return None
            outputs = output_prototypes_for_node(
                source_node,
                source_spec,
                input_prototypes,
                subgraphs,
                component_registry=component_registry,
                strict=False,
            )
            return outputs.get(source_port)
        return None

    for wire in spec.wires:
        init_proto = initializer_prototype(wire)
        if init_proto is not None:
            input_prototypes[(wire.target_node, wire.target_port)] = init_proto

    for _ in range(max(1, len(spec.nodes) + len(spec.wires) + 1)):
        changed = False
        for wire in spec.wires:
            init_proto = initializer_prototype(wire)
            key = (wire.target_node, wire.target_port)
            if init_proto is None or key in input_prototypes:
                continue
            input_prototypes[key] = init_proto
            changed = True
        for wire in spec.wires:
            source_spec = spec.nodes.get(wire.source_node)
            if source_spec is None:
                continue
            outputs = output_prototypes_for_node(
                wire.source_node,
                source_spec,
                input_prototypes,
                subgraphs,
                component_registry=component_registry,
                strict=False,
            )
            proto = outputs.get(wire.source_port)
            if proto is None:
                continue
            key = (wire.target_node, wire.target_port)
            current_shape = (
                shape_from_proto(input_prototypes[key]) if key in input_prototypes else None
            )
            next_shape = shape_from_proto(proto)
            if key not in input_prototypes:
                input_prototypes[key] = proto
                changed = True
            elif (
                current_shape is not None and next_shape is not None and current_shape != next_shape
            ):
                raise ValueError(
                    f"Graph wiring shape mismatch at {wire.target_node}.{wire.target_port}: "
                    f"existing prototype shape {current_shape!r}, "
                    f"{wire.source_node}.{wire.source_port} provides {next_shape!r}"
                )
            if key in input_prototypes:
                current_dtype = getattr(input_prototypes[key], "dtype", None)
                next_dtype = getattr(proto, "dtype", None)
                if (
                    current_dtype is not None
                    and next_dtype is not None
                    and current_dtype != next_dtype
                ):
                    raise ValueError(
                        f"Graph wiring dtype mismatch at {wire.target_node}.{wire.target_port}: "
                        f"existing prototype dtype {current_dtype}, "
                        f"{wire.source_node}.{wire.source_port} provides {next_dtype}"
                    )
        if not changed:
            break

    for wire in spec.wires:
        if (wire.target_node, wire.target_port) in input_prototypes:
            continue
        target_spec = spec.nodes.get(wire.target_node)
        if target_spec is None or target_spec.type not in STATEFUL_PROTOTYPE_TYPES:
            continue
        source_spec = spec.nodes.get(wire.source_node)
        if source_spec is None:
            continue
        outputs = output_prototypes_for_node(
            wire.source_node,
            source_spec,
            input_prototypes,
            subgraphs,
            component_registry=component_registry,
            strict=True,
        )
        if wire.source_port not in outputs:
            raise ValueError(
                f"Wire {wire.source_node}.{wire.source_port} -> "
                f"{wire.target_node}.{wire.target_port} references a source port "
                "with no inferable output prototype"
            )

    return input_prototypes


def normalize_stateful_prototypes(
    spec: GraphSpec,
    input_prototypes: Mapping[tuple[str, str], Any] | None = None,
    component_registry: Any = None,
) -> GraphSpec:
    spec = normalize_acausal_interfaces_for_graph(spec)
    subgraphs = dict(spec.subgraphs or {})
    node_inputs = infer_node_input_prototypes(
        spec,
        input_prototypes or {},
        subgraphs,
        component_registry=component_registry,
    )
    nodes: dict[str, ComponentSpec] = {}
    normalized_subgraphs: dict[str, GraphSubgraphSpec] = {}

    for node_name, node_spec in spec.nodes.items():
        params = dict(node_spec.params)
        if node_spec.type in STATEFUL_PROTOTYPE_TYPES:
            params = validate_or_add_input_shape(
                params,
                node_inputs.get((node_name, "input")),
                node_name=node_name,
                node_type=node_spec.type,
            )
        nodes[node_name] = node_spec.model_copy(update={"params": params})

        if node_name in subgraphs:
            raw_subgraph = subgraphs[node_name]
            if not isinstance(raw_subgraph, GraphSpec):
                normalized_subgraphs[node_name] = raw_subgraph
                continue
            subgraph = require_causal_subgraph(
                raw_subgraph,
                node_name=node_name,
                node_type=node_spec.type,
                consumer="normalize_stateful_prototypes",
            )
            nested_inputs = {
                (node, port): node_inputs[(node_name, graph_port)]
                for graph_port, (node, port) in subgraph.input_bindings.items()
                if (node_name, graph_port) in node_inputs
            }
            normalized_subgraphs[node_name] = normalize_stateful_prototypes(
                subgraph,
                nested_inputs,
                component_registry=component_registry,
            )

    return spec.model_copy(
        update={
            "nodes": nodes,
            "subgraphs": normalized_subgraphs or None,
        }
    )


def normalize_derived_dimensions(
    spec: GraphSpec,
    input_prototypes: Mapping[tuple[str, str], Any] | None = None,
    component_registry: Any = None,
) -> GraphSpec:
    """Materialize declared derived builder dimensions into node parameters.

    Rules read from the same prototype propagation path used by stateful shape
    normalization. Missing parameters are filled from resolved graph structure;
    explicit mismatches raise instead of being overwritten.
    """

    subgraphs = dict(spec.subgraphs or {})
    node_inputs = infer_node_input_prototypes(
        spec,
        input_prototypes or {},
        subgraphs,
        component_registry=component_registry,
    )
    nodes = dict(spec.nodes)

    for index, rule in enumerate(spec.derived_dimensions):
        node = nodes.get(rule.node)
        if node is None:
            raise DerivedDimensionError(
                f"Derived dimension rule {index} targets missing node {rule.node!r}",
                rule=rule,
            )
        if rule.source != "input_port":
            raise DerivedDimensionError(
                f"Derived dimension rule {index} uses unsupported source {rule.source!r}",
                rule=rule,
            )
        if rule.port not in node.input_ports:
            raise DerivedDimensionError(
                f"Derived dimension rule {index} targets missing input port "
                f"{rule.node}.{rule.port}",
                rule=rule,
            )

        shape = shape_from_proto(node_inputs.get((rule.node, rule.port)))
        if shape is None:
            raise DerivedDimensionError(
                f"Derived dimension rule {index} for {rule.node}.{rule.param} could not "
                f"resolve a serializable prototype for input port {rule.port!r}",
                rule=rule,
            )
        axis = rule.axis if rule.axis >= 0 else len(shape) + rule.axis
        if axis < 0 or axis >= len(shape):
            raise DerivedDimensionError(
                f"Derived dimension rule {index} for {rule.node}.{rule.param} axis "
                f"{rule.axis} is out of range for resolved shape {shape!r}",
                rule=rule,
            )
        derived = int(shape[axis])
        if derived <= 0:
            raise DerivedDimensionError(
                f"Derived dimension rule {index} for {rule.node}.{rule.param} resolved "
                f"non-positive dimension {derived}",
                rule=rule,
                derived=derived,
            )

        params = dict(node.params)
        if rule.param in params:
            declared = params[rule.param]
            if declared is None:
                raise DerivedDimensionConflict(
                    f"Derived dimension conflict for {rule.node}.{rule.param}: declared "
                    f"null but derived {derived} from {rule.node}.{rule.port} shape {shape!r}; "
                    "omit the parameter when using derived_dimensions",
                    rule=rule,
                    declared=declared,
                    derived=derived,
                )
            if not isinstance(declared, bool) and isinstance(declared, int) and declared == derived:
                continue
            raise DerivedDimensionConflict(
                f"Derived dimension conflict for {rule.node}.{rule.param}: declared "
                f"{declared!r} but derived {derived} from {rule.node}.{rule.port} "
                f"shape {shape!r}",
                rule=rule,
                declared=declared,
                derived=derived,
            )
        params[rule.param] = derived
        nodes[rule.node] = node.model_copy(update={"params": params})

    normalized_subgraphs: dict[str, GraphSubgraphSpec] = {}
    for node_name, subgraph in subgraphs.items():
        if not isinstance(subgraph, GraphSpec):
            normalized_subgraphs[node_name] = subgraph
            continue
        causal_subgraph = require_causal_subgraph(
            subgraph,
            node_name=node_name,
            node_type=nodes[node_name].type if node_name in nodes else "<unknown>",
            consumer="normalize_derived_dimensions",
        )
        nested_inputs = {
            (node, port): node_inputs[(node_name, graph_port)]
            for graph_port, (node, port) in causal_subgraph.input_bindings.items()
            if (node_name, graph_port) in node_inputs
        }
        normalized_subgraphs[node_name] = normalize_derived_dimensions(
            causal_subgraph,
            nested_inputs,
            component_registry=component_registry,
        )

    return spec.model_copy(
        update={
            "nodes": nodes,
            "subgraphs": normalized_subgraphs or None,
        }
    )
