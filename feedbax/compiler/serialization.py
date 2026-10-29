from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence, cast

from feedbax.runtime.graph import Component, ComponentBinding, Graph, Wire
from feedbax.contracts.component import (
    DynamicPortPolicyError,
    derive_dynamic_port_layout,
    validate_dynamic_port_layout,
)
from feedbax.contracts.graph import (
    ComponentSpec,
    GraphSpec,
    WireSpec,
    require_causal_subgraph,
    validate_subgraph_domain,
)
from feedbax.contracts.array_values import (
    ArrayValueSpec,
    _parse_array_value_payload,
    materialize_array_value,
)
from feedbax.component_registry import format_missing_interior_message, required_interior_domain
from feedbax.runtime.parameter_constraints import (
    apply_parameter_constraints,
    normalize_parameter_constraints,
)
from feedbax.compiler.builders import (
    _unsupported_component_message,
    build_component,
)
from feedbax.compiler.domain_compilers import get_domain_compiler
from feedbax.compiler.prototypes import (
    prototypes_from_task_bindings,
)
from feedbax.component_registry import builtin_domain_registry
from feedbax.contracts.domain import CAUSAL_DOMAIN_ID


__all__ = ["graph_to_spec", "prototypes_from_task_bindings"]


def _materialize_component_param_value(
    value: Any,
    *,
    declarations: dict[tuple[str, ...], ArrayValueSpec],
    path: tuple[str, ...],
) -> Any:
    declaration = _parse_array_value_payload(value)
    if declaration is not None:
        declarations[path] = declaration
        return materialize_array_value(declaration)
    if isinstance(value, Mapping):
        return {
            key: _materialize_component_param_value(
                item,
                declarations=declarations,
                path=(*path, str(key)),
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _materialize_component_param_value(
                item,
                declarations=declarations,
                path=(*path, str(index)),
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            _materialize_component_param_value(
                item,
                declarations=declarations,
                path=(*path, str(index)),
            )
            for index, item in enumerate(value)
        )
    return value


def _materialize_graph_component_params(
    spec: GraphSpec,
) -> tuple[GraphSpec, dict[str, dict[tuple[str, ...], ArrayValueSpec]]]:
    nodes: dict[str, ComponentSpec] = {}
    declarations_by_node: dict[str, dict[tuple[str, ...], ArrayValueSpec]] = {}
    for node_id, node in spec.nodes.items():
        declarations: dict[tuple[str, ...], ArrayValueSpec] = {}
        nodes[node_id] = node.model_copy(
            update={
                "params": {
                    key: _materialize_component_param_value(
                        value,
                        declarations=declarations,
                        path=(key,),
                    )
                    for key, value in node.params.items()
                }
            }
        )
        if declarations:
            declarations_by_node[node_id] = declarations
    return spec.model_copy(update={"nodes": nodes}), declarations_by_node


def graph_to_spec(graph: Any, component_registry: Any | None = None) -> GraphSpec:
    """Serialize a runtime graph through its component declarations."""

    if not isinstance(graph, Graph):
        raise TypeError("graph_to_spec requires feedbax.runtime.graph.Graph")
    registry = component_registry
    if registry is None:
        from feedbax.component_registry import ComponentRegistry

        registry = ComponentRegistry(load_user_components=False)

    nodes: dict[str, ComponentSpec] = {}
    subgraphs: dict[str, GraphSpec] = {}
    for name, component in graph.nodes.items():
        binding = graph.component_bindings.get(name)
        if isinstance(component, Graph):
            if binding is None:
                raise ValueError(
                    f"Programmatic composite node {name!r} has no declared component binding"
                )
            meta = registry.get(binding.type_id)
            if meta is None or not meta.is_composite:
                raise ValueError(
                    f"Runtime Graph node {name!r} is not bound to a declared composite type"
                )
            if binding.param_schema_version != meta.param_schema_version:
                raise ValueError(
                    f"Composite node {name!r} has unsupported bound parameter schema version "
                    f"{binding.param_schema_version!r}"
                )
            type_id = binding.type_id
            params = {}
            param_schema_version = binding.param_schema_version
            subgraphs[name] = graph_to_spec(component, registry)
        else:
            type_id, params, param_schema_version = registry.serialize_component(
                component,
                type_id=binding.type_id if binding is not None else None,
                param_schema_version=(
                    binding.param_schema_version if binding is not None else None
                ),
            )

        layout = registry.dynamic_port_layout(type_id, params)
        if layout is not None and (
            tuple(component.input_ports) != tuple(layout.input_ports)
            or tuple(component.output_ports) != tuple(layout.output_ports)
        ):
            raise DynamicPortPolicyError(
                f"{type_id} node {name!r} runtime ports do not match the ports "
                "derived from its declared parameter contract: "
                f"runtime inputs={tuple(component.input_ports)!r}, "
                f"outputs={tuple(component.output_ports)!r}; "
                f"declared inputs={tuple(layout.input_ports)!r}, "
                f"outputs={tuple(layout.output_ports)!r}"
            )
        nodes[name] = ComponentSpec(
            type=type_id,
            params=params,
            param_schema_version=param_schema_version,
            input_ports=list(component.input_ports),
            output_ports=list(component.output_ports),
        )

    return GraphSpec(
        nodes=nodes,
        wires=[
            WireSpec(
                source_node=wire.source_node,
                source_port=wire.source_port,
                target_node=wire.target_node,
                target_port=wire.target_port,
                temporality=cast(Literal["instant", "recurrent"], wire.temporality),
                recurrent_initializer=wire.recurrent_initializer,
            )
            for wire in graph.wires
        ],
        input_ports=list(graph.input_ports),
        output_ports=list(graph.output_ports),
        input_bindings=dict(graph.input_bindings),
        output_bindings=dict(graph.output_bindings),
        subgraphs=subgraphs or None,
        retained_observables=getattr(graph, "retained_observables", None),
        parameter_constraints=list(getattr(graph, "parameter_constraints", ())),
        metadata=None,
    )


def _instantiate_graph(
    spec: GraphSpec,
    component_registry: Any,
) -> Graph:
    """Instantiate a resolved graph for the graph compiler."""
    execution_registry = component_registry
    metadata_registry = component_registry
    component_bindings = {
        node_name: ComponentBinding(
            type_id=node.type,
            param_schema_version=node.param_schema_version,
        )
        for node_name, node in spec.nodes.items()
        if node.param_schema_version is not None
    }
    spec, authored_array_values = _materialize_graph_component_params(spec)

    nodes: dict[str, Component] = {}
    for node_name, node_spec in spec.nodes.items():
        node_type = node_spec.type
        node_params, _ = _split_lowering_metadata(node_spec)
        required_domain = required_interior_domain(node_type, metadata_registry)
        if required_domain is None and metadata_registry is not execution_registry:
            required_domain = required_interior_domain(node_type, execution_registry)
        if required_domain is None:
            unsupported_message = _unsupported_component_message(
                node_name,
                node_type,
                execution_registry,
            )
            if unsupported_message is not None:
                raise NotImplementedError(unsupported_message)
        params = node_params
        declared_input_ports, declared_output_ports = _validate_dynamic_component_ports(
            node_name,
            node_type,
            params,
            node_spec.input_ports,
            node_spec.output_ports,
            component_registry=metadata_registry,
        )

        if required_domain is not None:
            subgraph = (spec.subgraphs or {}).get(node_name)
            if subgraph is None:
                if node_type == "Subgraph":
                    raise ValueError(f"Missing subgraph spec for '{node_name}'")
                if node_type == "Network":
                    raise ValueError(
                        f"Network node {node_name!r} has no subgraph. "
                        "Open it in Studio to generate the internal architecture, then save again."
                    )
                raise ValueError(
                    format_missing_interior_message(
                        node_name=node_name,
                        node_type=node_type,
                        domain_id=required_domain,
                    )
                )
            validate_subgraph_domain(
                subgraph,
                expected_domain=required_domain,
                node_name=node_name,
                node_type=node_type,
                consumer="compile_graph",
            )
            if required_domain != CAUSAL_DOMAIN_ID:
                domain = builtin_domain_registry().get(required_domain)
                if domain is None:
                    raise ValueError(
                        f"Node {node_name!r} ({node_type}) requires unknown "
                        f"interior domain {required_domain!r}"
                    )
                if domain.compiler_id is None:
                    raise ValueError(
                        f"Interior domain {required_domain!r} for node {node_name!r} "
                        "has no compiler"
                    )
                if (
                    domain.interior_schema_id is not None
                    and getattr(subgraph, "schema_id", None) != domain.interior_schema_id
                ):
                    raise ValueError(
                        f"Node {node_name!r} ({node_type}) interior schema "
                        f"{getattr(subgraph, 'schema_id', None)!r} does not match "
                        f"domain {required_domain!r} schema {domain.interior_schema_id!r}"
                    )
                compiler = get_domain_compiler(domain.compiler_id)
                nodes[node_name] = compiler(subgraph, node_name, execution_registry)
                continue
            causal_subgraph = require_causal_subgraph(
                subgraph,
                node_name=node_name,
                node_type=node_type,
                consumer="compile_graph",
            )
            nodes[node_name] = _instantiate_graph(causal_subgraph, metadata_registry)
            continue
        if spec.subgraphs and node_name in spec.subgraphs:
            causal_subgraph = require_causal_subgraph(
                spec.subgraphs[node_name],
                node_name=node_name,
                node_type=node_type,
                consumer="compile_graph",
            )
            nodes[node_name] = _instantiate_graph(causal_subgraph, metadata_registry)
            continue
        delta_A_declaration = authored_array_values.get(node_name, {}).get(("delta_A",))
        if node_type == "StructuralLinearStateSpace" and delta_A_declaration is not None:
            params = {**params, "_authored_delta_A_value_spec": delta_A_declaration}
        component = build_component(
            node_name,
            node_type,
            params,
            component_registry=execution_registry,
            params_resolved=True,
        )
        if (
            tuple(component.input_ports) != declared_input_ports
            or tuple(component.output_ports) != declared_output_ports
        ):
            meta = metadata_registry.get(node_type)
            if meta is not None and meta.dynamic_port_policy is not None:
                raise DynamicPortPolicyError(
                    f"{node_type} node {node_name!r} runtime ports do not match its "
                    "policy-materialized GraphSpec namespace: "
                    f"runtime inputs={tuple(component.input_ports)!r}, "
                    f"outputs={tuple(component.output_ports)!r}; "
                    f"spec inputs={declared_input_ports!r}, outputs={declared_output_ports!r}"
                )
        nodes[node_name] = component

    wires = tuple(
        Wire(
            wire.source_node,
            wire.source_port,
            wire.target_node,
            wire.target_port,
            wire.temporality,
            wire.recurrent_initializer,
        )
        for wire in spec.wires
    )

    input_bindings = {
        name: (binding[0], binding[1]) for name, binding in spec.input_bindings.items()
    }
    output_bindings = {
        name: (binding[0], binding[1]) for name, binding in spec.output_bindings.items()
    }

    graph = Graph(
        nodes=nodes,
        wires=wires,
        input_ports=tuple(spec.input_ports),
        output_ports=tuple(spec.output_ports),
        input_bindings=input_bindings,
        output_bindings=output_bindings,
        parameter_constraints=normalize_parameter_constraints(spec.parameter_constraints),
        component_bindings=component_bindings,
    )
    return apply_parameter_constraints(graph)


def _split_lowering_metadata(
    node_spec: ComponentSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    params = dict(node_spec.params)
    if node_spec.type != "Sum" or "channel_adapter" not in params:
        return params, {}
    lowering_metadata = {"channel_adapter": params.pop("channel_adapter")}
    return params, lowering_metadata


def _validate_dynamic_component_ports(
    node_name: str,
    node_type: str,
    params: Mapping[str, Any],
    input_ports: Sequence[str],
    output_ports: Sequence[str],
    *,
    component_registry: Any,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    meta = component_registry.get(node_type)
    if meta is None or meta.dynamic_port_policy is None:
        return tuple(input_ports), tuple(output_ports)
    layout = derive_dynamic_port_layout(meta.dynamic_port_policy, params)
    declared_inputs = tuple(input_ports) or layout.input_ports
    declared_outputs = tuple(output_ports) or layout.output_ports
    try:
        validate_dynamic_port_layout(
            meta.dynamic_port_policy,
            params,
            input_ports=declared_inputs,
            output_ports=declared_outputs,
        )
    except DynamicPortPolicyError as exc:
        value = params.get(meta.dynamic_port_policy.count_param, "<missing>")
        raise DynamicPortPolicyError(
            f"{node_type} node {node_name!r} has invalid dynamic ports for "
            f"{meta.dynamic_port_policy.count_param}={value!r}: {exc}"
        ) from exc
    return declared_inputs, declared_outputs
