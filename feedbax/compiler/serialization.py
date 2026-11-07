from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from typing import Any, Literal, Mapping, Sequence, cast

from jaxtyping import PRNGKeyArray

from feedbax.models.feedback import feedback_state_view
from feedbax.runtime._graph_key_schedule import derive_key
from feedbax.runtime.trial_history import trial_history_plan
from feedbax.runtime.graph import AuthoredArrayEncoding, Component, ComponentBinding, Graph, GraphNode, Wire
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
from feedbax.component_registry.declarations import DeclaredComponent
from feedbax.component_registry import (
    format_missing_interior_message,
    required_interior_domain,
)
from feedbax.runtime.parameter_constraints import normalize_parameter_constraints
from feedbax.compiler.builders import (
    _unsupported_component_message,
    build_component,
)
from feedbax.compiler.domain_compilers import get_domain_compiler
from feedbax.compiler.resolution import GraphParameterResolution
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


def _array_encoding_slot(
    params: dict[str, Any], path: tuple[str, ...]
) -> tuple[Any, str | int] | None:
    container: Any = params
    for part in path[:-1]:
        if isinstance(container, dict) and part in container:
            container = container[part]
        elif isinstance(container, list) and part.isdecimal() and int(part) < len(container):
            container = container[int(part)]
        else:
            return None
    part = path[-1]
    if isinstance(container, dict) and part in container:
        return container, part
    if isinstance(container, list) and part.isdecimal() and int(part) < len(container):
        return container, int(part)
    return None


def _serialized_value_digest(value: Any) -> bytes:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).digest()


def _capture_authored_arrays(
    params: dict[str, Any], declarations: Mapping[tuple[str, ...], ArrayValueSpec]
) -> tuple[AuthoredArrayEncoding, ...]:
    encodings = []
    for path, declaration in sorted(declarations.items()):
        if (slot := _array_encoding_slot(params, path)) is not None:
            container, key = slot
            encodings.append(
                AuthoredArrayEncoding(path, declaration, _serialized_value_digest(container[key]))
            )
    return tuple(encodings)


def _restore_authored_arrays(
    params: dict[str, Any], encodings: tuple[AuthoredArrayEncoding, ...]
) -> None:
    for encoding in encodings:
        if (slot := _array_encoding_slot(params, encoding.path)) is not None:
            container, key = slot
            if _serialized_value_digest(container[key]) == encoding.baseline_sha256:
                container[key] = encoding.declaration.model_dump(mode="json")


def graph_to_spec(graph: Any, component_registry: Any | None = None) -> GraphSpec:
    """Serialize a runtime graph through its component declarations."""

    if not isinstance(graph, Graph):
        raise TypeError("graph_to_spec requires feedbax.runtime.graph.Graph")
    registry = component_registry
    if registry is None:
        from feedbax.component_registry import ComponentRegistry

        registry = ComponentRegistry()

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
            if binding.revision != meta.revision:
                raise ValueError(
                    f"Composite node {name!r} has unsupported bound revision "
                    f"{binding.revision!r}"
                )
            type_id = binding.type_id
            params = {}
            revision = binding.revision
            subgraphs[name] = graph_to_spec(component, registry)
        else:
            if not isinstance(component, Component):
                raise ValueError(
                    f"Graph node {name!r} ({type(component).__name__}) is Python-only: "
                    "portable export requires a Component with a reconstructible configuration"
                )
            type_id, params, revision = registry.serialize_component(
                component,
                type_id=binding.type_id if binding is not None else None,
                revision=(
                    binding.revision if binding is not None else None
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
        if binding is not None:
            _restore_authored_arrays(params, binding.authored_arrays)
            if binding.trainable_override is not None:
                params = {**params, "trainable": binding.trainable_override}
        nodes[name] = ComponentSpec(
            type=type_id,
            params=params,
            revision=revision,
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


def _subgraph_leaf_keys(
    node_name: str, keys: dict[tuple[str, ...], PRNGKeyArray]
) -> dict[tuple[str, ...], PRNGKeyArray]:
    if () in keys:
        raise ValueError(f"Initialization key for composite node {node_name!r} must name a leaf")
    return keys


def _instantiate_graph(
    spec: GraphSpec,
    component_registry: Any,
    resolution: GraphParameterResolution,
    *,
    construction_key: PRNGKeyArray,
    initialization_keys: Mapping[tuple[str, ...], PRNGKeyArray] | None = None,
) -> Graph:
    """Instantiate a resolved graph for the graph compiler."""
    keys_by_node: dict[str, dict[tuple[str, ...], PRNGKeyArray]] = {}
    for (node_name, *path), key in (initialization_keys or {}).items():
        keys_by_node.setdefault(node_name, {})[tuple(path)] = key
    execution_registry = component_registry
    metadata_registry = component_registry
    spec, authored_array_values = _materialize_graph_component_params(spec)

    nodes: dict[str, GraphNode] = {}
    component_bindings: dict[str, ComponentBinding] = {}
    for node_name, node_spec in spec.nodes.items():
        node_keys = keys_by_node.pop(node_name, {})
        node_type = node_spec.type
        node_params, lowering_metadata = _split_lowering_metadata(node_spec, metadata_registry)
        if node_spec.revision is not None:
            component_bindings[node_name] = ComponentBinding(
                type_id=node_type,
                revision=node_spec.revision,
                trainable_override=lowering_metadata.get("trainable"),
            )
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
                if node_type == "feedbax.Subgraph":
                    raise ValueError(f"Missing subgraph spec for '{node_name}'")
                if node_type == "feedbax.Network":
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
                if node_keys:
                    raise ValueError(
                        f"Cannot supply initialization keys to acausal node {node_name!r}"
                    )
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
            nodes[node_name] = _instantiate_graph(
                causal_subgraph,
                metadata_registry,
                resolution.subgraphs[node_name],
                construction_key=derive_key(construction_key, "construction", node_name),
                initialization_keys=_subgraph_leaf_keys(node_name, node_keys),
            )
            continue
        if spec.subgraphs and node_name in spec.subgraphs:
            causal_subgraph = require_causal_subgraph(
                spec.subgraphs[node_name],
                node_name=node_name,
                node_type=node_type,
                consumer="compile_graph",
            )
            nodes[node_name] = _instantiate_graph(
                causal_subgraph,
                metadata_registry,
                resolution.subgraphs[node_name],
                construction_key=derive_key(construction_key, "construction", node_name),
                initialization_keys=_subgraph_leaf_keys(node_name, node_keys),
            )
            continue
        meta = metadata_registry.get(node_type)
        if meta is not None and meta.compiler.build_output_prototypes:
            params = {
                **params,
                **{
                    parameter: resolution.node_outputs[(node_name, port)]
                    for parameter, port in meta.compiler.build_output_prototypes.items()
                },
            }
        if node_keys:
            if (
                node_keys.keys() != {()}
                or meta is None
                or "key" not in meta.params.build_context_fields
            ):
                raise ValueError(
                    f"Initialization keys at {node_name!r} must address a leaf accepting a key"
                )
            params = {**params, "key": node_keys[()]}
        elif meta is not None and "key" in meta.params.build_context_fields:
            params = {**params, "key": derive_key(construction_key, "construction", node_name)}
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
            if meta is not None and meta.dynamic_port_policy is not None:
                raise DynamicPortPolicyError(
                    f"{node_type} node {node_name!r} runtime ports do not match its "
                    "policy-materialized GraphSpec namespace: "
                    f"runtime inputs={tuple(component.input_ports)!r}, "
                    f"outputs={tuple(component.output_ports)!r}; "
                    f"spec inputs={declared_input_ports!r}, outputs={declared_output_ports!r}"
                )
        if declarations := authored_array_values.get(node_name):
            binding = component_bindings[node_name]
            assert meta is not None
            component_bindings[node_name] = replace(
                binding,
                authored_arrays=_capture_authored_arrays(meta.params.dump(component), declarations),
            )
        nodes[node_name] = component

    if keys_by_node:
        raise ValueError(f"Initialization keys name unknown nodes: {sorted(keys_by_node)}")

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

    return Graph(
        nodes=nodes,
        wires=wires,
        input_ports=tuple(spec.input_ports),
        output_ports=tuple(spec.output_ports),
        input_bindings=input_bindings,
        output_bindings=output_bindings,
        parameter_constraints=normalize_parameter_constraints(spec.parameter_constraints),
        component_bindings=component_bindings,
        state_consistency_fn=trial_history_plan(nodes, wires),
        state_view_fn=feedback_state_view(
            nodes, wires, input_bindings=input_bindings, output_bindings=output_bindings
        ),
    )


def _split_lowering_metadata(
    node_spec: ComponentSpec,
    component_registry: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate graph-level controls from the component's own parameters."""
    params = dict(node_spec.params)
    lowering_metadata = {}
    if node_spec.type in ("Sum", "feedbax.Sum") and "channel_adapter" in params:
        lowering_metadata["channel_adapter"] = params.pop("channel_adapter")
    if isinstance(params.get("trainable"), bool):
        declaration = component_registry.get(node_spec.type)
        if declaration is not None and not _component_owns_trainability(declaration):
            lowering_metadata["trainable"] = params.pop("trainable")
    return params, lowering_metadata


def _component_owns_trainability(declaration: DeclaredComponent) -> bool:
    contract = declaration.params
    return (
        "trainable" in contract.model.model_fields or "trainable" in contract.build_context_fields
    )


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
