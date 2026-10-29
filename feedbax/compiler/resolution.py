"""Dependency-driven completion of component parameters and graph prototypes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax.numpy as jnp
import jax.tree as jt

from feedbax.component_registry import (
    ComponentRegistry,
    format_missing_interior_message,
    required_interior_domain,
)
from feedbax.component_registry.declarations import MissingPrototypeInput
from feedbax.contracts.acausal_interface import normalize_acausal_interfaces_for_graph
from feedbax.contracts.domain import CAUSAL_DOMAIN_ID
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.compiler.prototypes import (
    DerivedDimensionConflict,
    DerivedDimensionError,
    STATEFUL_PROTOTYPE_TYPES,
    array_proto_from_shape,
    explicit_proto,
    shape_from_proto,
    validate_or_add_input_shape,
    _lookup_output_prototype_fn,
)
from feedbax.runtime.graph_channel_adapters import materialize_additive_channel_adapters


class ParameterResolutionError(ValueError):
    """A parameter or prototype constraint failure at one authored node."""

    def __init__(self, authored_path: str, cause: Exception):
        super().__init__(str(cause))
        self.authored_path = authored_path
        self.cause = cause


@dataclass(frozen=True)
class GraphParameterResolution:
    graph: GraphSpec
    node_inputs: dict[tuple[str, str], Any]
    node_outputs: dict[tuple[str, str], Any]
    errors: tuple[ParameterResolutionError, ...]

    @property
    def graph_outputs(self) -> dict[str, Any]:
        return {
            port: self.node_outputs[tuple(binding)]
            for port, binding in self.graph.output_bindings.items()
            if tuple(binding) in self.node_outputs
        }


def resolve_component_node(node: ComponentSpec, registry: Any) -> ComponentSpec:
    from feedbax.compiler.serialization import _split_lowering_metadata

    resolver = getattr(registry, "resolve_component_spec", None)
    should_resolve = getattr(registry, "should_resolve_component_spec", None)
    if (
        callable(resolver)
        and callable(should_resolve)
        and should_resolve(
            node.type,
            param_schema_version=node.param_schema_version,
        )
    ):
        params, lowering_metadata = _split_lowering_metadata(node)
        resolution = resolver(
            node.type,
            params,
            param_schema_version=node.param_schema_version,
            complete_params=False,
        )
        return node.model_copy(
            update={
                "type": resolution.type_id,
                "params": {**resolution.params, **lowering_metadata},
                "param_schema_version": resolution.param_schema_version,
            }
        )
    if required_interior_domain(node.type, registry) is not None:
        return node
    if registry.get(node.type) is None:
        raise ValueError(
            f"Unregistered component type {node.type!r}. "
            f"Known component types: {registry.names()!r}"
        )
    return node


def _complete_component_node(
    node_name: str,
    node: ComponentSpec,
    registry: Any,
) -> ComponentSpec:
    from feedbax.compiler.serialization import (
        _split_lowering_metadata,
        _validate_dynamic_component_ports,
    )

    meta = registry.get(node.type)
    if meta is None:
        return node
    if meta.is_composite:
        return node.model_copy(update={"params": {}})
    params, lowering_metadata = _split_lowering_metadata(node)
    context = {
        key: value for key, value in params.items() if key in meta.params.build_context_fields
    }
    effective = meta.params.require(
        {key: value for key, value in params.items() if key not in context}
    ).model_dump(mode="python")
    params = {**effective, **context}
    inputs, outputs = _validate_dynamic_component_ports(
        node_name,
        node.type,
        params,
        node.input_ports,
        node.output_ports,
        component_registry=registry,
    )
    return node.model_copy(
        update={
            "params": {**params, **lowering_metadata},
            "input_ports": list(inputs),
            "output_ports": list(outputs),
        }
    )


def _prototype_signature(proto: Any) -> Any:
    leaves, structure = jt.flatten(proto)
    return structure, tuple(
        (getattr(leaf, "shape", None), getattr(leaf, "dtype", None)) for leaf in leaves
    )


class _Completion:
    def __init__(self, graph: GraphSpec, registry: Any, path: str, identities_resolved: bool):
        self.registry = registry
        self.path = path
        self.graph = normalize_acausal_interfaces_for_graph(graph)
        self.nodes: dict[str, ComponentSpec] = {}
        self.completed: dict[str, ComponentSpec] = {}
        self.inputs: dict[tuple[str, str], Any] = {}
        self.outputs: dict[tuple[str, str], Any] = {}
        self.published: set[str] = set()
        self.deferred: dict[str, ValueError] = {}
        self.children: dict[str, _Completion] = {}
        self.errors: dict[str, ParameterResolutionError] = {}
        for name, node in self.graph.nodes.items():
            try:
                self.nodes[name] = (
                    node if identities_resolved else resolve_component_node(node, registry)
                )
            except (KeyError, TypeError, ValueError) as exc:
                self.fail(name, exc)
                self.nodes[name] = node
        for name, subgraph in (self.graph.subgraphs or {}).items():
            if isinstance(subgraph, GraphSpec):
                self.children[name] = _Completion(
                    subgraph,
                    registry,
                    f"{path}/subgraphs/{_token(name)}",
                    identities_resolved,
                )

    def fail(self, name: str, cause: Exception) -> None:
        self.errors[name] = (
            cause
            if isinstance(cause, ParameterResolutionError)
            else ParameterResolutionError(f"{self.path}/nodes/{_token(name)}", cause)
        )

    def put_input(self, key: tuple[str, str], proto: Any) -> bool:
        if key in self.inputs:
            if _prototype_signature(self.inputs[key]) != _prototype_signature(proto):
                old_shape, new_shape = shape_from_proto(self.inputs[key]), shape_from_proto(proto)
                kind = "shape" if old_shape != new_shape else "dtype"
                raise ValueError(
                    f"Graph wiring {kind} mismatch at {key[0]}.{key[1]}: "
                    f"existing prototype {self.inputs[key]!r}, source provides {proto!r}"
                )
            return False
        self.inputs[key] = proto
        return True

    def seed(self, external: Mapping[tuple[str, str], Any]) -> bool:
        changed = False
        for key, proto in external.items():
            if key[0] != "__graph__":
                try:
                    changed = self.put_input(key, proto) or changed
                except ValueError as exc:
                    self.fail(key[0], exc)
        for port, binding in self.graph.input_bindings.items():
            proto = external.get(("__graph__", port))
            if proto is not None:
                try:
                    changed = self.put_input(tuple(binding), proto) or changed
                except ValueError as exc:
                    self.fail(binding[0], exc)
        for wire in self.graph.wires:
            initial = wire.recurrent_initializer
            proto = None
            if initial is not None:
                kind = initial.get("kind")
                if kind == "zeros":
                    proto = array_proto_from_shape(initial.get("shape"))
                elif kind == "constant" and "value" in initial:
                    proto = jnp.zeros_like(
                        jnp.asarray(initial["value"], dtype=initial.get("dtype"))
                    )
                elif kind == "graph-input":
                    proto = external.get(("__graph__", initial.get("source")))
                elif kind == "node-output":
                    proto = self.outputs.get(
                        (initial.get("source_node"), initial.get("source_port"))
                    )
            if proto is not None:
                try:
                    changed = self.put_input((wire.target_node, wire.target_port), proto) or changed
                except ValueError as exc:
                    self.fail(wire.target_node, exc)
        return changed

    def dimensions(self, name: str) -> list[tuple[Any, int]] | None:
        result = []
        node = self.nodes[name]
        for index, rule in enumerate(self.graph.derived_dimensions):
            if rule.node != name:
                continue
            if rule.source != "input_port" or rule.port not in node.input_ports:
                raise DerivedDimensionError(
                    f"Derived dimension rule {index} targets unsupported input {name}.{rule.port}",
                    rule=rule,
                )
            shape = shape_from_proto(self.inputs.get((name, rule.port)))
            if shape is None:
                return None
            axis = rule.axis if rule.axis >= 0 else len(shape) + rule.axis
            if not 0 <= axis < len(shape) or shape[axis] <= 0:
                raise DerivedDimensionError(
                    f"Derived dimension rule {index} for {name}.{rule.param} axis {rule.axis} "
                    f"cannot select a positive dimension from shape {shape!r}",
                    rule=rule,
                )
            result.append((rule, int(shape[axis])))
        return result

    def complete(self, name: str) -> bool:
        if name in self.completed:
            node = self.completed[name]
            if node.type in STATEFUL_PROTOTYPE_TYPES and (name, "input") in self.inputs:
                validate_or_add_input_shape(
                    node.params,
                    self.inputs[(name, "input")],
                    node_name=name,
                    node_type=node.type,
                )
            return False
        node = self.nodes[name]
        dimensions = self.dimensions(name)
        if dimensions is None:
            return False
        params = dict(node.params)
        for rule, derived in dimensions:
            if rule.param in params and params[rule.param] is None:
                raise DerivedDimensionConflict(
                    f"Derived dimension conflict for {name}.{rule.param}: declared null but derived {derived}; "
                    "omit the parameter when using derived_dimensions",
                    rule=rule,
                    declared=None,
                    derived=derived,
                )
            if rule.param not in params:
                params[rule.param] = derived
        prototype = None
        if node.type in STATEFUL_PROTOTYPE_TYPES:
            prototype = self.inputs.get((name, "input"))
            if prototype is None:
                prototype = explicit_proto(params, node_name=name, node_type=node.type)
            if prototype is None:
                return False
            params = validate_or_add_input_shape(
                params, prototype, node_name=name, node_type=node.type
            )
        final = _complete_component_node(
            name, node.model_copy(update={"params": params}), self.registry
        )
        for rule, derived in dimensions:
            declared = final.params.get(rule.param)
            if isinstance(declared, bool) or not isinstance(declared, int) or declared != derived:
                raise DerivedDimensionConflict(
                    f"Derived dimension conflict for {name}.{rule.param}: "
                    + (
                        f"declared {node.params[rule.param]!r}, "
                        if rule.param in node.params
                        else ""
                    )
                    + f"finalized {declared!r} "
                    f"but derived {derived} from {name}.{rule.port}; omit conflicting authored values "
                    "or correct the parameter normalizer",
                    rule=rule,
                    declared=declared,
                    derived=derived,
                )
        if prototype is not None:
            validate_or_add_input_shape(
                final.params, prototype, node_name=name, node_type=node.type
            )
        self.completed[name] = final
        return True

    def publish(self, name: str) -> bool:
        if name in self.published or name not in self.completed:
            return False
        node = self.completed[name]
        domain = required_interior_domain(node.type, self.registry)
        if domain is not None and name not in (self.graph.subgraphs or {}):
            if node.type == "Network":
                raise ValueError(
                    f"Network node {name!r} requires a subgraph; no subgraph was provided. "
                    "Open it in Studio to generate the internal architecture, then save again."
                )
            raise ValueError(
                format_missing_interior_message(
                    node_name=name,
                    node_type=node.type,
                    domain_id=domain,
                )
            )
        if node.type in STATEFUL_PROTOTYPE_TYPES and (name, "input") in self.inputs:
            validate_or_add_input_shape(
                node.params,
                self.inputs[(name, "input")],
                node_name=name,
                node_type=node.type,
            )
        if name in self.children:
            outputs = self.children[name].graph_outputs()
        elif required_interior_domain(node.type, self.registry) not in (None, CAUSAL_DOMAIN_ID):
            outputs = {port: jnp.zeros(()) for port in node.output_ports}
        else:
            fn = _lookup_output_prototype_fn(self.registry, node.type)
            if fn is None:
                if any(wire.source_node == name for wire in self.graph.wires):
                    raise ValueError(
                        f"Node {name!r} of component type {node.type!r} has no registered output prototype rule"
                    )
                self.published.add(name)
                return False
            inputs = {
                port: self.inputs[(name, port)]
                for port in node.input_ports
                if (name, port) in self.inputs
            }
            if (
                node.input_ports
                and not inputs
                and not any(wire.source_node == name for wire in self.graph.wires)
            ):
                return False
            from feedbax.compiler.serialization import _materialize_component_param_value

            params = _materialize_component_param_value(node.params, declarations={}, path=())
            try:
                outputs = fn(params, inputs)
            except MissingPrototypeInput as exc:
                self.deferred[name] = ValueError(f"{node.type} node {name!r}: {exc}")
                return False
            if not isinstance(outputs, Mapping):
                raise TypeError(f"Output prototype rule for {node.type!r} must return a mapping")
        changed = False
        for port, proto in outputs.items():
            key = (name, port)
            if key not in self.outputs:
                self.outputs[key] = proto
                changed = True
            elif _prototype_signature(self.outputs[key]) != _prototype_signature(proto):
                raise ValueError(f"Output prototype changed after completion at {name}.{port}")
        if name not in self.children or all(port in outputs for port in node.output_ports):
            self.published.add(name)
        return changed

    def advance(self, external: Mapping[tuple[str, str], Any]) -> bool:
        changed = self.seed(external)
        for name in sorted(self.nodes):
            if name in self.errors:
                continue
            try:
                changed = self.complete(name) or changed
                if name in self.children:
                    child = self.children[name]
                    nested = {
                        (node, port): self.inputs[(name, graph_port)]
                        for graph_port, (node, port) in child.graph.input_bindings.items()
                        if (name, graph_port) in self.inputs
                    }
                    changed = child.advance(nested) or changed
                changed = self.publish(name) or changed
            except (KeyError, TypeError, ValueError) as exc:
                self.fail(name, exc)
        for wire in self.graph.wires:
            proto = self.outputs.get((wire.source_node, wire.source_port))
            if proto is not None and wire.target_node not in self.errors:
                try:
                    changed = self.put_input((wire.target_node, wire.target_port), proto) or changed
                except ValueError as exc:
                    self.fail(wire.target_node, exc)
        return changed

    def graph_outputs(self) -> dict[str, Any]:
        return {
            port: self.outputs[tuple(binding)]
            for port, binding in self.graph.output_bindings.items()
            if tuple(binding) in self.outputs
        }

    def finish(self) -> GraphParameterResolution:
        subgraphs = dict(self.graph.subgraphs or {})
        errors = list(self.errors.values())
        for name, child in self.children.items():
            resolved = child.finish()
            subgraphs[name] = resolved.graph
            errors.extend(resolved.errors)
        for name, node in self.nodes.items():
            if name in self.errors or name in self.completed:
                continue
            for wire in self.graph.wires:
                if wire.target_node == name and wire.source_node in self.deferred:
                    errors.append(
                        ParameterResolutionError(
                            f"{self.path}/nodes/{_token(wire.source_node)}",
                            self.deferred[wire.source_node],
                        )
                    )
            rule = next((rule for rule in self.graph.derived_dimensions if rule.node == name), None)
            cause = (
                DerivedDimensionError(
                    f"Derived dimension for {name}.{rule.param} could not resolve a prototype "
                    f"for input port {rule.port!r}",
                    rule=rule,
                )
                if rule is not None
                else ValueError(
                    f"{node.type} node {name!r} port 'input' requires an input prototype. "
                    "Connect a completed source or provide input_shape."
                )
            )
            errors.append(ParameterResolutionError(f"{self.path}/nodes/{_token(name)}", cause))
        graph = self.graph.model_copy(
            update={"nodes": {**self.nodes, **self.completed}, "subgraphs": subgraphs or None}
        )
        return GraphParameterResolution(graph, self.inputs, self.outputs, tuple(errors))


def _token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def resolve_graph_parameters(
    graph: GraphSpec,
    input_prototypes: Mapping[tuple[str, str], Any],
    component_registry: Any = None,
    *,
    identities_resolved: bool = False,
    allow_incomplete: bool = False,
) -> GraphParameterResolution:
    """Complete each node once and infer solely from its final parameter meaning."""
    registry = component_registry or ComponentRegistry(load_user_components=False)
    graph = graph if identities_resolved else materialize_additive_channel_adapters(graph)
    state = _Completion(graph, registry, "/graph", identities_resolved)
    while state.advance(input_prototypes):
        pass
    result = state.finish()
    if result.errors and not allow_incomplete:
        raise result.errors[0] from result.errors[0].cause
    return result
