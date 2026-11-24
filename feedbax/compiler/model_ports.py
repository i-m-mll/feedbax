"""Project model port declarations from the component registry."""

from __future__ import annotations

from functools import cache

from feedbax.component_registry import ComponentRegistry
from feedbax.component_registry.declarations import DeclaredComponent
from feedbax.compiler.graph_schema import _component_port_dimension
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.component import derive_dynamic_port_count, derive_dynamic_port_layout
from feedbax.contracts.model_document import ModelDocument, SignalType
from feedbax.contracts.strict_json import strict_json_loads
from feedbax.compiler.model_edit import Direction, Port, Record, model_graph_problems


class ModelRegistryPorts:
    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
        self._layout = cache(self._derive_layout)

    def _leaf(self, component: Record) -> DeclaredComponent | str:
        declaration = self.registry.get(component["type"])
        if declaration is None or declaration.is_composite:
            return f"unknown leaf type {component['type']!r}"
        return declaration

    def registration_problem(self, component: Record, _domain: str | None) -> str | None:
        if component["kind"] == "leaf" and isinstance(declaration := self._leaf(component), str):
            return declaration
        return None

    def placement_problem(self, component: Record, domain: str | None) -> str | None:
        unavailable = "Component is unavailable in this interior"
        if component["kind"] == "composite":
            nested = component["interior"]["physical_domain"]
            return None if domain is None or domain == nested else unavailable
        declaration = self._leaf(component)
        if isinstance(declaration, str):
            return declaration
        if component.get("revision") not in (None, declaration.revision):
            return unavailable
        if domain is None:
            return None if declaration.domain == "feedbax.domain.causal" else unavailable
        allowed = declaration.domain == "feedbax.domain.acausal" or (
            domain in {"planar_multibody", "spatial_multibody"}
            and declaration.domain == "feedbax.domain.mechanics"
        )
        return None if allowed else unavailable

    def port(self, component: Record, direction: Direction, port: str) -> Port | None:
        return self.layout(component).get((direction, port))

    def problems(self, document: ModelDocument):
        return model_graph_problems(document, ports=self.port, component_problem=self.placement_problem)

    def layout(self, component: Record) -> dict[tuple[Direction, str], Port]:
        return self._layout(
            component["type"], canonical_json_v2_bytes(component["fields"]),
            canonical_json_v2_bytes(component["port_counts"]),
        )

    def next_input(self, component: Record) -> tuple[str, str, int] | None:
        if component["kind"] != "leaf":
            return None
        declaration = self.registry.get(component["type"])
        policy = declaration.dynamic_port_policy if declaration else None
        if policy is None or policy.count_param is None or policy.auto_resize_minimum is None:
            return None
        name = policy.count_param
        fields = dict(component["fields"])
        if name in component["port_counts"]:
            fields[name] = component["port_counts"][name]
        params = declaration.params.require(fields).model_dump(mode="python")
        count = derive_dynamic_port_count(policy, params) + 1
        try:
            params = declaration.params.require(params | {name: count}).model_dump(mode="python")
        except ValueError:
            return None
        layout = derive_dynamic_port_layout(policy, params)
        return layout.input_ports[-1], name, count

    def _derive_layout(
        self, type_id: str, fields: bytes, port_counts: bytes,
    ) -> dict[tuple[Direction, str], Port]:
        declaration = self.registry.get(type_id)
        if declaration is None:
            return {}
        policy = declaration.dynamic_port_policy
        params = strict_json_loads(fields)
        counts = strict_json_loads(port_counts)
        for period in ("dt", "control_dt"):
            if period in declaration.params.model.model_fields:
                params.setdefault(period, declaration.params.model.model_fields[period].default)
        if policy and policy.count_param in counts:
            params[policy.count_param] = counts[policy.count_param]
        dynamic = self.registry.dynamic_port_layout(type_id, params)
        types = declaration.port_types
        inputs = dynamic.input_ports if dynamic is not None else declaration.input_ports
        outputs = dynamic.output_ports if dynamic is not None else declaration.output_ports
        result: dict[tuple[Direction, str], Port] = {}
        for direction, ports, type_map in (
            ("input", inputs, types.inputs if types else {}),
            ("output", outputs, types.outputs if types else {}),
        ):
            for port in ports:
                spec = type_map.get(port)
                if spec and spec.kind == "conserving":
                    result["physical", port] = Port(spec.physical_domain)
                    continue
                dimension = _component_port_dimension(
                    component_type=type_id, node_params=params, port=port, direction=direction,
                )
                shape = dimension["shape"] if dimension is not None else spec.shape if spec else None
                result[direction, port] = Port(
                    SignalType(
                        base=spec.dtype if spec else None,
                        shape=tuple(shape) if shape is not None else None,
                    ),
                    reads_state_only=direction == "output" and port in declaration.compiler.state_only_outputs,
                )
        return result
