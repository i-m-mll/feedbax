from __future__ import annotations

import inspect
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from feedbax.acausal.rotational import (
    AngleSensor,
    AngularVelocitySensor,
    GearRatio,
    Inertia,
    RotationalDamper,
    RotationalGround,
    TorqueSensor,
    TorqueSource,
    TorsionalSpring,
)
from feedbax.acausal.translational import (
    ForceSensor,
    ForceSource,
    Ground,
    LinearDamper,
    LinearSpring,
    Mass,
    PositionSensor,
    PrescribedMotion,
    VelocitySensor,
)
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.acausal import AcausalGraphSpec
from feedbax.contracts.acausal_interface import (
    AcausalInterfaceError,
    derive_acausal_interface,
    normalize_acausal_graph_interfaces,
)
from feedbax.contracts.component import (
    COMPONENT_DEFINITION_PORT_KIND_MIGRATION_ID,
    COMPONENT_DEFINITION_SCHEMA_ID,
    COMPONENT_DEFINITION_SCHEMA_VERSION,
    COMPONENT_DEFINITION_SCHEMA_VERSION_V1,
    ComponentDefinition,
)
from feedbax.contracts.domain import ACAUSAL_DOMAIN_ID
from feedbax.contracts.graph import ComponentSpec, GraphSpec
from feedbax.contracts.graphs.builders import build_component
from feedbax.contracts.graphs.normalization import normalize_graph_for_studio_authoring
from feedbax.contracts.migrations import default_spec_registry
from feedbax.web.api import components as components_api


ELEMENT_TYPES = (
    Mass,
    LinearSpring,
    LinearDamper,
    Ground,
    ForceSource,
    PrescribedMotion,
    PositionSensor,
    VelocitySensor,
    ForceSensor,
    Inertia,
    TorsionalSpring,
    RotationalDamper,
    RotationalGround,
    TorqueSource,
    GearRatio,
    AngleSensor,
    AngularVelocitySensor,
    TorqueSensor,
)


def _acausal_graph(nodes: dict[str, ComponentSpec]) -> AcausalGraphSpec:
    return AcausalGraphSpec(
        physical_domain="translational",
        nodes=nodes,
        connections=[],
        solver={"solver_type": "euler", "dt": 0.001},
    )


def _constructor_param_defaults(element_type: type[Any]) -> dict[str, Any]:
    signature = inspect.signature(element_type.__init__)
    params = {}
    for name, parameter in signature.parameters.items():
        if name in {"self", "name"}:
            continue
        params[name] = None if parameter.default is inspect.Parameter.empty else parameter.default
    return params


def test_api_components_include_acausal_elements_and_boundary_adapters() -> None:
    app = FastAPI()
    app.include_router(components_api.router, prefix="/api/components")
    client = TestClient(app)

    response = client.get("/api/components")

    assert response.status_code == 200
    definitions = {
        item.name: item
        for item in [
            ComponentDefinition.model_validate(raw) for raw in response.json()["data"]["components"]
        ]
    }
    expected_names = {element_type.__name__ for element_type in ELEMENT_TYPES} | {
        "ActuationInput",
        "SensorOutput",
        "BoundaryPort",
    }
    assert expected_names <= set(definitions)

    for name in expected_names:
        definition = definitions[name]
        assert definition.domain == ACAUSAL_DOMAIN_ID
        assert definition.port_types is not None
        declared_ports = [
            *definition.port_types.inputs.values(),
            *definition.port_types.outputs.values(),
        ]
        assert any(port.kind == "conserving" for port in declared_ports)


@pytest.mark.parametrize("element_type", ELEMENT_TYPES)
def test_acausal_element_param_schemas_match_constructors(element_type: type[Any]) -> None:
    registry = ComponentRegistry(load_user_components=False)
    meta = registry.get(element_type.__name__)

    assert meta is not None
    assert meta.domain == ACAUSAL_DOMAIN_ID
    expected_defaults = _constructor_param_defaults(element_type)
    assert {schema.name: schema.default for schema in meta.param_schema} == expected_defaults
    assert meta.port_types is not None
    assert set(meta.input_ports) == set(meta.port_types.inputs)
    assert all(port.kind == "conserving" for port in meta.port_types.inputs.values())


def test_component_definition_v1_migration_defaults_legacy_ports_to_signal() -> None:
    payload = {
        "schema_id": COMPONENT_DEFINITION_SCHEMA_ID,
        "schema_version": COMPONENT_DEFINITION_SCHEMA_VERSION_V1,
        "name": "LegacyGain",
        "category": "Math",
        "description": "Legacy signal-only component.",
        "input_ports": ["input"],
        "output_ports": ["output"],
        "port_types": {
            "inputs": {"input": {"dtype": "vector"}},
            "outputs": {"output": {"dtype": "vector"}},
        },
    }

    result = default_spec_registry.migrate("ComponentDefinition", payload)
    definition = ComponentDefinition.model_validate(result.payload)

    assert result.migration_records[0].migration_id == COMPONENT_DEFINITION_PORT_KIND_MIGRATION_ID
    assert definition.schema_version == COMPONENT_DEFINITION_SCHEMA_VERSION
    assert definition.port_types is not None
    assert definition.port_types.inputs["input"].kind == "signal"
    assert definition.port_types.outputs["output"].kind == "signal"


def test_derived_acausal_interface_orders_adapters_by_order_then_port_name() -> None:
    graph = _acausal_graph(
        {
            "late": ComponentSpec(
                type="ActuationInput",
                params={"port_name": "z", "order": 2},
            ),
            "alpha": ComponentSpec(
                type="ActuationInput",
                params={"port_name": "a", "order": 1},
            ),
            "beta": ComponentSpec(
                type="SensorOutput",
                params={"port_name": "b", "order": 1},
            ),
            "early": ComponentSpec(
                type="SensorOutput",
                params={"port_name": "a", "order": 1},
            ),
        }
    )

    interface = derive_acausal_interface(graph)

    assert interface.input_ports == ("a", "z")
    assert interface.output_ports == ("a", "b")


def test_derived_acausal_interface_rejects_duplicate_adapter_names() -> None:
    graph = _acausal_graph(
        {
            "one": ComponentSpec(type="ActuationInput", params={"port_name": "u"}),
            "two": ComponentSpec(type="ActuationInput", params={"port_name": "u"}),
        }
    )

    with pytest.raises(AcausalInterfaceError, match="Duplicate ActuationInput"):
        derive_acausal_interface(graph)


def test_acausal_system_outer_ports_recomputed_from_interior_adapters() -> None:
    graph = GraphSpec(
        nodes={
            "plant": ComponentSpec(
                type="AcausalSystem",
                params={},
                input_ports=["stale_input"],
                output_ports=["stale_output"],
            )
        },
        subgraphs={
            "plant": _acausal_graph(
                {
                    "act": ComponentSpec(
                        type="ActuationInput",
                        params={"port_name": "force_cmd", "order": 0},
                    ),
                    "sense": ComponentSpec(
                        type="SensorOutput",
                        params={"port_name": "position", "order": 0},
                    ),
                }
            )
        },
    )

    normalized = normalize_graph_for_studio_authoring(graph)

    assert normalized.nodes["plant"].input_ports == ["force_cmd"]
    assert normalized.nodes["plant"].output_ports == ["position"]


def test_nested_acausal_composite_exposes_boundary_port_names() -> None:
    inner = _acausal_graph(
        {
            "right": ComponentSpec(
                type="BoundaryPort",
                params={"port_name": "right", "order": 1},
            ),
            "left": ComponentSpec(
                type="BoundaryPort",
                params={"port_name": "left", "order": 0},
            ),
        }
    )
    outer = AcausalGraphSpec(
        physical_domain="translational",
        nodes={"nested": ComponentSpec(type="AcausalSystem", input_ports=["stale"])},
        connections=[],
        solver={"solver_type": "euler", "dt": 0.001},
        subgraphs={"nested": inner},
    )

    normalized = normalize_acausal_graph_interfaces(outer)

    assert normalized.nodes["nested"].input_ports == ["left", "right"]
    assert normalized.nodes["nested"].output_ports == []


def test_non_causal_domain_component_cannot_be_built_by_causal_builder() -> None:
    registry = ComponentRegistry(load_user_components=False)

    with pytest.raises(ValueError, match=ACAUSAL_DOMAIN_ID):
        build_component("mass", "Mass", {}, component_registry=registry)
