from __future__ import annotations

import inspect

import pytest

from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.component import (
    COMPONENT_DEFINITION_DYNAMIC_PORT_POLICY_MIGRATION_ID,
    COMPONENT_DEFINITION_SCHEMA_ID,
    COMPONENT_DEFINITION_SCHEMA_VERSION,
    COMPONENT_DEFINITION_SCHEMA_VERSION_V1,
    COMPONENT_DEFINITION_SCHEMA_VERSION_V2,
    ComponentDefinition,
    DynamicPortPolicy,
    PortType,
    derive_dynamic_port_layout,
    validate_dynamic_port_layout,
)
from feedbax.contracts.migrations import default_spec_registry


def test_builtin_mux_and_demux_policies_preserve_dynamic_port_layouts() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    mux = registry.get("Mux")
    demux = registry.get("Demux")

    assert mux is not None and mux.dynamic_port_policy is not None
    assert demux is not None and demux.dynamic_port_policy is not None
    validate_dynamic_port_layout(
        mux.dynamic_port_policy,
        mux.default_params,
        input_ports=mux.input_ports,
        output_ports=mux.output_ports,
    )
    validate_dynamic_port_layout(
        demux.dynamic_port_policy,
        demux.default_params,
        input_ports=demux.input_ports,
        output_ports=demux.output_ports,
    )
    assert derive_dynamic_port_layout(
        mux.dynamic_port_policy,
        {"n_inputs": 3},
    ).model_dump() == {
        "input_ports": ["in_0", "in_1", "in_2"],
        "output_ports": ["output"],
    }
    assert derive_dynamic_port_layout(
        demux.dynamic_port_policy,
        {"sizes": [2, 1, 3]},
    ).model_dump() == {
        "input_ports": ["input"],
        "output_ports": ["out_0", "out_1", "out_2"],
    }


def test_dynamic_arity_derivation_is_policy_driven_for_external_component() -> None:
    policy = DynamicPortPolicy(
        count_param="channels",
        count_mode="sequence_length",
        direction="input",
        fixed_output_ports=["combined"],
        generated_name_template="channel_{index}",
        generated_index_origin=1,
        dynamic_port_type=PortType(dtype="scalar"),
    )

    assert "component_type" not in inspect.signature(derive_dynamic_port_layout).parameters
    assert derive_dynamic_port_layout(policy, {"channels": ["left", "right"]}).model_dump() == {
        "input_ports": ["channel_1", "channel_2"],
        "output_ports": ["combined"],
    }


def test_dynamic_port_layout_validation_fails_closed() -> None:
    policy = DynamicPortPolicy(
        count_param="width",
        count_mode="integer",
        direction="output",
        fixed_input_ports=["input"],
        generated_name_template="result_{index}",
        dynamic_port_type=PortType(dtype="vector"),
    )

    with pytest.raises(ValueError, match="dynamic port layout mismatch"):
        validate_dynamic_port_layout(
            policy,
            {"width": 2},
            input_ports=["input"],
            output_ports=["result_0"],
        )


def test_registry_exports_external_dynamic_port_policy_in_v3_definition() -> None:
    registry = ComponentRegistry(load_user_components=False, discover_plugins=False)
    registry.register_component_type(
        "example.VariableFanIn",
        lambda params: params,
        param_schema=[{"name": "fan_in", "type": "int", "default": 2}],
        input_ports=["source_0", "source_1"],
        output_ports=["output"],
        dynamic_port_policy={
            "count_param": "fan_in",
            "count_mode": "integer",
            "direction": "input",
            "fixed_output_ports": ["output"],
            "generated_name_template": "source_{index}",
            "dynamic_port_type": {"dtype": "vector"},
        },
    )

    definition = next(item for item in registry.list_all() if item.name == "example.VariableFanIn")
    payload = definition.model_dump(mode="json")

    assert payload["schema_id"] == COMPONENT_DEFINITION_SCHEMA_ID
    assert payload["schema_version"] == COMPONENT_DEFINITION_SCHEMA_VERSION
    assert payload["dynamic_port_policy"]["count_param"] == "fan_in"
    assert ComponentDefinition.model_validate(payload) == definition


def test_component_definition_v2_migrates_to_optional_dynamic_port_policy() -> None:
    payload = {
        "schema_id": COMPONENT_DEFINITION_SCHEMA_ID,
        "schema_version": COMPONENT_DEFINITION_SCHEMA_VERSION_V2,
        "name": "StaticLegacy",
        "category": "Test",
        "description": "A v2 definition without a dynamic-port policy.",
    }

    result = default_spec_registry.migrate("ComponentDefinition", payload)
    definition = ComponentDefinition.model_validate(result.payload)

    assert [record.migration_id for record in result.migration_records] == [
        COMPONENT_DEFINITION_DYNAMIC_PORT_POLICY_MIGRATION_ID
    ]
    assert definition.schema_version == COMPONENT_DEFINITION_SCHEMA_VERSION
    assert definition.dynamic_port_policy is None


def test_component_definition_policy_declares_v1_and_v2_migrations() -> None:
    policy = default_spec_registry.resolve("ComponentDefinition").policy

    assert policy is not None
    assert policy.stance == "migrate"
    assert policy.supported_old_versions == (
        COMPONENT_DEFINITION_SCHEMA_VERSION_V1,
        COMPONENT_DEFINITION_SCHEMA_VERSION_V2,
    )
