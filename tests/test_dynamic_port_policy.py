from __future__ import annotations

import inspect
from typing import Any

import pytest
from pydantic import ValidationError

import feedbax.contracts as contracts
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.component import (
    COMPONENT_DEFINITION_DYNAMIC_PORT_POLICY_MIGRATION_ID,
    COMPONENT_DEFINITION_PORT_KIND_MIGRATION_ID,
    COMPONENT_DEFINITION_SCHEMA_ID,
    COMPONENT_DEFINITION_SCHEMA_VERSION,
    COMPONENT_DEFINITION_SCHEMA_VERSION_V1,
    COMPONENT_DEFINITION_SCHEMA_VERSION_V2,
    ComponentDefinition,
    DynamicPortPolicy,
    DynamicPortPolicyError,
    PortType,
    derive_dynamic_port_count,
    derive_dynamic_port_layout,
    validate_dynamic_port_layout,
)
from feedbax.contracts.migrations import default_spec_registry


def test_builtin_mux_and_demux_policies_preserve_dynamic_port_layouts() -> None:
    registry = ComponentRegistry(load_user_components=False)
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
    ).model_dump(mode="json") == {
        "input_ports": ["in_0", "in_1", "in_2"],
        "output_ports": ["output"],
    }
    assert derive_dynamic_port_layout(
        demux.dynamic_port_policy,
        {"sizes": [2, 1, 3]},
    ).model_dump(mode="json") == {
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
    assert derive_dynamic_port_layout(policy, {"channels": ["left", "right"]}).model_dump(
        mode="json"
    ) == {
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

    with pytest.raises(DynamicPortPolicyError, match="dynamic port layout mismatch"):
        validate_dynamic_port_layout(
            policy,
            {"width": 2},
            input_ports=["input"],
            output_ports=["result_0"],
        )


def test_registry_exports_external_dynamic_port_policy_in_v3_definition() -> None:
    registry = ComponentRegistry(load_user_components=False)
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


@pytest.mark.parametrize(
    "fixed_input_ports, message",
    [
        (("",), "must be non-empty"),
        (("input", "input"), "must be unique"),
    ],
)
def test_policy_rejects_invalid_fixed_port_names(
    fixed_input_ports: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        DynamicPortPolicy(
            count_param="count",
            count_mode="integer",
            direction="output",
            fixed_input_ports=fixed_input_ports,
            generated_name_template="out_{index}",
            dynamic_port_type=PortType(dtype="vector"),
        )


@pytest.mark.parametrize(
    "template",
    [
        "out",
        "{{index}}",
        "{{index}}_{index}",
        "out_{other}",
        "out_{index.value}",
        "out_{index[0]}",
        "out_{index!r}",
        "out_{index:03d}",
        "out_{index:}",
        "out_{index",
        "out_{}",
    ],
)
def test_policy_rejects_non_declarative_generated_name_templates(template: str) -> None:
    with pytest.raises(ValidationError, match="only unformatted"):
        DynamicPortPolicy(
            count_param="count",
            count_mode="integer",
            direction="output",
            generated_name_template=template,
            dynamic_port_type=PortType(dtype="vector"),
        )


def test_derivation_rejects_generated_fixed_name_collision() -> None:
    policy = DynamicPortPolicy(
        count_param="count",
        count_mode="integer",
        direction="input",
        fixed_input_ports=("in_0",),
        generated_name_template="in_{index}",
        dynamic_port_type=PortType(dtype="vector"),
    )

    with pytest.raises(DynamicPortPolicyError, match="collide with fixed input"):
        derive_dynamic_port_layout(policy, {"count": 1})


def test_derivation_defensively_rejects_duplicate_generated_names() -> None:
    policy = DynamicPortPolicy(
        count_param="count",
        count_mode="integer",
        direction="output",
        generated_name_template="out_{index}",
        dynamic_port_type=PortType(dtype="vector"),
    ).model_copy(update={"generated_name_template": "same"})

    with pytest.raises(DynamicPortPolicyError, match="must be unique"):
        derive_dynamic_port_layout(policy, {"count": 2})


def test_derivation_wraps_bypassed_template_formatting_errors() -> None:
    policy = DynamicPortPolicy(
        count_param="count",
        count_mode="integer",
        direction="output",
        generated_name_template="out_{index}",
        dynamic_port_type=PortType(dtype="vector"),
    ).model_copy(update={"generated_name_template": None})

    with pytest.raises(DynamicPortPolicyError, match="only unformatted"):
        derive_dynamic_port_layout(policy, {"count": 1})


@pytest.mark.parametrize(
    "params, message",
    [
        ({}, "missing dynamic-port parameter"),
        ({"count": True}, "must be an integer"),
        ({"count": 1.5}, "must be an integer"),
    ],
)
def test_integer_count_derivation_rejects_invalid_values(
    params: dict[str, Any],
    message: str,
) -> None:
    policy = DynamicPortPolicy(
        count_param="count",
        count_mode="integer",
        direction="output",
        generated_name_template="out_{index}",
        dynamic_port_type=PortType(dtype="vector"),
    )

    with pytest.raises(DynamicPortPolicyError, match=message):
        derive_dynamic_port_count(policy, params)


@pytest.mark.parametrize("value", ["abc", {"size": 1}, 3])
def test_sequence_count_derivation_rejects_non_sequences(value: Any) -> None:
    policy = DynamicPortPolicy(
        count_param="items",
        count_mode="sequence_length",
        direction="output",
        generated_name_template="out_{index}",
        dynamic_port_type=PortType(dtype="vector"),
    )

    with pytest.raises(DynamicPortPolicyError, match="must be a sequence"):
        derive_dynamic_port_count(policy, {"items": value})


def test_count_derivation_enforces_minimum() -> None:
    policy = DynamicPortPolicy(
        count_param="count",
        count_mode="integer",
        direction="output",
        generated_name_template="out_{index}",
        minimum_count=2,
        dynamic_port_type=PortType(dtype="vector"),
    )

    with pytest.raises(DynamicPortPolicyError, match="minimum is 2"):
        derive_dynamic_port_count(policy, {"count": 1})


def test_component_definition_rejects_unknown_schema_version() -> None:
    with pytest.raises(ValidationError, match="schema_version"):
        ComponentDefinition.model_validate(
            {
                "schema_id": COMPONENT_DEFINITION_SCHEMA_ID,
                "schema_version": "feedbax.spec.component_definition.v99",
                "name": "FutureDefinition",
                "category": "Test",
                "description": "Unsupported future schema.",
            }
        )


def test_component_definition_v1_migration_records_v2_then_v3() -> None:
    result = default_spec_registry.migrate(
        "ComponentDefinition",
        {
            "schema_id": COMPONENT_DEFINITION_SCHEMA_ID,
            "schema_version": COMPONENT_DEFINITION_SCHEMA_VERSION_V1,
            "name": "LegacyDefinition",
            "category": "Test",
            "description": "Legacy schema.",
        },
    )

    assert [record.migration_id for record in result.migration_records] == [
        COMPONENT_DEFINITION_PORT_KIND_MIGRATION_ID,
        COMPONENT_DEFINITION_DYNAMIC_PORT_POLICY_MIGRATION_ID,
    ]
    assert result.payload["schema_version"] == COMPONENT_DEFINITION_SCHEMA_VERSION


def test_public_contract_exports_dynamic_port_policy_helpers() -> None:
    assert contracts.DynamicPortPolicy is DynamicPortPolicy
    assert contracts.DynamicPortPolicyError is DynamicPortPolicyError
    assert contracts.derive_dynamic_port_count is derive_dynamic_port_count
    assert contracts.derive_dynamic_port_layout is derive_dynamic_port_layout
    assert contracts.validate_dynamic_port_layout is validate_dynamic_port_layout
    assert contracts.DynamicPortLayout.__name__ == "DynamicPortLayout"


def test_policy_plumbing_is_instance_scoped() -> None:
    assert (
        "dynamic_port_policy"
        in inspect.signature(ComponentRegistry.register_component_type).parameters
    )
