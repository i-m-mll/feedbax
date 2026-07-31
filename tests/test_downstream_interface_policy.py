"""Focused contract tests for downstream policy admission and named imports."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
from pathlib import Path

import pytest

from feedbax.plugins import (
    DOWNSTREAM_INTERFACE_POLICY_ID,
    DOWNSTREAM_POLICY_EFFECTIVE_RELEASE,
    DOWNSTREAM_PROTOCOL_CURRENT,
    DOWNSTREAM_PROTOCOL_MINIMUM,
    PLUGIN_DECLARATION_SCHEMA_VERSION_V1,
    BootstrapError,
    BootstrapErrorCode,
    PluginDeclaration,
    PluginRegistration,
    UnsupportedDownstreamProtocolVersion,
    bootstrap_application,
    new_registration_context,
    validate_downstream_protocol_version,
)


ROOT = Path(__file__).resolve().parents[1]

GUARANTEED_IMPORTS = {
    "feedbax.plugins": (
        "DOWNSTREAM_INTERFACE_POLICY_ID",
        "DOWNSTREAM_PROTOCOL_CURRENT",
        "DOWNSTREAM_PROTOCOL_MINIMUM",
        "DOWNSTREAM_POLICY_EFFECTIVE_RELEASE",
        "UnsupportedDownstreamProtocolVersion",
        "validate_downstream_protocol_version",
        "BootstrapError",
        "BootstrapErrorCode",
        "BootstrapState",
        "FamilyRequirement",
        "PluginDeclaration",
        "PluginDependency",
        "PluginProvenance",
        "PluginRegistration",
        "RegistrationContext",
        "RegistryKey",
        "bootstrap_application",
        "discover_plugin_registrations",
        "new_registration_context",
    ),
    "feedbax.lowering": (
        "LowererRegistration",
        "LoweredContribution",
        "LowererExecutionError",
        "OrderedLowererRegistry",
    ),
    "feedbax.component_registry": (
        "ComponentBuilder",
        "ComponentMeta",
        "ComponentResolution",
        "ComponentRegistry",
        "ComponentMigration",
        "ComponentMigrationPack",
    ),
    "feedbax.contracts.migrations": (
        "SchemaMigration",
        "SpecSchemaFamily",
        "SpecFamilyMigrationPolicy",
        "SpecMigrationResult",
        "SpecSchemaRegistry",
        "UnknownSpecFamily",
        "UnsupportedMigrationPath",
        "UnsupportedSpecVersion",
        "MissingComponentOwner",
        "UnsupportedComponentMigration",
        "default_spec_registry",
        "migrate_structured_spec_payload",
        "migrate_graph_spec",
    ),
    "feedbax.contracts.graph": (
        "GRAPH_SPEC_SCHEMA_ID",
        "GRAPH_SPEC_SCHEMA_VERSION",
        "GRAPH_SPEC_SCHEMA_VERSION_V2",
        "GRAPH_SPEC_SCHEMA_VERSION_V3",
        "GRAPH_SPEC_SCHEMA_VERSION_V4",
        "LEGACY_GRAPH_SPEC_SCHEMA_VERSION",
        "ComponentSpec",
        "GraphProject",
        "GraphSpec",
        "ParamSchema",
        "ParamValue",
        "StudioValueSpec",
        "WireSpec",
    ),
    "feedbax.contracts.value_identity": (
        "VALUE_IDENTITY_SCHEMA_ID",
        "VALUE_IDENTITY_SCHEMA_VERSION",
        "ValueIdentityRecord",
        "authored_value_sha256",
        "semantic_value_sha256",
        "realization_value_sha256",
        "value_identity_record",
    ),
    "feedbax.contracts.material_dependencies": (
        "ADMISSION_WAIVER_SCHEMA_ID",
        "ADMISSION_WAIVER_SCHEMA_VERSION",
        "MATERIAL_DEPENDENCIES_SCHEMA_ID",
        "MATERIAL_DEPENDENCIES_SCHEMA_VERSION",
        "AdmissionWaiver",
        "IncidentalAdmissionFailure",
        "MaterialDependency",
        "MaterialDependencyAdmission",
        "MaterialDependencyObservation",
        "MaterialDependencySet",
        "MaterialDependencyValue",
        "dependency_value_sha256",
        "material_dependency_identity_sha256",
        "validate_material_dependency_admission",
    ),
    "feedbax.contracts.manifest": (
        "TRAINING_RUN_CERTIFICATION_SCHEMA_ID",
        "TRAINING_RUN_CERTIFICATION_SCHEMA_VERSION",
        "TRAINING_RUN_CERTIFICATION_MIGRATION_TABLE",
        "TrainingRunCertification",
        "training_run_certification",
    ),
    "feedbax.contracts.array_values": (
        "ARRAY_VALUE_SCHEMA_ID",
        "ARRAY_VALUE_SCHEMA_VERSION",
        "ArrayValueSpec",
        "ConstantArrayValueSpec",
        "SparseCooArrayValueSpec",
        "SparseCooEntrySpec",
        "materialize_array_value",
    ),
    "feedbax.contracts.component": (
        "COMPONENT_DEFINITION_SCHEMA_ID",
        "COMPONENT_DEFINITION_SCHEMA_VERSION",
        "COMPONENT_DEFINITION_SCHEMA_VERSION_V1",
        "COMPONENT_DEFINITION_SCHEMA_VERSION_V2",
        "ComponentDefinition",
        "DynamicPortLayout",
        "DynamicPortPolicy",
        "DynamicPortPolicyError",
        "derive_dynamic_port_count",
        "derive_dynamic_port_layout",
        "validate_dynamic_port_layout",
        "migrate_component_definition_payload",
        "migrate_component_definition_v1_to_v2_payload",
        "migrate_component_definition_v2_to_v3_payload",
    ),
}

GUARANTEED_ROOT_EXPORTS = {
    "feedbax": GUARANTEED_IMPORTS["feedbax.lowering"],
    "feedbax.contracts": (
        *GUARANTEED_IMPORTS["feedbax.contracts.value_identity"],
        *GUARANTEED_IMPORTS["feedbax.contracts.material_dependencies"],
        *GUARANTEED_IMPORTS["feedbax.contracts.manifest"],
        *GUARANTEED_IMPORTS["feedbax.contracts.array_values"],
    ),
}


def _bootstrap_protocol(version: int) -> None:
    registration = PluginRegistration(
        PluginDeclaration("tests.downstream_policy", "1", version),
        lambda _context: None,
    )
    asyncio.run(
        bootstrap_application(
            new_registration_context(local_component_source=None),
            registrations=(registration,),
        )
    )


def test_guaranteed_imports_resolve_from_named_public_namespaces() -> None:
    for module_name, symbols in {**GUARANTEED_IMPORTS, **GUARANTEED_ROOT_EXPORTS}.items():
        module = importlib.import_module(module_name)
        missing = [symbol for symbol in symbols if not hasattr(module, symbol)]
        assert not missing, f"{module_name} is missing guaranteed symbols {missing!r}"


def test_policy_constants_bind_effective_release_and_numeric_window() -> None:
    assert DOWNSTREAM_INTERFACE_POLICY_ID == "feedbax.downstream-interface-stability.v1"
    assert DOWNSTREAM_POLICY_EFFECTIVE_RELEASE == "0.2.0"
    assert DOWNSTREAM_PROTOCOL_MINIMUM == 1
    assert DOWNSTREAM_PROTOCOL_CURRENT == 1


def test_current_downstream_protocol_is_admitted_by_unified_bootstrap() -> None:
    assert validate_downstream_protocol_version(DOWNSTREAM_PROTOCOL_CURRENT) == 1
    _bootstrap_protocol(DOWNSTREAM_PROTOCOL_CURRENT)


def test_minimum_downstream_protocol_is_admitted_by_unified_bootstrap() -> None:
    assert validate_downstream_protocol_version(DOWNSTREAM_PROTOCOL_MINIMUM) == 1
    _bootstrap_protocol(DOWNSTREAM_PROTOCOL_MINIMUM)


@pytest.mark.parametrize("version", [0, 2, True, "1", None])
def test_removed_future_or_nonnumeric_downstream_protocol_rejects(version: object) -> None:
    with pytest.raises(UnsupportedDownstreamProtocolVersion) as caught:
        validate_downstream_protocol_version(version)
    assert caught.value.code is BootstrapErrorCode.UNSUPPORTED_PROTOCOL
    assert caught.value.requested_version == version


def test_future_downstream_protocol_rejects_at_unified_bootstrap() -> None:
    with pytest.raises(UnsupportedDownstreamProtocolVersion) as caught:
        _bootstrap_protocol(DOWNSTREAM_PROTOCOL_CURRENT + 1)
    assert caught.value.plugin_id == "tests.downstream_policy"


def test_string_only_plugin_declaration_v1_rejects_without_fallback() -> None:
    registration = PluginRegistration(
        PluginDeclaration(
            "tests.legacy_declaration",
            "1",
            DOWNSTREAM_PROTOCOL_CURRENT,
            schema_version=PLUGIN_DECLARATION_SCHEMA_VERSION_V1,
        ),
        lambda _context: None,
    )
    with pytest.raises(BootstrapError) as caught:
        asyncio.run(
            bootstrap_application(
                new_registration_context(local_component_source=None),
                registrations=(registration,),
            )
        )
    assert caught.value.code is BootstrapErrorCode.UNSUPPORTED_PROTOCOL


def test_policy_checker_accepts_current_repository_contract() -> None:
    script = ROOT / "scripts" / "check_downstream_interface_policy.py"
    spec = importlib.util.spec_from_file_location("check_downstream_interface_policy", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_policy()


def test_policy_does_not_promote_runtime_namespace() -> None:
    document = (ROOT / "docs" / "design" / "downstream_interface_stability.md").read_text(
        encoding="utf-8"
    )
    guarantee_block = document.split("<!-- policy-guarantees:start -->", 1)[1].split(
        "<!-- policy-guarantees:end -->", 1
    )[0]
    assert "feedbax.runtime." not in guarantee_block
