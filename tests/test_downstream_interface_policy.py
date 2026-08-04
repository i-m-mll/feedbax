"""Focused contract tests for downstream policy admission and named imports."""

from __future__ import annotations

import asyncio
import copy
import importlib
import importlib.util
import json
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
POLICY_MANIFEST = (
    ROOT
    / "external"
    / "feedbax_conformance_fixture"
    / "src"
    / "feedbax_external_conformance"
    / "policy_manifest.v1.json"
)
_POLICY_PAYLOAD = json.loads(POLICY_MANIFEST.read_text(encoding="utf-8"))
_PLUGIN_ROW = next(
    row for row in _POLICY_PAYLOAD["guaranteed_rows"] if row["row_id"] == "plugin-bootstrap"
)
_PLUGIN_PUBLIC_NAMES = tuple(
    next(
        value["public_names"]
        for value in _PLUGIN_ROW["plugin_api"]["namespaces"]
        if value["namespace"] == "feedbax.plugins"
    )
)

GUARANTEED_IMPORTS = {
    "feedbax.plugins": _PLUGIN_PUBLIC_NAMES,
    "feedbax.orchestration.drivers": (
        "DRIVER_CAPABILITIES_SCHEMA_ID",
        "DRIVER_CAPABILITIES_SCHEMA_VERSION",
        "DRIVER_CAPABILITIES_SCHEMA_VERSION_V1",
        "DRIVER_CAPABILITIES_SCHEMA_VERSION_V2",
        "DRIVER_CAPABILITIES_SCHEMA_VERSION_V3",
        "DriverAuthority",
        "DriverCapabilityEnvelope",
        "DriverCapabilityFacts",
        "DriverConstructionContext",
        "DriverHook",
        "DriverRegistration",
        "DriverRegistry",
        "DriverStage",
        "DriverVenue",
        "RealizedDriverCapabilities",
    ),
    "feedbax.orchestration.bundle": (
        "DEPLOYMENT_POLICY_SCHEMA_ID",
        "DEPLOYMENT_POLICY_SCHEMA_VERSION",
        "DEPLOYMENT_POLICY_SCHEMA_VERSION_V1",
        "RUN_BUNDLE_SCHEMA_ID",
        "RUN_BUNDLE_SCHEMA_VERSION",
        "RUN_BUNDLE_SCHEMA_VERSION_V11",
        "DeploymentPolicy",
        "RunBundle",
    ),
    "feedbax.orchestration.assembly": (
        "RUN_ASSEMBLY_REQUEST_SCHEMA_ID",
        "RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION",
        "RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION_V5",
        "RunAssemblyRequest",
    ),
    "feedbax.orchestration": (
        "RUN_SET_STATE_SCHEMA_ID",
        "RUN_SET_STATE_SCHEMA_VERSION",
        "RUN_SET_STATE_SCHEMA_VERSION_V4",
        "EMERGENCY_RUN_SET_RECORD_SCHEMA_ID",
        "EMERGENCY_RUN_SET_RECORD_SCHEMA_VERSION",
        "ControlFilesystemPreflight",
        "ControlFilesystemPreflightError",
        "CustodyPreservationRequired",
        "EmergencyProviderIdentity",
        "EmergencyRunSetRecord",
        "PrimaryStatePersistenceError",
        "RunSetState",
        "RunSetStateStore",
        "StageEngine",
    ),
    "feedbax.contracts.row_index": (
        "ROW_INDEX_SCHEMA_ID",
        "ROW_INDEX_SCHEMA_VERSION",
        "ROW_INDEX_CUSTODY_SCHEMA_ID",
        "ROW_INDEX_CUSTODY_SCHEMA_VERSION",
        "RESOLVED_ROW_SET_SCHEMA_ID",
        "RESOLVED_ROW_SET_SCHEMA_VERSION",
        "AllRowsSelector",
        "AuthenticatedRowIndex",
        "ResolvedRowSet",
        "RowCustodyBinding",
        "RowIndexCustodyBindings",
        "RowIndexEntry",
        "RowSelectionError",
        "RowSelectionErrorCode",
        "RowSetSelector",
        "TagRowsSelector",
        "derive_row_label",
        "expand_row_selector",
        "normalize_row_tags",
    ),
    "feedbax.contracts.figure_roles": (
        "FIGURE_ROW_EXPANSION_REQUEST_SCHEMA_ID",
        "FIGURE_ROW_EXPANSION_REQUEST_SCHEMA_VERSION",
        "RESOLVED_FIGURE_INPUTS_SCHEMA_ID",
        "RESOLVED_FIGURE_INPUTS_SCHEMA_VERSION",
        "FigureRoleBindingContract",
        "FigureRoleReferenceError",
        "FigureRowExpansionRequest",
        "PerRowInputReference",
        "ResolvedFigureInput",
        "ResolvedFigureInputs",
        "SharedInputReference",
        "expand_figure_rows",
        "resolve_figure_input_roles",
        "row_namespace",
    ),
    "feedbax.contracts.experiment_envelope": (
        "EXPERIMENT_ENVELOPE_COMPILE_RESULT_SCHEMA_ID",
        "EXPERIMENT_ENVELOPE_COMPILE_RESULT_SCHEMA_VERSION",
        "ExperimentEnvelopeCompileRequest",
        "ExperimentEnvelopeCompileResult",
        "ExperimentEnvelopeRejection",
        "ExperimentEnvelopeRejectionCategory",
        "dispatch_experiment_envelope",
        "require_builtin_envelope_schema",
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


def test_driver_policy_schema_heads_match_reviewed_690_contract() -> None:
    drivers = importlib.import_module("feedbax.orchestration.drivers")
    bundle = importlib.import_module("feedbax.orchestration.bundle")
    assembly = importlib.import_module("feedbax.orchestration.assembly")

    assert drivers.DRIVER_CAPABILITIES_SCHEMA_ID == "feedbax.orchestration.driver-capabilities"
    assert drivers.DRIVER_CAPABILITIES_SCHEMA_VERSION == "3"
    assert drivers.DRIVER_CAPABILITIES_SCHEMA_VERSION_V1 == "1"
    assert drivers.DRIVER_CAPABILITIES_SCHEMA_VERSION_V2 == "2"
    assert bundle.DEPLOYMENT_POLICY_SCHEMA_VERSION == "feedbax.spec.deployment_policy.v2"
    assert bundle.RUN_BUNDLE_SCHEMA_VERSION == "feedbax.orchestration.run_bundle.v12"
    assert assembly.RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION == "feedbax.spec.run_assembly_request.v7"


def test_persistence_policy_schema_heads_match_reviewed_b85_contract() -> None:
    orchestration = importlib.import_module("feedbax.orchestration")

    assert orchestration.RUN_SET_STATE_SCHEMA_VERSION == "feedbax.orchestration.run_set_state.v5"
    assert orchestration.RUN_SET_STATE_SCHEMA_VERSION_V4 == "feedbax.orchestration.run_set_state.v4"
    assert (
        orchestration.EMERGENCY_RUN_SET_RECORD_SCHEMA_VERSION
        == "feedbax.orchestration.emergency_run_set_record.v1"
    )
    assert issubclass(orchestration.PrimaryStatePersistenceError, OSError)
    assert issubclass(orchestration.CustodyPreservationRequired, RuntimeError)


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


def test_plugin_api_manifest_pins_all_direct_rlrmp_entrypoint_imports() -> None:
    direct_imports = _PLUGIN_ROW["plugin_api"]["direct_entrypoint_imports"]
    assert len(direct_imports) == len(set(direct_imports)) == 12
    assert set(direct_imports).issubset(_PLUGIN_PUBLIC_NAMES)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("duplicate", "contains duplicates"),
        ("unclassified", "unclassified facade name"),
        ("method", "registry method.*is unavailable"),
        ("consumer", "consumer.*is unavailable"),
    ],
)
def test_plugin_api_checker_rejects_manifest_inventory_drift(
    mutation: str,
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = ROOT / "scripts" / "check_downstream_interface_policy.py"
    spec = importlib.util.spec_from_file_location(
        f"check_downstream_interface_policy_{mutation}", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row = copy.deepcopy(_PLUGIN_ROW)
    plugin_api = row["plugin_api"]
    if mutation == "duplicate":
        plugin_api["direct_entrypoint_imports"].append(plugin_api["direct_entrypoint_imports"][0])
    elif mutation == "unclassified":
        plugin_api["direct_entrypoint_imports"].append("ACCIDENTAL_ENTRYPOINT_NAME")
    elif mutation == "method":
        plugin_api["families"][0]["registry_methods"].append("accidental_method")
    else:
        plugin_api["families"][0]["public_consumers"].append("feedbax.analysis:accidental_consumer")
    monkeypatch.setattr(module, "_document_plugin_api", lambda _document: plugin_api)

    with pytest.raises(ValueError, match=match):
        module._check_plugin_api(row, "rendered inventory")


@pytest.mark.parametrize("mutation", ["reordered", "extra"])
def test_plugin_api_checker_rejects_facade_inventory_drift(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = ROOT / "scripts" / "check_downstream_interface_policy.py"
    spec = importlib.util.spec_from_file_location(
        f"check_downstream_interface_policy_facade_{mutation}", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plugin_api = copy.deepcopy(_PLUGIN_ROW["plugin_api"])
    facade_path = ROOT / "feedbax" / "plugins" / "__init__.py"
    original_literal_assignments = module._literal_assignments
    facade_assignments = original_literal_assignments(facade_path)
    facade_all = list(facade_assignments["__all__"])
    if mutation == "reordered":
        facade_all[0], facade_all[1] = facade_all[1], facade_all[0]
    else:
        facade_all.append("ACCIDENTAL_EXPORT")
    facade_assignments["__all__"] = facade_all

    monkeypatch.setattr(module, "_document_plugin_api", lambda _document: plugin_api)
    monkeypatch.setattr(
        module,
        "_literal_assignments",
        lambda path: facade_assignments
        if path == facade_path
        else original_literal_assignments(path),
    )

    with pytest.raises(ValueError, match="exact ordered inventory drifted"):
        module._check_plugin_api(copy.deepcopy(_PLUGIN_ROW), "rendered inventory")


def test_ratified_rows_bind_v14_and_have_no_pending_coverage() -> None:
    fixture = ROOT / "external" / "feedbax_conformance_fixture"
    manifest = json.loads(
        (fixture / "src/feedbax_external_conformance/policy_manifest.v1.json").read_text(
            encoding="utf-8"
        )
    )
    rows = {row["row_id"]: row for row in manifest["guaranteed_rows"]}
    for row_id in (
        "orchestration-lifecycle",
        "custody-persistence",
        "emergency-persistence",
        "result-role-binding",
        "figure-composition",
        "figure-role-references",
    ):
        assert rows[row_id]["coverage_status"] == "covered"
    assert rows["terminal-certification"]["coverage_status"] == "not-external-covered"
    assert rows["terminal-certification"]["case_ids"] == []
    assert "pending-final-sync" not in json.dumps(manifest)

    result_source = (fixture / "src/feedbax_external_conformance/result.py").read_text(
        encoding="utf-8"
    )
    assert 'Literal["feedbax.external_conformance.result.v14"]' in result_source
    assert "v12 cannot migrate to v14" in result_source
    assert "v13 cannot migrate to v14" in result_source


def test_ratified_policy_checker_rejects_residual_pending_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = ROOT / "scripts" / "check_downstream_interface_policy.py"
    spec = importlib.util.spec_from_file_location(
        "check_downstream_interface_policy_pending", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = json.loads(module.POLICY_MANIFEST.read_text(encoding="utf-8"))
    manifest["guaranteed_rows"][0]["coverage_status"] = "pending-final-sync"
    pending_manifest = tmp_path / "policy_manifest.v1.json"
    pending_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(module, "POLICY_MANIFEST", pending_manifest)

    with pytest.raises(ValueError, match="retains a pending-final-sync row"):
        module.check_policy()


def test_policy_does_not_promote_runtime_namespace() -> None:
    document = (ROOT / "docs" / "design" / "downstream_interface_stability.md").read_text(
        encoding="utf-8"
    )
    guarantee_block = document.split("<!-- policy-guarantees:start -->", 1)[1].split(
        "<!-- policy-guarantees:end -->", 1
    )[0]
    assert "feedbax.runtime." not in guarantee_block
