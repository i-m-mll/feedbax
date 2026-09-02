"""Focused contract tests for downstream policy admission and named imports."""

from __future__ import annotations

import asyncio
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import sys

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
sys.path.insert(0, str(ROOT / "external" / "feedbax_conformance_fixture" / "src"))
POLICY_MANIFEST = (
    ROOT
    / "external"
    / "feedbax_conformance_fixture"
    / "src"
    / "feedbax_external_conformance"
    / "policy_manifest.v2.json"
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

_GUARANTEED_APIS = tuple(
    (row["row_id"], entry["namespace"], tuple(entry["public_names"]))
    for row in _POLICY_PAYLOAD["guaranteed_rows"]
    for entry in row["public_api"]["namespaces"]
)


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
    for row_id, module_name, symbols in _GUARANTEED_APIS:
        module = importlib.import_module(module_name)
        missing = [symbol for symbol in symbols if not hasattr(module, symbol)]
        assert not missing, f"{row_id}: {module_name} is missing {missing!r}"


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


def test_plugin_api_manifest_pins_all_direct_downstream_entrypoint_imports() -> None:
    direct_imports = _PLUGIN_ROW["plugin_api"]["direct_entrypoint_imports"]
    assert len(direct_imports) == len(set(direct_imports)) == 10
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
        (fixture / "src/feedbax_external_conformance/policy_manifest.v2.json").read_text(
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
    assert "terminal-certification" not in rows
    assert "dynamic-component-definition" not in rows
    assert "feedbax.manifest.training_run_certification.v1" in rows[
        "orchestration-lifecycle"
    ]["schemas"]["current"]
    assert "feedbax.spec.component_definition.v3" in rows["component-registration"][
        "schemas"
    ]["current"]
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
    pending_manifest = tmp_path / "policy_manifest.v2.json"
    pending_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(module, "POLICY_MANIFEST", pending_manifest)

    with pytest.raises(ValueError, match="retains a pending-final-sync row"):
        module.check_policy()


def test_policy_checker_rejects_v1_manifest_instead_of_inventing_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = ROOT / "scripts" / "check_downstream_interface_policy.py"
    spec = importlib.util.spec_from_file_location(
        "check_downstream_interface_policy_v1", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = json.loads(module.POLICY_MANIFEST.read_text(encoding="utf-8"))
    manifest["schema_version"] = "feedbax.external_conformance.policy_manifest.v1"
    v1_manifest = tmp_path / "policy_manifest.v1.json"
    v1_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(module, "POLICY_MANIFEST", v1_manifest)

    with pytest.raises(ValueError, match="fixture policy manifest schema drifted"):
        module.check_policy()


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("| Status | Owner-ratified |", "| Status | Ratification-ready |", "field 'Status'"),
        (
            "| Ratification evidence | Base policy: protected `develop` merge ",
            "| Ratification evidence | Unverified merge ",
            "field 'Ratification evidence'",
        ),
        (
            "| Policy source head | Protected `develop` merge ",
            "| Policy source head | Unverified merge ",
            "field 'Policy source head'",
        ),
        (
            "| Result schema identity | `feedbax.external_conformance.result.v14` |",
            "| Result schema identity | `feedbax.external_conformance.result.v13` |",
            "field 'Result schema identity'",
        ),
        (
            "| Runtime result evidence | No concrete conformance result artifact or execution "
            "receipt is pinned in this policy |",
            "| Runtime result evidence | `feedbax.external_conformance.result.v14` |",
            "field 'Runtime result evidence'",
        ),
        (
            "that validated v14 result as an artifact",
            "that validated v13 result as an artifact",
            "current v14 result",
        ),
    ],
)
def test_policy_checker_rejects_ratification_metadata_staleness(
    old: str,
    new: str,
    match: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = ROOT / "scripts" / "check_downstream_interface_policy.py"
    spec = importlib.util.spec_from_file_location(
        "check_downstream_interface_policy_ratification_metadata", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    document = module.POLICY_DOCUMENT.read_text(encoding="utf-8")
    assert old in document
    stale_document = tmp_path / "downstream_interface_stability.md"
    stale_document.write_text(document.replace(old, new, 1), encoding="utf-8")
    monkeypatch.setattr(module, "POLICY_DOCUMENT", stale_document)

    with pytest.raises(ValueError, match=match):
        module.check_policy()


def test_policy_checker_requires_schema_and_runtime_evidence_domains_to_be_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = ROOT / "scripts" / "check_downstream_interface_policy.py"
    spec = importlib.util.spec_from_file_location(
        "check_downstream_interface_policy_evidence_domains", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    document = module.POLICY_DOCUMENT.read_text(encoding="utf-8")
    boundary = (
        "The result schema\nidentity names the required shape of a conformance result; "
        "it is not evidence\nthat the fixture ran."
    )
    assert boundary in document
    stale_document = tmp_path / "downstream_interface_stability.md"
    stale_document.write_text(
        document.replace(boundary, "The result schema is runtime evidence."), encoding="utf-8"
    )
    monkeypatch.setattr(module, "POLICY_DOCUMENT", stale_document)

    with pytest.raises(ValueError, match="conflates result schema identity with runtime evidence"):
        module.check_policy()


def test_policy_does_not_promote_runtime_namespace() -> None:
    document = (ROOT / "docs" / "design" / "downstream_interface_stability.md").read_text(
        encoding="utf-8"
    )
    guarantee_block = document.split("<!-- policy-guarantees:start -->", 1)[1].split(
        "<!-- policy-guarantees:end -->", 1
    )[0]
    assert "feedbax.runtime." not in guarantee_block
