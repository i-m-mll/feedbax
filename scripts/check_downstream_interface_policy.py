"""Validate the ratified downstream-interface policy and fixture mapping."""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
import re
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
POLICY_ID = "feedbax.downstream-interface-stability.v1"
POLICY_SCHEMA = "feedbax.external_conformance.policy_manifest.v1"
START = "<!-- feedbax-downstream-stability:start -->"
END = "<!-- feedbax-downstream-stability:end -->"
GUARANTEE_START = "<!-- policy-guarantees:start -->"
GUARANTEE_END = "<!-- policy-guarantees:end -->"
PLUGIN_API_START = "<!-- plugin-api-inventory:start -->"
PLUGIN_API_END = "<!-- plugin-api-inventory:end -->"
FIGURE_API_START = "<!-- figure-api-inventory:start -->"
FIGURE_API_END = "<!-- figure-api-inventory:end -->"
INSTRUCTION_FILES = (ROOT / "AGENTS.md", ROOT / "CLAUDE.md")
POLICY_DOCUMENT = ROOT / "docs" / "design" / "downstream_interface_stability.md"
FIXTURE_ROOT = ROOT / "external" / "feedbax_conformance_fixture"
POLICY_MANIFEST = FIXTURE_ROOT / "src" / "feedbax_external_conformance" / "policy_manifest.v1.json"
DRIVER_POLICY_SCHEMAS = {
    "current": [
        "feedbax.orchestration.driver-capabilities version 3",
        "feedbax.spec.deployment_policy.v2",
        "feedbax.spec.run_assembly_request.v6",
        "feedbax.orchestration.run_bundle.v12",
    ],
    "migrated": [
        "feedbax.spec.deployment_policy.v1",
        "feedbax.spec.run_assembly_request.v5",
        "feedbax.orchestration.run_bundle.v11",
    ],
    "rejected": [
        "feedbax.orchestration.driver-capabilities version 1",
        "feedbax.orchestration.driver-capabilities version 2",
        "older unsupported request and bundle versions",
        "unknown",
    ],
}
# Rows whose behavior is proved by focused in-repo tests rather than by an
# external conformance case. Adding a row here is a deliberate statement that no
# external case covers it, never a way to skip evidence.
RATIFIED_NON_EXTERNAL_ROWS = {
    "terminal-certification",
    # Draft rows raised by Feedbax issue 88d021d; ratified only by the owner's
    # approval and merge of the protected delivery that carries them.
    "report-surface",
    "evaluation-surface",
    "analysis-authoring",
}
DRAFT_POLICY_ROWS = {"report-surface", "evaluation-surface", "analysis-authoring"}
DRAFT_POLICY_MARKER = "## Pending owner ratification: envelope-layer prerequisite rows"


def _marked_block(path: Path, start_marker: str, end_marker: str) -> str:
    text = path.read_text(encoding="utf-8")
    if text.count(start_marker) != 1 or text.count(end_marker) != 1:
        raise ValueError(f"{path} must contain exactly one {start_marker!r}/{end_marker!r} pair")
    start = text.index(start_marker)
    end = text.index(end_marker, start) + len(end_marker)
    return text[start:end]


def _literal_assignments(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return values


def _document_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        match = re.fullmatch(r"\| ([^|]+?) \| (.+) \|", line)
        if match is not None:
            fields[match.group(1).strip()] = match.group(2).strip()
    return fields


def _document_rows(text: str) -> dict[str, tuple[str, ...]]:
    block = _marked_block(POLICY_DOCUMENT, GUARANTEE_START, GUARANTEE_END)
    rows: dict[str, tuple[str, ...]] = {}
    for line in block.splitlines():
        match = re.match(r"\| `([^`]+)` \|", line)
        if match is None:
            continue
        columns = tuple(column.strip() for column in line.strip().strip("|").split("|"))
        rows[match.group(1)] = tuple(re.findall(r"`([^`]+)`", columns[-1]))
    return rows


def _unique_strings(values: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not value for value in values
    ):
        raise ValueError(f"plugin API {field} must be a list of non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"plugin API {field} contains duplicates")
    return tuple(values)


def _validated_plugin_api(row: dict[str, object]) -> dict[str, object]:
    plugin_api = row.get("plugin_api")
    if not isinstance(plugin_api, dict) or set(plugin_api) != {
        "namespaces",
        "direct_entrypoint_imports",
        "families",
    }:
        raise ValueError("plugin-bootstrap.plugin_api has an invalid shape")
    namespaces = plugin_api["namespaces"]
    if not isinstance(namespaces, list) or not namespaces:
        raise ValueError("plugin API namespaces must be a non-empty list")
    namespace_names = []
    public_names: dict[str, tuple[str, ...]] = {}
    for index, value in enumerate(namespaces):
        if not isinstance(value, dict) or set(value) != {"namespace", "public_names"}:
            raise ValueError(f"plugin API namespace {index} has an invalid shape")
        namespace = value["namespace"]
        if not isinstance(namespace, str) or not namespace:
            raise ValueError(f"plugin API namespace {index} has an invalid name")
        namespace_names.append(namespace)
        public_names[namespace] = _unique_strings(
            value["public_names"], field=f"namespace {namespace} public_names"
        )
    if len(namespace_names) != len(set(namespace_names)):
        raise ValueError("plugin API namespaces contain duplicates")
    _unique_strings(plugin_api["direct_entrypoint_imports"], field="direct_entrypoint_imports")
    families = plugin_api["families"]
    if not isinstance(families, list) or not families:
        raise ValueError("plugin API families must be a non-empty list")
    expected_fields = {
        "key",
        "registry_type",
        "registry_methods",
        "callback_types",
        "support_types",
        "public_consumers",
    }
    family_keys = []
    for index, family in enumerate(families):
        if not isinstance(family, dict) or set(family) != expected_fields:
            raise ValueError(f"plugin API family {index} has an invalid shape")
        for field in ("key", "registry_type"):
            if not isinstance(family[field], str) or not family[field]:
                raise ValueError(f"plugin API family {index} has an invalid {field}")
        family_keys.append(family["key"])
        for field in (
            "registry_methods",
            "callback_types",
            "support_types",
            "public_consumers",
        ):
            _unique_strings(family[field], field=f"family {family['key']} {field}")
    if len(family_keys) != len(set(family_keys)):
        raise ValueError("plugin API family keys contain duplicates")
    return plugin_api


def _document_plugin_api(text: str) -> dict[str, object]:
    block = _marked_block(POLICY_DOCUMENT, PLUGIN_API_START, PLUGIN_API_END)
    namespace_line = next(
        (line for line in block.splitlines() if line.startswith("Namespace `")), None
    )
    direct_line = next(
        (line for line in block.splitlines() if line.startswith("Direct RLRMP")), None
    )
    if namespace_line is None or direct_line is None:
        raise ValueError("policy document omits the rendered plugin namespace or direct imports")
    namespace_tokens = re.findall(r"`([^`]+)`", namespace_line)
    families = []
    for line in block.splitlines():
        if not line.startswith("| `"):
            continue
        columns = [column.strip() for column in line.strip().strip("|").split("|")]
        if len(columns) != 6:
            raise ValueError("policy document plugin family row has the wrong column count")
        tokens = [re.findall(r"`([^`]+)`", column) for column in columns]
        if len(tokens[0]) != 1 or len(tokens[1]) != 1:
            raise ValueError("policy document plugin family identity is malformed")
        families.append(
            {
                "key": tokens[0][0],
                "registry_type": tokens[1][0],
                "registry_methods": tokens[2],
                "callback_types": tokens[3],
                "support_types": tokens[4],
                "public_consumers": tokens[5],
            }
        )
    return {
        "namespaces": [{"namespace": namespace_tokens[0], "public_names": namespace_tokens[1:]}],
        "direct_entrypoint_imports": re.findall(r"`([^`]+)`", direct_line),
        "families": families,
    }


def _check_plugin_api(row: dict[str, object], document: str) -> None:
    plugin_api = _validated_plugin_api(row)
    if _document_plugin_api(document) != plugin_api:
        raise ValueError("policy document plugin API inventory drifted from the manifest")

    declared_by_namespace = {
        value["namespace"]: tuple(value["public_names"]) for value in plugin_api["namespaces"]
    }
    facade_names = declared_by_namespace.get("feedbax.plugins")
    if facade_names is None:
        raise ValueError("plugin API must declare the feedbax.plugins facade")
    facade_assignments = _literal_assignments(ROOT / "feedbax" / "plugins" / "__init__.py")
    facade_all = tuple(facade_assignments["__all__"])
    non_guaranteed = tuple(facade_assignments["_NON_GUARANTEED_PLUGIN_EXPORTS"])
    if len(non_guaranteed) != len(set(non_guaranteed)) or set(facade_names) & set(non_guaranteed):
        raise ValueError("feedbax.plugins non-guaranteed export inventory is invalid")
    if facade_all != (*facade_names, *non_guaranteed):
        raise ValueError("feedbax.plugins __all__ exact ordered inventory drifted")
    facade = importlib.import_module("feedbax.plugins")
    for name in facade_names:
        if name not in facade_all or not hasattr(facade, name):
            raise ValueError(f"feedbax.plugins does not export declared name {name!r}")
    direct_imports = tuple(plugin_api["direct_entrypoint_imports"])
    if any(name not in facade_names for name in direct_imports):
        raise ValueError("direct RLRMP imports contain an unclassified facade name")

    keys = {key.family: key for key in facade.APPLICATION_REGISTRY_KEYS}
    classified_types = set()
    for family in plugin_api["families"]:
        key = keys.get(family["key"])
        if key is None:
            raise ValueError(f"plugin API declares unknown registry family {family['key']!r}")
        registry_type = getattr(facade, family["registry_type"], None)
        if registry_type is not key.expected_type:
            raise ValueError(f"plugin API registry type drifted for {family['key']!r}")
        classified_types.add(family["registry_type"])
        for method in family["registry_methods"]:
            if not callable(getattr(registry_type, method, None)):
                raise ValueError(
                    f"plugin API registry method {family['registry_type']}.{method} is unavailable"
                )
        for name in (*family["callback_types"], *family["support_types"]):
            classified_types.add(name)
            if name not in facade_names or not hasattr(facade, name):
                raise ValueError(f"plugin API family type {name!r} is unclassified or unavailable")
        for consumer in family["public_consumers"]:
            if consumer.count(":") != 1:
                raise ValueError(f"plugin API consumer {consumer!r} has an invalid public path")
            namespace, name = consumer.split(":")
            if not callable(getattr(importlib.import_module(namespace), name, None)):
                raise ValueError(f"plugin API consumer {consumer!r} is unavailable")
    accidental = classified_types - set(facade_names)
    if accidental:
        raise ValueError(f"plugin API family inventory contains accidental names {accidental!r}")


def _check_figure_api(row: dict[str, object], document: str) -> None:
    public_api = row.get("public_api")
    if not isinstance(public_api, dict) or set(public_api) != {"namespaces", "cli"}:
        raise ValueError("figure-composition.public_api has an invalid shape")
    namespaces = public_api["namespaces"]
    if not isinstance(namespaces, list) or not namespaces:
        raise ValueError("figure public API namespaces must be a non-empty list")
    declared: dict[str, tuple[str, ...]] = {}
    for index, value in enumerate(namespaces):
        if not isinstance(value, dict) or set(value) != {"namespace", "public_names"}:
            raise ValueError(f"figure public API namespace {index} has an invalid shape")
        namespace = value["namespace"]
        if not isinstance(namespace, str) or not namespace:
            raise ValueError(f"figure public API namespace {index} has an invalid name")
        declared[namespace] = _unique_strings(
            value["public_names"], field=f"figure namespace {namespace} public_names"
        )
    if len(declared) != len(namespaces):
        raise ValueError("figure public API namespaces contain duplicates")
    cli = _unique_strings(public_api["cli"], field="figure CLI")

    block = _marked_block(POLICY_DOCUMENT, FIGURE_API_START, FIGURE_API_END)
    documented: dict[str, tuple[str, ...]] = {}
    documented_cli: tuple[str, ...] | None = None
    for line in block.splitlines():
        if line.startswith("Namespace `"):
            tokens = re.findall(r"`([^`]+)`", line)
            documented[tokens[0]] = tuple(tokens[1:])
        elif line.startswith("CLI: "):
            documented_cli = tuple(re.findall(r"`([^`]+)`", line))
    if documented != declared or documented_cli != cli:
        raise ValueError("policy document figure API inventory drifted from the manifest")
    for namespace, names in declared.items():
        module = importlib.import_module(namespace)
        for name in names:
            if not hasattr(module, name):
                raise ValueError(f"figure public API name {namespace}:{name} is unavailable")
    cli_source = (ROOT / "feedbax" / "bin" / "figure.py").read_text(encoding="utf-8")
    if 'add_parser(\n        "resolve"' not in cli_source or '"--with-lineage"' not in cli_source:
        raise ValueError("feedbax-figure resolve CLI inventory drifted")
    semantics = row.get("semantics")
    if not isinstance(semantics, list) or not semantics:
        raise ValueError("figure-composition semantics must be a non-empty list")
    for statement in _unique_strings(semantics, field="figure semantics"):
        if statement not in document:
            raise ValueError(f"policy document omits figure semantic {statement!r}")


def check_policy() -> None:
    blocks = [_marked_block(path, START, END) for path in INSTRUCTION_FILES]
    if blocks[0] != blocks[1]:
        raise ValueError("Feedbax downstream-stability instruction blocks differ")

    bootstrap = _literal_assignments(ROOT / "feedbax" / "plugins" / "bootstrap.py")
    expected = {
        "policy_id": bootstrap["DOWNSTREAM_INTERFACE_POLICY_ID"],
        "current": bootstrap["DOWNSTREAM_PROTOCOL_CURRENT"],
        "minimum": bootstrap["DOWNSTREAM_PROTOCOL_MINIMUM"],
        "effective_release": bootstrap["DOWNSTREAM_POLICY_EFFECTIVE_RELEASE"],
    }
    if expected["policy_id"] != POLICY_ID:
        raise ValueError(f"unexpected policy identity in bootstrap: {expected['policy_id']!r}")
    if not isinstance(expected["current"], int) or not isinstance(expected["minimum"], int):
        raise ValueError("downstream protocol constants must be integers")
    if expected["minimum"] > expected["current"]:
        raise ValueError("minimum downstream protocol exceeds current")

    document = POLICY_DOCUMENT.read_text(encoding="utf-8")
    fields = _document_fields(document)
    required_fields = {
        "Policy identity": f"`{expected['policy_id']}`",
        "Effective release": f"Feedbax `{expected['effective_release']}`",
        "Extension protocol": (
            f"current `{expected['current']}`, minimum supported `{expected['minimum']}`"
        ),
    }
    for name, value in required_fields.items():
        if fields.get(name) != value:
            raise ValueError(f"policy field {name!r} drifted: {fields.get(name)!r} != {value!r}")

    instruction_block = blocks[0]
    for value in (
        str(expected["policy_id"]),
        f"version `{expected['current']}`",
        f"version `{expected['minimum']}`",
        f"release `{expected['effective_release']}`",
    ):
        if value not in instruction_block:
            raise ValueError(f"instruction block does not pin {value!r}")

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    fixture = tomllib.loads((FIXTURE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    if project["project"]["version"] != expected["effective_release"]:
        raise ValueError("project version does not match downstream policy effective release")
    expected_dependency = f"feedbax[analysis]=={expected['effective_release']}"
    if expected_dependency not in fixture["project"]["dependencies"]:
        raise ValueError("external fixture dependency does not pin the effective release")

    payload = json.loads(POLICY_MANIFEST.read_text(encoding="utf-8"))
    if payload.get("schema_version") != POLICY_SCHEMA:
        raise ValueError("fixture policy manifest schema drifted")
    if payload.get("policy_id") != expected["policy_id"]:
        raise ValueError("fixture policy identity drifted")
    if payload.get("effective_release") != expected["effective_release"]:
        raise ValueError("fixture policy effective release drifted")
    if payload.get("protocol") != {
        "current": expected["current"],
        "minimum": expected["minimum"],
    }:
        raise ValueError("fixture policy protocol window drifted")

    document_rows = _document_rows(document)
    manifest_rows = {row["row_id"]: row for row in payload.get("guaranteed_rows", [])}
    if set(document_rows) != set(manifest_rows):
        raise ValueError(
            "document/fixture policy rows differ: "
            f"document={sorted(document_rows)!r}, manifest={sorted(manifest_rows)!r}"
        )
    result_values = _literal_assignments(
        FIXTURE_ROOT / "src" / "feedbax_external_conformance" / "result.py"
    )
    result_source = (FIXTURE_ROOT / "src" / "feedbax_external_conformance" / "result.py").read_text(
        encoding="utf-8"
    )
    if 'Literal["feedbax.external_conformance.result.v14"]' not in result_source:
        raise ValueError("ratified policy requires the authoritative v14 result")
    if "v12 cannot migrate to v14" not in result_source:
        raise ValueError("v12 must reject rather than synthesize figure evidence")
    if "v13 cannot migrate to v14" not in result_source:
        raise ValueError("v13 must reject rather than synthesize figure-role evidence")
    if "pending-final-sync" in document or "pending-final-sync" in POLICY_MANIFEST.read_text(
        encoding="utf-8"
    ):
        raise ValueError("ratified policy retains a pending-final-sync row")
    case_ids = frozenset(result_values["REQUIRED_CASE_IDS"])
    driver_row = manifest_rows.get("orchestration-driver")
    if driver_row is None:
        raise ValueError("fixture policy manifest omits the orchestration-driver row")
    if driver_row.get("case_ids") != ["external_driver_plugin"]:
        raise ValueError("driver policy row must preserve the reviewed v11 external case")
    if driver_row.get("schemas") != DRIVER_POLICY_SCHEMAS:
        raise ValueError("driver policy schema and migration mapping drifted")
    _check_plugin_api(manifest_rows["plugin-bootstrap"], document)
    _check_figure_api(manifest_rows["figure-composition"], document)
    for row_id in (
        "orchestration-lifecycle",
        "custody-persistence",
        "emergency-persistence",
        "result-role-binding",
        "figure-composition",
        "figure-role-references",
    ):
        row = manifest_rows.get(row_id)
        if row is None or row.get("coverage_status") != "covered":
            raise ValueError(f"ratified policy row {row_id!r} is not covered")
    if manifest_rows["custody-persistence"].get("case_ids") != ["custody_persistence_recovery"]:
        raise ValueError("custody policy row must bind the landed external case")
    if manifest_rows["emergency-persistence"].get("schemas", {}).get("current") != [
        "feedbax.orchestration.emergency_run_set_record.v1"
    ]:
        raise ValueError("emergency persistence schema mapping drifted")
    if manifest_rows["result-role-binding"].get("schemas", {}).get("current") != [
        "feedbax.external_conformance.result.v14"
    ]:
        raise ValueError("result-role schema mapping drifted")
    for row_id, row in manifest_rows.items():
        cases = tuple(row.get("case_ids", ()))
        if tuple(document_rows[row_id]) != cases:
            raise ValueError(f"document/manifest case IDs differ for {row_id!r}")
        unknown_cases = sorted(set(cases) - case_ids)
        if unknown_cases:
            raise ValueError(f"policy row {row_id!r} names unknown cases {unknown_cases!r}")
        if row["coverage_status"] == "covered" and not cases:
            raise ValueError(f"covered policy row {row_id!r} has no external case")
        if row["coverage_status"] == "not-external-covered" and cases:
            raise ValueError(f"non-external policy row {row_id!r} fabricates case coverage")
        if row["coverage_status"] not in {"covered", "not-external-covered"}:
            raise ValueError(f"ratified policy row {row_id!r} has unresolved coverage status")
        schemas = row.get("schemas")
        if not isinstance(schemas, dict) or set(schemas) != {"current", "migrated", "rejected"}:
            raise ValueError(f"policy row {row_id!r} has an invalid schema mapping")
    observed_non_external = {
        row_id
        for row_id, row in manifest_rows.items()
        if row["coverage_status"] == "not-external-covered"
    }
    if observed_non_external != RATIFIED_NON_EXTERNAL_ROWS:
        raise ValueError("ratified non-external coverage rows drifted")
    if not DRAFT_POLICY_ROWS <= set(manifest_rows):
        raise ValueError("draft policy rows are missing from the fixture manifest")
    if DRAFT_POLICY_MARKER not in document:
        raise ValueError("draft policy rows are not marked pending owner ratification")
    facade_non_guaranteed = tuple(
        _literal_assignments(ROOT / "feedbax" / "plugins" / "__init__.py")[
            "_NON_GUARANTEED_PLUGIN_EXPORTS"
        ]
    )
    if "REPORT_RECIPES" not in facade_non_guaranteed:
        raise ValueError("REPORT_RECIPES must stay outside the guaranteed plugin inventory")

    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for proof in (
        "--result-out artifacts/downstream-conformance.json",
        "actions/upload-artifact@v4",
        "path: artifacts/downstream-conformance.json",
        "if-no-files-found: error",
    ):
        if proof not in workflow:
            raise ValueError(f"CI conformance artifact proof is missing {proof!r}")


def main() -> int:
    check_policy()
    print("Feedbax downstream-interface policy is internally consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
