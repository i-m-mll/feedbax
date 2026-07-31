"""Validate the ratified downstream-interface policy and fixture mapping."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[1]
POLICY_ID = "feedbax.downstream-interface-stability.v1"
POLICY_SCHEMA = "feedbax.external_conformance.policy_manifest.v1"
START = "<!-- feedbax-downstream-stability:start -->"
END = "<!-- feedbax-downstream-stability:end -->"
GUARANTEE_START = "<!-- policy-guarantees:start -->"
GUARANTEE_END = "<!-- policy-guarantees:end -->"
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
RATIFIED_NON_EXTERNAL_ROWS = {"terminal-certification"}
CONCRETE_FAMILY_PUBLIC_NAMES = {
    "APPLICATION_REGISTRY_KEYS",
    "ApplicationRegistryBundle",
    "COMPONENTS",
    "TRAINING_METHODS",
    "ROW_LOWERERS",
    "EXECUTION_PREPARATIONS",
    "ANALYSIS_RECIPES",
    "EVALUATION_RECIPES",
    "EVALUATION_BATCH_CONSUMERS",
    "EVALUATION_PRODUCT_UNION_FINALIZERS",
    "compile_training_method_authoring",
}


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
    if 'Literal["feedbax.external_conformance.result.v12"]' not in result_source:
        raise ValueError("ratified policy requires the authoritative v12 result")
    if "v11 cannot migrate to v12" not in result_source:
        raise ValueError("v11 must reject rather than synthesize custody and role evidence")
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
    if set(manifest_rows["plugin-bootstrap"].get("public_names", ())) != CONCRETE_FAMILY_PUBLIC_NAMES:
        raise ValueError("plugin policy omits or widens concrete-family public names")
    for row_id in (
        "orchestration-lifecycle",
        "custody-persistence",
        "emergency-persistence",
        "result-role-binding",
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
        "feedbax.external_conformance.result.v12"
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
