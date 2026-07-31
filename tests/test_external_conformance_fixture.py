from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest
from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SRC = ROOT / "external" / "feedbax_conformance_fixture" / "src"


@pytest.fixture
def fixture_package(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(FIXTURE_SRC))
    return importlib.import_module("feedbax_external_conformance")


def test_result_v1_migrates_with_unratified_role_slots(fixture_package) -> None:
    current = fixture_package.ConformanceResult(
        status="blocked",
        feedbax_version="0.1.2",
        feedbax_install_root="/installed/feedbax",
        fixture_install_root="/installed/fixture",
        cases={"foundation": True},
        lifecycle={
            "status": "blocked",
            "reason_code": "feedbax-7e7dac8-wheel-provenance-unavailable",
        },
    )
    legacy = current.model_dump(mode="json")
    legacy["schema_version"] = "feedbax.external_conformance.result.v1"
    legacy.pop("protocol_roles")

    migrated = fixture_package.load_result(legacy)

    assert migrated.schema_version == "feedbax.external_conformance.result.v2"
    assert migrated.protocol_roles.current is None
    assert migrated.protocol_roles.minimum is None


@pytest.mark.parametrize(
    "version",
    [None, "feedbax.external_conformance.result.v0", "feedbax.external_conformance.result.v3"],
)
def test_result_rejects_unsupported_versions(fixture_package, version: str | None) -> None:
    payload = {
        "schema_id": "feedbax.external_conformance.result",
        "schema_version": version,
        "status": "blocked",
        "feedbax_version": "0.1.2",
        "feedbax_install_root": "/installed/feedbax",
        "fixture_install_root": "/installed/fixture",
        "cases": {"foundation": True},
        "lifecycle": {
            "status": "blocked",
            "reason_code": "feedbax-7e7dac8-wheel-provenance-unavailable",
        },
    }

    with pytest.raises(ValueError, match="unsupported.*schema_version"):
        fixture_package.load_result(payload)


def test_result_model_rejects_extra_fields(fixture_package) -> None:
    with pytest.raises(ValidationError):
        fixture_package.ConformanceResult.model_validate(
            {
                "status": "blocked",
                "feedbax_version": "0.1.2",
                "feedbax_install_root": "/installed/feedbax",
                "fixture_install_root": "/installed/fixture",
                "cases": {"foundation": True},
                "lifecycle": {
                    "status": "blocked",
                    "reason_code": "feedbax-7e7dac8-wheel-provenance-unavailable",
                },
                "undeclared": True,
            }
        )


@pytest.mark.parametrize(
    ("status", "cases", "lifecycle"),
    [
        ("blocked", {"foundation": False}, {"status": "blocked", "reason_code": "gap"}),
        ("pass", {"foundation": True}, {"status": "blocked", "reason_code": "gap"}),
        ("blocked", {"foundation": True}, {"status": "blocked", "reason_code": None}),
        ("pass", {"foundation": True}, {"status": "pass", "reason_code": "stale-gap"}),
    ],
)
def test_result_rejects_inconsistent_outcomes(
    fixture_package,
    status: str,
    cases: dict[str, bool],
    lifecycle: dict[str, str | None],
) -> None:
    with pytest.raises(ValidationError):
        fixture_package.ConformanceResult(
            status=status,
            feedbax_version="0.1.2",
            feedbax_install_root="/installed/feedbax",
            fixture_install_root="/installed/fixture",
            cases=cases,
            lifecycle=lifecycle,
        )


def test_fixture_has_no_private_feedbax_imports() -> None:
    violations: list[str] = []
    for path in FIXTURE_SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names = [node.module]
            else:
                continue
            violations.extend(
                name
                for name in names
                if name.startswith("feedbax.")
                if any(part.startswith("_") for part in name.split(".")[1:])
            )
    assert violations == []


def test_ci_invokes_repository_clean_wheel_command() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "uv run --no-sync python scripts/run_external_conformance.py" in workflow
