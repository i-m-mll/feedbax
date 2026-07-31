from __future__ import annotations

import ast
import importlib
from pathlib import Path
import subprocess
import sys

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
        status="pass",
        feedbax_version="0.1.2",
        feedbax_install_root="/installed/feedbax",
        fixture_install_root="/installed/fixture",
        cases={"foundation": True},
        lifecycle={"status": "pass"},
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


def test_module_entrypoint_enters_network_denial_before_runner_import(tmp_path: Path) -> None:
    script = r"""
import contextlib
import importlib.abc
import importlib.machinery
import importlib.util
from pathlib import Path
import sys

fixture_src = Path(sys.argv[1])
result_path = Path(sys.argv[2])
sys.path.insert(0, str(fixture_src))

import feedbax_external_conformance as fixture
from feedbax_external_conformance import network

denial_active = False

@contextlib.contextmanager
def observed_network_denial():
    global denial_active
    assert not denial_active
    denial_active = True
    try:
        yield
    finally:
        denial_active = False

network.network_denied = observed_network_denial

class ObservedRunnerLoader(importlib.abc.Loader):
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def create_module(self, spec):
        create = getattr(self.wrapped, "create_module", None)
        return None if create is None else create(spec)

    def exec_module(self, module):
        assert denial_active, "runner loading started before network denial"
        self.wrapped.exec_module(module)
        assert "feedbax" in sys.modules
        assert denial_active, "network denial ended during runner/Feedbax import"

        def fake_run_fixture(*, source_root=None):
            assert denial_active
            return fixture.ConformanceResult(
                status="pass",
                feedbax_version="0.1.2",
                feedbax_install_root="/isolated/feedbax",
                fixture_install_root="/isolated/fixture",
                cases={"import_order": True},
                lifecycle={"status": "pass"},
            )

        module.run_fixture = fake_run_fixture

class ObservedRunnerFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "feedbax_external_conformance.runner":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        assert spec is not None and spec.loader is not None
        spec.loader = ObservedRunnerLoader(spec.loader)
        return spec

sys.meta_path.insert(0, ObservedRunnerFinder())
import feedbax_external_conformance.__main__ as entrypoint

assert "feedbax_external_conformance.runner" not in sys.modules
assert "feedbax" not in sys.modules
assert entrypoint.main(["--result", str(result_path)]) == 0
assert "feedbax_external_conformance.runner" in sys.modules
assert "feedbax" in sys.modules
assert not denial_active
"""
    subprocess.run(
        [sys.executable, "-c", script, str(FIXTURE_SRC), str(tmp_path / "result.json")],
        check=True,
    )


def test_ci_invokes_repository_clean_wheel_command() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "uv run --no-sync python scripts/run_external_conformance.py" in workflow
