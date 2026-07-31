from __future__ import annotations

import ast
import importlib
import os
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


def _required_cases(fixture_package, *, value: object = True) -> dict[str, object]:
    return dict.fromkeys(fixture_package.REQUIRED_CASE_IDS, value)


def _protocol_roles() -> dict[str, None]:
    return {"current": None, "minimum": None}


def test_result_v1_migrates_with_unratified_role_slots(fixture_package) -> None:
    current = fixture_package.ConformanceResult(
        status="pass",
        feedbax_version="0.1.2",
        feedbax_install_root="/installed/feedbax",
        fixture_install_root="/installed/fixture",
        protocol_roles=_protocol_roles(),
        cases=_required_cases(fixture_package),
        lifecycle={"status": "pass"},
    )
    legacy = current.model_dump(mode="json")
    legacy["schema_version"] = "feedbax.external_conformance.result.v1"
    legacy.pop("protocol_roles")

    migrated = fixture_package.load_result(legacy)

    assert migrated.schema_version == "feedbax.external_conformance.result.v2"
    assert migrated.protocol_roles.current is None
    assert migrated.protocol_roles.minimum is None


def test_result_v1_rejects_supplied_protocol_roles(fixture_package) -> None:
    payload = fixture_package.ConformanceResult(
        status="pass",
        feedbax_version="0.1.2",
        feedbax_install_root="/installed/feedbax",
        fixture_install_root="/installed/fixture",
        protocol_roles=_protocol_roles(),
        cases=_required_cases(fixture_package),
        lifecycle={"status": "pass"},
    ).model_dump(mode="json")
    payload["schema_version"] = "feedbax.external_conformance.result.v1"

    with pytest.raises(ValueError, match="v1 did not define protocol_roles"):
        fixture_package.load_result(payload)


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
        "cases": _required_cases(fixture_package),
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
                "protocol_roles": _protocol_roles(),
                "cases": _required_cases(fixture_package),
                "lifecycle": {
                    "status": "blocked",
                    "reason_code": "feedbax-7e7dac8-wheel-provenance-unavailable",
                },
                "undeclared": True,
            }
        )


@pytest.mark.parametrize(
    ("status", "cases_pass", "lifecycle"),
    [
        ("blocked", False, {"status": "blocked", "reason_code": "gap"}),
        ("pass", True, {"status": "blocked", "reason_code": "gap"}),
        ("blocked", True, {"status": "blocked", "reason_code": None}),
        ("pass", True, {"status": "pass", "reason_code": "stale-gap"}),
    ],
)
def test_result_rejects_inconsistent_outcomes(
    fixture_package,
    status: str,
    cases_pass: bool,
    lifecycle: dict[str, str | None],
) -> None:
    with pytest.raises(ValidationError):
        fixture_package.ConformanceResult(
            status=status,
            feedbax_version="0.1.2",
            feedbax_install_root="/installed/feedbax",
            fixture_install_root="/installed/fixture",
            protocol_roles=_protocol_roles(),
            cases=_required_cases(fixture_package, value=cases_pass),
            lifecycle=lifecycle,
        )


def test_result_v2_case_contract_is_exact(fixture_package) -> None:
    assert fixture_package.REQUIRED_CASE_IDS == (
        "ordered_registration",
        "component_registration_and_migration",
        "value_identity",
        "material_dependencies",
        "staged_exact_parent_migration",
        "public_lifecycle_recovery",
    )
    valid = {
        "status": "pass",
        "feedbax_version": "0.1.2",
        "feedbax_install_root": "/installed/feedbax",
        "fixture_install_root": "/installed/fixture",
        "protocol_roles": _protocol_roles(),
        "cases": _required_cases(fixture_package),
        "lifecycle": {"status": "pass"},
    }
    for cases in (
        {key: value for key, value in valid["cases"].items() if key != "public_lifecycle_recovery"},
        {**valid["cases"], "unexpected_case": True},
    ):
        with pytest.raises(ValidationError, match="must exactly match"):
            fixture_package.ConformanceResult.model_validate({**valid, "cases": cases})


def test_result_v2_case_values_are_strict_booleans(fixture_package) -> None:
    cases = _required_cases(fixture_package)
    cases["public_lifecycle_recovery"] = "yes"

    with pytest.raises(ValidationError, match="bool"):
        fixture_package.ConformanceResult(
            status="pass",
            feedbax_version="0.1.2",
            feedbax_install_root="/installed/feedbax",
            fixture_install_root="/installed/fixture",
            protocol_roles=_protocol_roles(),
            cases=cases,
            lifecycle={"status": "pass"},
        )


@pytest.mark.parametrize("slot", ["current", "minimum"])
def test_result_v2_protocol_roles_remain_unbound(fixture_package, slot: str) -> None:
    roles: dict[str, object] = _protocol_roles()
    roles[slot] = "unratified"
    with pytest.raises(ValidationError):
        fixture_package.ConformanceResult(
            status="pass",
            feedbax_version="0.1.2",
            feedbax_install_root="/installed/feedbax",
            fixture_install_root="/installed/fixture",
            protocol_roles=roles,
            cases=_required_cases(fixture_package),
            lifecycle={"status": "pass"},
        )


@pytest.mark.parametrize(
    "protocol_roles",
    [None, {}, {"current": None}, {"minimum": None}],
    ids=["absent", "empty", "missing-minimum", "missing-current"],
)
def test_result_v2_requires_explicit_protocol_role_slots(
    fixture_package,
    protocol_roles: dict[str, None] | None,
) -> None:
    payload = {
        "schema_id": "feedbax.external_conformance.result",
        "schema_version": "feedbax.external_conformance.result.v2",
        "status": "pass",
        "feedbax_version": "0.1.2",
        "feedbax_install_root": "/installed/feedbax",
        "fixture_install_root": "/installed/fixture",
        "cases": _required_cases(fixture_package),
        "lifecycle": {"status": "pass"},
    }
    if protocol_roles is not None:
        payload["protocol_roles"] = protocol_roles

    with pytest.raises(ValidationError, match="Field required"):
        fixture_package.load_result(payload)


@pytest.mark.parametrize(
    ("source", "qualified_name"),
    [
        ("from feedbax import _private\n", "feedbax._private"),
        ("from feedbax.public import _private\n", "feedbax.public._private"),
    ],
)
def test_private_feedbax_import_guard_rejects_imported_symbols(
    fixture_package,
    tmp_path: Path,
    source: str,
    qualified_name: str,
) -> None:
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    (fixture_root / "candidate.py").write_text(source, encoding="utf-8")
    runner = importlib.import_module(f"{fixture_package.__name__}.runner")

    with pytest.raises(AssertionError, match=qualified_name):
        runner._require_no_private_feedbax_imports(fixture_root)


def test_network_denial_rejects_promised_runner_tcp_apis(fixture_package) -> None:
    network = importlib.import_module(f"{fixture_package.__name__}.network")

    with network.network_denied():
        network._assert_outbound_tcp_denied()


def test_fixture_has_no_private_feedbax_imports() -> None:
    violations: list[str] = []
    for path in FIXTURE_SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names = [f"{node.module}.{alias.name}" for alias in node.names]
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
                protocol_roles={"current": None, "minimum": None},
                cases={case_id: True for case_id in fixture.REQUIRED_CASE_IDS},
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


def test_clean_wheel_runner_checks_installed_dependency_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = ROOT / "scripts" / "run_external_conformance.py"
    spec = importlib.util.spec_from_file_location("external_conformance_runner", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls: list[tuple[list[str], Path, dict[str, str]]] = []
    monkeypatch.setattr(
        module,
        "_run",
        lambda command, *, cwd, env: calls.append((command, cwd, env)),
    )
    python_executable = tmp_path / "venv" / "bin" / "python"
    environment = {"UV_CACHE_DIR": str(tmp_path / "cache")}

    module._check_installed_metadata(
        python_executable,
        cwd=tmp_path,
        env=environment,
    )

    assert calls == [
        (
            ["uv", "pip", "check", "--python", str(python_executable)],
            tmp_path,
            environment,
        )
    ]


def test_clean_wheel_wrapper_rejects_malformed_result(
    tmp_path: Path,
) -> None:
    script_path = ROOT / "scripts" / "run_external_conformance.py"
    spec = importlib.util.spec_from_file_location("external_conformance_runner", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result_path = tmp_path / "result.json"
    result_path.write_text(
        """{
  "schema_id": "feedbax.external_conformance.result",
  "schema_version": "feedbax.external_conformance.result.v2",
  "status": "pass",
  "feedbax_version": "0.1.2",
  "feedbax_install_root": "/isolated/feedbax",
  "fixture_install_root": "/isolated/fixture",
  "cases": {"ordered_registration": true},
  "lifecycle": {"status": "pass"}
}
""",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(FIXTURE_SRC)

    with pytest.raises(subprocess.CalledProcessError):
        module._load_installed_result(
            Path(sys.executable),
            result_path,
            cwd=tmp_path,
            env=environment,
        )


def test_local_lifecycle_uses_clean_interpreter_distribution_inventory(
    fixture_package,
    tmp_path: Path,
) -> None:
    lifecycle = importlib.import_module(f"{fixture_package.__name__}.lifecycle")
    driver = lifecycle._driver(tmp_path, None)

    assert driver.python_executable == sys.executable
    assert driver.freeze_lines is None


def test_local_lifecycle_child_is_fixed_print_only(fixture_package) -> None:
    lifecycle = importlib.import_module(f"{fixture_package.__name__}.lifecycle")
    compiled = lifecycle._LocalLifecycleCompiler().compile(
        authored={
            "schema_id": "feedbax.spec.studio.training_assembly",
            "schema_version": "feedbax.spec.studio.training_assembly.v1",
        },
        run_set_id="fixture",
        context=None,
    )

    assert [(row.row_id, row.launch.command) for row in compiled.rows] == [
        (
            "fixture-row",
            [
                sys.executable,
                "-c",
                "print('feedbax external conformance lifecycle')",
            ],
        )
    ]


def test_ci_invokes_repository_clean_wheel_command() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "uv run --no-sync python scripts/run_external_conformance.py" in workflow
