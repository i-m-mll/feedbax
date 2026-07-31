"""Clean-installed fixture orchestration and leakage checks."""

from __future__ import annotations

import ast
import importlib.metadata
import json
from pathlib import Path
import sys
from types import ModuleType

import feedbax
from feedbax.plugins import DOWNSTREAM_PROTOCOL_CURRENT, DOWNSTREAM_PROTOCOL_MINIMUM

from .cases import (
    check_component_registration_and_migration,
    check_component_param_array_values,
    check_dynamic_component_ports,
    check_exact_parent_migration,
    check_external_driver_plugin,
    check_material_dependencies,
    check_ordered_registration,
    check_resolved_evaluation_row_projection,
    check_unified_plugin_bootstrap,
    check_value_identity,
)
from .lifecycle import check_custody_persistence_recovery, check_public_lifecycle_recovery
from .result import ConformanceResult, LifecycleResult


def _package_root(module: ModuleType) -> Path:
    source = getattr(module, "__file__", None)
    if source is None:
        raise AssertionError(f"installed module {module.__name__!r} has no file")
    return Path(source).resolve().parent


def _require_clean_install(module: ModuleType, *, source_root: Path | None) -> Path:
    root = _package_root(module)
    if not root.is_relative_to(Path(sys.prefix).resolve()):
        raise AssertionError(f"{module.__name__} was not imported from the clean environment")
    if source_root is not None:
        if root.is_relative_to(source_root):
            raise AssertionError(f"{module.__name__} leaked from the source checkout")
        leaked_paths = [
            str(candidate)
            for raw in sys.path
            if raw
            if (candidate := Path(raw).resolve()).is_relative_to(source_root)
        ]
        if leaked_paths:
            raise AssertionError(f"source checkout leaked onto sys.path: {leaked_paths!r}")
    return root


def _require_noneditable_distribution(name: str) -> None:
    direct_url = importlib.metadata.distribution(name).read_text("direct_url.json")
    if direct_url is None:
        return
    payload = json.loads(direct_url)
    if payload.get("dir_info", {}).get("editable") is True:
        raise AssertionError(f"{name} was installed editably")


def _require_no_private_feedbax_imports(fixture_root: Path) -> None:
    for path in fixture_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names.extend(f"{node.module}.{alias.name}" for alias in node.names)
            for name in names:
                if name == "feedbax" or not name.startswith("feedbax."):
                    continue
                if any(part.startswith("_") for part in name.split(".")[1:]):
                    raise AssertionError(f"private Feedbax import in installed fixture: {name}")


def _require_loaded_feedbax_modules_under(root: Path) -> None:
    leaked: list[str] = []
    for name, module in tuple(sys.modules.items()):
        if name != "feedbax" and not name.startswith("feedbax."):
            continue
        source = getattr(module, "__file__", None)
        if source is not None and not Path(source).resolve().is_relative_to(root):
            leaked.append(f"{name}={Path(source).resolve()}")
    if leaked:
        raise AssertionError(f"Feedbax modules leaked outside installed wheel: {leaked!r}")


def run_fixture(*, source_root: Path | None = None) -> ConformanceResult:
    """Run bounded clean-installed cases with runner-process outbound TCP denied."""
    import feedbax_external_conformance

    resolved_source_root = source_root.resolve() if source_root is not None else None
    feedbax_root = _require_clean_install(feedbax, source_root=resolved_source_root)
    fixture_root = _require_clean_install(
        feedbax_external_conformance,
        source_root=resolved_source_root,
    )
    _require_noneditable_distribution("feedbax")
    _require_noneditable_distribution("feedbax-external-conformance")
    _require_no_private_feedbax_imports(fixture_root)
    if "feedbax_external_conformance.plugin" in sys.modules:
        raise AssertionError("fixture plugin entry point loaded before explicit discovery")

    cases = {
        "ordered_registration": check_ordered_registration(),
        "unified_plugin_bootstrap": check_unified_plugin_bootstrap(),
        "external_driver_plugin": check_external_driver_plugin(),
        "component_registration_and_migration": check_component_registration_and_migration(),
        "dynamic_component_ports": check_dynamic_component_ports(),
        "value_identity": check_value_identity(),
        "component_param_array_values": check_component_param_array_values(),
        "material_dependencies": check_material_dependencies(),
        "staged_exact_parent_migration": check_exact_parent_migration(),
        "resolved_evaluation_row_projection": check_resolved_evaluation_row_projection(),
        "public_lifecycle_recovery": check_public_lifecycle_recovery(),
        "custody_persistence_recovery": check_custody_persistence_recovery(),
    }

    _require_loaded_feedbax_modules_under(feedbax_root)
    return ConformanceResult(
        status="pass",
        feedbax_version=importlib.metadata.version("feedbax"),
        feedbax_install_root=str(feedbax_root),
        fixture_install_root=str(fixture_root),
        protocol_roles={
            "current": DOWNSTREAM_PROTOCOL_CURRENT,
            "minimum": DOWNSTREAM_PROTOCOL_MINIMUM,
        },
        cases=cases,
        lifecycle=LifecycleResult(status="pass"),
    )


__all__ = ["run_fixture"]
