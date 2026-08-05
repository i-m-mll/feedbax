"""The graph runtime must not import the Studio, model, or task layers.

``feedbax.runtime`` is the engine underneath ``feedbax.studio``,
``feedbax.models``, and ``feedbax.tasks``. A module-scope import in the other
direction makes the layering uncuttable and pulls those packages into every
``import feedbax.runtime``. This is a ratchet: the assertions below must stay at
zero.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

FORBIDDEN_PACKAGES = ("feedbax.models", "feedbax.studio", "feedbax.tasks")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _module_scope_imports(tree: ast.Module) -> list[tuple[int, str]]:
    """Return ``(lineno, target)`` for imports that execute at module scope."""

    found: list[tuple[int, str]] = []

    def walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(child, ast.Import):
                found.extend((child.lineno, alias.name) for alias in child.names)
            elif isinstance(child, ast.ImportFrom):
                if child.module and not child.level:
                    found.append((child.lineno, child.module))
            else:
                walk(child)

    walk(tree)
    return found


def test_runtime_has_no_module_scope_imports_of_studio_models_or_tasks() -> None:
    repo_root = _repo_root()
    runtime_dir = repo_root / "feedbax" / "runtime"
    assert runtime_dir.is_dir()

    violations: list[str] = []
    for path in sorted(runtime_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for lineno, target in _module_scope_imports(tree):
            if any(
                target == package or target.startswith(f"{package}.")
                for package in FORBIDDEN_PACKAGES
            ):
                violations.append(f"{path.relative_to(repo_root)}:{lineno}: {target}")

    assert violations == []


def test_importing_runtime_does_not_load_studio_models_or_tasks() -> None:
    repo_root = _repo_root()
    env = os.environ.copy()
    pythonpath = str(repo_root)
    if env.get("PYTHONPATH"):
        pythonpath = os.pathsep.join([pythonpath, env["PYTHONPATH"]])
    env["PYTHONPATH"] = pythonpath

    source = textwrap.dedent(
        """
        import importlib
        import json
        import sys

        for module_name in [
            "feedbax.runtime",
            "feedbax.runtime.parameter_constraints",
            "feedbax.runtime.retained_observables",
            "feedbax.runtime.task_bindings",
            "feedbax.runtime.task_data_roles",
            "feedbax.runtime.timeline_masks",
        ]:
            importlib.import_module(module_name)

        leaked = sorted(
            name for name in sys.modules
            if any(
                name == package or name.startswith(package + ".")
                for package in ("feedbax.models", "feedbax.studio", "feedbax.tasks")
            )
        )
        print(json.dumps({"leaked": leaked}))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["leaked"] == []


def test_relocated_contracts_stay_importable_from_their_original_paths() -> None:
    studio_protocol = importlib.import_module("feedbax.studio.protocol")
    task_data_roles = importlib.import_module("feedbax.runtime.task_data_roles")
    task_timeline_masks = importlib.import_module("feedbax.tasks.timeline_masks")
    runtime_timeline_masks = importlib.import_module("feedbax.runtime.timeline_masks")

    for name in (
        "GRAPH_BINDABLE_TASK_DATA_ROLES",
        "PROTOCOL_TASK_DATA_KINDS",
        "PROTOCOL_TASK_DATA_PATH_PREFIXES",
        "PROTOCOL_TASK_DATA_ROLES",
        "TASK_DATA_ROLES",
        "is_bindable_task_data",
        "task_data_role",
        "task_data_surface",
        "task_data_uses_protocol_path",
    ):
        assert getattr(studio_protocol, name) is getattr(task_data_roles, name)

    for name in (
        "TaskTimelineMask",
        "TaskTimelineMaskError",
        "align_time_mask",
        "build_task_timeline_mask",
    ):
        assert getattr(task_timeline_masks, name) is getattr(runtime_timeline_masks, name)


def test_parameter_role_resolvers_are_declared_by_component_types() -> None:
    parameter_constraints = importlib.import_module("feedbax.runtime.parameter_constraints")
    components = importlib.import_module("feedbax.runtime.components")
    networks = importlib.import_module("feedbax.models.networks")

    resolvers = parameter_constraints._PARAMETER_ROLE_RESOLVERS
    assert resolvers[components.Linear] is parameter_constraints.linear_parameter_role
    assert resolvers[components.GRU] is parameter_constraints.recurrent_parameter_role
    assert resolvers[components.LSTM] is parameter_constraints.recurrent_parameter_role
    # Declared next to the network type, not hardcoded in the runtime.
    assert resolvers[networks.VanillaRNN] is parameter_constraints.recurrent_parameter_role
