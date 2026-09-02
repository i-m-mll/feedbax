from __future__ import annotations

import ast
import re
import subprocess
import sys
import textwrap
import tomllib
from dataclasses import dataclass
from importlib.metadata import packages_distributions
from pathlib import Path


IMPORT_TO_DISTRIBUTION_OVERRIDES = {
    "jax_cookbook": "jax-cookbook",
    "ruamel": "ruamel-yaml",
    "sklearn": "scikit-learn",
}

# Existing optional namespaces have many import-time dependencies. Keep this list shrink-only:
# remove entries as those modules move optional imports behind lazy loaders or ImportError guards.
OPTIONAL_NAMESPACE_IMPORT_ALLOWLIST = {
    ("feedbax.analysis", "dill"): "analysis extra module imports dill at module import time",
    ("feedbax.analysis", "plotly"): "analysis extra module imports plotly at module import time",
    ("feedbax.analysis", "pyperclip"): "analysis extra module imports pyperclip at module import time",
    ("feedbax.analysis", "scikit-learn"): "analysis extra module imports sklearn at module import time",
    ("feedbax.analysis", "sqlalchemy"): "analysis extra module imports sqlalchemy at module import time",
    ("feedbax.bin.analysis", "plotly"): "analysis CLI belongs to the analysis extra",
    ("feedbax.persistence", "alembic"): "persistence extra module imports alembic at module import time",
    ("feedbax.persistence", "matplotlib"): (
        "persistence extra module imports matplotlib at module import time"
    ),
    ("feedbax.persistence", "plotly"): "persistence extra module imports plotly at module import time",
    ("feedbax.persistence", "pyexiv2"): "persistence extra module imports pyexiv2 at module import time",
    ("feedbax.persistence", "sqlalchemy"): (
        "persistence extra module imports sqlalchemy at module import time"
    ),
    ("feedbax.plot", "matplotlib"): "viz extra module imports matplotlib at module import time",
    ("feedbax.plot", "pandas"): "viz extra module imports pandas at module import time",
    ("feedbax.plot", "plotly"): "viz extra module imports plotly at module import time",
    ("feedbax.plot", "polars"): "viz extra module imports polars at module import time",
    ("feedbax.plot", "pyexiv2"): "viz extra module imports pyexiv2 at module import time",
    ("feedbax.plot", "pyperclip"): "viz extra module imports pyperclip at module import time",
    ("feedbax.plot", "scikit-learn"): "viz extra module imports sklearn at module import time",
    ("feedbax.plot", "seaborn"): "viz extra module imports seaborn at module import time",
    ("feedbax.web", "fastapi"): "web extra module imports fastapi at module import time",
    ("feedbax.web", "httpx"): "web extra module imports httpx at module import time",
    ("feedbax.web", "sqlalchemy"): "web extra module imports sqlalchemy at module import time",
    ("feedbax.web", "starlette"): "web extra module imports starlette at module import time",
    ("feedbax.web", "uvicorn"): "web extra module imports uvicorn at module import time",
}


@dataclass(frozen=True, order=True)
class ImportFinding:
    module: str
    path: Path
    line: int
    import_name: str
    distribution: str


class UnguardedImportVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.guarded_depth = 0
        self.imports: list[tuple[int, str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_If(self, node: ast.If) -> None:
        if _is_type_checking_test(node.test):
            self.guarded_depth += 1
            for child in node.body:
                self.visit(child)
            self.guarded_depth -= 1
            for child in node.orelse:
                self.visit(child)
            return
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        if any(
            handler.type is not None and _is_import_error_type(handler.type)
            for handler in node.handlers
        ):
            self.guarded_depth += 1
            for child in node.body:
                self.visit(child)
            self.guarded_depth -= 1
            for child in [*node.handlers, *node.orelse, *node.finalbody]:
                self.visit(child)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        if self.guarded_depth:
            return
        self.imports.extend((node.lineno, alias.name.split(".", maxsplit=1)[0]) for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self.guarded_depth or node.level or node.module is None:
            return
        self.imports.append((node.lineno, node.module.split(".", maxsplit=1)[0]))


def test_unconditional_feedbax_imports_are_base_dependencies() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = _load_pyproject(repo_root)
    base_dependencies = _project_dependencies(pyproject)

    findings = _find_unapproved_imports(
        sorted((repo_root / "feedbax").rglob("*.py")),
        repo_root=repo_root,
        base_dependencies=base_dependencies,
    )

    assert findings == []


def test_worker_identity_import_does_not_require_web_runtime_dependencies() -> None:
    script = textwrap.dedent(
        """
        import importlib
        import importlib.abc
        import sys

        blocked_roots = {"fastapi", "uvicorn"}
        blocked_modules = {"feedbax.web.worker.app"}

        class BlockWebRuntimeImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.partition(".")[0] in blocked_roots or fullname in blocked_modules:
                    raise ModuleNotFoundError(fullname)
                return None

        sys.meta_path.insert(0, BlockWebRuntimeImports())
        importlib.import_module("feedbax.web.worker.identity")

        assert not blocked_modules.intersection(sys.modules)
        assert not blocked_roots.intersection(sys.modules)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_extras_only_top_level_import_canary_is_flagged() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    canary_path = repo_root / "tests/fixtures/base_dependency_imports/extras_only_top_level.py"
    assert canary_path.exists(), "the extras-only top-level import canary fixture was removed"

    findings = _find_unapproved_imports(
        [canary_path],
        repo_root=repo_root,
        base_dependencies={"feedbax"},
    )

    assert findings == [
        ImportFinding(
            module="tests.fixtures.base_dependency_imports.extras_only_top_level",
            path=canary_path.relative_to(repo_root),
            line=1,
            import_name="plotly",
            distribution="plotly",
        )
    ]


def test_ruamel_yaml_must_remain_a_base_dependency() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = _load_pyproject(repo_root)
    base_dependencies = _project_dependencies(pyproject) - {"ruamel-yaml"}

    findings = _find_unapproved_imports(
        [repo_root / "feedbax/config/config.py", repo_root / "feedbax/config/yaml.py"],
        repo_root=repo_root,
        base_dependencies=base_dependencies,
    )

    assert any(finding.distribution == "ruamel-yaml" for finding in findings)


def _find_unapproved_imports(
    paths: list[Path],
    *,
    repo_root: Path,
    base_dependencies: set[str],
) -> list[ImportFinding]:
    import_to_distributions = packages_distributions()
    stdlib_modules = set(sys.stdlib_module_names) | {"__future__"}
    findings = []

    for path in paths:
        module = _module_name(path, repo_root=repo_root)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor = UnguardedImportVisitor()
        visitor.visit(tree)

        for line, import_name in visitor.imports:
            if import_name in stdlib_modules or import_name == "feedbax":
                continue
            distribution = _distribution_name(import_name, import_to_distributions)
            if distribution in base_dependencies or _allowlist_rationale(module, distribution):
                continue
            findings.append(
                ImportFinding(
                    module=module,
                    path=path.relative_to(repo_root),
                    line=line,
                    import_name=import_name,
                    distribution=distribution,
                )
            )

    return sorted(findings)


def _load_pyproject(repo_root: Path) -> dict[str, object]:
    return tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))


def _project_dependencies(pyproject: dict[str, object]) -> set[str]:
    project = pyproject["project"]
    assert isinstance(project, dict)
    dependencies = project["dependencies"]
    assert isinstance(dependencies, list)
    return {_normalize_distribution(str(dependency)) for dependency in dependencies}


def _distribution_name(
    import_name: str,
    import_to_distributions: dict[str, list[str]],
) -> str:
    if import_name in IMPORT_TO_DISTRIBUTION_OVERRIDES:
        return _normalize_distribution(IMPORT_TO_DISTRIBUTION_OVERRIDES[import_name])
    distributions = import_to_distributions.get(import_name)
    if not distributions:
        return _normalize_distribution(import_name)
    return _normalize_distribution(distributions[0])


def _normalize_distribution(value: str) -> str:
    name = re.split(r"[<>=!~;\[\s]", value, maxsplit=1)[0]
    return name.replace("_", "-").lower()


def _module_name(path: Path, *, repo_root: Path) -> str:
    relative = path.relative_to(repo_root).with_suffix("")
    return ".".join(relative.parts)


def _is_type_checking_test(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Name)
        and node.id == "TYPE_CHECKING"
        or isinstance(node, ast.Attribute)
        and node.attr == "TYPE_CHECKING"
    )


def _is_import_error_type(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id in {"ImportError", "ModuleNotFoundError"}
    if isinstance(node, ast.Tuple):
        return any(_is_import_error_type(element) for element in node.elts)
    return False


def _allowlist_rationale(module: str, distribution: str) -> str | None:
    matches = [
        rationale
        for (module_prefix, allowed_distribution), rationale in OPTIONAL_NAMESPACE_IMPORT_ALLOWLIST.items()
        if distribution == allowed_distribution
        and (module == module_prefix or module.startswith(f"{module_prefix}."))
    ]
    assert len(matches) <= 1
    return matches[0] if matches else None
