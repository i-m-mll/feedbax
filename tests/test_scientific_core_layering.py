"""Complete direct-import law for the scientific-core package boundary."""

from __future__ import annotations

import ast
from pathlib import Path


SCIENTIFIC_CORE_PACKAGES = (
    "acausal",
    "compiler",
    "component_registry",
    "components",
    "control",
    "execution",
    "intervene",
    "mechanics",
    "models",
    "objectives",
    "runtime",
    "tasks",
    "training",
    "workflow",
)

FORBIDDEN_IMPORTS = {
    "studio": ("feedbax.studio", "feedbax.web"),
    "controller": ("feedbax.orchestration.controller",),
    "provider_adapter": (
        "feedbax.integrations",
        "feedbax.orchestration.drivers",
    ),
    "persistence": ("feedbax.persistence",),
    "authoring": ("feedbax.bin", "feedbax.config", "feedbax.envelope"),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _absolute_imports(path: Path) -> tuple[tuple[int, str], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, name.name) for name in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.append((node.lineno, node.module))
    return tuple(imports)


def test_scientific_core_has_no_platform_or_authoring_imports() -> None:
    repo_root = _repo_root()
    violations: list[str] = []
    for package in SCIENTIFIC_CORE_PACKAGES:
        package_root = repo_root / "feedbax" / package
        assert package_root.is_dir(), package
        for path in sorted(package_root.rglob("*.py")):
            for lineno, target in _absolute_imports(path):
                for boundary, forbidden in FORBIDDEN_IMPORTS.items():
                    if any(
                        target == prefix or target.startswith(f"{prefix}.") for prefix in forbidden
                    ):
                        violations.append(
                            f"{path.relative_to(repo_root)}:{lineno}: {boundary}: {target}"
                        )

    assert violations == []
