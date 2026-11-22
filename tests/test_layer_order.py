"""The declared layer order of ``feedbax``.

Every module belongs to the layer of the longest prefix in ``LAYERS`` that names it. A module may import
its own layer and any layer below it, at any scope (module, function, class, ``TYPE_CHECKING``), and
nothing above it. Since no layer imports upward, the layers cannot form an import cycle.

``UPWARD`` lists each import that does not yet satisfy the order, with its reason. It may only shrink:
an import that now satisfies the order must be removed from it.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LAYERS: tuple[tuple[str, ...], ...] = (
    (
        # The root facade's exports are lazy, so it imports nothing at load.
        "feedbax", "feedbax._atomic", "feedbax._distribution_provenance", "feedbax._immutable_snapshot",
        "feedbax._interpolation", "feedbax._ode", "feedbax._secure_fs", "feedbax._vendor",
        "feedbax.checkpoint_coordinates", "feedbax.plugins", "feedbax.registry_errors", "feedbax.revision",
        # The Host bootstrap loads these from the installed files without the rest of Feedbax.
        "feedbax.host.io", "feedbax.host.jax_cache", "feedbax.host.launcher", "feedbax.host.process",
        "feedbax.host.protocol", "feedbax.host.python", "feedbax.host.source_roster", "feedbax.host.watchdog",
    ),
    ("feedbax.contracts",),
    ("feedbax.expression_builtins",),
    ("feedbax.dataset", "feedbax.dataset_replicates", "feedbax.dataset_sweep"),
    ("feedbax.runtime",),
    ("feedbax.expression_compile",),
    ("feedbax.builtins.runtime",),
    ("feedbax.component_registry",),
    ("feedbax.integrations",),
    ("feedbax.acausal",),
    ("feedbax.mechanics",),
    ("feedbax.builtins.acausal",),
    ("feedbax.builtins.mechanics",),
    ("feedbax.intervene",),
    ("feedbax.builtins.control",),
    ("feedbax.builtins.components",),
    ("feedbax.compiler",),
    ("feedbax.builtins.models",),
    ("feedbax.builtins.templates",),
    ("feedbax.mechanics.templates",),
    ("feedbax.models",),
    ("feedbax.tasks",),
    ("feedbax.objectives",),
    ("feedbax.execution",),
    ("feedbax.training",),
    ("feedbax.analysis",),
    ("feedbax.plot",),
    ("feedbax.builtins",),
    ("feedbax.lowering", "feedbax.pca"),
    ("feedbax.application",),
    ("feedbax.planner",),
    ("feedbax.config",),
    ("feedbax.persistence",),
    ("feedbax.validity",),
    ("feedbax.host",),
    ("feedbax.studio",),
    ("feedbax.project_folder", "feedbax.project_members", "feedbax.project_model_selection"),
    ("feedbax.jobs", "feedbax.project_retention"),
    ("feedbax.orchestration",),
    ("feedbax.web",),
    ("feedbax.bin", "feedbax.desktop", "feedbax.examples", "feedbax.governance", "feedbax.testing"),
    ("feedbax.__main__", "feedbax.cli"),
)

UPWARD: dict[tuple[str, str], str] = {
    ("feedbax/builtins/mechanics/direct_force_components.py", "feedbax.builtins.templates.mechanics_templates"): (
        "PointMass's default template is the point-mass template graph, which is built from PointMass"
    ),
    ("feedbax/compiler/serialization.py", "feedbax.models.feedback"): (
        "serialization recognizes the canonical feedback graph to expose its state view, which models defines"
    ),
    ("feedbax/component_registry/registry.py", "feedbax.compiler.acausal_compiler"): (
        "the registry checks an acausal template graph by compiling it"
    ),
    ("feedbax/component_registry/registry.py", "feedbax.compiler.normalization"): (
        "a component definition is stamped with versions by the compiler's normalization"
    ),
    ("feedbax/contracts/component.py", "feedbax.component_registry"): (
        "the v3 component-definition migration resolves historical metadata through the discovered registry"
    ),
    ("feedbax/contracts/model_input.py", "feedbax.tasks.registry"): (
        "reading a stored Python task input resolves its selection through the task registry it is given"
    ),
    ("feedbax/contracts/model_values.py", "feedbax.component_registry"): (
        "model owner facts are read from the component registry passed in (annotation only)"
    ),
    ("feedbax/contracts/rollout.py", "feedbax.compiler.graph"): (
        "rollout semantics take the compiled graph key schedule (annotation only)"
    ),
    ("feedbax/contracts/train_settings.py", "feedbax.training.methods"): (
        "Train settings are normalized against the training method registry passed in (annotation only)"
    ),
    ("feedbax/jobs.py", "feedbax.orchestration.session"): (
        "the job service hands remote jobs to execution sessions, whose job adapters import the job service"
    ),
    ("feedbax/jobs.py", "feedbax.orchestration.targets"): (
        "the job service resolves the execution target a job names; same cycle as orchestration.session"
    ),
}


def _module(path: Path) -> str:
    parts = path.relative_to(ROOT).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _imports(path: Path) -> list[str]:
    module = _module(path)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join([*anchor, *([node.module] if node.module else [])])
            found.extend(f"{base}.{alias.name}" for alias in node.names)
    return [_resolve(name) for name in found if name.startswith("feedbax.")]


def _resolve(name: str) -> str:
    """The module an imported name lives in: an imported attribute resolves to its module."""
    while name not in MODULES:
        name = name.rpartition(".")[0]
    return name


def _rank(module: str) -> tuple[int, str] | None:
    best = None
    for rank, layer in enumerate(LAYERS):
        for prefix in layer:
            if (module == prefix or (prefix != "feedbax" and module.startswith(prefix + "."))) and (best is None or len(prefix) > len(best[1])):
                best = (rank, prefix)
    return best


def _sources() -> list[Path]:
    return [path for path in sorted((ROOT / "feedbax").rglob("*.py")) if "_vendor" not in path.parts]


MODULES = frozenset(_module(path) for path in sorted((ROOT / "feedbax").rglob("*.py")))


def test_every_module_has_a_layer() -> None:
    assert [_module(path) for path in _sources() if _rank(_module(path)) is None] == []


def test_every_layer_prefix_names_a_module() -> None:
    assert [prefix for layer in LAYERS for prefix in layer if prefix not in MODULES] == []


def test_no_module_imports_a_higher_layer() -> None:
    upward = {}
    for path in _sources():
        source = _rank(_module(path))
        assert source is not None
        for target in _imports(path):
            rank = _rank(target)
            if rank is not None and rank[0] > source[0]:
                upward[(path.relative_to(ROOT).as_posix(), target)] = f"{source[1]} imports {rank[1]}"
    assert sorted(set(upward) - set(UPWARD)) == [], "\n".join(
        f"{path} imports {target} ({upward[(path, target)]})" for path, target in sorted(set(upward) - set(UPWARD))
    )
    assert sorted(set(UPWARD) - set(upward)) == [], "these imports now satisfy the order; remove them from UPWARD"
