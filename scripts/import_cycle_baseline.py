"""Static package-level import-cycle analysis and baseline maintenance for ``feedbax``.

The library currently has no enforced layering: most subpackages sit in a single
strongly connected component of the package-level import graph. A strict target
layering rule cannot precede the target architecture, so what is enforceable
today is a *ratchet*: the set of package pairs that participate in an import
cycle is recorded as a checked-in baseline, and the test suite fails when a new
cycle edge appears. The baseline may shrink freely; it may never grow silently.

Usage::

    # print the current analysis without touching the baseline
    uv run --no-sync python scripts/import_cycle_baseline.py --report

    # compare the working tree against the checked-in baseline (exit 1 on drift)
    uv run --no-sync python scripts/import_cycle_baseline.py --check

    # deliberately rewrite the baseline after removing cycle edges
    uv run --no-sync python scripts/import_cycle_baseline.py --write

Classification rules (all static, via :mod:`ast`; nothing is imported):

* **Module-scope imports count.** An import counts when it executes on import of
  the module: top level, inside module-scope ``if``/``try``/``with``/``for``
  blocks, and inside class bodies.
* **Function-local imports do not count.** Anything inside a ``def``, ``async
  def``, or ``lambda`` is deferred to call time and cannot create an import
  cycle. The codebase already dodges cycles this way file-by-file.
* **``if TYPE_CHECKING:`` bodies do not count.** They never execute at runtime.
  The ``else`` branch of such an ``if`` *does* count.
* **``try: ... except ImportError:`` imports do count.** Unlike optional
  third-party dependency probes, a guarded intra-package import still runs at
  import time and still closes a cycle.
* **Dynamic imports are invisible.** ``importlib.import_module`` and
  ``__import__`` calls are not resolved; this is a documented blind spot of a
  static analysis, not a supported way to evade the ratchet.

Package attribution: ``feedbax.<pkg>...`` belongs to package ``<pkg>`` when
``feedbax/<pkg>/`` is a package directory. Top-level modules such as
``feedbax/cli.py`` and ``feedbax/__init__.py`` belong to the synthetic root
package ``feedbax``. The root is a graph node like any other: ``feedbax`` is
mutually dependent with ``orchestration``, ``plugins``, and ``training``, and
dropping it would hide those cycles.

Reconciliation with the 2026-08-04 leanness audit (point-in-time; the figures
move as layering lanes land, ``--reconcile`` recomputes them)
------------------------------------------------------------------------------
The audit reported "1,839 classified edges, 20 packages in one SCC, 27 mutually
dependent pairs". Those three figures come from three *different*
configurations, each of which this analyzer reproduces exactly:

* **1,839 edges** = distinct ``feedbax`` -> ``feedbax`` import *statements* at
  *all* scopes (``--reconcile`` -> ``audit_all_scope_import_statements``, an
  exact match). The ratchet counts 1,622 edges from 1,608 statements: it
  excludes function-local and ``TYPE_CHECKING`` imports, which cannot close an
  import cycle, and resolves ``from p import a, b`` to whichever of ``p.a`` and
  ``p.b`` are real submodules, so one statement can yield several edges.
* **20 packages** = the SCC over *module-scope* edges with the synthetic root
  package excluded. Reproduced exactly, membership included. The ratchet's own
  figure is 24 because it keeps the root node, which pulls ``feedbax``, ``bin``,
  ``governance``, and ``integrations`` into the same component.
* **27 pairs** = mutually dependent pairs over *all-scope* edges with the root
  excluded. The ratchet's own figure is 20 (module scope, root included): seven
  audit pairs are mutual only through deferred function-local imports.

No figure is unexplained, and no configuration is silently matched.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

BASELINE_SCHEMA = "feedbax.import_cycle_baseline.v1"
"""Stable schema identity of the checked-in baseline document."""

ROOT_PACKAGE = "feedbax"
"""Synthetic package name for modules directly under ``feedbax/``."""

DEFAULT_BASELINE_PATH = Path("tests/fixtures/import_cycle/package_cycle_baseline.json")

REGENERATE_COMMAND = "uv run --no-sync python scripts/import_cycle_baseline.py --write"


@dataclass(frozen=True, order=True)
class ImportEdge:
    """One module-scope ``feedbax`` -> ``feedbax`` import statement."""

    source_package: str
    target_package: str
    source_module: str
    target_module: str
    line: int

    @property
    def is_cross_package(self) -> bool:
        return self.source_package != self.target_package

    def describe(self) -> str:
        return (
            f"{self.source_module}:{self.line} imports {self.target_module} "
            f"[{self.source_package} -> {self.target_package}]"
        )


@dataclass(frozen=True)
class PackageGraph:
    """Package-level import graph plus the edges that participate in cycles."""

    edges: tuple[ImportEdge, ...]
    packages: tuple[str, ...]
    package_edges: tuple[tuple[str, str], ...]
    components: tuple[tuple[str, ...], ...]

    @property
    def cycle_edges(self) -> tuple[tuple[str, str], ...]:
        """Package edges whose endpoints share a strongly connected component."""
        component_of = {
            package: index
            for index, component in enumerate(self.components)
            for package in component
        }
        cyclic = {index for index, component in enumerate(self.components) if len(component) > 1}
        return tuple(
            sorted(
                (source, target)
                for source, target in self.package_edges
                if component_of[source] in cyclic and component_of[source] == component_of[target]
            )
        )

    @property
    def largest_component(self) -> tuple[str, ...]:
        if not self.components:
            return ()
        return max(self.components, key=len)

    @property
    def mutual_pairs(self) -> tuple[tuple[str, str], ...]:
        package_edges = set(self.package_edges)
        return tuple(
            sorted(
                (source, target)
                for source, target in package_edges
                if source < target and (target, source) in package_edges
            )
        )

    def witnesses(self, package_edge: tuple[str, str]) -> tuple[ImportEdge, ...]:
        """Concrete import statements that produce ``package_edge``."""
        source, target = package_edge
        return tuple(
            edge
            for edge in self.edges
            if edge.source_package == source and edge.target_package == target
        )


class ModuleScopeImportVisitor(ast.NodeVisitor):
    """Collect imports that execute when the module is imported.

    Function and lambda bodies are skipped (deferred to call time); class bodies
    are visited (they execute at import time); ``if TYPE_CHECKING:`` bodies are
    skipped while their ``else`` branch is visited.
    """

    def __init__(self) -> None:
        self.imports: list[tuple[int, int, str | None, tuple[str, ...]]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        return

    def visit_If(self, node: ast.If) -> None:  # noqa: N802
        if _is_type_checking_test(node.test):
            for child in node.orelse:
                self.visit(child)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            self.imports.append((node.lineno, 0, alias.name, ()))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        names = tuple(alias.name for alias in node.names)
        self.imports.append((node.lineno, node.level, node.module, names))


class AllScopeImportVisitor(ast.NodeVisitor):
    """Collect every import statement regardless of scope.

    Only used by ``--reconcile`` to reproduce the audit's edge count; deferred
    imports cannot close an import cycle and are excluded from the ratchet.
    """

    def __init__(self) -> None:
        self.imports: list[tuple[int, int, str | None, tuple[str, ...]]] = []

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            self.imports.append((node.lineno, 0, alias.name, ()))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        self.imports.append(
            (node.lineno, node.level, node.module, tuple(alias.name for alias in node.names))
        )


def _is_type_checking_test(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "TYPE_CHECKING"
    if isinstance(node, ast.Attribute):
        return node.attr == "TYPE_CHECKING"
    return False


def source_paths(package_root: Path) -> list[Path]:
    return sorted(path for path in package_root.rglob("*.py") if path.is_file())


def module_name(path: Path, *, repo_root: Path) -> str:
    relative = path.relative_to(repo_root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) if parts else ROOT_PACKAGE


def known_modules(paths: Iterable[Path], *, repo_root: Path) -> set[str]:
    """Every importable ``feedbax`` module and package name implied by ``paths``."""
    names = {ROOT_PACKAGE}
    for path in paths:
        parts = module_name(path, repo_root=repo_root).split(".")
        for stop in range(1, len(parts) + 1):
            names.add(".".join(parts[:stop]))
    return names


def package_of(module: str, *, package_dirs: frozenset[str]) -> str:
    """Attribute a ``feedbax`` module to its top-level subpackage."""
    parts = module.split(".")
    if len(parts) < 2:
        return ROOT_PACKAGE
    return parts[1] if parts[1] in package_dirs else ROOT_PACKAGE


def _resolve_target(
    *,
    level: int,
    module: str | None,
    source_module: str,
    source_is_package: bool,
    modules: set[str],
) -> str | None:
    """Resolve an import statement to the ``feedbax`` module it depends on."""
    if level:
        base_parts = source_module.split(".")
        if not source_is_package:
            base_parts.pop()
        for _ in range(level - 1):
            if not base_parts:
                return None
            base_parts.pop()
        if not base_parts:
            return None
        base = ".".join(base_parts)
        target = f"{base}.{module}" if module else base
    else:
        if module is None:
            return None
        target = module

    if target != ROOT_PACKAGE and not target.startswith(f"{ROOT_PACKAGE}."):
        return None
    return target


def collect_edges(repo_root: Path, *, scope: str = "module") -> list[ImportEdge]:
    """Classify ``feedbax`` -> ``feedbax`` import statements.

    Args:
        repo_root: checkout root containing the ``feedbax`` package.
        scope: ``"module"`` for import-time edges only (the ratchet's semantics),
            or ``"all"`` to also count deferred function-local imports.
    """
    visitor_cls = ModuleScopeImportVisitor if scope == "module" else AllScopeImportVisitor
    package_root = repo_root / ROOT_PACKAGE
    paths = source_paths(package_root)
    modules = known_modules(paths, repo_root=repo_root)
    package_dirs = frozenset(
        name.split(".")[1] for name in modules if name.count(".") >= 1 and name != ROOT_PACKAGE
    ) & frozenset(path.name for path in package_root.iterdir() if path.is_dir())

    edges: list[ImportEdge] = []
    for path in paths:
        source_module = module_name(path, repo_root=repo_root)
        source_is_package = path.name == "__init__.py"
        source_package = package_of(source_module, package_dirs=package_dirs)

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        visitor = visitor_cls()
        visitor.visit(tree)

        for line, level, module, names in visitor.imports:
            base = _resolve_target(
                level=level,
                module=module,
                source_module=source_module,
                source_is_package=source_is_package,
                modules=modules,
            )
            if base is None:
                continue
            for target in _import_targets(base, names, modules=modules):
                edges.append(
                    ImportEdge(
                        source_package=source_package,
                        target_package=package_of(target, package_dirs=package_dirs),
                        source_module=source_module,
                        target_module=target,
                        line=line,
                    )
                )
    return sorted(edges)


def _import_targets(base: str, names: Sequence[str], *, modules: set[str]) -> Iterator[str]:
    """Yield the module(s) an import statement actually depends on.

    ``from feedbax.a import b`` depends on ``feedbax.a.b`` when that submodule
    exists and on ``feedbax.a`` otherwise (``b`` is then a plain attribute).
    ``import feedbax.a.b`` depends on ``feedbax.a.b``.
    """
    if not names:
        yield base
        return
    emitted = False
    for name in names:
        candidate = f"{base}.{name}"
        if candidate in modules:
            emitted = True
            yield candidate
    if not emitted:
        yield base


def build_graph(edges: Iterable[ImportEdge]) -> PackageGraph:
    edges = tuple(sorted(edges))
    cross = tuple(edge for edge in edges if edge.is_cross_package)
    packages = tuple(
        sorted({edge.source_package for edge in edges} | {edge.target_package for edge in edges})
    )
    package_edges = tuple(sorted({(edge.source_package, edge.target_package) for edge in cross}))
    components = tarjan_components(packages, package_edges)
    return PackageGraph(
        edges=cross,
        packages=packages,
        package_edges=package_edges,
        components=components,
    )


def tarjan_components(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
) -> tuple[tuple[str, ...], ...]:
    """Iterative Tarjan strongly-connected-components over a small digraph."""
    adjacency: dict[str, list[str]] = {node: [] for node in nodes}
    for source, target in edges:
        adjacency.setdefault(source, []).append(target)
        adjacency.setdefault(target, [])

    index_of: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    components: list[tuple[str, ...]] = []
    counter = 0

    for root in adjacency:
        if root in index_of:
            continue
        work: list[tuple[str, int]] = [(root, 0)]
        while work:
            node, child_index = work[-1]
            if child_index == 0:
                index_of[node] = lowlink[node] = counter
                counter += 1
                stack.append(node)
                on_stack.add(node)

            recursed = False
            neighbours = adjacency[node]
            while child_index < len(neighbours):
                neighbour = neighbours[child_index]
                child_index += 1
                if neighbour not in index_of:
                    work[-1] = (node, child_index)
                    work.append((neighbour, 0))
                    recursed = True
                    break
                if neighbour in on_stack:
                    lowlink[node] = min(lowlink[node], index_of[neighbour])
            if recursed:
                continue

            work[-1] = (node, child_index)
            work.pop()
            if lowlink[node] == index_of[node]:
                component: list[str] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                components.append(tuple(sorted(component)))
            if work:
                parent, _ = work[-1]
                lowlink[parent] = min(lowlink[parent], lowlink[node])

    return tuple(sorted(components))


def summarize(graph: PackageGraph, *, all_edges: Sequence[ImportEdge]) -> dict[str, object]:
    largest = graph.largest_component
    return {
        "classified_import_edges": len(all_edges),
        "cross_package_import_edges": len(graph.edges),
        "packages": len(graph.packages),
        "package_edges": len(graph.package_edges),
        "packages_in_largest_cycle": len(largest) if len(largest) > 1 else 0,
        "mutually_dependent_package_pairs": len(graph.mutual_pairs),
        "cycle_package_edges": len(graph.cycle_edges),
        "largest_cycle_packages": list(largest) if len(largest) > 1 else [],
    }


def analyze(repo_root: Path) -> tuple[PackageGraph, dict[str, object]]:
    all_edges = collect_edges(repo_root)
    graph = build_graph(all_edges)
    return graph, summarize(graph, all_edges=all_edges)


def _statements(edges: Iterable[ImportEdge]) -> set[tuple[str, int]]:
    """Distinct ``feedbax`` -> ``feedbax`` import *statements* behind ``edges``."""
    return {(edge.source_module, edge.line) for edge in edges}


def _drop_root(edges: Iterable[ImportEdge]) -> list[ImportEdge]:
    return [
        edge
        for edge in edges
        if edge.source_package != ROOT_PACKAGE and edge.target_package != ROOT_PACKAGE
    ]


def reconcile(repo_root: Path) -> dict[str, object]:
    """Reproduce the alternative configurations used by the 2026-08-04 audit.

    The audit reported three figures under three different configurations. This
    recomputes each of them so the ratchet's own numbers can be compared against
    them without guesswork. See the module docstring for the reconciliation.
    """
    module_scope = collect_edges(repo_root, scope="module")
    all_scope = collect_edges(repo_root, scope="all")
    module_no_root = build_graph(_drop_root(module_scope))
    all_no_root = build_graph(_drop_root(all_scope))
    return {
        "ratchet_module_scope_edges": len(module_scope),
        "ratchet_module_scope_import_statements": len(_statements(module_scope)),
        "audit_all_scope_import_statements": len(_statements(all_scope)),
        "audit_all_scope_edges": len(all_scope),
        "audit_packages_in_largest_cycle_module_scope_no_root": len(
            module_no_root.largest_component
        ),
        "audit_largest_cycle_packages_module_scope_no_root": list(
            module_no_root.largest_component
        ),
        "audit_mutual_pairs_all_scope_no_root": len(all_no_root.mutual_pairs),
        "audit_mutual_pairs_module_scope_no_root": len(module_no_root.mutual_pairs),
    }


def baseline_document(graph: PackageGraph, summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": BASELINE_SCHEMA,
        "description": (
            "Upper bound on package-level import-cycle edges. This set may shrink, "
            f"never grow. Regenerate deliberately with: {REGENERATE_COMMAND}"
        ),
        "summary": dict(summary),
        "cycle_edges": [list(edge) for edge in graph.cycle_edges],
    }


def load_baseline(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    schema = document.get("schema")
    if schema != BASELINE_SCHEMA:
        raise ValueError(
            f"{path}: unsupported import-cycle baseline schema {schema!r}; "
            f"expected {BASELINE_SCHEMA!r}. Regenerate with: {REGENERATE_COMMAND}"
        )
    return document


def baseline_edges(document: Mapping[str, object]) -> set[tuple[str, str]]:
    raw = document.get("cycle_edges")
    if not isinstance(raw, list):
        raise ValueError("import-cycle baseline document has no 'cycle_edges' list")
    return {(str(entry[0]), str(entry[1])) for entry in raw}


def new_cycle_edges(
    graph: PackageGraph,
    baseline: set[tuple[str, str]],
) -> list[tuple[str, str]]:
    return sorted(set(graph.cycle_edges) - baseline)


def removed_cycle_edges(
    graph: PackageGraph,
    baseline: set[tuple[str, str]],
) -> list[tuple[str, str]]:
    return sorted(baseline - set(graph.cycle_edges))


def format_violations(graph: PackageGraph, added: Sequence[tuple[str, str]]) -> str:
    lines = [
        f"{len(added)} new package-level import cycle edge(s) are not in the baseline.",
        "",
        "The import-cycle ratchet allows the baseline to shrink, never grow. Either",
        "route the new dependency so it does not close a package cycle, or, if this",
        "lane deliberately reshapes the cycle set, regenerate the baseline in the same",
        f"commit with:\n    {REGENERATE_COMMAND}",
        "",
    ]
    for package_edge in added:
        source, target = package_edge
        lines.append(f"  NEW CYCLE EDGE  {source} -> {target}")
        witnesses = graph.witnesses(package_edge)
        for witness in witnesses[:10]:
            lines.append(f"      {witness.describe()}")
        if len(witnesses) > 10:
            lines.append(f"      ... and {len(witnesses) - 10} further import site(s)")
        lines.append("")
    return "\n".join(lines)


def format_report(graph: PackageGraph, summary: Mapping[str, object]) -> str:
    lines = ["Feedbax package-level import analysis", ""]
    for key, value in summary.items():
        if key == "largest_cycle_packages":
            continue
        lines.append(f"  {key}: {value}")
    largest = summary.get("largest_cycle_packages")
    if isinstance(largest, list) and largest:
        lines.append("")
        lines.append("  largest cycle packages:")
        lines.append(f"    {', '.join(str(name) for name in largest)}")
    lines.append("")
    lines.append("  mutually dependent package pairs:")
    for source, target in graph.mutual_pairs:
        lines.append(f"    {source} <-> {target}")
    return "\n".join(lines)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--report", action="store_true", help="print the current analysis")
    mode.add_argument("--check", action="store_true", help="compare against the baseline")
    mode.add_argument("--write", action="store_true", help="rewrite the baseline deliberately")
    mode.add_argument(
        "--reconcile",
        action="store_true",
        help="recompute the alternative configurations used by the 2026-08-04 audit",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help=f"baseline path (default: {DEFAULT_BASELINE_PATH})",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    baseline_path = args.baseline or (repo_root / DEFAULT_BASELINE_PATH)

    if args.reconcile:
        print(json.dumps(reconcile(repo_root), indent=2))
        return 0

    graph, summary = analyze(repo_root)

    if args.write:
        document = baseline_document(graph, summary)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {baseline_path} ({len(graph.cycle_edges)} cycle edges)")
        return 0

    if args.check:
        baseline = baseline_edges(load_baseline(baseline_path))
        added = new_cycle_edges(graph, baseline)
        removed = removed_cycle_edges(graph, baseline)
        if removed:
            print(f"{len(removed)} baseline cycle edge(s) are gone; regenerate to lock the win:")
            for source, target in removed:
                print(f"  REMOVED  {source} -> {target}")
            print(f"    {REGENERATE_COMMAND}")
        if added:
            print(format_violations(graph, added), file=sys.stderr)
            return 1
        print(f"import-cycle ratchet holds ({len(graph.cycle_edges)} cycle edges <= baseline)")
        return 0

    if args.json:
        print(json.dumps(baseline_document(graph, summary), indent=2))
    else:
        print(format_report(graph, summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
