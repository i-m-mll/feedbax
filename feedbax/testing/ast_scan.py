"""Reusable building blocks for repository AST conformance scans.

All paths are anchored by an explicit caller-supplied root.  Visitors own the
domain-specific detection policy; this module only supplies traversal, source
location context, and stable labels for Python expressions.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Protocol, TypeVar


@dataclass(frozen=True)
class StructuralSite:
    """Conventional, hashable location for a structurally keyed AST finding.

    ``lineno`` is diagnostic information and intentionally excluded from the
    structural key so inserting unrelated lines does not grow an allowlist.
    Subclasses can extend ``structural_key`` with detector-specific fields.
    """

    relpath: str
    qualname: str
    lineno: int

    @property
    def structural_key(self) -> tuple[str, str]:
        return (self.relpath, self.qualname)


SiteT = TypeVar("SiteT", bound=StructuralSite)


class SiteVisitor(ast.NodeVisitor, Generic[SiteT]):
    """Base visitor which tracks nested Python scopes and collected sites."""

    def __init__(self, *, relpath: str) -> None:
        self.relpath = relpath
        self.sites: list[SiteT] = []
        self._scope: list[str] = []

    @property
    def qualname(self) -> str:
        """Current dotted scope, or ``<module>`` at module level."""

        return ".".join(self._scope) if self._scope else "<module>"

    def _visit_scope(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    visit_FunctionDef = _visit_scope
    visit_AsyncFunctionDef = _visit_scope
    visit_ClassDef = _visit_scope


class VisitorFactory(Protocol[SiteT]):
    """Construct a visitor for one root-relative source path."""

    def __call__(self, *, relpath: str) -> SiteVisitor[SiteT]: ...


def expr_string(node: ast.AST | None) -> str:
    """Return one canonical, compact label for an AST expression."""

    if node is None:
        return "<none>"
    try:
        return ast.unparse(node)
    except (AttributeError, ValueError, TypeError):
        return ast.dump(node, annotate_fields=False, include_attributes=False)


def target_label(node: ast.AST | None) -> str:
    """Return a stable label for a write/call target expression.

    Literal string and path-like binary targets retain their useful expression
    text; other expressions use the same canonical rendering as ``expr_string``.
    """

    return expr_string(node)


def scan_file(
    path: Path,
    *,
    root: Path,
    visitor_factory: VisitorFactory[SiteT],
) -> list[SiteT]:
    """Parse and scan one file, requiring it to reside below ``root``."""

    path = path.resolve()
    root = root.resolve()
    relpath = path.relative_to(root).as_posix()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    visitor = visitor_factory(relpath=relpath)
    visitor.visit(tree)
    return visitor.sites


def scan_tree(
    root: Path,
    *,
    visitor_factory: VisitorFactory[SiteT],
    pattern: str = "*.py",
    exclude: Callable[[Path], bool] | None = None,
) -> list[SiteT]:
    """Scan matching files recursively beneath an explicit root."""

    files = sorted(path for path in root.rglob(pattern) if not (exclude and exclude(path)))
    return scan_domain(files, root=root, visitor_factory=visitor_factory)


def scan_domain(
    paths: Iterable[Path],
    *,
    root: Path,
    visitor_factory: VisitorFactory[SiteT],
) -> list[SiteT]:
    """Scan an explicit collection of files in deterministic path order."""

    sites: list[SiteT] = []
    for path in sorted(paths, key=lambda value: value.as_posix()):
        sites.extend(scan_file(path, root=root, visitor_factory=visitor_factory))
    return sites
