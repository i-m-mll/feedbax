"""Read and compare first-party files without executing project code."""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Callable, Mapping
from copy import copy
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from hashlib import file_digest, sha256
from importlib import metadata
from os import path as os_path
from pathlib import Path
import stat
import symtable
import sys
from types import ModuleType

from packaging.utils import canonicalize_name

from feedbax.contracts.code_validity import SourceDifference, SourcePath
from feedbax.validity.occurrence import FileStamp, file_stamp


@dataclass(frozen=True)
class SourceFile:
    data: bytes
    sha256: str
    modified_ns: int


class SourceCatalog:
    """A filesystem view whose cached bytes are shared by capture and comparison."""

    def __init__(self, roots: Mapping[str, Path]):
        self._files: dict[Path, tuple[FileStamp, SourceFile]] = {}
        self._digests: dict[Path, tuple[FileStamp, str]] = {}
        self.set_roots(roots)

    def set_roots(self, roots: Mapping[str, Path]) -> None:
        self.roots = dict(sorted(
            ((name, root.resolve()) for name, root in roots.items()),
            key=lambda item: -len(item[1].parts),
        ))
        self._files.clear()
        self._digests.clear()

    def locate(self, path: Path) -> SourcePath | None:
        named = Path(os_path.abspath(path))
        target = named.resolve()
        if not any(target.is_relative_to(root) for root in self.roots.values()):
            return None
        for name, root in self.roots.items():
            if named != root and named.is_relative_to(root):
                return SourcePath(scope=name, path=named.relative_to(root).as_posix())
        for name, root in self.roots.items():
            if target != root and target.is_relative_to(root):
                return SourcePath(scope=name, path=target.relative_to(root).as_posix())
        return None

    def resolve(self, source: SourcePath) -> Path | None:
        if (named := self._named(source)) is None:
            return None
        path = named.resolve()
        return path if any(path.is_relative_to(item) for item in self.roots.values()) else None

    def _named(self, source: SourcePath) -> Path | None:
        root = self.roots.get(source.scope)
        return root / source.path if root is not None else None

    def file_digest(self, source: SourcePath) -> str | None:
        path = self.resolve(source)
        if path is None:
            return None
        try:
            metadata = path.stat()
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(metadata.st_mode):
            return None
        stamp = file_stamp(metadata)
        if (cached := self._digests.get(path)) is not None and cached[0] == stamp:
            return cached[1]
        with path.open("rb") as stream:
            digest = file_digest(stream, "sha256").hexdigest()
        self._digests[path] = stamp, digest
        return digest

    def read(self, source: SourcePath) -> SourceFile | None:
        named = self._named(source)
        if named is None:
            return None
        try:
            metadata = named.stat()
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(metadata.st_mode):
            return None
        stamp = file_stamp(metadata)
        if (cached := self._files.get(named)) is not None and cached[0] == stamp:
            return cached[1]
        path = self.resolve(source)
        if path is None:
            return None
        data = path.read_bytes()
        file = SourceFile(data, sha256(data).hexdigest(), metadata.st_mtime_ns)
        self._files[named] = stamp, file
        return file

    def compare_source(
        self, path: SourcePath, before: str, read_archived: Callable[[], bytes], entered: tuple[str, ...],
    ) -> SourceDifference | None:
        """Read archived bytes only when the present content differs."""
        try:
            current = self.read(path)
        except OSError:
            return SourceDifference(path=path, before=before, after=None, status="unreadable")
        if current is None:
            return SourceDifference(path=path, before=before, after=None, status="removed")
        if current.sha256 == before:
            return None
        return compare_source(path, read_archived(), before, current, entered)


@lru_cache(maxsize=1)
def plugin_packages() -> dict[str, str]:
    """Read declared plugin package owners without importing their packages."""
    return {
        entry.value: distribution
        for entry in metadata.entry_points(group="feedbax.plugins")
        if entry.dist is not None and entry.dist.name
        and (distribution := canonicalize_name(entry.dist.name)) != "feedbax"
    }


def loaded_source_roots(owners: Mapping[str, str]) -> dict[str, Path]:
    roots = {"feedbax": Path(__file__).resolve().parents[1]}
    for package, distribution in sorted(owners.items()):
        module = sys.modules.get(package)
        if not isinstance(module, ModuleType):
            continue
        if isinstance(filename := vars(module).get("__file__"), str):
            package_roots = (Path(filename).parent,)
        else:
            package_roots = tuple(Path(path) for path in vars(module).get("__path__", ()))
        depth = package.count(".")
        for index, package_root in enumerate(package_roots):
            if depth:
                package_root = package_root.parents[depth - 1]
            suffix = f":{index}" if len(package_roots) > 1 else ""
            roots[f"plugin:{distribution}{suffix}"] = package_root.resolve()
    return roots


@lru_cache(maxsize=16)
def _catalog(project_root: Path | None) -> SourceCatalog:
    return SourceCatalog(_catalog_roots(project_root, plugin_packages()))


def _catalog_roots(project_root: Path | None, owners: Mapping[str, str]) -> dict[str, Path]:
    roots = loaded_source_roots(owners)
    if project_root is not None:
        roots["project"] = project_root.resolve()
    return roots


def source_catalog(project_root: Path | None) -> SourceCatalog:
    result = _catalog(project_root)
    roots = _catalog_roots(project_root, plugin_packages())
    if result.roots != roots:
        result.set_roots(roots)
    return result


@lru_cache(maxsize=16)
def _admission_catalog(roots: tuple[tuple[str, Path], ...]) -> SourceCatalog:
    return SourceCatalog(dict(roots))


def admission_source_catalog(project_root: Path, owners: Mapping[str, str]) -> SourceCatalog:
    """Reuse source bytes while the composed package and project roots match."""
    return _admission_catalog(tuple(_catalog_roots(project_root, owners).items()))


@lru_cache(maxsize=512)
def _parse(data: bytes) -> ast.Module:
    module = ast.parse(data)
    symtable.symtable(data, "<source>", "exec")
    for node in ast.walk(module):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(node, clean=False) is not None:
                del node.body[0]
    return module


class _Projection(ast.NodeTransformer):
    def __init__(self, entered: tuple[str, ...]):
        self.entered = frozenset(entered)
        self.scope: list[str] = []
        self.bodies: dict[str, list[str]] = defaultdict(list)

    def generic_visit(self, node: ast.AST) -> ast.AST:
        projected = copy(node)
        for name, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                setattr(projected, name, self.visit(value))
            elif isinstance(value, list):
                setattr(projected, name, [self.visit(item) if isinstance(item, ast.AST) else item for item in value])
        return projected

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self.scope.append(node.name)
        result = self.generic_visit(node)
        self.scope.pop()
        return result

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST:
        self.scope.append(node.name)
        name = ".".join(self.scope)
        if name in self.entered or any(item.startswith(name + ".") for item in self.entered):
            body = ast.Module(body=[self.visit(item) for item in node.body], type_ignores=[])
            self.bodies[name].append(ast.dump(body))
        self.scope.pop()
        result = copy(node)
        result.body = []
        return result

    visit_AsyncFunctionDef = visit_FunctionDef


def _label(node: ast.stmt) -> str:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return ast.unparse(node)
    if isinstance(node, ast.Assign):
        return ", ".join(ast.unparse(target) for target in node.targets)
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        return ast.unparse(node.target)
    return f"{type(node).__name__} statement"


@lru_cache(maxsize=128)
def _fingerprint(data: bytes, entered: tuple[str, ...]) -> tuple[tuple[tuple[str, str], ...], dict[str, list[str]]]:
    projector = _Projection(entered)
    module = projector.visit(_parse(data))
    return tuple((ast.dump(node), _label(node)) for node in module.body), dict(projector.bodies)


def compare_source(
    path: SourcePath, archived: bytes, before: str, current: SourceFile, entered: tuple[str, ...],
) -> SourceDifference | None:
    """Count syntax changes outside the bodies of unentered functions."""
    try:
        old, old_bodies = _fingerprint(archived, entered)
        new, new_bodies = _fingerprint(current.data, entered)
    except (SyntaxError, UnicodeError, ValueError):
        return SourceDifference(path=path, before=before, after=current.sha256, status="invalid")
    statements = set()
    comparison = SequenceMatcher(a=[value for value, _ in old], b=[value for value, _ in new], autojunk=False)
    for tag, old_start, old_end, new_start, new_end in comparison.get_opcodes():
        if tag != "equal":
            statements.update(label for _, label in (*old[old_start:old_end], *new[new_start:new_end]))
    functions = tuple(sorted(
        name for name in old_bodies.keys() | new_bodies.keys()
        if old_bodies.get(name) != new_bodies.get(name)
    ))
    if statements or functions:
        return SourceDifference(
            path=path, before=before, after=current.sha256, status="changed",
            statements=tuple(sorted(statements)), functions=functions,
        )
    return None
