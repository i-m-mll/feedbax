"""Deny-by-default science-surface checker for downstream project repositories.

A project repository downstream of Feedbax is not allowed to accumulate
production Python at will.  An owner-ratified policy file names every permitted
production source path and the top-level symbols each of those paths may define,
plus path patterns that are banned forever.  Anything else fails closed.

The gate is deliberately Feedbax-owned so a project cannot weaken it, and the
policy text is resolved from a protected baseline ref with
``git show <baseline-ref>:<policy-path>``.  The working tree's own copy of the
policy is never consulted for authorization: a branch cannot authorize itself by
editing the file it is judged against.

Scanning is purely static (:mod:`ast`); project code is never imported.

Policy file format (TOML)::

    schema_version = 1

    # Production source roots, relative to the project root.  Every ``*.py``
    # file below these roots must be allowlisted.
    source_roots = ["src"]

    # Path patterns ratified as forbidden forever, in pathlib glob syntax
    # relative to the project root ("**" is recursive).  Any existing file or
    # directory matching one of these is a failure.
    banned_paths = ["src/<project>/engine", "src/<project>/engine/**"]

    [[allowed_file]]
    path = "src/<project>/__init__.py"
    symbols = ["__version__"]
    reason = "package marker"          # optional, free text

    [[allowed_file]]
    path = "src/<project>/study.py"
    symbols = ["STUDY_ID", "build_study"]

Top-level symbols are the names bound at module scope by ``def``, ``async def``,
``class``, and assignment statements, including those nested inside module-level
``if``/``try``/``with``/loop bodies.  Names bound by ``import`` statements are
not counted: an import binds a name that some other package already owns, and
requiring every ``from pathlib import Path`` to be ratified would bury the
science surface in noise.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


POLICY_SCHEMA_VERSION = 1
"""The only ``schema_version`` this checker accepts; unknown versions fail closed."""

_SKIP_DIR_NAMES = frozenset({".git", ".venv", "venv", "node_modules", "__pycache__"})

_BODY_FIELDS = ("body", "orelse", "finalbody")

_COMPOUND_STATEMENTS = (
    ast.If,
    ast.Try,
    ast.TryStar,
    ast.While,
    ast.For,
    ast.AsyncFor,
    ast.With,
    ast.AsyncWith,
)


class ScienceSurfacePolicyError(Exception):
    """Raised when the baseline policy is missing, unreadable, or unsupported."""


@dataclass(frozen=True)
class AllowedFile:
    """One ratified production source file and the symbols it may define."""

    path: str
    symbols: frozenset[str]
    reason: str = ""


@dataclass(frozen=True)
class ScienceSurfacePolicy:
    """Parsed, version-checked policy document."""

    schema_version: int
    source_roots: tuple[str, ...]
    banned_paths: tuple[str, ...]
    allowed_files: tuple[AllowedFile, ...]

    def allowed_file(self, relpath: str) -> AllowedFile | None:
        for entry in self.allowed_files:
            if entry.path == relpath:
                return entry
        return None


@dataclass(frozen=True)
class SurfaceViolation:
    """One offending path, or one offending symbol within a path."""

    kind: str
    path: str
    detail: str
    symbol: str | None = None

    @property
    def sort_key(self) -> tuple[int, str, str]:
        order = {
            "banned-path": 0,
            "unlisted-file": 1,
            "unparseable-file": 2,
            "unlisted-symbol": 3,
        }
        return (order.get(self.kind, 9), self.path, self.symbol or "")

    def render(self) -> str:
        located = f"{self.path}::{self.symbol}" if self.symbol else self.path
        return f"  {self.kind} {located}: {self.detail}"


@dataclass(frozen=True)
class ScienceSurfaceReport:
    """Outcome of one science-surface scan."""

    root: Path
    policy_path: str
    baseline_ref: str
    scanned_files: tuple[str, ...]
    violations: tuple[SurfaceViolation, ...]

    @property
    def is_clean(self) -> bool:
        return not self.violations

    def render(self) -> str:
        if self.is_clean:
            return (
                f"Project science-surface check passed: {len(self.scanned_files)} production "
                f"source file(s) under {self.root} match policy {self.policy_path} "
                f"@ {self.baseline_ref}."
            )
        lines = [
            "Project science-surface check FAILED.",
            f"  project root:  {self.root}",
            f"  policy:        {self.policy_path} @ {self.baseline_ref}",
            f"  violations:    {len(self.violations)}",
            "",
        ]
        lines.extend(violation.render() for violation in self.violations)
        lines.extend(
            (
                "",
                "This repository is deny-by-default for production Python: only the source"
                " paths and top-level symbols ratified in the baseline policy may exist.",
                "The correct home for new machinery is feedbax, or a prior protected-branch"
                " ratification of this policy.",
                f"This branch cannot authorize itself by editing {self.policy_path}:"
                f" the checker reads the policy only from {self.baseline_ref}.",
            )
        )
        return "\n".join(lines)


def read_baseline_policy_text(root: Path, policy_path: str, baseline_ref: str) -> str:
    """Return the policy text as it exists on ``baseline_ref``, never on the branch."""

    completed = subprocess.run(
        ["git", "--no-optional-locks", "show", f"{baseline_ref}:{policy_path}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ScienceSurfacePolicyError(
            f"Project science-surface check FAILED: the baseline ref {baseline_ref!r} has no "
            f"ratified science-surface policy at {policy_path!r} ({detail}). This repository is "
            "deny-by-default for production Python, and a branch cannot authorize itself by "
            "adding or editing that file: the policy must be ratified on the protected branch "
            "first."
        )
    return completed.stdout


def _require_string_list(data: Any, field: str, *, required: bool) -> tuple[str, ...]:
    if field not in data:
        if required:
            raise ScienceSurfacePolicyError(
                f"science-surface policy is missing required {field!r} (a list of strings)"
            )
        return ()
    value = data[field]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ScienceSurfacePolicyError(f"science-surface policy {field!r} must be a list of strings")
    return tuple(value)


def parse_policy(text: str) -> ScienceSurfacePolicy:
    """Parse and version-check policy text, failing closed on anything unexpected."""

    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ScienceSurfacePolicyError(f"science-surface policy is not valid TOML: {exc}") from exc

    if "schema_version" not in data:
        raise ScienceSurfacePolicyError(
            "science-surface policy has no 'schema_version'; an unversioned policy is rejected. "
            f"Ratify a policy declaring schema_version = {POLICY_SCHEMA_VERSION}."
        )
    schema_version = data["schema_version"]
    if schema_version != POLICY_SCHEMA_VERSION:
        raise ScienceSurfacePolicyError(
            f"science-surface policy schema_version {schema_version!r} is not supported; this "
            f"feedbax checker supports schema_version {POLICY_SCHEMA_VERSION} only."
        )

    source_roots = _require_string_list(data, "source_roots", required=True)
    if not source_roots:
        raise ScienceSurfacePolicyError(
            "science-surface policy 'source_roots' must name at least one production source root"
        )
    banned_paths = _require_string_list(data, "banned_paths", required=False)

    raw_files = data.get("allowed_file", [])
    if not isinstance(raw_files, list):
        raise ScienceSurfacePolicyError("science-surface policy 'allowed_file' must be a TOML array")
    allowed: list[AllowedFile] = []
    seen: set[str] = set()
    for raw in raw_files:
        if not isinstance(raw, dict):
            raise ScienceSurfacePolicyError("each [[allowed_file]] entry must be a TOML table")
        path = raw.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ScienceSurfacePolicyError("each [[allowed_file]] entry requires a 'path' string")
        if path in seen:
            raise ScienceSurfacePolicyError(f"duplicate [[allowed_file]] entry for {path!r}")
        seen.add(path)
        symbols = raw.get("symbols")
        if not isinstance(symbols, list) or not all(isinstance(item, str) for item in symbols):
            raise ScienceSurfacePolicyError(
                f"[[allowed_file]] {path!r} requires 'symbols' as a list of strings "
                "(use an empty list for a file that defines nothing)"
            )
        allowed.append(
            AllowedFile(path=path, symbols=frozenset(symbols), reason=str(raw.get("reason", "")))
        )

    return ScienceSurfacePolicy(
        schema_version=POLICY_SCHEMA_VERSION,
        source_roots=source_roots,
        banned_paths=banned_paths,
        allowed_files=tuple(allowed),
    )


def _bound_names(target: ast.expr) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for element in target.elts for name in _bound_names(element)]
    if isinstance(target, ast.Starred):
        return _bound_names(target.value)
    return []


def module_symbols(tree: ast.Module) -> tuple[str, ...]:
    """Return the sorted names bound at module scope by def/class/assignment."""

    names: set[str] = set()

    def walk(body: Iterable[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    names.update(_bound_names(target))
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                names.update(_bound_names(node.target))
            elif isinstance(node, _COMPOUND_STATEMENTS):
                for field in _BODY_FIELDS:
                    walk(getattr(node, field, ()) or ())
                for handler in getattr(node, "handlers", ()) or ():
                    walk(handler.body)

    walk(tree.body)
    return tuple(sorted(names))


def _is_skipped(path: Path, root: Path) -> bool:
    return any(part in _SKIP_DIR_NAMES for part in path.relative_to(root).parts)


def iter_production_files(root: Path, policy: ScienceSurfacePolicy) -> list[Path]:
    """Return every ``*.py`` file below the policy's declared source roots."""

    files: list[Path] = []
    for source_root in policy.source_roots:
        base = root / source_root
        if not base.is_dir():
            continue
        files.extend(path for path in base.rglob("*.py") if not _is_skipped(path, root))
    return sorted(set(files))


def _banned_violations(root: Path, policy: ScienceSurfacePolicy) -> list[SurfaceViolation]:
    violations: list[SurfaceViolation] = []
    reported: set[str] = set()
    for pattern in policy.banned_paths:
        try:
            matches = sorted(root.glob(pattern))
        except (ValueError, IndexError) as exc:
            raise ScienceSurfacePolicyError(
                f"science-surface policy banned_paths pattern {pattern!r} is not a valid glob: {exc}"
            ) from exc
        for match in matches:
            if _is_skipped(match, root) or match.relative_to(root).as_posix() in reported:
                continue
            reported.add(match.relative_to(root).as_posix())
            violations.append(
                SurfaceViolation(
                    kind="banned-path",
                    path=match.relative_to(root).as_posix(),
                    detail=f"matches ratified banned pattern {pattern!r}",
                )
            )
    return violations


def check_project_science_surface(
    *,
    root: Path,
    policy_path: str,
    baseline_ref: str,
) -> ScienceSurfaceReport:
    """Scan a project checkout against the policy ratified on ``baseline_ref``."""

    root = root.resolve()
    policy = parse_policy(read_baseline_policy_text(root, policy_path, baseline_ref))

    violations: list[SurfaceViolation] = _banned_violations(root, policy)
    scanned: list[str] = []
    for path in iter_production_files(root, policy):
        relpath = path.relative_to(root).as_posix()
        scanned.append(relpath)
        entry = policy.allowed_file(relpath)
        if entry is None:
            violations.append(
                SurfaceViolation(
                    kind="unlisted-file",
                    path=relpath,
                    detail="production source file is not in the ratified allowlist",
                )
            )
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relpath)
        except (SyntaxError, UnicodeDecodeError, ValueError) as exc:
            violations.append(
                SurfaceViolation(
                    kind="unparseable-file",
                    path=relpath,
                    detail=f"could not be parsed as Python ({exc})",
                )
            )
            continue
        for symbol in module_symbols(tree):
            if symbol not in entry.symbols:
                violations.append(
                    SurfaceViolation(
                        kind="unlisted-symbol",
                        path=relpath,
                        symbol=symbol,
                        detail="top-level symbol is not ratified for this file",
                    )
                )

    return ScienceSurfaceReport(
        root=root,
        policy_path=policy_path,
        baseline_ref=baseline_ref,
        scanned_files=tuple(scanned),
        violations=tuple(sorted(violations, key=lambda item: item.sort_key)),
    )


def build_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the checker's arguments to an existing parser."""

    parser.add_argument(
        "--root",
        default=".",
        help="Project repository root to scan (default: current directory).",
    )
    parser.add_argument(
        "--policy",
        required=True,
        help="Policy TOML path, relative to the project root.",
    )
    parser.add_argument(
        "--baseline-ref",
        required=True,
        help="Protected ref the ratified policy is read from (the branch copy is ignored).",
    )
    return parser


def run_cli(*, root: str, policy: str, baseline_ref: str) -> int:
    """Run one check and print its report; 0 clean, 1 on any failure."""

    try:
        report = check_project_science_surface(
            root=Path(root), policy_path=policy, baseline_ref=baseline_ref
        )
    except ScienceSurfacePolicyError as exc:
        print(str(exc))
        return 1
    print(report.render())
    return 0 if report.is_clean else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser(
        argparse.ArgumentParser(
            prog="python -m feedbax check-project-science-surface",
            description="Deny-by-default production-Python gate for downstream project repos.",
        )
    )
    args = parser.parse_args(argv)
    return run_cli(root=args.root, policy=args.policy, baseline_ref=args.baseline_ref)


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    raise SystemExit(main())
