"""Generic allowlists and shrink-only ratchets for conformance scans."""

from __future__ import annotations

import fnmatch
import json
import tomllib
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, Protocol, TypeVar


KeyT = TypeVar("KeyT", bound=Hashable)
SiteT = TypeVar("SiteT")
ScopeKind = Literal["python_scope", "file", "glob"]


class AllowlistError(AssertionError):
    """Raised when an allowlist is invalid or no longer matches a scan."""


@dataclass(frozen=True)
class Scope:
    """Location matcher for an allowlist entry."""

    kind: ScopeKind
    path: str
    qualname: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "python_scope" and not self.qualname:
            raise ValueError("python_scope requires qualname")
        if self.kind != "python_scope" and self.qualname is not None:
            raise ValueError(f"{self.kind} does not accept qualname")


@dataclass(frozen=True)
class AllowlistEntry(Generic[KeyT]):
    """One justified exception, optionally restricted to a finding key."""

    scope: Scope
    owner: str
    reason: str
    key: KeyT | None = None

    def __post_init__(self) -> None:
        if not self.owner.strip():
            raise ValueError("allowlist entry owner is required")
        if not self.reason.strip():
            raise ValueError("allowlist entry reason/rationale is required")


@dataclass(frozen=True)
class AllowlistDiff(Generic[KeyT, SiteT]):
    """Coverage result; a clean gate has neither unlisted nor dead entries."""

    unlisted: tuple[SiteT, ...]
    dead_entries: tuple[AllowlistEntry[KeyT], ...]

    @property
    def is_clean(self) -> bool:
        return not self.unlisted and not self.dead_entries

    def require_clean(self) -> None:
        if self.is_clean:
            return
        parts = []
        if self.unlisted:
            parts.append(f"unlisted findings: {list(self.unlisted)!r}")
        if self.dead_entries:
            parts.append(f"dead allowlist entries: {list(self.dead_entries)!r}")
        raise AllowlistError("; ".join(parts))


def scope_matches(scope: Scope, *, relpath: str, qualname: str | None = None) -> bool:
    """Match a location using python-scope, exact-file, or POSIX glob dispatch."""

    if scope.kind == "python_scope":
        return relpath == scope.path and qualname == scope.qualname
    if scope.kind == "file":
        return relpath == scope.path
    return fnmatch.fnmatchcase(relpath, scope.path)


def diff_allowlist(
    sites: Iterable[SiteT],
    entries: Sequence[AllowlistEntry[KeyT]],
    *,
    site_key: Callable[[SiteT], KeyT],
    site_location: Callable[[SiteT], tuple[str, str | None]],
) -> AllowlistDiff[KeyT, SiteT]:
    """Compare findings with justified entries using a caller-defined key shape."""

    site_list = list(sites)
    used: set[int] = set()
    unlisted: list[SiteT] = []
    for site in site_list:
        key = site_key(site)
        relpath, qualname = site_location(site)
        matching = [
            index
            for index, entry in enumerate(entries)
            if (entry.key is None or entry.key == key)
            and scope_matches(entry.scope, relpath=relpath, qualname=qualname)
        ]
        if not matching:
            unlisted.append(site)
        used.update(matching)
    dead = tuple(entry for index, entry in enumerate(entries) if index not in used)
    return AllowlistDiff(tuple(unlisted), dead)


def load_scoped_entries(
    path: Path,
    *,
    key_from_entry: Callable[[Mapping[str, Any]], KeyT | None] | None = None,
) -> list[AllowlistEntry[KeyT]]:
    """Load ``[[python_scope]]``, ``[[file]]``, and ``[[glob]]`` TOML entries."""

    data = tomllib.loads(path.read_text(encoding="utf-8"))
    entries: list[AllowlistEntry[KeyT]] = []
    for kind in ("python_scope", "file", "glob"):
        for raw in data.get(kind, []):
            if not isinstance(raw, dict):
                raise ValueError(f"{kind} entries must be TOML tables")
            reason = raw.get("reason", raw.get("rationale", ""))
            scope_path = raw.get("path") if kind != "glob" else raw.get("pattern")
            if not isinstance(scope_path, str):
                raise ValueError(f"{kind} entry requires {'pattern' if kind == 'glob' else 'path'}")
            key = key_from_entry(raw) if key_from_entry else None
            entries.append(
                AllowlistEntry(
                    Scope(kind, scope_path, raw.get("qualname")),
                    owner=str(raw.get("owner", "")),
                    reason=str(reason),
                    key=key,
                )
            )
    return entries


class RatchetBackend(Protocol[KeyT]):
    """Persistence adapter for a shrink-only set of keys."""

    def load(self) -> set[KeyT]: ...

    def write(self, keys: set[KeyT]) -> None: ...


@dataclass(frozen=True)
class JsonBaseline:
    """JSON-list backend for string ratchet keys."""

    path: Path

    def load(self) -> set[str]:
        if not self.path.exists():
            return set()
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"{self.path} must contain a JSON list of strings")
        return set(value)

    def write(self, keys: set[str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(sorted(keys), indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class TomlListBaseline:
    """Read-only TOML array-of-tables/list adapter with caller-defined keys.

    TOML rewriting is deliberately delegated to downstream code so comments
    and formatting are not destroyed by a generic serializer.
    """

    path: Path
    table: str
    key_from_entry: Callable[[Mapping[str, Any]], KeyT]

    def load(self) -> set[KeyT]:
        data = tomllib.loads(self.path.read_text(encoding="utf-8"))
        values = data.get(self.table, [])
        if not isinstance(values, list):
            raise ValueError(f"{self.table!r} must be a TOML list")
        return {self.key_from_entry(value) for value in values}

    def write(self, keys: set[KeyT]) -> None:
        raise NotImplementedError("generic TOML rewriting would discard comments; update downstream")


@dataclass(frozen=True)
class RatchetDiff(Generic[KeyT]):
    added: frozenset[KeyT]
    removed: frozenset[KeyT]

    @property
    def is_clean(self) -> bool:
        return not self.added and not self.removed


def compare_ratchet(current: Iterable[KeyT], baseline: Iterable[KeyT]) -> RatchetDiff[KeyT]:
    """Return growth and stale-baseline differences."""

    current_set = set(current)
    baseline_set = set(baseline)
    return RatchetDiff(frozenset(current_set - baseline_set), frozenset(baseline_set - current_set))


def enforce_ratchet(current: Iterable[KeyT], backend: RatchetBackend[KeyT]) -> RatchetDiff[KeyT]:
    """Require exact shrink-only-baseline agreement (growth and dead keys fail)."""

    diff = compare_ratchet(current, backend.load())
    if diff.added or diff.removed:
        raise AllowlistError(
            f"ratchet mismatch: added={sorted(map(repr, diff.added))}, "
            f"dead={sorted(map(repr, diff.removed))}"
        )
    return diff


def write_shrink_only(
    current: Iterable[KeyT],
    backend: RatchetBackend[KeyT],
    *,
    allow_growth: bool = False,
) -> set[KeyT]:
    """Persist current keys, refusing any set growth unless explicitly allowed."""

    current_set = set(current)
    previous = backend.load()
    added = current_set - previous
    if previous and added and not allow_growth:
        raise AllowlistError(f"refusing ratchet growth; new keys: {sorted(map(repr, added))}")
    backend.write(current_set)
    return current_set
