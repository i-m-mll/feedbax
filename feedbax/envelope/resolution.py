"""Content-pinned document lineage and inherited-value lookup.

Two envelope rules need to know what a path *already* means before a delta
applies: checked assertions, and the refusal of any authored leaf that merely
restates its inherited value. Both are answered here, against an ordered,
content-pinned lineage rather than against a merged view, so every answer can
name the document that owns it.

Within a document a dotted path resolves either by direct traversal or through an
authored patch list, because inherited facts are expressed in both forms: a base
states a value outright, while a delta document states it as an operation. The
patch forms recognized are ``add`` and ``replace`` entries carrying ``path`` and
``value``, which is the shape Feedbax's own composition deltas already use.

This is document lineage, which is a different question from the execution
lineage DAG in :mod:`feedbax.contracts.lineage`: that one records what ran and
what it descended from, while this one records what was *read* at compile time
and pins the bytes that were read.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from feedbax.contracts.authored_canonical import (
    CANONICAL_PIN_ALGORITHM,
    canonical_sha256,
)

_MISSING = object()

#: The chain keys a lineage walk follows out of one document, in priority order.
#: A document states where it came from with one of these blocks, each carrying a
#: ``ref``; the first one present wins, so a document that states both does not
#: fork the walk.
DEFAULT_CHAIN_KEYS: tuple[str, ...] = ("base", "parent", "parent_lock")


@dataclass(frozen=True)
class PinnedDocument:
    """One content-pinned document consulted while resolving inherited values.

    Attributes:
        ref: How the document was named, normally a repo-relative path.
        content_hash: The canonical hash of the bytes that were read.
        document: The parsed document.
        patch_scope_skips: Keys whose subtrees are not searched for patch
            bindings, so a collection of sibling records cannot leak one
            record's overrides into the whole document's inherited view.
    """

    ref: str
    content_hash: str
    document: Mapping[str, Any]
    patch_scope_skips: tuple[str, ...] = ()

    @classmethod
    def of(
        cls, ref: str, document: Mapping[str, Any], *, skips: tuple[str, ...] = ()
    ) -> "PinnedDocument":
        """Pin an already-parsed document under the canonical hash domain."""
        return cls(ref, canonical_sha256(document), document, skips)

    def pin(self) -> dict[str, str]:
        """Return this document's compile-lock pin record."""
        return {
            "ref": self.ref,
            "content_hash": self.content_hash,
            "pin_algorithm": CANONICAL_PIN_ALGORITHM,
        }


@dataclass(frozen=True)
class InheritedValue:
    """One inherited value and the document that owns it."""

    value: Any
    owner_ref: str


def load_pinned(
    repo_root: Path, ref: str, *, skips: tuple[str, ...] = ()
) -> PinnedDocument | None:
    """Load and content-pin one repo-relative JSON document, if it exists.

    Returns ``None`` rather than raising when the reference does not name a
    readable JSON object, because a lineage walk ends at the first link it cannot
    follow; a *required* parent is resolved by the caller, which knows whether an
    absent document is a rejection.
    """
    if not ref.endswith(".json"):
        return None
    path = repo_root / ref
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if not isinstance(document, dict):
        return None
    return PinnedDocument.of(ref, document, skips=skips)


def _traverse(document: Any, path: str) -> Any:
    current: Any = document
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _patch_bindings(node: Any, skips: tuple[str, ...], bindings: dict[str, Any]) -> None:
    if isinstance(node, Mapping):
        patches = node.get("patches")
        if isinstance(patches, Sequence) and not isinstance(patches, (str, bytes)):
            for patch in patches:
                if (
                    isinstance(patch, Mapping)
                    and isinstance(patch.get("path"), str)
                    and patch.get("op") in ("add", "replace")
                    and patch["path"] not in bindings
                ):
                    bindings[patch["path"]] = patch["value"]
        for key, value in node.items():
            if key not in skips:
                _patch_bindings(value, skips, bindings)
    elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        for item in node:
            _patch_bindings(item, skips, bindings)


@dataclass(frozen=True)
class Lineage:
    """Ordered, content-pinned documents supplying a parent's inherited values."""

    documents: tuple[PinnedDocument, ...]

    def lookup(self, path: str) -> InheritedValue | None:
        """Return the most specific inherited value for ``path``, or ``None``.

        The search is ordered: the nearest document that states the path owns it,
        whether it states the path directly, binds it in a patch list, or binds a
        prefix of it whose value contains the rest.
        """
        for pinned in self.documents:
            direct = _traverse(pinned.document, path)
            if direct is not _MISSING:
                return InheritedValue(direct, pinned.ref)
            bindings: dict[str, Any] = {}
            _patch_bindings(pinned.document, pinned.patch_scope_skips, bindings)
            if path in bindings:
                return InheritedValue(bindings[path], pinned.ref)
            for bound_path, bound_value in bindings.items():
                if path.startswith(f"{bound_path}."):
                    nested = _traverse(bound_value, path[len(bound_path) + 1 :])
                    if nested is not _MISSING:
                        return InheritedValue(nested, pinned.ref)
        return None

    def pins(self) -> list[dict[str, str]]:
        """Return the lineage as ordered compile-lock pin records."""
        return [pinned.pin() for pinned in self.documents]

    def prepend(self, pinned: PinnedDocument) -> "Lineage":
        """Return this lineage with one nearer document searched first."""
        return Lineage((pinned, *self.documents))


def build_lineage(
    repo_root: Path,
    parent: PinnedDocument,
    *,
    chain_keys: tuple[str, ...] = DEFAULT_CHAIN_KEYS,
    source_refs: Callable[[Mapping[str, Any]], Iterable[str]] | None = None,
) -> Lineage:
    """Walk a parent's chain into an ordered, content-pinned lineage.

    Args:
        repo_root: Root every reference is resolved against.
        parent: The already-pinned nearest parent, searched first.
        chain_keys: Blocks a document states its own origin in, in priority order.
        source_refs: How the project names additional documents a parent draws
            from beyond its own chain. Each is followed by the same rules.

    Returns:
        The lineage, nearest document first. The walk is cycle-safe: a reference
        already visited ends that branch rather than looping.
    """
    documents: list[PinnedDocument] = [parent]
    seen = {parent.ref}

    def follow(ref: Any) -> None:
        while isinstance(ref, str) and ref not in seen:
            pinned = load_pinned(repo_root, ref)
            if pinned is None:
                return
            documents.append(pinned)
            seen.add(ref)
            ref = None
            for key in chain_keys:
                block = pinned.document.get(key)
                if isinstance(block, Mapping) and isinstance(block.get("ref"), str):
                    ref = block["ref"]
                    break

    for source in source_refs(parent.document) if source_refs is not None else ():
        follow(source)
    for key in chain_keys:
        block = parent.document.get(key)
        if isinstance(block, Mapping) and isinstance(block.get("ref"), str):
            follow(block["ref"])
            break
    return Lineage(tuple(documents))


__all__ = [
    "DEFAULT_CHAIN_KEYS",
    "InheritedValue",
    "Lineage",
    "PinnedDocument",
    "build_lineage",
    "load_pinned",
]
