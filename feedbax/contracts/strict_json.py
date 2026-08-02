"""Strict JSON parsing for authority-boundary documents.

Python's :func:`json.loads` silently keeps the *last* value for a duplicated
object member name. At an authority boundary — a compile lock, a manifest, a
custody sidecar, a row index, a report or figure payload — a duplicated member
name means one document states two things about the same fact. Collapsing that
to "whichever came last" picks an authority at random, so the document is
refused here instead.

This module owns the single strict loader every such boundary routes through.
Refusal is the only behavioral difference: for a document with no duplicated
member name, :func:`strict_json_loads` returns exactly what
:func:`json.loads` returns, with the same object identity semantics, the same
member order, and the same scalar decoding.
"""

from __future__ import annotations

import json
import re
from typing import Any

__all__ = [
    "DuplicateJsonKeyError",
    "StrictJsonError",
    "strict_json_loads",
]

_BARE_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class StrictJsonError(ValueError):
    """Base class for strict-JSON refusals at an authority boundary."""


class DuplicateJsonKeyError(StrictJsonError):
    """One JSON object states the same member name more than once.

    Attributes:
        key: The duplicated member name.
        json_path: The JSON path of the duplicated member, such as
            ``$.provenance.parents[1].id``.
        ref: An optional caller-supplied name for the document.
    """

    def __init__(self, key: str, json_path: str, *, ref: str | None = None) -> None:
        self.key = key
        self.json_path = json_path
        self.ref = ref
        where = f" in {ref}" if ref else ""
        super().__init__(
            f"duplicate JSON object key {key!r} at {json_path}{where}: a document that states "
            "the same member twice states two authorities for one fact, and the standard "
            "last-value-wins parse would silently choose one of them"
        )


def _format_key(key: str) -> str:
    """Return one path segment for ``key``, quoting anything non-identifier."""
    if _BARE_KEY.match(key):
        return f".{key}"
    return f"[{json.dumps(key)}]"


def _locate(
    node: Any,
    duplicates: dict[int, tuple[str, ...]],
    path: str,
) -> tuple[str, str] | None:
    """Return the (json_path, key) of the first duplicate reachable from ``node``.

    The walk is depth-first in document order, so the reported path is the
    earliest duplicated member in the document.
    """
    if isinstance(node, dict):
        found = duplicates.get(id(node))
        if found is not None:
            return path + _format_key(found[0]), found[0]
        for key, value in node.items():
            located = _locate(value, duplicates, path + _format_key(key))
            if located is not None:
                return located
        return None
    if isinstance(node, list):
        for index, value in enumerate(node):
            located = _locate(value, duplicates, f"{path}[{index}]")
            if located is not None:
                return located
        return None
    return None


def strict_json_loads(data: str | bytes | bytearray, *, ref: str | None = None) -> Any:
    """Parse one JSON document, refusing any duplicated object member name.

    Args:
        data: The document bytes or text, exactly as :func:`json.loads` accepts.
        ref: Optional document name used in the refusal message.

    Returns:
        The same value :func:`json.loads` would return for a document with no
        duplicated member name, at any depth.

    Raises:
        DuplicateJsonKeyError: If any object in the document — at the top level
            or nested at any depth inside objects or arrays — states the same
            member name more than once.
        json.JSONDecodeError: If the document is not valid JSON.
    """
    duplicates: dict[int, tuple[str, ...]] = {}
    retained: list[dict[str, Any]] = []

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        obj: dict[str, Any] = {}
        repeated: list[str] = []
        for key, value in pairs:
            if key in obj:
                repeated.append(key)
            obj[key] = value
        if repeated:
            duplicates[id(obj)] = tuple(dict.fromkeys(repeated))
            # Hold a reference so no later allocation can reuse this ``id``.
            retained.append(obj)
        return obj

    parsed = json.loads(data, object_pairs_hook=hook)
    if duplicates:
        located = _locate(parsed, duplicates, "$")
        if located is None:  # pragma: no cover - unreachable while ids stay live
            key = next(iter(duplicates.values()))[0]
            raise DuplicateJsonKeyError(key, "$", ref=ref)
        json_path, key = located
        raise DuplicateJsonKeyError(key, json_path, ref=ref)
    return parsed
