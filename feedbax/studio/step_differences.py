"""A step's differences from the definition it is based on, stored as few patches as say them."""

from typing import Any

from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.matrix_core import apply_override_patches
from feedbax.contracts.project import DocumentReference

type _Path = tuple[str, ...]

_MISSING = object()


def _at(payload: Any, path: _Path) -> Any:
    """The value at ``path``, ``_MISSING`` when an entry along it is absent."""
    for name in path:
        if not isinstance(payload, dict) or name not in payload:
            return _MISSING
        payload = payload[name]
    return payload


def _crosses_list(payload: Any, path: _Path) -> bool:
    for name in path:
        if isinstance(payload, list):
            return True
        if not isinstance(payload, dict) or name not in payload:
            return False
        payload = payload[name]
    return False


def _atomic(value: Any) -> bool:
    if not isinstance(value, dict):
        return True
    try:
        DocumentReference.model_validate(value)
    except ValueError:
        return False
    return True


def difference_patches(before: dict[str, Any], after: dict[str, Any], at: _Path = ()) -> list[OverridePatch]:
    """The patches that turn ``before`` into ``after``: one per differing entry, a document reference written whole."""
    patches = []
    for name in dict.fromkeys((*before, *after)):
        path = (*at, name)
        if name not in after:
            patches.append(OverridePatch(op="remove", path=".".join(path)))
        elif name not in before:
            patches.append(OverridePatch(op="add", path=".".join(path), value=after[name]))
        elif before[name] != after[name]:
            if _atomic(before[name]) or _atomic(after[name]):
                patches.append(OverridePatch(op="replace", path=".".join(path), value=after[name]))
            else:
                patches.extend(difference_patches(before[name], after[name], path))
    return patches


def normalized_patches(before: dict[str, Any], patches: list[OverridePatch]) -> list[OverridePatch]:
    """``patches``, applied to ``before``, as one patch per written path whose value differs from ``before``'s.

    A path below another written path is said by that one. Patches that cross a list keep their order and are
    returned as they are, since an index names a different entry once an earlier patch inserts or removes one.
    """
    paths = [tuple(patch.path.split(".")) for patch in patches]
    after = apply_override_patches(before, patches)
    if any(_crosses_list(before, path) or _crosses_list(after, path) for path in paths):
        return patches
    written = list(dict.fromkeys(paths))
    kept = [path for path in written if not any(other != path and path[:len(other)] == other for other in written)]
    normalized: dict[_Path, OverridePatch] = {}
    for path in kept:
        old, new = _at(before, path), _at(after, path)
        if old == new:
            continue
        if new is _MISSING:
            normalized[path] = OverridePatch(op="remove", path=".".join(path))
            continue
        if old is _MISSING:
            path = next(path[:depth] for depth in range(1, len(path) + 1) if _at(before, path[:depth]) is _MISSING)
            normalized[path] = OverridePatch(op="add", path=".".join(path), value=_at(after, path))
            continue
        normalized[path] = OverridePatch(op="replace", path=".".join(path), value=new)
    return list(normalized.values())
