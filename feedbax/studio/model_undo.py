"""Reversible deltas for the addressed model-edit transaction."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.matrix_core import apply_value_patch_in_place
from feedbax.contracts.project import (
    ProjectValueEdit, RemoveProjectValue, SetProjectValue, json_pointer_parts,
)
from feedbax.contracts.project_sources import pointer


def inverse_delta(
    before: dict[str, Any], after: dict[str, Any], path: tuple[str, ...] = (),
) -> Iterator[ProjectValueEdit]:
    """Describe the changes that restore a transaction's exact authored values."""
    for name, previous in before.items():
        target = (*path, name)
        if name not in after:
            yield SetProjectValue(op="add", path=pointer(target), value=previous)
            continue
        current = after[name]
        if isinstance(previous, dict) and isinstance(current, dict):
            yield from inverse_delta(previous, current, target)
        elif canonical_json_v2_bytes(previous) != canonical_json_v2_bytes(current):
            yield SetProjectValue(op="replace", path=pointer(target), value=previous)
    for name in after:
        if name not in before:
            yield RemoveProjectValue(op="remove", path=pointer((*path, name)))


def apply_delta(target: dict[str, Any], patches: tuple[ProjectValueEdit, ...]) -> None:
    for patch in patches:
        apply_value_patch_in_place(
            target, json_pointer_parts(patch.path), patch.op,
            patch.value if isinstance(patch, SetProjectValue) else None, path=patch.path,
        )
