"""Utilities for working with Equinox ``StateIndex`` leaves."""

from __future__ import annotations

from typing import Any

from equinox.nn import StateIndex
import jax.tree as jt
from jaxtyping import PyTree


def _align_state_index_like(target: StateIndex, template: StateIndex) -> StateIndex:
    aligned = StateIndex(target.init)
    object.__setattr__(aligned, "marker", template.marker)
    return aligned


def align_state_indices_like(target_tree: PyTree, template_tree: PyTree) -> PyTree:
    """Return ``target_tree`` with ``StateIndex`` markers aligned to ``template_tree``.

    Non-``StateIndex`` leaves are preserved from ``target_tree``. A ``StateIndex`` leaf
    must match another ``StateIndex`` leaf at the same PyTree position.
    """

    def _align(target_leaf: Any, template_leaf: Any) -> Any:
        target_is_state_index = isinstance(target_leaf, StateIndex)
        template_is_state_index = isinstance(template_leaf, StateIndex)
        if target_is_state_index != template_is_state_index:
            raise TypeError(
                "StateIndex marker alignment requires matching StateIndex leaves at "
                "corresponding PyTree positions"
            )
        if target_is_state_index:
            return _align_state_index_like(target_leaf, template_leaf)
        return target_leaf

    try:
        return jt.map(
            _align,
            target_tree,
            template_tree,
            is_leaf=lambda leaf: isinstance(leaf, StateIndex),
        )
    except ValueError as exc:
        raise ValueError(
            "target_tree and template_tree must have the same PyTree structure for "
            "StateIndex marker alignment"
        ) from exc
