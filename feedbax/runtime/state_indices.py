"""Utilities for working with Equinox ``StateIndex`` leaves."""

from __future__ import annotations

from typing import Any, TypeVar

import jax.tree_util as jtu

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


_Model = TypeVar("_Model")


def prepare_component(component: _Model) -> _Model:
    """Return a model with state markers determined by typed PyTree occurrence paths.

    Initial values remain dynamic leaves and are retained unchanged. Preparing the
    outer complete model gives independently composed children distinct state slots.
    """

    def prepare_index(path, value):
        if not isinstance(value, StateIndex):
            return value
        marker = tuple(
            (type(entry), type(entry.key), entry.key)
            if isinstance(entry, jtu.DictKey)
            else (type(entry), entry)
            for entry in path
        )
        index = StateIndex(value.init)
        object.__setattr__(index, "marker", marker)
        return index

    return jtu.tree_map_with_path(
        prepare_index, component, is_leaf=lambda value: isinstance(value, StateIndex)
    )
