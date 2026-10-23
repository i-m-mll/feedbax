"""PyTree selection API for fluent tree manipulation.

Provides a chainable interface for selecting, modifying, and partitioning
leaves in PyTrees. Compatible with JAX transformations (jit, vmap).

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar, overload

import equinox as eqx
import jax.tree as jt
from jaxtyping import PyTree

T = TypeVar("T")
S = TypeVar("S")


class Selection(Generic[T]):
    """A chainable selection over a PyTree.

    Selections allow fluent specification of which leaves to operate on,
    with operations like `set`, `apply`, `partition`, etc.

    The selection is defined by a combination of:
    - `where`: A function `tree -> subtree` (like eqx.tree_at's first arg)
    - `filter_spec`: A boolean PyTree or predicate function
    - `is_leaf`: A function determining what counts as a leaf

    These can be chained via `at()`, `at_instances_of()`, and `where()`.
    """

    __slots__ = ("_tree", "_where", "_filter_spec", "_is_leaf")

    def __init__(
        self,
        tree: PyTree[Any, T],
        where: Callable[[PyTree], PyTree] | None = None,
        filter_spec: PyTree[bool] | Callable[[Any], bool] | None = None,
        is_leaf: Callable[[Any], bool] | None = None,
    ):
        """Initialize a Selection.

        Args:
            tree: The PyTree to select from.
            where: A function that extracts the subtree to operate on.
                   Similar to eqx.tree_at's first argument.
            filter_spec: Either a boolean PyTree matching the structure of
                         the (sub)tree, or a predicate function applied to leaves.
            is_leaf: A function determining what counts as a leaf during traversal.
        """
        self._tree = tree
        self._where = where
        self._filter_spec = filter_spec
        self._is_leaf = is_leaf

    @property
    def tree(self) -> PyTree[Any, T]:
        """The original tree this selection operates on."""
        return self._tree

    def at(self, where: Callable[[PyTree], PyTree]) -> Selection[T]:
        """Refine the selection by specifying a subtree accessor.

        Args:
            where: A function that extracts a subtree from the current selection.
                   If chained, the new `where` is composed with the existing one.

        Returns:
            A new Selection with the refined accessor.

        Example:
            >>> select(model).at(lambda m: m.layers).at(lambda l: l.weights)
            # Equivalent to: select(model).at(lambda m: m.layers.weights)
        """
        if self._where is None:
            new_where = where
        else:
            # Compose the accessors
            old_where = self._where
            new_where = lambda tree: where(old_where(tree))

        return Selection(
            self._tree,
            where=new_where,
            filter_spec=self._filter_spec,
            is_leaf=self._is_leaf,
        )

    def at_instances_of(self, *types: type) -> Selection[T]:
        """Select all leaves that are instances of the given types.

        Args:
            *types: One or more types to match against.

        Returns:
            A new Selection targeting only leaves of the specified types.

        Example:
            >>> select(model).at_instances_of(jnp.ndarray)
            # Selects all array leaves
        """
        type_predicate = lambda x: isinstance(x, types)
        return self._add_filter(type_predicate, is_leaf=type_predicate)

    def where(self, predicate: Callable[[Any], bool]) -> Selection[T]:
        """Filter the selection to leaves satisfying a predicate.

        Args:
            predicate: A function returning True for leaves to include.

        Returns:
            A new Selection with the predicate filter applied.

        Example:
            >>> select(model).where(lambda x: hasattr(x, 'shape') and x.shape[0] > 10)
        """
        return self._add_filter(predicate)

    def _add_filter(
        self,
        predicate: Callable[[Any], bool],
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> Selection[T]:
        """Internal: add a filter predicate to the selection."""
        # Combine with existing filter_spec if present
        if self._filter_spec is None:
            new_filter = predicate
        elif callable(self._filter_spec):
            old_pred = self._filter_spec
            new_filter = lambda x: old_pred(x) and predicate(x)
        else:
            # Existing filter_spec is a boolean PyTree; we can't easily
            # compose with a predicate, so we convert to a combined approach
            # by storing the predicate and applying both at operation time
            new_filter = predicate

        # Combine is_leaf specifications
        new_is_leaf = is_leaf if is_leaf is not None else self._is_leaf

        return Selection(
            self._tree,
            where=self._where,
            filter_spec=new_filter,
            is_leaf=new_is_leaf,
        )

    def _get_target_tree(self) -> PyTree:
        """Get the subtree that operations will apply to."""
        if self._where is None:
            return self._tree
        return self._where(self._tree)

    def _get_filter_spec(self, target: PyTree) -> PyTree[bool]:
        """Compute the boolean filter spec for the target subtree."""
        if self._filter_spec is None:
            # No filter: select everything
            return jt.map(lambda _: True, target, is_leaf=self._is_leaf)
        elif callable(self._filter_spec):
            # Predicate function: apply to each leaf
            return jt.map(self._filter_spec, target, is_leaf=self._is_leaf)
        else:
            # Already a boolean PyTree
            return self._filter_spec

    def get(self) -> PyTree:
        """Return the selected portion of the tree.

        Returns:
            A PyTree containing only the selected leaves (others are None).
        """
        target = self._get_target_tree()
        filter_spec = self._get_filter_spec(target)
        selected, _ = eqx.partition(target, filter_spec, is_leaf=self._is_leaf)
        return selected

    def count(self) -> int:
        """Return the number of selected leaves.

        Returns:
            The count of leaves matching the selection criteria.
        """
        target = self._get_target_tree()
        filter_spec = self._get_filter_spec(target)
        # Count True values in the filter spec
        return sum(jt.leaves(filter_spec))

    def set(self, value: Any) -> PyTree[Any, T]:
        """Set all selected leaves to a single value.

        Args:
            value: The value to set at all selected positions.

        Returns:
            A new tree with selected leaves replaced by `value`.

        Example:
            >>> select(model).at_instances_of(jnp.ndarray).set(jnp.zeros(10))
        """
        if self._where is not None and self._filter_spec is None:
            # Simple case: use eqx.tree_at directly
            return eqx.tree_at(
                self._where,
                self._tree,
                value,
            )

        # Complex case: need to handle filter_spec
        target = self._get_target_tree()
        filter_spec = self._get_filter_spec(target)

        # Create replacement function
        replace_fn = lambda _: value

        new_target = eqx.tree_at(
            lambda t: t,
            target,
            replace_fn=lambda t: jt.map(
                lambda leaf, keep: replace_fn(leaf) if keep else leaf,
                t,
                filter_spec,
                is_leaf=self._is_leaf,
            ),
        )

        # If we had a where accessor, update the original tree
        if self._where is not None:
            return eqx.tree_at(self._where, self._tree, new_target)
        return new_target

    def apply(self, fn: Callable[[Any], Any]) -> PyTree[Any, T]:
        """Apply a function to all selected leaves.

        Args:
            fn: A function to apply to each selected leaf.

        Returns:
            A new tree with `fn` applied to selected leaves.

        Example:
            >>> select(model).at_instances_of(jnp.ndarray).apply(lambda x: x * 2)
        """
        if self._where is not None and self._filter_spec is None:
            # Simple case: use eqx.tree_at with replace_fn
            return eqx.tree_at(
                self._where,
                self._tree,
                replace_fn=fn,
            )

        # Complex case: need to handle filter_spec
        target = self._get_target_tree()
        filter_spec = self._get_filter_spec(target)

        new_target = jt.map(
            lambda leaf, keep: fn(leaf) if keep else leaf,
            target,
            filter_spec,
            is_leaf=self._is_leaf,
        )

        # If we had a where accessor, update the original tree
        if self._where is not None:
            return eqx.tree_at(self._where, self._tree, new_target)
        return new_target

    def partition(self) -> tuple[PyTree, PyTree]:
        """Split the tree into selected and non-selected parts.

        Returns:
            A tuple (selected, rest) where:
            - `selected`: PyTree with only selected leaves (others are None)
            - `rest`: PyTree with only non-selected leaves (others are None)

        Example:
            >>> trainable, frozen = select(model).at_instances_of(jnp.ndarray).partition()
        """
        target = self._get_target_tree()
        filter_spec = self._get_filter_spec(target)
        return eqx.partition(target, filter_spec, is_leaf=self._is_leaf)

    @staticmethod
    def combine(selected: PyTree, rest: PyTree, is_leaf: Callable | None = None) -> PyTree:
        """Combine selected and rest parts back into a complete tree.

        This is the inverse of `partition()`.

        Args:
            selected: The selected portion (from partition).
            rest: The non-selected portion (from partition).
            is_leaf: Optional leaf predicate for combination.

        Returns:
            The recombined tree.

        Example:
            >>> trainable, frozen = select(model).partition()
            >>> # ... modify trainable ...
            >>> model = Selection.combine(trainable, frozen)
        """
        return eqx.combine(selected, rest, is_leaf=is_leaf)

    def __repr__(self) -> str:
        parts = [f"Selection({type(self._tree).__name__}"]
        if self._where is not None:
            parts.append(", where=<accessor>")
        if self._filter_spec is not None:
            if callable(self._filter_spec):
                parts.append(", filter=<predicate>")
            else:
                parts.append(", filter=<spec>")
        if self._is_leaf is not None:
            parts.append(", is_leaf=<func>")
        parts.append(")")
        return "".join(parts)


def select(tree: PyTree[Any, T]) -> Selection[T]:
    """Create a Selection over a PyTree.

    This is the main entry point for the selection API.

    Args:
        tree: The PyTree to select from.

    Returns:
        A Selection object for fluent chaining.

    Example:
        >>> # Set all arrays to zeros
        >>> model = select(model).at_instances_of(jnp.ndarray).apply(jnp.zeros_like)

        >>> # Partition trainable from frozen
        >>> trainable, frozen = select(model).where(eqx.is_array).partition()
    """
    return Selection(tree)


# Compatibility shim for eqx.tree_at-style usage
def tree_at(
    where: Callable[[PyTree], PyTree],
    tree: PyTree[Any, T],
    value: Any = None,
    replace_fn: Callable[[Any], Any] | None = None,
) -> PyTree[Any, T]:
    """Functional tree modification (thin wrapper around eqx.tree_at).

    Provides a consistent API alongside the Selection class.

    Args:
        where: A function extracting the subtree to modify.
        tree: The tree to modify.
        value: The replacement value (mutually exclusive with replace_fn).
        replace_fn: A function to compute replacement (mutually exclusive with value).

    Returns:
        A new tree with the modification applied.
    """
    if value is not None and replace_fn is not None:
        raise ValueError("Cannot specify both `value` and `replace_fn`")

    if replace_fn is not None:
        return eqx.tree_at(where, tree, replace_fn=replace_fn)
    return eqx.tree_at(where, tree, value)
