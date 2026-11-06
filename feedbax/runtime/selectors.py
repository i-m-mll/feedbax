"""PyTree selection API for fluent tree manipulation.

Provides a chainable interface for selecting, modifying, and partitioning
leaves in PyTrees. Compatible with JAX transformations (jit, vmap).

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import equinox as eqx
import jax
import jax.tree as jt
from jax.tree_util import KeyPath, PyTreeDef, keystr
from jaxtyping import PyTree
import numpy as np

T = TypeVar("T")


_Accessor = Callable[[PyTree], PyTree]
_Predicate = Callable[[Any], bool]


@dataclass(frozen=True)
class _At:
    accessor: _Accessor


@dataclass(frozen=True)
class _Where:
    predicate: _Predicate


@dataclass(frozen=True)
class _InstancesOf:
    types: tuple[type, ...]

    def matches(self, value: Any) -> bool:
        return isinstance(value, self.types)


def _is_none(value: Any) -> bool:
    return value is None


def _predicate_decision(predicate: _Predicate, value: Any) -> bool:
    decision = predicate(value)
    if isinstance(decision, jax.core.Tracer):
        raise ValueError("Selection predicates must return a concrete boolean while tracing")
    if isinstance(decision, (bool, np.bool_)) or (
        isinstance(decision, (jax.Array, np.ndarray))
        and decision.shape == ()
        and decision.dtype == np.dtype(bool)
    ):
        return bool(decision)
    raise ValueError("Selection predicates must return a scalar boolean")


class _AccessorMarker:
    pass


class _AccessorReplacement(_AccessorMarker):
    """One replacement or an ordered group, as determined by tree_at."""

    def __init__(self, target: PyTree):
        self.target = target
        self.group: tuple[_AccessorMarker, ...] | None = None

    def __len__(self) -> int:
        return len(self.target)

    def __iter__(self) -> Iterator[_AccessorMarker]:
        if self.group is None:
            self.group = tuple(_AccessorMarker() for _ in range(len(self)))
        return iter(self.group)


@dataclass(frozen=True)
class _Scope:
    target: PyTree
    insert: _Accessor
    units: PyTree | None = None

    def at(
        self, accessor: _Accessor, *, project_coverage: bool = False
    ) -> tuple[_Scope, dict[KeyPath, KeyPath]]:
        target = accessor(self.target)
        replacement = _AccessorReplacement(target)
        # tree_at uses the replacement itself for a single subtree and iterates
        # it for a group of locations, in the accessor's order.
        marked = eqx.tree_at(accessor, self.target, replace=replacement, is_leaf=_is_none)
        if replacement.group is None:
            units = replacement
        else:
            _, group_structure = eqx.tree_flatten_one_level(target)
            units = group_structure.unflatten(replacement.group)
        markers, structure = jt.flatten(units)
        paths = {}
        if project_coverage:
            locations = {
                value: path
                for path, value in jt.leaves_with_path(marked)
                if isinstance(value, _AccessorMarker)
            }
            paths = {path: locations[value] for path, value in jt.leaves_with_path(units)}

        def insert(updated: PyTree) -> PyTree:
            values = dict(zip(markers, structure.flatten_up_to(updated), strict=True))
            return self.insert(
                jt.map(
                    lambda leaf: values[leaf] if isinstance(leaf, _AccessorMarker) else leaf, marked
                )
            )

        return _Scope(target, insert, units), paths


class _Decisions:
    """Selection coverage in the current scope, including boolean prefixes."""

    def __init__(self, values: dict[KeyPath, bool]):
        self.values = values
        self.descendants: dict[KeyPath, set[bool]] = {}
        for path, keep in values.items():
            for depth in range(len(path)):
                self.descendants.setdefault(path[:depth], set()).add(keep)

    @classmethod
    def from_spec(cls, spec: PyTree[bool], target: PyTree) -> _Decisions:
        pairs, structure = jt.flatten_with_path(spec)
        try:
            structure.flatten_up_to(target)
        except ValueError as exc:
            raise ValueError("Selection filter_spec must be a prefix of its target") from exc
        if any(not isinstance(value, bool) for _, value in pairs):
            raise ValueError("Selection filter_spec must contain boolean values")
        return cls(dict(pairs))

    def ancestor(self, path: KeyPath) -> bool | None:
        for depth in range(len(path), -1, -1):
            decision = self.values.get(path[:depth])
            if decision is not None:
                return decision
        return None

    def includes(self, path: KeyPath) -> bool:
        decision = self.ancestor(path)
        if decision is not None:
            return decision
        descendants = self.descendants.get(path, set())
        if len(descendants) > 1:
            raise ValueError(
                f"Selection mask partially covers the atomic subtree at {keystr(path) or '<root>'}"
            )
        return True in descendants

    def project(self, paths: dict[KeyPath, KeyPath]) -> _Decisions:
        values = {}
        sources = {}
        for destination, source in paths.items():
            decision = self.ancestor(source)
            if decision is not None:
                values[destination] = decision
            else:
                sources[source] = destination
        if sources:
            for path, keep in self.values.items():
                for depth in range(len(path)):
                    destination = sources.get(path[:depth])
                    if destination is not None:
                        values[destination + path[depth:]] = keep
                        break
        return _Decisions(values)


@dataclass(frozen=True)
class _ResolvedSelection:
    """The editable units of an accessor target and their selection decisions."""

    leaves: list[Any]
    structure: PyTreeDef
    selected: list[bool]
    insert: _Accessor

    def project(self, selected: bool) -> PyTree:
        return self.structure.unflatten(
            leaf if keep == selected else None
            for leaf, keep in zip(self.leaves, self.selected, strict=True)
        )

    def apply(self, fn: Callable[[Any], Any]) -> PyTree:
        return self.insert(
            self.structure.unflatten(
                fn(leaf) if keep else leaf
                for leaf, keep in zip(self.leaves, self.selected, strict=True)
            )
        )


class Selection(Generic[T]):
    """A chainable selection over a PyTree.

    Selections allow fluent specification of which leaves to operate on,
    with operations like `set`, `apply`, `partition`, etc.

    The selection is defined by a combination of:
    - `where`: A function `tree -> subtree` (like eqx.tree_at's first arg)
    - `filter_spec`: A boolean PyTree or predicate function
    - `is_leaf`: A function determining what counts as a leaf

    Refinements via `at()`, `at_instances_of()`, and `where()` are applied in order.
    An accessor without filters selects whole subtrees as single units. Filters
    traverse within that scope, retaining previous decisions when the scope narrows.
    """

    __slots__ = ("_tree", "_where", "_filter_spec", "_is_leaf", "_refinements")

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
            filter_spec: A boolean PyTree prefix of the (sub)tree, or a predicate
                         function applied to its leaves.
            is_leaf: A function determining what counts as a leaf during traversal.
        """
        self._tree = tree
        self._where = where
        self._filter_spec = filter_spec
        self._is_leaf = is_leaf
        self._refinements: tuple[_At | _Where | _InstancesOf, ...] = ()

    @property
    def tree(self) -> PyTree[Any, T]:
        """The original tree this selection operates on."""
        return self._tree

    def at(self, where: Callable[[PyTree], PyTree]) -> Selection[T]:
        """Refine the selection by specifying a subtree accessor.

        Args:
            where: A structural accessor returning a subtree or sequence of subtrees
                   within the current scope.

        Returns:
            A new Selection with the refined accessor.

        Example:
            >>> select(model).at(lambda m: m.layers).at(lambda l: l.weights)
            # Equivalent to: select(model).at(lambda m: m.layers.weights)
        """
        return self._refine(_At(where))

    def at_instances_of(self, *types: type) -> Selection[T]:
        """Select matching types and treat matching subtrees as single units.

        A matching subtree must be uniformly selected or excluded by earlier filters.

        Args:
            *types: One or more types to match against.

        Returns:
            A new Selection targeting only leaves of the specified types.

        Example:
            >>> select(model).at_instances_of(jnp.ndarray)
            # Selects all array leaves
        """
        return self._refine(_InstancesOf(types))

    def where(self, predicate: Callable[[Any], bool]) -> Selection[T]:
        """Filter the selection to leaves satisfying a predicate.

        Args:
            predicate: A function returning True for leaves to include.

        Returns:
            A new Selection with the predicate filter applied.

        Example:
            >>> select(model).where(lambda x: hasattr(x, 'shape') and x.shape[0] > 10)
        """
        return self._refine(_Where(predicate))

    def _refine(self, refinement: _At | _Where | _InstancesOf) -> Selection[T]:
        result = Selection(self._tree, self._where, self._filter_spec, self._is_leaf)
        result._refinements = (*self._refinements, refinement)
        return result

    def _resolve(self) -> _ResolvedSelection:
        scope = _Scope(self._tree, lambda tree: tree)
        if self._where is not None:
            scope, _ = scope.at(self._where)
        decisions = None
        explicit = None
        refinements = self._refinements
        if callable(self._filter_spec):
            refinements = (_Where(self._filter_spec), *refinements)
        elif self._filter_spec is not None:
            explicit = _Decisions.from_spec(self._filter_spec, scope.target)

        boundaries: list[_Predicate] = []
        if self._is_leaf is not None:
            boundaries.append(self._is_leaf)

        def is_leaf(value: Any) -> bool:
            return any(boundary(value) for boundary in boundaries)

        def includes(path: KeyPath) -> bool:
            return (decisions is None or decisions.includes(path)) and (
                explicit is None or explicit.includes(path)
            )

        # Only refinements that precede a scope or boundary change need path coverage.
        predicates = []
        while refinements and isinstance(refinements[-1], _Where):
            predicates.append(refinements[-1].predicate)
            refinements = refinements[:-1]
        predicates.reverse()
        terminal_type = None
        if refinements and isinstance(refinements[-1], _InstancesOf):
            terminal_type = refinements[-1].matches
            refinements = refinements[:-1]

        flattened = None
        for refinement in refinements:
            if isinstance(refinement, _At):
                scope, paths = scope.at(
                    refinement.accessor,
                    project_coverage=decisions is not None or explicit is not None,
                )
                flattened = None
                if decisions is not None:
                    decisions = decisions.project(paths)
                if explicit is not None:
                    explicit = explicit.project(paths)
                continue
            if isinstance(refinement, _InstancesOf):
                boundaries.append(refinement.matches)
                flattened = jt.flatten_with_path(scope.target, is_leaf=is_leaf)
                pairs, _ = flattened
                values = {path: refinement.matches(leaf) and includes(path) for path, leaf in pairs}
            else:
                if flattened is None:
                    flattened = jt.flatten_with_path(
                        scope.target, is_leaf=is_leaf if boundaries else None
                    )
                pairs, _ = flattened
                values = {
                    path: includes(path) and _predicate_decision(refinement.predicate, leaf)
                    for path, leaf in pairs
                }
            decisions = _Decisions(values)

        if terminal_type is not None:
            boundaries.append(terminal_type)
            flattened = None

        def passes_predicates(leaf: Any) -> bool:
            return not predicates or all(_predicate_decision(p, leaf) for p in predicates)

        unfiltered = decisions is None and explicit is None
        if unfiltered and terminal_type is None and not predicates and scope.units is not None:
            structure = jt.structure(scope.units)
            leaves = structure.flatten_up_to(scope.target)
            selected = [True] * len(leaves)
        elif unfiltered:
            leaves, structure = jt.flatten(scope.target, is_leaf=is_leaf if boundaries else None)
            selected = [
                (terminal_type is None or terminal_type(leaf)) and passes_predicates(leaf)
                for leaf in leaves
            ]
        else:
            if flattened is None:
                flattened = jt.flatten_with_path(
                    scope.target, is_leaf=is_leaf if boundaries else None
                )
            pairs, structure = flattened
            leaves = [leaf for _, leaf in pairs]
            selected = [
                (terminal_type is None or terminal_type(leaf))
                and includes(path)
                and passes_predicates(leaf)
                for path, leaf in pairs
            ]
        return _ResolvedSelection(leaves, structure, selected, scope.insert)

    def get(self) -> PyTree:
        """Return the selected portion of the tree.

        Returns:
            A PyTree containing only the selected leaves (others are None).
        """
        return self._resolve().project(True)

    def count(self) -> int:
        """Return the number of selected leaves.

        Returns:
            The count of leaves matching the selection criteria.
        """
        return sum(self._resolve().selected)

    def set(self, value: Any) -> PyTree[Any, T]:
        """Set all selected leaves to a single value.

        Args:
            value: The value to set at all selected positions.

        Returns:
            A new tree with selected leaves replaced by `value`.

        Example:
            >>> select(model).at_instances_of(jnp.ndarray).set(jnp.zeros(10))
        """
        return self.apply(lambda _: value)

    def apply(self, fn: Callable[[Any], Any]) -> PyTree[Any, T]:
        """Apply a function to all selected leaves.

        Args:
            fn: A function to apply to each selected leaf.

        Returns:
            A new tree with `fn` applied to selected leaves.

        Example:
            >>> select(model).at_instances_of(jnp.ndarray).apply(lambda x: x * 2)
        """
        resolved = self._resolve()
        return resolved.apply(fn) if any(resolved.selected) else self._tree

    def partition(self) -> tuple[PyTree, PyTree]:
        """Split the tree into selected and non-selected parts.

        Returns:
            A tuple (selected, rest) where:
            - `selected`: PyTree with only selected leaves (others are None)
            - `rest`: PyTree with only non-selected leaves (others are None)

        Example:
            >>> trainable, frozen = select(model).at_instances_of(jnp.ndarray).partition()
        """
        resolved = self._resolve()
        return resolved.project(True), resolved.project(False)

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

        def merge(left: PyTree, right: PyTree) -> PyTree:
            if left is None:
                return right
            if right is None or (is_leaf is not None and is_leaf(left)):
                return left
            children, structure = eqx.tree_flatten_one_level(left)
            if structure.num_nodes == structure.num_leaves == 1:
                return left
            return structure.unflatten(
                merge(a, b) for a, b in zip(children, structure.flatten_up_to(right), strict=True)
            )

        return merge(selected, rest)

    def __repr__(self) -> str:
        parts = [f"Selection({type(self._tree).__name__}"]
        if self._where is not None or any(isinstance(r, _At) for r in self._refinements):
            parts.append(", where=<accessor>")
        if self._filter_spec is not None:
            if callable(self._filter_spec):
                parts.append(", filter=<predicate>")
            else:
                parts.append(", filter=<spec>")
        if any(isinstance(r, _Where) for r in self._refinements):
            parts.append(", where=<predicate>")
        if any(isinstance(r, _InstancesOf) for r in self._refinements):
            parts.append(", types=<types>")
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
