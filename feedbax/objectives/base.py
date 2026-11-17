"""Shared objective module and tree of weighted term values."""

import functools as ft
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Optional, Self, Tuple, TypeVar

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, ArrayLike

U = TypeVar("U")


@jtu.register_pytree_node_class
@dataclass
class TermTree[T](Mapping[str, "TermTree"]):
    label: str
    names: Tuple[str, ...]
    children: Tuple[Self, ...] = ()
    value: Optional[Array] = None  # only for leaf
    weight: float = 1.0
    leaf_fn: Callable[[Array], Array] = jnp.mean
    originator: Optional[T] = None

    def tree_flatten(self):
        """Flatten for PyTree; only include numeric/JAX parts dynamically."""
        # dynamic leaves (these are mapped by vmap/jit):
        # weight is included here (not in aux) so that changing the weight value
        # does NOT change the treedef. This allows loss_update_func to update
        # weights without breaking tree_set(history.loss, new_losses, idx).
        # weight is kept as-is (no jnp.asarray) so that Python float weights
        # stay as Python non-array leaves, avoiding traced-bool issues in
        # filter_jit's partition(out, is_array) output processing.
        children = (self.children, self.value, self.weight)
        # static metadata (these are carried in aux data, not mapped):
        aux = (self.label, self.names, self.leaf_fn, self.originator)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        label, names, leaf_fn, originator = aux_data
        kids, value, weight = children
        return cls(
            label=label,
            names=names,
            children=kids,
            value=value,
            weight=weight,
            leaf_fn=leaf_fn,
            originator=originator,
        )

    @staticmethod
    def leaf(
        label: str,
        value: Array,
        *,
        weight: float = 1.0,
        leaf_fn: Callable[[Array], Array] = jnp.mean,
        originator: Optional[T] = None,
    ) -> "TermTree":
        return TermTree(
            label=label,
            names=(),
            children=(),
            value=value,
            leaf_fn=leaf_fn,
            weight=weight,  # originator=originator
        )

    @staticmethod
    def branch(
        label: str,
        mapping: Mapping[str, "TermTree"],
        *,
        weight: float = 1.0,
        leaf_fn: Callable[[Array], Array] = jnp.mean,
        originator: Optional[T] = None,  #! I don't think this should use the class generic T
    ) -> "TermTree":
        # Freeze order for JIT-stability
        names = tuple(mapping.keys())
        kids = tuple(mapping[k] for k in names)
        return TermTree(
            label=label,
            names=names,
            children=kids,
            value=None,
            weight=weight,
            leaf_fn=leaf_fn,
            # originator=originator,
        )

    def with_weight(self, weight: float) -> Self:
        return TermTree(
            label=self.label,
            names=self.names,
            children=self.children,
            value=self.value,
            leaf_fn=self.leaf_fn,
            weight=weight,
        )

    def fold(
        self,
        *,
        on_leaf: Callable[["TermTree"], U],
        on_branch: Callable[["TermTree", Tuple[U, ...]], U],
    ) -> U:
        if self.value is not None:
            return on_leaf(self)
        child_vals = tuple(c.fold(on_leaf=on_leaf, on_branch=on_branch) for c in self.children)
        return on_branch(self, child_vals)

    def aggregate(
        self,
        *,
        leaf_fn: Optional[
            Callable[[Array], Array]
        ] = None,  # uniform override; None → use node.leaf_fn
        plus: Callable[[Array, Array], Array] = jnp.add,  # how to combine siblings
        times: Callable[[Array, Array], Array] = jnp.multiply,  # how to apply weights
        zero: Array | float = 0.0,  # identity for `plus`
    ) -> Array:
        # (a) how to evaluate a leaf:
        def _on_leaf(node: "TermTree") -> Array:
            f = node.leaf_fn if leaf_fn is None else leaf_fn
            v = f(node.value)  # scalar (or broadcastable) is expected
            return times(jnp.asarray(node.weight), v)

        # (b) how to evaluate a branch:
        def _on_branch(node: "TermTree", kids: Tuple[Array, ...]) -> Array:
            acc = ft.reduce(plus, kids, jnp.asarray(zero))
            return times(jnp.asarray(node.weight), acc)

        return self.fold(on_leaf=_on_leaf, on_branch=_on_branch)

    @property
    def total(self) -> Array:
        """Return the node-weighted scalar sum of all leaves in this tree."""
        return self.aggregate()

    def map(self, fn: Callable[[Array], ArrayLike], check_value: Callable = eqx.is_array) -> Self:
        """Apply a function to all `value` arrays in this `TermTree`.

        Handle the recursion explicitly rather than using `jt.map` and excluding `weights` etc.
        """
        if self.value is not None:
            # Leaf node
            new_value = fn(self.value) if check_value(self.value) else self.value
            return eqx.tree_at(lambda t: t.value, self, new_value)
        else:
            # Branch node - recursively apply to children
            new_children = tuple(child.map(fn) for child in self.children)
            return eqx.tree_at(lambda t: t.children, self, new_children)

    def _walk_leaves(
        self,
        *,
        include_root: bool,
        apply_weights: bool,
    ):
        """Yield (path:str, node:TermTree, cum_w:float)."""

        def visit(node: "TermTree", path_parts: list[str], cum_w: float):
            if node.value is not None:
                # leaf
                path = "/".join(path_parts) if path_parts else node.label
                yield path, node, (cum_w * node.weight if apply_weights else 1.0)
                return

            next_cum_w = cum_w * (node.weight if apply_weights else 1.0)
            base_parts = [node.label] if include_root and not path_parts else path_parts
            for name, child in zip(node.names, node.children):
                yield from visit(child, base_parts + [name], next_cum_w)

        yield from visit(self, [], 1.0)

    def flatten(
        self,
        *,
        apply_weights: bool = True,
        apply_leaf_fn: bool = False,
        include_root: bool = False,
    ) -> dict[str, Array]:
        """
        Returns {'path/to/leaf': Array} for all leaves, WITHOUT applying `node.leaf_fn`.

        If `apply_weights=True`, multiplies the raw leaf arrays by the cumulative
        product of weights along the path (including the leaf's own `weight`).
        """
        out: dict[str, Array] = {}
        for path, node, w in self._walk_leaves(
            include_root=include_root, apply_weights=apply_weights
        ):
            #! TODO: Make this `dict[str, Array | None]` and keep `None` leaves
            arr = node.value if node.value is not None else jnp.asarray(0.0)
            arr = self.leaf_fn(arr) if apply_leaf_fn else arr
            out[path] = (w * arr) if apply_weights else arr
        return out

    def iter_items(self, *, include_root: bool = False):
        """Yield (path, leaf_node) for all leaves (no weights, no reduction)."""
        for path, node, _ in self._walk_leaves(include_root=include_root, apply_weights=False):
            if node.value is not None:
                yield path, node

    def __len__(self):
        return len(self.children)

    def __iter__(self):
        return iter(self.names)

    def __getitem__(self, k: str) -> Self:
        i = self.names.index(k)
        return self.children[i]


def is_termtree(x):
    return isinstance(x, TermTree)
