"""A model Delay's data layout, shared by initialization and raw state recording."""

from dataclasses import dataclass

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import PyTree


@dataclass(frozen=True)
class DelayLeaf:
    path: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: np.dtype

    def carry_shape(self, steps: int) -> tuple[int, ...]:
        return (steps, *self.shape) if steps > 1 else self.shape

    def accepts(self, value: float) -> bool:
        if self.dtype.kind == "b":
            return value in (0, 1)
        if self.dtype.kind in "iu":
            bounds = np.iinfo(self.dtype)
            return value.is_integer() and bounds.min <= value <= bounds.max
        return abs(value) <= np.finfo(self.dtype).max


def _key_part(key: jtu.KeyEntry) -> str:
    match key:
        case jtu.GetAttrKey(name=name):
            return name
        case jtu.DictKey(key=value):
            return str(value)
        case jtu.SequenceKey(idx=index):
            return str(index)
        case jtu.FlattenedIndexKey(key=index):
            return str(index)
        case _:
            return str(key)


@dataclass(frozen=True)
class DelayLayout:
    tree: jtu.PyTreeDef
    leaves: tuple[DelayLeaf, ...]
    steps: int

    @classmethod
    def from_prototype(cls, prototype: PyTree, steps: int) -> "DelayLayout":
        paths, tree = jtu.tree_flatten_with_path(prototype)
        leaves = []
        for path, prototype_leaf in paths:
            if not hasattr(prototype_leaf, "shape") or not hasattr(prototype_leaf, "dtype"):
                raise ValueError("Delay requires a numeric sample at each signal leaf.")
            dtype = np.dtype(prototype_leaf.dtype)
            if dtype.kind not in "biuf":
                raise ValueError(f"Delay cannot carry a {dtype.name} signal leaf.")
            leaves.append(DelayLeaf(tuple(map(_key_part, path)), tuple(prototype_leaf.shape), dtype))
        if not leaves:
            raise ValueError("Delay has no resolved signal layout.")
        if len({leaf.path for leaf in leaves}) != len(leaves):
            raise ValueError("Delay signal leaves do not have distinct paths.")
        return cls(tree, tuple(leaves), steps)

    def validate_initial(self, value: float) -> None:
        for leaf in self.leaves:
            if not leaf.accepts(value):
                key = ".".join(leaf.path)
                field = f" at {key}" if key else ""
                raise ValueError(f"Delay Initial must fit the {leaf.dtype.name} signal{field}.")

    def filled(self, value: float) -> PyTree:
        return self.tree.unflatten([
            jnp.full(leaf.carry_shape(self.steps), value, dtype=leaf.dtype)
            for leaf in self.leaves
        ])

    def seeded(self, value: PyTree) -> PyTree:
        samples, tree = jtu.tree_flatten(value)
        if tree != self.tree or any(
            sample.shape != leaf.shape or sample.dtype != leaf.dtype
            for leaf, sample in zip(self.leaves, samples, strict=True)
        ):
            raise ValueError("Delay Initial does not fit the resolved signal layout.")
        return tree.unflatten([
            jnp.broadcast_to(sample, leaf.carry_shape(self.steps))
            for leaf, sample in zip(self.leaves, samples, strict=True)
        ])
