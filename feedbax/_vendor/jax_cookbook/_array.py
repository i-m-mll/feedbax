# Modified by Feedbax: internal imports use the bundled package namespace.
from collections.abc import Sequence
from typing import Any, Literal, Optional, TypeVar

import jax.numpy as jnp
import jax.tree as jt
from equinox import Module, field
from jaxtyping import Array, ArrayLike, PyTree

import feedbax._vendor.jax_cookbook.tree as jtree
from feedbax._vendor.jax_cookbook._func import is_type


class MaskedArray(Module):
    """Array with associated boolean mask indicating valid (unmasked) elements.

    This is used to represent arrays where some elements may be padding or invalid
    (e.g., due to alignment operations). The mask is True where data is valid,
    False where it is padding/invalid.

    Note: This is a PyTree node (via equinox.Module). When used with JAX tree
    operations like stacking, the data and mask arrays are transformed separately.
    If you need to treat MaskedArray as an atomic leaf in tree operations, use
    `is_leaf=is_type(MaskedArray)` in your tree traversal.

    Attributes:
        data: The array data (including padded/masked regions).
        mask: Boolean array with same shape as data, True where data is valid.
    """
    data: Array
    mask: Array  # bool array, True where data is valid

    def unwrap(self, invalid_value=jnp.nan):
        """Convert to a regular array with masked (invalid) values replaced.

        Args:
            invalid_value: Value to use for masked elements. Defaults to NaN.

        Returns:
            Array with same shape as `data`, where masked elements are set to `invalid_value`.
        """
        return jnp.where(self.mask, self.data, invalid_value)


class ArrayLikeWrapper(Module):
    """Metadata-carrying wrapper for ArrayLike objects.

    Metadata fields are static, and thus excluded from JAX transformations. Any modifications to
    the metadata to remain valid after such transformations, must be implemented separately.
    """

    value: ArrayLike
    axes_names: Optional[Sequence[str]] = field(default=None, static=True)
    label: Optional[str | Sequence[str]] = field(default=None, static=True)

    def __check_init__(self):
        if self.axes_names is not None:
            if isinstance(self.value, Array):
                if len(self.axes_names) < self.value.ndim:
                    if Ellipsis not in (self.axes_names[0], self.axes_names[-1]):
                        raise ValueError(
                            f"axes_names length ({len(self.axes_names)}) does not match number of "
                            "axes in value ({self.value.ndim}), nor is there a leading/trailing "
                            "ellipsis"
                        )
                    if self.axes_names.count(Ellipsis) > 1:
                        raise ValueError(
                            "axes_names can only contain a single ellipsis (...) at either its start "
                            "or end."
                        )
                elif len(self.axes_names) > self.value.ndim:
                    raise ValueError(
                        f"axes_names length ({len(self.axes_names)}) cannot exceed number of axes "
                        f"in value ({self.value.ndim})"
                    )
                elif Ellipsis in self.axes_names:
                    raise ValueError(
                        "axes_names cannot contain ellipsis when length matches array rank"
                    )
            else:
                raise ValueError("axes_names should not be provided for non-array values")

        if self.label is not None:
            if not isinstance(self.label, str) and not (
                isinstance(self.label, Sequence) and all(isinstance(x, str) for x in self.label)
            ):
                raise TypeError(
                    f"label must be a string or sequence of strings; got {type(self.label)}"
                )


def unwrap_arraylikes(tree: PyTree[ArrayLikeWrapper]) -> PyTree[ArrayLike]:
    """Unwrap ArrayLikeWrapper objects in a PyTree.

    Non-wrapper leaves are passed through unchanged.
    """
    return jt.map(
        lambda x: x.value if isinstance(x, ArrayLikeWrapper) else x,
        tree,
        is_leaf=is_type(ArrayLikeWrapper),
    )


T = TypeVar("T")


def unwrap_arraylikes_and_labels(
    tree: PyTree[ArrayLikeWrapper, "T"],
    label_fmt: Literal["short", "medium", "full"] = "medium",
) -> tuple[PyTree[ArrayLike, "T"], PyTree[Optional[str], "T"]]:
    """Unwrap ArrayLikeWrapper objects in a PyTree and return both values and labels.

    Notes:
        If the wrapper's label is a string, it is returned as-is. If it is a sequence
        of strings, then:
            - "short": first element (or "" if empty)
            - "medium": "/"-joined
            - "full": "/"-joined (same as "medium" for now)
    """

    def _label_for(wrapper: ArrayLikeWrapper | Any) -> Optional[str]:
        if not isinstance(wrapper, ArrayLikeWrapper):
            return None
        label = wrapper.label
        if label is None:
            return None
        if isinstance(label, str):
            return label
        if isinstance(label, Sequence):
            if not label:
                return ""
            if label_fmt == "short":
                return label[0]
            return "/".join(label)
        return str(label)

    return jtree.unzip(
        jt.map(
            lambda x: (x.value, _label_for(x))
            if isinstance(x, ArrayLikeWrapper)
            else (x, None),
            tree,
            is_leaf=is_type(ArrayLikeWrapper),
        )
    )


def part_by_idx(arr: Array, idxs: Array, axis: int = 0) -> tuple[Array, Array]:
    """
    Split `arr` into (selected, remainder) along a specified axis.

    Args:
        arr: Array to split.
        idxs: Indices along `axis` to select.
        axis: Axis along which to select (default: 0).

    Returns:
        selected: arr indexed by `idxs` along `axis`
        remainder: arr with those indices excluded
    """
    # Normalize axis
    axis = axis % arr.ndim
    n = arr.shape[axis]

    # Create boolean mask along the target axis
    mask = jnp.zeros(n, dtype=bool).at[idxs].set(True)

    # Use jnp.take and boolean indexing with `jax.vmap`-compatible broadcasting
    selected = jnp.take(arr, idxs, axis=axis)
    remainder = jnp.compress(~mask, arr, axis=axis)
    return selected, remainder
