"""Some recipes in JAX + Equinox.

:copyright: Copyright 2024-2025 by MLL
:license: Apache 2.0, see LICENSE for details.
"""

from ._array import (
    ArrayLikeWrapper,
    MaskedArray,
    part_by_idx,
    unwrap_arraylikes,
    unwrap_arraylikes_and_labels,
)
from ._func import (
    allf,
    anyf,
    compose,
    hash_callable,
    identity,
    is_not_type,
    is_type,
    notf,
)
from ._io import arrays_to_lists, load, load_with_hyperparameters, save
from ._ldict import LDict, LDictConstructor
from ._types import (
    is_module,
    is_none,
)
from ._vmap import (
    MultiVmapAxes,
    expand_axes_spec,
    natural_to_sequential_axes,
    unkwarg_key,
    vmap_multi,
)
from ._where import (
    where_attr_strs_to_func,
    where_func_to_strs,
)
from .progress import (
    map_rich,
)

__all__ = [
    "ArrayLikeWrapper",
    "LDict",
    "LDictConstructor",
    "MaskedArray",
    "MultiVmapAxes",
    "allf",
    "anyf",
    "arrays_to_lists",
    "compose",
    "expand_axes_spec",
    "hash_callable",
    "identity",
    "is_module",
    "is_none",
    "is_not_type",
    "is_type",
    "load",
    "load_with_hyperparameters",
    "map_rich",
    "natural_to_sequential_axes",
    "notf",
    "part_by_idx",
    "save",
    "unwrap_arraylikes",
    "unwrap_arraylikes_and_labels",
    "unkwarg_key",
    "vmap_multi",
    "where_attr_strs_to_func",
    "where_func_to_strs",
]
