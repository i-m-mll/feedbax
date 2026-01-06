"""Conveniences for debugging; use `from _debug import *`."""

import jax.tree as jt
import plotly.graph_objects as go
from jax_cookbook import is_module, is_none, is_type
from jax_cookbook._array import part_by_idx
from jax_cookbook.tree import (
    first_leaf as fs,
)
from jax_cookbook.tree import (
    first_leaf_shape as fsh,
)

from feedbax._experiments.misc import location_inspect as loc
from feedbax._experiments.tree_utils import (
    ldict_verbose_label_fn,
    tree_level_labels,
)
from feedbax._experiments.tree_utils import (
    pp2 as pp,
)


def tll(*args, **kwargs):
    return tree_level_labels(
        *args,
        label_fn=ldict_verbose_label_fn,
        **kwargs,
    )


def lf(tree, type_=None):
    if type_ is not None:
        is_leaf = is_type(type_)
    else:
        is_leaf = None
    leaves = jt.leaves(tree, is_leaf=is_leaf)
    if not leaves:
        return None
    else:
        return leaves[0]


def lff(tree):
    return lf(tree, is_type(go.Figure))
