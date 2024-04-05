"""

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from collections.abc import Mapping, Sequence, Callable
import logging
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PyTree

from feedbax._model import AbstractModel
from feedbax._staged import AbstractStagedModel

if TYPE_CHECKING:
    from feedbax.intervene import AbstractIntervenor


logger = logging.getLogger(__name__)


def _clear_intervenors_dict(
    intervenors: Mapping[str, Sequence["AbstractIntervenor"]],
    scheduled_only: bool,
):
    """Return a new mapping with all intervenors removed."""
    if scheduled_only:
        return {
            stage:
                [
                    intervenor for intervenor in stage_intervenors
                    if intervenor.label[0] == '_'
                ]
            for stage, stage_intervenors in intervenors.items()
        }
    else:
        return {stage: [] for stage in intervenors}


def remove_intervenors(
    model: AbstractModel,
    where: Callable[[AbstractModel], PyTree] = lambda model: model,
    scheduled_only: bool = False,
) -> AbstractModel:
    """Return a model with all intervenors removed at `where`."""
    return eqx.tree_at(
        where,
        model,
        jax.tree_map(
            lambda submodel: eqx.tree_at(
                lambda submodel: submodel.intervenors,
                submodel,
                _clear_intervenors_dict(submodel.intervenors, scheduled_only),
            ),
            where(model),
            # Can't do `isinstance(x, AbstractModel)` because of circular import
            is_leaf=lambda x: getattr(x, "model_spec", None) is not None,
        ),
    )


def remove_all_intervenors(
    tree: PyTree,
    scheduled_only: bool = False,
) -> PyTree:
    """Return a model with all intervenors removed."""
    # TODO: Avoid importing `AbstractStagedModel`? (circular import issues)
    if isinstance(tree, AbstractStagedModel):
        tree = eqx.tree_at(
            lambda model: model.intervenors,
            tree,
            _clear_intervenors_dict(tree.intervenors, scheduled_only),
        )
    leaves, treedef = eqx.tree_flatten_one_level(tree)
    return jtu.tree_unflatten(treedef, [
        (
            remove_all_intervenors(leaf, scheduled_only)
            if isinstance(leaf, AbstractModel)
            else leaf
        )
        for leaf in leaves
    ])


