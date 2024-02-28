"""Tools for dimensionality reduction.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import logging
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Shaped


logger = logging.getLogger(__name__)


class PCAResults(eqx.Module):
    """
    Attributes:
        components: Principal components.
        singular_values: Singular values $S$.
    """

    components: Array
    singular_values: Array


def pca(
    x: Shaped[Array, "*batch features"],
    **kwargs: Any,
) -> PCAResults:
    """Principal component analysis.

    Takes the last axis of `x` as features. Flattens the preceding dimensions,
    then mean subtracts the features and performs singular value decomposition.
    Unflattens the PCs back to the original shape before returning.

    Arguments:
        x: Input features.
        **kwargs: Additional arguments for `jax.numpy.linalg.svd`.
    """
    X = x.reshape(-1, x.shape[-1])
    X -= X.mean(axis=0)
    U, S, Vt = jnp.linalg.svd(X, full_matrices=False, **kwargs)
    # L = S ** 2 / (X.shape[0] - 1)

    return PCAResults(
        components=(U @ jnp.diag(S)).reshape(*x.shape),
        singular_values=S,
    )
