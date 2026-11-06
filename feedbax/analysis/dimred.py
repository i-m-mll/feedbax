"""Tools for dimensionality reduction.

:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from math import prod
from typing import Any

import equinox as eqx
from jaxtyping import Array, Shaped

from feedbax.analysis.pca import _fit_pca


class PCAResults(eqx.Module):
    """
    Attributes:
        components: Principal-component scores with shape ``(*batch, rank)``.
        singular_values: Singular values $S$.
    """

    components: Array
    singular_values: Array


def pca(
    x: Shaped[Array, "*batch features"],
    **kwargs: Any,
) -> PCAResults:
    """Principal component analysis.

    Takes the last axis of `x` as features and flattens all leading axes into
    samples. Centers each feature and uses a reduced SVD. The scores retain the
    leading shape and replace the feature axis with the number of returned
    singular values, normally `min(samples, features)`. Score signs follow the SVD.

    Arguments:
        x: Input features.
        **kwargs: Additional arguments for `jax.numpy.linalg.svd`.
    """
    *batch_shape, feature_count = x.shape
    samples = x.reshape(prod(batch_shape), feature_count)
    _, _, (left, singular_values, _) = _fit_pca(samples, svd_options=kwargs)

    return PCAResults(
        components=(left * singular_values).reshape(*batch_shape, singular_values.size),
        singular_values=singular_values,
    )
