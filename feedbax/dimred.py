"""

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import logging

import jax.numpy as jnp


logger = logging.getLogger(__name__)


def pca(x):
    """Principal component analysis.
    
    Takes the last axis of `x` as features. Flattens the preceding dimensions,
    then mean subtracts the features and performs singular value decomposition.
    Unflattens the PCs back to the original shape before returning.
    """    
    X = x.reshape(-1, x.shape[-1])
    X -= X.mean(axis=0)
    U, S, Vt = jnp.linalg.svd(X, full_matrices=False)
    L = S * 2 / (X.shape[0] - 1)
    PCs = (U @ jnp.diag(S)).reshape(*x.shape)
    return L, Vt, PCs 