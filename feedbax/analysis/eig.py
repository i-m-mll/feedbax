"""Batched eigenvalues and eigenvector bases."""

from collections import namedtuple

import jax
from jaxtyping import Array, Inexact

DecompResults = namedtuple("DecompResults", ["vals", "vecs_l", "vecs_r"])


def eig(
    arr: Inexact[Array, "... n n"],
    *,
    compute_left_eigenvectors: bool = True,
    compute_right_eigenvectors: bool = True,
    **kwargs,
) -> DecompResults:
    """Compute eigenvalues and requested eigenvectors of square matrices.

    The basis fields contain column eigenvectors in eigenvalue order, or
    `None` when that basis is omitted. The left columns are eigenvectors of
    the adjoint for the conjugated eigenvalues. Keep the output flags static
    under JIT. Additional keywords are forwarded to `jax.lax.linalg.eig`.
    """
    outputs = iter(
        jax.lax.linalg.eig(
            arr,
            compute_left_eigenvectors=compute_left_eigenvectors,
            compute_right_eigenvectors=compute_right_eigenvectors,
            **kwargs,
        )
    )
    eigvals = next(outputs)
    eigvecs_l = next(outputs) if compute_left_eigenvectors else None
    eigvecs_r = next(outputs) if compute_right_eigenvectors else None
    return DecompResults(eigvals, eigvecs_l, eigvecs_r)
