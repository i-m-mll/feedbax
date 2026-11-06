from collections import namedtuple

import jax
import jax.numpy as jnp
import jax.tree as jt
from equinox import field
from jaxtyping import Array, Float, Inexact

import feedbax._vendor.jax_cookbook.tree as jtree
from feedbax._vendor.jax_cookbook import LDict
from feedbax.analysis.analysis import AbstractAnalysis, AbstractAnalysisPorts, InputOf
from feedbax.analysis.types import AnalysisInputData

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


def svd(arr: Float[Array, "... m n"], **kwargs) -> DecompResults:
    """Compute singular values and vectors of matrices."""
    singvecs_l, singvals, singvecs_r_adj = jax.lax.linalg.svd(arr, **kwargs)
    return DecompResults(singvals, singvecs_l, singvecs_r_adj.conj().T)


class DecompPorts(AbstractAnalysisPorts):
    matrices: InputOf[Float[Array, "... m n"]]


class SquareDecompPorts(AbstractAnalysisPorts):
    """Input ports for Eigendecomposition analysis."""

    matrices: InputOf[Float[Array, "... m m"]]


class Eig(AbstractAnalysis[SquareDecompPorts]):
    Ports = SquareDecompPorts
    inputs: SquareDecompPorts = field(
        default_factory=SquareDecompPorts, converter=SquareDecompPorts.converter
    )

    def compute(
        self,
        data: AnalysisInputData,
        *,
        matrices,
        **kwargs,
    ):
        return jtree.unzip(jt.map(eig, matrices), tuple_cls=DecompResults)


class SVD(AbstractAnalysis[DecompPorts]):
    Ports = DecompPorts
    inputs: DecompPorts = field(default_factory=DecompPorts, converter=DecompPorts.converter)

    def compute(
        self,
        data: AnalysisInputData,
        *,
        matrices,
        **kwargs,
    ):
        return jtree.unzip(jt.map(svd, matrices), tuple_cls=DecompResults)


def complex_to_polar_abs_angle(arr: Array) -> LDict[str, Array]:
    """
    Converts complex numbers to polar coordinates with symmetric angles.

    This is useful when working with the distribution of eigenvalues of discrete
    systems, where the absolute angle describes frequency and the magnitude describes
    stability.

    Args:
      arr: A JAX array of complex numbers with shape (..., n).

    Returns:
      A tuple containing two JAX arrays:
      - angles: The angles in radians, mapped to the interval [0, pi],
                symmetric about the real axis. Shape (..., n).
      - magnitudes: The magnitudes (radii) of the complex numbers.
                    Shape (..., n).
    """
    # Calculate magnitudes (absolute values)
    magnitude = jnp.abs(arr)
    # Calculate standard angles in (-pi, pi] and take absolute value for [0, pi]
    angle = jnp.abs(jnp.angle(arr))
    return LDict.of("component")(dict(angle=angle, magnitude=magnitude))
    # return jnp.stack([angles, magnitudes], axis=0)
