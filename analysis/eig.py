from collections import namedtuple
from functools import partial

from equinox import field
import jax
import jax.numpy as jnp
import jax.tree as jt
import plotly.graph_objects as go
from jax_cookbook import is_type
from jaxtyping import PyTree, Float, Array

import jax_cookbook.tree as jtree

from rlrmp.analysis.analysis import AbstractAnalysis, AbstractAnalysisPorts, InputOf
from rlrmp.types import AnalysisInputData, LDict, TreeNamespace
from rlrmp.types import Polar


DecompResults = namedtuple("DecompResults", ["vals", "vecs_l", "vecs_r"])


# @partial(jax.jit, device=jax.devices('cpu')[0])
def eig(    
    arr: Float[Array, "... m n"],
    **kwargs,
) -> DecompResults:
    """Compute eigenvalues and eigenvectors of square matrices."""
    eigvals, eigvecs_l, eigvecs_r = jax.lax.linalg.eig(arr, **kwargs)
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