from collections.abc import Mapping
from functools import wraps
from types import MappingProxyType
from typing import Literal as L
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
import numpy as np
from feedbax.misc import batch_reshape  # for flattening/unflattening
from jax_cookbook.misc import crop_to_shortest
from jaxtyping import Array
from sklearn.neighbors import KDTree

from feedbax.analysis.analysis import (
    AbstractAnalysis,
    AbstractAnalysisPorts,
    Data,
    InputOf,
)
from feedbax.tree_utils import getitem_at_level
from feedbax.types import AnalysisInputData


class TanglingPorts(AbstractAnalysisPorts):
    """Input ports for Tangling analysis."""

    state: InputOf[Array]


class Tangling(AbstractAnalysis[TanglingPorts]):
    Ports = TanglingPorts
    inputs: TanglingPorts = eqx.field(
        default_factory=TanglingPorts, converter=TanglingPorts.converter
    )
    fig_params: Mapping = MappingProxyType(dict())
    variant: Optional[str] = None
    eps: float = 1e-6  # TODO: Allow for `lambda states: ...`
    t_axis: int = -2  # time step axis in arrays
    method: L["direct", "kdtree"] = "direct"
    # Number of nearest neighbours to consider for the KD-tree variant. This is
    # a compromise between speed and the probability of hitting the true
    # maximiser – in practice 10–50 is usually enough because the tangling
    # ratio falls off quickly with distance.
    k_neighbours: int = 50
    # Leaf size forwarded to ``sklearn.neighbors.KDTree``. Kept here so that it
    # can be tuned from configs if desired.
    leaf_size: int = 40

    def compute(self, data: AnalysisInputData, *, state, hps_common, **kwargs) -> dict:
        #! Should probably be hps_common.dt, top-level
        dt = hps_common.train.model.dt
        if self.variant is not None:
            state = getitem_at_level("task_variant", self.variant, state)
        flow = jt.map(lambda x: self._flow_field(x, dt), state)
        tangling = jt.map(lambda x, dxdt: self._tangling(x, dxdt), state, flow)
        return tangling

    def _tangling(self, x: Array, dxdt: Array) -> Array:
        # Delegate to the decorated core function which handles
        # timestep-cropping and batch flattening automatically.
        return _get_tangling_core(
            self.eps,
            self.method,
            self.k_neighbours,
            self.leaf_size,
            t_axis=self.t_axis,
        )(x, dxdt)

    def _flow_field(self, arr: Array, dt: float) -> Array:
        # Simple finite difference approximation of the flow field.
        # Assume `arr` has shape (..., timestep, dim)
        # return (arr[..., 1:, :] - arr[..., :-1, :]) / dt
        arr_prev = jax.lax.slice_in_dim(arr, start_index=0, limit_index=-1, axis=self.t_axis)
        arr_next = jax.lax.slice_in_dim(arr, start_index=1, limit_index=None, axis=self.t_axis)
        return (arr_next - arr_prev) / dt


@jax.jit
def _tangling_direct(x_flat: jnp.ndarray, v_flat: jnp.ndarray, eps: float) -> jnp.ndarray:
    """O(N²) reference implementation using full pairwise differences."""

    # Broadcast differences: Δx_{ij} and Δv_{ij}
    dx = x_flat[:, None, :] - x_flat[None, :, :]  # (N, N, dim)
    dv = v_flat[:, None, :] - v_flat[None, :, :]

    dist2 = jnp.sum(dx**2, axis=-1) + eps  # (N, N)
    vel2 = jnp.sum(dv**2, axis=-1)  # (N, N)

    ratio = vel2 / dist2  # (N, N)

    # Maximum over the *second* axis gives the tangling for every i.
    return jnp.max(ratio, axis=1)


def _tangling_kdtree(
    x_flat: jnp.ndarray,
    v_flat: jnp.ndarray,
    *,
    eps: float,
    k: int = 50,
    leaf_size: int = 40,
) -> jnp.ndarray:
    """Approximate O(N log N) computation using a KD-tree.

    CPU side:  *scikit-learn* KD-tree query giving neighbour ``idxs`` and
    squared distances ``dists``.
    JIT side:  heavy arithmetic (gather, differences, reductions) performed
    by :func:`_tangling_from_neighbors` which is `@jax.jit`-compiled.
    """

    # ---------- CPU: build / query KD-tree --------------------------------
    x_np = np.asarray(x_flat)  # zero-copy for host arrays
    n_samples = x_np.shape[0]

    # Include self – we will drop it right away but this guarantees at least
    # one neighbour even if k == 1.
    k_eff = min(k + 1, n_samples)

    tree = KDTree(x_np, leaf_size=leaf_size)
    dists, idxs = tree.query(x_np, k=k_eff)

    # Remove self-index (idxs[:,0] == i)
    idxs = idxs[:, 1:]
    dists = dists[:, 1:]

    # Convert to JAX arrays *once* – afterwards everything happens in jit land.
    idxs_jax = jnp.asarray(idxs, dtype=jnp.int32)
    dists_jax = jnp.asarray(dists)

    return _tangling_from_neighbors(x_flat, v_flat, idxs_jax, dists_jax, eps)


@jax.jit
def _tangling_from_neighbors(
    x: jnp.ndarray,
    v: jnp.ndarray,
    idxs: jnp.ndarray,
    dists: jnp.ndarray,
    eps: float,
) -> jnp.ndarray:
    """Compute tangling given neighbour indices & distances (JIT-friendly)."""

    # Gather neighbour velocities (N, k, dim)
    v_neigh = v[idxs]
    v_owner = v[:, None, :]

    dv = v_owner - v_neigh
    vel2 = jnp.sum(dv**2, axis=-1)

    dist2 = dists**2 + eps

    ratios = vel2 / dist2

    return jnp.max(ratios, axis=1)


def _get_tangling_core(
    eps: float,
    method: str,
    k_neigh: int,
    leaf_size: int,
    t_axis: int,
):
    @crop_to_shortest(axis=t_axis)
    @batch_reshape
    def _tangling_core(
        x: jnp.ndarray,
        dxdt: jnp.ndarray,
    ) -> jnp.ndarray:
        """Vectorised core tangling computation callable from the class method."""

        # At this point `batch_reshape` has collapsed *all* leading dims (including
        # the time axis) into a single dimension, so `x` and `dxdt` have shape
        # (N, dim).

        if method == "direct":
            return _tangling_direct(x, dxdt, eps)
        elif method == "kdtree":
            return _tangling_kdtree(
                x,
                dxdt,
                eps=eps,
                k=k_neigh,
                leaf_size=leaf_size,
            )
        else:
            raise ValueError(f"Unknown tangling method '{method}'.")

    return _tangling_core
