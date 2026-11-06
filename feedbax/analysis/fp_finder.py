"""Facilities for finding fixed points of dynamical systems.

Derived from the JAX [implementation](https://github.com/google-research/computation-thru-dynamics/blob/94b47aff4cd029aa9df3d0402047110bc3f3519d/fixed_point_finder/fixed_points.py)
of Fixed Point Finder, by David Sussillo et al.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree as jt
import numpy as np
import optax
from equinox import Module
from feedbax._vendor.jax_cookbook import is_module
from jaxtyping import Array, Bool, Float, PRNGKeyArray, PyTree

from feedbax.analysis.math import squareform_pdist

logger = logging.getLogger(__name__)


class FPFilteredResults(Module):
    fps: Float[Array, "candidates state"]
    losses: Float[Array, "candidates"]  # noqa: F821
    masks: dict[str, Bool[Array, "candidates"]]  # noqa: F821
    counts: dict[str, int]


class FixedPointFinder(Module):
    """JAX-transformable optimizer for batches of fixed-point candidates."""

    optimizer: optax.GradientTransformation
    enable_log: bool = True

    @eqx.filter_jit
    def __call__(
        self,
        func: Callable[[Float[Array, "state"]], Float[Array, "state"]],  # noqa: F821
        candidates: Float[Array, "n_candidates state"],
        loss_tol: Float[Array, "1"],
        n_batches: int = 10_000,
        n_batches_per_iter: int = 200,
        *,
        key: PRNGKeyArray,
    ) -> tuple[int, Float[Array, "n_candidates state"], Float[Array, "n_candidates"]]:  # noqa: F821
        """Optimize trainable candidates to minimize their mean fixed-point MSE.

        Check the current mean loss against `loss_tol` before the first update
        and after each chunk of up to `n_batches_per_iter` updates. Stop when
        the mean loss is at or below the tolerance, no trainable candidates
        remain, or `n_batches` updates have run.

        Freeze candidates with nonfinite states, losses, or gradients and report
        their losses as infinite. A finite candidate with a nonfinite gradient
        keeps its loss if it is already strictly below the tolerance.
        Return the update count, candidates, and losses in ascending loss order.
        The random key is reserved for future stochastic dynamics.
        """
        if n_batches_per_iter <= 0:
            raise ValueError("n_batches_per_iter must be positive.")

        # Differentiate each residual independently so invalid adjoints stay in their own row.
        candidate_value_and_grad = jax.vmap(jax.value_and_grad(_get_candidate_fp_loss_fn(func)))
        loss_tol = jnp.asarray(loss_tol).reshape(())

        def evaluate(params):
            losses, grads = candidate_value_and_grad(params)
            finite = jnp.all(jnp.isfinite(params), axis=1) & jnp.isfinite(losses)
            trainable = finite & jnp.all(jnp.isfinite(grads), axis=1)
            losses = jnp.where(finite & (trainable | (losses < loss_tol)), losses, jnp.inf)
            return losses, grads, trainable

        def cond_fn(state):
            i_batch, _, losses, _, trainable, _ = state

            def above_tolerance():
                mean_loss = jnp.mean(losses, where=trainable)
                return mean_loss > loss_tol

            return (
                (i_batch < n_batches)
                & jnp.any(trainable)
                & jax.lax.cond(i_batch % n_batches_per_iter == 0, above_tolerance, lambda: True)
            )

        def body_fn(state):
            i_batch, params, losses, grads, trainable, opt_state = state
            mask = trainable[:, None]
            accumulation_dtype = jnp.result_type(grads.dtype, jnp.float32)
            count = jnp.maximum(jnp.sum(trainable), 1).astype(accumulation_dtype)
            grads = (jnp.where(mask, grads, 0).astype(accumulation_dtype) / count).astype(
                grads.dtype
            )
            updates, opt_state = self.optimizer.update(grads, opt_state, jnp.where(mask, params, 0))
            params = jnp.where(mask, optax.apply_updates(params, updates), params)
            new_losses, grads, new_trainable = evaluate(params)
            losses = jnp.where(trainable, new_losses, losses)
            return i_batch + 1, params, losses, grads, trainable & new_trainable, opt_state

        losses, grads, trainable = evaluate(candidates)
        opt_state = self.optimizer.init(jnp.where(trainable[:, None], candidates, 0))
        initial_state = 0, candidates, losses, grads, trainable, opt_state
        i_batch, fps, losses, _, _, _ = jax.lax.while_loop(cond_fn, body_fn, initial_state)
        loss_sort_idxs = jnp.argsort(losses)

        return i_batch, fps[loss_sort_idxs], losses[loss_sort_idxs]

    @eqx.filter_jit
    def find_and_filter(
        self,
        func: Callable[[Float[Array, "state"]], Float[Array, "state"]],  # noqa: F821
        candidates: Float[Array, "n_candidates state"],
        loss_tol: Float[Array, "1"] = jnp.array(1e-6),
        n_batches: int = 10_000,
        n_batches_per_iter: int = 200,
        outlier_tol: Optional[float] = None,
        unique_tol: Optional[float] = None,
        distance_norm: int | str | None = 2,
        verbose: bool = False,
        *,
        key: PRNGKeyArray,  # TODO
    ) -> FPFilteredResults:
        """Finds fixed points and filters by loss, and (optionally) duplicates and outliers."""
        n_updates, fps, losses = self(
            func, candidates, loss_tol, n_batches, n_batches_per_iter, key=key
        )
        if self.enable_log:
            jax.debug.callback(self._log, candidates.shape[0], n_updates)
        masks, counts = exclude_points(
            fps, losses, loss_tol, outlier_tol, unique_tol, distance_norm, verbose
        )
        return FPFilteredResults(fps, losses, masks, counts)

    def _log(self, n_candidates, n_updates):
        logger.debug(
            "Optimization of %d candidates finished after %d updates.", n_candidates, n_updates
        )


def exclude_points(
    points: Float[Array, "points dims"],
    values: Float[Array, "points"],  # noqa: F821
    value_tol: Float[Array, "1"],
    outlier_tol: Optional[float] = None,
    unique_tol: Optional[float] = None,
    ord: int | str | None = 2,
    verbose: bool = False,
):
    """Filter finite points in stages and return cumulative masks and pass counts.

    First require a finite value strictly below `value_tol`. If `outlier_tol`
    is set and multiple points pass, require another passing point within that
    distance. Then, if `unique_tol` is set, consider the survivors in increasing
    value order, breaking ties by input order. Retain a point if no previously
    retained point is within `unique_tol`.

    Both distance thresholds are inclusive and use the norm specified by `ord`.
    The masks align with the input rows. A disabled stage passes its input
    through. When verbose, report how many points each stage rejected.
    """
    meets_tolerance = jnp.isfinite(values) & (values < value_tol)
    meets_tolerance &= jnp.all(jnp.isfinite(points), axis=1)
    not_outlier = meets_tolerance
    unique = meets_tolerance
    n = len(points)

    if n > 1 and (outlier_tol is not None or unique_tol is not None):
        distances = squareform_pdist(points, ord=ord)

        if outlier_tol is not None:
            neighbors = (distances <= outlier_tol) & ~jnp.eye(n, dtype=bool)
            has_neighbor = jnp.any(neighbors & meets_tolerance[None, :], axis=1)
            not_outlier = meets_tolerance & (has_neighbor | (jnp.sum(meets_tolerance) == 1))

        unique = not_outlier
        if unique_tol is not None:
            order = jnp.argsort(values, stable=True)

            def choose(available, idx):
                keep = available[idx]
                available = available & ~(keep & (distances[idx] <= unique_tol))
                return available, keep

            _, keep_in_order = jax.lax.scan(choose, not_outlier, order)
            unique = jnp.zeros_like(not_outlier).at[order].set(keep_in_order)

    masks = {
        "meets_all_criteria": unique,
        "meets_tolerance": meets_tolerance,
        "not_outlier": not_outlier,
        "unique": unique,
    }
    counts = jt.map(jnp.sum, masks)

    if verbose:
        jax.debug.print(
            "Filtering {total} points: {invalid} failed tolerance, "
            "{outliers} outliers, {duplicates} duplicates, {accepted} accepted.",
            total=n,
            invalid=n - counts["meets_tolerance"],
            outliers=counts["meets_tolerance"] - counts["not_outlier"],
            duplicates=counts["not_outlier"] - counts["unique"],
            accepted=counts["meets_all_criteria"],
        )

    return masks, counts


def fp_adam_optimizer(
    learning_rate: float = 0.2,
    decay_steps: int = 1,
    decay_rate: float = 0.9999,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-5,
):
    """Default Adam optimizer for finding fixed points.

    Incorporates an exponential learning rate decay schedule.
    """
    schedule = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=decay_steps,
        decay_rate=decay_rate,
    )
    optimizer = optax.adam(
        learning_rate=schedule,
        b1=b1,
        b2=b2,
        eps=eps,
    )
    return optimizer


def get_fp_loss_fn(func):
    """Returns the batched fixed point MSE loss for a given function."""
    return jax.vmap(_get_candidate_fp_loss_fn(func))


def _get_candidate_fp_loss_fn(func):
    def loss_fn(x):
        return jnp.mean((x - func(x)) ** 2)

    return loss_fn


def get_total_fp_loss_fn(func):
    loss_fn = get_fp_loss_fn(func)

    def total_loss_fn(x):
        return jnp.mean(loss_fn(x))

    return total_loss_fn


N_KEEP_MAX = 10


def take_top_fps(
    fpf_results: PyTree[FPFilteredResults],
    n_keep: int = 1,
    warn: bool = True,
):
    """Slice out the leading FPs.

    Argument:
        fpf_results: Tree of fixed point finder results, containing all optimized candidate fixed points.
        n_keep: The number of top FPs to keep. The maximum number that can be returned is 10.
        warn: Whether to log a warning if some qualifying FPs are excluded from the result.
    """

    @partial(jnp.vectorize, signature="(n,m),(n)->(p,m)")
    def take_top_fp(fps, mask):
        # Get the actual count of True values
        n_matches = jnp.sum(mask)
        # Get fixed-size indices using size parameter
        idxs = jnp.where(mask, size=N_KEEP_MAX)[0]  # Or whatever max size needed
        # Create a mask for valid indices (< n_matches and < n_keep)
        idx_mask = jnp.arange(n_keep) < jnp.minimum(n_matches, n_keep)
        # Create output array of fixed size n_keep
        result = jnp.full((n_keep, fps.shape[1]), jnp.nan)
        # Only set values where idx_mask is True
        result = result.at[
            jnp.arange(n_keep)
        ].set(
            jnp.where(
                idx_mask[..., None],  # Shape will be (5, 1) to broadcast with (5, 100)
                fps[idxs][:n_keep],
                result,
            )
        )
        return result

    if warn:
        # get the idxs of any conditions that have more or less than 1 FP
        idxs_multi_fps = jt.map(
            lambda results: np.where(results.counts["meets_all_criteria"] > n_keep),
            fpf_results,
            is_leaf=is_module,
        )
        # TODO: Warn how many FPs, and give the indices
        if np.any(np.concatenate(jt.leaves(idxs_multi_fps))):
            logger.warning("Some FPs satisfied criteria but were omitted")

    return jt.map(
        lambda results: take_top_fp(results.fps, results.masks["meets_all_criteria"]),
        fpf_results,
        is_leaf=is_module,
    )
