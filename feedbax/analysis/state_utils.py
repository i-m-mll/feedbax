from functools import cache

import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt
import numpy as np
from equinox import filter_vmap as vmap
from feedbax._vendor.jax_cookbook import LDict, MaskedArray, is_type
from jaxtyping import Array, Float, PyTree

from feedbax.config.defaults import REPLICATE_CRITERION


def _get_replicate_idxs_fn(replicate_info, key):
    def _replicate_idxs_fn(std):
        return replicate_info[std][key][REPLICATE_CRITERION]

    return _replicate_idxs_fn


def get_best_replicate(tree, *, replicate_info, axis: int = 1, keep_axis: bool = False, **kwargs):
    _original_replicate_idxs_fn = _get_replicate_idxs_fn(replicate_info, "best_replicates")

    if keep_axis:
        _replicate_idxs_fn = lambda std: jnp.array([_original_replicate_idxs_fn(std)])
    else:
        _replicate_idxs_fn = _original_replicate_idxs_fn

    def _take_best_replicate(x, std):
        if eqx.is_array(x):
            return jnp.take(x, _replicate_idxs_fn(std), axis=axis)
        else:
            return x

    def _process_node(node):
        if LDict.is_of("train__pert__std")(node):
            return LDict.of("train__pert__std")(
                {
                    std: jt.map(lambda x: _take_best_replicate(x, std), subtree)
                    for std, subtree in node.items()
                }
            )
        elif eqx.is_array(node):
            #! This is for the case where the user passes a single
            return jnp.take(node, replicate_info["best_replicates"][REPLICATE_CRITERION], axis=axis)
        else:
            return node

    return jt.map(
        _process_node,
        tree,
        is_leaf=LDict.is_of("train__pert__std"),
    )


def get_align_epoch_start(
    epoch_starts: Array,
    *,
    time_axis: int = -2,
    trial_axis: int = -3,
    anchor: int | str = "max",
    pad_value=0,
):
    """Return a transform aligning each trial's chosen epoch start to an output index.

    The anchor can be the latest start ("max"), earliest start ("min"), or a
    fixed index. Samples shifted before index zero are discarded. The common
    output time length is the input length plus the largest rightward shift,
    if any. Each result leaf is a MaskedArray with a same-shape validity mask.
    """

    def align_epochs(values):
        start_idxs = np.asarray(epoch_starts, dtype=np.int64)
        min_start = int(start_idxs.min())

        if anchor == "max":
            anchor_idx = int(start_idxs.max())
        elif anchor == "min":
            anchor_idx = min_start
        elif isinstance(anchor, str):
            raise ValueError(f"Unsupported anchor='{anchor}'. Use 'max', 'min', or an int.")
        else:
            anchor_idx = int(anchor)

        max_right_shift = max(0, anchor_idx - min_start)

        @cache
        def _alignment_indices(n_time):
            output_time = jnp.arange(n_time + max_right_shift)
            # Shifts at or below -n_time discard the entire trial.
            bounded_anchor = max(anchor_idx, min_start - n_time)
            shifts = jnp.asarray(np.maximum(bounded_anchor - start_idxs, -n_time))
            source_time = output_time[None, :] - shifts[:, None]
            valid = (0 <= source_time) & (source_time < n_time)
            return jnp.where(valid, source_time, 0), valid

        def _align_vars(states):
            x = jnp.moveaxis(states, (trial_axis, time_axis), (0, 1))
            n_trials, n_time, *feature_shape = x.shape
            if n_trials != start_idxs.size:
                raise ValueError(
                    f"Trial axis size ({n_trials}) does not match start_idxs ({start_idxs.size})."
                )

            source_time, valid = _alignment_indices(n_time)
            valid = valid.reshape(valid.shape + (1,) * len(feature_shape))
            output_shape = (*source_time.shape, *feature_shape)
            pad = jnp.asarray(pad_value, dtype=x.dtype)
            if n_time:
                trial = jnp.arange(n_trials)[:, None]
                aligned = jnp.where(valid, x[trial, source_time], pad)
            else:
                aligned = jnp.broadcast_to(pad, output_shape)
            masks = jnp.broadcast_to(valid, output_shape)
            return MaskedArray(
                jnp.moveaxis(aligned, (0, 1), (trial_axis, time_axis)),
                jnp.moveaxis(masks, (0, 1), (trial_axis, time_axis)),
            )

        return jt.map(_align_vars, values)

    return align_epochs


def trim_tails(
    tree: PyTree[MaskedArray],
    tolerance: float = 1.0,
    time_axis: int = -2,
    trial_axis: int = -3,
) -> PyTree[MaskedArray]:
    """Trim ragged ends of MaskedArray instances to prevent choppy aggregations.
    
    When trials have different valid lengths after alignment (e.g., from
    `get_align_epoch_start`), aggregated plots show discontinuities as trials
    drop out. This function truncates to a consistent endpoint across all trials.
    
    Args:
        tree: PyTree of MaskedArray instances to trim
        tolerance: Fraction of trials required to be valid at a timestep.
            1.0 = truncate at the shortest trial's endpoint
            0.9 = keep timesteps where ≥90% of trials remain valid
        time_axis: Axis along which to truncate (default -2)
        trial_axis: Axis representing trials/replicates (default -3)
    
    Returns:
        PyTree of MaskedArray instances with trimmed data and masks
    """
    # Collect all MaskedArray instances
    masked_arrays = jt.leaves(tree, is_leaf=is_type(MaskedArray))
    
    if not masked_arrays:
        return tree
    
    # Find minimum valid timesteps to keep across all arrays
    min_valid_steps = float('inf')
    
    for ma in masked_arrays:
        # Normalize axes to positive indices
        ndim = ma.mask.ndim
        time_ax = time_axis if time_axis >= 0 else ndim + time_axis
        trial_ax = trial_axis if trial_axis >= 0 else ndim + trial_axis
        
        # Move trial and time axes to front: (n_trials, n_timesteps, ...)
        mask_reordered = jnp.moveaxis(ma.mask, [trial_ax, time_ax], [0, 1])
        
        # Check validity per (trial, timestep) by reducing over other dimensions
        axes_to_check = tuple(range(2, mask_reordered.ndim))
        if axes_to_check:
            # If a trial has ANY valid data at a timestep, count it as valid
            valid_per_trial_time = jnp.any(mask_reordered, axis=axes_to_check)
        else:
            valid_per_trial_time = mask_reordered
        # Shape: (n_trials, n_timesteps)
        
        # Count how many trials are valid at each timestep
        n_trials = valid_per_trial_time.shape[0]
        trials_valid_per_timestep = jnp.sum(valid_per_trial_time, axis=0)  # Shape: (n_timesteps,)
        
        # Apply tolerance threshold
        threshold = n_trials * tolerance
        
        # Find last timestep where enough trials are valid
        valid_timesteps = jnp.where(trials_valid_per_timestep >= threshold)[0]
        if valid_timesteps.size > 0:
            # +1 to convert from index to length (exclusive end)
            last_valid_timestep = int(valid_timesteps[-1]) + 1
            min_valid_steps = min(min_valid_steps, last_valid_timestep)
    
    # If no valid timesteps found or no truncation needed, return unchanged
    if min_valid_steps == float('inf') or min_valid_steps == 0:
        return tree
    
    # Truncate all MaskedArray instances to min_valid_steps
    def _trim_ma(x):
        if isinstance(x, MaskedArray):
            time_ax = time_axis if time_axis >= 0 else x.data.ndim + time_axis
            # Build slice tuple: all axes get :, time axis gets :min_valid_steps
            slices = tuple(
                slice(None, min_valid_steps) if i == time_ax else slice(None)
                for i in range(x.data.ndim)
            )
            return MaskedArray(
                data=x.data[slices],
                mask=x.mask[slices],
            )
        return x
    
    return jt.map(_trim_ma, tree, is_leaf=is_type(MaskedArray))


def unsqueezer(axis: int):
    """Return a function that unsqueezes an array along the specified axis."""

    def _unsqueeze(x):
        return jnp.expand_dims(x, axis=axis)

    return _unsqueeze


frob = lambda x: jnp.linalg.norm(x, axis=(-1, -2), ord="fro")


def output_corr(
    activities: Float[Array, "evals replicates conditions time hidden"],
    weights: Float[Array, "replicates outputs hidden"],
):
    # center the activities in time
    activities = activities - jnp.mean(activities, axis=-2, keepdims=True)

    def corr(x, w):
        z = jnp.dot(x, w.T)
        return frob(z) / (frob(w) * frob(x))

    corrs = vmap(
        # Vmap over evals and reach conditions (activities only)
        vmap(vmap(corr, in_axes=(0, None)), in_axes=(0, None)),
        # Vmap over replicates (appears in both activities and weights)
        in_axes=(1, 0),
    )(activities, weights)

    # Return the replicate axis to the same position as in `activities`
    return jnp.moveaxis(corrs, 0, 1)
