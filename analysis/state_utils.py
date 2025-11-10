from collections.abc import Callable
from functools import partial
from types import MappingProxyType
from typing import ClassVar, Optional

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import jax_cookbook.tree as jtree
from equinox import filter_vmap as vmap
from feedbax.intervene import AbstractIntervenor
from feedbax.task import AbstractTask
from jax import lax
from jax_cookbook import MaskedArray, is_module, is_type
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from feedbax_experiments.analysis.analysis import get_validation_trial_specs
from feedbax_experiments.constants import REPLICATE_CRITERION
from feedbax_experiments.misc import dynamic_slice_with_padding
from feedbax_experiments.types import LDict, TreeNamespace


def angle_between_vectors(v2, v1):
    """Return the signed angle between two 2-vectors."""
    return jnp.arctan2(
        v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0],
        v1[..., 0] * v2[..., 0] + v1[..., 1] * v2[..., 1],
    )


def get_lateral_distance(
    pos: Float[Array, "*batch conditions time xy=2"],
    pos_endpoints: Float[Array, "point=2 conditions xy=2"],
) -> Float[Array, "*batch conditions time"]:
    """Compute the lateral distance of points from the straight line connecting init and goal positions.

    Arguments:
        pos: Trajectories of positions.
        pos_endpoints: Initial and goal reference positions for each condition.

    Returns:
        Trajectories of lateral distances to the straight line between endpoints.
    """
    init_pos, goal_pos = pos_endpoints

    # Calculate the vectors from 1) inits to goals, and 2) inits to trajectory positions
    direction_vec = goal_pos - init_pos
    point_vec = pos - init_pos[..., None, :]

    # Calculate the cross product between the line vector and the point vector
    # This is the area of the parallelogram they form.
    cross_product = jnp.cross(direction_vec[..., None, :], point_vec)

    # Obtain the parallelogram heights (i.e. the lateral distances) by dividing
    # by the length of the line vectors.
    line_length = jnp.linalg.norm(direction_vec, axis=-1)
    # lateral_dist = jnp.abs(cross_product) / line_length
    lateral_dist = cross_product / line_length[..., None]

    return lateral_dist


def get_pos_endpoints(trial_specs):
    """Given a set of `SimpleReaches` trial specifications, return the stacked start and end positions."""
    return jnp.stack(
        [
            trial_specs.inits["mechanics.effector"].pos,
            jnp.take(trial_specs.targets["mechanics.effector.pos"].value, -1, axis=-2),
        ],
        axis=0,
    )


def _get_eval_ensemble(models, task: AbstractTask):
    def eval_ensemble(key):
        return task.eval_ensemble(
            models,
            n_replicates=jtree.infer_batch_size(models, exclude=is_type(AbstractIntervenor)),
            # Each member of the model ensemble will be evaluated on the same trials
            ensemble_random_trials=False,
            key=key,
        )

    return eval_ensemble


@eqx.filter_jit
def vmap_eval_ensemble(
    key: PRNGKeyArray,
    hps: TreeNamespace,
    models: eqx.Module,
    task: AbstractTask,
):
    """Evaluate an ensemble of models on `n` random repeats of a task's validation set."""
    return eqx.filter_vmap(_get_eval_ensemble(models, task))(jr.split(key, hps.task.eval_n))


def get_constant_task_input_fn(x, n_steps, n_trials):
    return lambda trial_spec, key: (jnp.full((n_trials, n_steps), x, dtype=float))


def get_step_task_input_fn(x1, x2, step_step, n_steps, n_trials):
    def input_fn(trial_spec, key):
        # Create array of x1 values
        inputs = jnp.full((n_trials, n_steps), x1, dtype=float)
        inputs = inputs.at[:, step_step:].set(x2)

        return inputs

    return input_fn


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


get_best_model_replicate = jtree.filter_wrap(
    lambda x: not is_type(AbstractIntervenor)(x),
    is_leaf=is_type(AbstractIntervenor),
)(partial(get_best_replicate, axis=0, keep_axis=True))
get_best_model_replicate.__doc__ = """Variant of `get_best_replicate` that filters out intervenors from the tree.
    
    This is necessary when getting the best replicate from a tree of models, rather than states,
    since intervention parameters generally do not have a replicate batch axis. 
    """


def exclude_bad_replicates(tree, *, replicate_info, axis=0):
    _replicate_idxs_fn = _get_replicate_idxs_fn(replicate_info, "included_replicates")

    def _process_std_subtree(tree_by_std):
        return LDict.of("train__pert__std")(
            {
                std: jt.map(
                    #! TODO: Store included replicates as ints (not bools) in the first place!
                    lambda arr: jnp.take(arr, _replicate_idxs_fn(std).nonzero()[0], axis=axis),
                    subtree,
                )
                for std, subtree in tree_by_std.items()
            }
        )

    #
    return jt.map(
        _process_std_subtree,
        tree,
        is_leaf=LDict.is_of("train__pert__std"),
    )


# def get_align_epoch_start(epoch_idx: int):
# def align_epochs(vars, *, data):
#     def _align_vars_by_task(task, all_states_for_task):
#         trial_specs = task.validation_trials
#         if trial_specs.timeline.epoch_bounds is None:
#             raise ValueError("No epoch bounds defined in the task timeline.")
#         start_idxs = trial_specs.timeline.epoch_bounds[:, epoch_idx]

#         # TODO: Simple case: pad trials to same length such that the epoch start idxs are all
#         # TODO: at the same index in the resulting array. Also return a mask of the same shape,
#         # TODO: indicating which elements correspond to actual trial data, versus padding.
#         def _align_vars(states):
#             pass

#         return jt.map(_align_vars, all_states_for_task, is_leaf=is_module)

#     return jt.map(_align_vars_by_task, data.tasks, vars, is_leaf=is_module)

# return align_epochs


def get_align_epoch_start(
    epoch_idx: int,
    *,
    time_axis: int = -2,
    trial_axis: int = -3,
    anchor: int | str = "max",  # "max" | "min" | int
    pad_value=0,
):
    """
    Returns a function align_epochs(vars, *, data) that:
      - Pads/aligns each replicate's time axis so the chosen epoch's start aligns across replicates.
      - Returns a PyTree mirroring `vars`, where each leaf is a MaskedArray instance.
    """

    def _norm_axis(ax: int, ndim: int) -> int:
        return ax if ax >= 0 else ax + ndim

    def align_epochs(vars, *, data):
        def _align_vars_by_task(task, all_states_for_task):
            # Grab per-replicate start indices for the requested epoch
            trial_specs = task.validation_trials
            if trial_specs.timeline.epoch_bounds is None:
                raise ValueError("No epoch bounds defined in the task timeline.")
            # shape: (n_replicates, n_epochs+1?) -> we take the 'start' column for the epoch
            start_idxs = jnp.asarray(
                trial_specs.timeline.epoch_bounds[:, epoch_idx], dtype=jnp.int32
            )

            # Compute the anchor index (in output time coordinates) and per-replicate left padding.
            if isinstance(anchor, str):
                if anchor == "max":
                    anchor_idx = int(jnp.max(start_idxs))
                elif anchor == "min":
                    anchor_idx = int(jnp.min(start_idxs))
                else:
                    raise ValueError(f"Unsupported anchor='{anchor}'. Use 'max', 'min', or an int.")
            else:
                anchor_idx = int(anchor)

            # Left pad needed so each replicate's epoch start lands at anchor_idx
            # (For 'max', these are >= 0; for a custom anchor, negative values mean you'd be cropping;
            # here we clamp to 0 to preserve padding semantics. You can change this if you prefer cropping.)
            raw_left = anchor_idx - start_idxs
            pad_left_per_rep = jnp.maximum(raw_left, 0).astype(jnp.int32)

            # We want a single output length for every replicate:
            #   L = T + max_left_pad
            # So later-starting replicates get more right pad, earlier ones more left pad.
            def _align_vars(states):
                arr = states  # a leaf array with replicate and time axes present
                tr_axis = _norm_axis(trial_axis, arr.ndim)
                t_ax = _norm_axis(time_axis, arr.ndim)

                # Move replicate -> axis 0, time -> axis 1
                x = jnp.moveaxis(arr, tr_axis, 0)
                # After moving replicate to 0, time's index may have shifted by +1 if rep_ax < t_ax
                t_ax_after_rep0 = t_ax + 1 if tr_axis > t_ax else t_ax
                x = jnp.moveaxis(x, t_ax_after_rep0, 1)  # shape: (R, T, ...)

                R = x.shape[0]
                T = x.shape[1]

                # Sanity: replicate count must match start_idxs length
                if R != start_idxs.shape[0]:
                    raise ValueError(
                        f"Replicate axis size ({R}) does not match start_idxs ({start_idxs.shape[0]})."
                    )

                # Max left pad determines the common output length.
                max_left = int(jnp.max(pad_left_per_rep))
                L = T + max_left  # common time length after alignment

                # Per-replicate insert at (pad_left, 0, 0, ...)
                def _insert_one(x_r, pad_left_r):
                    # x_r shape: (T, ...)
                    out_shape = (L,) + x_r.shape[1:]
                    out = jnp.full(out_shape, pad_value, dtype=x_r.dtype)
                    start_tuple = (jnp.asarray(pad_left_r, dtype=jnp.int32),) + (0,) * (
                        x_r.ndim - 1
                    )
                    out = lax.dynamic_update_slice(out, x_r, start_tuple)

                    # Boolean mask marking real data
                    mask = lax.dynamic_update_slice(
                        jnp.zeros(out_shape, dtype=bool),
                        jnp.ones_like(x_r, dtype=bool),
                        start_tuple,
                    )
                    return out, mask

                aligned, masks = eqx.filter_vmap(_insert_one, in_axes=(0, 0), out_axes=(0, 0))(
                    x, pad_left_per_rep
                )

                # Move axes back to original positions
                aligned = jnp.moveaxis(aligned, 1, t_ax_after_rep0)  # time back
                aligned = jnp.moveaxis(aligned, 0, tr_axis)  # replicates back

                masks = jnp.moveaxis(masks, 1, t_ax_after_rep0)
                masks = jnp.moveaxis(masks, 0, tr_axis)

                return MaskedArray(aligned, masks)

            return jt.map(_align_vars, all_states_for_task)

        result = jt.map(_align_vars_by_task, data.tasks, vars, is_leaf=is_module)
        return result

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
    
    Example:
        >>> analysis = (
        ...     Profiles()
        ...     .after_transform(
        ...         lambda vars, *, data: get_align_epoch_start(2)(vars, data=data),
        ...         dependency_names="vars"
        ...     )
        ...     .after_transform(
        ...         partial(trim_tails, tolerance=1.0),
        ...         dependency_names="vars"
        ...     )
        ... )
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


def get_symmetric_accel_decel_epochs(states):
    speed = jnp.linalg.norm(states.mechanics.effector.vel, axis=-1)
    idxs_max_speed = jnp.argmax(speed, axis=-1)
    return LDict.of("epoch")(
        {
            "accel": (None, idxs_max_speed),
            #! Assume decel is the same length as accel, after the peak speed.
            "decel": (idxs_max_speed, 2 * idxs_max_speed),
        }
    )


def get_segment_trials_fn(slice_bounds_fn, axis=-2):
    def segment_trials(all_states, **kwargs):
        def _segment_states(states):
            return jt.map(
                lambda slice_bounds: jt.map(
                    lambda arr: dynamic_slice_with_padding(
                        arr,
                        slice_end_idxs=slice_bounds[1],
                        axis=axis,
                        slice_start_idxs=slice_bounds[0],
                    ),
                    states,
                ),
                slice_bounds_fn(states),
                is_leaf=is_type(tuple),
            )

        return jt.map(_segment_states, all_states, is_leaf=is_module)

    return segment_trials


def get_trial_start_positions(task: AbstractTask) -> Array:
    return get_validation_trial_specs(task).inits["mechanics.effector"].pos


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
