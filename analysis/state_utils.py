from collections.abc import Callable
from functools import partial
from types import MappingProxyType
from typing import ClassVar, Optional

import equinox as eqx
from equinox import filter_vmap as vmap
import jax.numpy as jnp 
import jax.random as jr
import jax.tree as jt
from jaxtyping import Array, Float, PRNGKeyArray

from feedbax.intervene import AbstractIntervenor
from feedbax.task import AbstractTask
import jax_cookbook.tree as jtree
from jax_cookbook import is_type, is_module

from rlrmp.analysis.analysis import get_validation_trial_specs
from rlrmp.constants import REPLICATE_CRITERION
from rlrmp.misc import dynamic_slice_with_padding
from rlrmp.types import LDict, TreeNamespace


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
    return jnp.stack([
        trial_specs.inits['mechanics.effector'].pos, 
        jnp.take(trial_specs.targets['mechanics.effector.pos'].value, -1, axis=-2),
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
    return eqx.filter_vmap(_get_eval_ensemble(models, task))(
        jr.split(key, hps.eval_n)
    )


def get_constant_task_input_fn(x, n_steps, n_trials):
    return lambda trial_spec, key: (
        jnp.full((n_trials, n_steps), x, dtype=float)
    )


def get_step_task_input_fn(x1, x2, step_step, n_steps, n_trials):
    def input_func(trial_spec, key):
        # Create array of x1 values
        inputs = jnp.full((n_trials, n_steps), x1, dtype=float)
        inputs = inputs.at[:, step_step:].set(x2)

        return inputs

    return input_func

def _get_replicate_idxs_func(replicate_info, key):
    def _replicate_idxs_func(std):
        return replicate_info[std][key][REPLICATE_CRITERION]
    return _replicate_idxs_func


def get_best_replicate(tree, *, replicate_info, axis: int = 1, keep_axis: bool = False, **kwargs):
    _original_replicate_idxs_func = _get_replicate_idxs_func(replicate_info, 'best_replicates')
    
    if keep_axis:    
        _replicate_idxs_func = lambda std: jnp.array([_original_replicate_idxs_func(std)])
    else:
        _replicate_idxs_func = _original_replicate_idxs_func
    
    def _take_best_replicate(x):
        if eqx.is_array(x):
            return jnp.take(x, _replicate_idxs_func(0), axis=axis)
        else:
            return x
    
    def _process_node(node):
        if LDict.is_of("train__pert__std")(node):
            return LDict.of("train__pert__std")({
                std: jt.map(_take_best_replicate, subtree)
                for std, subtree in node.items()
            })
        elif eqx.is_array(node):
            #! This is for the case where the user passes a single 
            return jnp.take(node, replicate_info['best_replicates'][REPLICATE_CRITERION], axis=axis)
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
get_best_model_replicate.__doc__ = (
    """Variant of `get_best_replicate` that filters out intervenors from the tree.
    
    This is necessary when getting the best replicate from a tree of models, rather than states,
    since intervention parameters generally do not have a replicate batch axis. 
    """
)


def exclude_bad_replicates(tree, *, replicate_info, axis=0):
    _replicate_idxs_func = _get_replicate_idxs_func(replicate_info, 'included_replicates')
    
    def _process_std_subtree(tree_by_std):
        return LDict.of('train__pert__std')({
            std: jt.map(
                #! TODO: Store included replicates as ints (not bools) in the first place!
                lambda arr:  jnp.take(arr, _replicate_idxs_func(std).nonzero()[0], axis=axis),
                subtree,
            )
            for std, subtree in tree_by_std.items()
        }) 

    # 
    return jt.map(
        _process_std_subtree,
        tree,
        is_leaf=LDict.is_of('train__pert__std'),
    )


def get_symmetric_accel_decel_epochs(states):
    speed = jnp.linalg.norm(states.mechanics.effector.vel, axis=-1)
    idxs_max_speed = jnp.argmax(speed, axis=-1)
    return LDict.of("epoch")({
        "accel": (None, idxs_max_speed),
        #! Assume decel is the same length as accel, after the peak speed. 
        "decel": (idxs_max_speed, 2 * idxs_max_speed),
    })


def get_segment_trials_func(slice_bounds_func, axis=-2):
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
                slice_bounds_func(states),
                is_leaf=is_type(tuple),
            )

        return jt.map(_segment_states, all_states, is_leaf=is_module)

    return segment_trials


def get_trial_start_positions(task: AbstractTask) -> Array:
    return get_validation_trial_specs(task).inits['mechanics.effector'].pos


def unsqueezer(axis: int):
    """Return a function that unsqueezes an array along the specified axis."""
    def _unsqueeze(x):
        return jnp.expand_dims(x, axis=axis)
    return _unsqueeze


frob = lambda x: jnp.linalg.norm(x, axis=(-1, -2), ord='fro')


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