from collections.abc import Callable, Sequence

import equinox as eqx
import jax.numpy as jnp
import jax.tree as jt
from feedbax.intervene import AddNoise, CurlField, FixedField, TimeSeriesParam

from feedbax_experiments.analysis.state_utils import vmap_eval_ensemble
from feedbax_experiments.types import TreeNamespace

FB_INTERVENOR_LABEL = "FeedbackPert"
PLANT_INTERVENOR_LABEL = "DisturbanceField"


PLANT_DISTURBANCE_CLASSES = {
    "curl": CurlField,
    "constant": FixedField,
    "gusts": FixedField,
    "noise": AddNoise,
}


def orthogonal_field(trial_spec, _, key):
    init_pos = trial_spec.inits["mechanics.effector"].pos
    goal_pos = jnp.take(trial_spec.targets["mechanics.effector.pos"].value, -1, axis=-2)
    direction_vec = goal_pos - init_pos
    direction_vec = direction_vec / jnp.linalg.norm(direction_vec)
    return jnp.array([-direction_vec[1], direction_vec[0]])


#! TODO: Control start and end time via hps
def get_fixed_gust_fn(hps: TreeNamespace, start_prop: float = 0.1, end_prop: float = 0.15):
    """Returns a fixed orthogonal field that is active during a fixed portion of the trial."""
    n_steps = hps.task.n_steps - 1
    active_idxs = jnp.arange(int(n_steps * start_prop), int(n_steps * end_prop))
    active_ts = jnp.zeros((n_steps,), dtype=bool).at[active_idxs].set(True)

    def fixed_gust_fn(scale: float):
        return FixedField.with_params(
            scale=scale,
            field=orthogonal_field,
            active=TimeSeriesParam(active_ts),
        )

    return fixed_gust_fn


PLANT_PERT_FNS = {
    "curl": lambda hps: lambda scale: CurlField.with_params(
        #! amplitude=amplitude,
        scale=scale,
    ),
    "constant": lambda hps: lambda scale: FixedField.with_params(
        scale=scale,
        field=orthogonal_field,
    ),
    #! TODO: Maybe eval on a single large gust.
    "gusts": get_fixed_gust_fn,
    # "noise": lambda scale: AddNoise.with_params(
    #     scale=scale,
    # ),
}


def task_with_pert_amp(task, pert_amp, intervenor_label):
    """Returns a task with the given disturbance amplitude."""
    return eqx.tree_at(
        lambda task: task.intervention_specs.validation[intervenor_label].intervenor.params.scale,
        task,
        pert_amp,
    )


def get_pert_amp_vmap_eval_fn(
    where_pert_amps_in_hps: Callable[[TreeNamespace], Sequence[float]],
    intervenor_label: str,
):
    """Returns a function for evaluating models across a range of perturbation amplitudes.

    Args:
        where_pert_amps_in_hps: Callable that selects the sequence of amplitudes from the tree of hyperparameters.
        intervenor_label: The same argument passed to `schedule_intervenor` when setting up the task+models, which
            identifies which intervention to scale by the vmap argument.
    """

    def eval_fn(key_eval, hps, models, task):
        """Vmap over impulse amplitude."""

        states = eqx.filter_vmap(
            lambda amplitude: vmap_eval_ensemble(
                key_eval,
                hps,
                models,
                task_with_pert_amp(task, amplitude, intervenor_label),
            ),
        )(jnp.array(where_pert_amps_in_hps(hps)))

        # I am not sure why this moveaxis is necessary.
        # I tried using `out_axes=2` (with or without `in_axes=0`) and
        # the result has the trial (axis 0) and replicate (axis 1) swapped.
        # (I had expected vmap to simply insert the new axis in the indicated position.)
        return jt.map(
            lambda arr: jnp.moveaxis(arr, 0, 2),
            states,
        )

    return eval_fn
