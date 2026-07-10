"""Pre-built loss functions for common tasks.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import jax.numpy as jnp

from feedbax.objectives.loss import (
    AbstractLoss,
    CompositeLoss,
    TargetSpec,
    TargetStateLoss,
    target_final_state,
    target_zero,
)

class EffectorFixationLoss(AbstractLoss):
    """Penalize effector position error while a delayed-reach trial is in hold."""

    label: str = "Effector maintains fixation"

    def term(self, states, trial_specs, model):
        assert states is not None, "EffectorFixationLoss requires states"
        assert trial_specs is not None, "EffectorFixationLoss requires trial_specs"

        target_spec = trial_specs.targets.get("mechanics.effector.pos")
        if not isinstance(target_spec, TargetSpec) or target_spec.value is None:
            raise ValueError("EffectorFixationLoss requires an effector position TargetSpec")

        effector_pos = states.mechanics.effector.pos[:, 1:]
        loss = jnp.sum((effector_pos - target_spec.value) ** 2, axis=-1)

        hold = trial_specs.inputs.hold
        if hold.ndim == loss.ndim + 1 and hold.shape[-1] == 1:
            hold = jnp.squeeze(hold, axis=-1)
        return jnp.sum(loss * hold.astype(loss.dtype), axis=-1)


def simple_reach_loss(
    effector_position: float = 1.0,
    effector_final_velocity: float = 1.0,
    nn_output: float = 1e-5,
    nn_hidden: float = 1e-5,
) -> CompositeLoss:
    """A typical loss function for a simple reaching task.

    Arguments:
        loss_term_weights: Maps loss term names to term weights. If `None`,
            a typical set of default weights is used.
        discount_exp: The exponent of the power function used to discount
            the position error, back in time from the end of trials. Larger
            values lead to penalties that are more concentrated at the end
            of trials. If zero, all time steps are weighted equally.
    """
    return CompositeLoss(
        dict(
            effector_position=TargetStateLoss(
                "Effector position",
                where=lambda state: state.mechanics.effector.pos,
                norm=lambda x: jnp.sum(x**2, axis=-1),
                # norm=lambda *args, **kwargs: (
                #     # Euclidean distance
                #     jnp.linalg.norm(*args, axis=-1, **kwargs) ** 2
                # ),
            ),
            effector_final_velocity=TargetStateLoss(
                "Effector final velocity",
                where=lambda state: state.mechanics.effector.vel,
                # By indexing out the final timestep only, this loss must
                # be paired with an `AbstractTask` that supplies a
                # single-timestep target value.
                spec=target_zero & target_final_state,
            ),
            nn_output=TargetStateLoss(
                "Command",
                where=lambda state: state.efferent.output,
                spec=target_zero,
            ),
            nn_hidden=TargetStateLoss(
                "NN activity",
                where=lambda state: state.net.hidden,
                spec=target_zero,
            ),
        ),
        weights=dict(
            effector_position=effector_position,
            effector_final_velocity=effector_final_velocity,
            nn_output=nn_output,
            nn_hidden=nn_hidden,
        ),
    )

def delayed_reach_loss(
    effector_fixation: float = 1.0,
    effector_position: float = 1.0,
    effector_final_velocity: float = 1.0,
    nn_output: float = 1e-4,
    nn_hidden: float = 1e-5,
) -> CompositeLoss:
    """A typical loss function for a `DelayedReaches` task."""
    return CompositeLoss(
        dict(
            # these assume a particular PyTree structure to the states returned by the model
            # which is why we simply instantiate them
            effector_fixation=EffectorFixationLoss(),
            effector_position=TargetStateLoss(
                "Effector position",
                where=lambda state: state.mechanics.effector.pos,
                norm=lambda x: jnp.sum(x**2, axis=-1),
                # norm=lambda *args, **kwargs: (
                #     # Euclidean distance
                #     jnp.linalg.norm(*args, axis=-1, **kwargs) ** 2
                # ),
            ),
            effector_final_velocity=TargetStateLoss(
                "Effector final velocity",
                where=lambda state: state.mechanics.effector.vel,
                # By indexing out the final timestep only, this loss must
                # be paired with an `AbstractTask` that supplies a
                # single-timestep target value.
                spec=target_zero & target_final_state,
            ),
            nn_output=TargetStateLoss(
                "Command",
                where=lambda state: state.efferent.output,
                spec=target_zero,
            ),
            nn_hidden=TargetStateLoss(
                "NN activity",
                where=lambda state: state.net.hidden,
                spec=target_zero,
            ),
        ),
        weights=dict(
            effector_fixation=effector_fixation,
            effector_position=effector_position,
            effector_final_velocity=effector_final_velocity,
            nn_output=nn_output,
            nn_hidden=nn_hidden,
        ),
    )
