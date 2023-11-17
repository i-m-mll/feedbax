"""Pre-built loss functions for common tasks."""

from typing import Dict, Optional

import jax.numpy as jnp

from feedbax.loss import (
    CompositeLossFunc,
    EffectorPositionLossFunc,
    EffectorFinalVelocityLossFunc,
    NetworkOutputLossFunc,
    NetworkActivityLossFunc,
)


def simple_reach_loss(
    n_steps: int, 
    loss_term_weights: Optional[Dict[str, float]] = None,
    discount_exp: int = 6,
):
    """A typical loss function for a simple reach task.
    
    Includes power function discounting of position error, back in time from
    the end of trials. If the exponent `discount_exp` is zero, there is no
    discounting.
    
    TODO: 
    - Maybe activity loss shouldn't be included by default.
    """
    if loss_term_weights is None:
        # TODO: maybe move this to a common area for default parameters
        loss_term_weights = dict(
            effector_position=1.,
            effector_final_velocity=1.,
            nn_output=1e-5,
            nn_activity=1e-5,
        )
    if discount_exp == 0:
        discount = 1.
    else: 
        discount = jnp.linspace(1. / n_steps, 1., n_steps) ** discount_exp
    return CompositeLossFunc(
        dict(
            # these assume a particular PyTree structure to the states returned by the model
            # which is why we simply instantiate them 
            effector_position=EffectorPositionLossFunc(discount=discount),
            effector_final_velocity=EffectorFinalVelocityLossFunc(),
            nn_output=NetworkOutputLossFunc(),  # the "control" loss
            nn_activity=NetworkActivityLossFunc(),
        ),
        weights=loss_term_weights,
    )