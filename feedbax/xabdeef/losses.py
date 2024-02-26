"""Pre-built loss functions for common tasks.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from collections.abc import Mapping
import logging
from typing import Optional

import jax.numpy as jnp

from feedbax.loss import (
    CompositeLoss,
    EffectorPositionLoss,
    EffectorFinalVelocityLoss,
    NetworkOutputLoss,
    NetworkActivityLoss,
    power_discount,
)


logger = logging.getLogger(__name__)


def simple_reach_loss(
    loss_term_weights: Optional[Mapping[str, float]] = None,
    discount_exp: int = 6,
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
    if loss_term_weights is None:
        # TODO: maybe move this to a common area for default parameters
        loss_term_weights = dict(
            effector_position=1.,
            effector_final_velocity=1.,
            nn_output=1e-5,
            nn_hidden=1e-5,
        )
    return CompositeLoss(
        dict(
            # these assume a particular PyTree structure to the states returned by the model
            # which is why we simply instantiate them 
            effector_position=EffectorPositionLoss(
                discount_func=lambda n_steps: power_discount(n_steps, discount_exp)),
            effector_final_velocity=EffectorFinalVelocityLoss(),
            nn_output=NetworkOutputLoss(),  # the "control" loss
            nn_hidden=NetworkActivityLoss(),
        ),
        weights=loss_term_weights,
    )