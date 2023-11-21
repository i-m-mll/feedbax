"""

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

from functools import cached_property
import logging 
from typing import Any, Callable, Optional

import equinox as eqx
from equinox import field
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from feedbax.mechanics.arm import TwoLink
from feedbax.mechanics.muscle import VirtualMuscle
from feedbax.types import CartesianState2D


logger = logging.getLogger(__name__)

N_DIM = 2


class TwoLinkMuscledState(eqx.Module):
    theta: Float[Array, "links=2"]
    d_theta: Float[Array, "links=2"]
    activation: Float[Array, "muscles"]
    torque: Float[Array, "links=2"] = field(
        default_factory=lambda: jnp.zeros(2))


class TwoLinkMuscled(eqx.Module):
    """
    
    NOTE: 
    - The six muscle groups given by the default parameters are (see 
      Lillicrap & Scott 2013):
        - shoulder flexors: pectoralis major, deltoid anterior
        - shoulder extensors: deltoid posterior, deltoid middle 
        - elbow flexors: brachialis, brachioradialis, extensor carpi radialis longus
        - elbow extensors: triceps lateral, triceps long 
        - bijoint flexors: biceps long, biceps short 
        - bijoint extensors: dorsoepitrochlearis, triceps long
    
    TODO: 
    - the activator should be defined as part of the muscle model, I think?
        - though this class probably still needs to pass around activations so
          they can be stored/initialized by `Recursive` and the user
    - could maybe subclass `TwoLink`
        - then wouldn't need to sub `system.twolink` for `system` when interchanging
    - don't convert `theta0` and $f0$ inside the default arguments list
    """
    
    muscle_model: VirtualMuscle  
    activator: eqx.Module
    twolink: TwoLink
    moment_arms: Float[Array, "links=2 muscles"]
    theta0: Float[Array, "links=2 muscles"] 
    l0: Float[Array, "muscles"] 
    f0: Float[Array, "muscles"] 
    
    forward_kinematics: Callable 
    inverse_kinematics: Callable
    effector: Callable
    update_state_given_effector_force: Callable
    _forward_pos: Callable
    
    def __init__(
        self, 
        muscle_model, 
        activator,
        twolink=TwoLink(),
        moment_arms=jnp.array(((2.0, -2.0, 0.0, 0.0, 1.50, -2.0),  # [cm]
                               (0.0, 0.0, 2.0, -2.0, 2.0, -1.50))),
        theta0=2 * jnp.pi * jnp.array(((15.0, 4.88, 0.0, 0.0, 4.5, 2.12), # [rad]
                                       (0.0, 0.0, 80.86, 109.32, 92.96, 91.52))) / 360.,  
        l0=jnp.array((7.32, 3.26, 6.4, 4.26, 5.95, 4.04)),  # [cm]
        f0=1. #31.8 * jnp.array((22., 12., 18., 14., 5., 10.)),  # [N] = [N/cm^2] * [cm^2]
    ):
        self.twolink = twolink
        self.muscle_model = muscle_model
        self.activator = activator
        self.moment_arms = moment_arms  
        self.theta0 = theta0
        self.l0 = l0
        self.f0 = f0
        
        #! alias these so this class behaves like `Mechanics` expects
        self.forward_kinematics = self.twolink.forward_kinematics
        self.inverse_kinematics = self.twolink.inverse_kinematics
        self.effector = self.twolink.effector
        self._forward_pos = self.twolink._forward_pos
        self.update_state_given_effector_force = \
            self.twolink.update_state_given_effector_force

    @jax.named_scope("fbx.TwoLinkMuscled.vector_field")
    def vector_field(self, t, state, args):
        u = args 

        muscle_length = self._muscle_length(state.theta)
        muscle_velocity = self._muscle_velocity(state.d_theta)
        
        d_activation = self.activator.vector_field(t, state.activation, u)
        
        tension = self.muscle_model(muscle_length, muscle_velocity, u)
        torque = self.moment_arms @ (self.f0 * tension)
        
        d_joints = self.twolink.vector_field(t, state, torque)
        d_theta, dd_theta = d_joints.theta, d_joints.d_theta
                
        return TwoLinkMuscledState(d_theta, dd_theta, d_activation)

    def _muscle_length(self, theta):
        moment_arms, l0, theta0 = self.theta0, self.l0, self.theta0
        l = 1 + (moment_arms[0] * (theta0[0] - theta[0]) + moment_arms[1] * (theta0[1] - theta[1])) / l0
        return l
    
    def _muscle_velocity(self, d_theta):
        moment_arms, l0 = self.theta0, self.l0
        v = (moment_arms[0] * d_theta[0] + moment_arms[1] * d_theta[1]) / l0
        return v
    
    def init(self, effector_state):
        if effector_state is None:
            effector_state = CartesianState2D(
                pos=self._forward_pos(jnp.zeros(2))[0][-1], 
                vel=jnp.zeros(N_DIM),
            )
        theta, torque = self.inverse_kinematics(effector_state)         
        return TwoLinkMuscledState(
            theta=theta, 
            d_theta=jnp.zeros_like(theta), 
            activation=jnp.zeros(self.control_size),
            torque=torque,
        )
    
    @property
    def control_size(self):
        return self.l0.shape[0]
    
    @cached_property
    def state_size(self):
        return self.twolink.state_size + self.control_size


