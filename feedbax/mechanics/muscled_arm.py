
from functools import cached_property
from typing import Any, Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from feedbax.mechanics.arm import TwoLink
from feedbax.mechanics.muscle import VirtualMuscle

class TwoLinkMuscled(eqx.Module):
    """
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
    moment_arms: Float[Array, "links=2 muscles"] = eqx.field(
        static=True, converter=jnp.asarray)
    theta0: Float[Array, "links=2 muscles"] = eqx.field(static=True)
    l0: Float[Array, "muscles"] = eqx.field(static=True)
    f0: Float[Array, "muscles"] = eqx.field(static=True)
    
    def __init__(
        self, 
        muscle_model, 
        activator,
        twolink=TwoLink(),
        moment_arms=((2.0, -2.0, 0.0, 0.0, 1.50, -2.0),  # [cm]
                     (0.0, 0.0, 2.0, -2.0, 2.0, -1.50)),
        theta0=2 * jnp.pi * jnp.array(((15.0, 4.88, 0.0, 0.0, 4.5, 2.12), # [rad]
                                       (0.0, 0.0, 80.86, 109.32, 92.96, 91.52))) / 360.,  
        l0=jnp.array((7.32, 3.26, 6.4, 4.26, 5.95, 4.04)),  # [cm]
        f0=1. #31.8 * jnp.array((18., 14., 22., 12., 5., 10.)),  # [N] = [N/cm^2] * [cm^2]
    ):
        self.twolink = twolink
        self.muscle_model = muscle_model
        self.activator = activator
        self.moment_arms = moment_arms  
        self.theta0 = theta0
        self.l0 = l0
        self.f0 = f0

    def vector_field(self, t, y, args):
        theta, d_theta, activation = y 
        u = args 

        muscle_length = self._muscle_length(theta)
        muscle_velocity = self._muscle_velocity(d_theta)
        
        d_activation = self.activator.vector_field(t, activation, u)
        
        tension = self.muscle_model(muscle_length, muscle_velocity, u)
        torque = self.moment_arms @ (self.f0 * tension)
        
        d_theta, dd_theta = self.twolink.vector_field(t, (theta, d_theta), torque)
        
        return d_theta, dd_theta, d_activation

    def _muscle_length(self, theta):
        moment_arms, l0, theta0 = self.theta0, self.l0, self.theta0
        l = 1 + (moment_arms[0] * (theta0[0] - theta[0]) + moment_arms[1] * (theta0[1] - theta[1])) / l0
        return l
    
    def _muscle_velocity(self, d_theta):
        moment_arms, l0 = self.theta0, self.l0
        v = (moment_arms[0] * d_theta[0] + moment_arms[1] * d_theta[1]) / l0
        return v
    
    def init_state(self, state):
        # TODO
        pass
    
    @property
    def control_size(self):
        return len(self.l0)
    
    @cached_property
    def state_size(self):
        return self.twolink.state_size + self.control_size


