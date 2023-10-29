""" 


:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

from functools import cached_property
import logging
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
import numpy as np

from feedbax.types import CartesianState2D
from feedbax.utils import SINCOS_GRAD_SIGNS


logger = logging.getLogger(__name__)


N_DIM = 2


class TwoLinkState(eqx.Module):
    theta: Float[Array, "2"]
    d_theta: Float[Array, "2"]


class TwoLink(eqx.Module):
    l: Float[Array, "2"] = eqx.field(converter=jnp.asarray)  # [L] lengths of arm segments
    m: Float[Array, "2"] = eqx.field(converter=jnp.asarray)  # [M] masses of segments
    I: Float[Array, "2"] = eqx.field(converter=jnp.asarray)  # [M L^2] moments of inertia of segments
    s: Float[Array, "2"] = eqx.field(converter=jnp.asarray)  # [L] distance from joint center to segment COM
    B: Float[Array, "2 2"] = eqx.field(converter=jnp.asarray)  # [M L^2 T^-1] joint friction matrix
    inertia_gain: float   
    
    def __init__(
            self,
            l=(0.30, 0.33),  # [m]
            m=(1.4, 1.0),  # [kg]
            I=(0.025, 0.045),  # [kg m^2]
            s=(0.11, 0.16),  # [m]
            B=((0.05, 0.025),  # [kg m^2 s^-1]
               (0.025, 0.05)),
    ):
        self.l = l
        self.m = m
        self.I = I
        self.s = s
        self.B = B
        self.inertia_gain = 1.0
    
        #! initialize cached properties used by JAX operations
        # otherwise their initialization is a side effect
        self._a  
    
    @jax.named_scope("fbx.TwoLink.vector_field")
    def vector_field(self, t, state, args):
        theta, d_theta = state.theta, state.d_theta
        input_torque = args

        # centripetal and coriolis torques 
        c_vec = jnp.array((
            -d_theta[1] * (2 * d_theta[0] + d_theta[1]),
            d_theta[0] ** 2
        )) * self._a[1] * jnp.sin(theta[1])  
        
        # inertia matrix that maps torques -> angular accelerations
        cs1 = jnp.cos(theta[1])
        tmp = self._a[2] + self._a[1] * cs1
        inertia_mat = jnp.array(((self._a[0] + 2 * self._a[1] * cs1, tmp),
                                 (tmp, self._a[2] * jnp.ones_like(cs1))))
        
        net_torque = input_torque - c_vec.T - jnp.matmul(d_theta, self.B.T)
        
        dd_theta = jnp.linalg.inv(inertia_mat) @ net_torque
        
        return TwoLinkState(d_theta, dd_theta)
    
    def init(
        self, 
        effector_state: Optional[CartesianState2D] = None,
    ):
        if effector_state is None:
            effector_state = CartesianState2D(
                jnp.zeros(N_DIM), 
                jnp.zeros(N_DIM)
            )
        theta = self.inverse_kinematics(effector_state)        
        return TwoLinkState(theta, jnp.zeros_like(theta))

    @cached_property
    def _a(self):
        # this is a cached_property to avoid polluting the module's fields with private attributes
        return (
            self.I[0] + self.I[1] + self.m[1] * self.l[0] ** 2, # + m[1]*s[1]**2 + m[0]*s[0]**2
            self.m[1] * self.l[0] * self.s[1],
            self.I[1],  # + m[1] * s[1] ** 2
        )
    
    @cached_property
    def _lsqpm(self):
        lsq = self.l ** 2
        return (lsq[0] - lsq[1], lsq[0] + lsq[1])
    
    @property 
    def control_size(self) -> int:
        return 2
    
    @property
    def state_size(self) -> int:
        return 2 * 2  # two joints, angle and angular velocity
    
    @property 
    def n_links(self) -> int:
        return 2

    @jax.named_scope("fbx.TwoLink.inverse_kinematics")
    def inverse_kinematics(
            self,
            effector_state: CartesianState2D
    ) -> Float[Array, "2"]: # TwoLinkState
        """Convert Cartesian effector position to joint angles for a two-link arm.
        
        NOTE: 
        
        - This is the "inverse kinematics" problem.
        - This gives the "elbow down" or "righty" solution. The "elbow up" or
        "lefty" solution is given by `theta0 = gamma + alpha` and 
        `theta1 = beta - pi`.
        - No solution exists if `dsq` is outside of `(l[0] - l[1], l[0] + l[1])`.
        - See https://robotics.stackexchange.com/a/6393 which also covers
        how to convert velocity.
        
        TODO:
        - Convert velocity using the effector Jacobian.
        - Try to generalize to n-link arm using Jacobian of forward kinematics?
        - Unit test round trip with `forward_kinematics`.
        """
        pos = effector_state.pos
        l, lsqpm = self.l, self._lsqpm
        dsq = jnp.sum(pos ** 2)

        alpha = jnp.arccos((lsqpm[0] + dsq) / (2 * l[0] * jnp.sqrt(dsq)))
        gamma = jnp.arctan2(pos[1], pos[0])
        theta0 = gamma - alpha
        
        beta = jnp.arccos((lsqpm[1] - dsq) / (2 * l[0] * l[1]))
        theta1 = np.pi - beta

        angles = jnp.stack([theta0, theta1], axis=-1)

        return angles    

    @jax.named_scope("fbx.TwoLink.forward_kinematics")
    def forward_kinematics(
            self,
            state: TwoLinkState
    ) -> CartesianState2D:
        """Convert angular state to Cartesian state.
        
        NOTE:
        - See https://robotics.stackexchange.com/a/6393; which suggests the 
        Denavit-Hartenberg method, which uses a matrix for each joint, 
        transforming its angle into a change in position relative to the 
        preceding joint.
        
        TODO:
        - Any point to reducing memory by only calculating the last link?
        """
        angle_sum = jnp.cumsum(state.theta)  # links
        length_components = self.l * jnp.array([jnp.cos(angle_sum),
                                                jnp.sin(angle_sum)])  # xy, links
        xy_position = jnp.cumsum(length_components, axis=1)  # xy, links
        
        ang_vel_sum = jnp.cumsum(state.d_theta)  # links
        xy_velocity = jnp.cumsum(SINCOS_GRAD_SIGNS[1] * length_components[:, ::-1] 
                                 * ang_vel_sum,
                                 axis=1)  # xy, links
        return CartesianState2D(xy_position.T, xy_velocity.T)

    def effector(self, state: TwoLinkState) -> CartesianState2D:
        """Return the Cartesian state of the end of the arm."""
        return jax.tree_map(
            lambda x: x[-1],  # last link
            self.forward_kinematics(state),
        )