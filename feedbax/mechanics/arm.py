""" 


:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

from functools import cached_property
import logging
from typing import Optional

import equinox as eqx
from equinox import field
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
import numpy as np

from feedbax.state import AbstractState, CartesianState2D, StateBounds
from feedbax.utils import SINCOS_GRAD_SIGNS


logger = logging.getLogger(__name__)


N_DIM = 2


class TwoLinkState(AbstractState):
    theta: Float[Array, "... ndim=2"]
    d_theta: Float[Array, "... ndim=2"]
    torque: Float[Array, "... ndim=2"] = field(default_factory=lambda: jnp.zeros(2))


class TwoLink(eqx.Module):
    l: Float[Array, "links=2"] = field(converter=jnp.asarray)  # [L] lengths of arm segments
    m: Float[Array, "links=2"] = field(converter=jnp.asarray)  # [M] masses of segments
    I: Float[Array, "links=2"] = field(converter=jnp.asarray)  # [M L^2] moments of inertia of segments
    s: Float[Array, "links=2"] = field(converter=jnp.asarray)  # [L] distance from joint center to segment COM
    B: Float[Array, "2 2"] = field(converter=jnp.asarray)  # [M L^2 T^-1] joint friction matrix
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
        
        # Normally, `state.torque` only contains the torque induced by
        # linear force on the effector, as determined by inverse kinematics;
        # whereas `input_torque` is direct torque control.
        net_torque = (
            state.torque  
            + input_torque 
            - c_vec.T 
            - jnp.matmul(d_theta, self.B.T)
        )
        
        dd_theta = jnp.linalg.inv(inertia_mat) @ net_torque
        
        return TwoLinkState(d_theta, dd_theta)
    
    def init(
        self, 
        effector_state: Optional[CartesianState2D] = None,
    ):
        if effector_state is None:
            effector_state = CartesianState2D(
                pos=self._forward_pos(jnp.zeros(2))[0][-1], 
                vel=jnp.zeros(N_DIM),
            )
        theta, torque = self.inverse_kinematics(effector_state)        
        return TwoLinkState(
            theta=theta, 
            dtheta=jnp.zeros_like(theta),
            torque=torque,
        )

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
    ) -> Float[Array, "2"]: # TODO: should be TwoLinkState
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

        theta = jnp.stack([theta0, theta1], axis=-1)
        
        if effector_state.force is not None:
            torque = self.effector_force_to_torques(theta, effector_state.force)
        else:
            torque = None

        return theta, torque
    
    def update_state_given_effector_force(
        self, 
        state: TwoLinkState, 
        effector_force: jax.Array,
    ):
        """Update a state PyTree with torques inferred from effector force."""
        torque = self.effector_force_to_torques(
            state.theta, 
            effector_force,
        )
        
        return eqx.tree_at(
            lambda state: state.torque, 
            state, 
            torque + state.torque
        )
    
    def effector_force_to_torques(
        self, 
        theta: jax.Array, 
        effector_force: jax.Array
    ):
        """Return the torques induced by a force on the effector."""
        torque = self.effector_jac(theta).T @ effector_force
        return torque
    
    def effector_jac(self, theta):
        """Jacobian of effector position with respect to joint angles."""
        jac, _ = jax.jacfwd(self._forward_pos, has_aux=True)(theta)
        return jac[-1]
    
    def _forward_pos(self, theta):
        """Separated from `forward_kinematics` for use in `effector_jac`."""
        angle_sum = jnp.cumsum(theta)  # links
        length_components = self.l * jnp.array([jnp.cos(angle_sum),
                                                jnp.sin(angle_sum)])  # xy, links
        xy_pos = jnp.cumsum(length_components, axis=1)  # xy, links
        return xy_pos.T, length_components

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
        xy_pos, length_components = self._forward_pos(state.theta)
        
        ang_vel_sum = jnp.cumsum(state.d_theta)  # links
        xy_vel = jnp.cumsum(SINCOS_GRAD_SIGNS[1] * length_components[:, ::-1] 
                            * ang_vel_sum,
                            axis=1).T  # xy, links
        
        # TODO: add force conversion?
        # It might be very important NOT to convert force here.
        # Otherwise we could get positive feedback where the effector force
        # is passed to the next time step and converted to torques, then
        # converted to effector forces and added back to the torques, etc.

        return CartesianState2D(
            pos=xy_pos, 
            vel=xy_vel,
            force=jnp.zeros_like(xy_vel),
        )

    def effector(self, state: TwoLinkState) -> CartesianState2D:
        """Return the Cartesian state of the end of the arm."""
        return jax.tree_map(
            lambda x: x[-1],  # last link
            self.forward_kinematics(state),
        )
    
    @property
    def bounds(self) -> StateBounds[TwoLinkState]:
        """Suggested bounds on the state space.
        
        Angle limits adopted from MotorNet (TODO cite).
        """
        return StateBounds(
            low=TwoLinkState(
                theta=jnp.array([0., 0.]),
                d_theta=None,
                torque=None,
            ), 
            high=TwoLinkState(
                theta=jnp.deg2rad(jnp.array([140., 160.])),
                d_theta=None,
                torque=None,
            ),
        )