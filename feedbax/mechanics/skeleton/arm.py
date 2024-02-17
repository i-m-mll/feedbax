""" 


:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

from functools import cached_property
import logging
from typing import Optional

import equinox as eqx
from equinox import field
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
import numpy as np

from feedbax.mechanics.skeleton import AbstractSkeleton, AbstractSkeletonState
from feedbax.state import CartesianState2D, StateBounds
from feedbax.misc import SINCOS_GRAD_SIGNS, corners_2d


logger = logging.getLogger(__name__)


N_DIM = 2


class TwoLinkState(AbstractSkeletonState):
    angle: Float[Array, "... links=2 ndim=2"] = field(default_factory=lambda: jnp.zeros(2))
    d_angle: Float[Array, "... links=2 ndim=2"] = field(default_factory=lambda: jnp.zeros(2))
    torque: Float[Array, "... links=2 ndim=2"] = field(default_factory=lambda: jnp.zeros(2))
            

class TwoLink(AbstractSkeleton[TwoLinkState]):
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
    
    @jax.named_scope("fbx.TwoLink")
    def vector_field(self, t, state, input):
        angle, d_angle = state.angle, state.d_angle
        input_torque = input

        # centripetal and coriolis torques 
        c_vec = jnp.array((
            -d_angle[1] * (2 * d_angle[0] + d_angle[1]),
            d_angle[0] ** 2
        )) * self._a[1] * jnp.sin(angle[1])  
        
        # inertia matrix that maps torques -> angular accelerations
        cs1 = jnp.cos(angle[1])
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
            - jnp.matmul(d_angle, self.B.T)
        )
        
        dd_angle = jnp.linalg.inv(inertia_mat) @ net_torque
        
        return TwoLinkState(d_angle, dd_angle)
    
    def init(
        self, 
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> TwoLinkState:
        return TwoLinkState()

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
    def input_size(self) -> int:
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
    ) -> TwoLinkState: 
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
        - Try to generalize to n-link arm using Jacobian of forward kinematics?
        """
        pos = effector_state.pos
        l, lsqpm = self.l, self._lsqpm
        dsq = jnp.sum(pos ** 2)
        
        alpha = jnp.arccos((lsqpm[0] + dsq) / (2 * l[0] * jnp.sqrt(dsq)))
        gamma = jnp.arctan2(pos[1], pos[0])
        theta0 = gamma - alpha
        
        beta = jnp.arccos((lsqpm[1] - dsq) / (2 * l[0] * l[1]))
        theta1 = np.pi - beta

        angle = jnp.stack([theta0, theta1], axis=-1)
        
        # TODO: don't calculate Jacobian twice, for `d_angle` and `torque`
        d_angle = jnp.linalg.inv(self.effector_jac(angle)) @ effector_state.vel
        
        if effector_state.force is not None:
            torque = self.effector_force_to_torques(angle, effector_state.force)
        else:
            torque = None

        return TwoLinkState(angle=angle, d_angle=d_angle, torque=torque)
    
    def update_state_given_effector_force(
        self, 
        effector_force: jax.Array,
        state: TwoLinkState, 
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> TwoLinkState:
        """Add torques inferred from effector force to the state PyTree.
        
        TODO:
        
        - generalize to force anywhere on arm
        """
        torque = self.effector_force_to_torques(
            state.angle, 
            effector_force,
        )
        
        return eqx.tree_at(
            lambda state: state.torque, 
            state, 
            torque + state.torque
        )
    
    def effector_force_to_torques(
        self, 
        angle: jax.Array, 
        effector_force: jax.Array
    ):
        """Return the torques induced by a force on the effector."""
        torque = self.effector_jac(angle).T @ effector_force
        return torque
    
    def effector_jac(self, angle):
        """Jacobian of effector position with respect to joint angles."""
        jac, _ = jax.jacfwd(self._forward_pos, has_aux=True)(angle)
        return jac[-1]
    
    def _forward_pos(self, angle):
        """Separated from `forward_kinematics` for use in `effector_jac`."""
        angle_sum = jnp.cumsum(angle)  # links
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
        xy_pos, length_components = self._forward_pos(state.angle)
        
        ang_vel_sum = jnp.cumsum(state.d_angle)  # links
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
                angle=jnp.array([0., 0.]),
                d_angle=None,
                torque=None,
            ), 
            high=TwoLinkState(
                angle=jnp.deg2rad(jnp.array([140., 160.])),
                d_angle=None,
                torque=None,
            ),
        )   
    
def twolink_workspace_test(workspace: Float[Array, "bounds=2 xy=2"], twolink: TwoLink):
    """Tests whether a rectangular workspace is reachable by the two-link arm.
    
    TODO: Take the angle bounds into account.
    """
    r = sum(twolink.l)
    lengths = jnp.sum(corners_2d(workspace) ** 2, axis=0) ** 0.5
    if jnp.any(lengths > r):
        return False 
    return True