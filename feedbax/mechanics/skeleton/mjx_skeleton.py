"""MuJoCo/MJX skeleton implementing feedbax's AbstractSkeleton interface.

Uses ``mjx.forward()`` to compute forward dynamics without integrating,
exposing a standard ``vector_field()`` returning ``(qvel, qacc)`` so that
Diffrax handles the time integration.
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx
from equinox import field
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, Scalar

from feedbax.mechanics.skeleton.skeleton import AbstractSkeleton, AbstractSkeletonState
from feedbax.runtime.state import CartesianState, StateBounds


class MJXSkeletonState(AbstractSkeletonState):
    """Configuration state for an MJX-based skeleton.

    Attributes:
        qpos: Generalized positions, shape ``(nq,)``.
        qvel: Generalized velocities, shape ``(nv,)``.
        effector_force: World-space XY force held at the effector site, shape ``(2,)``.
    """

    qpos: Float[Array, " nq"] = field(default_factory=lambda: jnp.zeros(1))
    qvel: Float[Array, " nv"] = field(default_factory=lambda: jnp.zeros(1))
    effector_force: Float[Array, "2"] = field(default_factory=lambda: jnp.zeros(2))


class MJXSkeleton(AbstractSkeleton[MJXSkeletonState]):
    """Skeleton dynamics via MuJoCo/MJX forward computation.

    Calls ``mjx.forward()`` to compute accelerations without integrating,
    providing a ``vector_field()`` compatible with Diffrax ODE solvers.

    Attributes:
        mjx_model: The MJX model (on device).
        effector_site_id: Site index for the end-effector.
        effector_body_id: Body index for the end-effector body.
        nq: Number of generalized positions (static).
        nv: Number of generalized velocities (static).
        nbody: Number of bodies (static).
    """

    mjx_model: object  # mjx.Model — not typed to avoid import at class-definition time
    effector_site_id: int = field(static=True)
    effector_body_id: int = field(static=True)
    nq: int = field(static=True)
    nv: int = field(static=True)
    nbody: int = field(static=True)
    _bounds: StateBounds  # Cached at init from concrete model data

    def __init__(
        self,
        mjx_model,
        effector_site_id: int,
        effector_body_id: int,
    ):
        from mujoco import mjtJoint
        import numpy as np

        if not 0 <= effector_site_id < mjx_model.nsite:
            raise ValueError(f"Invalid effector site index {effector_site_id}")
        site_body_id = int(mjx_model.site_bodyid[effector_site_id])
        if effector_body_id != site_body_id:
            raise ValueError(
                f"Effector site {effector_site_id} belongs to body {site_body_id}, "
                f"but body {effector_body_id} was specified"
            )

        self.mjx_model = mjx_model
        self.effector_site_id = effector_site_id
        self.effector_body_id = site_body_id
        self.nq = int(mjx_model.nq)
        self.nv = int(mjx_model.nv)
        self.nbody = int(mjx_model.nbody)

        scalar_joint = np.isin(
            mjx_model.jnt_type, [mjtJoint.mjJNT_HINGE, mjtJoint.mjJNT_SLIDE]
        )
        limited = np.asarray(mjx_model.jnt_limited, dtype=bool) & scalar_joint
        if np.any(limited):
            addresses = mjx_model.jnt_qposadr[limited]
            joint_ranges = np.asarray(mjx_model.jnt_range)[limited]
            low = np.full(self.nq, -np.inf, dtype=joint_ranges.dtype)
            high = np.full(self.nq, np.inf, dtype=joint_ranges.dtype)
            low[addresses], high[addresses] = joint_ranges.T
            self._bounds = StateBounds(
                low=MJXSkeletonState(
                    qpos=jnp.asarray(low),
                    qvel=None,
                    effector_force=None,
                ),
                high=MJXSkeletonState(
                    qpos=jnp.asarray(high),
                    qvel=None,
                    effector_force=None,
                ),
            )
        else:
            self._bounds = StateBounds(low=None, high=None)

    def _reconstruct_data(self, state: MJXSkeletonState):
        """Build MJX data with the current generalized positions and velocities."""
        from mujoco import mjx

        return mjx.make_data(self.mjx_model).replace(qpos=state.qpos, qvel=state.qvel)

    def _kinematics_data(self, state: MJXSkeletonState):
        """Compute world positions and the center-of-mass motion bases."""
        from mujoco import mjx

        data = mjx.kinematics(self.mjx_model, self._reconstruct_data(state))
        return mjx.com_pos(self.mjx_model, data)

    def _dynamics_data(self, state: MJXSkeletonState, ctrl: Array):
        """Prepare controls and the current body COM wrench for dynamics."""
        from mujoco import mjx

        data = mjx.kinematics(self.mjx_model, self._reconstruct_data(state))
        force = jnp.pad(state.effector_force, (0, 1))
        moment_arm = data.site_xpos[self.effector_site_id] - data.xipos[self.effector_body_id]
        wrench = jnp.concatenate((force, jnp.cross(moment_arm, force)))
        return data.replace(
            ctrl=ctrl, xfrc_applied=data.xfrc_applied.at[self.effector_body_id].set(wrench)
        )

    @jax.named_scope("fbx.MJXSkeleton.vector_field")
    def vector_field(
        self,
        t: Scalar,
        state: MJXSkeletonState,
        input: Array,
    ) -> MJXSkeletonState:
        """Return time derivatives of the MJX skeleton state.

        Calls ``mjx.forward()`` to compute accelerations (qacc) without
        integrating, then returns ``(qvel, qacc, zeros)`` as the derivative.

        Args:
            t: Time (unused — MJX dynamics are time-invariant).
            state: Current skeleton state.
            input: Actuator controls, shape ``(nu,)``.

        Returns:
            MJXSkeletonState with (d_qpos=qvel, d_qvel=qacc, d_effector_force=0).
        """
        from mujoco import mjx

        data = self._dynamics_data(state, input)
        data = mjx.forward(self.mjx_model, data)

        return MJXSkeletonState(
            qpos=state.qvel,  # d(qpos)/dt = qvel
            qvel=data.qacc,   # d(qvel)/dt = qacc
            effector_force=jnp.zeros_like(state.effector_force),
        )

    @jax.named_scope("fbx.MJXSkeleton.forward_kinematics")
    def forward_kinematics(self, state: MJXSkeletonState) -> CartesianState:
        """Return world-space XY positions and velocities at every body origin."""
        from mujoco import mjx

        data = mjx.com_vel(self.mjx_model, self._kinematics_data(state))
        # MJX expresses cvel in world axes at the root subtree's center of mass.
        offset = data.xpos - data.subtree_com[self.mjx_model.body_rootid]
        vel = data.cvel[:, 3:] + jnp.cross(data.cvel[:, :3], offset)

        pos = data.xpos[:, :2]
        return CartesianState(pos=pos, vel=vel[:, :2], force=jnp.zeros_like(pos))

    @jax.named_scope("fbx.MJXSkeleton.effector")
    def effector(self, state: MJXSkeletonState) -> CartesianState:
        """Return the Cartesian state of the end-effector site.

        The velocity is the world-space site position Jacobian applied to
        generalized velocity, projected onto the XY plane.

        Args:
            state: Current skeleton state.

        Returns:
            CartesianState with 2D effector position, velocity, and zero force.
        """
        from mujoco import mjx

        data = self._kinematics_data(state)
        pos = data.site_xpos[self.effector_site_id]
        jac_pos, _ = mjx.jac(self.mjx_model, data, pos, self.effector_body_id)
        vel = data.qvel @ jac_pos

        return CartesianState(pos=pos[:2], vel=vel[:2], force=jnp.zeros(2))

    def update_state_given_effector_force(
        self,
        effector_force: Array,
        state: MJXSkeletonState,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> MJXSkeletonState:
        """Hold a world-space XY force at the effector site until the next update."""
        force = jnp.asarray(effector_force, dtype=state.effector_force.dtype)
        return eqx.tree_at(lambda s: s.effector_force, state, force)

    @jax.named_scope("fbx.MJXSkeleton.step")
    def step(self, state: MJXSkeletonState, ctrl: Array) -> MJXSkeletonState:
        """One discrete MuJoCo step via ``mjx.step()``.

        Unlike ``vector_field()`` (which returns derivatives for Diffrax
        integration), this method performs a full discrete-time integration
        step using MuJoCo's native integrator. The COM wrench is evaluated
        at the start of each timestep and held through its internal stages
        (including RK4). The site force carries over to the next timestep.

        Args:
            state: Current skeleton state.
            ctrl: Actuator controls, shape ``(nu,)``.

        Returns:
            Updated MJXSkeletonState after one MuJoCo timestep.
        """
        from mujoco import mjx

        data = self._dynamics_data(state, ctrl)
        data = mjx.step(self.mjx_model, data)

        return MJXSkeletonState(
            qpos=data.qpos,
            qvel=data.qvel,
            effector_force=state.effector_force,
        )

    def inverse_kinematics(self, effector_state: CartesianState) -> MJXSkeletonState:
        """Not implemented for MJX skeletons.

        MJX has no analytical inverse kinematics. Numerical IK is future work.
        MJXPlant cannot be used in SimpleFeedback graphs requiring IK.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "MJXSkeleton does not support inverse kinematics. "
            "MJX has no analytical IK; numerical IK is future work."
        )

    def init(self, *, key: PRNGKeyArray) -> MJXSkeletonState:
        """Initialize skeleton state with small random perturbation.

        Args:
            key: PRNG key for random initialization.

        Returns:
            Initial MJXSkeletonState.
        """
        from mujoco import mjx

        data = mjx.make_data(self.mjx_model)
        qpos = data.qpos + jax.random.normal(key, (self.nq,)) * 0.01
        data = data.replace(qpos=qpos)
        data = mjx.forward(self.mjx_model, data)

        return MJXSkeletonState(
            qpos=data.qpos,
            qvel=data.qvel,
            effector_force=jnp.zeros(2),
        )

    @property
    def input_size(self) -> int:
        """Number of actuator inputs."""
        return int(self.mjx_model.nu)

    @property
    def bounds(self) -> StateBounds[MJXSkeletonState]:
        """State bounds derived from MuJoCo joint limits (cached at init)."""
        return self._bounds
