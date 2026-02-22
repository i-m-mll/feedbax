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
import jax.tree as jt
from jaxtyping import Array, Float, PRNGKeyArray, Scalar

from feedbax.mechanics.skeleton.skeleton import AbstractSkeleton, AbstractSkeletonState
from feedbax.state import CartesianState, StateBounds


class MJXSkeletonState(AbstractSkeletonState):
    """Configuration state for an MJX-based skeleton.

    Attributes:
        qpos: Generalized positions, shape ``(nq,)``.
        qvel: Generalized velocities, shape ``(nv,)``.
        xfrc_applied: External forces on bodies, shape ``(nbody, 6)``.
    """

    qpos: Float[Array, " nq"] = field(default_factory=lambda: jnp.zeros(1))
    qvel: Float[Array, " nv"] = field(default_factory=lambda: jnp.zeros(1))
    xfrc_applied: Float[Array, "nbody 6"] = field(
        default_factory=lambda: jnp.zeros((1, 6))
    )


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
        import numpy as np

        self.mjx_model = mjx_model
        self.effector_site_id = effector_site_id
        self.effector_body_id = effector_body_id
        self.nq = int(mjx_model.nq)
        self.nv = int(mjx_model.nv)
        self.nbody = int(mjx_model.nbody)

        # Compute bounds eagerly from concrete model data (not inside JIT)
        jnt_range = np.asarray(mjx_model.jnt_range)  # (njnt, 2)
        has_limits = np.any(jnt_range != 0)
        if has_limits:
            self._bounds = StateBounds(
                low=MJXSkeletonState(
                    qpos=jnp.array(jnt_range[:self.nq, 0]),
                    qvel=None,
                    xfrc_applied=None,
                ),
                high=MJXSkeletonState(
                    qpos=jnp.array(jnt_range[:self.nq, 1]),
                    qvel=None,
                    xfrc_applied=None,
                ),
            )
        else:
            self._bounds = StateBounds(low=None, high=None)

    def _reconstruct_data(
        self, state: MJXSkeletonState, ctrl: Array | None = None
    ):
        """Build an mjx.Data from skeleton state for forward computation."""
        from mujoco import mjx

        data = mjx.make_data(self.mjx_model)
        data = data.replace(
            qpos=state.qpos,
            qvel=state.qvel,
            xfrc_applied=state.xfrc_applied,
        )
        if ctrl is not None:
            data = data.replace(ctrl=ctrl)
        return data

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
            MJXSkeletonState with (d_qpos=qvel, d_qvel=qacc, d_xfrc=0).
        """
        from mujoco import mjx

        data = self._reconstruct_data(state, ctrl=input)
        data = mjx.forward(self.mjx_model, data)

        return MJXSkeletonState(
            qpos=state.qvel,  # d(qpos)/dt = qvel
            qvel=data.qacc,   # d(qvel)/dt = qacc
            xfrc_applied=jnp.zeros_like(state.xfrc_applied),
        )

    @jax.named_scope("fbx.MJXSkeleton.forward_kinematics")
    def forward_kinematics(self, state: MJXSkeletonState) -> CartesianState:
        """Compute Cartesian states of all bodies via forward kinematics.

        Args:
            state: Current skeleton state.

        Returns:
            CartesianState with positions and velocities of all bodies (2D).
        """
        from mujoco import mjx

        data = self._reconstruct_data(state)
        data = mjx.forward(self.mjx_model, data)

        # Extract 2D positions and velocities for all bodies
        pos = data.xpos[:, :2]  # (nbody, 2)
        vel = data.cvel[:, 3:5]  # (nbody, 2) — linear velocity components
        force = jnp.zeros_like(pos)

        return CartesianState(pos=pos, vel=vel, force=force)

    @jax.named_scope("fbx.MJXSkeleton.effector")
    def effector(self, state: MJXSkeletonState) -> CartesianState:
        """Return the Cartesian state of the end-effector site.

        Uses ``data.site_xpos`` for position and ``data.site_xvelp`` for
        velocity — no finite-difference approximation needed.

        Args:
            state: Current skeleton state.

        Returns:
            CartesianState with 2D effector position, velocity, and zero force.
        """
        from mujoco import mjx

        data = self._reconstruct_data(state)
        data = mjx.forward(self.mjx_model, data)

        pos = data.site_xpos[self.effector_site_id, :2]

        # MJX provides site velocities via site_xvelp (linear part)
        # site_xvelp has shape (nsite, 3)
        if hasattr(data, 'site_xvelp') and data.site_xvelp is not None:
            vel = data.site_xvelp[self.effector_site_id, :2]
        else:
            # Fallback: use body velocity
            vel = data.cvel[self.effector_body_id, 3:5]

        return CartesianState(
            pos=pos,
            vel=vel,
            force=jnp.zeros(2),
        )

    def update_state_given_effector_force(
        self,
        effector_force: Array,
        state: MJXSkeletonState,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> MJXSkeletonState:
        """Store an effector-space force in ``xfrc_applied``.

        Converts a 2D force into MuJoCo's 6D ``xfrc_applied`` format
        ``[fx, fy, 0, 0, 0, 0]`` on the effector body. Does NOT call
        ``mjx.forward()`` — just stores the force (same pattern as
        PointMass storing force in CartesianState).

        Args:
            effector_force: 2D force vector, shape ``(2,)``.
            state: Current skeleton state.
            key: Unused (satisfies AbstractSkeleton interface).

        Returns:
            Updated MJXSkeletonState with force in xfrc_applied.
        """
        xfrc = jnp.zeros_like(state.xfrc_applied)
        force_6d = jnp.zeros(6).at[:2].set(effector_force)
        xfrc = xfrc.at[self.effector_body_id].set(force_6d)
        return eqx.tree_at(
            lambda s: s.xfrc_applied, state, xfrc,
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
            xfrc_applied=jnp.zeros((self.nbody, 6)),
        )

    @property
    def input_size(self) -> int:
        """Number of actuator inputs."""
        return int(self.mjx_model.nu)

    @property
    def bounds(self) -> StateBounds[MJXSkeletonState]:
        """State bounds derived from MuJoCo joint limits (cached at init)."""
        return self._bounds
