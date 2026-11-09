"""Stateful MuJoCo XLA skeleton used by native and continuous physics backends."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox import field
from jaxtyping import Array, PRNGKeyArray, Scalar

from feedbax.mechanics.skeleton.skeleton import AbstractSkeleton, AbstractSkeletonState
from feedbax.runtime.state import CartesianState, SpatialCartesianState, StateBounds

if TYPE_CHECKING:
    from mujoco.mjx import Data

    from feedbax.mechanics.compiled_model import CompiledPhysicalModel


class AppliedWrench(eqx.Module):
    """Force, free moment and application point in the same world or body frame.

    A planar wrench has XY force and point and a one-element Z moment. A
    spatial wrench has XYZ force, point and moment. A body-frame point is
    relative to the body's origin, and a world-frame point is absolute.
    """

    force: Array
    moment: Array
    point: Array
    frame: Literal["world", "body"] = field(static=True)


class MusclePathKinematics(NamedTuple):
    """Lengths (m), lengthening speeds (m/s), and d(length)/d(generalized displacement)."""

    length: Array
    velocity: Array
    jacobian: Array


class _BodyWrenches(eqx.Module):
    force: Array
    moment: Array
    point: Array
    body_frame: Array


class MJXSkeletonState(AbstractSkeletonState):
    """Physical state, held loads and the provider's native continuation vector."""

    qpos: Array = field(default_factory=lambda: jnp.zeros(1))
    qvel: Array = field(default_factory=lambda: jnp.zeros(1))
    effector_force: Array = field(default_factory=lambda: jnp.zeros(2))
    act: Array = field(default_factory=lambda: jnp.zeros(0))
    time: Array = field(default_factory=lambda: jnp.zeros(()))
    continuation: Array = field(default_factory=lambda: jnp.zeros(0))
    wrenches: _BodyWrenches | None = None


class MJXSkeleton(AbstractSkeleton[MJXSkeletonState]):
    """Shared MJX state, observation and dynamics for planar and spatial models."""

    mjx_model: object
    _warp_data_template: object | None
    descriptor: CompiledPhysicalModel | None = field(static=True)
    effector_site_id: int = field(static=True)
    effector_body_id: int = field(static=True)
    dimension: Literal[2, 3] = field(static=True)
    nq: int = field(static=True)
    nv: int = field(static=True)
    nbody: int = field(static=True)
    constraint_rows: int = field(static=True)
    _path_tendons: tuple[int, ...] = field(static=True)
    _path_actuators: tuple[int, ...] = field(static=True)
    _path_segments: tuple[tuple[int, int], ...] = field(static=True)
    _quaternions: tuple[tuple[int, int], ...] = field(static=True)
    _continuation_mask: int = field(static=True)
    _bounds: StateBounds

    def __init__(
        self,
        mjx_model,
        effector_site_id: int,
        effector_body_id: int,
        *,
        descriptor: CompiledPhysicalModel | None = None,
        warp_data_template=None,
        warp_constraint_rows: int | None = None,
    ):
        from mujoco import mjtJoint, mjtState, mjx

        if not 0 <= effector_site_id < mjx_model.nsite:
            raise ValueError(f"Invalid effector site index {effector_site_id}")
        site_body_id = int(mjx_model.site_bodyid[effector_site_id])
        if effector_body_id != site_body_id:
            raise ValueError(
                f"Effector site {effector_site_id} belongs to body {site_body_id}, "
                f"but body {effector_body_id} was specified"
            )
        self.mjx_model = mjx_model
        self._warp_data_template = warp_data_template
        self.descriptor = descriptor
        self.effector_site_id = effector_site_id
        self.effector_body_id = site_body_id
        self.dimension = 2 if descriptor is None else descriptor.dimension
        self.nq = int(mjx_model.nq)
        self.nv = int(mjx_model.nv)
        self.nbody = int(mjx_model.nbody)
        paths = () if descriptor is None else descriptor.paths
        self._path_tendons = tuple(path.tendon_id for path in paths)
        self._path_actuators = tuple(
            path.actuator_id for path in paths if path.actuator_id is not None
        )
        if len(self._path_actuators) != len(paths):
            raise ValueError("The authored MJX skeleton requires muscle path transmissions")
        self._path_segments = tuple(
            (descriptor.site_ids[a], descriptor.site_ids[b])
            for path in paths
            for a, b in zip(path.sites, path.sites[1:])
        )
        if sorted(self._path_actuators) != list(range(self.input_size, int(mjx_model.nu))):
            raise ValueError("Authored muscle transmissions must follow native controls")
        # Include rows inactive in the initial pose.
        self.constraint_rows = (
            mjx.make_data(mjx_model, impl="jax")._impl.efc_J.shape[0]
            if warp_constraint_rows is None else warp_constraint_rows
        )
        self._continuation_mask = int(
            mjtState.mjSTATE_WARMSTART | mjtState.mjSTATE_EQ_ACTIVE
            | mjtState.mjSTATE_MOCAP_POS | mjtState.mjSTATE_MOCAP_QUAT
        )
        if mjx_model.nhistory or mjx_model.npluginstate or mjx_model.nplastic:
            raise ValueError("MJX continuation does not support history, plugins or plasticity")

        self._quaternions = ()
        if descriptor is not None:
            self._quaternions = tuple(
                (joint.qpos.stop - 4, joint.qvel.stop - 3)
                for joint in descriptor.joints if joint.kind in {"ball", "free"}
            )
        elif np.isin(
            mjx_model.jnt_type, [int(mjtJoint.mjJNT_BALL), int(mjtJoint.mjJNT_FREE)],
        ).any():
            raise ValueError("Ball and free joints require a spatial compiled descriptor")
        if self._quaternions and self.dimension != 3:
            raise ValueError("Ball and free joints require a spatial compiled descriptor")

        scalar_joint = np.isin(
            mjx_model.jnt_type, [int(mjtJoint.mjJNT_HINGE), int(mjtJoint.mjJNT_SLIDE)],
        )
        limited = np.asarray(mjx_model.jnt_limited, dtype=bool) & scalar_joint
        self._bounds = StateBounds(low=None, high=None)
        if np.any(limited):
            addresses = mjx_model.jnt_qposadr[limited]
            ranges = np.asarray(mjx_model.jnt_range)[limited]
            low = np.full(self.nq, -np.inf, dtype=ranges.dtype)
            high = np.full(self.nq, np.inf, dtype=ranges.dtype)
            low[addresses], high[addresses] = ranges.T

            def bound(value):
                return MJXSkeletonState(
                    qpos=jnp.asarray(value), qvel=None, effector_force=None,
                    act=None, time=None, continuation=None,
                )

            self._bounds = StateBounds(low=bound(low), high=bound(high))

    @classmethod
    def from_descriptor(
        cls, descriptor: CompiledPhysicalModel, *, backend: Literal["jax", "warp"] = "jax",
    ) -> MJXSkeleton:
        from mujoco import mjx

        if descriptor.effector_site is None:
            raise ValueError("An executable MuJoCo model requires an effector site")
        site_id = descriptor.site_ids[descriptor.effector_site]
        model = descriptor.model
        if backend == "jax":
            mjx_model, template, constraint_rows = mjx.put_model(model, impl="jax"), None, None
        elif backend == "warp":
            from feedbax.mechanics.skeleton.mjx_warp import make_warp_model_and_data

            mjx_model, template, constraint_rows = make_warp_model_and_data(model)
        else:
            raise ValueError(f"Unknown MJX implementation {backend!r}")
        return cls(
            mjx_model, site_id, int(model.site_bodyid[site_id]),
            descriptor=descriptor, warp_data_template=template,
            warp_constraint_rows=constraint_rows,
        )

    @property
    def implementation(self) -> Literal["jax", "warp"]:
        return "jax" if self._warp_data_template is None else "warp"

    def _fresh_data(self):
        from mujoco import mjx

        if self._warp_data_template is not None:
            return self._warp_data_template
        return mjx.make_data(self.mjx_model, impl="jax")

    def retract_configuration(self, qpos: Array) -> Array:
        """Project quaternion coordinates onto the unit sphere."""
        for start, _ in self._quaternions:
            quat = qpos[start:start + 4]
            norm = jnp.linalg.norm(quat)
            norm = eqx.error_if(
                norm, ~jnp.isfinite(norm) | (norm <= jnp.finfo(quat.dtype).eps),
                "MuJoCo configuration contains an invalid quaternion",
            )
            qpos = qpos.at[start:start + 4].set(quat / norm)
        return qpos

    def configuration_derivative(self, qpos: Array, qvel: Array) -> Array:
        """MuJoCo uses scalar-first quaternions and body-local angular velocity."""
        pieces = []
        previous_pos = previous_vel = 0
        for pos, vel in self._quaternions:
            pieces.append(qvel[previous_vel:vel])
            scalar, vector = qpos[pos], qpos[pos + 1:pos + 4]
            omega = qvel[vel:vel + 3]
            pieces.append(0.5 * jnp.concatenate((
                jnp.array([-jnp.dot(vector, omega)]),
                scalar * omega + jnp.cross(vector, omega),
            )))
            previous_pos, previous_vel = pos + 4, vel + 3
        pieces.append(qvel[previous_vel:previous_vel + self.nq - previous_pos])
        return jnp.concatenate(pieces)

    def project_state(self, state: MJXSkeletonState) -> MJXSkeletonState:
        state = self._cast_state(state)
        return replace(state, qpos=self.retract_configuration(state.qpos))

    def _cast_state(self, state: MJXSkeletonState) -> MJXSkeletonState:
        if self._warp_data_template is None:
            return state
        return jax.tree.map(
            lambda value: jnp.asarray(value, dtype=self._warp_data_template.qpos.dtype), state,
        )

    def _reconstruct_data(self, state: MJXSkeletonState):
        from mujoco import mjx

        model = self.mjx_model
        data = self._fresh_data()
        if state.continuation.size:
            data = mjx.set_state(model, data, state.continuation, self._continuation_mask)
        return data.replace(
            qpos=self.retract_configuration(state.qpos), qvel=state.qvel,
            act=state.act, time=state.time,
        )

    def _kinematics_data(self, state: MJXSkeletonState):
        from mujoco import mjx

        data = mjx.kinematics(self.mjx_model, self._reconstruct_data(state))
        return mjx.com_pos(self.mjx_model, data)

    def _with_wrenches(self, state: MJXSkeletonState, data: Data) -> Data:
        wrenches = data.xfrc_applied
        if state.wrenches is not None:
            body = state.wrenches
            local = body.body_frame[:, None] > 0.5

            def world_vector(value):
                return jnp.where(local, jnp.einsum("bij,bj->bi", data.xmat, value), value)

            force = world_vector(body.force)
            point = world_vector(body.point) + jnp.where(local, data.xpos, 0)
            moment = world_vector(body.moment) + jnp.cross(point - data.xipos, force)
            wrenches = jnp.concatenate((force, moment), axis=1)

        site_force = jnp.pad(state.effector_force, (0, 3 - self.dimension))
        arm = data.site_xpos[self.effector_site_id] - data.xipos[self.effector_body_id]
        wrench = jnp.concatenate((site_force, jnp.cross(arm, site_force)))
        return data.replace(
            xfrc_applied=wrenches.at[self.effector_body_id].add(wrench),
        )

    def _dynamics_data(self, state: MJXSkeletonState, ctrl: Array) -> Data:
        from mujoco import mjx

        data = mjx.kinematics(self.mjx_model, self._reconstruct_data(state))
        if self._warp_data_template is not None:
            ctrl = jnp.asarray(ctrl, dtype=data.ctrl.dtype)
        return self._with_wrenches(state, data).replace(
            ctrl=jnp.pad(ctrl, (0, len(self._path_actuators))),
        )

    def continuous_stage(self, state: MJXSkeletonState) -> Data:
        """Evaluate the position and velocity of a constraint-free integration stage."""
        from mujoco import mjx

        if self.constraint_rows:
            raise ValueError("A coupled MJX stage requires zero compiled constraint rows")
        model = self.mjx_model
        if self._warp_data_template is not None:
            state = self._cast_state(state)
            data = mjx.tendon(model, self._kinematics_data(state))
            return self._with_wrenches(state, data)
        data = mjx.fwd_position(model, self._reconstruct_data(state))
        return mjx.fwd_velocity(model, self._with_wrenches(state, data))

    def muscle_path_kinematics(self, stage: Data) -> MusclePathKinematics:
        """Read authored paths in descriptor order using MJX's nv-column tendon Jacobian."""
        tendons = np.asarray(self._path_tendons, dtype=int)
        lengths = stage.ten_length[tendons]
        if self._path_segments:
            first, second = np.asarray(self._path_segments).T
            delta = stage.site_xpos[first] - stage.site_xpos[second]
            invalid = ~jnp.isfinite(delta).all() | (delta == 0).all(axis=1).any()
            lengths = eqx.error_if(
                lengths, invalid, "A muscle path has a coincident or nonfinite fixed segment",
            )
        if self._warp_data_template is not None:
            from feedbax.mechanics.skeleton.mjx_warp import tendon_jacobian

            jacobian = tendon_jacobian(self.mjx_model, stage, tendons)
        else:
            jacobian = stage._impl.ten_J[tendons]
        return MusclePathKinematics(lengths, jacobian @ stage.qvel, jacobian)

    def tendon_controls(self, tension: Array) -> Array:
        """Set muscle shortening tensions (N); leave other actuator controls at zero."""
        if tension.shape != (len(self._path_actuators),):
            raise ValueError(f"Expected {len(self._path_actuators)} muscle tensions")
        ids = np.asarray(self._path_actuators, dtype=int)
        return jnp.zeros(int(self.mjx_model.nu), dtype=tension.dtype).at[ids].set(tension)

    def continuous_derivative(
        self, state: MJXSkeletonState, stage: Data, ctrl: Array,
    ) -> MJXSkeletonState:
        """Finish a prepared constraint-free stage with its evaluated native controls."""
        from mujoco import mjx

        model = self.mjx_model
        if self._warp_data_template is not None:
            controls = jnp.asarray(ctrl, dtype=stage.ctrl.dtype)
            data = mjx.forward(model, stage.replace(ctrl=controls))
            return self._derivative(state, data, self._require_warp_capacity(data, data.qacc))
        data = mjx.fwd_actuation(model, stage.replace(ctrl=ctrl))
        data = mjx.fwd_acceleration(model, data)
        return self._derivative(state, data, data.qacc_smooth)

    def _derivative(self, state: MJXSkeletonState, data: Data, acceleration: Array):
        return replace(
            jax.tree.map(jnp.zeros_like, state),
            qpos=self.configuration_derivative(data.qpos, data.qvel),
            qvel=acceleration, act=data.act_dot, time=jnp.ones_like(state.time),
        )

    @jax.named_scope("fbx.MJXSkeleton.vector_field")
    def vector_field(self, t: Scalar, state: MJXSkeletonState, input: Array) -> MJXSkeletonState:
        from mujoco import mjx

        data = mjx.forward(self.mjx_model, self._dynamics_data(state, input))
        return self._derivative(state, data, data.qacc)

    def _cartesian(self, pos, vel, rotation) -> CartesianState:
        pos = pos[..., :self.dimension]
        vel = vel[..., :self.dimension]
        force = jnp.zeros_like(pos)
        if self.dimension == 3:
            return SpatialCartesianState(pos=pos, vel=vel, force=force, orientation=rotation)
        return CartesianState(pos=pos, vel=vel, force=force)

    @jax.named_scope("fbx.MJXSkeleton.forward_kinematics")
    def forward_kinematics(self, state: MJXSkeletonState) -> CartesianState:
        from mujoco import mjx

        model = self.mjx_model
        data = self._kinematics_data(state)
        if self._warp_data_template is not None:
            vel = jax.vmap(
                lambda pos, body: data.qvel @ mjx.jac(model, data, pos, body)[0],
            )(data.xpos, jnp.arange(self.nbody))
        else:
            data = mjx.com_vel(model, data)
            offset = data.xpos - data.subtree_com[model.body_rootid]
            vel = data.cvel[:, 3:] + jnp.cross(data.cvel[:, :3], offset)
        return self._cartesian(data.xpos, vel, data.xmat)

    @jax.named_scope("fbx.MJXSkeleton.effector")
    def effector(self, state: MJXSkeletonState) -> CartesianState:
        from mujoco import mjx

        data = self._kinematics_data(state)
        site = self.effector_site_id
        pos = data.site_xpos[site]
        jac_pos, _ = mjx.jac(self.mjx_model, data, pos, self.effector_body_id)
        return self._cartesian(pos, data.qvel @ jac_pos, data.site_xmat[site])

    def update_state_given_effector_force(
        self, effector_force: Array, state: MJXSkeletonState, *, key: PRNGKeyArray | None = None,
    ) -> MJXSkeletonState:
        force = jnp.asarray(effector_force, dtype=state.effector_force.dtype)
        if force.shape != (self.dimension,):
            raise ValueError(f"Effector force requires shape ({self.dimension},), got {force.shape}")
        return replace(state, effector_force=force)

    def with_wrench(
        self, state: MJXSkeletonState, body_id: int, wrench: AppliedWrench,
    ) -> MJXSkeletonState:
        """Replace the held load on one body. The next integration stage transforms it."""
        if not 0 < body_id < self.nbody or wrench.frame not in {"body", "world"}:
            raise ValueError("Wrench requires a non-world body and a world or body frame")
        force, moment, point = (
            jnp.asarray(value, dtype=state.qpos.dtype)
            for value in (wrench.force, wrench.moment, wrench.point)
        )
        if force.shape != (self.dimension,) or point.shape != (self.dimension,):
            raise ValueError(f"Wrench force and point require shape ({self.dimension},)")
        if moment.shape != (1 if self.dimension == 2 else 3,):
            raise ValueError(f"Wrench moment does not match the {self.dimension}D model")
        if self.dimension == 2:
            force, point, moment = jnp.pad(force, (0, 1)), jnp.pad(point, (0, 1)), jnp.pad(moment, (2, 0))
        body = state.wrenches
        if body is None:
            body = self._empty_wrenches(state.qpos.dtype)
        return replace(state, wrenches=_BodyWrenches(
            body.force.at[body_id].set(force), body.moment.at[body_id].set(moment),
            body.point.at[body_id].set(point), body.body_frame.at[body_id].set(wrench.frame == "body"),
        ))

    @jax.named_scope("fbx.MJXSkeleton.step")
    def step(self, state: MJXSkeletonState, ctrl: Array) -> MJXSkeletonState:
        """Advance the authored integrator with controls and COM wrenches held for the step."""
        from mujoco import mjx

        state = self._cast_state(state)
        data = mjx.step(self.mjx_model, self._dynamics_data(state, ctrl))
        return replace(
            state, qpos=self._require_warp_capacity(data, data.qpos),
            qvel=data.qvel, act=data.act, time=data.time,
            continuation=mjx.get_state(self.mjx_model, data, self._continuation_mask),
        )

    def _require_warp_capacity(self, data: Data, value: Array) -> Array:
        if self._warp_data_template is None:
            return value
        return eqx.error_if(
            value, jnp.any(data._impl.overflow != 0),
            "MJX-Warp exceeded its allocated collision or constraint capacity",
            on_error="raise",
        )

    def inverse_kinematics(self, effector_state: CartesianState) -> MJXSkeletonState:
        raise NotImplementedError("Initialize an MJX model with generalized coordinates")

    def _empty_wrenches(self, dtype) -> _BodyWrenches:
        zeros = jnp.zeros((self.nbody, 3), dtype=dtype)
        return _BodyWrenches(zeros, zeros, zeros, jnp.zeros(self.nbody, dtype=dtype))

    def reset_continuation(self, state: MJXSkeletonState) -> MJXSkeletonState:
        """Reinitialize provider buffers after transferring the common physical state."""
        from mujoco import mjx

        fresh = self._fresh_data()
        state = jax.tree.map(lambda value: jnp.asarray(value, dtype=fresh.qpos.dtype), state)
        data = self._reconstruct_data(state).replace(qacc_warmstart=fresh.qacc_warmstart)
        return replace(state, continuation=mjx.get_state(
            self.mjx_model, data, self._continuation_mask,
        ))

    def init(self, *, key: PRNGKeyArray) -> MJXSkeletonState:
        from mujoco import mjx

        data = self._fresh_data()
        return MJXSkeletonState(
            qpos=data.qpos, qvel=data.qvel, act=data.act, time=data.time,
            effector_force=jnp.zeros(self.dimension, dtype=data.qpos.dtype),
            continuation=mjx.get_state(self.mjx_model, data, self._continuation_mask),
            wrenches=self._empty_wrenches(data.qpos.dtype),
        )

    @property
    def input_size(self) -> int:
        return int(self.mjx_model.nu) - len(self._path_actuators)

    @property
    def bounds(self) -> StateBounds[MJXSkeletonState]:
        return self._bounds
