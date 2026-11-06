"""Tests for feedbax.mechanics.skeleton.mjx_skeleton and feedbax.mechanics.mjx_plant."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.mechanics.backend import MJXBackend
from feedbax.mechanics.body import default_3link_bounds, sample_preset
from feedbax.mechanics.mechanics import Mechanics
from feedbax.mechanics.model_builder import (
    ChainConfig,
    SimConfig,
    build_model,
    get_body_id,
    get_site_id,
    to_mjx,
)
from feedbax.mechanics.skeleton.mjx_skeleton import MJXSkeleton
from feedbax.mechanics.mjx_plant import MJXPlant
from feedbax.mechanics.plant import PlantState
from feedbax.runtime.state import CartesianState


pytestmark = pytest.mark.optional_mjx


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def chain_config():
    return ChainConfig(n_joints=3)


@pytest.fixture
def sim_config():
    return SimConfig(dt=0.002, episode_duration=2.0)


@pytest.fixture
def preset(key):
    return sample_preset(default_3link_bounds(), key)


@pytest.fixture
def mj_model(preset, chain_config, sim_config):
    return build_model(preset, chain_config, sim_config)


@pytest.fixture
def mjx_model(mj_model):
    return to_mjx(mj_model)


@pytest.fixture
def skeleton(mj_model, mjx_model, chain_config):
    return MJXSkeleton(
        mjx_model=mjx_model,
        effector_site_id=get_site_id(mj_model, "effector"),
        effector_body_id=get_body_id(mj_model, f"link{chain_config.n_joints - 1}"),
    )


@pytest.fixture
def plant(preset, chain_config, sim_config):
    """Build MJXPlant via from_body_preset so moment_arms/muscle_gear are set."""
    return MJXPlant.from_body_preset(preset, chain_config, sim_config, clip_states=False)


@pytest.fixture
def plant_from_preset(preset, chain_config, sim_config):
    return MJXPlant.from_body_preset(preset, chain_config, sim_config, clip_states=False)


@pytest.fixture
def state(plant, key):
    return plant.init(key=key)


class TestMJXSkeletonInit:
    def test_shapes(self, skeleton, key):
        state = skeleton.init(key=key)
        assert state.qpos.shape == (skeleton.nq,)
        assert state.qvel.shape == (skeleton.nv,)

    def test_finite(self, skeleton, key):
        state = skeleton.init(key=key)
        assert jnp.all(jnp.isfinite(state.qpos))
        assert jnp.all(jnp.isfinite(state.qvel))


class TestMJXSkeletonVectorField:
    def test_derivative_shapes(self, skeleton, key):
        state = skeleton.init(key=key)
        ctrl = jnp.zeros(skeleton.input_size)
        d_state = skeleton.vector_field(0.0, state, ctrl)
        assert d_state.qpos.shape == state.qpos.shape
        assert d_state.qvel.shape == state.qvel.shape

    def test_finite_derivatives(self, skeleton, key):
        state = skeleton.init(key=key)
        ctrl = jnp.ones(skeleton.input_size) * 0.5
        d_state = skeleton.vector_field(0.0, state, ctrl)
        assert jnp.all(jnp.isfinite(d_state.qpos))
        assert jnp.all(jnp.isfinite(d_state.qvel))


class TestMJXSkeletonEffector:
    def test_effector_position(self, skeleton, key):
        state = skeleton.init(key=key)
        eff = skeleton.effector(state)
        assert isinstance(eff, CartesianState)
        assert eff.pos.shape == (2,)
        assert jnp.all(jnp.isfinite(eff.pos))

    def test_effector_plausible(self, skeleton, key):
        state = skeleton.init(key=key)
        eff = skeleton.effector(state)
        # End-effector should be within reasonable workspace (< 2m from origin)
        assert jnp.linalg.norm(eff.pos) < 2.0


class TestMJXSkeletonForwardKinematics:
    def test_returns_cartesian(self, skeleton, key):
        state = skeleton.init(key=key)
        fk = skeleton.forward_kinematics(state)
        assert isinstance(fk, CartesianState)
        assert fk.pos.shape[1] == 2  # 2D positions
        assert fk.vel.shape[1] == 2


@pytest.fixture
def offset_hinge(key):
    import mujoco

    model = mujoco.MjModel.from_xml_string("""
        <mujoco>
            <option gravity="0 0 0" timestep="0.04" integrator="Euler"/>
            <worldbody>
                <body name="link">
                    <joint name="hinge" axis="0 0 1"/>
                    <inertial pos="0.2 0.1 0" mass="1" diaginertia="0.1 0.1 0.1"/>
                    <site name="tip" pos="0.8 -0.3 0"/>
                </body>
            </worldbody>
        </mujoco>
    """)
    skeleton = MJXSkeleton(to_mjx(model), get_site_id(model, "tip"), get_body_id(model, "link"))
    state = eqx.tree_at(
        lambda s: (s.qpos, s.qvel),
        skeleton.init(key=key),
        (jnp.array([0.6]), jnp.array([1.7])),
    )
    return model, skeleton, state


def _site_velocity_reference(model, skeleton, state):
    import mujoco

    data = mujoco.MjData(model)
    positions = []
    epsilon = 1e-5
    for dt in (-epsilon, epsilon):
        data.qpos[:] = np.asarray(state.qpos) + dt * np.asarray(state.qvel, dtype=float)
        mujoco.mj_kinematics(model, data)
        positions.append(data.site_xpos[skeleton.effector_site_id, :2].copy())
    return (positions[1] - positions[0]) / (2 * epsilon)


def test_effector_velocity_matches_site_motion(offset_hinge):
    model, skeleton, state = offset_hinge

    expected = _site_velocity_reference(model, skeleton, state)

    np.testing.assert_allclose(skeleton.effector(state).vel, expected, atol=1e-6)


def test_effector_force_satisfies_virtual_work(offset_hinge):
    model, skeleton, state = offset_hinge
    force = jnp.array([1.4, -0.9])
    application_state = eqx.tree_at(lambda s: s.qpos, state, jnp.array([-0.4]))
    loaded = skeleton.update_state_given_effector_force(force, application_state)
    loaded = eqx.tree_at(lambda s: s.qpos, loaded, state.qpos)
    ctrl = jnp.zeros(skeleton.input_size)

    unloaded_acc = skeleton.vector_field(0.0, state, ctrl).qvel[0]
    loaded_acc = skeleton.vector_field(0.0, loaded, ctrl).qvel[0]
    site_power = force @ _site_velocity_reference(model, skeleton, state)
    # Hinge inertia about its origin follows from the parallel-axis theorem.
    hinge_inertia = model.body_inertia[1, 2] + model.body_mass[1] * np.sum(model.body_ipos[1] ** 2)
    joint_power = hinge_inertia * (loaded_acc - unloaded_acc) * state.qvel[0]

    np.testing.assert_allclose(joint_power, site_power, atol=1e-6)


def test_native_substeps_apply_held_force_at_moving_site(offset_hinge, key):
    import mujoco

    model, skeleton, state = offset_hinge
    force = jnp.array([1.4, -0.9])
    substeps = 3
    dt = model.opt.timestep * substeps
    plant = MJXPlant(skeleton)
    backend = MJXBackend(control_dt=dt, sub_dt=model.opt.timestep)
    mechanics = Mechanics(plant, dt, backend=backend, key=key)
    storage = eqx.nn.State(mechanics)
    initial = storage.get(mechanics.state_index)
    initial = eqx.tree_at(
        lambda s: (s.plant.skeleton, s.effector.force), initial, (state, force),
    )
    storage = storage.set(mechanics.state_index, initial)

    inputs = {"force": jnp.zeros(plant.input_size)}
    outputs, storage = mechanics(inputs, storage, key=key)
    stepped = outputs["state"].plant.skeleton

    data = mujoco.MjData(model)
    data.qpos[:] = state.qpos
    data.qvel[:] = state.qvel
    for _ in range(substeps):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        data.qfrc_applied[:] = 0
        mujoco.mj_applyFT(
            model, data, np.array([*force, 0.0]), np.zeros(3),
            data.site_xpos[skeleton.effector_site_id], skeleton.effector_body_id,
            data.qfrc_applied,
        )
        mujoco.mj_step(model, data)

    np.testing.assert_allclose(stepped.qpos, data.qpos, atol=1e-6)
    np.testing.assert_allclose(stepped.qvel, data.qvel, atol=1e-6)

    # With no fresh disturbance, the next control step releases the site force.
    outputs, _ = mechanics(inputs, storage, key=key)
    stepped = outputs["state"].plant.skeleton
    data.qfrc_applied[:] = 0
    for _ in range(substeps):
        mujoco.mj_step(model, data)

    np.testing.assert_allclose(stepped.qpos, data.qpos, atol=1e-6)
    np.testing.assert_allclose(stepped.qvel, data.qvel, atol=1e-6)


class TestMJXSkeletonIK:
    def test_raises(self, skeleton):
        dummy = CartesianState(
            pos=jnp.zeros(2), vel=jnp.zeros(2), force=jnp.zeros(2),
        )
        with pytest.raises(NotImplementedError):
            skeleton.inverse_kinematics(dummy)


class TestMJXPlantInit:
    def test_shapes(self, state, plant):
        assert state.skeleton.qpos.shape == (plant.skeleton.nq,)
        assert state.muscles is None

    def test_from_preset(self, plant_from_preset, key):
        state = plant_from_preset.init(key=key)
        assert state.skeleton.qpos.shape[0] > 0
        assert jnp.all(jnp.isfinite(state.skeleton.qpos))


class TestMJXPlantInputSize:
    def test_matches_n_muscles(self, plant):
        # input_size is n_muscles (policy output dim), not n_joints (MuJoCo nu).
        assert plant.input_size == plant.moment_arms.shape[0]


class TestMJXPlantVectorField:
    def test_returns_plant_state(self, plant, state):
        ctrl = jnp.zeros(plant.input_size)
        d_state = plant.vector_field(0.0, state, ctrl)
        assert isinstance(d_state, PlantState)
        assert d_state.muscles is None
        assert jnp.all(jnp.isfinite(d_state.skeleton.qpos))
        assert jnp.all(jnp.isfinite(d_state.skeleton.qvel))


class TestMJXPlantDiffraxIntegration:
    def test_euler_100_steps(self, plant, state):
        """Integrate MJXPlant with Diffrax Euler for 100 steps."""
        import diffrax as dfx

        term = dfx.ODETerm(plant.vector_field)
        solver = dfx.Euler()
        dt = 0.002
        ctrl = jnp.zeros(plant.input_size)

        for _ in range(100):
            state = plant.kinematics_update(ctrl, state)
            new_state, _, _, _, _ = solver.step(
                term, 0, dt, state, ctrl, None, made_jump=False,
            )
            state = new_state

        assert jnp.all(jnp.isfinite(state.skeleton.qpos))
        assert jnp.all(jnp.isfinite(state.skeleton.qvel))


class TestMJXPlantVmap:
    def test_vmap_4_states(self, plant, key):
        """vmap over 4 plants with different initial states."""
        keys = jax.random.split(key, 4)
        states = jax.vmap(plant.init)(key=keys)

        ctrl = jnp.zeros((4, plant.input_size))
        d_states = jax.vmap(plant.vector_field, in_axes=(None, 0, 0))(
            0.0, states, ctrl,
        )
        assert d_states.skeleton.qpos.shape == (4, plant.skeleton.nq)
        assert jnp.all(jnp.isfinite(d_states.skeleton.qpos))
