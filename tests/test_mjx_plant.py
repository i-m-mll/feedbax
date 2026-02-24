"""Tests for feedbax.mechanics.skeleton.mjx_skeleton and feedbax.mechanics.mjx_plant."""

import jax
import jax.numpy as jnp
import pytest

from feedbax.mechanics.body import default_3link_bounds, sample_preset
from feedbax.mechanics.model_builder import (
    ChainConfig,
    SimConfig,
    build_model,
    get_body_id,
    get_site_id,
    to_mjx,
)
from feedbax.mechanics.skeleton.mjx_skeleton import MJXSkeleton, MJXSkeletonState
from feedbax.mechanics.mjx_plant import MJXPlant
from feedbax.mechanics.plant import PlantState
from feedbax.state import CartesianState


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def chain_config():
    return ChainConfig(n_joints=3, muscles_per_joint=2)


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
def plant(skeleton):
    return MJXPlant(skeleton=skeleton, clip_states=False)


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
        assert state.xfrc_applied.shape == (skeleton.nbody, 6)

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
        assert d_state.xfrc_applied.shape == state.xfrc_applied.shape

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

    def test_effector_velocity(self, skeleton, key):
        state = skeleton.init(key=key)
        eff = skeleton.effector(state)
        assert eff.vel.shape == (2,)
        assert jnp.all(jnp.isfinite(eff.vel))

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


class TestMJXSkeletonForce:
    def test_force_stored(self, skeleton, key):
        state = skeleton.init(key=key)
        force = jnp.array([1.5, -0.5])
        new_state = skeleton.update_state_given_effector_force(force, state)
        expected = jnp.zeros(6).at[:2].set(force)
        assert jnp.allclose(
            new_state.xfrc_applied[skeleton.effector_body_id], expected
        )


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
    def test_matches_model(self, plant, mjx_model):
        assert plant.input_size == int(mjx_model.nu)


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
