"""Tests for feedbax.mechanics.model_builder."""

import jax
import jax.numpy as jnp
import pytest

from feedbax.mechanics.body import BodyPreset, default_3link_bounds, sample_preset
from feedbax.mechanics.model_builder import (
    ChainConfig,
    SimConfig,
    build_model,
    get_body_id,
    get_site_id,
)
from feedbax.mechanics.muscle_config import (
    default_6muscle_2link_topology,
    default_monoarticular_topology,
)


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
    bounds = default_3link_bounds()
    return sample_preset(bounds, key)


@pytest.fixture
def model(preset, chain_config, sim_config):
    return build_model(preset, chain_config, sim_config)


class TestChainConfig:
    def test_n_muscles_default(self):
        cfg = ChainConfig(n_joints=3)
        assert cfg.n_muscles == 6  # 3 joints * 2 muscles/joint

    def test_n_muscles_custom_topology(self):
        topo = default_6muscle_2link_topology()
        cfg = ChainConfig(n_joints=2, muscle_topology=topo)
        assert cfg.n_muscles == 6

    def test_defaults(self):
        cfg = ChainConfig()
        assert cfg.n_joints == 3
        assert cfg.muscle_topology is not None
        assert cfg.muscle_topology.n_joints == 3


class TestSimConfig:
    def test_defaults(self):
        cfg = SimConfig()
        assert cfg.dt == 0.002
        assert cfg.episode_duration == 2.0


class TestBuildModel:
    def test_compiles(self, model):
        assert model is not None
        assert model.nq == 3  # 3 hinge joints
        assert model.nv == 3

    def test_actuators(self, model, chain_config):
        # Now one torque actuator per joint, not per muscle.
        assert model.nu == chain_config.n_joints

    def test_effector_site(self, model):
        sid = get_site_id(model, "effector")
        assert sid >= 0

    def test_body_ids(self, model):
        for i in range(3):
            bid = get_body_id(model, f"link{i}")
            assert bid >= 0

    def test_site_not_found(self, model):
        sid = get_site_id(model, "nonexistent")
        assert sid == -1


def _mjx_available() -> bool:
    try:
        from mujoco import mjx
        return True
    except ImportError:
        return False


class TestMJXTransfer:
    @pytest.mark.skipif(
        not _mjx_available(), reason="mujoco-mjx not installed"
    )
    def test_to_mjx(self, model):
        from feedbax.mechanics.model_builder import to_mjx
        mjx_model = to_mjx(model)
        assert mjx_model.nq == model.nq
        assert mjx_model.nv == model.nv
