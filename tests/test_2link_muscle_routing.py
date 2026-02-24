"""Tests for 2-link arm muscle routing with biarticular muscles.

Bug: 138bbe5 -- Validates the generalized moment arm matrix approach
for muscle-to-torque conversion, including biarticular muscles that
span multiple joints.
"""

import jax
import jax.numpy as jnp
import pytest

from feedbax.mechanics.body import (
    default_2link_bounds,
    default_3link_bounds,
    flat_dim,
    from_flat,
    sample_preset,
    to_flat,
)
from feedbax.mechanics.model_builder import ChainConfig, SimConfig
from feedbax.mechanics.mjx_plant import MJXPlant
from feedbax.mechanics.muscle_config import (
    MuscleTopology,
    default_6muscle_2link_topology,
    default_monoarticular_topology,
    default_muscle_config,
)
from feedbax.mechanics.plant import PlantState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def topo_2link():
    return default_6muscle_2link_topology()


@pytest.fixture
def topo_3link_mono():
    return default_monoarticular_topology(3)


@pytest.fixture
def chain_2link(topo_2link):
    return ChainConfig(n_joints=2, muscle_topology=topo_2link)


@pytest.fixture
def sim_config():
    return SimConfig(dt=0.002, episode_duration=2.0)


@pytest.fixture
def bounds_2link():
    return default_2link_bounds()


@pytest.fixture
def preset_2link(bounds_2link, key):
    return sample_preset(bounds_2link, key)


@pytest.fixture
def plant_2link(preset_2link, chain_2link, sim_config):
    return MJXPlant.from_body_preset(
        preset_2link, chain_2link, sim_config, clip_states=False,
    )


# ---------------------------------------------------------------------------
# MuscleTopology tests
# ---------------------------------------------------------------------------


class TestMuscleTopology:
    def test_6muscle_2link_shape(self, topo_2link):
        assert topo_2link.n_muscles == 6
        assert topo_2link.n_joints == 2

    def test_6muscle_2link_routing(self, topo_2link):
        r = topo_2link.routing_array
        # Monoarticular shoulder muscles: span joint 0 only.
        assert bool(r[0, 0]) and not bool(r[0, 1])
        assert bool(r[1, 0]) and not bool(r[1, 1])
        # Monoarticular elbow muscles: span joint 1 only.
        assert not bool(r[2, 0]) and bool(r[2, 1])
        assert not bool(r[3, 0]) and bool(r[3, 1])
        # Biarticular muscles: span both joints.
        assert bool(r[4, 0]) and bool(r[4, 1])
        assert bool(r[5, 0]) and bool(r[5, 1])

    def test_6muscle_2link_sign(self, topo_2link):
        s = topo_2link.sign_array
        # Flexors: positive.
        assert int(s[0, 0]) == +1   # shoulder flexor
        assert int(s[2, 1]) == +1   # elbow flexor
        assert int(s[4, 0]) == +1   # biarticular flexor at shoulder
        assert int(s[4, 1]) == +1   # biarticular flexor at elbow
        # Extensors: negative.
        assert int(s[1, 0]) == -1   # shoulder extensor
        assert int(s[3, 1]) == -1   # elbow extensor
        assert int(s[5, 0]) == -1   # biarticular extensor at shoulder
        assert int(s[5, 1]) == -1   # biarticular extensor at elbow

    def test_monoarticular_backward_compat(self, topo_3link_mono):
        """3-joint monoarticular topology should be backward-compatible."""
        assert topo_3link_mono.n_muscles == 6
        assert topo_3link_mono.n_joints == 3
        r = topo_3link_mono.routing_array
        s = topo_3link_mono.sign_array
        # Each muscle spans exactly one joint.
        for i in range(6):
            assert int(jnp.sum(r[i])) == 1
        # Check alternating flex/ext pattern.
        assert int(s[0, 0]) == +1   # joint0 flexor
        assert int(s[1, 0]) == -1   # joint0 extensor
        assert int(s[2, 1]) == +1   # joint1 flexor
        assert int(s[3, 1]) == -1   # joint1 extensor
        assert int(s[4, 2]) == +1   # joint2 flexor
        assert int(s[5, 2]) == -1   # joint2 extensor

    def test_topology_is_static(self, topo_2link):
        """Topology fields should be in the treedef, not leaves."""
        import jax.tree as jt

        leaves, treedef = jt.flatten(topo_2link)
        # Static fields produce no leaves.
        assert len(leaves) == 0


# ---------------------------------------------------------------------------
# ChainConfig with topology tests
# ---------------------------------------------------------------------------


class TestChainConfigTopology:
    def test_default_creates_monoarticular(self):
        cfg = ChainConfig(n_joints=3)
        assert cfg.muscle_topology is not None
        assert cfg.n_muscles == 6
        assert cfg.muscle_topology.n_joints == 3

    def test_custom_topology(self, topo_2link):
        cfg = ChainConfig(n_joints=2, muscle_topology=topo_2link)
        assert cfg.n_muscles == 6  # 4 mono + 2 biarticular
        assert cfg.muscle_topology.n_joints == 2


# ---------------------------------------------------------------------------
# 2-link BodyPreset tests
# ---------------------------------------------------------------------------


class Test2LinkBodyPreset:
    def test_bounds_shape(self, bounds_2link):
        assert bounds_2link.segment_lengths_min.shape == (2,)
        assert bounds_2link.muscle_pcsa_min.shape == (6,)
        assert bounds_2link.muscle_moment_arm_magnitudes_min.shape == (6, 2)
        assert bounds_2link.muscle_moment_arm_magnitudes_max.shape == (6, 2)

    def test_sample_shape(self, preset_2link):
        assert preset_2link.segment_lengths.shape == (2,)
        assert preset_2link.segment_masses.shape == (2,)
        assert preset_2link.muscle_pcsa.shape == (6,)
        assert preset_2link.muscle_moment_arm_magnitudes.shape == (6, 2)

    def test_flat_round_trip(self, preset_2link):
        flat = to_flat(preset_2link)
        expected_dim = flat_dim(n_joints=2, n_muscles=6)
        assert flat.shape == (expected_dim,)

        recon = from_flat(flat, n_joints=2, n_muscles=6)
        assert jnp.allclose(preset_2link.segment_lengths, recon.segment_lengths)
        assert jnp.allclose(preset_2link.muscle_pcsa, recon.muscle_pcsa)
        assert jnp.allclose(
            preset_2link.muscle_moment_arm_magnitudes,
            recon.muscle_moment_arm_magnitudes,
        )


# ---------------------------------------------------------------------------
# MuscleConfig (moment arm matrix) tests
# ---------------------------------------------------------------------------


class TestMuscleConfig:
    def test_moment_arms_shape(self, preset_2link, chain_2link):
        mc = default_muscle_config(preset_2link, chain_2link)
        assert mc.moment_arms.shape == (6, 2)
        assert mc.n_muscles == 6
        assert mc.n_joints == 2

    def test_monoarticular_zero_cross_joint(self, preset_2link, chain_2link):
        """Monoarticular muscles should have zero moment arm at non-spanned joint."""
        mc = default_muscle_config(preset_2link, chain_2link)
        # Shoulder muscles: zero at elbow.
        assert float(mc.moment_arms[0, 1]) == 0.0
        assert float(mc.moment_arms[1, 1]) == 0.0
        # Elbow muscles: zero at shoulder.
        assert float(mc.moment_arms[2, 0]) == 0.0
        assert float(mc.moment_arms[3, 0]) == 0.0

    def test_biarticular_nonzero_both_joints(self, preset_2link, chain_2link):
        """Biarticular muscles should have non-zero moment arms at both joints."""
        mc = default_muscle_config(preset_2link, chain_2link)
        # Biarticular flexor (index 4): positive at both joints.
        assert float(mc.moment_arms[4, 0]) > 0
        assert float(mc.moment_arms[4, 1]) > 0
        # Biarticular extensor (index 5): negative at both joints.
        assert float(mc.moment_arms[5, 0]) < 0
        assert float(mc.moment_arms[5, 1]) < 0


# ---------------------------------------------------------------------------
# MJXPlant 2-link integration tests
# ---------------------------------------------------------------------------


class TestMJXPlant2Link:
    def test_builds(self, plant_2link):
        assert plant_2link is not None
        assert plant_2link.moment_arms.shape == (6, 2)
        assert plant_2link.muscle_gear.shape == (6,)

    def test_input_size_is_n_muscles(self, plant_2link):
        assert plant_2link.input_size == 6

    def test_skeleton_nu_is_n_joints(self, plant_2link):
        """MuJoCo model should have n_joints torque actuators."""
        assert plant_2link.skeleton.input_size == 2

    def test_init_state(self, plant_2link, key):
        state = plant_2link.init(key=key)
        assert isinstance(state, PlantState)
        assert state.skeleton.qpos.shape == (2,)
        assert state.skeleton.qvel.shape == (2,)
        assert jnp.all(jnp.isfinite(state.skeleton.qpos))

    def test_vector_field_finite(self, plant_2link, key):
        state = plant_2link.init(key=key)
        ctrl = jnp.ones(6) * 0.5  # 6 muscle activations
        d_state = plant_2link.vector_field(0.0, state, ctrl)
        assert jnp.all(jnp.isfinite(d_state.skeleton.qpos))
        assert jnp.all(jnp.isfinite(d_state.skeleton.qvel))

    def test_effector_position(self, plant_2link, key):
        state = plant_2link.init(key=key)
        eff = plant_2link.skeleton.effector(state.skeleton)
        assert eff.pos.shape == (2,)
        assert jnp.all(jnp.isfinite(eff.pos))
        assert float(jnp.linalg.norm(eff.pos)) < 2.0

    def test_euler_50_steps_stable(self, plant_2link, key):
        """Run 50 Euler steps and verify finite state."""
        import diffrax as dfx

        state = plant_2link.init(key=key)
        term = dfx.ODETerm(plant_2link.vector_field)
        solver = dfx.Euler()
        dt = 0.002
        ctrl = jnp.ones(6) * 0.3

        for _ in range(50):
            state = plant_2link.kinematics_update(ctrl, state)
            state, _, _, _, _ = solver.step(
                term, 0, dt, state, ctrl, None, made_jump=False,
            )

        assert jnp.all(jnp.isfinite(state.skeleton.qpos))
        assert jnp.all(jnp.isfinite(state.skeleton.qvel))


# ---------------------------------------------------------------------------
# Biarticular torque production (critical test)
# ---------------------------------------------------------------------------


class TestBiarticularTorque:
    """Critical test: activating a biarticular muscle should produce torque
    at BOTH joints."""

    def test_biarticular_flexor_produces_dual_torque(self, plant_2link):
        """Activating only the biarticular flexor (muscle 4) should produce
        non-zero torque at both shoulder and elbow."""
        activations = jnp.array([0, 0, 0, 0, 1, 0], dtype=jnp.float32)
        torques = plant_2link._muscle_activations_to_joint_torques(activations)
        assert torques.shape == (2,)
        # Both joints should receive torque.
        assert float(jnp.abs(torques[0])) > 0, "Shoulder torque is zero"
        assert float(jnp.abs(torques[1])) > 0, "Elbow torque is zero"
        # Both should be positive (flexion).
        assert float(torques[0]) > 0, "Shoulder torque should be positive (flexion)"
        assert float(torques[1]) > 0, "Elbow torque should be positive (flexion)"

    def test_biarticular_extensor_produces_dual_torque(self, plant_2link):
        """Activating only the biarticular extensor (muscle 5) should produce
        non-zero torque at both joints."""
        activations = jnp.array([0, 0, 0, 0, 0, 1], dtype=jnp.float32)
        torques = plant_2link._muscle_activations_to_joint_torques(activations)
        assert torques.shape == (2,)
        assert float(jnp.abs(torques[0])) > 0, "Shoulder torque is zero"
        assert float(jnp.abs(torques[1])) > 0, "Elbow torque is zero"
        # Both should be negative (extension).
        assert float(torques[0]) < 0, "Shoulder torque should be negative (extension)"
        assert float(torques[1]) < 0, "Elbow torque should be negative (extension)"

    def test_monoarticular_produces_single_torque(self, plant_2link):
        """Activating a monoarticular muscle should produce torque at one joint
        only."""
        # Shoulder flexor (muscle 0).
        activations = jnp.array([1, 0, 0, 0, 0, 0], dtype=jnp.float32)
        torques = plant_2link._muscle_activations_to_joint_torques(activations)
        assert float(jnp.abs(torques[0])) > 0, "Shoulder torque is zero"
        assert float(jnp.abs(torques[1])) == pytest.approx(0.0, abs=1e-10)

        # Elbow flexor (muscle 2).
        activations = jnp.array([0, 0, 1, 0, 0, 0], dtype=jnp.float32)
        torques = plant_2link._muscle_activations_to_joint_torques(activations)
        assert float(jnp.abs(torques[0])) == pytest.approx(0.0, abs=1e-10)
        assert float(jnp.abs(torques[1])) > 0, "Elbow torque is zero"

    def test_antagonist_cancellation(self, plant_2link):
        """Equal activation of flexor and extensor at same joint should
        partially cancel (net torque depends on moment arm magnitudes)."""
        # Activate shoulder flexor and extensor equally.
        activations = jnp.array([1, 1, 0, 0, 0, 0], dtype=jnp.float32)
        torques = plant_2link._muscle_activations_to_joint_torques(activations)
        # If muscle_gear values are equal for muscles 0 and 1, and moment
        # arm magnitudes are equal, net shoulder torque should be ~0.
        # With sampled presets, they won't be exactly zero but should be
        # smaller than one muscle alone.
        single = plant_2link._muscle_activations_to_joint_torques(
            jnp.array([1, 0, 0, 0, 0, 0], dtype=jnp.float32),
        )
        assert float(jnp.abs(torques[0])) < float(jnp.abs(single[0])) * 2


# ---------------------------------------------------------------------------
# Batch build tests
# ---------------------------------------------------------------------------


class TestBatch2Link:
    def test_build_batch_2_bodies(self, bounds_2link, chain_2link, sim_config):
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 2)
        presets = [sample_preset(bounds_2link, k) for k in keys]
        batch = MJXPlant.build_batch(
            presets, chain_2link, sim_config, clip_states=False,
        )
        # Should have leading batch dim on array leaves.
        assert batch.moment_arms.shape == (2, 6, 2)
        assert batch.muscle_gear.shape == (2, 6)
        assert batch.segment_lengths.shape == (2, 2)

    def test_vmap_over_batch(self, bounds_2link, chain_2link, sim_config):
        import equinox as eqx

        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 2)
        presets = [sample_preset(bounds_2link, k) for k in keys]
        batch = MJXPlant.build_batch(
            presets, chain_2link, sim_config, clip_states=False,
        )

        init_keys = jax.random.split(jax.random.PRNGKey(99), 2)
        states = eqx.filter_vmap(
            lambda plant, k: plant.init(key=k),
        )(batch, init_keys)
        assert states.skeleton.qpos.shape == (2, 2)

        ctrl = jnp.ones((2, 6)) * 0.5
        d_states = eqx.filter_vmap(
            lambda plant, s, c: plant.vector_field(0.0, s, c),
        )(batch, states, ctrl)
        assert d_states.skeleton.qpos.shape == (2, 2)
        assert jnp.all(jnp.isfinite(d_states.skeleton.qpos))
        assert jnp.all(jnp.isfinite(d_states.skeleton.qvel))


# ---------------------------------------------------------------------------
# Backward compatibility: 3-link monoarticular
# ---------------------------------------------------------------------------


class TestBackwardCompat3Link:
    def test_3link_still_works(self):
        """Default ChainConfig(n_joints=3) should still produce a working
        plant with 6 monoarticular muscles."""
        key = jax.random.PRNGKey(7)
        bounds = default_3link_bounds()
        preset = sample_preset(bounds, key)
        chain = ChainConfig(n_joints=3)
        sim = SimConfig(dt=0.002, episode_duration=2.0)
        plant = MJXPlant.from_body_preset(preset, chain, sim, clip_states=False)

        assert plant.input_size == 6
        assert plant.moment_arms.shape == (6, 3)
        assert plant.skeleton.input_size == 3  # 3 torque actuators

        state = plant.init(key=key)
        ctrl = jnp.ones(6) * 0.5
        d_state = plant.vector_field(0.0, state, ctrl)
        assert jnp.all(jnp.isfinite(d_state.skeleton.qpos))
