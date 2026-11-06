"""Tests for feedbax.mechanics.backend and Mechanics backend integration.

Tests both DiffraxBackend and MJXBackend, along with Mechanics integration,
backward compatibility, vmap, and gradient checkpointing.

:copyright: Copyright 2025 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from feedbax.runtime.graph import init_state_from_component
from feedbax.mechanics import (
    DiffraxBackend,
    MJXBackend,
    Mechanics,
    MechanicsState,
    PhysicsBackend,
    PhysicsState,
)
from feedbax.mechanics.plant import DirectForceInput
from feedbax.mechanics.skeleton.pointmass import PointMass
from feedbax.runtime.state import CartesianState

try:
    import mujoco  # noqa: F401
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

requires_mujoco = pytest.mark.skipif(
    not HAS_MUJOCO, reason="mujoco not installed"
)


# ---------------------------------------------------------------------------
# Fixtures: PointMass-based (no MuJoCo dependency)
# ---------------------------------------------------------------------------

@pytest.fixture
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def pointmass_plant():
    """A simple DirectForceInput wrapping a PointMass."""
    return DirectForceInput(PointMass(mass=1.0, damping=0.0), clip_states=False)


@pytest.fixture
def dt():
    return 0.01


@pytest.fixture
def diffrax_backend(dt):
    """DiffraxBackend with single substep (matches legacy behavior)."""
    return DiffraxBackend(control_dt=dt)


@pytest.fixture
def diffrax_backend_multi(dt):
    """DiffraxBackend with 5 substeps."""
    return DiffraxBackend(control_dt=dt, sub_dt=dt / 5)


# ---------------------------------------------------------------------------
# Fixtures: MJX-based
# ---------------------------------------------------------------------------

@pytest.fixture
def chain_config():
    pytest.importorskip("mujoco")
    from feedbax.mechanics.model_builder import ChainConfig
    return ChainConfig(n_joints=3)


@pytest.fixture
def sim_config():
    pytest.importorskip("mujoco")
    from feedbax.mechanics.model_builder import SimConfig
    return SimConfig(dt=0.002, episode_duration=2.0)


@pytest.fixture
def preset(key):
    pytest.importorskip("mujoco")
    from feedbax.mechanics.body import default_3link_bounds, sample_preset
    return sample_preset(default_3link_bounds(), key)


@pytest.fixture
def mjx_plant(preset, chain_config, sim_config):
    pytest.importorskip("mujoco")
    from feedbax.mechanics.mjx_plant import MJXPlant
    return MJXPlant.from_body_preset(
        preset, chain_config, sim_config, clip_states=False
    )


@pytest.fixture
def mjx_backend(sim_config):
    """MJXBackend with frame_skip=5."""
    return MJXBackend(
        control_dt=sim_config.dt * 5,
        sub_dt=sim_config.dt,
        n_substeps=5,
    )


@pytest.fixture
def mjx_backend_single(sim_config):
    """MJXBackend with single substep."""
    return MJXBackend(
        control_dt=sim_config.dt,
        sub_dt=sim_config.dt,
        n_substeps=1,
    )


# ===========================================================================
# DiffraxBackend tests
# ===========================================================================

class TestDiffraxBackendInit:
    def test_default_substeps(self, dt):
        """Single substep when sub_dt == control_dt (default)."""
        backend = DiffraxBackend(control_dt=dt)
        assert backend.n_substeps == 1
        assert backend.sub_dt == dt

    def test_multi_substeps(self, dt):
        """Multiple substeps when sub_dt < control_dt."""
        backend = DiffraxBackend(control_dt=dt, sub_dt=dt / 5)
        assert backend.n_substeps == 5

    def test_protocol(self, diffrax_backend):
        """DiffraxBackend satisfies PhysicsBackend protocol."""
        assert isinstance(diffrax_backend, PhysicsBackend)


class TestDiffraxBackendSubstep:
    def test_constant_force_transition(self, diffrax_backend, pointmass_plant, dt, key):
        """Euler stepping preserves both axes and carries velocity into position."""
        state = diffrax_backend.init_state(pointmass_plant, key=key)
        force = jnp.array([1.0, -2.0])
        initial = state.effector
        state = diffrax_backend.substep(pointmass_plant, state, force)
        assert jnp.allclose(state.effector.pos, initial.pos + initial.vel * dt)
        assert jnp.allclose(state.effector.vel, initial.vel + force * dt)
        state = diffrax_backend.substep(pointmass_plant, state, force)
        assert jnp.allclose(
            state.effector.pos, initial.pos + 2 * initial.vel * dt + force * dt**2
        )
        assert jnp.allclose(state.effector.vel, initial.vel + 2 * force * dt)


class TestDiffraxBackendObserve:
    def test_observe_matches_effector(
        self, diffrax_backend, pointmass_plant, key
    ):
        """observe() returns the same effector as in the state."""
        state = diffrax_backend.init_state(pointmass_plant, key=key)
        observed = diffrax_backend.observe(pointmass_plant, state)
        assert isinstance(observed, CartesianState)
        assert jnp.allclose(observed.pos, state.effector.pos)


class TestDiffraxBackendLegacyEquivalence:
    def test_recurrent_steps_match_legacy(self, pointmass_plant, dt, key):
        """DiffraxBackend single substep matches legacy Mechanics."""
        # Legacy mechanics
        legacy = Mechanics(pointmass_plant, dt, key=key)
        legacy_state = init_state_from_component(legacy)

        # Backend mechanics (single substep = legacy equivalent)
        backend = DiffraxBackend(control_dt=dt)
        backend_mech = Mechanics(pointmass_plant, dt, backend=backend, key=key)
        backend_state = init_state_from_component(backend_mech)

        forces = (jnp.array([0.5, -0.3]), jnp.array([-0.2, 0.8]), jnp.zeros(2))
        for i, force in enumerate(forces):
            step_key = jax.random.fold_in(key, i)
            legacy_out, legacy_state = legacy(
                {"force": force}, legacy_state, key=step_key
            )
            backend_out, backend_state = backend_mech(
                {"force": force}, backend_state, key=step_key
            )
            assert jnp.allclose(
                legacy_out["effector"].pos, backend_out["effector"].pos, atol=1e-6
            )
            assert jnp.allclose(
                legacy_out["effector"].vel, backend_out["effector"].vel, atol=1e-6
            )


# ===========================================================================
# MJXBackend tests
# ===========================================================================

class TestMJXBackendInit:
    def test_n_substeps(self, sim_config):
        """n_substeps computed from timing when not explicit."""
        backend = MJXBackend(control_dt=0.01, sub_dt=0.002)
        assert backend.n_substeps == 5

    @pytest.mark.parametrize("count", [3, 0, 5.0, True])
    def test_explicit_n_substeps_must_agree_with_periods(self, count):
        with pytest.raises(ValueError, match="n_substeps"):
            MJXBackend(control_dt=0.01, sub_dt=0.002, n_substeps=count)

    def test_protocol(self, mjx_backend):
        """MJXBackend satisfies PhysicsBackend protocol."""
        assert isinstance(mjx_backend, PhysicsBackend)


@requires_mujoco
@pytest.mark.optional_mjx
class TestMJXBackendState:
    def test_init_state(self, mjx_backend, mjx_plant, key):
        """init_state produces PhysicsState with aux=None."""
        state = mjx_backend.init_state(mjx_plant, key=key)
        assert isinstance(state, PhysicsState)
        assert state.aux is None
        assert state.effector.pos.shape == (2,)

    def test_init_state_finite(self, mjx_backend, mjx_plant, key):
        """init_state produces finite values."""
        state = mjx_backend.init_state(mjx_plant, key=key)
        assert jnp.all(jnp.isfinite(state.effector.pos))
        assert jnp.all(jnp.isfinite(state.plant.skeleton.qpos))


@requires_mujoco
@pytest.mark.optional_mjx
class TestMJXBackendSubstep:
    def test_substep_returns_physics_state(
        self, mjx_backend_single, mjx_plant, key
    ):
        """Single MJX substep returns PhysicsState."""
        state = mjx_backend_single.init_state(mjx_plant, key=key)
        action = jnp.zeros(mjx_plant.input_size)
        new_state = mjx_backend_single.substep(mjx_plant, state, action)
        assert isinstance(new_state, PhysicsState)
        assert new_state.aux is None

    def test_substep_finite(self, mjx_backend_single, mjx_plant, key):
        """MJX substep produces finite output."""
        state = mjx_backend_single.init_state(mjx_plant, key=key)
        action = jnp.zeros(mjx_plant.input_size)
        new_state = mjx_backend_single.substep(mjx_plant, state, action)
        assert jnp.all(jnp.isfinite(new_state.plant.skeleton.qpos))
        assert jnp.all(jnp.isfinite(new_state.plant.skeleton.qvel))
        assert jnp.all(jnp.isfinite(new_state.effector.pos))

    def test_substep_shapes_preserved(
        self, mjx_backend_single, mjx_plant, key
    ):
        """Shapes are preserved across substeps."""
        state = mjx_backend_single.init_state(mjx_plant, key=key)
        action = jnp.zeros(mjx_plant.input_size)
        new_state = mjx_backend_single.substep(mjx_plant, state, action)
        assert new_state.plant.skeleton.qpos.shape == state.plant.skeleton.qpos.shape
        assert new_state.plant.skeleton.qvel.shape == state.plant.skeleton.qvel.shape


@requires_mujoco
@pytest.mark.optional_mjx
class TestMJXBackendMultiSubstep:
    def test_multi_substep_accumulation(self, mjx_backend, mjx_plant, key):
        """Multiple substeps should accumulate more change than a single one."""
        state = mjx_backend.init_state(mjx_plant, key=key)
        action = jnp.ones(mjx_plant.input_size) * 0.5

        # Run multiple substeps manually
        multi_state = state
        for _ in range(mjx_backend.n_substeps):
            multi_state = mjx_backend.substep(mjx_plant, multi_state, action)

        # Run single substep
        single_state = mjx_backend.substep(mjx_plant, state, action)

        # Multi-substep should have moved further from initial
        multi_delta = jnp.linalg.norm(
            multi_state.plant.skeleton.qpos - state.plant.skeleton.qpos
        )
        single_delta = jnp.linalg.norm(
            single_state.plant.skeleton.qpos - state.plant.skeleton.qpos
        )
        assert multi_delta > single_delta


@requires_mujoco
@pytest.mark.optional_mjx
class TestMJXSkeletonStep:
    """Tests for MJXSkeleton.step() method."""

    def test_step_returns_state(self, key):
        """step() returns an MJXSkeletonState."""
        from feedbax.mechanics.body import default_3link_bounds, sample_preset
        from feedbax.mechanics.model_builder import (
            ChainConfig, SimConfig, build_model, get_body_id, get_site_id, to_mjx,
        )
        from feedbax.mechanics.skeleton.mjx_skeleton import MJXSkeleton, MJXSkeletonState

        preset = sample_preset(default_3link_bounds(), key)
        cc = ChainConfig(n_joints=3)
        sc = SimConfig(dt=0.002, episode_duration=2.0)
        mj = build_model(preset, cc, sc)
        mjx_model = to_mjx(mj)
        skel = MJXSkeleton(
            mjx_model, get_site_id(mj, "effector"),
            get_body_id(mj, "link2"),
        )
        state = skel.init(key=key)
        ctrl = jnp.zeros(skel.input_size)
        new_state = skel.step(state, ctrl)
        assert isinstance(new_state, MJXSkeletonState)
        assert new_state.qpos.shape == state.qpos.shape
        assert new_state.qvel.shape == state.qvel.shape
        assert jnp.all(jnp.isfinite(new_state.qpos))

    def test_step_finite_with_nonzero_ctrl(self, key):
        """step() with nonzero control remains finite."""
        from feedbax.mechanics.body import default_3link_bounds, sample_preset
        from feedbax.mechanics.model_builder import (
            ChainConfig, SimConfig, build_model, get_body_id, get_site_id, to_mjx,
        )
        from feedbax.mechanics.skeleton.mjx_skeleton import MJXSkeleton

        preset = sample_preset(default_3link_bounds(), key)
        cc = ChainConfig(n_joints=3)
        sc = SimConfig(dt=0.002, episode_duration=2.0)
        mj = build_model(preset, cc, sc)
        mjx_model = to_mjx(mj)
        skel = MJXSkeleton(
            mjx_model, get_site_id(mj, "effector"),
            get_body_id(mj, "link2"),
        )
        state = skel.init(key=key)
        ctrl = jnp.ones(skel.input_size) * 0.5
        new_state = skel.step(state, ctrl)
        assert jnp.all(jnp.isfinite(new_state.qpos))
        assert jnp.all(jnp.isfinite(new_state.qvel))


# ===========================================================================
# Mechanics integration tests
# ===========================================================================

class TestMechanicsDiffraxBackend:
    @pytest.mark.parametrize(
        ("control_dt", "sub_dt", "expected_pos"),
        [(0.1, 0.02, 0.004), (np.float32(0.06817), np.float32(0.00401), 0.0021868936)],
    )
    def test_diffrax_backend_multi_substep(
        self, pointmass_plant, control_dt, sub_dt, expected_pos, key,
    ):
        """Unit force on a unit mass measures elapsed time through velocity."""
        backend = DiffraxBackend(control_dt=control_dt, sub_dt=sub_dt)
        mech = Mechanics(pointmass_plant, backend=backend, key=key)
        state = init_state_from_component(mech)
        force = jnp.array([1.0, 0.0])
        initial = state.get(mech.state_index).effector
        out, _ = mech({"force": force}, state, key=key)
        assert mech.dt == float(control_dt)
        assert jnp.allclose(out["effector"].vel - initial.vel, jnp.array([control_dt, 0.0]))
        assert jnp.allclose(out["effector"].pos - initial.pos, jnp.array([expected_pos, 0.0]))

    @pytest.mark.parametrize("sub_dt", [0.03, 0.2, 0.0, -0.02, float("nan"), float("inf")])
    def test_refuses_incoherent_diffrax_substep(self, sub_dt):
        with pytest.raises(ValueError, match="sub_dt"):
            DiffraxBackend(control_dt=0.1, sub_dt=sub_dt)

    def test_refuses_conflicting_mechanics_dt(self, pointmass_plant):
        with pytest.raises(ValueError, match=r"Mechanics\.dt.*backend\.control_dt"):
            Mechanics(pointmass_plant, dt=0.1, backend=DiffraxBackend(0.2))

    @pytest.mark.parametrize("dt", [0.0, -0.1, float("inf"), float("nan")])
    def test_refuses_invalid_control_period(self, pointmass_plant, dt):
        with pytest.raises(ValueError, match=r"Mechanics\.dt"):
            Mechanics(pointmass_plant, dt=dt)
        with pytest.raises(ValueError, match="control_dt"):
            DiffraxBackend(control_dt=dt)


@requires_mujoco
@pytest.mark.optional_mjx
class TestMechanicsMJXBackend:
    def test_refuses_backend_period_that_disagrees_with_model(self, mjx_plant, sim_config):
        sub_dt = 2 * sim_config.dt
        backend = MJXBackend(control_dt=5 * sub_dt, sub_dt=sub_dt)
        with pytest.raises(ValueError, match=r"model\.opt\.timestep.*backend\.sub_dt"):
            Mechanics(mjx_plant, backend=backend)

    def test_mjx_backend_runs(self, mjx_plant, sim_config, key):
        """Mechanics with MJXBackend runs without error."""
        backend = MJXBackend(
            control_dt=sim_config.dt * 5,
            sub_dt=sim_config.dt,
            n_substeps=5,
        )
        mech = Mechanics(mjx_plant, sim_config.dt * 5, backend=backend, key=key)
        state = init_state_from_component(mech)
        force = jnp.zeros(mjx_plant.input_size)
        out, _ = mech({"force": force}, state, key=key)
        assert jnp.all(jnp.isfinite(out["effector"].pos))
        assert isinstance(out["state"], MechanicsState)

    def test_mjx_backend_multi_step(self, mjx_plant, sim_config, key):
        """MJX backend Mechanics runs multiple steps."""
        backend = MJXBackend(
            control_dt=sim_config.dt * 5,
            sub_dt=sim_config.dt,
            n_substeps=5,
        )
        mech = Mechanics(mjx_plant, sim_config.dt * 5, backend=backend, key=key)
        state = init_state_from_component(mech)
        force = jnp.zeros(mjx_plant.input_size)

        for i in range(10):
            step_key = jax.random.fold_in(key, i)
            out, state = mech({"force": force}, state, key=step_key)

        assert jnp.all(jnp.isfinite(out["effector"].pos))


class TestMechanicsVmap:
    def test_vmap_diffrax_backend(self, pointmass_plant, dt, key):
        """Mechanics with DiffraxBackend is vmap-compatible."""
        backend = DiffraxBackend(control_dt=dt)
        mech = Mechanics(pointmass_plant, dt, backend=backend, key=key)
        state = init_state_from_component(mech)

        batch_size = 4
        forces = jnp.arange(batch_size * 2.0).reshape(batch_size, 2) * 0.1
        keys = jax.random.split(key, batch_size)

        def step_one(force, step_key):
            return mech({"force": force}, state, key=step_key)

        batch_out, _ = jax.vmap(step_one)(forces, keys)
        serial = [
            mech({"force": force}, init_state_from_component(mech), key=step_key)[0]["effector"]
            for force, step_key in zip(forces, keys)
        ]
        assert jnp.allclose(batch_out["effector"].pos, jnp.stack([out.pos for out in serial]))
        assert jnp.allclose(batch_out["effector"].vel, jnp.stack([out.vel for out in serial]))


class TestMechanicsRemat:
    def test_remat_substep_diffrax(self, pointmass_plant, dt, key):
        """Gradient checkpointing on DiffraxBackend substeps."""
        backend = DiffraxBackend(control_dt=dt, sub_dt=dt / 3)
        mech = Mechanics(
            pointmass_plant, dt, backend=backend,
            remat_substep=True, key=key,
        )
        state = init_state_from_component(mech)
        force = jnp.array([0.5, -0.3])

        out, _ = mech({"force": force}, state, key=key)
        assert jnp.all(jnp.isfinite(out["effector"].pos))

    def test_remat_no_remat_same_result(self, pointmass_plant, dt, key):
        """Remat and non-remat paths produce identical results."""
        backend = DiffraxBackend(control_dt=dt, sub_dt=dt / 3)

        mech_remat = Mechanics(
            pointmass_plant, dt, backend=backend,
            remat_substep=True, key=key,
        )
        mech_no_remat = Mechanics(
            pointmass_plant, dt, backend=backend,
            remat_substep=False, key=key,
        )

        state_r = init_state_from_component(mech_remat)
        state_n = init_state_from_component(mech_no_remat)

        force = jnp.array([0.5, -0.3])
        step_key = jax.random.PRNGKey(99)

        out_r, _ = mech_remat({"force": force}, state_r, key=step_key)
        out_n, _ = mech_no_remat({"force": force}, state_n, key=step_key)

        assert jnp.allclose(out_r["effector"].pos, out_n["effector"].pos, atol=1e-7)
        assert jnp.allclose(out_r["effector"].vel, out_n["effector"].vel, atol=1e-7)
