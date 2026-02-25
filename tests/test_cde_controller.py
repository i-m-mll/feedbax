"""Tests for feedbax.nn_cde (CDENetwork, CDENetworkState).

Bug: 1d2192d
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import equinox as eqx
from equinox.nn import State

from feedbax.nn_cde import CDENetwork, CDENetworkState
from feedbax.graph import init_state_from_component


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

OBS_DIM = 6
HIDDEN_DIM = 16
OUT_SIZE = 2
VF_WIDTH = 32
VF_DEPTH = 2


@pytest.fixture
def key():
    return jr.PRNGKey(42)


@pytest.fixture
def net(key):
    return CDENetwork(
        obs_dim=OBS_DIM,
        hidden_dim=HIDDEN_DIM,
        out_size=OUT_SIZE,
        vf_width=VF_WIDTH,
        vf_depth=VF_DEPTH,
        key=key,
    )


@pytest.fixture
def state(net):
    return init_state_from_component(net)


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_attributes(self, net):
        assert net.obs_dim == OBS_DIM
        assert net.hidden_dim == HIDDEN_DIM
        assert net.out_size == OUT_SIZE

    def test_h0_shape(self, net):
        assert net.h0.shape == (HIDDEN_DIM,)

    def test_readout_shape(self, net):
        assert net.readout.weight.shape == (OUT_SIZE, HIDDEN_DIM)

    def test_vector_field_output_size(self, net):
        """Vector field MLP should output hidden_dim * obs_dim."""
        h_dummy = jnp.zeros(HIDDEN_DIM)
        out = net.vector_field(h_dummy)
        assert out.shape == (HIDDEN_DIM * OBS_DIM,)

    def test_initial_state_shapes(self, net, state):
        net_state: CDENetworkState = state.get(net.state_index)
        assert net_state.input.shape == (OBS_DIM,)
        assert net_state.hidden.shape == (HIDDEN_DIM,)
        assert net_state.output.shape == (OUT_SIZE,)
        assert net_state.obs_prev.shape == (OBS_DIM,)

    def test_ports(self, net):
        assert net.input_ports == ("input", "feedback")
        assert net.output_ports == ("output", "hidden")


# ---------------------------------------------------------------------------
# Single step tests
# ---------------------------------------------------------------------------

class TestSingleStep:
    def test_output_shapes(self, net, state, key):
        obs = jr.normal(key, (OBS_DIM,))
        inputs = {"input": obs}
        outputs, new_state = net(inputs, state, key=key)

        assert outputs["output"].shape == (OUT_SIZE,)
        assert outputs["hidden"].shape == (HIDDEN_DIM,)

    def test_action_bounds(self, net, state, key):
        """Actions should be in [0, 1] due to sigmoid."""
        obs = 10.0 * jr.normal(key, (OBS_DIM,))
        inputs = {"input": obs}
        outputs, _ = net(inputs, state, key=key)
        action = outputs["output"]
        assert jnp.all(action >= 0.0)
        assert jnp.all(action <= 1.0)

    def test_feedback_only_input(self, net, state, key):
        """Network should work with only feedback (no 'input' port)."""
        feedback = jr.normal(key, (OBS_DIM,))
        inputs = {"feedback": feedback}
        outputs, _ = net(inputs, state, key=key)
        assert outputs["output"].shape == (OUT_SIZE,)

    def test_combined_input_feedback(self, key):
        """Network with split input + feedback dims."""
        input_dim = 4
        feedback_dim = 3
        total_obs = input_dim + feedback_dim
        net = CDENetwork(
            obs_dim=total_obs,
            hidden_dim=HIDDEN_DIM,
            out_size=OUT_SIZE,
            key=key,
        )
        state = init_state_from_component(net)
        inputs = {
            "input": jr.normal(key, (input_dim,)),
            "feedback": jr.normal(jr.fold_in(key, 1), (feedback_dim,)),
        }
        outputs, _ = net(inputs, state, key=key)
        assert outputs["output"].shape == (OUT_SIZE,)

    def test_requires_at_least_one_input(self, net, state, key):
        with pytest.raises(ValueError, match="at least one input"):
            net({}, state, key=key)


# ---------------------------------------------------------------------------
# obs_prev tracking
# ---------------------------------------------------------------------------

class TestObsPrevTracking:
    def test_obs_prev_updated(self, net, state, key):
        """obs_prev should equal the observation from the previous step."""
        obs1 = jr.normal(key, (OBS_DIM,))
        _, state1 = net({"input": obs1}, state, key=key)
        net_state1: CDENetworkState = state1.get(net.state_index)
        assert jnp.allclose(net_state1.obs_prev, obs1)

    def test_obs_prev_two_steps(self, net, state, key):
        """After two steps, obs_prev should be the second observation."""
        k1, k2 = jr.split(key)
        obs1 = jr.normal(k1, (OBS_DIM,))
        obs2 = jr.normal(k2, (OBS_DIM,))

        _, state1 = net({"input": obs1}, state, key=k1)
        _, state2 = net({"input": obs2}, state1, key=k2)

        net_state2: CDENetworkState = state2.get(net.state_index)
        assert jnp.allclose(net_state2.obs_prev, obs2)


# ---------------------------------------------------------------------------
# Zero dX test
# ---------------------------------------------------------------------------

class TestZeroDX:
    def test_hidden_unchanged_when_obs_equals_prev(self, net, state, key):
        """If obs == obs_prev, the CDE step should not change hidden state."""
        obs = jr.normal(key, (OBS_DIM,))
        # First step: sets obs_prev = obs
        _, state1 = net({"input": obs}, state, key=key)
        h_after_first: CDENetworkState = state1.get(net.state_index)

        # Second step with same obs: dX = 0, so hidden should not change
        _, state2 = net({"input": obs}, state1, key=key)
        h_after_second: CDENetworkState = state2.get(net.state_index)

        assert jnp.allclose(h_after_first.hidden, h_after_second.hidden, atol=1e-6)


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------

class TestGradients:
    def test_grad_single_step(self, net, state, key):
        """Gradients should flow through a single CDE step."""
        obs = jr.normal(key, (OBS_DIM,))

        def loss_fn(net):
            outputs, _ = net({"input": obs}, state, key=key)
            return jnp.sum(outputs["output"])

        grads = eqx.filter_grad(loss_fn)(net)
        # Check that at least the readout weight has non-zero gradients
        assert not jnp.allclose(grads.readout.weight, 0.0)

    def test_grad_multi_step(self, net, state, key):
        """Gradients should flow through a multi-step unroll."""
        n_steps = 5
        keys = jr.split(key, n_steps)
        observations = jr.normal(jr.fold_in(key, 99), (n_steps, OBS_DIM))

        def loss_fn(net):
            s = state
            total = 0.0
            for i in range(n_steps):
                outputs, s = net({"input": observations[i]}, s, key=keys[i])
                total = total + jnp.sum(outputs["output"])
            return total

        grads = eqx.filter_grad(loss_fn)(net)
        # Vector field should receive gradients through multi-step unroll
        vf_leaves = jax.tree.leaves(grads.vector_field)
        has_nonzero = any(not jnp.allclose(leaf, 0.0) for leaf in vf_leaves)
        assert has_nonzero, "Vector field should have non-zero gradients"


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------

class TestJIT:
    def test_jit_step(self, net, state, key):
        obs = jr.normal(key, (OBS_DIM,))

        @eqx.filter_jit
        def step(net, state, obs, key):
            return net({"input": obs}, state, key=key)

        outputs, new_state = step(net, state, obs, key)
        assert outputs["output"].shape == (OUT_SIZE,)
        assert outputs["hidden"].shape == (HIDDEN_DIM,)


# ---------------------------------------------------------------------------
# vmap batch compatibility
# ---------------------------------------------------------------------------

class TestVmap:
    def test_vmap_over_observations(self, net, state, key):
        """vmap over a batch of observations."""
        batch_size = 8
        obs_batch = jr.normal(key, (batch_size, OBS_DIM))
        keys = jr.split(key, batch_size)

        @eqx.filter_vmap(in_axes=(None, None, 0, 0))
        def batched_step(net, state, obs, key):
            return net({"input": obs}, state, key=key)

        outputs, _ = batched_step(net, state, obs_batch, keys)
        assert outputs["output"].shape == (batch_size, OUT_SIZE)
        assert outputs["hidden"].shape == (batch_size, HIDDEN_DIM)


# ---------------------------------------------------------------------------
# Integration with SimpleFeedback
# ---------------------------------------------------------------------------

class TestSimpleFeedbackIntegration:
    @pytest.mark.xfail(
        reason="Pre-existing bug: Channel.change_input() uses "
               "dataclasses.replace which fails with custom __init__",
        strict=True,
    )
    def test_wired_into_simple_feedback(self, key):
        """Wire CDENetwork into SimpleFeedback directly (bypassing task module).

        This tests the actual integration: CDENetwork plugged into the
        SimpleFeedback sensorimotor graph with mechanics and feedback channels.
        """
        from feedbax.bodies import SimpleFeedback
        from feedbax.mechanics import Mechanics
        from feedbax.mechanics.plant import DirectForceInput
        from feedbax.mechanics.skeleton.pointmass import PointMass
        from feedbax.noise import Normal
        from feedbax.state import CartesianState

        k1, k2, k3 = jr.split(key, 3)

        system = PointMass(mass=1.0, damping=0.0)
        mechanics = Mechanics(DirectForceInput(system), dt=0.05)

        feedback_spec = dict(
            where=lambda state: (
                state.plant.skeleton.pos,
                state.plant.skeleton.vel,
            ),
            delay=0,
            noise_func=Normal(std=0.0),
        )

        # obs_dim = pos(2) + vel(2) + task_input(2) = 6
        # We'll provide a 2D task input (target position)
        task_input_dim = 2
        obs_dim = 4 + task_input_dim  # pos + vel + task input

        net = CDENetwork(
            obs_dim=obs_dim,
            hidden_dim=8,
            out_size=system.input_size,  # 2D force
            vf_width=16,
            vf_depth=1,
            key=k1,
        )

        model = SimpleFeedback(
            net,
            mechanics,
            feedback_spec=feedback_spec,
            key=k2,
        )

        assert isinstance(model.net, CDENetwork)

        # Initialize state
        state = model.init_state(key=k2)

        # Create a sequence of task inputs (target positions over time)
        n_steps = 9
        target_pos = jnp.broadcast_to(
            jnp.array([0.5, 0.5]),
            (n_steps, 2),
        )
        task_inputs = CartesianState(
            pos=target_pos,
            vel=jnp.zeros_like(target_pos),
            force=jnp.zeros_like(target_pos),
        )

        # Run the full sensorimotor loop
        outputs, final_state = model(
            {"input": task_inputs},
            state,
            key=k3,
            n_steps=n_steps,
        )

        # Check that effector output exists with correct time dimension
        assert outputs["effector"] is not None
        # The effector output is a CartesianState; check pos shape
        assert outputs["effector"].pos.shape[0] == n_steps

    @pytest.mark.xfail(
        reason="Pre-existing circular import: feedbax.task -> feedbax.plot "
               "-> feedbax.types -> feedbax.task",
        strict=True,
    )
    def test_full_model_builder(self, key):
        """Build a complete model with cde_controller_nn and run one trial.

        This test requires the task module which has a circular import issue.
        """
        from feedbax.task import SimpleReaches
        from feedbax.loss import CompositeLoss
        from feedbax.xabdeef.models import cde_controller_nn

        k1, k2, k3 = jr.split(key, 3)

        loss_func = CompositeLoss(terms={})
        task = SimpleReaches(
            loss_func=loss_func,
            workspace=jnp.array([[-1.0, -1.0], [1.0, 1.0]]),
            n_steps=10,
        )

        model = cde_controller_nn(
            task, n_steps=10, dt=0.05, hidden_dim=8,
            vf_width=16, vf_depth=1, key=k1,
        )
        assert isinstance(model.net, CDENetwork)
