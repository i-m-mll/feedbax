"""Tests for control system blocks and signal processing components."""

import jax
import jax.numpy as jnp

from feedbax.runtime.graph import init_state_from_component

# Control components
from feedbax.control.continuous import Derivative, Integrator, StateSpace, TransferFunction
from feedbax.control.discrete import IntegratorDiscrete, UnitDelay, ZeroOrderHold
from feedbax.control.pid import PID, PIDDiscrete

# Signal components in components.py
from feedbax.runtime.components import DeadZone, Demux, Mux, RateLimiter, Switch

# Filter components
from feedbax.runtime.filters import BandPassFilter, HighPassFilter


KEY = jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# Helper: run a component for N steps with a given input sequence
# ---------------------------------------------------------------------------

def _run_steps(component, inputs_seq, state=None):
    """Run component for len(inputs_seq) steps, return list of output dicts."""
    if state is None:
        state = init_state_from_component(component)
    step = jax.jit(lambda inputs, state: component(inputs, state, key=KEY))
    outputs = []
    for step_inputs in inputs_seq:
        out, state = step(step_inputs, state)
        outputs.append(out)
    return outputs, state


# ===========================================================================
# Integrator
# ===========================================================================

class TestIntegrator:
    def test_multidimensional_trajectory(self):
        comp = Integrator(dt=0.02, n_dims=3, initial_value=0.5)
        inputs = jnp.array([[1.0, -2.0, 0.0], [3.0, 1.0, -4.0], [-2.0, 0.0, 5.0]])
        outputs, _ = _run_steps(comp, [{"input": value} for value in inputs])
        expected = jnp.array([[0.52, 0.46, 0.5], [0.58, 0.48, 0.42], [0.54, 0.48, 0.52]])
        assert jnp.allclose(jnp.stack([out["output"] for out in outputs]), expected)


# ===========================================================================
# Derivative
# ===========================================================================

class TestDerivative:
    def test_derivative_of_ramp_is_constant(self):
        """Derivative of a ramp (linearly increasing input) should be constant."""
        dt = 0.01
        comp = Derivative(dt=dt, n_dims=1)
        state = init_state_from_component(comp)
        step = jax.jit(lambda inputs, state: comp(inputs, state, key=KEY))

        # Ramp input: 0, dt, 2*dt, ...
        n_steps = 50
        derivatives = []
        for i in range(n_steps):
            val = jnp.array([i * dt])
            out, state = step({"input": val}, state)
            derivatives.append(out["output"])

        # After the first step, derivative should be ~1.0 (slope of ramp)
        for d in derivatives[1:]:
            assert jnp.allclose(d, jnp.array([1.0]), atol=1e-5)


# ===========================================================================
# StateSpace
# ===========================================================================

class TestStateSpace:
    def test_first_order_system(self):
        """Simple first-order system: x' = -x + u, y = x."""
        dt = 0.01
        A = jnp.array([[-1.0]])
        B = jnp.array([[1.0]])
        C = jnp.array([[1.0]])
        D = jnp.array([[0.0]])
        comp = StateSpace(A, B, C, D, dt=dt)
        state = init_state_from_component(comp)

        # Step response: should converge toward 1.0 for u=1
        n_steps = 1000
        inputs_seq = [{"input": jnp.array([1.0])} for _ in range(n_steps)]
        outputs, _ = _run_steps(comp, inputs_seq, state)

        # After many steps, should be close to steady state y=1.0
        final = outputs[-1]["output"]
        assert jnp.allclose(final, jnp.array([1.0]), atol=0.05)

    def test_impulse_response(self):
        """Impulse into a first-order system produces decaying response."""
        dt = 0.01
        A = jnp.array([[-1.0]])
        B = jnp.array([[1.0]])
        C = jnp.array([[1.0]])
        D = jnp.array([[0.0]])
        comp = StateSpace(A, B, C, D, dt=dt)
        state = init_state_from_component(comp)

        # Impulse at step 0
        out0, state = comp({"input": jnp.array([1.0])}, state, key=KEY)

        # Subsequent zero inputs: response should decay
        prev_val = jnp.abs(out0["output"][0])
        for _ in range(100):
            out, state = comp({"input": jnp.array([0.0])}, state, key=KEY)
            curr_val = jnp.abs(out["output"][0])
            assert curr_val <= prev_val + 1e-6  # monotonically decaying
            prev_val = curr_val


# ===========================================================================
# TransferFunction
# ===========================================================================

class TestTransferFunction:
    def test_first_order_lowpass(self):
        """H(s) = 1/(s+1) should behave like a first-order system."""
        dt = 0.01
        comp = TransferFunction(num=[1.0], den=[1.0, 1.0], dt=dt)
        state = init_state_from_component(comp)

        # Step response
        n_steps = 1000
        inputs_seq = [{"input": jnp.array([1.0])} for _ in range(n_steps)]
        outputs, _ = _run_steps(comp, inputs_seq, state)

        # Should converge to ~1.0
        final = outputs[-1]["output"]
        assert jnp.allclose(final, jnp.array([1.0]), atol=0.05)

    def test_static_gain(self):
        """H(s) = 5 (pure gain) should immediately output 5*u."""
        comp = TransferFunction(num=[5.0], den=[1.0], dt=0.01)
        state = init_state_from_component(comp)
        out, _ = comp({"input": jnp.array([2.0])}, state, key=KEY)
        assert jnp.allclose(out["output"], jnp.array([10.0]), atol=1e-5)


# ===========================================================================
# PID
# ===========================================================================

class TestPID:
    def test_proportional_only(self):
        """P-only controller: output = Kp * error."""
        comp = PID(Kp=2.0, Ki=0.0, Kd=0.0, dt=0.01, n_dims=1)
        state = init_state_from_component(comp)
        out, _ = comp({"error": jnp.array([3.0])}, state, key=KEY)
        assert jnp.allclose(out["output"], jnp.array([6.0]), atol=1e-5)

    def test_integral_accumulation(self):
        """PI controller: integral accumulates over steps."""
        dt = 0.01
        comp = PID(Kp=0.0, Ki=1.0, Kd=0.0, dt=dt, n_dims=1)
        state = init_state_from_component(comp)

        n_steps = 10
        inputs_seq = [{"error": jnp.array([1.0])} for _ in range(n_steps)]
        outputs, _ = _run_steps(comp, inputs_seq, state)

        # After 10 steps: integral = 10 * 1.0 * dt = 0.1, Ki * integral = 0.1
        expected = n_steps * dt
        assert jnp.allclose(outputs[-1]["output"], jnp.array([expected]), atol=1e-5)

    def test_anti_windup(self):
        """Integral should saturate at integral_limit."""
        dt = 0.01
        comp = PID(Kp=0.0, Ki=1.0, Kd=0.0, dt=dt, integral_limit=0.05, n_dims=1)
        state = init_state_from_component(comp)

        # Run enough steps to exceed the limit
        inputs_seq = [{"error": jnp.array([1.0])} for _ in range(100)]
        outputs, _ = _run_steps(comp, inputs_seq, state)

        # Output should be clamped at Ki * integral_limit = 1.0 * 0.05
        assert jnp.allclose(outputs[-1]["output"], jnp.array([0.05]), atol=1e-5)

    def test_tracks_step_setpoint(self):
        """PI controller should track a step setpoint over time."""
        dt = 0.01
        comp = PID(Kp=1.0, Ki=5.0, Kd=0.0, dt=dt, n_dims=1)
        state = init_state_from_component(comp)

        # Simulate closed-loop tracking (simplified: error = setpoint - pid_output)
        setpoint = 1.0
        output_val = jnp.array([0.0])
        for _ in range(500):
            error = jnp.array([setpoint]) - output_val
            out, state = comp({"error": error}, state, key=KEY)
            # Simple plant: output follows PID output directly (gain=1 integrator)
            output_val = output_val + out["output"] * dt

        # Should be close to setpoint
        assert jnp.abs(output_val[0] - setpoint) < 0.1


# ===========================================================================
# PIDDiscrete
# ===========================================================================

class TestPIDDiscrete:
    def test_proportional_step(self):
        """Step error with P-only should give proportional output."""
        comp = PIDDiscrete(Kp=2.0, Ki=0.0, Kd=0.0, dt=0.01, n_dims=1)
        state = init_state_from_component(comp)
        step = jax.jit(lambda inputs, state: comp(inputs, state, key=KEY))

        out, state = step({"error": jnp.array([1.0])}, state)
        # Velocity form: dp = Kp*(e - 0) = 2.0, u = 0 + 2.0 = 2.0
        assert jnp.allclose(out["output"], jnp.array([2.0]), atol=1e-5)


# ===========================================================================
# Mux / Demux
# ===========================================================================

class TestMuxDemux:
    def test_mux_concatenates(self):
        """Mux should concatenate inputs into one vector."""
        comp = Mux(n_inputs=3)
        state = init_state_from_component(comp)
        step = jax.jit(lambda inputs, state: comp(inputs, state, key=KEY))
        inputs = {
            "in_0": jnp.array([1.0, 2.0]),
            "in_1": jnp.array([3.0]),
            "in_2": jnp.array([4.0, 5.0]),
        }
        out, _ = step(inputs, state)
        expected = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert jnp.allclose(out["output"], expected)

    def test_demux_splits(self):
        """Demux should split vector into chunks of specified sizes."""
        comp = Demux(sizes=[2, 1, 2])
        state = init_state_from_component(comp)
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out, _ = comp({"input": x}, state, key=KEY)
        assert jnp.allclose(out["out_0"], jnp.array([1.0, 2.0]))
        assert jnp.allclose(out["out_1"], jnp.array([3.0]))
        assert jnp.allclose(out["out_2"], jnp.array([4.0, 5.0]))


# ===========================================================================
# Switch
# ===========================================================================

class TestSwitch:
    def test_elementwise(self):
        """Switch operates element-wise on arrays."""
        comp = Switch(threshold=0.0)
        state = init_state_from_component(comp)
        step = jax.jit(lambda inputs, state: comp(inputs, state, key=KEY))
        out, _ = step(
            {
                "condition": jnp.array([1.0, -1.0, 0.0]),
                "true_input": jnp.array([1.0, 1.0, 1.0]),
                "false_input": jnp.array([0.0, 0.0, 0.0]),
            },
            state,
        )
        expected = jnp.array([1.0, 0.0, 0.0])
        assert jnp.allclose(out["output"], expected)


# ===========================================================================
# DeadZone
# ===========================================================================

class TestDeadZone:
    def test_inside_dead_band(self):
        """Inputs within dead band produce zero output."""
        comp = DeadZone(threshold=0.5)
        state = init_state_from_component(comp)
        step = jax.jit(lambda inputs, state: comp(inputs, state, key=KEY))
        out, _ = step({"input": jnp.array([0.3, -0.2, 0.0])}, state)
        assert jnp.allclose(out["output"], jnp.zeros(3), atol=1e-6)

    def test_outside_dead_band(self):
        """Inputs outside dead band are shifted toward zero."""
        comp = DeadZone(threshold=0.5)
        state = init_state_from_component(comp)
        out, _ = comp({"input": jnp.array([1.0, -1.0])}, state, key=KEY)
        expected = jnp.array([0.5, -0.5])
        assert jnp.allclose(out["output"], expected, atol=1e-6)


# ===========================================================================
# RateLimiter
# ===========================================================================

class TestRateLimiter:
    def test_limits_rate(self):
        """Large step change is rate-limited across multiple steps."""
        dt = 0.01
        max_rate = 10.0  # max 0.1 per step
        comp = RateLimiter(max_rate=max_rate, dt=dt, n_dims=1)
        state = init_state_from_component(comp)
        step = jax.jit(lambda inputs, state: comp(inputs, state, key=KEY))

        # Apply a large step from 0 to 5
        out1, state = step({"input": jnp.array([5.0])}, state)
        # Should only move max_rate * dt = 0.1
        assert jnp.allclose(out1["output"], jnp.array([0.1]), atol=1e-6)

        # Second step: another 0.1
        out2, state = step({"input": jnp.array([5.0])}, state)
        assert jnp.allclose(out2["output"], jnp.array([0.2]), atol=1e-6)

    def test_no_limit_when_slow(self):
        """Small changes within rate limit pass through unchanged."""
        comp = RateLimiter(max_rate=100.0, dt=0.01, n_dims=1)
        state = init_state_from_component(comp)
        out, _ = comp({"input": jnp.array([0.5])}, state, key=KEY)
        assert jnp.allclose(out["output"], jnp.array([0.5]), atol=1e-6)


# ===========================================================================
# UnitDelay
# ===========================================================================

class TestUnitDelay:
    def test_output_is_previous_input(self):
        """Output at step k should be the input from step k-1."""
        comp = UnitDelay(n_dims=2, initial_value=0.0)
        state = init_state_from_component(comp)
        step = jax.jit(lambda inputs, state: comp(inputs, state, key=KEY))

        # Step 0: output should be initial value (0)
        out0, state = step({"input": jnp.array([1.0, 2.0])}, state)
        assert jnp.allclose(out0["output"], jnp.zeros(2))

        # Step 1: output should be input from step 0
        out1, state = step({"input": jnp.array([3.0, 4.0])}, state)
        assert jnp.allclose(out1["output"], jnp.array([1.0, 2.0]))

        # Step 2: output should be input from step 1
        out2, state = step({"input": jnp.array([5.0, 6.0])}, state)
        assert jnp.allclose(out2["output"], jnp.array([3.0, 4.0]))


# ===========================================================================
# ZeroOrderHold
# ===========================================================================

class TestZeroOrderHold:
    def test_hold_behavior(self):
        """Samples input on first step, holds for hold_steps."""
        comp = ZeroOrderHold(hold_steps=3, n_dims=1)
        state = init_state_from_component(comp)
        step = jax.jit(lambda inputs, state: comp(inputs, state, key=KEY))

        # Step 0: counter=0, should sample
        out0, state = step({"input": jnp.array([10.0])}, state)
        assert jnp.allclose(out0["output"], jnp.array([10.0]))

        # Steps 1, 2: should hold
        out1, state = step({"input": jnp.array([20.0])}, state)
        assert jnp.allclose(out1["output"], jnp.array([10.0]))

        out2, state = step({"input": jnp.array([30.0])}, state)
        assert jnp.allclose(out2["output"], jnp.array([10.0]))

        # Step 3: counter wraps, should sample again
        out3, state = step({"input": jnp.array([40.0])}, state)
        assert jnp.allclose(out3["output"], jnp.array([40.0]))


# ===========================================================================
# IntegratorDiscrete
# ===========================================================================

class TestIntegratorDiscrete:
    def test_accumulation(self):
        """Pure accumulator with dt=1 should sum inputs."""
        comp = IntegratorDiscrete(dt=1.0, n_dims=1)
        state = init_state_from_component(comp)

        inputs_seq = [{"input": jnp.array([float(i)])} for i in range(1, 6)]
        outputs, _ = _run_steps(comp, inputs_seq, state)

        # After inputs 1,2,3,4,5: sum = 15
        assert jnp.allclose(outputs[-1]["output"], jnp.array([15.0]), atol=1e-5)


# ===========================================================================
# HighPassFilter
# ===========================================================================

class TestHighPassFilter:
    def test_blocks_dc(self):
        """A high-pass filter should attenuate a constant (DC) input over time."""
        comp = HighPassFilter(tau=0.1, dt=0.01, n_dims=1)
        state = init_state_from_component(comp)
        step = jax.jit(lambda inputs, state: comp(inputs, state, key=KEY))

        # Apply constant input for many steps
        for _ in range(500):
            out, state = step({"input": jnp.array([1.0])}, state)

        # Output should be close to zero (DC rejected)
        assert jnp.abs(out["output"][0]) < 0.1

    def test_passes_transient(self):
        """A step change should produce a nonzero transient."""
        comp = HighPassFilter(tau=0.1, dt=0.01, n_dims=1)
        state = init_state_from_component(comp)

        # First step from zero: should have significant output
        out, _ = comp({"input": jnp.array([1.0])}, state, key=KEY)
        assert jnp.abs(out["output"][0]) > 0.5


# ===========================================================================
# BandPassFilter
# ===========================================================================

class TestBandPassFilter:
    def test_step_transient_decays(self):
        """A band-pass filter passes the step transient and rejects its DC tail."""
        comp = BandPassFilter(tau_low=0.1, tau_high=0.01, dt=0.01, n_dims=1)
        state = init_state_from_component(comp)
        step = jax.jit(lambda inputs, state: comp(inputs, state, key=KEY))
        out, _ = step({"input": jnp.array([1.0])}, state)
        assert 0.0 < out["output"][0] < 1.0
        for _ in range(200):
            out, state = step({"input": jnp.array([1.0])}, state)
        assert jnp.abs(out["output"][0]) < 1e-5


# ===========================================================================
# Vmap (batch) test for stateful components
# ===========================================================================

class TestVmapBatch:
    def test_integrator_vmap(self):
        """Integrator should work with vmap over a batch of inputs."""
        comp = Integrator(dt=0.01, n_dims=2)
        state = init_state_from_component(comp)

        @jax.jit
        @jax.vmap
        def batched_step(inputs):
            return comp({"input": inputs}, state, key=KEY)

        batch_inputs = jnp.arange(8.0).reshape(4, 2)
        out, _ = batched_step(batch_inputs)
        assert jnp.allclose(out["output"], batch_inputs * 0.01)

    def test_pid_vmap(self):
        """PID should work with vmap over a batch of errors."""
        comp = PID(Kp=1.0, Ki=0.1, Kd=0.01, dt=0.01, n_dims=3)
        state = init_state_from_component(comp)

        @jax.jit
        @jax.vmap
        def batched_step(errors):
            return comp({"error": errors}, state, key=KEY)

        batch_errors = jnp.arange(24.0).reshape(8, 3) / 10
        out, _ = batched_step(batch_errors)
        serial = jnp.stack([
            comp({"error": error}, init_state_from_component(comp), key=KEY)[0]["output"]
            for error in batch_errors
        ])
        assert jnp.allclose(out["output"], serial)

    def test_unit_delay_vmap(self):
        """Each batched delay carries its own previous input."""
        comp = UnitDelay(n_dims=2)
        state = init_state_from_component(comp)
        batch_inputs = jnp.arange(8.0).reshape(4, 2)

        @jax.jit
        @jax.vmap
        def first_step(inputs):
            return comp({"input": inputs}, state, key=KEY)

        out, states = first_step(batch_inputs)
        assert jnp.allclose(out["output"], jnp.zeros_like(batch_inputs))
        next_out, _ = jax.jit(jax.vmap(
            lambda inputs, state: comp({"input": inputs}, state, key=KEY)
        ))(-batch_inputs, states)
        assert jnp.allclose(next_out["output"], batch_inputs)
