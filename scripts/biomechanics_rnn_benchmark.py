"""Closed-loop reach qualification using the production Feedbax graph and trainer.

A fixture supplies executable mechanics. The benchmark owns the controller,
trial banks, objective, and measurement, and can replay them with any matched
fixture or physics timestep.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from math import isfinite
import resource
import sys
from time import perf_counter
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from equinox.nn import State
from jaxtyping import Array

from feedbax.mechanics.mechanics import Mechanics
from feedbax.models.controllers import recurrent_controller
from feedbax.models.feedback import SimpleFeedback
from feedbax.objectives.loss import CompositeLoss, TargetSpec, TargetStateLoss
from feedbax.runtime.channel import ChannelSpec
from feedbax.runtime.feedback import StateSelector
from feedbax.runtime.graph import RolloutStepContext, RolloutStepHookResult
from feedbax.runtime.mapping import WhereDict
from feedbax.runtime.noise import Normal
from feedbax.tasks.task import TaskTrialSpec, TrialTimeline, prepare_trial
from feedbax.training.trainer import grad_wrap_abstract_loss, training_step_for_abstract_loss


@dataclass(frozen=True)
class ReachSpec:
    control_dt: float = 0.01
    steps: int = 60
    hidden_units: int = 64
    batch_size: int = 16
    precision: str = "float32"
    sensory_delay_steps: int = 1
    motor_delay_steps: int = 0
    position_noise_m: float = 0.001
    velocity_noise_m_s: float = 0.01
    motor_noise: float = 0.005
    min_radius_m: float = 0.06
    max_radius_m: float = 0.09
    pulse_force_n: float = 0.5
    pulse_start: int = 25
    pulse_steps: int = 5
    learning_rate: float = 0.003
    gradient_clip: float = 1.0
    terminal_position_weight: float = 1.0
    running_position_weight: float = 0.4
    terminal_speed_weight: float = 0.025
    command_weight: float = 0.002


@dataclass(frozen=True)
class SuccessCriteria:
    position_p90_m: float
    speed_p90_m_s: float
    command_rms: float

    def met_by(self, metrics: ReachMetrics) -> bool:
        return all(isfinite(float(value)) for value in metrics) and (
            float(metrics.position_p90_m) <= self.position_p90_m
            and float(metrics.speed_p90_m_s) <= self.speed_p90_m_s
            and float(metrics.command_rms) <= self.command_rms
        )


class ReachBank(NamedTuple):
    displacement: Array
    pulse_sign: Array
    keys: Array


class ReachBatch(NamedTuple):
    trials: TaskTrialSpec
    state: State
    keys: Array


class ReachMetrics(NamedTuple):
    loss: Array
    position_p90_m: Array
    speed_p90_m_s: Array
    command_rms: Array


@dataclass(frozen=True)
class KernelTiming:
    compile_s: float
    synchronized_s: tuple[float, ...]
    peak_process_rss_bytes: int


@dataclass(frozen=True)
class LearningPoint:
    updates: int
    training_s: float
    elapsed_s: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class TrainingResult:
    model: SimpleFeedback
    optimizer_state: Any
    curve: tuple[LearningPoint, ...]
    time_to_target_s: float | None
    updates_to_target: int | None
    updates: int
    training_s: float
    elapsed_s: float
    peak_process_rss_bytes: int
    stop_reason: str


def _position(state):
    return state.mechanics.effector.pos


def _velocity(state):
    return state.mechanics.effector.vel


def _command(state):
    return state.efferent.output


def _apply_distal_force(context: RolloutStepContext) -> RolloutStepHookResult | None:
    if context.node_name != "mechanics":
        return None
    mechanics = context.component
    state = mechanics.state_view(context.state)
    effector = replace(state.effector, force=context.step_inputs["disturbance"])
    return RolloutStepHookResult(
        state=context.state.set(mechanics.state_index, replace(state, effector=effector)),
    )


def make_controller(mechanics: Mechanics, spec: ReachSpec, *, seed: int) -> SimpleFeedback:
    """Compose the production controller and feedback graph over a mechanics fixture."""
    if mechanics.dt != spec.control_dt:
        raise ValueError("The benchmark control period must match the mechanics control period")
    if mechanics.plant.input_size != 6:
        raise ValueError("The reaching controller requires six muscle excitations")
    if spec.precision == "float64" and not jax.config.x64_enabled:
        raise ValueError("Enable JAX x64 before building a float64 benchmark")
    net = recurrent_controller(
        6,
        spec.hidden_units,
        out_size=mechanics.plant.input_size,
        out_nonlinearity="sigmoid",
        dtype=jnp.dtype(spec.precision),
        key=jax.random.key(seed),
    )
    model = SimpleFeedback(
        net,
        mechanics,
        feedback_spec=(
            ChannelSpec(
                where=StateSelector(("effector.pos",)),
                delay=spec.sensory_delay_steps,
                noise_func=Normal(std=spec.position_noise_m),
            ),
            ChannelSpec(
                where=StateSelector(("effector.vel",)),
                delay=spec.sensory_delay_steps,
                noise_func=Normal(std=spec.velocity_noise_m_s),
            ),
        ),
        motor_delay=spec.motor_delay_steps,
        motor_noise_func=Normal(std=spec.motor_noise),
    )
    # The trial hook consumes the disturbance; it has no wire to the controller.
    model = eqx.tree_at(lambda graph: graph.input_ports, model, ("input", "disturbance"))
    return model.prepare()


def _loss(spec: ReachSpec) -> CompositeLoss:
    terminal = jnp.asarray([-1])
    dt = jnp.asarray(spec.control_dt, dtype=spec.precision)
    zero = jnp.asarray(0.0, dtype=spec.precision)
    return CompositeLoss(
        {
            "position": TargetStateLoss(
                "Position",
                _position,
                spec=TargetSpec(time_idxs=terminal),
            ),
            "running_position": TargetStateLoss(
                "Running position",
                _position,
                spec=TargetSpec(discount=dt),
            ),
            "speed": TargetStateLoss(
                "Speed",
                _velocity,
                spec=TargetSpec(value=zero, time_idxs=terminal),
            ),
            "command": TargetStateLoss(
                "Command",
                _command,
                spec=TargetSpec(value=zero, discount=dt),
            ),
        },
        weights={
            "position": spec.terminal_position_weight,
            "running_position": spec.running_position_weight,
            "speed": spec.terminal_speed_weight,
            "command": spec.command_weight,
        },
    )


def process_peak_rss_bytes() -> int:
    """Return process high-water RSS, reported in bytes by macOS and KiB by Linux."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _timed_compile(fn, args):
    start = perf_counter()
    compiled = fn.lower(*args).compile()
    return compiled, perf_counter() - start


def _measure(fn, args, compile_s: float, repeats: int) -> KernelTiming:
    jax.block_until_ready(fn(*args))
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        jax.block_until_ready(fn(*args))
        durations.append(perf_counter() - start)
    return KernelTiming(compile_s, tuple(durations), process_peak_rss_bytes())


def _controller_filter(model):
    mask = jax.tree.map(lambda _: False, model)
    return eqx.tree_at(lambda graph: graph.net, mask, jax.tree.map(eqx.is_inexact_array, model.net))


def copy_controller(source: SimpleFeedback, destination: SimpleFeedback) -> SimpleFeedback:
    """Copy controller weights to another fixture without changing its state indices."""
    parameters = eqx.filter(source.net, eqx.is_inexact_array)
    destination_net = eqx.combine(
        parameters, eqx.filter(destination.net, eqx.is_inexact_array, inverse=True)
    )
    return eqx.tree_at(lambda graph: graph.net, destination, destination_net)


class ReachBenchmark:
    """One fixture with shared bank generation, loss, replay, and training entry points."""

    def __init__(
        self,
        model: SimpleFeedback,
        spec: ReachSpec,
        qpos_path: str,
        start_qpos: Sequence[float],
    ):
        self.model = model
        self.spec = spec
        self.qpos_path = qpos_path
        self.start_qpos = jnp.asarray(start_qpos, dtype=spec.precision)
        self.loss = _loss(spec)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(spec.gradient_clip),
            optax.adam(spec.learning_rate),
        )
        self.trainable = _controller_filter(model)
        self.objective = grad_wrap_abstract_loss(self.loss, rollout_step_hook=_apply_distal_force)
        self.forward = eqx.filter_jit(self._forward)
        self.evaluate = eqx.filter_jit(self._evaluate)
        self.update = eqx.filter_jit(self._update)
        self.directional = eqx.filter_jit(self._directional)
        self._record_pullback = eqx.filter_jit(self._pullback)
        self._backward = eqx.filter_jit(lambda pullback, loss: pullback(jnp.ones_like(loss)))
        initial_trial = self._trial(jnp.zeros((2,), dtype=spec.precision), 0)
        self.initial_state = prepare_trial(model, initial_trial).init_state
        self.center = model.mechanics.state_view(self.initial_state).effector.pos
        if self.center.shape != (2,):
            raise ValueError("The reaching benchmark requires planar effector coordinates")

    def train_bank(self, seed: int, update: int) -> ReachBank:
        key = jax.random.fold_in(jax.random.key(seed), update)
        angle_key, radius_key, pulse_key, rollout_key = jax.random.split(key, 4)
        dtype = self.start_qpos.dtype
        shape = (self.spec.batch_size,)
        angles = jax.random.uniform(angle_key, shape, dtype=dtype, maxval=2 * jnp.pi)
        radii = jax.random.uniform(
            radius_key,
            shape,
            dtype=dtype,
            minval=self.spec.min_radius_m,
            maxval=self.spec.max_radius_m,
        )
        pulse_sign = jax.random.randint(pulse_key, shape, -1, 2)
        return ReachBank(
            radii[:, None] * jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=-1),
            pulse_sign,
            jax.random.split(rollout_key, shape[0]),
        )

    def held_out_bank(self, seed: int, directions: int = 8) -> ReachBank:
        """Pair each held-out goal with quiet and disturbed trials, using disjoint keys."""
        dtype = self.start_qpos.dtype
        angles = (jnp.arange(directions, dtype=dtype) + 0.5) * (2 * jnp.pi / directions)
        unit = jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=-1)
        radii = jnp.asarray([self.spec.min_radius_m, self.spec.max_radius_m], dtype=dtype)
        goals = (radii[:, None, None] * unit).reshape(-1, 2)
        disturbed = jnp.where(jnp.arange(len(goals)) % 2 == 0, 1, -1)
        return ReachBank(
            jnp.concatenate((goals, goals)),
            jnp.concatenate((jnp.zeros_like(disturbed), disturbed)),
            jax.random.split(jax.random.key(seed), 2 * len(goals)),
        )

    def _trial(self, target: Array, disturbance: Array | int) -> TaskTrialSpec:
        return TaskTrialSpec(
            inits=WhereDict({self.qpos_path: self.start_qpos}),
            inputs={
                "input": jnp.broadcast_to(target, (self.spec.steps, 2)),
                "disturbance": jnp.broadcast_to(disturbance, (self.spec.steps, 2)).astype(
                    target.dtype
                ),
            },
            targets=WhereDict(
                {
                    _position: {
                        "Position": TargetSpec(value=target),
                        "Running position": TargetSpec(value=target),
                    }
                }
            ),
            timeline=TrialTimeline(n_steps=self.spec.steps, target_start_step=0),
        )

    def prepare(self, bank: ReachBank) -> ReachBatch:
        displacement = bank.displacement.astype(self.start_qpos.dtype)
        amplitude = jnp.linalg.norm(displacement, axis=-1, keepdims=True)
        perpendicular = jnp.stack((-displacement[:, 1], displacement[:, 0]), axis=-1) / jnp.where(
            amplitude > 0, amplitude, 1
        )
        ticks = jnp.arange(self.spec.steps)
        pulse = (ticks >= self.spec.pulse_start) & (
            ticks < self.spec.pulse_start + self.spec.pulse_steps
        )
        forces = (
            self.spec.pulse_force_n
            * bank.pulse_sign[:, None, None]
            * pulse[None, :, None]
            * perpendicular[:, None, :]
        )
        trials = jax.vmap(self._trial)(self.center + displacement, forces)
        states = jax.tree.map(
            lambda value: jnp.broadcast_to(value, (len(displacement), *value.shape)),
            self.initial_state,
        )
        return ReachBatch(trials, states, bank.keys)

    def optimizer_state(self, model: SimpleFeedback):
        return self.optimizer.init(eqx.filter(model, self.trainable))

    def _forward(self, model, batch):
        diff, frozen = eqx.partition(model, self.trainable)
        return self.objective(diff, frozen, *batch)[0]

    def _evaluate(self, model, batch):
        diff, frozen = eqx.partition(model, self.trainable)
        value, (_, states) = self.objective(diff, frozen, *batch)
        goal = batch.trials.inputs["input"][:, -1]
        return ReachMetrics(
            value,
            jnp.quantile(jnp.linalg.norm(_position(states)[:, -1] - goal, axis=-1), 0.9),
            jnp.quantile(jnp.linalg.norm(_velocity(states)[:, -1], axis=-1), 0.9),
            jnp.sqrt(jnp.mean(jnp.clip(_command(states), 0, 1) ** 2)),
        )

    def _update(self, model, opt_state, batch):
        model, opt_state, losses, _ = training_step_for_abstract_loss(
            self.optimizer,
            model,
            opt_state,
            self.loss,
            self.trainable,
            *batch,
            rollout_step_hook=_apply_distal_force,
        )
        return model, opt_state, losses.total

    def _pullback(self, model, batch):
        diff, frozen = eqx.partition(model, self.trainable)
        return jax.vjp(lambda params: self.objective(params, frozen, *batch)[0], diff)

    def _directional(self, model, batch, direction):
        value, pullback = self._pullback(model, batch)
        (gradient,) = self._backward(pullback, value)
        return value, sum(jax.tree.leaves(jax.tree.map(jnp.vdot, gradient, direction)))

    def direction(self, model: SimpleFeedback, seed: int):
        params = eqx.filter(model, self.trainable)
        leaves, structure = jax.tree.flatten(params)
        keys = jax.random.split(jax.random.key(seed), len(leaves))
        vectors = [
            jax.random.normal(key, leaf.shape, leaf.dtype)
            for leaf, key in zip(leaves, keys, strict=True)
        ]
        norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in vectors))
        return jax.tree.unflatten(structure, [x / norm for x in vectors])

    def directional_check(self, model, batch, direction, epsilons: Sequence[float]):
        """Report a central finite-difference sweep using the exact same trial/noise keys."""
        loss, derivative = self.directional(model, batch, direction)
        pairs = []
        for epsilon in epsilons:
            if epsilon <= 0:
                raise ValueError("Finite-difference steps must be positive")
            delta = jax.tree.map(lambda x: epsilon * x, direction)
            plus = self.forward(eqx.apply_updates(model, delta), batch)
            minus = self.forward(eqx.apply_updates(model, jax.tree.map(jnp.negative, delta)), batch)
            pairs.append(
                {
                    "epsilon": epsilon,
                    "loss_plus": float(plus),
                    "loss_minus": float(minus),
                    "central": float((plus - minus) / (2 * epsilon)),
                }
            )
        return {"loss": float(loss), "reverse": float(derivative), "finite_differences": pairs}

    def time_kernels(self, model, batch, *, repeats: int = 5) -> dict[str, KernelTiming]:
        """Time forward, VJP recording, pullback only, and a production optimizer update.

        Peak RSS is cumulative within the process. Profile each comparison row
        in a fresh process when using its memory measurements.
        """
        if repeats < 1:
            raise ValueError("At least one synchronized timing repeat is required")
        results = {}
        args = (model, batch)
        for label, kernel in (("forward", self.forward), ("vjp_record", self._record_pullback)):
            compiled, compile_s = _timed_compile(kernel, args)
            results[label] = _measure(compiled, args, compile_s, repeats)
        loss, pullback = compiled(*args)
        args = (pullback, loss)
        backward, compile_s = _timed_compile(self._backward, args)
        results["backward"] = _measure(backward, args, compile_s, repeats)
        del pullback, loss, args
        args = (model, self.optimizer_state(model), batch)
        update, compile_s = _timed_compile(self.update, args)
        results["update"] = _measure(update, args, compile_s, repeats)
        return results

    def train(
        self,
        *,
        train_seed: int,
        held_out: ReachBatch,
        criteria: SuccessCriteria,
        update_cap: int,
        elapsed_cap_s: float,
        peak_rss_cap_bytes: int,
        evaluate_every: int,
        on_point: Callable[[LearningPoint], None] | None = None,
    ) -> TrainingResult:
        """Train to the first observed held-out success or a declared run cap."""
        if min(update_cap, elapsed_cap_s, peak_rss_cap_bytes, evaluate_every) <= 0:
            raise ValueError("Training caps and evaluation period must be positive")
        model = self.model
        opt_state = self.optimizer_state(model)
        start = perf_counter()
        update_s = 0.0
        curve = []

        def evaluate(step):
            metrics = self.evaluate(model, held_out)
            point = LearningPoint(
                step,
                update_s,
                perf_counter() - start,
                {name: float(value) for name, value in metrics._asdict().items()},
            )
            curve.append(point)
            if on_point is not None:
                on_point(point)
            return (
                point.elapsed_s <= elapsed_cap_s
                and process_peak_rss_bytes() <= peak_rss_cap_bytes
                and criteria.met_by(metrics)
            )

        success = evaluate(0)
        step = 0
        reason = "target" if success else "update_cap"
        while not success and step < update_cap:
            if process_peak_rss_bytes() > peak_rss_cap_bytes:
                reason = "memory_cap"
                break
            if perf_counter() - start >= elapsed_cap_s:
                reason = "elapsed_cap"
                break
            batch = self.prepare(self.train_bank(train_seed, step))
            update_start = perf_counter()
            model, opt_state, loss = jax.block_until_ready(self.update(model, opt_state, batch))
            update_s += perf_counter() - update_start
            step += 1
            if not isfinite(float(loss)):
                reason = "nonfinite_update"
                break
            if process_peak_rss_bytes() > peak_rss_cap_bytes:
                reason = "memory_cap"
                break
            if perf_counter() - start >= elapsed_cap_s:
                reason = "elapsed_cap"
                break
            if step % evaluate_every == 0 or step == update_cap:
                success = evaluate(step)
                if success:
                    reason = "target"
        elapsed = perf_counter() - start
        return TrainingResult(
            model=model,
            optimizer_state=opt_state,
            curve=tuple(curve),
            time_to_target_s=curve[-1].elapsed_s if success else None,
            updates_to_target=step if success else None,
            updates=step,
            training_s=update_s,
            elapsed_s=elapsed,
            peak_process_rss_bytes=process_peak_rss_bytes(),
            stop_reason=reason,
        )
