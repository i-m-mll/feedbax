"""Run a local benchmark row against the frozen Feedbax qualification protocol."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from feedbax._atomic import write_bytes_atomic
from feedbax._ode import restore_storage_dtypes
from feedbax.mechanics.backend import PhysicsState
from scripts.biomechanics_fixture import (
    SHOULDER_WRAP_MAX_DISPLACEMENT_RAD,
    START_QPOS,
    build_mechanics,
    matched_physics,
)
from scripts.biomechanics_rnn_benchmark import (
    ReachBenchmark,
    ReachSpec,
    SuccessCriteria,
    make_controller,
    process_peak_rss_bytes,
)


def _identity() -> dict:
    root = Path(__file__).resolve().parent.parent
    return {
        "source": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=5,
        ).strip(),
        "machine": [platform.system(), platform.release(), platform.machine(), platform.processor()],
        "device": str(jax.devices()[0]),
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("jax", "jaxlib", "mujoco-core", "mujoco-mjx")},
    }


def _make_benchmark(args, lock):
    spec = ReachSpec(**lock["reach"])
    if args.phase == "gradient":
        spec = ReachSpec(**{**asdict(spec), "batch_size": 1, "precision": args.precision})
    if args.phase == "timing" and args.batch_one:
        spec = ReachSpec(**{**asdict(spec), "batch_size": 1})
    jax.config.update("jax_enable_x64", spec.precision == "float64")
    mechanics = build_mechanics(
        args.substeps, engine=args.engine, law_profile=args.law, solver=args.solver,
        wrap_shoulder=args.wrap_shoulder,
    )
    model = make_controller(mechanics, spec, seed=args.seed)
    coordinate = "qpos" if args.engine == "mjx" else "angle"
    benchmark = ReachBenchmark(model, spec, f"mechanics.plant.skeleton.{coordinate}", START_QPOS)
    model_identity = getattr(getattr(mechanics, "descriptor", None), "identity", None)
    return benchmark, model, model_identity


@eqx.filter_jit
def _euler_stage_envelope(mechanics, history, commands):
    """Replay each observed tick with its exact held command and stored force."""
    plant, backend = mechanics.plant, mechanics.backend

    def replay_tick(inputs):
        before, command, force = inputs
        loaded = plant.skeleton.update_state_given_effector_force(force, before.plant.skeleton)
        initial = PhysicsState(
            plant=restore_storage_dtypes(replace(before.plant, skeleton=loaded), before.plant),
            effector=replace(before.effector, force=force),
            aux=before.solver,
        )
        start = initial.plant.skeleton.qpos[0]

        def substep(_, carry):
            state, lo, hi = carry
            state = backend.substep(plant, state, command)
            shoulder = state.plant.skeleton.qpos[0]
            return state, jnp.minimum(lo, shoulder), jnp.maximum(hi, shoulder)

        final, lo, hi = jax.lax.fori_loop(0, backend.n_substeps, substep, (initial, start, start))
        return lo, hi, final.plant.skeleton.qpos, final.plant.skeleton.qvel

    def replay_trial(trial, control):
        before = jax.tree.map(lambda value: value[:-1], trial)
        force = trial.plant.skeleton.effector_force[1:]
        return jax.lax.map(replay_tick, (before, control, force))

    lo, hi, qpos, qvel = jax.vmap(replay_trial)(history, commands)
    reference = history.plant.skeleton

    def compare(replayed, observed):
        error = jnp.abs(replayed - observed)
        scale = jnp.maximum(1.0, jnp.maximum(jnp.abs(replayed), jnp.abs(observed)))
        tolerance = 32 * jnp.finfo(replayed.dtype).eps * scale
        return jnp.max(error), jnp.max(tolerance), jnp.all(error <= tolerance)

    terminal = reference.qpos[:, -1, 0]
    return (
        jnp.minimum(jnp.min(lo), jnp.min(terminal)),
        jnp.maximum(jnp.max(hi), jnp.max(terminal)),
        compare(qpos, reference.qpos[:, 1:]),
        compare(qvel, reference.qvel[:, 1:]),
    )


def _snapshot(benchmark, model, batch, *, wrap_shoulder: bool = False) -> dict:
    diff, frozen = eqx.partition(model, benchmark.trainable)
    replay_euler = wrap_shoulder and isinstance(model.mechanics.backend.solver, diffrax.Euler)
    if replay_euler and model.force_lp is not None:
        raise ValueError("Wrapped Euler replay requires the captured efferent signal to drive mechanics")

    @eqx.filter_jit
    def capture(weights, static, data):
        _, (_, states) = benchmark.objective(weights, static, *data)
        skeletal = states.mechanics.plant.skeleton
        qpos = skeletal.qpos if hasattr(skeletal, "qpos") else skeletal.angle
        qvel = skeletal.qvel if hasattr(skeletal, "qvel") else skeletal.d_angle
        summary = (
            states.mechanics.effector.pos[:, -1],
            jnp.min(qpos, axis=(0, 1)),
            jnp.max(qpos, axis=(0, 1)),
            jnp.max(jnp.abs(qvel), axis=(0, 1)),
        )
        replay_inputs = None
        if replay_euler:
            initial = jax.vmap(model.mechanics.state_view)(data.state)
            history = jax.tree.map(
                lambda before, after: jnp.concatenate((before[:, None], after), axis=1),
                initial, states.mechanics,
            )
            replay_inputs = (history, states.efferent.output)
        return summary, replay_inputs

    values, replay_inputs = capture(diff, frozen, batch)
    names = ("endpoints_m", "qpos_min_rad", "qpos_max_rad", "qvel_abs_max_rad_s")
    snapshot = {
        name: np.asarray(value).tolist()
        for name, value in zip(names, values, strict=True)
    }
    if wrap_shoulder:
        start = START_QPOS[0]
        limit = SHOULDER_WRAP_MAX_DISPLACEMENT_RAD
        sampled_lo = snapshot["qpos_min_rad"][0]
        sampled_hi = snapshot["qpos_max_rad"][0]
        lo = min(start, sampled_lo)
        hi = max(start, sampled_hi)
        snapshot["shoulder_wrap"] = {
            "sample_period_s": benchmark.spec.control_dt,
            "observed_min_rad": lo,
            "observed_max_rad": hi,
            "qualified_min_rad": start - limit,
            "qualified_max_rad": start + limit,
            "all_sampled_poses_qualified": start - limit <= sampled_lo <= sampled_hi <= start + limit,
        }
        if replay_inputs is not None:
            history, commands = replay_inputs
            stage_lo, stage_hi, position, velocity = _euler_stage_envelope(
                model.mechanics, history, commands,
            )
            lo, hi = float(stage_lo), float(stage_hi)
            matched = bool(position[2]) and bool(velocity[2])
            snapshot["shoulder_wrap"]["euler_replay"] = {
                "sample_period_s": model.mechanics.backend.sub_dt,
                "samples_per_trial": 1 + commands.shape[1] * model.mechanics.backend.n_substeps,
                "observed_min_rad": lo,
                "observed_max_rad": hi,
                "qpos_max_abs_error_rad": float(position[0]),
                "qpos_max_tolerance_rad": float(position[1]),
                "qvel_max_abs_error_rad_s": float(velocity[0]),
                "qvel_max_tolerance_rad_s": float(velocity[1]),
                "matches_feedback_ticks": matched,
                "all_sampled_poses_qualified": matched and start - limit <= lo <= hi <= start + limit,
            }
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("physics", "rehearsal", "gradient", "timing", "train", "held-out-replay"),
    )
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engine", choices=("mjx", "analytical"), default="mjx")
    parser.add_argument("--law", choices=("rigid", "activation", "compliant"), default="rigid")
    parser.add_argument("--solver", choices=("euler", "tsit5"), default="euler")
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--batch-one", action="store_true")
    parser.add_argument("--wrap-shoulder", action="store_true",
                        help="Use the qualified exterior cylinder shoulder pair on MJX.")
    parser.add_argument("--load-net", type=Path)
    parser.add_argument("--save-net", type=Path)
    args = parser.parse_args()
    if args.wrap_shoulder and (args.engine != "mjx" or args.phase == "physics"):
        parser.error("--wrap-shoulder requires an MJX row without the analytical physics comparator")
    if args.load_net is not None and args.phase in ("physics", "rehearsal", "train"):
        parser.error("--load-net is for evaluation and timing")
    if args.save_net is not None and args.phase != "train":
        parser.error("--save-net requires the train phase")
    if args.phase == "held-out-replay" and (not args.wrap_shoulder or args.load_net is None):
        parser.error("held-out-replay requires --wrap-shoulder and --load-net")
    if args.phase == "train" and args.wrap_shoulder and args.save_net is None:
        parser.error("wrapped training requires --save-net for held-out replay")
    jax.config.update("jax_enable_compilation_cache", False)
    lock = json.loads(args.lock.read_text())
    report = {
        "phase": args.phase,
        "engine": args.engine,
        "law": args.law,
        "solver": args.solver,
        "substeps": args.substeps,
        "seed": args.seed,
        "protocol": lock,
        "wrap_shoulder": args.wrap_shoulder,
    }
    if args.phase == "physics":
        jax.config.update("jax_enable_x64", True)
        report["max_abs_error"] = matched_physics(
            args.substeps, law_profile=args.law, solver=args.solver,
        )
    else:
        benchmark, model, model_identity = _make_benchmark(args, lock)
        report.update(model_identity=model_identity, reach=asdict(benchmark.spec))
        if args.load_net is not None:
            checkpoint = args.load_net.read_bytes()
            net = eqx.tree_deserialise_leaves(BytesIO(checkpoint), model.net)
            model = eqx.tree_at(lambda graph: graph.net, model, net)
            report["loaded_checkpoint_sha256"] = sha256(checkpoint).hexdigest()
        train_seed = lock["seed_streams"]["train"] + args.seed
        if args.phase in ("rehearsal", "gradient", "timing"):
            batch = benchmark.prepare(benchmark.train_bank(train_seed, 0))
        if args.phase == "rehearsal":
            start = perf_counter()
            updated, _, loss = jax.block_until_ready(
                benchmark.update(model, benchmark.optimizer_state(model), batch),
            )
            delta = sum(jnp.sum((after - before)**2) for before, after in zip(
                jax.tree.leaves(eqx.filter(model.net, eqx.is_inexact_array)),
                jax.tree.leaves(eqx.filter(updated.net, eqx.is_inexact_array)),
                strict=True,
            ))
            report.update(loss=float(loss), weight_delta_l2=float(jnp.sqrt(delta)),
                          first_update_s=perf_counter() - start)
        elif args.phase == "gradient":
            report["directions"] = [
                benchmark.directional_check(
                    model, batch, benchmark.direction(model, seed),
                    lock["gradient"]["epsilons"][args.precision],
                )
                for seed in lock["seed_streams"]["directions"]
            ]
            report["trajectory"] = _snapshot(
                benchmark, model, batch, wrap_shoulder=args.wrap_shoulder,
            )
        elif args.phase == "timing":
            report["kernels"] = {
                label: asdict(timing)
                for label, timing in benchmark.time_kernels(
                    model, batch, repeats=lock["timing_repeats"],
                ).items()
            }
        elif args.phase == "held-out-replay":
            held_out = benchmark.prepare(benchmark.held_out_bank(lock["seed_streams"]["held_out"]))
            report["held_out_trajectory"] = _snapshot(
                benchmark, model, held_out, wrap_shoulder=True,
            )
        else:
            held_out = benchmark.prepare(benchmark.held_out_bank(lock["seed_streams"]["held_out"]))
            budget = lock["training"]
            result = benchmark.train(
                train_seed=train_seed, held_out=held_out, criteria=SuccessCriteria(**lock["success"]),
                on_point=lambda point: print(json.dumps(asdict(point)), flush=True), **budget,
            )
            model = result.model
            report.update(
                curve=[asdict(point) for point in result.curve],
                elapsed_s=result.elapsed_s,
                training_s=result.training_s,
                time_to_target_s=result.time_to_target_s,
                updates_to_target=result.updates_to_target,
                updates=result.updates,
                stop_reason=result.stop_reason,
            )
        if args.save_net is not None:
            buffer = BytesIO()
            eqx.tree_serialise_leaves(buffer, model.net)
            checkpoint = buffer.getvalue()
            write_bytes_atomic(args.save_net, checkpoint)
            report["saved_checkpoint_sha256"] = sha256(checkpoint).hexdigest()
    report["peak_process_rss_bytes"] = process_peak_rss_bytes()
    report["identity"] = _identity()
    result = json.dumps(report, indent=2, allow_nan=False) + "\n"
    write_bytes_atomic(args.out, result.encode())


if __name__ == "__main__":
    main()
