"""Run a local benchmark row against the frozen Feedbax qualification protocol."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from scripts.biomechanics_fixture import START_QPOS, build_mechanics, matched_physics
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
    )
    model = make_controller(mechanics, spec, seed=args.seed)
    coordinate = "qpos" if args.engine == "mjx" else "angle"
    benchmark = ReachBenchmark(model, spec, f"mechanics.plant.skeleton.{coordinate}", START_QPOS)
    model_identity = getattr(getattr(mechanics, "descriptor", None), "identity", None)
    return benchmark, model, model_identity


def _snapshot(benchmark, model, batch) -> dict:
    diff, frozen = eqx.partition(model, benchmark.trainable)

    @eqx.filter_jit
    def capture(weights, static, data):
        _, (_, states) = benchmark.objective(weights, static, *data)
        skeletal = states.mechanics.plant.skeleton
        qpos = skeletal.qpos if hasattr(skeletal, "qpos") else skeletal.angle
        qvel = skeletal.qvel if hasattr(skeletal, "qvel") else skeletal.d_angle
        return (
            states.mechanics.effector.pos[:, -1],
            jnp.min(qpos, axis=(0, 1)),
            jnp.max(qpos, axis=(0, 1)),
            jnp.max(jnp.abs(qvel), axis=(0, 1)),
        )

    names = ("endpoints_m", "qpos_min_rad", "qpos_max_rad", "qvel_abs_max_rad_s")
    return {
        name: np.asarray(value).tolist()
        for name, value in zip(names, capture(diff, frozen, batch), strict=True)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("physics", "rehearsal", "gradient", "timing", "train"))
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engine", choices=("mjx", "analytical"), default="mjx")
    parser.add_argument("--law", choices=("rigid", "activation"), default="rigid")
    parser.add_argument("--solver", choices=("euler", "tsit5"), default="euler")
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--batch-one", action="store_true")
    parser.add_argument("--load-net", type=Path)
    parser.add_argument("--save-net", type=Path)
    args = parser.parse_args()
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
            net = eqx.tree_deserialise_leaves(args.load_net, model.net)
            model = eqx.tree_at(lambda graph: graph.net, model, net)
        train_seed = lock["seed_streams"]["train"] + args.seed
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
            report["trajectory"] = _snapshot(benchmark, model, batch)
        elif args.phase == "timing":
            report["kernels"] = {
                label: asdict(timing)
                for label, timing in benchmark.time_kernels(
                    model, batch, repeats=lock["timing_repeats"],
                ).items()
            }
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
            eqx.tree_serialise_leaves(args.save_net, model.net)
    report["peak_process_rss_bytes"] = process_peak_rss_bytes()
    report["identity"] = _identity()
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
