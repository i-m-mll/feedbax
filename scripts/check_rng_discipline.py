#!/usr/bin/env python3
"""Narrow static checks for reviewed Feedbax RNG discipline regressions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Check:
    path: str
    bad: str
    message: str


CHECKS = (
    Check(
        "feedbax/tasks/task.py",
        "key_trials, key_dependencies = jr.split(key)",
        "validation_trials must split trial/intervenor/dependency streams explicitly",
    ),
    Check(
        "feedbax/tasks/task.py",
        "trial_specs = self.get_validation_trials(key)",
        "validation trial generation must receive the trial stream",
    ),
    Check(
        "feedbax/training/rl/tasks.py",
        "hold_fk = _sample_reachable_pos(k2, segment_lengths)",
        "HOLD target sampling must not reuse the REACH target key",
    ),
    Check(
        "feedbax/training/rl/tasks.py",
        "hold_raw = jax.random.uniform(\n        k2,",
        "HOLD raw target sampling must not reuse the REACH target key",
    ),
    Check(
        "feedbax/intervene/intervene.py",
        "lambda x: self.noise_func(key, x)",
        "AddNoise must split the call key per PyTree signal leaf",
    ),
    Check(
        "feedbax/intervene/schedule.py",
        "key=key)",
        "intervenor parameter callable leaves must receive split keys",
    ),
    Check(
        "feedbax/analysis/grad.py",
        "key=jax.random.PRNGKey(0)",
        "Hutchinson reducers must not capture a fixed PRNG key as a default",
    ),
    Check(
        "feedbax/analysis/execution.py",
        'states_pickle_path = states_pkl_dir / f"{eval_info.hash}.pkl"',
        "legacy evaluation state cache filenames must include the PRNG key digest",
    ),
)


def main() -> int:
    failures: list[str] = []
    for check in CHECKS:
        text = (REPO_ROOT / check.path).read_text()
        if check.bad in text:
            failures.append(f"{check.path}: {check.message}")

    if failures:
        print("RNG discipline check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
