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
        "feedbax/builtins/runtime/signal_transforms.py",
        "lambda x: self.noise_func(key, x)",
        "AddNoise must split the call key per PyTree signal leaf",
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
