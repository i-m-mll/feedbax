#!/usr/bin/env python
"""
Unified entrypoint.

Usage:
  run train    [<args>...]     # forwards to feedbax_experiments.scripts.train:main()
  run analysis [<args>...]     # forwards to feedbax_experiments.scripts.run_analysis:main()

All arguments after the subcommand are passed through unchanged to the target
script's own argparse logic, so existing flags continue to work.
"""

import argparse
import importlib
import os
import runpy
import sys
from collections.abc import Callable
from types import ModuleType
from typing import Optional, Sequence

from ._setup import setup

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"


def _call_module_main(mod: ModuleType, argv: Sequence[str]) -> int:
    """
    Try to call mod.main() if present; otherwise run it as __main__
    with sys.argv patched to the sub-argv.
    """
    # If module exposes a `main()`, prefer it (cleaner; avoids top-level side effects).
    main_fn: Optional[Callable] = getattr(mod, "main", None)
    if callable(main_fn):
        # Patch sys.argv so their argparse sees only the subcommand args.
        old_argv = sys.argv
        try:
            sys.argv = [sys.argv[0]] + list(argv)
            _ = main_fn()  # ? Could just pass `list(argv)`, assuming it accepts optional argv
            return int(0)
        finally:
            sys.argv = old_argv

    # Fallback: execute the module as if run via `python -m <mod>`
    old_argv = sys.argv
    try:
        sys.argv = [mod.__name__] + list(argv)
        # run_module returns the globals dict; no exit code, assume 0 if no exception.
        runpy.run_module(mod.__name__, run_name="__main__")
        return 0
    finally:
        sys.argv = old_argv


def _dispatch(cmd: str, sub_argv: Sequence[str]) -> int:
    # Try candidates in order until one imports successfully.
    mod = importlib.import_module("." + cmd, package="feedbax.bin")
    if mod is not None:
        return _call_module_main(mod, sub_argv)
    raise SystemExit(f"Could not locate a module for subcommand '{cmd}'. ")


def parse_args(argv: Sequence[str] | None = None) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(
        prog="run",
        description="Unified entrypoint for training and analysis.",
        add_help=False,  # we'll let subcommands handle their own -h/--help
    )
    parser.add_argument("command", choices=("train", "analysis"))
    # Everything after the subcommand is passed through verbatim
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    ns = parser.parse_args(argv)
    # If user wrote `run train --help`, they probably want train's help.
    if ns.rest and ns.rest[0] == "--":
        ns.rest = ns.rest[1:]
    return ns.command, ns.rest


def main(argv: Sequence[str] | None = None) -> int:
    cmd, sub_argv = parse_args(argv)
    if cmd not in ("analysis", "train"):
        raise SystemExit(f"Unknown command '{cmd}'; expected 'train' or 'analysis'.")
    # Switch to slightly different naming conventions for subpackages (versus CLI commands)
    if cmd == "train":
        domain = "training"
    else:
        domain = "analysis"
    # Set up logging, warnings, and globals config for the chosen domain
    setup(argv=sub_argv, domain=domain, queue_mode="flush")
    # Import and run the chosen script
    return _dispatch(cmd, sub_argv)


if __name__ == "__main__":
    raise SystemExit(main())
