#!/usr/bin/env python
"""From the command line, train some models by loading a config and passing to `train_and_save_models`.

Takes a single positional argument: the path to the YAML config.
"""

import argparse
import logging
import os
import sys
from functools import partial
from pathlib import Path

# NOTE: JAX arrays are not directly picklable if they contain device memory references.
# Since we're using pickle to cache states which may contain JAX arrays, we rely on JAX's
# implicit handling of arrays during pickling (it should work for CPU arrays and most
# host-accessible device arrays).
import jax.random as jr
import plotly.io as pio

from feedbax_experiments.analysis.execution import FigDumpManager, run_analysis_module
from feedbax_experiments.config import (
    PATHS,
    PLOTLY_CONFIG,
    PRNG_CONFIG,
    load_batch_config,
    load_config,
)
from feedbax_experiments.misc import deep_merge
from feedbax_experiments.plugins import EXPERIMENT_REGISTRY

logger = logging.getLogger(os.path.basename(__file__))


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run analysis modules on trained models.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--single",
        metavar="MODULE_KEY",
        help="Run single analysis module (e.g., part2.plant_perts)",
    )
    mode.add_argument(
        "--batched",
        metavar="BATCH_NAME",
        help="Run batch analysis (e.g., part2/plant_perts_sweeps)",
    )
    parser.add_argument(
        "--fig-dump-dir",
        type=str,
        default=PATHS.figures_dump,
        help="Directory to dump figures.",
    )
    parser.add_argument(
        "--fig-dump-formats",
        type=str,
        default="html,webp,svg",
        help="Format(s) to dump figures in, comma-separated (e.g., 'html,png,pdf')",
    )
    parser.add_argument(
        "--no-pickle",
        action="store_true",
        help="Do not use pickle for states (don't load existing or save new).",
    )
    parser.add_argument(
        "--retain-past-fig-dumps", action="store_true", help="Do not save states to pickle."
    )
    parser.add_argument(
        "--states-pkl-dir",
        type=str,
        default=None,
        help="Alternative directory for state pickle files (default: PATHS.cache/'states')",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the analysis.")
    parser.add_argument(
        "--plotly-template",
        type=str,
        default=None,
        help="Plotly template to use for figures (default: 'simple_white').",
    )
    parser.add_argument(
        "--show-duplicate-warnings",
        action="store_true",
        help="If set, all occurrences of each distinct warning message are shown.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv or sys.argv[1:])

    pio.templates.default = args.plotly_template or PLOTLY_CONFIG.templates.default

    if args.seed is None:
        key = jr.PRNGKey(PRNG_CONFIG.seed)
    else:
        key = jr.PRNGKey(args.seed)

    # Parse the figure dump formats
    fig_dump_formats = args.fig_dump_formats.split(",")

    # Set states pickle directory
    states_pkl_dir = Path(args.states_pkl_dir) if args.states_pkl_dir else PATHS.cache / "states"

    fig_dump_manager = FigDumpManager(root=Path(args.fig_dump_dir))

    # Clear existing figures if not retaining past dumps
    if not args.retain_past_fig_dumps:
        fig_dump_manager.clear_all_figures()
        logger.info(f"Deleted existing dump figures in {args.fig_dump_dir}")

    run_analysis_func = partial(
        run_analysis_module,
        fig_dump_formats=fig_dump_formats,
        no_pickle=args.no_pickle if args.single else True,  #! For now, don't pickle in batched mode
        states_pkl_dir=states_pkl_dir,
        key=key,
    )

    if args.single:
        module_key = args.single
        module_config = load_config(
            module_key, config_type="analysis", registry=EXPERIMENT_REGISTRY
        )
        fig_dump_dir = fig_dump_manager.prepare_module_dir(module_key, module_config)

        data, common_inputs, all_analyses, all_results, all_figs = run_analysis_func(
            module_key=module_key,
            module_config=module_config,
            fig_dump_dir=fig_dump_dir,
        )
    else:
        batched_spec = load_batch_config(domain="analysis", config_key=args.batched)

        for module_key, run_params_list in batched_spec.items():
            module_config_base = load_config(
                module_key, config_type="analysis", registry=EXPERIMENT_REGISTRY
            )
            fig_dump_manager.prepare_module_dir(module_key, module_config_base)
            for i, run_params in enumerate(run_params_list):
                logger.info(
                    f"Running analysis module {module_key}, run {i + 1} of {len(run_params_list)}"
                )

                module_config = deep_merge(module_config_base, run_params)
                fig_dump_dir = fig_dump_manager.prepare_run_dir(module_key, run_params)

                # Do not keep results in memory across runs
                run_analysis_func(
                    module_key=module_key,
                    module_config=module_config,
                    fig_dump_dir=fig_dump_dir,
                )

    logger.info("All analyses complete. Exiting.")


if __name__ == "__main__":
    main()
