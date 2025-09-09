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
import jax
import jax.random as jr
import plotly.io as pio
from jax_cookbook.progress import progress_session

from feedbax_experiments.analysis.execution import (
    FigDumpManager,
    check_records_for_analysis,
    run_analysis_module,
)
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
        "--states-pkl-dir",
        type=str,
        default=None,
        help="Alternative directory for state pickle files (default: PATHS.cache/'states')",
    )
    parser.add_argument(
        "--clear-fig-dumps",
        type=str,
        choices=("none", "module", "all"),
        default="module",
        help="Clear existing figure dumps: 'none', 'module' (default), or 'all'.",
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
    parser.add_argument(
        "--memory-warn-gb",
        type=float,
        default=30.0,
        help="Warn if estimated memory usage exceeds this value (in GB).",
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

    fig_dump_manager = FigDumpManager(
        root=Path(args.fig_dump_dir),
        clear_existing=args.clear_fig_dumps if args.clear_fig_dumps != "all" else "module",
    )

    # Clear existing figures if not retaining past dumps
    if args.clear_fig_dumps == "all":
        fig_dump_manager.clear_all_figures()

    run_analysis_fn = partial(
        run_analysis_module,
        fig_dump_formats=fig_dump_formats,
        no_pickle=args.no_pickle if args.single else True,  #! For now, don't pickle in batched mode
        states_pkl_dir=states_pkl_dir,
        memory_warn_gb=args.memory_warn_gb,
        key=key,
    )

    with progress_session():  # Keep the global `rich` progress region alive
        if args.single:
            module_key = args.single
            module_config = load_config(
                module_key, config_type="analysis", registry=EXPERIMENT_REGISTRY
            )
            fig_dump_dir = fig_dump_manager.prepare_module_dir(module_key, module_config)

            data, common_inputs, all_analyses, all_results, all_figs = run_analysis_fn(
                module_key=module_key,
                module_config=module_config,
                fig_dump_dir=fig_dump_dir,
            )
        else:
            batched_spec = load_batch_config(domain="analysis", config_key=args.batched)

            module_configs_base = {
                module_key: load_config(
                    module_key, config_type="analysis", registry=EXPERIMENT_REGISTRY
                )
                for module_key in batched_spec.keys()
            }

            module_configs = {
                module_key: [
                    deep_merge(module_configs_base[module_key], run_params)
                    for run_params in run_params_list
                ]
                for module_key, run_params_list in batched_spec.items()
            }

            # First: Make sure all the required models are actually in the db, so we don't
            # raise an exception later
            for module_key in batched_spec:
                for module_config in module_configs[module_key]:
                    check_records_for_analysis(module_key, module_config)

            for module_key in batched_spec:
                fig_dump_manager.prepare_module_dir(module_key, module_configs_base[module_key])
                for i, module_config in enumerate(module_configs[module_key]):
                    run_params_list = batched_spec[module_key]

                    logger.info(
                        f"Running analysis module {module_key}, run {i + 1} of {len(run_params_list)}"
                    )

                    fig_dump_dir = fig_dump_manager.prepare_run_dir(module_key, run_params_list[i])

                    # Do not keep results in memory across runs
                    run_analysis_fn(
                        module_key=module_key,
                        module_config=module_config,
                        fig_dump_dir=fig_dump_dir,
                    )

                    jax.clear_caches()

    logger.info("All analyses complete. Exiting.")


if __name__ == "__main__":
    main()
