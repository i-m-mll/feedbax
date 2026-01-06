#!/usr/bin/env python
"""From the command line, train some models according to some configuration.

Load the config and pass it to `train_and_save_models`.

Takes a single positional argument: the path to the YAML config.
"""

import argparse
import logging
import os
from functools import partial
from pathlib import Path

import equinox as eqx
import feedbax
import jax
import jax.random as jr
import jax.tree as jt
import jax_cookbook.tree as jtree
import optax
from jax_cookbook import is_type
from jax_cookbook.progress import piter

import feedbax_experiments
from feedbax._experiments.config import (
    PATHS,
    PRNG_CONFIG,
    load_batch_config,
    load_config,
)
from feedbax._experiments.database import ModelRecord, db_session
from feedbax._experiments.misc import deep_merge, discard, log_version_info
from feedbax._experiments.plugins import EXPERIMENT_REGISTRY
from feedbax._experiments.training.post_training import process_model_post_training
from feedbax._experiments.training.train import (
    prepare_to_train,
    train_and_save,
    train_and_save_from_config,
)
from feedbax._experiments.tree_utils import filter_varying_leaves
from feedbax._experiments.types import TreeNamespace

logger = logging.getLogger(os.path.basename(__file__))


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train models based on configuration.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--single", metavar="MODULE_KEY", help="Train single module (e.g., part2)")
    mode.add_argument("--batched", metavar="BATCHED_KEY", help="Train batch modules (e.g., part2)")
    parser.add_argument(
        "--untrained-only",
        action="store_false",
        help="Only train models which appear not to have been trained yet.",
    )
    parser.add_argument(
        "--postprocess", action="store_false", help="Postprocess each model after training."
    )
    parser.add_argument(
        "--force-postprocess",
        action="store_true",
        help="Force re-postprocessing of models even if already postprocessed.",
    )
    parser.add_argument(
        "--n-std-exclude",
        type=int,
        default=2,
        help="In postprocessing, exclude model replicates with n_std greater than this value.",
    )
    parser.add_argument("--no-figures", action="store_true", help="Save figures in postprocessing.")
    parser.add_argument(
        "--fig-dump-dir",
        type=str,
        default=Path(PATHS.figures_dump) / "train",
        help="Directory to dump post-training figures (default: none).",
    )
    parser.add_argument(
        "--fig-dump-formats",
        type=str,
        default="html,png,svg",
        help="Format(s) to dump figures in, comma-separated (e.g., 'html,png,pdf')",
    )
    parser.add_argument(
        "--clear-fig-dumps",
        action="store_true",
        help="Clear figure dump directory before training",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the training.")
    parser.add_argument(
        "--show-duplicate-warnings",
        action="store_true",
        help="If set, all occurrences of each distinct warning message are shown.",
    )

    return parser


def main():
    args = build_arg_parser().parse_args()

    if args.seed is None:
        key = jr.PRNGKey(PRNG_CONFIG.seed)
    else:
        key = jr.PRNGKey(args.seed)

    # Parse figure dump formats and path
    fig_dump_formats = args.fig_dump_formats.split(",")
    fig_dump_path = Path(args.fig_dump_dir) if args.fig_dump_dir else None

    # Clear figure dump directory if requested
    if fig_dump_path and args.clear_fig_dumps:
        if fig_dump_path.exists():
            import shutil

            # Count files before clearing for logging
            n_files = sum(1 for _ in fig_dump_path.rglob("*") if _.is_file())
            shutil.rmtree(fig_dump_path)
            logger.info(f"Cleared figure dump directory: {fig_dump_path} ({n_files} files removed)")
        else:
            logger.info(f"Figure dump directory does not exist, nothing to clear: {fig_dump_path}")

    version_info = log_version_info(
        jax,
        eqx,
        optax,
        git_modules=(feedbax, feedbax_experiments),
    )

    if args.single:
        module_key = args.single
        module_config = load_config(
            module_key, config_type="training", registry=EXPERIMENT_REGISTRY
        )

        trained_models, train_histories, model_records = train_and_save_from_config(
            expt_key=module_key,
            config=module_config,
            untrained_only=args.untrained_only,
            postprocess=args.postprocess,
            force_postprocess=args.force_postprocess,
            n_std_exclude=args.n_std_exclude,
            save_figures=not args.no_figures,
            fig_dump_path=fig_dump_path,
            fig_dump_formats=fig_dump_formats,
            fig_dump_params=None,  # Single mode: no varying params to extract
            version_info=version_info,
            key=key,
        )

    else:
        batched_spec = load_batch_config(domain="training", config_key=args.batched)

        module_configs_base = {
            module_key: load_config(
                module_key, config_type="training", registry=EXPERIMENT_REGISTRY
            )
            for module_key in batched_spec.keys()
        }

        module_configs_list = [
            (module_key, deep_merge(module_configs_base[module_key], run_params))
            for module_key, run_params_list in batched_spec.items()
            for run_params in run_params_list
        ]

        # First: Check which of the requested models actually need to be trained or post-processed
        training_status, all_hps, model_records = jtree.unzip(
            [
                prepare_to_train(*pair)
                for pair in piter(
                    module_configs_list,
                    description="Preparing to train",
                )
            ]
        )

        # Flatten everything to lists for easier manipulation
        all_hps_flat = jt.leaves(all_hps, is_leaf=is_type(TreeNamespace))
        training_status_flat = jt.leaves(training_status)
        model_records_flat = jt.leaves(model_records, is_leaf=is_type(ModelRecord))

        # Compute varying params from ALL hps (before any filtering)
        all_hps_flat_diff = filter_varying_leaves(all_hps_flat)

        # Do this here rather than using `train_and_save_from_config`, so logs are summary
        # rather than repetitive
        if args.untrained_only:
            n_total = len(jt.leaves(all_hps, is_leaf=is_type(TreeNamespace)))
            all_hps_to_train, all_hps_already_trained = eqx.partition(
                all_hps,
                jt.map(lambda status: status == "untrained", training_status),
                is_leaf=is_type(TreeNamespace),
            )

            n_already_trained = len(
                jt.leaves(all_hps_already_trained, is_leaf=is_type(TreeNamespace))
            )

            if n_already_trained > 0:
                logger.info(
                    f"Skipping training of {n_already_trained} (of {n_total}) already-trained "
                    "models."
                )

            if args.postprocess or args.force_postprocess:
                # Determine which models need postprocessing
                if args.force_postprocess:
                    # Force re-postprocessing of all models (both already postprocessed and not)
                    to_postprocess = list(enumerate(model_records_flat))
                    to_postprocess_diff = all_hps_flat_diff
                else:
                    # Only postprocess models that weren't postprocessed yet
                    to_postprocess = [
                        (i, record)
                        for i, (record, status) in enumerate(
                            zip(model_records_flat, training_status_flat)
                        )
                        if status == "not_postprocessed"
                    ]
                    to_postprocess_diff = [
                        diff
                        for diff, status in zip(all_hps_flat_diff, training_status_flat)
                        if status == "not_postprocessed"
                    ]

                if len(to_postprocess) > 0:
                    if args.force_postprocess:
                        logger.info(
                            f"Force re-postprocessing {len(to_postprocess)} models "
                            "(including already postprocessed ones)."
                        )
                    else:
                        logger.info(
                            f"Postprocessing {len(to_postprocess)} models which were previously "
                            "trained but not postprocessed."
                        )
                    with db_session(autocommit=False) as db:
                        for (run_number, record), varying_params in piter(
                            zip(to_postprocess, to_postprocess_diff),
                            total=len(to_postprocess),
                            description="Postprocessing",
                        ):
                            process_model_post_training(
                                db,
                                record,
                                n_std_exclude=args.n_std_exclude,
                                process_all=True,
                                save_figures=not args.no_figures,
                                dump_path=fig_dump_path,
                                dump_formats=fig_dump_formats,
                                dump_params=varying_params,
                                run_number=run_number,
                            )
        else:
            # Not filtering - train everything
            all_hps_to_train = all_hps

        # Filter to models that need training
        all_hps_to_train_flat = jt.leaves(all_hps_to_train, is_leaf=is_type(TreeNamespace))

        if len(all_hps_to_train_flat) == 0:
            logger.info("No models to train. Exiting.")
            return

        # Get corresponding varying params for models to train
        if args.untrained_only:
            # Filter varying params to match untrained models
            all_hps_to_train_flat_diff = [
                diff
                for diff, status in zip(all_hps_flat_diff, training_status_flat)
                if status == "untrained"
            ]
        else:
            # Training all - use all varying params
            all_hps_to_train_flat_diff = all_hps_flat_diff

        logger.info(f"Training {len(all_hps_to_train_flat)} models.")

        for run_number, (hps, hps_diff) in piter(
            enumerate(zip(all_hps_to_train_flat, all_hps_to_train_flat_diff)),
            total=len(all_hps_to_train_flat),
            description="Training models",
        ):
            discard(
                train_and_save(
                    hps,
                    postprocess=args.postprocess,
                    n_std_exclude=args.n_std_exclude,
                    save_figures=not args.no_figures,
                    fig_dump_path=fig_dump_path,
                    fig_dump_formats=fig_dump_formats,
                    fig_dump_params=hps_diff,
                    run_number=run_number,
                    version_info=version_info,
                    key=key,
                )
            )

        logger.info("All training runs complete. Exiting.")


if __name__ == "__main__":
    main()
