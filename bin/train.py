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
from jax_cookbook import is_type, map_rich
from jax_cookbook.progress import piter

import feedbax_experiments
from feedbax_experiments.config import (
    PATHS,
    PRNG_CONFIG,
    load_batch_config,
    load_config,
)
from feedbax_experiments.database import ModelRecord, db_session
from feedbax_experiments.misc import deep_merge, discard, log_version_info
from feedbax_experiments.plugins import EXPERIMENT_REGISTRY
from feedbax_experiments.training.post_training import process_model_post_training
from feedbax_experiments.training.train import (
    prepare_to_train,
    train_and_save,
    train_and_save_from_config,
)
from feedbax_experiments.types import TreeNamespace

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
        default="html",
        help="Format(s) to dump figures in, comma-separated (e.g., 'html,png,pdf')",
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
                    records_for_postprocessing = model_records
                else:
                    # Only postprocess models that weren't postprocessed yet
                    records_for_postprocessing = eqx.filter(
                        model_records,
                        jt.map(lambda status: status == "not_postprocessed", training_status),
                        is_leaf=is_type(ModelRecord),
                    )

                to_postprocess = jt.leaves(records_for_postprocessing, is_leaf=is_type(ModelRecord))

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
                        map_rich(
                            lambda record: process_model_post_training(
                                db,
                                record,
                                n_std_exclude=args.n_std_exclude,
                                process_all=True,
                                save_figures=not args.no_figures,
                                dump_path=fig_dump_path,
                                dump_formats=fig_dump_formats,
                            ),
                            to_postprocess,
                            is_leaf=is_type(ModelRecord),
                            description="Postprocessing",
                        )
        else:
            all_hps_to_train = all_hps

        all_hps_to_train_flat = jt.leaves(all_hps_to_train, is_leaf=is_type(TreeNamespace))

        if len(all_hps_to_train_flat) == 0:
            logger.info("No models to train. Exiting.")
            return

        logger.info(f"Training {len(all_hps_to_train_flat)} models.")

        map_rich(
            # Don't save results in memory
            lambda hps: discard(
                train_and_save(
                    hps,
                    postprocess=args.postprocess,
                    n_std_exclude=args.n_std_exclude,
                    save_figures=not args.no_figures,
                    fig_dump_path=fig_dump_path,
                    fig_dump_formats=fig_dump_formats,
                    version_info=version_info,
                    key=key,
                )
            ),
            all_hps_to_train_flat,
            is_leaf=is_type(TreeNamespace),
            description="Training models",
        )

        logger.info("All training runs complete. Exiting.")


if __name__ == "__main__":
    main()
