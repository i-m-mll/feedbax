#!/usr/bin/env python
"""From the command line, train some models according to some configuration.

Load the config and pass it to `train_and_save_models`.

Takes a single positional argument: the path to the YAML config.
"""

import logging
from functools import partial

from feedbax_experiments import enable_logging_handlers
from feedbax_experiments.misc import deep_merge
from feedbax_experiments.plugins import EXPERIMENT_REGISTRY

enable_logging_handlers()

import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import argparse

import equinox as eqx
import feedbax
import jax
import jax.random as jr
import optax

import feedbax_experiments
from feedbax_experiments._warnings import enable_warning_dedup
from feedbax_experiments.config import (
    PRNG_CONFIG,
    load_batch_config,
    load_config,
)
from feedbax_experiments.misc import log_version_info
from feedbax_experiments.training.train import train_and_save_models

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
        "--n-std-exclude",
        type=int,
        default=2,
        help="In postprocessing, exclude model replicates with n_std greater than this value.",
    )
    parser.add_argument(
        "--save-figures", action="store_true", help="Save figures in postprocessing."
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

    # Optionally install warning de-duplication.
    if not args.show_duplicate_warnings:
        enable_warning_dedup()

    if args.seed is None:
        key = jr.PRNGKey(PRNG_CONFIG.seed)
    else:
        key = jr.PRNGKey(args.seed)

    version_info = log_version_info(
        jax,
        eqx,
        optax,
        git_modules=(feedbax, feedbax_experiments),
    )

    train_func = partial(
        train_and_save_models,
        untrained_only=args.untrained_only,
        postprocess=args.postprocess,
        n_std_exclude=args.n_std_exclude,
        save_figures=args.save_figures,
        version_info=version_info,
        key=key,
    )

    if args.single:
        module_key = args.single
        module_config = load_config(
            module_key, config_type="training", registry=EXPERIMENT_REGISTRY
        )

        trained_models, train_histories, model_records = train_func(
            config=module_config,
            expt_name=module_key,
        )
    else:
        batched_spec = load_batch_config(domain="training", config_key=args.batched)

        for module_key, run_params_list in batched_spec.items():
            module_config_base = load_config(
                module_key, config_type="training", registry=EXPERIMENT_REGISTRY
            )
            for i, run_params in enumerate(run_params_list):
                logger.info(
                    f"Training models for experiment {module_key}, "
                    f"run {i + 1} of {len(run_params_list)}"
                )

                module_config = deep_merge(module_config_base, run_params)

                #! TODO: Keep these in memory for all runs?
                trained_models, train_histories, model_records = train_func(
                    config=module_config,
                    expt_name=module_key,
                )

        logger.info("All training runs complete. Exiting.")


if __name__ == "__main__":
    main()
