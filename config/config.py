import logging
import os
import shlex
from copy import deepcopy
from cProfile import label
from functools import reduce
from importlib import resources
from itertools import product
from pathlib import Path
from re import sub
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, TypeVar

import jax.tree as jt
import yaml
from pytest import File

from feedbax_experiments.misc import deep_merge
from feedbax_experiments.types import TreeNamespace, dict_to_namespace

logger = logging.getLogger(__name__)


CONFIG_DIR_ENV_VAR_NAME = "RLRMP_CONFIG_DIR"
DEFAULT_CONFIG_FILENAME = "default.yml"


T = TypeVar("T", bound=SimpleNamespace)


def get_user_config_dir():
    """Get user config directory from environment variable, or return None."""
    env_config_dir = os.environ.get(CONFIG_DIR_ENV_VAR_NAME)
    if env_config_dir is None:
        return
    else:
        return Path(env_config_dir).expanduser()


def _load_defaults_hierarchy(
    name_parts: list[str],
    config_type: str,
) -> tuple[dict, list[str | None]]:
    """Load hierarchical default configs from root to parent directory.

    For name_parts=['part1', 'plant_perts'] and config_type='analysis':
    - Try to load feedbax_experiments.config.modules.analysis/default.yml
    - Try to load feedbax_experiments.config.modules.analysis.part1/default.yml
    - Return merged result
    """
    merged_config = {}
    base_subpackage = f"feedbax_experiments.config.modules.{config_type}"

    # Generate all subpackage paths from root to parent directory
    subpackage_paths: list[str | None] = [base_subpackage]
    for i in range(len(name_parts) - 1):  # Exclude the final config name
        subpath_parts = name_parts[: i + 1]
        subpackage_paths.append(".".join([base_subpackage, *subpath_parts]))

    # Load and merge each default.yml that exists
    for i, subpackage_name in enumerate(subpackage_paths):
        try:
            assert subpackage_name is not None
            with resources.open_text(subpackage_name, DEFAULT_CONFIG_FILENAME) as f:
                default_config = yaml.safe_load(f)
                if default_config is None:
                    subpackage_paths[i] = None  # Mark as not found
                    pass
                merged_config = deep_merge(merged_config, default_config)
        except (FileNotFoundError, ModuleNotFoundError):
            pass

    return merged_config, subpackage_paths


def load_config(
    name: str, config_type: Optional[Literal["training", "analysis"]] = None, *, registry=None
):
    """Load the contents of a project YAML config file resource as a nested dict."""
    name_parts = name.split(".")
    config_name = name_parts[-1]

    # If the user has specified a config directory, try to load the config from it
    user_config_dir = get_user_config_dir()
    if user_config_dir is not None:
        subpath = "/".join(name_parts[:-1])
        if config_type is not None:
            try:
                with open(
                    user_config_dir / "modules" / config_type / subpath / f"{config_name}.yml"
                ) as f:
                    return yaml.safe_load(f)
            except FileNotFoundError:
                logger.info(
                    f"Config file modules/{config_type}/{subpath}/{config_name}.yml not found in user config directory "
                    f"`{user_config_dir}`. Falling back to package resources."
                )
        else:
            try:
                with open(user_config_dir / subpath / f"{name}.yml") as f:
                    return yaml.safe_load(f)
            except FileNotFoundError:
                logger.info(
                    f"Config file {subpath}/{config_name}.yml not found in user config directory "
                    f"`{user_config_dir}`. Falling back to package resources."
                )

    # Load hierarchical defaults if config_type is specified
    if config_type is not None:
        merged_config, paths = _load_defaults_hierarchy(name_parts, config_type)
        if paths:
            # `None` corresponds to an empty defaults.yml
            paths_used = [p for p in paths if p is not None]
            if len(paths_used) == 1:
                logger.info(f"Loaded default.yml config from: {paths_used[0]}")
            else:
                logger.info(
                    f"Loaded default.yml configs hierarchically from: {', '.join(paths_used)}"
                )
    else:
        merged_config = {}

    # Load the final config and merge with defaults
    if registry is not None and config_type is not None:
        # Use registry to find the appropriate package
        try:
            package_name, resource_root = registry.get_config_resource_root(name)
            subpackage_name = f"{resource_root}.modules.{config_type}"
            subpackage_name = ".".join([subpackage_name, *name_parts[:-1]])
        except ValueError:
            # Fall back to default package if not found in registry
            if config_type is None:
                subpackage_name = "feedbax_experiments.config"
            else:
                subpackage_name = f"feedbax_experiments.config.modules.{config_type}"
            subpackage_name = ".".join([subpackage_name, *name_parts[:-1]])
    else:
        # Legacy behavior - load from feedbax_experiments
        if config_type is None:
            subpackage_name = "feedbax_experiments.config"
        else:
            subpackage_name = f"feedbax_experiments.config.modules.{config_type}"
        subpackage_name = ".".join([subpackage_name, *name_parts[:-1]])

    # Load the specific config file
    with resources.open_text(subpackage_name, f"{config_name}.yml") as f:
        final_config = yaml.safe_load(f) or {}

    logger.info(f"Loaded run config from resource {subpackage_name}/{config_name}.yml")

    # Merge defaults with final config (final config takes precedence)
    return deep_merge(merged_config, final_config)


def load_config_as_ns(
    name: str,
    config_type: Optional[Literal["training", "analysis"]] = None,
    to_type: type[T] = TreeNamespace,
) -> T:
    """Load the contents of a project YAML config file resource as a namespace."""
    return dict_to_namespace(load_config(name, config_type), to_type=to_type)


def _setup_paths(paths_ns: TreeNamespace):
    base_path = Path(paths_ns.base)

    def _setup_path(path_str: str):
        if path_str == "base":
            return base_path
        else:
            path = base_path / path_str
            path.mkdir(parents=True, exist_ok=True)
            return path

    return jt.map(_setup_path, paths_ns)


def _normalize_log_level(label: str, lvl: str | int) -> int:
    if isinstance(lvl, str):
        lvl = lvl.strip().upper()
        # this returns an int for a valid name, or the same object if invalid
        try:
            lvl = logging.getLevelNamesMapping()[lvl]
        except KeyError:
            raise ValueError(f"Invalid {label} specified in YAML config: {lvl!r}")
    if not isinstance(lvl, int):
        raise ValueError(f"Cannot parse log level {lvl!r}")
    return lvl


def _setup_logging(logging_ns: TreeNamespace):
    for label in ["file_level", "console_level", "pkg_console_levels", "pkgs_own_files"]:
        tree = getattr(logging_ns, label, None)
        if tree is None:
            continue
        tree_normalized = jt.map(
            lambda x: _normalize_log_level(label, x),
            tree,
        )
        setattr(logging_ns, label, tree_normalized)

    return logging_ns
