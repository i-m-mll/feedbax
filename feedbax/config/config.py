import logging
import os
import shlex
from copy import deepcopy
from cProfile import label
from dataclasses import dataclass
from functools import reduce
from importlib import resources
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, TextIO, TypeVar
from unittest.mock import DEFAULT

import jax.tree as jt
from ruamel.yaml import YAML

from feedbax.config.yaml import get_yaml_loader
from feedbax._experiments.misc import deep_merge
from feedbax.plugins.registry import ExperimentRegistry
from feedbax.types import TreeNamespace, dict_to_namespace

logger = logging.getLogger(__name__)


CONFIG_DIR_ENV_VAR_NAME = "FEEDBAX_EXPERIMENTS_CONFIG_DIR"
DEFAULT_CONFIG_FILENAME = "default"


T = TypeVar("T", bound=SimpleNamespace)


def _maybe_open_yaml(resource_root: str, stem: str) -> Optional[dict]:
    """
    Return parsed YAML from package resources, or None if missing.
    - Uses importlib.resources.files (open_text is deprecated).
    - Wraps the stream so constructors can use loader.stream.name.
    """
    try:
        path = resources.files(resource_root) / f"{stem}.yml"
    except ModuleNotFoundError:
        return None

    if not path.is_file():
        return None

    yaml = get_yaml_loader(typ="safe")

    try:
        # If you want a *real* filesystem path for .name (even from a zip),
        # materialize it with as_file; otherwise you can skip as_file and use str(res).
        with resources.as_file(path) as real_path:
            with open(real_path, "r", encoding="utf-8") as f:
                return yaml.load(f) or {}
    except (FileNotFoundError, ModuleNotFoundError):
        return None


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
    resource_root: str,
) -> tuple[dict, list[str | None]]:
    """Load hierarchical default configs from root to parent directory.

    For name_parts=['part1', 'plant_perts'] and config_type='analysis':
    - Try to load feedbax.config.modules.analysis/default.yml
    - Try to load feedbax.config.modules.analysis.part1/default.yml
    - Return merged result
    """
    base_subpackage = f"{resource_root}.modules.{config_type}"

    # Generate all subpackage paths from root to parent directory
    subpackage_paths: list[str | None] = [base_subpackage]
    for i in range(len(name_parts) - 1):  # Exclude the final config name
        subpath_parts = name_parts[: i + 1]
        subpackage_paths.append(".".join([base_subpackage, *subpath_parts]))

    # Load and merge each default.yml that exists
    merged_config = {}
    for i, subpackage_name in enumerate(subpackage_paths):
        assert subpackage_name is not None
        default_config = _maybe_open_yaml(subpackage_name, DEFAULT_CONFIG_FILENAME)
        if default_config is None:
            subpackage_paths[i] = None
            continue
        merged_config = deep_merge(merged_config, default_config or {})

    return merged_config, subpackage_paths


def load_config(
    name: str,
    config_type: Optional[Literal["training", "analysis"]] = None,
    *,
    registry: Optional[ExperimentRegistry] = None,
) -> dict[str, Any]:
    """
    Load a YAML config as a dict.

    Addressing:
      - Global configs (config_type is None):   name = "pkg/resource" or "resource"
      - Module configs (training/analysis):     name = "pkg/part1.some_module" or "part1.some_module"

    Precedence (globals):
      1) package override:   {pkg}.config/{resource}.yml  (if registry provided / determined)
      2) user config dir:    {user_config_dir}/{resource}.yml
      3) base fallback:      feedbax.config/{resource}.yml

    Module configs (training/analysis):
      - No user-dir lookup.
      - Defaults hierarchy is rooted at the owning package’s resource root.
    """
    yaml = get_yaml_loader(typ="safe")

    if config_type is None:
        # -------- GLOBAL CONFIGS --------
        # name is "pkg/resource" or just "resource"
        resource_name = name.split("/")[-1]  # trailing token is always the filename stem

        if registry is None:
            # No registry: user config dir -> base fallback
            user_config_dir = get_user_config_dir()
            if user_config_dir is not None:
                upath = user_config_dir / f"{resource_name}.yml"
                if upath.exists():
                    with open(upath, "r", encoding="utf-8") as f:
                        return yaml.load(f) or {}
                else:
                    logger.info(
                        f"Config file {resource_name}.yml not found in user config directory "
                        f"`{user_config_dir}`. Falling back to base resources."
                    )

            data = _maybe_open_yaml("feedbax.config", resource_name)
            if data is None:
                raise ValueError(
                    f"Global config '{resource_name}.yml' not found in base package resources."
                )
            return data

        # Registry provided: determine package for globals.
        # Accept "pkg/resource", or (if only one package is registered) "resource".
        if "/" in name:
            pkg = name.split("/", 1)[0]
            # Validate + compute resource root via registry
            _pkg_name, resource_root = registry.get_config_resource_root(name)
        else:
            single = registry.single_package_name()
            if not single:
                pkgs = "', '".join(sorted(registry._packages.keys()))
                raise ValueError(
                    f"Global config '{name}' is ambiguous with multiple packages registered. "
                    f"Use '<package>/{name}'. Installed packages: '{pkgs}'."
                )
            pkg = single
            _pkg_name, resource_root = registry.get_config_resource_root(f"{pkg}/{resource_name}")

        # Precedence: package override -> user config dir -> base
        data = _maybe_open_yaml(resource_root, resource_name)
        if data is not None:
            return data

        user_config_dir = get_user_config_dir()
        if user_config_dir is not None:
            upath = user_config_dir / f"{resource_name}.yml"
            if upath.exists():
                with open(upath, "r", encoding="utf-8") as f:
                    return yaml.load(f) or {}

        data = _maybe_open_yaml("feedbax.config", resource_name)
        if data is not None:
            return data

        raise ValueError(
            f"Global config '{resource_name}.yml' not found in {resource_root}, "
            f"user config dir, or feedbax.config."
        )

    # -------- MODULE CONFIGS (training/analysis) --------
    # We prefer to have a registry to resolve the owning package.
    if registry is not None:
        package_name, resource_root = registry.get_config_resource_root(name, domain=config_type)
    else:
        # Fallback: base package only (not recommended; you generally want a registry here)
        package_name = "feedbax"
        resource_root = "feedbax.config"

    # Derive the relative (dotted) module path (parents + leaf) from `name`
    if "/" in name:
        relative_key = name.split("/", 1)[1]  # dotted module path after '<pkg>/'
    elif name.startswith(f"{package_name}."):
        # Back-compat if someone passed dotted "<pkg>.<...>"
        relative_key = name[len(package_name) + 1 :]
    else:
        relative_key = name

    rel_parts = relative_key.split(".")
    leaf = rel_parts[-1]
    parents = rel_parts[:-1]

    # Defaults hierarchy rooted at the owning package
    merged_defaults, paths = _load_defaults_hierarchy(
        rel_parts,  # IMPORTANT: use relative (package-stripped) parts
        config_type,  # "training" | "analysis"
        resource_root,  # e.g., "rlrmp.config"
    )

    if paths:
        used = [p for p in paths if p is not None]
        if len(used) == 1:
            logger.info(f"Loaded default {config_type} config from: {used[0]}")
        elif used:
            logger.info(f"Loaded {config_type} defaults hierarchically from: {', '.join(used)}")

    # Final config file under {pkg}.config.modules.{domain}.<parents>/<leaf>.yml
    base = f"{resource_root}.modules.{config_type}"
    subpackage = ".".join([base, *parents]) if parents else base

    final_config = _maybe_open_yaml(subpackage, leaf)
    if final_config is None:
        raise ValueError(
            f"{config_type.capitalize()} module config '{leaf}.yml' not found under {subpackage}"
        )

    logger.info(f"Loaded {config_type} module config from resource {subpackage}/{leaf}.yml")
    return deep_merge(merged_defaults, final_config)


def load_config_as_ns(
    name: str,
    config_type: Optional[Literal["training", "analysis"]] = None,
    to_type: type[T] = TreeNamespace,
    registry: Optional[ExperimentRegistry] = None,
) -> T:
    """Load the contents of a project YAML config file resource as a namespace."""
    return dict_to_namespace(load_config(name, config_type, registry=registry), to_type=to_type)


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
