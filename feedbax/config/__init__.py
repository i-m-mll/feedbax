from typing import Optional

import feedbax.plugins
from feedbax.plugins.registry import ExperimentRegistry
from feedbax.config.namespace import TreeNamespace

from .batch import load_batch_config
from .config import (
    CONFIG_DIR_ENV_VAR_NAME,
    _setup_logging,
    _setup_paths,
    load_config,
    load_config_as_ns,
)

# Load project-wide configuration from YAML resources in the `config` subpackage
# These aren't populated until we call `configure_globals_for_package(...)` at the execution
# entrypoint, e.g. `run_analysis.py`
CONSTANTS = TreeNamespace()
LOGGING = TreeNamespace()
PATHS = TreeNamespace()
PLOTLY_CONFIG = TreeNamespace()
PRNG_CONFIG = TreeNamespace()
STRINGS = TreeNamespace()


def _overwrite_namespace(dst: TreeNamespace, src: TreeNamespace) -> None:
    dst.__dict__.clear()
    dst.__dict__.update(src.__dict__)


def configure_globals_for_package(
    package_name: str, registry: Optional[ExperimentRegistry]
) -> None:
    """Load package-scoped global resources for `package_name` with precedence:
    package override -> user config dir -> base feedbax.config."""
    # Using your existing load_config(..., registry=...) behavior:
    constants = load_config_as_ns(f"{package_name}/constants", registry=registry)
    logging_ns = _setup_logging(load_config_as_ns(f"{package_name}/logging", registry=registry))
    paths = _setup_paths(load_config_as_ns(f"{package_name}/paths", registry=registry))
    plotly_cfg = load_config_as_ns(f"{package_name}/plotly", registry=registry)
    prng_cfg = load_config_as_ns(f"{package_name}/prng", registry=registry)
    strings = load_config_as_ns(f"{package_name}/strings", registry=registry)

    _overwrite_namespace(CONSTANTS, constants)
    _overwrite_namespace(LOGGING, logging_ns)
    _overwrite_namespace(PATHS, paths)
    _overwrite_namespace(PLOTLY_CONFIG, plotly_cfg)
    _overwrite_namespace(PRNG_CONFIG, prng_cfg)
    _overwrite_namespace(STRINGS, strings)


# Populate the global namespaces from the base feedbax config BEFORE triggering
# plugin discovery. Discovery imports experiment packages (e.g. `rlrmp`), whose
# own import chains read populated config values such as
# `STRINGS.hps_level_label_sep` at import time (e.g. as default argument values
# in `feedbax.config.hyperparams`). Loading the base config first guarantees
# those reads succeed; if a single experiment package is then discovered, we
# re-load with package-override precedence below. See issue ccfe63a.
configure_globals_for_package("feedbax", None)

# Access `EXPERIMENT_REGISTRY` only now — this is what triggers plugin discovery.
# Deferring it to here (rather than as an import-time side effect of
# `feedbax.plugins`) means `feedbax.config` is already importable and its
# globals are populated when experiment packages import back into it.
EXPERIMENT_REGISTRY = feedbax.plugins.EXPERIMENT_REGISTRY

single_package_name = EXPERIMENT_REGISTRY.single_package_name()
if single_package_name is not None:
    # Re-load with package-override precedence now that the single experiment
    # package is registered.
    configure_globals_for_package(single_package_name, EXPERIMENT_REGISTRY)


__all__ = [
    "CONFIG_DIR_ENV_VAR_NAME",
    "CONSTANTS",
    "LOGGING",
    "PATHS",
    "PLOTLY_CONFIG",
    "PRNG_CONFIG",
    "STRINGS",
    "TreeNamespace",
    "load_config",
    "load_config_as_ns",
    "load_batch_config",
]
