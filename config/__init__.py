from feedbax_experiments.plugins.registry import ExperimentRegistry
from feedbax_experiments.types import TreeNamespace

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


def configure_globals_for_package(package_name: str, registry: ExperimentRegistry) -> None:
    """Load package-scoped global resources for `package_name` with precedence:
    package override -> user config dir -> base feedbax_experiments.config."""
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
