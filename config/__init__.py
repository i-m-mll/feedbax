from feedbax_experiments.types import TreeNamespace

from .batch import load_batch_config
from .config import (
    CONFIG_DIR_ENV_VAR_NAME,
    _setup_logging,
    _setup_paths,
    load_config,
    load_config_as_ns,
)

# Initialize experiment registry once at package import
from feedbax_experiments.plugins import discover_experiment_packages
EXPERIMENT_REGISTRY = discover_experiment_packages()

# Load project-wide configuration from YAML resources in the `config` subpackage
CONSTANTS: TreeNamespace = load_config_as_ns("constants")
LOGGING: TreeNamespace = _setup_logging(load_config_as_ns("logging"))
PATHS: TreeNamespace = _setup_paths(load_config_as_ns("paths"))
PLOTLY_CONFIG: TreeNamespace = load_config_as_ns("plotly")
PRNG_CONFIG: TreeNamespace = load_config_as_ns("prng")
STRINGS: TreeNamespace = load_config_as_ns("strings")
