"""Backward compatibility shim - config is now in feedbax.config.config."""
from feedbax.config.config import *
from feedbax.config.config import (
    CONFIG_DIR_ENV_VAR_NAME,
    load_config,
    load_config_as_ns,
    get_user_config_dir,
    _setup_logging,
    _setup_paths,
)
