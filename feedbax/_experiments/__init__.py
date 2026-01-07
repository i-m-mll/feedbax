import logging
import os
from pathlib import Path

import plotly.io as pio

logger = logging.getLogger(__package__)
logger.addHandler(logging.NullHandler())


# Lazy imports to avoid circular dependency with feedbax.config
def __getattr__(name):
    if name == "enable_logging_handlers":
        from feedbax._experiments._logging import enable_logging_handlers
        return enable_logging_handlers
    elif name == "CONFIG_DIR_ENV_VAR_NAME":
        from feedbax._experiments.config import CONFIG_DIR_ENV_VAR_NAME
        return CONFIG_DIR_ENV_VAR_NAME
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
