import logging
import os
from pathlib import Path

import plotly.io as pio

from rlrmp._logging import enable_logging_handlers
from rlrmp.config import (
    CONFIG_DIR_ENV_VAR_NAME,
    LOGGING,
    PLOTLY_CONFIG,
)

logger = logging.getLogger(__package__)
logger.addHandler(logging.NullHandler())

# Set the default Plotly theme for the project
pio.templates.default = PLOTLY_CONFIG.templates.default
