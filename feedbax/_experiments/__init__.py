import logging
import os
from pathlib import Path

import plotly.io as pio

from feedbax._experiments._logging import enable_logging_handlers
from feedbax._experiments.config import CONFIG_DIR_ENV_VAR_NAME

logger = logging.getLogger(__package__)
logger.addHandler(logging.NullHandler())
