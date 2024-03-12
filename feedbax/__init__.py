"""
:copyright: Copyright 2023-2024 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

import logging
import logging.handlers as loghandlers
import os
import warnings

from feedbax._io import save, load
from feedbax._model import (
    AbstractModel,
    ModelInput,
    wrap_stateless_callable,
    wrap_stateless_keyless_callable,
)
from feedbax._staged import (
    AbstractStagedModel,
    ModelStage,
    pformat_model_spec,
    pprint_model_spec,
)
from feedbax.state import AbstractState
from feedbax._tree import (
    get_ensemble,
    random_split_like_tree,
    tree_array_bytes,
    tree_call,
    tree_map_unzip,
    tree_set,
    tree_stack,
    tree_struct_bytes,
    tree_take,
    tree_unzip,
)

# logging.config.fileConfig('../logging.conf')

if os.environ.get("FEEDBAX_DEBUG", False) == "True":
    DEFAULT_LOG_LEVEL = "DEBUG"
else:
    DEFAULT_LOG_LEVEL = "INFO"

LOG_LEVEL = os.environ.get("FEEDBAX_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()

logger = logging.getLogger(__package__)
logger.setLevel(LOG_LEVEL)

file_handler = loghandlers.RotatingFileHandler(
    f"{__package__}.log",
    maxBytes=1_000_000,
    backupCount=1,
)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s,%(lineno)d: %(message)s",
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logging.captureWarnings(True)

logger.info("Logger configured.")
