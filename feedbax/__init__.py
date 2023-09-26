"""TODO

:copyright: Copyright 2023 by MLL <mll@mll.bio>.
:license: Apache 2.0, see LICENSE for details.
"""

import logging 
import os 

# logging.config.fileConfig('../logging.conf')
LOG_LEVEL = os.environ.get('FEEDBAX_LOG_LEVEL', 'DEBUG').upper()

logger = logging.getLogger(__package__)
logger.setLevel(LOG_LEVEL)

file_handler = logging.handlers.RotatingFileHandler(
    f'{__package__}.log',
    maxBytes=1_000_000,
    backupCount=1,
)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s,%(lineno)d: %(message)s",
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info('Logger configured.')