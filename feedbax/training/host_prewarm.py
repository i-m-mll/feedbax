"""The Host-only preload for a managed Train worker profile."""

import os

from feedbax.training_host_worker import BOOTSTRAP_ENV, prewarm

if BOOTSTRAP_ENV in os.environ:
    prewarm(os.environ[BOOTSTRAP_ENV])
