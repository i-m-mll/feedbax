"""Backward compatibility shim - plugins is now in feedbax.plugins."""
from feedbax.plugins import *
from feedbax.plugins import (
    EXPERIMENT_REGISTRY,
    ExperimentRegistry,
    get_default_registry,
    discover_experiment_packages,
)
