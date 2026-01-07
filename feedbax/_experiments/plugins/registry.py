"""Backward compatibility shim - registry is now in feedbax.plugins.registry."""
from feedbax.plugins.registry import *
from feedbax.plugins.registry import (
    PackageMetadata,
    ExperimentRegistry,
    get_default_registry,
)
