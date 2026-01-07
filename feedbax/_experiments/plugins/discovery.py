"""Backward compatibility shim - discovery is now in feedbax.plugins.discovery."""
from feedbax.plugins.discovery import *
from feedbax.plugins.discovery import (
    discover_experiment_packages,
    register_package_from_module_info,
)
