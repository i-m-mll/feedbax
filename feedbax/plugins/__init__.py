"""Plugin system for feedbax experiments framework."""

from .discovery import discover_experiment_packages
from .registry import ExperimentRegistry, get_default_registry

# Initialize experiment registry once at package import
EXPERIMENT_REGISTRY = discover_experiment_packages()

__all__ = [
    "EXPERIMENT_REGISTRY",
    "ExperimentRegistry",
    "get_default_registry",
    "discover_experiment_packages",
]
