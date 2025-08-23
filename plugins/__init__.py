"""Plugin system for feedbax experiments framework."""

from .registry import ExperimentRegistry, get_default_registry
from .discovery import discover_experiment_packages

__all__ = ["ExperimentRegistry", "get_default_registry", "discover_experiment_packages"]