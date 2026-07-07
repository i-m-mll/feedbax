"""Plugin system for feedbax experiments framework."""

from typing import TYPE_CHECKING

from .discovery import (
    discover_experiment_packages,
    feedbax_plugin_entry_points,
    load_training_method_plugins,
)
from .registry import ExperimentRegistry, get_default_registry

# `EXPERIMENT_REGISTRY` is discovered lazily on first access rather than as an
# import-time side effect. Running `discover_experiment_packages()` at import
# time creates a circular import: `feedbax.config` imports `feedbax.plugins`
# before its own module-level globals (`PATHS`, etc.) are populated, discovery
# loads experiment-package entry points (e.g. `rlrmp`), and those packages
# import back into the still-partially-initialized `feedbax.config`. Deferring
# discovery until the registry is actually used lets `feedbax.config` finish
# initializing first. See issue ccfe63a.
_EXPERIMENT_REGISTRY: "ExperimentRegistry | None" = None

if TYPE_CHECKING:
    EXPERIMENT_REGISTRY: ExperimentRegistry


def _get_experiment_registry() -> "ExperimentRegistry":
    """Return the discovered experiment registry, running discovery once."""
    global _EXPERIMENT_REGISTRY
    if _EXPERIMENT_REGISTRY is None:
        _EXPERIMENT_REGISTRY = discover_experiment_packages()
    return _EXPERIMENT_REGISTRY


def __getattr__(name: str):
    if name == "EXPERIMENT_REGISTRY":
        return _get_experiment_registry()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EXPERIMENT_REGISTRY",
    "ExperimentRegistry",
    "get_default_registry",
    "discover_experiment_packages",
    "feedbax_plugin_entry_points",
    "load_training_method_plugins",
]
