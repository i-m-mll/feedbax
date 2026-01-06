"""Backward-compatibility shim for feedbax_experiments.

This package re-exports from feedbax._experiments during the transition period.
Import paths like `from feedbax_experiments.analysis import X` continue to work.

DEPRECATED: Direct imports from feedbax_experiments will be removed in a future version.
            Migrate to: `from feedbax.analysis import X` (once reorganization is complete)
"""
import warnings

# Issue deprecation warning on import
warnings.warn(
    "Importing from 'feedbax_experiments' is deprecated. "
    "Please update imports to use 'feedbax._experiments' or the final 'feedbax.*' paths.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the internal module's contents
from feedbax._experiments import *
from feedbax import _experiments

# Make the package's __file__ and __path__ point to the actual implementation
__file__ = _experiments.__file__
__path__ = _experiments.__path__

# Re-export submodules for attribute access (e.g., feedbax_experiments.analysis)
from feedbax._experiments import (
    analysis,
    bin,
    config,
    dashboard,
    plugins,
    training,
)

# Re-export top-level modules
from feedbax._experiments import (
    colors,
    constants,
    database,
    hyperparams,
    misc,
    perturbations,
    plot,
    plot_utils,
    setup_utils,
    tree_utils,
    types,
    _debug,
    _logging,
    _warnings,
)

# Re-export common utilities
from feedbax._experiments._logging import enable_logging_handlers

__all__ = [
    "analysis",
    "bin",
    "colors",
    "config",
    "constants",
    "dashboard",
    "database",
    "hyperparams",
    "misc",
    "perturbations",
    "plot",
    "plot_utils",
    "plugins",
    "setup_utils",
    "training",
    "tree_utils",
    "types",
    "enable_logging_handlers",
]
