from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from threading import RLock
from typing import TYPE_CHECKING, Iterator, Optional

from feedbax.config.namespace import TreeNamespace

if TYPE_CHECKING:
    from feedbax.plugins.registry import ExperimentRegistry

from .batch import load_batch_config
from .config import (
    CONFIG_DIR_ENV_VAR_NAME,
    _setup_logging,
    _setup_paths,
    load_config,
    load_config_as_ns,
)

# Load project-wide configuration from YAML resources in the `config` subpackage
# These aren't populated until we call `configure_globals_for_package(...)` at the execution
# entrypoint, e.g. `run_analysis.py`
CONSTANTS = TreeNamespace()
LOGGING = TreeNamespace()
PATHS = TreeNamespace()
PLOTLY_CONFIG = TreeNamespace()
PRNG_CONFIG = TreeNamespace()
STRINGS = TreeNamespace()

_CONFIG_GLOBAL_NAMES = (
    "CONSTANTS",
    "LOGGING",
    "PATHS",
    "PLOTLY_CONFIG",
    "PRNG_CONFIG",
    "STRINGS",
)
_CONFIG_GLOBAL_LOCK = RLock()


def _overwrite_namespace(dst: TreeNamespace, src: TreeNamespace) -> None:
    dst.__dict__.clear()
    dst.__dict__.update(src.__dict__)


def snapshot_config_globals() -> dict[str, TreeNamespace]:
    """Return a deep snapshot of process-global config namespaces."""
    with _CONFIG_GLOBAL_LOCK:
        return {name: deepcopy(globals()[name]) for name in _CONFIG_GLOBAL_NAMES}


def reset_config_globals(snapshot: dict[str, TreeNamespace] | None = None) -> None:
    """Restore process-global config namespaces.

    If ``snapshot`` is omitted, base Feedbax config is loaded. Supplying an
    explicit snapshot is the xdist-safe path for tests that temporarily mutate
    module-level namespaces.
    """
    with _CONFIG_GLOBAL_LOCK:
        if snapshot is None:
            configure_globals_for_package("feedbax", None)
            return
        for name in _CONFIG_GLOBAL_NAMES:
            _overwrite_namespace(globals()[name], deepcopy(snapshot[name]))


@contextmanager
def config_globals_context() -> Iterator[dict[str, TreeNamespace]]:
    """Context manager that restores global config namespaces on exit."""
    snapshot = snapshot_config_globals()
    try:
        yield snapshot
    finally:
        reset_config_globals(snapshot)


def configure_globals_for_package(
    package_name: str, registry: Optional[ExperimentRegistry]
) -> None:
    """Load package-scoped global resources for `package_name` with precedence:
    package override -> user config dir -> base feedbax.config."""
    with _CONFIG_GLOBAL_LOCK:
        # Using your existing load_config(..., registry=...) behavior:
        constants = load_config_as_ns(f"{package_name}/constants", registry=registry)
        logging_ns = _setup_logging(load_config_as_ns(f"{package_name}/logging", registry=registry))
        paths = _setup_paths(load_config_as_ns(f"{package_name}/paths", registry=registry))
        plotly_cfg = load_config_as_ns(f"{package_name}/plotly", registry=registry)
        prng_cfg = load_config_as_ns(f"{package_name}/prng", registry=registry)
        strings = load_config_as_ns(f"{package_name}/strings", registry=registry)

        _overwrite_namespace(CONSTANTS, constants)
        _overwrite_namespace(LOGGING, logging_ns)
        _overwrite_namespace(PATHS, paths)
        _overwrite_namespace(PLOTLY_CONFIG, plotly_cfg)
        _overwrite_namespace(PRNG_CONFIG, prng_cfg)
        _overwrite_namespace(STRINGS, strings)


# Populate base configuration without plugin discovery. Application composition
# may explicitly reconfigure for one bootstrapped experiment package later.
configure_globals_for_package("feedbax", None)


__all__ = [
    "CONFIG_DIR_ENV_VAR_NAME",
    "CONSTANTS",
    "LOGGING",
    "PATHS",
    "PLOTLY_CONFIG",
    "PRNG_CONFIG",
    "STRINGS",
    "TreeNamespace",
    "config_globals_context",
    "load_config",
    "load_config_as_ns",
    "load_batch_config",
    "reset_config_globals",
    "snapshot_config_globals",
]
