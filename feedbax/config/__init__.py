from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from threading import RLock
from typing import Iterator

from feedbax.config.namespace import TreeNamespace

from .config import (
    CONFIG_DIR_ENV_VAR_NAME,
    _setup_logging,
    _setup_paths,
    load_config,
    load_config_as_ns,
)

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
            _load_base_globals()
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


def _load_base_globals() -> None:
    with _CONFIG_GLOBAL_LOCK:
        constants = load_config_as_ns("constants")
        logging_ns = _setup_logging(load_config_as_ns("logging"))
        paths = _setup_paths(load_config_as_ns("paths"))
        plotly_cfg = load_config_as_ns("plotly")
        prng_cfg = load_config_as_ns("prng")
        strings = load_config_as_ns("strings")

        _overwrite_namespace(CONSTANTS, constants)
        _overwrite_namespace(LOGGING, logging_ns)
        _overwrite_namespace(PATHS, paths)
        _overwrite_namespace(PLOTLY_CONFIG, plotly_cfg)
        _overwrite_namespace(PRNG_CONFIG, prng_cfg)
        _overwrite_namespace(STRINGS, strings)


_load_base_globals()


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
    "reset_config_globals",
    "snapshot_config_globals",
]
