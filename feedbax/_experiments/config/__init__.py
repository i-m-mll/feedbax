"""Backward compatibility shim - config is now in feedbax.config."""

# Use lazy imports to avoid circular dependency during feedbax.config initialization
_EXPORTED_NAMES = [
    "CONFIG_DIR_ENV_VAR_NAME",
    "CONSTANTS",
    "LOGGING",
    "PATHS",
    "PLOTLY_CONFIG",
    "PRNG_CONFIG",
    "STRINGS",
    "load_config",
    "load_config_as_ns",
    "load_batch_config",
    "configure_globals_for_package",
]

def __getattr__(name):
    if name in _EXPORTED_NAMES or name == "__all__":
        import feedbax.config as _config
        if name == "__all__":
            return _EXPORTED_NAMES
        return getattr(_config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return _EXPORTED_NAMES
