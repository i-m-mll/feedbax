"""The lazy public interface for installed Feedbax declarations."""

from importlib import import_module


_PUBLIC_ATTR_MODULES = {
    "ApplicationRegistryBundle": ".application",
    "TRAINING_METHODS": ".application",
    "OPERATIONS": ".application",
    "FIGURE_TEMPLATES": ".application",
    "compose_application": ".composition",
    "load_application": ".composition",
    "COMPONENTS": ".application",
    "PYTHON_TASKS": ".application",
    "BootstrapError": ".bootstrap",
    "BootstrapErrorCode": ".bootstrap",
    "BootstrapState": ".bootstrap",
    "PluginProvenance": ".bootstrap",
    "RegistryKey": ".bootstrap",
}

# Submodules the eager facade used to bind as attributes of this package.
_PUBLIC_SUBMODULES = ("application", "bootstrap")

__all__ = list(_PUBLIC_ATTR_MODULES)


def __getattr__(name: str):
    if name in _PUBLIC_SUBMODULES:
        value = import_module(f".{name}", __name__)
    else:
        try:
            module_name = _PUBLIC_ATTR_MODULES[name]
        except KeyError as exc:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
        value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__, *_PUBLIC_SUBMODULES})
