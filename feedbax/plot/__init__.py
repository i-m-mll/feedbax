"""Plotting public API with optional backends loaded on first use."""

from importlib import import_module

import plotly.io as pio


_PUBLIC_ATTR_MODULES = {
    "AxesLabels": ".misc",
    "adjust_color_brightness": ".colors",
    "constructor_catalog": ".constructors",
    "get_figure_constructor": ".constructors",
    "get_figure_piece": ".constructors",
    "get_figure_template": ".constructors",
    "loss_history": ".plotly",
    "loss_history_compare": ".plotly",
    "profiles": ".profiles",
    "register_figure_constructor": ".constructors",
    "register_figure_piece": ".constructors",
    "register_figure_template": ".constructors",
    "registered_figure_constructors": ".constructors",
    "registered_figure_pieces": ".constructors",
    "registered_figure_templates": ".constructors",
    "sample_colorscale_unique": ".colors",
    "save_figure": ".io",
    "save_figure_with_spec": ".io",
    "trajectories": ".trajectories",
    "trajectories_2D": ".trajectories",
    "trajectories_3D": ".trajectories",
    "unregister_figure_constructor": ".constructors",
    "unregister_figure_piece": ".constructors",
    "unregister_figure_template": ".constructors",
}


def __getattr__(name: str):
    try:
        module_name = _PUBLIC_ATTR_MODULES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = list(_PUBLIC_ATTR_MODULES)

pio.templates.default = "plotly_white"
