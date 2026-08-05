"""Plotting public API with optional backends loaded on first use."""

from importlib import import_module


DEFAULT_TEMPLATE = "plotly_white"

# Plotly's own out-of-the-box default. A process whose default is anything else has had
# one deliberately chosen — by `feedbax-analysis --plotly-template`, by project config,
# or by a notebook — and that choice outranks this package's default.
_PLOTLY_STOCK_DEFAULT = "plotly"

_default_template_applied = False


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


def apply_default_template() -> str:
    """Install the Feedbax default Plotly template, once, on first use of the plot surface.

    Assigning `plotly.io.templates.default` loads and validates the named template out of
    Plotly's bundled package data, which costs well over a hundred milliseconds. Doing
    that at import time made every console script that transitively imports this package
    pay for a plotting global it may never touch, so the mutation is deferred until
    something actually uses the plotting surface.

    Deferring moves the assignment later in the process, so it must not overwrite a
    template someone chose on purpose in the meantime. Two rules keep that from
    happening: the assignment is one-shot, and it only fires while the process default is
    still Plotly's own. Together they reproduce what an import-time assignment gave —
    the Feedbax default unless a caller says otherwise — without depending on the order
    in which plotting modules happen to be imported.

    Returns:
        The Plotly template name in effect for this process.
    """
    global _default_template_applied

    # Deferred so that importing this package does not import or mutate Plotly.
    import plotly.io as pio  # noqa: PLC0415

    if not _default_template_applied:
        _default_template_applied = True
        if pio.templates.default == _PLOTLY_STOCK_DEFAULT:
            pio.templates.default = DEFAULT_TEMPLATE
    return pio.templates.default


def __getattr__(name: str):
    try:
        module_name = _PUBLIC_ATTR_MODULES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    apply_default_template()
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = [*_PUBLIC_ATTR_MODULES, "DEFAULT_TEMPLATE", "apply_default_template"]
