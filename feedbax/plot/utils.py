import functools
import inspect
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import jax
import plotly.graph_objects as go

from feedbax._vendor.jax_cookbook import is_type
from feedbax.config import STRINGS
from feedbax.plot.io import safe_html_id
from feedbax.plot.lifecycle import close_figure, is_matplotlib_figure
from feedbax.plot.ranges import calculate_array_minmax as calculate_array_minmax


def with_caller_logger(func):
    """Provide the caller's logger to wrapped functions that accept `logger=`."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if "logger" not in kwargs:
            caller_module = None
            caller_frame = inspect.currentframe()
            if caller_frame is not None:
                caller_module = inspect.getmodule(caller_frame.f_back)
            if caller_module is not None:
                kwargs["logger"] = logging.getLogger(caller_module.__name__)
            else:
                kwargs["logger"] = logging.getLogger(func.__module__)
        return func(*args, **kwargs)

    return wrapper


def _format_if_abbrev(s: str) -> str:
    if s in STRINGS.figures.abbrev_terms:
        s += "."
    return s


def get_label_str(s: str):
    """Converts flattened hyperparameter keys to labels that can be used in plots."""
    # TODO: Optionally add in line breaks if the string is long
    subs = [_format_if_abbrev(sub) for sub in s.split("_") if sub]
    return " ".join(subs).capitalize()


def figs_flatten_with_paths(figs):
    return jax.tree_util.tree_flatten_with_path(figs, is_leaf=is_type(go.Figure))[0]


def copy_fig_json(fig):
    """Copy Plotly figure's JSON representation to clipboard.

    I use this to embed interactive figures in my Obsidian notes: https://github.com/i-m-mll/obsidian-paste-as-embed
    """
    import pyperclip as clip

    fig = go.Figure(fig)
    fig.update_layout(
        width=700,
        height=600,
        margin=dict(l=10, r=10, t=0, b=10),
        legend=dict(
            yanchor="top",
            y=0.9,
            xanchor="right",
        ),
    )
    clip.copy(str(fig.to_json()))


EXTS_WITH_EXIF = ["jpg", "jpeg", "tif", "tiff", "webp"]


@with_caller_logger
def savefig(
    fig,
    label,
    fig_dir: Path,
    image_formats: Sequence[str],
    transparent=True,
    metadata: Optional[dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
    close: bool = False,
    **kwargs,
):
    try:
        path = str(fig_dir / f"{label}") + ".{ext}"

        if is_matplotlib_figure(fig):
            for ext in image_formats:
                fig.savefig(
                    path.format(ext=ext),
                    transparent=transparent,
                    **kwargs,
                )

        elif isinstance(fig, go.Figure):
            fig.update_layout(meta=metadata)

            for ext in image_formats:
                path_i = path.format(ext=ext)
                if ext == "html":
                    # Plotly mints a random UUID for the container element
                    # otherwise, so two executions of the same figure would
                    # write different bytes while agreeing in every observable
                    # respect. Analysis figure artifacts are digest
                    # authenticated, and rebuild-as-verification compares those
                    # digests, so a random id would report false drift forever.
                    # The label is already unique within the output directory,
                    # and each render is a standalone document. `kwargs` is
                    # shared with the other formats in this loop, so the id goes
                    # into a copy rather than into `kwargs` itself.
                    fig.write_html(path_i, **{"div_id": safe_html_id(label), **kwargs})
                elif ext == "json":
                    fig.write_json(path_i, engine="auto", **kwargs)
                else:
                    width = getattr(fig.layout, "width", None)
                    height = getattr(fig.layout, "height", None)
                    fig.write_image(path_i, scale=2, width=width, height=height, **kwargs)

                    if metadata is not None and ext in EXTS_WITH_EXIF:
                        import pyexiv2

                        pyexiv2.registerNs("http://example.com/ns/custom/", "custom")
                        metadata_xmp = {f"Xmp.custom.{k}": v for k, v in metadata.items()}
                        metadata_xmp["Xmp.dc.description"] = json.dumps(metadata, indent=2)
                        with pyexiv2.Image(path_i) as img:
                            img.modify_xmp(metadata_xmp)
    finally:
        if close:
            close_figure(fig)
