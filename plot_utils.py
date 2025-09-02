import os
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import feedbax.plot as fbp
import ipywidgets as widgets
import jax
import jax.numpy as jnp
import jax.tree as jt
import matplotlib.figure as mplfig
import numpy as np
import plotly
import plotly.graph_objects as go
import pyperclip as clip
from IPython.display import HTML, display
from jax_cookbook import is_type

from feedbax_experiments.config import STRINGS
from feedbax_experiments.misc import filename_join


def _format_if_abbrev(s: str) -> str:
    if s in STRINGS.figures.abbrev_terms:
        s += "."
    return s


def get_label_str(s: str):
    """Converts flattened hyperparameter keys to labels that can be used in plots."""
    # TODO: Optionally add in line breaks if the string is long
    subs = [_format_if_abbrev(sub) for sub in s.split("_") if sub]
    return " ".join(subs).capitalize()


def get_savefig_fn(fig_dir: Path, suffix=""):
    """Returns a function that saves Matplotlib and Plotly figures to file in a given directory.

    This is convenient in notebooks, where all figures made within a single notebook are generally
    saved to the same directory.
    """

    def savefig(fig, label, ext=".svg", transparent=True, subdir: Optional[str] = None, **kwargs):
        if subdir is not None:
            save_dir = fig_dir / subdir
            save_dir.mkdir(exist_ok=True, parents=True)
        else:
            save_dir = fig_dir

        label = filename_join([label, suffix])

        if isinstance(fig, mplfig.Figure):
            fig.savefig(
                str(save_dir / f"{label}{ext}"),
                transparent=transparent,
                **kwargs,
            )

        elif isinstance(fig, go.Figure):
            # Save HTML for easy viewing, and JSON for embedding.
            # fig.write_html(save_dir / f'{label}.html')
            fig.write_json(save_dir / f"{label}.json")

            width = getattr(fig.layout, "width", None)
            height = getattr(fig.layout, "height", None)
            # Also save PNG for easy browsing and sharing
            fig.write_image(save_dir / f"{label}.png", scale=2, width=width, height=height)
            # fig.write_image(save_dir / f'{label}.webp', scale=2)

    return savefig


def figs_flatten_with_paths(figs):
    return jax.tree_util.tree_flatten_with_path(figs, is_leaf=is_type(go.Figure))[0]


def figleaves(tree):
    return jt.leaves(tree, is_leaf=is_type(go.Figure))


def add_context_annotation(
    fig: go.Figure,
    train_condition_strs: Optional[Sequence[str]] = None,
    perturbations: Optional[dict[str, tuple[float, Optional[int], Optional[int]]]] = None,
    n=None,
    i_trial=None,
    i_replicate=None,
    i_condition=None,
    y=1.1,
    **kwargs,
) -> go.Figure:
    """Annotates a figure with details about sample size, trials, replicates."""
    lines = []
    if train_condition_strs is not None:
        for condition_str in train_condition_strs:
            lines.append(f"Trained on {condition_str}")

    if perturbations is not None:
        for label, (amplitude, start, end) in perturbations.items():
            if amplitude is None:
                amplitude_str = ""
            else:
                amplitude_str = f" amplitude {amplitude:.2g}"

            line = f"Response to{amplitude_str} {label} "
            match (start, end):
                case (None, None):
                    line += "constant over trial"
                case (None, _):
                    line += f"from trial start to step {end}"
                case (_, None):
                    line += f"from step {start} to trial end"
                case (_, _):
                    line += f"from step {start} to {end}"

            lines.append(line)

    match (n, i_trial, i_replicate):
        case (None, None, None):
            pass
        case (n, None, None):
            lines.append(f"N = {n}")
        case (n, i_trial, None):
            lines.append(f"Single evaluation (#{i_trial}) of N = {n} model replicates")
        case (n, None, i_replicate):
            lines.append(f"N = {n} evaluations of model replicate {i_replicate}")
        case (None, i_trial, i_replicate):
            lines.append(f"Single evaluation (#{i_trial}) of model replicate {i_replicate}")
        case _:
            raise ValueError("Invalid combination of n, i_trial, and i_replicate for annotation")

    if i_condition is not None:
        lines.append(f"For single task condition ({i_condition})")

    # Adjust the layout of the figure to make room for the annotation
    if (margin_t := fig.layout.margin.t) is None:  # type: ignore
        margin_t = 100

    fig.update_layout(margin_t=margin_t + 5 * len(lines))

    if (height := fig.layout.height) is not None:  # type: ignore
        fig.update_layout(height=height + 5 * len(lines))

    fig.add_annotation(
        dict(
            text="<br>".join(lines),
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=y,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=12),  # Adjust font size as needed
            name="context_annotation",
            **kwargs,
        )
    )

    return fig


def get_merged_context_annotation(*figs):
    """Given figures with annotations added by `add_context_annotation`, return the text of a merged annotation.

    Note that this does not add the text as an annotation to any figure.
    """
    annotations_text = [
        next(iter(fig.select_annotations(selector=dict(name="context_annotation")))).text
        for fig in figs
    ]
    annotation_unique_lines = set(sum([text.split("<br>") for text in annotations_text], []))
    merged_annotation = "<br>".join(reversed(sorted(annotation_unique_lines)))
    return merged_annotation


def toggle_bounds_visibility(fig):
    """Toggle the visibility of traces with 'bound' in their names."""

    def toggle_visibility_if_bound(trace):
        if "bound" in trace.name:
            if trace.visible is None:
                trace.visible = False
            else:
                trace.visible = not trace.visible

    fig.for_each_trace(toggle_visibility_if_bound)


def plotly_vscode_latex_fix():
    """Fixes LaTeX rendering in Plotly figures in VS Code."""
    if os.environ.get("VSCODE_PID") is not None:
        plotly.offline.init_notebook_mode()
        display(
            HTML(
                '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'
            )
        )


def copy_fig_json(fig):
    """Copy Plotly figure's JSON representation to clipboard.

    I use this to embed interactive figures in my Obsidian notes: https://github.com/i-m-mll/obsidian-paste-as-embed
    """
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


class PlotlyFigureWidget:
    """Wraps a Plotly figure to display with widget interface elements in notebooks.

    In particular, adds buttons for copying a figure as PNG or JSON.

    Written with the help of Claude 3.5 Sonnet.
    """

    def __init__(
        self, fig: go.Figure, bg_fig: Optional[go.Figure] = None, annotation: Optional[str] = None
    ):
        """
        Initialize the widget with a Plotly figure.

        Args:
            fig (go.Figure): A Plotly graph objects figure
        """
        self.annotation = annotation
        self.fig = fig
        self.bg_fig = bg_fig

        if annotation is not None:
            self.fig.add_annotation(
                text=annotation,
                xref="paper",
                yref="paper",
                x=1,
                y=0,
                xanchor="right",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=8),
            )

        self.create_widgets()

    def create_widgets(self):
        """Create and arrange the widgets"""
        # Create the figure widgets
        if self.bg_fig:
            # Get divs for both figures
            main_div = self.fig.to_html(full_html=False)
            bg_div = self.bg_fig.to_html(full_html=False)

            print(main_div)

            container = widgets.HTML(
                value=f"""
                <div style="position: relative; width: 800px; height: 600px;">
                    <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; opacity: 0.3;">
                        {bg_div}
                    </div>
                    <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;">
                        {main_div}
                    </div>
                </div>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                """
            )
        else:
            container = go.FigureWidget(self.fig)

        # Create the buttons
        self.copy_png_button = widgets.Button(
            description="PNG", icon="copy", layout=widgets.Layout(width="auto", margin="2px")
        )

        self.copy_json_button = widgets.Button(
            description="JSON", icon="copy", layout=widgets.Layout(width="auto", margin="2px")
        )

        # Add button click handlers
        self.copy_png_button.on_click(self.copy_png)
        self.copy_json_button.on_click(self.copy_json)

        # Create label
        label = widgets.HTML(
            value="Copy as:", layout=widgets.Layout(margin="5px 2px", font_size="12px")
        )

        # Create button container (vertical stack)
        button_box = widgets.VBox(
            [label, self.copy_png_button, self.copy_json_button],
            layout=widgets.Layout(margin="0px 10px", align_items="center"),
        )

        # Combine figure and buttons horizontally
        self.container = widgets.HBox(
            [container, button_box], layout=widgets.Layout(align_items="center")
        )

    def copy_png(self, b):
        """Handler for PNG copy button"""
        fig = go.Figure(self.fig)
        # fig.update_layout(
        #     width=700, height=600,
        #     margin=dict(l=10, r=10, t=0, b=10),
        # )
        img_bytes = fig.to_image(format="png", scale=2)
        with open("/tmp/fig.png", "wb") as imgf:
            imgf.write(img_bytes)
        os.system("xclip -selection clipboard -t image/png -i /tmp/fig.png")

    def copy_json(self, b):
        """Handler for JSON copy button"""
        return copy_fig_json(self.fig)

    def show(self):
        """Display the widget"""
        display(self.container)


def get_underlay_fig(fig):
    fig = go.Figure(fig)
    fig.update_traces(opacity=0.3, line_color="grey")
    fig.update_layout(showlegend=False, xaxis_visible=False, yaxis_visible=False)
    return fig


def calculate_array_minmax(arrays_dict, indices=None, padding=0.05):
    """Calculate min and max values from multiple arrays with optional padding.

    Arguments:
        arrays_dict: A dictionary or LDict of arrays.
        indices: Optional indices to select from the last dimension of each array.
            If None, uses the full arrays.
        padding: Percentage of range to add as padding on both sides (default: 0.05 or 5%).

    Returns:
        Tuple of (min_val, max_val) with specified padding.
    """
    all_values = []
    for array in arrays_dict.values():
        # Extract specified indices if provided, otherwise use full array
        if indices is not None:
            # Select indices from last dimension
            selected_values = array[..., indices]
        else:
            selected_values = array
        all_values.append(selected_values)

    # Combine all values and find min/max
    if all_values:
        all_values_flat = np.concatenate([v.flatten() for v in all_values])
        min_val = np.min(all_values_flat)
        max_val = np.max(all_values_flat)

        # Add padding
        range_val = max_val - min_val
        min_val -= padding * range_val
        max_val += padding * range_val
    else:
        min_val, max_val = None, None

    return min_val, max_val
