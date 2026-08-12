from collections.abc import Sequence
from numbers import Real
import re
from typing import Any

import numpy as np
import plotly.colors as plc
import plotly.io as pio
from _plotly_utils.basevalidators import ColorValidator
from plotly.colors import convert_colors_to_same_type, sample_colorscale

from feedbax.plot import apply_default_template


_COLOR_VALIDATOR = ColorValidator("color", "scatter")


def default_colors() -> Any:
    """Return the colorway of the active default Plotly template.

    Reading the colorway forces Plotly to load the default template, so this is resolved
    on demand rather than at import: this module is imported by the analysis and
    orchestration entry points, which must not pay for a plotting global they may never
    use. The Feedbax default template is applied first so the colorway is the one this
    package has always returned, not Plotly's stock default.
    """
    apply_default_template()
    return pio.templates[pio.templates.default].layout.colorway  # pyright: ignore


def __getattr__(name: str):
    if name == "DEFAULT_COLORS":
        value = default_colors()
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def color_add_alpha(rgb_str: str, alpha: float) -> str:
    """Add an alpha channel to an RGB or six-digit hex color."""
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
        raise ValueError("alpha must be a finite number between 0 and 1")
    alpha_value = float(alpha)
    if not np.isfinite(alpha_value) or not 0 <= alpha_value <= 1:
        raise ValueError("alpha must be a finite number between 0 and 1")
    if alpha_value == 0:
        alpha_value = 0.0
    alpha_str = np.format_float_positional(alpha_value, trim="0")

    rgb_match = re.fullmatch(r"rgb\(([^()]*)\)", rgb_str, flags=re.IGNORECASE)
    if rgb_match is not None:
        components = rgb_match.group(1).split(",")
        if len(components) == 3:
            literal = f"rgba({rgb_match.group(1)}, {alpha_str})"
            try:
                return _COLOR_VALIDATOR.validate_coerce(literal)
            except ValueError:
                pass

            normalized_components = []
            for component in components:
                stripped = component.strip()
                percentage = stripped.endswith("%")
                number_text = stripped[:-1] if percentage else stripped
                decimal_pattern = r"(?:\d+(?:\.\d*)?|\.\d+)"
                if not re.fullmatch(
                    rf"[+]?{decimal_pattern}(?:[eE][+-]?\d+)?", number_text
                ):
                    break
                plotly_safe = re.fullmatch(decimal_pattern, number_text) and all(
                    character == " " or not character.isspace() for character in component
                )
                if plotly_safe:
                    normalized_components.append(component)
                else:
                    value = float(number_text)
                    if not np.isfinite(value):
                        break
                    maximum = 100.0 if percentage else 255.0
                    normalized_number = np.format_float_positional(
                        min(value, maximum), trim="-"
                    )
                    normalized_components.append(
                        f"{normalized_number}{'%' if percentage else ''}"
                    )
            else:
                normalized = f"rgba({','.join(normalized_components)}, {alpha_str})"
                try:
                    return _COLOR_VALIDATOR.validate_coerce(normalized)
                except ValueError:
                    pass

    if re.fullmatch(r"#[0-9a-fA-F]{6}", rgb_str):
        red, green, blue = (int(rgb_str[index : index + 2], 16) for index in (1, 3, 5))
        return f"rgba({red}, {green}, {blue}, {alpha_str})"

    raise ValueError("color must use rgb(r,g,b) or #rrggbb format")


def arr_to_rgb(arr):
    return f"rgb({', '.join(map(str, arr))})"


def adjust_color_brightness(colors, factor=0.8):
    colors_arr = np.array(plc.convert_colors_to_same_type(colors, colortype="tuple")[0])
    return list(map(arr_to_rgb, factor * colors_arr))


def is_cyclical_colorscale(colorscale) -> bool:
    """Whether a colorscale ends on the color it starts with, so its ends coincide."""
    colors = plc.get_colorscale(colorscale)
    return colors[0][1] == colors[-1][1]


def sample_colorscale_unique(colorscale, samplepoints: int, **kwargs):
    """Helper to ensure we don't get repeat colors when using cyclical colorscales.

    Also avoids the division-by-zero error that `sample_colorscale` raises when `samplepoints == 1`.
    """
    if samplepoints == 1:
        n_sample = 2
        idxs = slice(1, None)
    elif is_cyclical_colorscale(colorscale):
        n_sample = samplepoints + 1
        idxs = slice(None, -1)
    else:
        n_sample = samplepoints
        idxs = slice(None)

    return sample_colorscale(colorscale, n_sample, **kwargs)[idxs]


def sample_colorscale_at(colorscale, positions: Sequence[float], **kwargs):
    """Sample a colorscale at explicit normalized positions rather than uniformly.

    ``positions`` are fractions of the scale in ``[0, 1]``, one per sample, in
    the caller's order. Cyclical colorscales are compressed onto
    ``[0, (n - 1) / n]`` so the last of ``n`` positions cannot land on the
    wrap-around duplicate of the first color — the same rule
    `sample_colorscale_unique` applies to uniform positions, which this
    function therefore reproduces exactly when the positions are uniform.
    """
    n = len(positions)
    if n > 1 and is_cyclical_colorscale(colorscale):
        cyclical_scale = (n - 1) / n
        positions = [position * cyclical_scale for position in positions]
    return sample_colorscale(colorscale, list(positions), **kwargs)


def _compute_colors(
    base_shape: tuple[int, ...],
    colors,
    colorscale: str,
    colorscale_axis: int,
    stride: int,
):
    """Return (color_sequence[C,3], colors_broadcast[*batch,3], color_idxs[*batch], default_labels, idxs_or_None).
    Works on the *per-leaf* shape (*batch, time, d). Also returns the indices to slice along
    the colorscale axis when `stride != 1`.
    """
    batch_shape = base_shape[:-2]  # *batch
    if colorscale_axis < 0:
        colorscale_axis += len(base_shape)
    if colorscale_axis >= len(base_shape) - 2:
        raise ValueError(f"colorscale_axis {colorscale_axis} points to a non-batch dimension")

    C_full = base_shape[colorscale_axis]
    if stride != 1:
        idxs = np.arange(C_full)[::stride]
        C = len(idxs)
    else:
        idxs = None
        C = C_full

    # Determine color sequence for C groups
    if colors is None:
        color_sequence = np.array(
            sample_colorscale_unique(colorscale, C, colortype="tuple")
        )  # (C,3)
    elif isinstance(colors, str):
        converted, _ = convert_colors_to_same_type([colors])
        color_sequence = np.array(converted[0] * C).reshape(C, -1)
    else:
        if len(colors) != C:
            raise ValueError("Length of colors must match number of color groups (after stride)")
        converted, _ = convert_colors_to_same_type(list(colors))
        color_sequence = np.array(converted)

    # Build per-group color indices and broadcasted RGBs across other batch dims
    color_idxs = np.arange(C).reshape((C,) + (1,) * (len(batch_shape) - 1))
    full_shape = (C,) + tuple(
        batch_shape[i] for i in range(len(batch_shape)) if i != colorscale_axis
    )
    color_idxs = np.broadcast_to(color_idxs, full_shape)

    colors_broadcast = np.broadcast_to(
        np.expand_dims(
            color_sequence, axis=tuple(1 + np.arange(len(full_shape) - 1))
        ),  # (C,1,1,...,3)
        full_shape + (3,),
    )

    default_labels = list(range(C))
    return color_sequence, colors_broadcast, color_idxs, default_labels, idxs
