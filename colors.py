import logging
import warnings
from collections.abc import Callable, Hashable, Sequence
from types import SimpleNamespace
from typing import Literal, NamedTuple, Optional, TypeVar, Union

import feedbax.plotly as fbp
import jax.tree as jt
import jax_cookbook.tree as jtree
import plotly.colors as plc
from jax_cookbook import is_type
from jaxtyping import PyTree

from feedbax_experiments.config import PLOTLY_CONFIG
from feedbax_experiments.types import LDict, TreeNamespace

logger = logging.getLogger(__name__)


MEAN_LIGHTEN_FACTOR = PLOTLY_CONFIG.mean_lighten_factor
LIGHTEN_FACTORS = TreeNamespace(normal=1, dark=MEAN_LIGHTEN_FACTOR)


T = TypeVar("T")


class ColorscaleSpec(NamedTuple):
    sequence_func: Callable[[TreeNamespace], Sequence]
    colorscale: Optional[Union[str, Sequence[str], Sequence[tuple]]] = None


# Colorscales
COLORSCALES: dict[str, str] = dict(
    train__pert__std="viridis",
    pert__amp="plotly3",
    sisu="thermal",
    reach_condition="phase",
    replicate="twilight",
    trial="Tealgrn",
    # pert_var=plc.qualitative.D3,
)

DISCRETE_COLORSCALES = dict(
    pert_var=plc.qualitative.D3,  # list[str]
)


"""
Default colorscales to try to set up, based on hyperparameters.
Values are hyperparameter where-functions so we can try to load them one-by-one.
"""
COMMON_COLOR_SPECS = {
    k: ColorscaleSpec(func)
    for k, func in dict(
        # sisu=
        pert__amp=lambda hps: hps.pert.amp,
        train__pert__std=lambda hps: hps.train.pert.std,
        # pert_var=  #?
        #  reach_condition=  #?
        sisu=lambda hps: hps.sisu,
        trial=lambda hps: range(hps.eval_n),
    ).items()
}


def is_discrete_colorscale(colorscale):
    """Determine if a colorscale is discrete (a sequence of colors) or continuous (a string name)."""
    return isinstance(colorscale, Sequence) and not isinstance(colorscale, str)


def get_variable_values(
    sequence_func: Callable[[TreeNamespace], Sequence], hps: TreeNamespace
) -> Optional[Sequence]:
    """Safely get variable values from hyperparameters using the provided function.

    Args:
        sequence_func: Function that extracts a sequence of values from hyperparameters
        hps: Hyperparameters to extract values from

    Returns:
        Sequence of values or None if extraction failed or returned empty values
    """
    try:
        values = sequence_func(hps)
        if values is None or len(values) == 0:
            return None
        return values
    except AttributeError:
        # This happens when the function tries to access attributes
        # that don't exist in this hyperparameter set
        return None


def get_colors_dicts_from_discrete(
    keys: Sequence[Hashable],
    colors: Sequence[str] | Sequence[tuple],
    lighten_factor: PyTree[float, "T"] = LIGHTEN_FACTORS,
    colortype: Literal["rgb", "tuple"] = "rgb",
    label: Optional[str] = None,
) -> PyTree[dict[Hashable, str | tuple], "T"]:
    """Create color dictionaries from a discrete set of colors.

    Args:
        keys: The values to map to colors
        colors: The colors to use (will cycle if there are more keys than colors)
        lighten_factor: Factor to adjust brightness by for each variant
        colortype: Output color format ('rgb' or 'tuple')
        label: Optional label for the LDict

    Returns:
        PyTree of dictionaries mapping keys to colors
    """

    def _get_colors(colors, factor):
        colors = fbp.adjust_color_brightness(colors, factor)
        return plc.convert_colors_to_same_type(colors, colortype=colortype)[0]

    if label is not None:
        dict_constructor = LDict.of(label)
    else:
        dict_constructor = dict

    # Cycle colors if there are more keys than colors
    if len(keys) > len(colors):
        warnings.warn(
            f"More values ({len(keys)}) than discrete colors ({len(colors)}), for '{label}'. Colors will cycle."
        )
        colors_cycled = []
        for i in range(len(keys)):
            colors_cycled.append(colors[i % len(colors)])
        colors = colors_cycled

    return jt.map(
        lambda f: dict_constructor(zip(keys, _get_colors(colors, f))),
        lighten_factor,
    )


def setup_colors(
    hps: PyTree[TreeNamespace], var_funcs: dict[str, ColorscaleSpec]
) -> tuple[PyTree[dict], dict]:
    """Get all the colorscales we might want for our analyses, given the experiment hyperparameters.

    Args:
        hps: Hyperparameters tree
        var_funcs: Dictionary mapping variable names to `ColorscaleSpecs`

    Returns:
        Tuple of (PyTree of color mappings, updated colorscales dictionary)
    """
    # Create updated colorscales dictionary
    colorscales = COLORSCALES.copy()
    for k, spec in var_funcs.items():
        if spec.colorscale is not None:
            colorscales[k] = spec.colorscale

    def process_variable(hps, var_name, spec):
        # Get variable values
        values = get_variable_values(spec.sequence_func, hps)
        if values is None:
            logger.info(f"'{var_name}' values unspecified in hyperparams; no colorscale set")
            return None

        # Get colorscale from updated dictionary
        colorscale = colorscales.get(var_name)
        if colorscale is None:
            logger.warn(f"no colorscale determined for variable '{var_name}'")
            return None

        # Handle discrete or continuous colorscales
        if is_discrete_colorscale(colorscale):
            colors = colorscale
        else:
            colors = fbp.sample_colorscale_unique(colorscale, len(values))

        return get_colors_dicts_from_discrete(
            values, colors, lighten_factor=LIGHTEN_FACTORS, label=var_name
        )

    colors = jt.map(
        lambda hps: {
            k: result
            for k, v in var_funcs.items()
            if (result := process_variable(hps, k, v)) is not None
        },
        hps,
        is_leaf=is_type(TreeNamespace),
    )

    return colors, colorscales
