import logging
import warnings
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, NamedTuple, Optional, TypeVar, Union

import feedbax.plot as fbp
import jax.tree as jt
import plotly.colors as plc
from jax_cookbook import is_type
from jaxtyping import PyTree

from feedbax.config import PLOTLY_CONFIG
from feedbax.config.namespace import TreeNamespace
from jax_cookbook import LDict

logger = logging.getLogger(__name__)


MEAN_LIGHTEN_FACTOR = PLOTLY_CONFIG.mean_lighten_factor
LIGHTEN_FACTORS = TreeNamespace(normal=1, dark=MEAN_LIGHTEN_FACTOR)


T = TypeVar("T")


class ColorscaleSpec(NamedTuple):
    sequence_fn: Callable[[TreeNamespace], Sequence]
    colorscale: Optional[Union[str, Sequence[str], Sequence[tuple]]] = None


@dataclass(frozen=True)
class ColorConfig:
    """Registered color defaults with explicit schema identity."""

    schema_id: str
    schema_version: str
    colorscales: Mapping[str, str | Sequence[str] | Sequence[tuple]] = field(
        default_factory=dict
    )
    color_specs: Mapping[str, ColorscaleSpec] = field(default_factory=dict)


def _trial_sequence(hps: TreeNamespace) -> Sequence[int]:
    return range(hps.eval_n)


# Generic colorscales shipped by Feedbax. Project vocabulary such as
# perturbation amplitudes, SISU settings, or task-condition names belongs in a
# caller-supplied ColorConfig.
COLORSCALES: dict[str, str | Sequence[str] | Sequence[tuple]] = dict(
    replicate="twilight",
    trial="Tealgrn",
)

DISCRETE_COLORSCALES = dict(
    category=plc.qualitative.D3,
)


"""
Generic default colorscales to try to set up, based on caller-provided
hyperparameters. Values are hyperparameter where-functions so we can try to
load them one-by-one.
"""
COMMON_COLOR_SPECS = {
    "trial": ColorscaleSpec(_trial_sequence),
}

GENERIC_COLOR_CONFIG = ColorConfig(
    schema_id="feedbax.plot.color_config.generic",
    schema_version="1",
    colorscales=MappingProxyType(COLORSCALES),
    color_specs=MappingProxyType(COMMON_COLOR_SPECS),
)

_COLOR_CONFIGS: dict[str, ColorConfig] = {
    GENERIC_COLOR_CONFIG.schema_id: GENERIC_COLOR_CONFIG,
}


def register_color_config(config: ColorConfig, *, replace: bool = False) -> ColorConfig:
    """Register a named color configuration for project-specific analysis defaults."""
    if config.schema_id in _COLOR_CONFIGS and not replace:
        raise ValueError(f"Color config {config.schema_id!r} is already registered")
    _COLOR_CONFIGS[config.schema_id] = config
    return config


def get_color_config(schema_id: str) -> ColorConfig:
    """Return a registered color configuration by stable schema id."""
    try:
        return _COLOR_CONFIGS[schema_id]
    except KeyError as exc:
        raise ValueError(f"Unknown color config {schema_id!r}") from exc


def is_discrete_colorscale(colorscale):
    """Determine if a colorscale is discrete (a sequence of colors) or continuous (a string name)."""
    return isinstance(colorscale, Sequence) and not isinstance(colorscale, str)


def get_variable_values(
    sequence_fn: Callable[[TreeNamespace], Sequence], hps: TreeNamespace
) -> Optional[Sequence]:
    """Safely get variable values from hyperparameters using the provided function.

    Args:
        sequence_fn: Function that extracts a sequence of values from hyperparameters
        hps: Hyperparameters to extract values from

    Returns:
        Sequence of values or None if extraction failed or returned empty values
    """
    try:
        values = sequence_fn(hps)
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
    hps: PyTree[TreeNamespace],
    var_fns: Mapping[str, ColorscaleSpec] | None = None,
    *,
    color_config: ColorConfig | str | None = None,
    colorscales: Mapping[str, str | Sequence[str] | Sequence[tuple]] | None = None,
) -> tuple[PyTree[dict], dict]:
    """Get all the colorscales we might want for our analyses, given the experiment hyperparameters.

    Args:
        hps: Hyperparameters tree
        var_fns: Dictionary mapping variable names to `ColorscaleSpecs`
        color_config: Registered color config or schema id to seed defaults from.
        colorscales: Extra or overriding colorscale mappings.

    Returns:
        Tuple of (PyTree of color mappings, updated colorscales dictionary)
    """
    if isinstance(color_config, str):
        color_config = get_color_config(color_config)
    elif color_config is None:
        color_config = GENERIC_COLOR_CONFIG

    if var_fns is None:
        var_fns = dict(color_config.color_specs)
    else:
        var_fns = dict(color_config.color_specs) | dict(var_fns)

    # Create updated colorscales dictionary
    resolved_colorscales = dict(color_config.colorscales)
    if colorscales is not None:
        resolved_colorscales.update(colorscales)
    for k, spec in var_fns.items():
        if spec.colorscale is not None:
            resolved_colorscales[k] = spec.colorscale

    def process_variable(hps, var_name, spec):
        # Get variable values
        values = get_variable_values(spec.sequence_fn, hps)
        if values is None:
            logger.info(f"'{var_name}' values unspecified in hyperparams; no colorscale set")
            return None

        # Get colorscale from updated dictionary
        colorscale = resolved_colorscales.get(var_name)
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
            for k, v in var_fns.items()
            if (result := process_variable(hps, k, v)) is not None
        },
        hps,
        is_leaf=is_type(TreeNamespace),
    )

    return colors, resolved_colorscales
