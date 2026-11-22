"""The structure of a field's value, as Studio draws its parts."""

from math import isfinite

from feedbax.contracts.expression_values import CallValue, DrawValue, FieldValue, LiteralValue, Site
from feedbax.contracts.studio_field import StudioDensity, StudioValueStructure
from feedbax.expression_registry import FunctionRegistry
from feedbax.expression_text import print_field

_DENSITY_POINTS = 48


def value_structure(value: FieldValue, site: Site, functions: FunctionRegistry) -> StudioValueStructure:
    arguments = value.args if isinstance(value, DrawValue | CallValue) else ()
    return StudioValueStructure(
        value=value, arguments=[print_field(argument, site) for argument in arguments],
        density=_density(value, functions) if isinstance(value, DrawValue) else None,
    )


def _density(draw: DrawValue, functions: FunctionRegistry) -> StudioDensity | None:
    function = functions.resolve(draw.family)
    if function is None or function.density is None:
        return None
    if not all(isinstance(arg, LiteralValue) and type(arg.value) in (int, float) for arg in draw.args):
        return None
    if len({arg.unit for arg in draw.args}) > 1:
        return None
    density = function.density(*(float(arg.value) for arg in draw.args))
    if density is None or not (isfinite(density.low) and isfinite(density.high)):
        return None
    import numpy as np

    values = np.asarray(density.pdf(np.linspace(density.low, density.high, _DENSITY_POINTS)), dtype=float)
    peak = values.max()
    if not peak > 0:
        return None
    return StudioDensity(low=density.low, high=density.high, values=(values / peak).round(4).tolist())
