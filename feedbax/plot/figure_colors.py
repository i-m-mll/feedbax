"""Resolve named figure color scales over the values drawn by their groups."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import inf, isfinite, nextafter
from typing import Any

from plotly.colors import qualitative, sample_colorscale

from feedbax.contracts.figure_document import ColorScale, ContinuousColorScale, PaletteValue, ordinal_key


@dataclass(frozen=True)
class ColorSample:
    value: PaletteValue
    label: str
    field_label: str
    unit: str | None


@dataclass(frozen=True)
class ResolvedColorScale:
    colors: Mapping[tuple[bool, PaletteValue], str]
    limits: tuple[float, float]
    axis: Mapping[str, Any]


def _fraction(value: float, lo: float, hi: float) -> float:
    if value <= lo:
        return 0.0
    if value >= hi:
        return 1.0
    extent = hi - lo
    return (value - lo) / extent if isfinite(extent) else (value / 2 - lo / 2) / (hi / 2 - lo / 2)


def resolve_color_scale(scale: ColorScale, samples: Sequence[ColorSample]) -> ResolvedColorScale:
    unique = {ordinal_key(sample.value): sample for sample in samples}
    title = scale.title or samples[0].field_label
    colorbar: dict[str, Any] = {"title": title, "thickness": 14}
    if isinstance(scale, ContinuousColorScale):
        if any(type(sample.value) not in (int, float) or not isfinite(sample.value) for sample in samples):
            raise ValueError("A continuous color scale requires finite numbers.")
        if len({sample.unit for sample in samples}) != 1:
            raise ValueError("A continuous color scale requires one unit across its bound fields.")
        values = [float(sample.value) for sample in unique.values()]
        lo, hi = scale.domain if scale.domain is not None else (min(values), max(values))
        if lo == hi:
            below, above = nextafter(lo, -inf), nextafter(hi, inf)
            colorbar["tickvals"] = [lo]
            lo, hi = (below if isfinite(below) else lo), (above if isfinite(above) else hi)
            fractions = [0.5] * len(values)
        else:
            fractions = [_fraction(value, lo, hi) for value in values]
        colors = dict(zip(unique, sample_colorscale(scale.palette, fractions), strict=True))
        plotly_scale = scale.palette
    else:
        keys = tuple(ordinal_key(value) for value in scale.domain) if scale.domain is not None else tuple(unique)
        if any(key not in keys for key in unique):
            raise ValueError("A plotted category is outside its ordinal color domain.")
        palette = {"category10": qualitative.D3, "plotly": qualitative.Plotly, "dark24": qualitative.Dark24}[scale.palette]
        colors = {key: palette[index % len(palette)] for index, key in enumerate(keys)}
        size = len(keys)
        plotly_scale = [
            (edge / size, colors[key])
            for index, key in enumerate(keys)
            for edge in (index, index + 1)
        ]
        lo, hi = -0.5, size - 0.5
        colorbar.update(
            tickmode="array", tickvals=list(range(size)),
            ticktext=[unique[key].label if key in unique else str(key[1]) for key in keys],
        )
    axis = {"cmin": lo, "cmax": hi, "colorscale": plotly_scale, "showscale": scale.legend, "colorbar": colorbar}
    return ResolvedColorScale(colors, (lo, hi), axis)
