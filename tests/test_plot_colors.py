from __future__ import annotations

import math

import plotly.graph_objects as go
import pytest

from feedbax.plot.colors import color_add_alpha


@pytest.mark.parametrize(
    ("color", "alpha", "expected"),
    [
        ("rgb(31,119,180)", 0.6, "rgba(31,119,180, 0.6)"),
        ("rgb(31, 119, 180)", 0.6, "rgba(31, 119, 180, 0.6)"),
        ("rgb(0.25, .5, 1.0)", 0.6, "rgba(0.25, .5, 1.0, 0.6)"),
        ("rgb(1.,.5,3)", 0.5, "rgba(1.,.5,3, 0.5)"),
        ("RGB(1,2,3)", 0.5, "rgba(1,2,3, 0.5)"),
        ("rgb(256,2,3)", 0.5, "rgba(256,2,3, 0.5)"),
        ("rgb(+1,2,3)", 0.5, "rgba(1,2,3, 0.5)"),
        ("rgb(1e2,2,3)", 0.5, "rgba(100,2,3, 0.5)"),
        ("rgb(1e300,2,3)", 0.5, "rgba(255,2,3, 0.5)"),
        ("rgb(1,\t2,3)", 0.5, "rgba(1,2,3, 0.5)"),
        ("rgb(0%,50.5%,100%)", 0.4, "rgba(0%,50.5%,100%, 0.4)"),
        ("rgb(101%,0%,0%)", 0.4, "rgba(101%,0%,0%, 0.4)"),
        ("rgb(100%,0,0%)", 0.4, "rgba(100%,0,0%, 0.4)"),
        ("#2563eb", 0.2, "rgba(37, 99, 235, 0.2)"),
        ("#FFFFFF", 1.0, "rgba(255, 255, 255, 1.0)"),
    ],
)
def test_color_add_alpha_supports_rgb_and_six_digit_hex(
    color: str, alpha: float, expected: str
) -> None:
    result = color_add_alpha(color, alpha)

    assert result == expected
    assert go.Scatter(fillcolor=result).fillcolor == result


@pytest.mark.parametrize(
    "color",
    [
        "#fff",
        "red",
        "rgba(1,2,3,0.5)",
        "rgb(1,2)",
        "rgb(a,2,3)",
        "rgb(nan,2,3)",
        "rgb(inf,2,3)",
        "rgb(-1%,0%,0%)",
        "rgb(nan%,0%,0%)",
        "rgb(100%%,0%,0%)",
    ],
)
def test_color_add_alpha_rejects_unsupported_or_invalid_colors(color: str) -> None:
    with pytest.raises(ValueError, match="rgb\\(r,g,b\\) or #rrggbb"):
        color_add_alpha(color, 0.5)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, math.nan, math.inf, True, "0.5"])
def test_color_add_alpha_rejects_invalid_alpha(alpha: object) -> None:
    with pytest.raises(ValueError, match="finite number between 0 and 1"):
        color_add_alpha("rgb(1,2,3)", alpha)  # type: ignore[arg-type]


def test_color_add_alpha_percentage_output_is_accepted_by_plotly() -> None:
    fillcolor = color_add_alpha("rgb(100%, 0%, 0%)", 0.25)

    assert fillcolor == "rgba(100%, 0%, 0%, 0.25)"
    assert go.Scatter(fillcolor=fillcolor).fillcolor == fillcolor
