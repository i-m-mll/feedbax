from __future__ import annotations

import math

import pytest

from feedbax.plot.colors import color_add_alpha


@pytest.mark.parametrize(
    ("color", "alpha", "expected"),
    [
        ("rgb(31,119,180)", 0.6, "rgba(31,119,180, 0.6)"),
        ("rgb(31, 119, 180)", 0.6, "rgba(31, 119, 180, 0.6)"),
        ("#2563eb", 0.2, "rgba(37, 99, 235, 0.2)"),
        ("#FFFFFF", 1.0, "rgba(255, 255, 255, 1.0)"),
    ],
)
def test_color_add_alpha_supports_rgb_and_six_digit_hex(
    color: str, alpha: float, expected: str
) -> None:
    assert color_add_alpha(color, alpha) == expected


@pytest.mark.parametrize(
    "color",
    ["#fff", "red", "rgba(1,2,3,0.5)", "rgb(1,2)", "rgb(256,2,3)", "rgb(a,2,3)"],
)
def test_color_add_alpha_rejects_unsupported_or_invalid_colors(color: str) -> None:
    with pytest.raises(ValueError, match="rgb\\(r,g,b\\) or #rrggbb"):
        color_add_alpha(color, 0.5)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, math.nan, math.inf, True, "0.5"])
def test_color_add_alpha_rejects_invalid_alpha(alpha: object) -> None:
    with pytest.raises(ValueError, match="finite number between 0 and 1"):
        color_add_alpha("rgb(1,2,3)", alpha)  # type: ignore[arg-type]
