from __future__ import annotations

from feedbax.config.namespace import TreeNamespace
from feedbax.plot.color_setup import (
    COLORSCALES,
    COMMON_COLOR_SPECS,
    ColorConfig,
    ColorscaleSpec,
    setup_colors,
)


def test_generic_color_defaults_do_not_ship_project_vocabulary() -> None:
    assert "train__pert__std" not in COLORSCALES
    assert "pert__amp" not in COLORSCALES
    assert "sisu" not in COLORSCALES
    assert "train__pert__std" not in COMMON_COLOR_SPECS
    assert "sisu" not in COMMON_COLOR_SPECS


def test_setup_colors_accepts_caller_supplied_project_config() -> None:
    config = ColorConfig(
        schema_id="test.rlrmp_like.colors",
        schema_version="1",
        colorscales={
            "train__pert__std": "viridis",
            "sisu": "thermal",
        },
        color_specs={
            "train__pert__std": ColorscaleSpec(lambda hps: hps.train.pert.std),
            "sisu": ColorscaleSpec(lambda hps: hps.sisu),
        },
    )
    hps = TreeNamespace(
        train=TreeNamespace(pert=TreeNamespace(std=[0.0, 0.1])),
        sisu=[0.0, 0.5],
    )

    colors, colorscales = setup_colors(hps, color_config=config)

    assert colorscales["train__pert__std"] == "viridis"
    assert colorscales["sisu"] == "thermal"
    assert set(colors["train__pert__std"].normal) == {0.0, 0.1}
    assert set(colors["sisu"].normal) == {0.0, 0.5}
