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
    assert "train__pert__std" not in COMMON_COLOR_SPECS


def test_setup_colors_accepts_caller_supplied_project_config() -> None:
    config = ColorConfig(
        schema_id="test.experiment.colors",
        schema_version="1",
        colorscales={
            "train__pert__std": "viridis",
            "gain": "thermal",
        },
        color_specs={
            "train__pert__std": ColorscaleSpec(lambda hps: hps.train.pert.std),
            "gain": ColorscaleSpec(lambda hps: hps.gain),
        },
    )
    hps = TreeNamespace(
        train=TreeNamespace(pert=TreeNamespace(std=[0.0, 0.1])),
        gain=[0.0, 0.5],
    )

    colors, colorscales = setup_colors(hps, color_config=config)

    assert colorscales["train__pert__std"] == "viridis"
    assert colorscales["gain"] == "thermal"
    assert set(colors["train__pert__std"].normal) == {0.0, 0.1}
    assert set(colors["gain"].normal) == {0.0, 0.5}
