"""Import-order coverage for the explicit application bootstrap boundary."""

from __future__ import annotations

import asyncio
import importlib

import pytest

from feedbax.plugins import (
    PluginDeclaration,
    PluginRegistration,
)
from feedbax.plugins import bootstrap as bootstrap_module
from feedbax.plugins.composition import compose_application


def test_importing_plugins_performs_no_discovery(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        bootstrap_module.importlib.metadata,
        "entry_points",
        lambda **_kwargs: calls.append("discover") or (),
    )

    importlib.reload(importlib.import_module("feedbax.plugins"))

    assert calls == []


def test_typed_compose_discovers_once_after_import(monkeypatch) -> None:
    calls: list[str] = []
    registration = PluginRegistration(
        PluginDeclaration("tests.import_order", "1", 1),
        lambda _context: calls.append("register"),
    )

    class EntryPoint:
        name = "import-order"
        value = "tests.import_order:PLUGIN_REGISTRATION"
        dist = None

        def load(self):
            calls.append("load")
            return registration

    monkeypatch.setattr(
        bootstrap_module.importlib.metadata,
        "entry_points",
        lambda **_kwargs: (EntryPoint(),),
    )

    state = asyncio.run(compose_application(local_component_source=None))

    assert calls == ["load", "register"]
    assert state.provenance[0].plugin_id == "tests.import_order"


@pytest.mark.parametrize(
    "module_name",
    (
        "feedbax.__main__",
        "feedbax.bin.analysis",
        "feedbax.bin.figure",
        "feedbax.bin.orchestrate",
        "feedbax.bin.provider",
        "feedbax.bin.run",
        "feedbax.bin.studio_pipeline",
        "feedbax.web.api.components",
        "feedbax.web.app",
        "feedbax.web.worker.app",
    ),
)
def test_public_composition_roots_use_the_shared_composer(module_name: str) -> None:
    module = importlib.import_module(module_name)

    assert module.compose_application is compose_application


def test_config_globals_are_populated_without_plugin_side_effects() -> None:
    config = importlib.import_module("feedbax.config")

    assert config.STRINGS.hps_level_label_sep == "__"
    assert hasattr(config.PATHS, "base")
