import importlib
import importlib.util
import tomllib
from pathlib import Path

import pytest


def test_feedbax_misc_is_not_a_root_compatibility_module():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("feedbax.misc")


def test_legacy_dashboard_package_is_retired():
    assert importlib.util.find_spec("feedbax.dashboard") is None
    assert importlib.util.find_spec("feedbax.bin.dashboard") is None


def test_legacy_dashboard_extra_is_retired():
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))

    optional_dependencies = pyproject["project"]["optional-dependencies"]
    assert "dashboard" not in optional_dependencies
    assert all(
        "dash" not in dependency.lower()
        for dependency in optional_dependencies["all"]
    )
