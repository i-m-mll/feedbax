import importlib
import importlib.util
import subprocess
import sys
import textwrap
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


def test_analysis_import_does_not_require_persistence_or_viz_extras():
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        blocked = {"alembic", "matplotlib", "polars", "pyexiv2"}

        class BlockOptionalImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.partition(".")[0] in blocked:
                    raise ModuleNotFoundError(fullname)
                return None

        sys.meta_path.insert(0, BlockOptionalImports())
        import feedbax.analysis
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
