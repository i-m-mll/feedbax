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


@pytest.mark.parametrize(
    "module_name",
    ["feedbax.analysis", "feedbax.analysis.aligned", "feedbax.analysis.execution"],
)
def test_analysis_import_does_not_require_persistence_or_viz_extras(module_name: str):
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        blocked = {"matplotlib", "polars", "pyexiv2"}

        class BlockOptionalImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.partition(".")[0] in blocked:
                    raise ModuleNotFoundError(fullname)
                return None

        sys.meta_path.insert(0, BlockOptionalImports())
        import importlib
        importlib.import_module(sys.argv[1])
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, module_name],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_project_onto_direction_does_not_require_matplotlib():
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockMatplotlibImport(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.partition(".")[0] == "matplotlib":
                    raise ModuleNotFoundError(fullname)
                return None

        sys.meta_path.insert(0, BlockMatplotlibImport())
        import jax.numpy as jnp
        from feedbax.analysis.aligned import project_onto_direction

        var = jnp.array([[[[3.0, 4.0]]]])
        direction = jnp.array([[1.0, 0.0]])
        projected = project_onto_direction(var, direction)
        expected = jnp.array([[[[3.0, 4.0]]]])
        assert jnp.allclose(projected, expected)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
