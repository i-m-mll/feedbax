"""Detect declarations that cannot be supplied by Feedbax's installed entry point."""

from importlib.util import find_spec
import os
import subprocess
import sys
from textwrap import dedent

import pytest
from tests.support.hang import HANG_SECONDS


@pytest.mark.skipif(
    any(find_spec(package) is None for package in ("fastapi", "mujoco", "polars")),
    reason="The whole Feedbax tree imports its optional web, MuJoCo, and visualization adapters",
)
def test_concrete_declarations_live_in_builtins(tmp_path):
    script = dedent("""
        import feedbax
        from importlib import import_module
        from inspect import isabstract, isclass, isfunction
        from pkgutil import walk_packages

        from feedbax import Component
        from feedbax.contracts.portable import ComponentOwner
        from feedbax.plugins.declarations import callable_declaration
        from feedbax.tasks.registry import PythonTask

        def import_failed(name):
            raise RuntimeError(f"Cannot inspect Feedbax declaration placement in {name}")

        misplaced = set()
        for item in walk_packages(feedbax.__path__, "feedbax.", onerror=import_failed):
            module = import_module(item.name)
            if module.__name__ == "feedbax.builtins" or module.__name__.startswith("feedbax.builtins."):
                continue
            for value in tuple(vars(module).values()):
                if not (isclass(value) or isfunction(value)) or value.__module__ != module.__name__:
                    continue
                if isclass(value):
                    if value in (ComponentOwner, Component, PythonTask):
                        continue
                    declared = issubclass(value, (ComponentOwner, PythonTask)) and not isabstract(value)
                else:
                    declared = callable_declaration(value) is not None
                if declared:
                    misplaced.add(f"{module.__name__}.{value.__qualname__}")
        assert not misplaced, "\\n".join(sorted(misplaced))
        """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=os.environ | {
            "HOME": str(tmp_path),
            "FEEDBAX_WEB_DATA": str(tmp_path / "web"),
            "FEEDBAX_HOST_ROOT": str(tmp_path / "host"),
            "FEEDBAX_TRAJECTORIES_DIR": str(tmp_path / "trajectories"),
        },
        text=True, capture_output=True, timeout=HANG_SECONDS,
    )
    assert result.returncode == 0, result.stderr
