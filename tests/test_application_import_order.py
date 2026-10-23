"""Fresh-process regressions for application import layering."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module_name",
    (
        "feedbax.config",
        "feedbax.analysis.evaluation",
        "feedbax.bin.analysis",
        "feedbax.orchestration.drivers.runpod",
        "feedbax.plugins",
        "feedbax.web.app",
    ),
)
def test_application_modules_import_in_fresh_process(module_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(repo_root), env.get("PYTHONPATH"))))

    result = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_config_import_does_not_load_application_registries() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(repo_root), env.get("PYTHONPATH"))))
    source = """
import sys
import feedbax.config

assert "feedbax.plugins" not in sys.modules
assert "feedbax.plugins.application" not in sys.modules
assert "feedbax.analysis" not in sys.modules
"""

    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
