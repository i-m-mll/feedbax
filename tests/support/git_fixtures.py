from __future__ import annotations

import os
import subprocess
from pathlib import Path


def run_fixture_git(root: Path, *args: str) -> str:
    """Run Git inside a synthetic test repository without host Git configuration."""
    return subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "--no-optional-locks",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "user.name=Feedbax Test Fixture",
            *args,
        ],
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"},
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
