from __future__ import annotations

import os
import subprocess
from pathlib import Path


def run_fixture_git(root: Path, *args: str) -> str:
    """Run Git inside a synthetic test repository without host Git configuration."""
    env = {
        key: value for key, value in os.environ.items()
        if key != "GIT_CONFIG" and not key.startswith("GIT_CONFIG_")
    }
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
        env={**env, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"},
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
