from __future__ import annotations

import os
import subprocess
from pathlib import Path


def fixture_git_environment() -> dict[str, str]:
    """Return the process environment with every source of host Git configuration removed.

    Host configuration reaches Git through global and system files and through
    ``GIT_CONFIG_COUNT``/``GIT_CONFIG_KEY_n``/``GIT_CONFIG_PARAMETERS``; any of
    them can turn on commit signing with a key the fixture cannot use.
    """
    env = {
        key: value for key, value in os.environ.items()
        if key != "GIT_CONFIG" and not key.startswith("GIT_CONFIG_")
    }
    return {**env, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}


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
        env=fixture_git_environment(),
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
