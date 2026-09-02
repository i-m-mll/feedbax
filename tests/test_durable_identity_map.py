"""Focused drift gate for the generated durable identity map."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_durable_identity_map_matches_source() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "scripts/check_durable_identity_map.py", "--check"],
        cwd=repo_root,
        check=True,
        timeout=30,
    )
