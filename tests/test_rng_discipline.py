from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_rng_discipline_static_gate_passes() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts/check_rng_discipline.py")],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
