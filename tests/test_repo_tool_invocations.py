from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pyright_entrypoint_does_not_depend_on_console_script_shebang() -> None:
    result = subprocess.run(
        [REPO_ROOT / "scripts" / "pyright.sh", "--version"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.startswith("pyright ")
    assert "scripts/pyright.sh" in (REPO_ROOT / "makefile").read_text(encoding="utf-8")
    assert "scripts/pyright.sh" in (
        REPO_ROOT / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")
