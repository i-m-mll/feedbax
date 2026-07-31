"""Verify mirrored Feedbax test-policy instruction blocks stay in sync."""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

from check_downstream_interface_policy import check_policy as check_downstream_policy


ROOT = Path(__file__).resolve().parents[1]
START = "<!-- feedbax-test-policy:start -->"
END = "<!-- feedbax-test-policy:end -->"
POLICY_FILES = (ROOT / "AGENTS.md", ROOT / "CLAUDE.md")


def extract_policy(path: Path) -> list[str]:
    """Return the marked policy block from an instruction file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        start = lines.index(START)
        end = lines.index(END)
    except ValueError as exc:
        raise ValueError(f"{path.name} is missing Feedbax test-policy markers") from exc
    if end <= start:
        raise ValueError(f"{path.name} has malformed Feedbax test-policy markers")
    return lines[start + 1 : end]


def main() -> int:
    blocks = {path.name: extract_policy(path) for path in POLICY_FILES}
    agents = blocks["AGENTS.md"]
    claude = blocks["CLAUDE.md"]
    if agents == claude:
        check_downstream_policy()
        print("Feedbax instruction test-policy blocks match.")
        print("Feedbax downstream-stability instruction blocks match policy constants.")
        return 0

    diff = difflib.unified_diff(
        agents,
        claude,
        fromfile="AGENTS.md",
        tofile="CLAUDE.md",
        lineterm="",
    )
    print("Feedbax instruction test-policy blocks differ:", file=sys.stderr)
    for line in diff:
        print(line, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
