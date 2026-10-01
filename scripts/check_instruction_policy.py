"""Verify Feedbax's root instruction policy and downstream contract."""

from __future__ import annotations

from pathlib import Path

from check_downstream_interface_policy import check_policy as check_downstream_policy


ROOT = Path(__file__).resolve().parents[1]
START = "<!-- feedbax-test-policy:start -->"
END = "<!-- feedbax-test-policy:end -->"
POLICY_FILE = ROOT / "AGENTS.md"


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
    extract_policy(POLICY_FILE)
    check_downstream_policy()
    print("Feedbax AGENTS.md policy markers and downstream contract are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
