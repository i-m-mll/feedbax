#!/usr/bin/env python
"""Retired training CLI entrypoint."""

from __future__ import annotations

from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Refuse a retired command by name and point at the one that replaced it."""
    del argv
    raise SystemExit(
        "`feedbax train` has been retired. Use executor run specs via "
        "`feedbax execute-training-run-spec <spec>`."
    )


if __name__ == "__main__":
    main()
