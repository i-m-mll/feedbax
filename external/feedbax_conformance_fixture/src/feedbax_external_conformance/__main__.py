"""Command-line entrypoint for the external conformance fixture."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import run_fixture


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args(argv)
    result = run_fixture(source_root=args.source_root)
    payload = result.model_dump_json(indent=2) + "\n"
    if args.result is None:
        print(payload, end="")
    else:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(payload, encoding="utf-8")
        print(args.result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
