#!/usr/bin/env python
"""Retired training CLI entrypoint."""


def main() -> None:
    raise SystemExit(
        "feedbax-train has been retired. Use executor run specs via "
        "`python -m feedbax execute-training-run-spec <spec>`."
    )


if __name__ == "__main__":
    main()
