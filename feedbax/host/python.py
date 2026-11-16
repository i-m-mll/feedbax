"""Command lines for Feedbax-owned Python processes."""

import sys


def python_argv(interpreter: str, *arguments: str) -> list[str]:
    return [interpreter, "-B", *arguments]


def same_python_argv(*arguments: str) -> list[str]:
    isolation = ("-I",) if sys.flags.isolated else ()
    return python_argv(sys.executable, *isolation, *arguments)
