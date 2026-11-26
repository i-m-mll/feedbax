"""Classify browser fixture commands and give seed actions their offline cases."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import sys


type _Cases = tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class FileSeed:
    """Returns one or more Studio create-file envelopes."""

    command: Callable[..., object]
    cases: _Cases = ((),)


@dataclass(frozen=True)
class HomeSeed:
    """Writes an ordinary project under the fixture's private Studio home."""

    command: Callable[..., object]
    cases: _Cases = ((),)
    fixed_project: str | None = None
    """A command that accepts just a home uses this project name itself."""


@dataclass(frozen=True)
class FolderSeed:
    """Writes to a project folder passed as an absolute Path."""

    command: Callable[..., object]
    cases: _Cases = ((),)


@dataclass(frozen=True)
class NonSeed:
    """A browser fixture command that does not produce initial authored content."""

    command: Callable[..., object]
    reason: str


type FixtureAction = FileSeed | HomeSeed | FolderSeed | NonSeed


def print_action(actions: Mapping[str, FixtureAction]) -> None:
    """Run a fixture CLI whose first argument names the action."""
    action, *arguments = sys.argv[1:]
    print(json.dumps(actions[action].command(*arguments)))
