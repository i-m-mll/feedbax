"""The unified ``feedbax`` console entry point routes; it never reinterprets.

One thing to type, one ``--help`` that names everything, and no second command
inventory that can quietly disagree with the engine's. These tests hold that:
every advertised command reaches exactly the main that implements it, the
argument list arrives unmodified, the delegate's exit code comes back unchanged,
and the engine inventory is read from the engine parser rather than copied.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Sequence

import pytest

from feedbax import cli

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clear_engine_command_cache():
    cli.engine_commands.cache_clear()
    yield
    cli.engine_commands.cache_clear()


def _console_scripts() -> dict[str, str]:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return payload["project"]["scripts"]


# --- the entry point exists and is the one advertised -----------------------


def test_the_package_installs_a_unified_feedbax_console_script() -> None:
    assert _console_scripts()["feedbax"] == "feedbax.cli:main"


def test_every_delegated_command_still_has_its_own_console_script() -> None:
    """The per-command scripts are guaranteed surface, so they stay installed."""
    scripts = _console_scripts()

    for command in cli.DELEGATED_COMMANDS:
        assert scripts[f"feedbax-{command.name}"] == f"{command.module}:{command.attribute}"


def test_the_absorbed_commands_are_exactly_the_six_named_scripts() -> None:
    assert tuple(command.name for command in cli.DELEGATED_COMMANDS) == (
        "run",
        "analysis",
        "figure",
        "train",
        "provider",
        "orchestrate",
    )


# --- dispatch ---------------------------------------------------------------


def _capture(monkeypatch: pytest.MonkeyPatch, name: str, result: Any) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_main(argv: Sequence[str] | None = None) -> Any:
        calls.append(list(argv or ()))
        return result

    command = cli.DELEGATED_BY_NAME[name]
    monkeypatch.setattr(cli.DelegatedCommand, "entrypoint", lambda self: fake_main)
    assert command.name == name
    return calls


@pytest.mark.parametrize(
    "name", ["run", "analysis", "figure", "train", "provider", "orchestrate"]
)
def test_a_delegated_command_receives_its_own_arguments_verbatim(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    calls = _capture(monkeypatch, name, 0)

    code = cli.main([name, "sub", "--flag", "value", "--", "-x"])

    assert code == 0
    assert calls == [["sub", "--flag", "value", "--", "-x"]]


def test_a_delegated_command_with_no_arguments_gets_an_empty_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture(monkeypatch, "provider", 0)

    assert cli.main(["provider"]) == 0
    assert calls == [[]]


@pytest.mark.parametrize("result", [0, 1, 2, 7, None])
def test_a_delegate_exit_code_is_returned_unchanged(
    monkeypatch: pytest.MonkeyPatch, result: int | None
) -> None:
    _capture(monkeypatch, "analysis", result)

    assert cli.main(["analysis", "run"]) == (0 if result is None else result)


def test_a_delegate_returning_a_non_exit_code_is_a_programming_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture(monkeypatch, "figure", "fine")

    with pytest.raises(TypeError, match="non-exit-code value"):
        cli.main(["figure", "resolve"])


def test_an_engine_command_reaches_the_engine_main_with_the_command_included(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import feedbax.__main__ as engine

    calls: list[list[str]] = []
    monkeypatch.setattr(engine, "main", lambda argv: calls.append(list(argv)) or 0)

    assert cli.main(["preflight-experiment-envelope", "e.json", "--repo-root", "."]) == 0
    assert calls == [["preflight-experiment-envelope", "e.json", "--repo-root", "."]]


def test_the_retired_train_command_refuses_by_its_unified_name() -> None:
    with pytest.raises(SystemExit, match="`feedbax train` has been retired"):
        cli.main(["train"])


# --- one inventory, not two -------------------------------------------------


def test_the_engine_inventory_is_read_from_the_engine_parser() -> None:
    from feedbax.__main__ import engine_command_names

    assert cli.engine_commands() == engine_command_names()
    assert "preflight-experiment-envelope" in cli.engine_commands()
    assert "check-project-science-surface" in cli.engine_commands()


def test_no_engine_command_collides_with_a_delegated_command() -> None:
    assert not set(cli.engine_commands()) & set(cli.DELEGATED_BY_NAME)


def test_a_command_added_to_the_engine_parser_becomes_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The unified surface follows the engine; it does not need its own edit."""
    import feedbax.__main__ as engine

    monkeypatch.setattr(engine, "engine_command_names", lambda: ("invented-engine-command",))
    cli.engine_commands.cache_clear()
    calls: list[list[str]] = []
    monkeypatch.setattr(engine, "main", lambda argv: calls.append(list(argv)) or 0)

    assert cli.main(["invented-engine-command", "--x"]) == 0
    assert calls == [["invented-engine-command", "--x"]]


# --- usage and refusal ------------------------------------------------------


def test_help_lists_every_delegated_and_engine_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["--help"]) == 0

    out = capsys.readouterr().out
    for command in cli.DELEGATED_COMMANDS:
        assert f"  {command.name}" in out
    for name in cli.engine_commands():
        assert name in out


@pytest.mark.parametrize("flag", ["-h", "--help", "help"])
def test_every_help_spelling_succeeds(
    capsys: pytest.CaptureFixture[str], flag: str
) -> None:
    assert cli.main([flag]) == 0
    assert "usage: feedbax <command>" in capsys.readouterr().out


def test_no_command_is_a_usage_error_on_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == cli.EXIT_USAGE

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "usage: feedbax <command>" in captured.err


def test_an_unknown_command_is_refused_rather_than_guessed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["analize", "run"]) == cli.EXIT_USAGE

    captured = capsys.readouterr()
    assert "unknown command 'analize'" in captured.err
    assert captured.out == ""
