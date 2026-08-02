"""The managed agent-instructions block: install, update, refuse, never guess.

A tool that rewrites a file a human also writes in has exactly one obligation:
touch only what it owns, and stop rather than improvise when it cannot tell what
it owns. These tests are that obligation, stated as a matrix — first install,
update, no-op, local edit, unknown schema, future template, duplicated and
unpaired markers, and every arrangement of the two conventional agent files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from feedbax.governance import agent_instructions as ai
from feedbax.governance.cli import run_instructions

HUMAN_ABOVE = "# My project\n\nNotes I wrote myself.\n"
HUMAN_BELOW = "\n## My own section\n\nMore notes.\n"


def _block(
    *,
    body: str | None = None,
    template: int | None = None,
    schema: str | None = None,
    sha256: str | None = None,
    generated_by: str = "feedbax-0.0.1",
) -> str:
    text = ai.template_body() if body is None else body
    header = "\n".join(
        (
            f"schema={schema or ai.AGENT_INSTRUCTIONS_SCHEMA_VERSION}",
            f"template={ai.AGENT_INSTRUCTIONS_TEMPLATE_VERSION if template is None else template}",
            f"generated-by={generated_by}",
            f"sha256={sha256 or ai.body_sha256(text)}",
        )
    )
    return (
        f"{ai.BLOCK_START_MARKER}\n{header}\n{ai.BLOCK_HEADER_END}\n{text}{ai.BLOCK_END_MARKER}\n"
    )


# --- the template ships and is the only source of the body -------------------


def test_the_template_is_shipped_as_package_data() -> None:
    from importlib import resources

    resource = resources.files(ai.AGENT_INSTRUCTIONS_TEMPLATE_PACKAGE).joinpath(
        ai.AGENT_INSTRUCTIONS_TEMPLATE_NAME
    )

    assert resource.is_file()
    assert ai.template_body() == resource.read_text(encoding="utf-8")


def test_the_template_covers_every_section_the_contract_requires() -> None:
    body = ai.template_body()

    for heading in (
        "The mental model",
        "Project map",
        "The experiment model",
        "Naming: a name states a concept, never a value",
        "The residence boundary",
        "What a science plugin may be",
        "Command orientation",
        "Science authorization",
        "Durable formats migrate or reject",
        "Generated custody is hands-off",
        "When something does not fit",
    ):
        assert heading in body, heading


def test_the_template_names_no_particular_project() -> None:
    """Framework text is generic text; a real project name here would be a bug."""
    body = ai.template_body().lower()

    for forbidden in ("rlrmp", "quillon", "spinnaker"):
        assert forbidden not in body, forbidden


def test_the_block_header_states_its_own_identity() -> None:
    rendered = ai.render_block()
    parsed = ai.parse_block(rendered)

    assert parsed is not None
    assert parsed.schema == ai.AGENT_INSTRUCTIONS_SCHEMA_VERSION
    assert parsed.template_version == ai.AGENT_INSTRUCTIONS_TEMPLATE_VERSION
    assert parsed.declared_sha256 == ai.body_sha256(ai.template_body())
    assert parsed.body == ai.template_body()
    assert ai.classify(parsed) is ai.BlockStatus.FRESH


# --- install and update ------------------------------------------------------


def test_a_first_install_prepends_exactly_one_block(tmp_path: Path) -> None:
    target = tmp_path / "AGENTS.md"
    target.write_text(HUMAN_ABOVE)

    ai.install(tmp_path, target=target)

    text = target.read_text()
    assert text.count(ai.BLOCK_START_MARKER) == 1
    assert text.count(ai.BLOCK_END_MARKER) == 1
    assert text.endswith(HUMAN_ABOVE)


def test_an_update_replaces_only_the_bytes_between_the_markers(tmp_path: Path) -> None:
    target = tmp_path / "AGENTS.md"
    target.write_text(HUMAN_ABOVE + _block(body="Stale text.\n") + HUMAN_BELOW)

    ai.install(tmp_path, target=target)

    text = target.read_text()
    assert text.startswith(HUMAN_ABOVE)
    assert text.endswith(HUMAN_BELOW)
    assert "Stale text." not in text
    assert ai.classify(ai.parse_block(text)) is ai.BlockStatus.FRESH


def test_reinstalling_a_fresh_block_changes_no_bytes(tmp_path: Path) -> None:
    target = tmp_path / "AGENTS.md"
    ai.install(tmp_path, target=target)
    before = target.read_bytes()

    report = ai.install(tmp_path, target=target)

    assert target.read_bytes() == before
    assert [outcome.action for outcome in report.outcomes] == ["unchanged"]


def test_a_newer_feedbax_version_alone_does_not_rewrite_the_block(tmp_path: Path) -> None:
    """Freshness is the template hash, not the version that happened to write it."""
    target = tmp_path / "AGENTS.md"
    target.write_text(_block(generated_by="feedbax-0.0.1"))

    report = ai.install(tmp_path, target=target)

    assert [outcome.action for outcome in report.outcomes] == ["unchanged"]
    assert "feedbax-0.0.1" in target.read_text()


def test_a_dry_run_writes_nothing(tmp_path: Path) -> None:
    target = tmp_path / "AGENTS.md"

    report = ai.install(tmp_path, target=target, dry_run=True)

    assert not target.exists()
    assert report.dry_run and report.changed
    assert "would created" in report.describe()


def test_standalone_mode_writes_a_whole_generated_fragment(tmp_path: Path) -> None:
    report = ai.install(tmp_path, mode="standalone")

    fragment = tmp_path / ai.STANDALONE_FRAGMENT_PATH
    assert fragment.read_text() == ai.render_block()
    assert [outcome.action for outcome in report.outcomes] == ["created"]
    assert [outcome.action for outcome in ai.install(tmp_path, mode="standalone").outcomes] == [
        "unchanged"
    ]


def test_an_unknown_mode_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ai.AgentInstructionsError, match="unknown instructions mode"):
        ai.install(tmp_path, mode="freestyle")


# --- refusal matrix ----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (_block() + _block(), "exactly one"),
        (ai.BLOCK_START_MARKER + "\nschema=x\n", "unpaired"),
        (ai.BLOCK_END_MARKER + "\n", "unpaired"),
        (ai.BLOCK_END_MARKER + "\n" + _block().replace(ai.BLOCK_END_MARKER, ""), "precedes"),
    ],
)
def test_unsafe_markers_refuse_without_writing(
    tmp_path: Path, text: str, match: str
) -> None:
    target = tmp_path / "AGENTS.md"
    target.write_text(text)

    with pytest.raises(ai.AgentInstructionsError, match=match):
        ai.install(tmp_path, target=target)
    assert target.read_text() == text


def test_an_unknown_schema_refuses_rather_than_being_reinterpreted(tmp_path: Path) -> None:
    target = tmp_path / "AGENTS.md"
    text = _block(schema="someone_else.agent_instructions.v1")
    target.write_text(text)

    with pytest.raises(ai.AgentInstructionsError, match="does not know"):
        ai.install(tmp_path, target=target)
    assert target.read_text() == text


def test_a_missing_header_field_refuses(tmp_path: Path) -> None:
    target = tmp_path / "AGENTS.md"
    text = _block().replace("generated-by=feedbax-0.0.1\n", "")
    target.write_text(text)

    with pytest.raises(ai.AgentInstructionsError, match="exactly"):
        ai.install(tmp_path, target=target)
    assert target.read_text() == text


def test_a_non_integer_template_version_refuses(tmp_path: Path) -> None:
    target = tmp_path / "AGENTS.md"
    current = f"template={ai.AGENT_INSTRUCTIONS_TEMPLATE_VERSION}"
    target.write_text(_block().replace(current, "template=one"))

    with pytest.raises(ai.AgentInstructionsError, match="not an integer"):
        ai.install(tmp_path, target=target)


def test_an_older_feedbax_never_downgrades_a_newer_block(tmp_path: Path) -> None:
    target = tmp_path / "AGENTS.md"
    text = _block(body="Written by a future Feedbax.\n", template=99)
    target.write_text(text)

    with pytest.raises(ai.AgentInstructionsError, match="newer than this Feedbax"):
        ai.install(tmp_path, target=target)
    assert target.read_text() == text


def test_a_locally_edited_block_is_restored_not_preserved(tmp_path: Path) -> None:
    target = tmp_path / "AGENTS.md"
    edited = _block().replace("The mental model", "My own model")
    target.write_text(edited)

    ai.install(tmp_path, target=target)

    assert ai.classify(ai.parse_block(target.read_text())) is ai.BlockStatus.FRESH


# --- the two conventional agent files ---------------------------------------


def test_a_new_repository_gets_one_real_file_and_one_relative_link(tmp_path: Path) -> None:
    ai.install(tmp_path)

    agents, claude = (tmp_path / name for name in ai.AGENT_FILE_NAMES)
    assert agents.is_file() and not agents.is_symlink()
    assert claude.is_symlink() and Path(claude.readlink()) == Path("AGENTS.md")


def test_an_existing_symlink_is_followed_not_repointed(tmp_path: Path) -> None:
    agents = tmp_path / "AGENTS.md"
    agents.write_text(HUMAN_ABOVE)
    claude = tmp_path / "CLAUDE.md"
    claude.symlink_to("AGENTS.md")

    report = ai.install(tmp_path)

    assert [outcome.path for outcome in report.outcomes] == [agents.resolve()]
    assert Path(claude.readlink()) == Path("AGENTS.md")
    assert agents.read_text().endswith(HUMAN_ABOVE)


def test_a_missing_second_file_becomes_a_link_to_the_one_that_exists(
    tmp_path: Path,
) -> None:
    claude = tmp_path / "CLAUDE.md"
    claude.write_text(HUMAN_ABOVE)

    ai.install(tmp_path)

    agents = tmp_path / "AGENTS.md"
    assert agents.is_symlink() and Path(agents.readlink()) == Path("CLAUDE.md")
    assert claude.read_text().endswith(HUMAN_ABOVE)


def test_two_divergent_regular_files_are_both_written_and_neither_is_merged(
    tmp_path: Path,
) -> None:
    agents = tmp_path / "AGENTS.md"
    claude = tmp_path / "CLAUDE.md"
    agents.write_text("# Agents-only notes\n")
    claude.write_text("# Claude-only notes\n")

    ai.install(tmp_path)

    assert agents.read_text().endswith("# Agents-only notes\n")
    assert claude.read_text().endswith("# Claude-only notes\n")
    assert not agents.is_symlink() and not claude.is_symlink()
    assert "Claude-only" not in agents.read_text()
    assert "Agents-only" not in claude.read_text()


def test_two_identical_regular_files_are_still_not_converted_to_a_link(
    tmp_path: Path,
) -> None:
    """Conversion is a deliberate reconciliation, never a side effect of install."""
    for name in ai.AGENT_FILE_NAMES:
        (tmp_path / name).write_text(HUMAN_ABOVE)

    ai.install(tmp_path)

    assert not (tmp_path / "CLAUDE.md").is_symlink()


def test_a_broken_symlink_refuses_rather_than_being_replaced(tmp_path: Path) -> None:
    claude = tmp_path / "CLAUDE.md"
    claude.symlink_to("AGENTS.md")

    with pytest.raises(ai.AgentInstructionsError, match="broken symlink"):
        ai.install(tmp_path)
    assert claude.is_symlink()


# --- check -------------------------------------------------------------------


def test_a_fresh_repository_checks_clean(tmp_path: Path) -> None:
    ai.install(tmp_path)

    report = ai.check(tmp_path)

    assert report.status is ai.BlockStatus.FRESH
    assert report.exit_code == 0


@pytest.mark.parametrize(
    ("text", "status"),
    [
        (None, ai.BlockStatus.MISSING),
        ("# nothing generated here\n", ai.BlockStatus.MISSING),
        (_block(body="Older revision.\n", template=0), ai.BlockStatus.STALE),
        (_block(body="Edited.\n"), ai.BlockStatus.EDITED),
        (_block(sha256="0" * 64), ai.BlockStatus.EDITED),
        (_block(template=99), ai.BlockStatus.FUTURE),
        (_block() + _block(), ai.BlockStatus.MALFORMED),
        (_block(schema="other.v1"), ai.BlockStatus.MALFORMED),
    ],
)
def test_every_unhealthy_state_gets_its_own_nonzero_code(
    tmp_path: Path, text: str | None, status: ai.BlockStatus
) -> None:
    target = tmp_path / "AGENTS.md"
    if text is not None:
        target.write_text(text)

    report = ai.check(tmp_path, target=target)

    assert report.status is status
    assert report.exit_code == ai.BLOCK_STATUS_EXIT_CODES[status]
    assert report.exit_code != 0


def test_the_unhealthy_exit_codes_are_all_distinct() -> None:
    codes = list(ai.BLOCK_STATUS_EXIT_CODES.values())

    assert len(set(codes)) == len(codes)
    assert ai.BLOCK_STATUS_EXIT_CODES[ai.BlockStatus.FRESH] == 0
    assert 0 not in [code for status, code in ai.BLOCK_STATUS_EXIT_CODES.items()
                     if status is not ai.BlockStatus.FRESH]


def test_a_broken_link_is_reported_rather_than_crashing(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").symlink_to("AGENTS.md")

    report = ai.check(tmp_path)

    assert report.status is ai.BlockStatus.MALFORMED
    assert "broken symlink" in report.describe()


def test_check_reports_the_worst_state_across_both_files(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text(_block())
    (tmp_path / "CLAUDE.md").write_text(_block(template=99))

    report = ai.check(tmp_path)

    assert report.status is ai.BlockStatus.FUTURE


def test_check_never_writes(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text(_block(body="Edited.\n"))
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    ai.check(tmp_path)

    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


# --- the command surface -----------------------------------------------------


def test_the_install_command_reports_and_succeeds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_instructions(["install", str(tmp_path)]) == 0

    assert "created" in capsys.readouterr().out
    assert (tmp_path / "AGENTS.md").is_file()


def test_the_install_command_refuses_with_exit_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "AGENTS.md"
    target.write_text(_block(template=99))

    assert run_instructions(["install", str(tmp_path)]) == 2

    assert "newer than this Feedbax" in capsys.readouterr().err
    assert target.read_text() == _block(template=99)


def test_the_check_command_returns_the_state_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_instructions(["check", str(tmp_path)]) == (
        ai.BLOCK_STATUS_EXIT_CODES[ai.BlockStatus.MISSING]
    )
    assert "missing" in capsys.readouterr().err

    run_instructions(["install", str(tmp_path)])
    capsys.readouterr()

    assert run_instructions(["check", str(tmp_path)]) == 0
    assert "fresh" in capsys.readouterr().out


def test_the_commands_default_to_the_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    assert run_instructions(["install"]) == 0
    assert run_instructions(["check"]) == 0


def test_the_unified_entry_point_routes_the_family(tmp_path: Path) -> None:
    from feedbax import cli

    assert cli.main(["instructions", "install", str(tmp_path), "--dry-run"]) == 0
    assert not (tmp_path / "AGENTS.md").exists()
    assert "instructions" in cli.usage()


def test_init_leaves_the_instructions_check_clean(tmp_path: Path) -> None:
    """The two commands agree: what `init` installs, `check` calls fresh."""
    from feedbax.governance.project_init import initialize

    root = tmp_path / "cadence-study"
    root.mkdir()
    initialize(root)

    assert ai.check(root).exit_code == 0
