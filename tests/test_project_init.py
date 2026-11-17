"""Transactional managed-project initialization and repository scaffolding."""

from __future__ import annotations

import json
import os
import subprocess
import tomllib
from pathlib import Path

import pytest

from feedbax.contracts.project_io import load_text_project
from feedbax.governance import project_init
from feedbax.governance.agent_instructions import (
    AGENT_FILE_NAMES,
    BlockStatus,
    classify,
    parse_block,
)
from feedbax.governance.cli import run_init
from feedbax.governance.science_surface import POLICY_SCHEMA_VERSION


PROJECT_DECLARATION_FILENAME = "spinnaker-study.feedbax"


def _project(tmp_path: Path, name: str = "spinnaker-study") -> Path:
    root = tmp_path / name
    root.mkdir()
    return root


def _entries(root: Path) -> set[str]:
    return {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }


# --- the exact inventory ----------------------------------------------------


def test_a_new_project_gets_exactly_the_project_files(tmp_path: Path) -> None:
    root = _project(tmp_path)

    report = project_init.initialize(root)

    assert report.exit_code == project_init.EXIT_OK
    assert _entries(root) == set(project_init.entry_paths(report.project, report.package))
    assert set(project_init.entry_paths("spinnaker-study", "spinnaker_study")) == {
        PROJECT_DECLARATION_FILENAME,
        ".gitignore",
        "ci/feedbax-science-surface.toml",
        "pyproject.toml",
        "src/spinnaker_study/__init__.py",
        "AGENTS.md",
    }


def test_names_are_derived_from_the_directory(tmp_path: Path) -> None:
    root = _project(tmp_path, "wave-cadence")

    report = project_init.initialize(root)

    assert report.project == "wave-cadence"
    assert report.package == "wave_cadence"


def test_explicit_names_override_the_derivation(tmp_path: Path) -> None:
    root = _project(tmp_path)

    report = project_init.initialize(root, project="quillon", package="quill")

    assert (report.project, report.package) == ("quillon", "quill")
    assert (root / "src/quill/__init__.py").exists()


def test_nothing_speculative_is_created(tmp_path: Path) -> None:
    root = _project(tmp_path)

    project_init.initialize(root)

    for absent in ("results", "documents", "steps", ".git"):
        assert not (root / absent).exists(), absent
    assert (root / ".gitignore").read_text(encoding="utf-8") == "/results/\n"
    assert (root / "src" / "spinnaker_study" / "__init__.py").read_text().count("\n") == 1


def test_the_output_rule_is_root_anchored(tmp_path: Path) -> None:
    root = _project(tmp_path)
    project_init.initialize(root)
    (root / "results").mkdir()
    (root / "results" / "lock.json").touch()
    (root / "web" / "src" / "results").mkdir(parents=True)
    (root / "web" / "src" / "results" / "contract.ts").touch()
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    subprocess.run(["git", "init", "-q", root], check=True, env=env)

    root_output = subprocess.run(
        ["git", "-C", root, "check-ignore", "--no-index", "-q", "results/lock.json"],
        check=False,
        env=env,
    )
    nested_source = subprocess.run(
        [
            "git",
            "-C",
            root,
            "check-ignore",
            "--no-index",
            "-q",
            "web/src/results/contract.ts",
        ],
        check=False,
        env=env,
    )

    assert root_output.returncode == 0
    assert nested_source.returncode == 1


# --- what it writes is what Feedbax reads -----------------------------------


def test_the_written_declaration_loads_through_the_real_loader(tmp_path: Path) -> None:
    root = _project(tmp_path)

    project_init.initialize(root)

    text = load_text_project(root)
    assert text.configuration.project == "spinnaker-study"
    assert text.configuration.store == "results"
    assert text.project.documents == {}
    assert text.project.steps == {}


def test_the_written_policy_authorizes_only_the_marker_init_created(tmp_path: Path) -> None:
    root = _project(tmp_path)

    project_init.initialize(root)

    policy = tomllib.loads((root / project_init.DEFAULT_POLICY_REF).read_text())
    assert policy["schema_version"] == POLICY_SCHEMA_VERSION
    assert policy["source_roots"] == ["src"]
    assert policy["banned_paths"] == []
    assert [entry["path"] for entry in policy["allowed_file"]] == [
        "src/spinnaker_study/__init__.py"
    ]
    assert policy["allowed_file"][0]["symbols"] == []


def test_the_policy_never_inventories_source_it_did_not_create(tmp_path: Path) -> None:
    """An existing repository is not silently authorized by initializing it."""
    root = _project(tmp_path)
    (root / "pyproject.toml").write_text('[project]\nname = "already-here"\n')
    (root / "src" / "already_here").mkdir(parents=True)
    (root / "src" / "already_here" / "engine.py").write_text("def compile_everything():\n    ...\n")

    project_init.initialize(root)

    policy = tomllib.loads((root / project_init.DEFAULT_POLICY_REF).read_text())
    assert "allowed_file" not in policy


def test_the_written_pyproject_is_a_minimal_installable_project(tmp_path: Path) -> None:
    root = _project(tmp_path)

    project_init.initialize(root)

    document = tomllib.loads((root / "pyproject.toml").read_text())
    assert document["project"]["name"] == "spinnaker-study"
    assert any(item.startswith("feedbax") for item in document["project"]["dependencies"])
    assert document["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "src/spinnaker_study"
    ]


def test_the_agent_files_are_one_real_agents_file(tmp_path: Path) -> None:
    root = _project(tmp_path)

    project_init.initialize(root)

    agents, claude = (root / name for name in AGENT_FILE_NAMES)
    assert agents.is_file() and not agents.is_symlink()
    assert not claude.exists() and not claude.is_symlink()
    assert classify(parse_block(agents.read_text())) is BlockStatus.FRESH


# --- rerun: validate, never rewrite -----------------------------------------


def test_a_second_run_changes_no_durable_bytes(tmp_path: Path) -> None:
    root = _project(tmp_path)
    project_init.initialize(root)
    before = {path: (root / path).read_bytes() for path in _entries(root)}

    report = project_init.initialize(root)

    assert report.exit_code == project_init.EXIT_OK
    assert {path: (root / path).read_bytes() for path in _entries(root)} == before


def test_a_second_run_reports_durable_files_as_validated(tmp_path: Path) -> None:
    root = _project(tmp_path)
    project_init.initialize(root)

    report = project_init.initialize(root)

    actions = {outcome.path: outcome.action for outcome in report.outcomes}
    assert actions[PROJECT_DECLARATION_FILENAME] == "validated"
    assert actions[project_init.DEFAULT_POLICY_REF] == "validated"
    assert actions[AGENT_FILE_NAMES[0]] == "unchanged"


def test_a_customized_declaration_is_kept_not_reset(tmp_path: Path) -> None:
    root = _project(tmp_path)
    project_init.initialize(root)
    document = json.loads((root / PROJECT_DECLARATION_FILENAME).read_text())
    document["store"] = "studies"
    (root / PROJECT_DECLARATION_FILENAME).write_text(json.dumps(document, indent=2) + "\n")

    project_init.initialize(root)

    assert load_text_project(root).configuration.store == "studies"


def test_an_existing_gitignore_keeps_its_bytes_and_gains_the_declared_rule(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    project_init.initialize(root)
    original = b"*.secret\r\n# project rule with opaque bytes: \xff"
    (root / ".gitignore").write_bytes(original)
    declaration = json.loads((root / PROJECT_DECLARATION_FILENAME).read_text())
    declaration["store"] = "build/experiment-results"
    (root / PROJECT_DECLARATION_FILENAME).write_text(
        json.dumps(declaration, indent=2) + "\n", encoding="utf-8"
    )

    project_init.initialize(root)

    assert (root / ".gitignore").read_bytes() == original + b"\n/build/experiment-results/\n"


def test_a_second_run_does_not_duplicate_the_output_ignore_rule(tmp_path: Path) -> None:
    root = _project(tmp_path)
    project_init.initialize(root)

    project_init.initialize(root)

    assert (root / ".gitignore").read_text(encoding="utf-8") == "/results/\n"


def test_an_existing_python_project_keeps_its_own_packaging(tmp_path: Path) -> None:
    root = _project(tmp_path)
    pyproject = root / "pyproject.toml"
    pyproject.write_text('[project]\nname = "already-here"\nversion = "1.2.3"\n')

    report = project_init.initialize(root)

    actions = {outcome.path: outcome.action for outcome in report.outcomes}
    assert actions["pyproject.toml"] == "skipped"
    assert actions[f"src/{report.package}/__init__.py"] == "skipped"
    assert tomllib.loads(pyproject.read_text())["project"]["version"] == "1.2.3"
    assert not (root / "src").exists()


# --- refusal and transactionality -------------------------------------------


def test_an_invalid_declaration_refuses_the_whole_run(tmp_path: Path) -> None:
    root = _project(tmp_path)
    (root / PROJECT_DECLARATION_FILENAME).write_text('{"schema_id": "someone.else"}\n')

    report = project_init.initialize(root)

    assert report.exit_code == project_init.EXIT_CONFLICT
    assert [outcome.path for outcome in report.conflicts] == [PROJECT_DECLARATION_FILENAME]
    assert _entries(root) == {PROJECT_DECLARATION_FILENAME}


def test_an_unsupported_policy_version_refuses_the_whole_run(tmp_path: Path) -> None:
    root = _project(tmp_path)
    policy = root / project_init.DEFAULT_POLICY_REF
    policy.parent.mkdir(parents=True)
    policy.write_text("schema_version = 99\n")

    report = project_init.initialize(root)

    assert report.exit_code == project_init.EXIT_CONFLICT
    assert _entries(root) == {project_init.DEFAULT_POLICY_REF}


def test_a_dry_run_writes_nothing_and_reports_everything(tmp_path: Path) -> None:
    root = _project(tmp_path)

    report = project_init.initialize(root, dry_run=True)

    assert report.dry_run
    assert _entries(root) == set()
    assert {outcome.action for outcome in report.outcomes} == {"created"}
    assert "would created" in report.describe()


def test_a_dry_run_and_the_real_run_agree_on_every_outcome(tmp_path: Path) -> None:
    root = _project(tmp_path)

    planned = project_init.initialize(root, dry_run=True)
    applied = project_init.initialize(root)

    assert [(o.path, o.action) for o in planned.outcomes] == [
        (o.path, o.action) for o in applied.outcomes
    ]


def test_a_failed_write_leaves_the_repository_as_it_found_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    real_write = project_init.write_atomic
    calls: list[Path] = []

    def failing_write(path: Path, content: str | bytes) -> None:
        calls.append(path)
        if len(calls) == 3:
            raise OSError("disk went away")
        real_write(path, content)

    monkeypatch.setattr(project_init, "write_atomic", failing_write)

    with pytest.raises(OSError, match="disk went away"):
        project_init.initialize(root)

    assert _entries(root) == set()


def test_a_failed_write_restores_an_existing_gitignore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    original = b"*.secret\r\nopaque=\xff"
    (root / ".gitignore").write_bytes(original)
    real_write = project_init.write_atomic
    calls = 0

    def failing_write(path: Path, content: str | bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("disk went away")
        real_write(path, content)

    monkeypatch.setattr(project_init, "write_atomic", failing_write)

    with pytest.raises(OSError, match="disk went away"):
        project_init.initialize(root)

    assert (root / ".gitignore").read_bytes() == original


# --- the bootstrap boundary is stated, not hidden ---------------------------


def test_the_report_says_the_science_policy_is_not_yet_authoritative(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)

    described = project_init.initialize(root).describe()

    assert project_init.DEFAULT_POLICY_REF in described
    assert "not authoritative until it is committed and ratified" in described
    assert "fails closed" in described


# --- the command surface -----------------------------------------------------


def test_the_command_initializes_the_stated_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path)

    assert run_init([str(root)]) == 0

    assert (root / PROJECT_DECLARATION_FILENAME).exists()
    assert "created" in capsys.readouterr().out


def test_the_command_defaults_to_the_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    monkeypatch.chdir(root)

    assert run_init([]) == 0
    assert (root / PROJECT_DECLARATION_FILENAME).exists()


def test_the_command_refuses_with_exit_two_and_reports_on_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(tmp_path)
    (root / PROJECT_DECLARATION_FILENAME).write_text("{}\n")

    assert run_init([str(root)]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "conflict" in captured.err
    assert "Nothing was written." in captured.err


def test_the_unified_entry_point_routes_init(tmp_path: Path) -> None:
    from feedbax import cli

    root = _project(tmp_path)

    assert cli.main(["init", str(root), "--dry-run"]) == 0
    assert _entries(root) == set()
    assert "init" in cli.usage()
