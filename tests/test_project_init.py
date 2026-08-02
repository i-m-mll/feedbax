"""``feedbax init`` creates exactly seven things, twice, without ever rewriting.

The value of an initializer is not that it saves typing; it is that the thing it
produces is *exactly* what the framework expects, and that running it again on a
live repository cannot damage it. These tests pin both: the exact inventory, the
exact refusal behavior, and the guarantee that a second run validates rather
than overwrites.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from feedbax.contracts.experiment_envelope_dialect import ExperimentEnvelopeLayer
from feedbax.contracts.project_experiment import (
    PROJECT_DECLARATION_FILENAME,
    load_project_declaration,
)
from feedbax.governance import project_init
from feedbax.governance.agent_instructions import (
    AGENT_FILE_NAMES,
    BlockStatus,
    classify,
    parse_block,
)
from feedbax.governance.cli import run_init
from feedbax.governance.science_surface import POLICY_SCHEMA_VERSION


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


def test_a_new_project_gets_exactly_the_seven_entries(tmp_path: Path) -> None:
    root = _project(tmp_path)

    report = project_init.initialize(root)

    assert report.exit_code == project_init.EXIT_OK
    assert _entries(root) == set(project_init.entry_paths(report.package))
    assert set(project_init.entry_paths("spinnaker_study")) == {
        PROJECT_DECLARATION_FILENAME,
        "specs/experiment/authoring_budget.json",
        "ci/feedbax-science-surface.toml",
        "pyproject.toml",
        "src/spinnaker_study/__init__.py",
        "AGENTS.md",
        "CLAUDE.md",
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
    """No example envelope, no fake science, no empty generated custody."""
    root = _project(tmp_path)

    project_init.initialize(root)

    for absent in ("generated", "specs/experiment/example.envelope.json", ".git", ".gitignore"):
        assert not (root / absent).exists(), absent
    assert list((root / "specs/experiment").iterdir()) == [
        root / "specs/experiment/authoring_budget.json"
    ]
    assert (root / "src" / "spinnaker_study" / "__init__.py").read_text().count("\n") == 1


# --- what it writes is what Feedbax reads -----------------------------------


def test_the_written_declaration_loads_through_the_real_loader(tmp_path: Path) -> None:
    root = _project(tmp_path)

    project_init.initialize(root)

    declaration = load_project_declaration(root)
    assert declaration.project == "spinnaker-study"
    assert declaration.envelope_directory == project_init.DEFAULT_ENVELOPE_DIRECTORY
    assert declaration.output_directory == project_init.DEFAULT_OUTPUT_DIRECTORY
    assert declaration.authoring_budget.resource_id == project_init.DEFAULT_BUDGET_REF
    assert declaration.authoring_budget.root.joinpath(
        declaration.authoring_budget.document_name
    ).is_file()


def test_the_written_budget_states_a_section_for_every_dialect_layer(tmp_path: Path) -> None:
    root = _project(tmp_path)

    project_init.initialize(root)

    document = json.loads((root / project_init.DEFAULT_BUDGET_REF).read_text())
    assert set(document["layers"]) == {layer.value for layer in ExperimentEnvelopeLayer}


def test_the_ratified_training_caps_are_the_framework_defaults() -> None:
    training = project_init.DEFAULT_AUTHORING_BUDGET_LAYERS["training"]

    assert training["max_lines"] == 128
    assert training["max_ast_nodes"] == 160
    assert training["max_depth"] == 16


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


def test_the_agent_files_are_one_real_file_and_one_relative_link(tmp_path: Path) -> None:
    root = _project(tmp_path)

    project_init.initialize(root)

    agents, claude = (root / name for name in AGENT_FILE_NAMES)
    assert agents.is_file() and not agents.is_symlink()
    assert claude.is_symlink()
    assert Path(claude.readlink()) == Path(AGENT_FILE_NAMES[0])
    assert claude.read_text() == agents.read_text()
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
    assert actions[project_init.DEFAULT_BUDGET_REF] == "validated"
    assert actions[project_init.DEFAULT_POLICY_REF] == "validated"
    assert actions[AGENT_FILE_NAMES[0]] == "unchanged"


def test_a_customized_declaration_is_kept_not_reset(tmp_path: Path) -> None:
    root = _project(tmp_path)
    project_init.initialize(root)
    document = json.loads((root / PROJECT_DECLARATION_FILENAME).read_text())
    document["envelope_directory"] = "studies"
    (root / PROJECT_DECLARATION_FILENAME).write_text(json.dumps(document, indent=2) + "\n")

    project_init.initialize(root)

    assert load_project_declaration(root).envelope_directory == "studies"


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


def test_an_invalid_budget_refuses_the_whole_run(tmp_path: Path) -> None:
    root = _project(tmp_path)
    budget = root / project_init.DEFAULT_BUDGET_REF
    budget.parent.mkdir(parents=True)
    budget.write_text('{"schema_id": "feedbax.spec.authoring_budget", "schema_version": "v9"}\n')

    report = project_init.initialize(root)

    assert report.exit_code == project_init.EXIT_CONFLICT
    assert _entries(root) == {project_init.DEFAULT_BUDGET_REF}


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
    assert {outcome.action for outcome in report.outcomes} == {"created", "linked"}
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

    def failing_write(path: Path, text: str) -> None:
        calls.append(path)
        if len(calls) == 3:
            raise OSError("disk went away")
        real_write(path, text)

    monkeypatch.setattr(project_init, "write_atomic", failing_write)

    with pytest.raises(OSError, match="disk went away"):
        project_init.initialize(root)

    assert _entries(root) == set()


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
