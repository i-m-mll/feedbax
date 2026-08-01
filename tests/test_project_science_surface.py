"""Deny-by-default project science-surface gate.

Every test builds a throwaway Git repository under ``tmp_path`` with a ratified
policy on its baseline branch, then exercises the real
``check-project-science-surface`` subcommand against branch-side mutations.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import feedbax.__main__ as feedbax_cli


BASELINE_BRANCH = "baseline"
POLICY_PATH = "governance/science_surface_policy.toml"

RATIFIED_POLICY = """
schema_version = 1
source_roots = ["src"]
banned_paths = ["src/orchard/pressroom", "src/orchard/pressroom/**"]

[[allowed_file]]
path = "src/orchard/__init__.py"
symbols = ["__version__"]
reason = "package marker"

[[allowed_file]]
path = "src/orchard/canopy.py"
symbols = ["CANOPY_LABEL", "measure_canopy"]
"""

INIT_SOURCE = '__version__ = "0.1.0"\n'

CANOPY_SOURCE = '''"""Ratified canopy surface."""

from pathlib import Path

CANOPY_LABEL = "canopy"


def measure_canopy(root: Path) -> int:
    return len(str(root))
'''


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "--no-optional-locks", *args],
        cwd=root,
        check=True,
        capture_output=True,
        env={
            "HOME": str(root),
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "Surface Test",
            "GIT_AUTHOR_EMAIL": "surface@example.invalid",
            "GIT_COMMITTER_NAME": "Surface Test",
            "GIT_COMMITTER_EMAIL": "surface@example.invalid",
        },
    )


def _write(root: Path, relpath: str, text: str) -> None:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _commit(root: Path, message: str) -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "--quiet", "--no-gpg-sign", "-m", message)


@pytest.fixture
def project_repo(tmp_path: Path) -> Path:
    """A tiny project checkout whose baseline branch carries a ratified policy."""

    root = tmp_path / "orchard-project"
    root.mkdir()
    _git(root, "init", "--quiet", f"--initial-branch={BASELINE_BRANCH}")
    _write(root, POLICY_PATH, RATIFIED_POLICY)
    _write(root, "src/orchard/__init__.py", INIT_SOURCE)
    _write(root, "src/orchard/canopy.py", CANOPY_SOURCE)
    _commit(root, "ratify science-surface policy")
    _git(root, "checkout", "--quiet", "-b", "work")
    return root


def _run(root: Path) -> int:
    return feedbax_cli.main(
        [
            "check-project-science-surface",
            "--root",
            str(root),
            "--policy",
            POLICY_PATH,
            "--baseline-ref",
            BASELINE_BRANCH,
        ]
    )


def test_clean_project_passes(project_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(project_repo) == 0
    assert "passed" in capsys.readouterr().out


def test_unlisted_production_file_fails(
    project_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(project_repo, "src/orchard/understory.py", "def sample_understory():\n    return 1\n")

    assert _run(project_repo) == 1

    out = capsys.readouterr().out
    assert "unlisted-file src/orchard/understory.py" in out
    assert "deny-by-default" in out
    assert "The correct home for new machinery is feedbax" in out
    assert "branch cannot authorize itself" in out


def test_new_top_level_symbol_in_listed_file_fails(
    project_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = project_repo / "src/orchard/canopy.py"
    path.write_text(
        path.read_text(encoding="utf-8") + "\n\ndef prune_canopy() -> None:\n    return None\n",
        encoding="utf-8",
    )

    assert _run(project_repo) == 1

    out = capsys.readouterr().out
    assert "unlisted-symbol src/orchard/canopy.py::prune_canopy" in out
    assert "unlisted-file" not in out


def test_symbols_hidden_in_module_level_blocks_are_detected(
    project_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = project_repo / "src/orchard/canopy.py"
    path.write_text(
        path.read_text(encoding="utf-8")
        + "\n\nif True:\n    SHADOW_TABLE = {}\n\n    def shadow_solver() -> None:\n"
        "        return None\n",
        encoding="utf-8",
    )

    assert _run(project_repo) == 1

    out = capsys.readouterr().out
    assert "unlisted-symbol src/orchard/canopy.py::SHADOW_TABLE" in out
    assert "unlisted-symbol src/orchard/canopy.py::shadow_solver" in out


def test_imports_are_not_treated_as_project_symbols(project_repo: Path) -> None:
    path = project_repo / "src/orchard/canopy.py"
    path.write_text(
        "import json\nfrom pathlib import Path\n\n" + path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    assert _run(project_repo) == 0


def test_banned_path_recreation_fails(
    project_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(project_repo, "src/orchard/pressroom/__init__.py", "")
    _write(project_repo, "src/orchard/pressroom/rollout.py", "def roll() -> None:\n    return None\n")

    assert _run(project_repo) == 1

    out = capsys.readouterr().out
    assert "banned-path src/orchard/pressroom:" in out
    assert "banned-path src/orchard/pressroom/rollout.py:" in out
    assert "matches ratified banned pattern" in out


def test_branch_edited_policy_cannot_authorize_itself(
    project_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(project_repo, "src/orchard/understory.py", "def sample_understory():\n    return 1\n")
    _write(
        project_repo,
        POLICY_PATH,
        RATIFIED_POLICY
        + '\n[[allowed_file]]\npath = "src/orchard/understory.py"\n'
        'symbols = ["sample_understory"]\n',
    )
    _commit(project_repo, "self-authorize new production file")

    assert _run(project_repo) == 1

    out = capsys.readouterr().out
    assert "unlisted-file src/orchard/understory.py" in out
    assert f"branch cannot authorize itself by editing {POLICY_PATH}" in out


def test_policy_absent_from_baseline_fails(
    project_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(project_repo, "governance/new_policy.toml", RATIFIED_POLICY)
    _commit(project_repo, "introduce policy on the branch only")

    exit_code = feedbax_cli.main(
        [
            "check-project-science-surface",
            "--root",
            str(project_repo),
            "--policy",
            "governance/new_policy.toml",
            "--baseline-ref",
            BASELINE_BRANCH,
        ]
    )

    assert exit_code == 1
    out = capsys.readouterr().out
    assert "has no ratified science-surface policy" in out
    assert "cannot authorize itself by adding or editing that file" in out


def test_unversioned_baseline_policy_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "unversioned-project"
    root.mkdir()
    _git(root, "init", "--quiet", f"--initial-branch={BASELINE_BRANCH}")
    _write(root, POLICY_PATH, 'source_roots = ["src"]\n')
    _write(root, "src/orchard/__init__.py", INIT_SOURCE)
    _commit(root, "unversioned policy")

    assert _run(root) == 1
    assert "no 'schema_version'" in capsys.readouterr().out


def test_unknown_policy_schema_version_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "future-policy-project"
    root.mkdir()
    _git(root, "init", "--quiet", f"--initial-branch={BASELINE_BRANCH}")
    _write(root, POLICY_PATH, 'schema_version = 2\nsource_roots = ["src"]\n')
    _write(root, "src/orchard/__init__.py", INIT_SOURCE)
    _commit(root, "future policy")

    assert _run(root) == 1
    out = capsys.readouterr().out
    assert "schema_version 2 is not supported" in out
    assert "supports schema_version 1 only" in out
