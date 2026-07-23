"""Tests for the Feedbax provenance verifier (Mandible-Issue 7e7dac8).

These tests exercise ``resolve_feedbax_provenance``/``check_feedbax_provenance``
against real, isolated ``tmp_path`` Git checkouts (never the real ``feedbax``
package source) by monkeypatching ``revision.feedbax.__file__`` so each test is
pytest-xdist-safe and leaves no ambient process/module state behind.
"""

from __future__ import annotations

import subprocess
import warnings
from pathlib import Path

import pytest

from feedbax.orchestration import revision
from feedbax.orchestration.revision import (
    FeedbaxProvenance,
    FeedbaxRevisionError,
    check_feedbax_provenance,
    resolve_feedbax_provenance,
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        env={**revision._GIT_ENVIRONMENT, "HOME": str(root)},
    )


def _init_clean_repo(tmp_path: Path) -> tuple[Path, str]:
    """Create a tmp Git checkout with one committed package file and return its HEAD."""
    package_root = tmp_path / "checkout" / "feedbax"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("# stub package\n")
    repo_root = package_root.parent
    _git(repo_root, "init", "--quiet")
    _git(repo_root, "config", "user.email", "test@example.com")
    _git(repo_root, "config", "user.name", "Test")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "--quiet", "-m", "initial")
    head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        env={**revision._GIT_ENVIRONMENT, "HOME": str(repo_root)},
    ).stdout.strip()
    return package_root, head


@pytest.fixture
def clean_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    package_root, head = _init_clean_repo(tmp_path)
    monkeypatch.setattr(revision.feedbax, "__file__", str(package_root / "__init__.py"))
    return package_root, head


def test_resolve_feedbax_provenance_reports_clean_origin(
    clean_checkout: tuple[Path, str],
) -> None:
    package_root, head = clean_checkout

    provenance = resolve_feedbax_provenance()

    assert provenance == FeedbaxProvenance(
        source_path=package_root, revision=head, dirty=False
    )


def test_check_feedbax_provenance_accepts_matching_clean_pin(
    clean_checkout: tuple[Path, str],
) -> None:
    _package_root, head = clean_checkout

    result = check_feedbax_provenance(head)

    assert result is not None
    assert result.revision == head
    assert result.dirty is False


def test_check_feedbax_provenance_refuses_mismatch(
    clean_checkout: tuple[Path, str],
) -> None:
    _package_root, head = clean_checkout
    other_revision = "a" * 40
    assert other_revision != head

    with pytest.raises(FeedbaxRevisionError, match="mismatch") as excinfo:
        check_feedbax_provenance(other_revision)

    assert head in str(excinfo.value)
    assert other_revision in str(excinfo.value)


def test_check_feedbax_provenance_refuses_dirty_tree(
    clean_checkout: tuple[Path, str],
) -> None:
    package_root, head = clean_checkout
    (package_root / "__init__.py").write_text("# uncommitted edit\n")

    with pytest.raises(FeedbaxRevisionError, match="uncommitted changes"):
        check_feedbax_provenance(head)


def test_resolve_feedbax_provenance_detects_dirty_tree(
    clean_checkout: tuple[Path, str],
) -> None:
    package_root, _head = clean_checkout
    (package_root / "new_file.py").write_text("# untracked\n")

    provenance = resolve_feedbax_provenance()

    assert provenance.dirty is True


def test_check_feedbax_provenance_refuses_unverifiable_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A directory that is not a Git working tree at all: origin is unresolvable.
    package_root = tmp_path / "not-a-repo"
    package_root.mkdir()
    monkeypatch.setattr(revision.feedbax, "__file__", str(package_root / "__init__.py"))

    with pytest.raises(FeedbaxRevisionError, match="cannot resolve"):
        check_feedbax_provenance("a" * 40)


def test_check_feedbax_provenance_skips_with_warning_when_pin_absent(
    clean_checkout: tuple[Path, str],
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = check_feedbax_provenance(None)

    assert result is None
    assert any("skipping provenance verification" in str(item.message) for item in caught)


def test_check_feedbax_provenance_rejects_malformed_pin(
    clean_checkout: tuple[Path, str],
) -> None:
    with pytest.raises(FeedbaxRevisionError, match="full lowercase Git commit"):
        check_feedbax_provenance("not-a-commit")


def test_check_feedbax_provenance_override_bypasses_mismatch_with_warning(
    clean_checkout: tuple[Path, str],
) -> None:
    _package_root, head = clean_checkout
    other_revision = "b" * 40
    assert other_revision != head

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = check_feedbax_provenance(other_revision, override=True)

    assert result is not None
    assert result.revision == head
    assert any("override in effect" in str(item.message) for item in caught)


def test_check_feedbax_provenance_override_bypasses_dirty_tree_with_warning(
    clean_checkout: tuple[Path, str],
) -> None:
    package_root, head = clean_checkout
    (package_root / "__init__.py").write_text("# uncommitted edit\n")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = check_feedbax_provenance(head, override=True)

    assert result is not None
    assert result.dirty is True
    assert any("override in effect" in str(item.message) for item in caught)


def test_check_feedbax_provenance_override_bypasses_unverifiable_with_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root = tmp_path / "not-a-repo"
    package_root.mkdir()
    monkeypatch.setattr(revision.feedbax, "__file__", str(package_root / "__init__.py"))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = check_feedbax_provenance("a" * 40, override=True)

    assert result is None
    assert any("unverifiable provenance" in str(item.message) for item in caught)


def test_check_feedbax_provenance_override_is_never_the_default(
    clean_checkout: tuple[Path, str],
) -> None:
    _package_root, head = clean_checkout
    other_revision = "c" * 40

    with pytest.raises(FeedbaxRevisionError):
        check_feedbax_provenance(other_revision)
