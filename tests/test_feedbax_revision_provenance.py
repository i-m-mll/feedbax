"""Tests for the Feedbax provenance verifier (Mandible-Issue 7e7dac8).

These tests exercise ``resolve_feedbax_provenance``/``check_feedbax_provenance``
against real, isolated ``tmp_path`` Git checkouts (never the real ``feedbax``
package source) by monkeypatching ``revision.feedbax.__file__`` so each test is
pytest-xdist-safe and leaves no ambient process/module state behind.

The second half of the module covers the authoring/launch split of the
revision-pin gate (Mandible-Issue 0c2b295): ``assert_feedbax_revision_pin``
accepts an ancestor of the installed revision, while
``assert_feedbax_revision_exact`` — the launch-time boundary — still demands
identity.
"""

from __future__ import annotations

import subprocess
import warnings
from pathlib import Path

import pytest

from feedbax.orchestration import revision
from feedbax.orchestration.revision import (
    FeedbaxProvenance,
    FeedbaxRevisionAncestryWarning,
    FeedbaxRevisionError,
    assert_feedbax_revision_exact,
    assert_feedbax_revision_pin,
    check_feedbax_provenance,
    resolve_feedbax_provenance,
)

# A syntactically valid commit that no test repository can contain.
FABRICATED_REVISION = "0123456789abcdef0123456789abcdef01234567"


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


# --- Authoring/launch split of the revision-pin gate (Mandible-Issue 0c2b295) ---


def _git_stdout(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        env={**revision._GIT_ENVIRONMENT, "HOME": str(root)},
    ).stdout.strip()


def _init_history_repo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Build a checkout with a base commit, an advance, and an abandoned branch.

    Returns the package directory and the ``base``/``abandoned``/``head``
    commits. ``base`` is an ancestor of ``head``; ``abandoned`` exists but is not
    reachable from ``head``.
    """
    package_root = tmp_path / "checkout" / "feedbax"
    package_root.mkdir(parents=True)
    repo_root = package_root.parent
    _git(repo_root, "init", "--quiet", "--initial-branch=develop")
    _git(repo_root, "config", "user.email", "test@example.com")
    _git(repo_root, "config", "user.name", "Test")
    (package_root / "__init__.py").write_text("# base\n")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "--quiet", "-m", "base")
    base = _git_stdout(repo_root, "rev-parse", "HEAD")
    _git(repo_root, "checkout", "--quiet", "-b", "abandoned")
    (package_root / "__init__.py").write_text("# abandoned line of work\n")
    _git(repo_root, "commit", "--quiet", "-am", "abandoned")
    abandoned = _git_stdout(repo_root, "rev-parse", "HEAD")
    _git(repo_root, "checkout", "--quiet", "develop")
    (package_root / "__init__.py").write_text("# advanced\n")
    _git(repo_root, "commit", "--quiet", "-am", "advance")
    head = _git_stdout(repo_root, "rev-parse", "HEAD")
    return package_root, {"base": base, "abandoned": abandoned, "head": head}


@pytest.fixture
def history_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, str]]:
    package_root, revisions = _init_history_repo(tmp_path)
    monkeypatch.setattr(revision.feedbax, "__file__", str(package_root / "__init__.py"))
    return package_root, revisions


def test_authoring_pin_accepts_an_ancestor_of_the_installed_revision(
    history_checkout: tuple[Path, dict[str, str]],
) -> None:
    _package_root, revisions = history_checkout

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert assert_feedbax_revision_pin(revisions["base"]) == revisions["base"]


def test_authoring_pin_accepts_the_installed_revision_itself(
    history_checkout: tuple[Path, dict[str, str]],
) -> None:
    _package_root, revisions = history_checkout

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert assert_feedbax_revision_pin(revisions["head"]) == revisions["head"]


def test_authoring_pin_refuses_a_fabricated_commit(
    history_checkout: tuple[Path, dict[str, str]],
) -> None:
    _package_root, revisions = history_checkout

    with pytest.raises(FeedbaxRevisionError, match="not an ancestor") as excinfo:
        assert_feedbax_revision_pin(FABRICATED_REVISION)

    message = str(excinfo.value)
    assert FABRICATED_REVISION in message
    assert revisions["head"] in message
    assert "unknown to the complete checkout" in message


def test_authoring_pin_refuses_a_commit_on_an_abandoned_branch(
    history_checkout: tuple[Path, dict[str, str]],
) -> None:
    _package_root, revisions = history_checkout

    with pytest.raises(FeedbaxRevisionError, match="not an ancestor") as excinfo:
        assert_feedbax_revision_pin(revisions["abandoned"])

    message = str(excinfo.value)
    assert f"locked={revisions['abandoned']}" in message
    assert f"loaded={revisions['head']}" in message
    assert "not reachable from the loaded revision" in message


@pytest.mark.parametrize(
    "malformed",
    ["not-a-commit", "abc123", "0123456789ABCDEF0123456789abcdef01234567", ""],
)
def test_authoring_pin_refuses_a_malformed_pin(
    history_checkout: tuple[Path, dict[str, str]], malformed: str
) -> None:
    with pytest.raises(FeedbaxRevisionError, match="full lowercase Git commit"):
        assert_feedbax_revision_pin(malformed)


def test_authoring_pin_warns_when_the_install_has_no_git_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A wheel-style install: the package directory is not inside a checkout, so
    # ancestry is undecidable in principle rather than answered "no".
    package_root = tmp_path / "site-packages" / "feedbax"
    package_root.mkdir(parents=True)
    monkeypatch.setattr(revision.feedbax, "__file__", str(package_root / "__init__.py"))

    with pytest.warns(FeedbaxRevisionAncestryWarning, match="no resolvable Git history"):
        result = assert_feedbax_revision_pin(FABRICATED_REVISION)

    assert result == FABRICATED_REVISION


def test_authoring_pin_warns_when_a_shallow_checkout_lacks_the_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root, revisions = _init_history_repo(tmp_path)
    shallow_root = tmp_path / "shallow"
    _git(
        tmp_path,
        "clone",
        "--quiet",
        "--depth",
        "1",
        "--branch",
        "develop",
        package_root.parent.as_uri(),
        str(shallow_root),
    )
    assert _git_stdout(shallow_root, "rev-parse", "--is-shallow-repository") == "true"
    monkeypatch.setattr(revision.feedbax, "__file__", str(shallow_root / "feedbax" / "__init__.py"))

    with pytest.warns(FeedbaxRevisionAncestryWarning, match="shallow or partial"):
        result = assert_feedbax_revision_pin(revisions["base"])

    assert result == revisions["base"]


def test_authoring_pin_warns_when_a_partial_checkout_lacks_the_object(
    history_checkout: tuple[Path, dict[str, str]],
) -> None:
    package_root, _revisions = history_checkout
    _git(package_root.parent, "config", "remote.origin.promisor", "true")

    with pytest.warns(FeedbaxRevisionAncestryWarning, match="shallow or partial"):
        result = assert_feedbax_revision_pin(FABRICATED_REVISION)

    assert result == FABRICATED_REVISION


def test_undeterminable_ancestry_is_promotable_to_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root = tmp_path / "site-packages" / "feedbax"
    package_root.mkdir(parents=True)
    monkeypatch.setattr(revision.feedbax, "__file__", str(package_root / "__init__.py"))

    with warnings.catch_warnings():
        warnings.simplefilter("error", FeedbaxRevisionAncestryWarning)
        with pytest.raises(FeedbaxRevisionAncestryWarning):
            assert_feedbax_revision_pin(FABRICATED_REVISION)


def test_launch_exact_check_still_refuses_an_ancestor(
    history_checkout: tuple[Path, dict[str, str]],
) -> None:
    """The split must not leak ancestry tolerance into the launch boundary."""
    _package_root, revisions = history_checkout

    with pytest.raises(FeedbaxRevisionError, match="mismatch") as excinfo:
        assert_feedbax_revision_exact(revisions["base"])

    assert (
        f"Feedbax revision pin mismatch: "
        f"locked={revisions['base']} loaded={revisions['head']}" == str(excinfo.value)
    )


def test_launch_exact_check_accepts_and_returns_the_loaded_revision(
    history_checkout: tuple[Path, dict[str, str]],
) -> None:
    _package_root, revisions = history_checkout

    assert assert_feedbax_revision_exact(revisions["head"]) == revisions["head"]


def test_launch_exact_check_refuses_a_malformed_pin(
    history_checkout: tuple[Path, dict[str, str]],
) -> None:
    with pytest.raises(FeedbaxRevisionError) as excinfo:
        assert_feedbax_revision_exact("not-a-commit")

    assert str(excinfo.value) == "locked Feedbax revision pin must be a full lowercase Git commit"


def test_launch_exact_check_fails_closed_without_git_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root = tmp_path / "site-packages" / "feedbax"
    package_root.mkdir(parents=True)
    monkeypatch.setattr(revision.feedbax, "__file__", str(package_root / "__init__.py"))

    with pytest.raises(FeedbaxRevisionError, match="cannot resolve"):
        assert_feedbax_revision_exact(FABRICATED_REVISION)
