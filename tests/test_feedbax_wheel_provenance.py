"""Real-wheel acceptance tests for Feedbax distribution provenance."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from feedbax.orchestration import revision
from feedbax.orchestration.revision import (
    FEEDBAX_DISTRIBUTION_PROVENANCE_SCHEMA_VERSION,
    FeedbaxRevisionError,
    assert_feedbax_revision_exact,
    resolve_feedbax_provenance,
    resolve_feedbax_revision,
)


def _run(
    args: list[str],
    *,
    cwd: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C",
        },
    )


def _copy_source_checkout(destination: Path) -> str:
    root = Path(__file__).resolve().parents[1]
    required_roots = {
        ".gitignore",
        "feedbax",
        "hatch_build.py",
        "pyproject.toml",
        "README.md",
        "LICENSE",
    }
    listed = _run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=root,
    ).stdout.splitlines()
    for relative_text in listed:
        relative = Path(relative_text)
        if relative.parts[0] not in required_roots:
            continue
        source = root / relative
        if not source.is_file():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    _run(["git", "init", "--quiet", "--initial-branch=develop"], cwd=destination)
    _run(["git", "config", "user.email", "wheel-test@example.com"], cwd=destination)
    _run(["git", "config", "user.name", "Wheel Test"], cwd=destination)
    _run(["git", "config", "commit.gpgsign", "false"], cwd=destination)
    _run(["git", "add", "--force", "-A"], cwd=destination)
    _run(["git", "commit", "--quiet", "-m", "wheel fixture"], cwd=destination)
    return _run(["git", "rev-parse", "HEAD"], cwd=destination).stdout.strip()


@pytest.fixture(scope="module")
def installed_wheel(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, str]:
    root = tmp_path_factory.mktemp("feedbax-wheel-provenance")
    source = root / "source"
    source.mkdir()
    revision_id = _copy_source_checkout(source)
    wheel_dir = root / "wheel"
    wheel_dir.mkdir()
    _run(
        ["uv", "build", "--wheel", "--directory", str(source), "--out-dir", str(wheel_dir)],
        cwd=root,
    )
    wheel = next(wheel_dir.glob("feedbax-*.whl"))
    environment = root / "environment"
    _run(["uv", "venv", "--system-site-packages", str(environment)], cwd=root)
    python = environment / "bin" / "python"
    _run(
        ["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)],
        cwd=root,
    )
    package_root = Path(
        _run(
            [
                str(python),
                "-c",
                "import pathlib, feedbax; print(pathlib.Path(feedbax.__file__).parent)",
            ],
            cwd=root,
        ).stdout.strip()
    )
    assert package_root.is_relative_to(environment)
    assert not (package_root.parent / ".git").exists()
    assert (package_root / "models" / "cde.py").is_file()
    observed_revision = _run(
        [
            str(python),
            "-c",
            (
                "from feedbax.orchestration.revision import "
                "assert_feedbax_revision_exact, resolve_feedbax_revision; "
                f"assert_feedbax_revision_exact({revision_id!r}); "
                "print(resolve_feedbax_revision())"
            ),
        ],
        cwd=root,
    ).stdout.strip()
    assert observed_revision == revision_id
    return source, package_root, revision_id


@pytest.fixture
def wheel_package(
    installed_wheel: tuple[Path, Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, str]:
    _source, installed_package, revision_id = installed_wheel
    package_root = tmp_path / "site-packages" / "feedbax"
    shutil.copytree(installed_package, package_root)
    monkeypatch.setattr(revision.feedbax, "__file__", str(package_root / "__init__.py"))
    return package_root, revision_id


def test_noneditable_wheel_resolves_and_passes_exact_gate(
    wheel_package: tuple[Path, str],
) -> None:
    package_root, revision_id = wheel_package

    assert resolve_feedbax_revision() == revision_id
    assert assert_feedbax_revision_exact(revision_id) == revision_id
    assert resolve_feedbax_provenance().source_path == package_root
    assert resolve_feedbax_provenance().dirty is False


def test_wheel_exact_gate_rejects_mismatch(wheel_package: tuple[Path, str]) -> None:
    _package_root, revision_id = wheel_package
    mismatched = "a" * 40 if revision_id != "a" * 40 else "b" * 40

    with pytest.raises(FeedbaxRevisionError, match="mismatch"):
        assert_feedbax_revision_exact(mismatched)


def test_wheel_rejects_missing_identity(wheel_package: tuple[Path, str]) -> None:
    package_root, _revision_id = wheel_package
    (package_root / "_distribution_provenance.json").unlink()

    with pytest.raises(FeedbaxRevisionError, match="neither a Git-owned checkout"):
        resolve_feedbax_revision()


def test_wheel_rejects_tampered_package_bytes(wheel_package: tuple[Path, str]) -> None:
    package_root, _revision_id = wheel_package
    with (package_root / "__init__.py").open("ab") as stream:
        stream.write(b"\n# tampered\n")

    with pytest.raises(FeedbaxRevisionError, match="does not match commit"):
        resolve_feedbax_revision()


def test_wheel_rejects_malformed_and_unsupported_identity(
    wheel_package: tuple[Path, str],
) -> None:
    package_root, _revision_id = wheel_package
    provenance_path = package_root / "_distribution_provenance.json"
    payload = json.loads(provenance_path.read_text())
    payload["schema_version"] = "feedbax.distribution_provenance.v0"
    provenance_path.write_text(json.dumps(payload))
    with pytest.raises(FeedbaxRevisionError, match="schema is unsupported"):
        resolve_feedbax_revision()

    provenance_path.write_text("{")
    with pytest.raises(FeedbaxRevisionError, match="unreadable or malformed"):
        resolve_feedbax_revision()


def test_wheel_rejects_conflicting_checkout_identity(
    wheel_package: tuple[Path, str],
) -> None:
    package_root, wheel_revision = wheel_package
    checkout_root = package_root.parent
    _run(["git", "init", "--quiet"], cwd=checkout_root)
    _run(["git", "config", "user.email", "conflict@example.com"], cwd=checkout_root)
    _run(["git", "config", "user.name", "Conflict Test"], cwd=checkout_root)
    _run(["git", "config", "commit.gpgsign", "false"], cwd=checkout_root)
    _run(["git", "add", "--force", "feedbax/__init__.py"], cwd=checkout_root)
    _run(["git", "commit", "--quiet", "-m", "conflicting checkout"], cwd=checkout_root)
    checkout_revision = _run(["git", "rev-parse", "HEAD"], cwd=checkout_root).stdout.strip()
    assert checkout_revision != wheel_revision

    with pytest.raises(FeedbaxRevisionError, match="conflicting Feedbax revision identities"):
        resolve_feedbax_revision()


def test_wheel_build_rejects_dirty_source(
    installed_wheel: tuple[Path, Path, str],
    tmp_path: Path,
) -> None:
    clean_source, _installed_package, _revision_id = installed_wheel
    dirty_source = tmp_path / "dirty-source"
    _run(["git", "clone", "--quiet", str(clean_source), str(dirty_source)], cwd=tmp_path)
    with (dirty_source / "feedbax" / "__init__.py").open("a") as stream:
        stream.write("\n# dirty build\n")

    result = _run(
        [
            "uv",
            "build",
            "--wheel",
            "--directory",
            str(dirty_source),
            "--out-dir",
            str(tmp_path / "wheel"),
        ],
        cwd=tmp_path,
        check=False,
    )

    assert result.returncode != 0
    assert "must be built from a clean source checkout" in result.stderr
    assert (
        FEEDBAX_DISTRIBUTION_PROVENANCE_SCHEMA_VERSION
        == "feedbax.distribution_provenance.v1"
    )
