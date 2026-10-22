"""Distribution identity rejection and real wheel/sdist acceptance.

Small authenticated packages exercise identity rejection without building Feedbax.
The ``slow`` acceptance scenarios build and install the actual package, prove
public model imports, and protect source immutability on success and failure.
Their shared immutable build fixtures run together under xdist's loadgroup scheduler.
"""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from feedbax._distribution_provenance import build_provenance_from_git
from feedbax.orchestration import revision
from feedbax.orchestration.revision import (
    FEEDBAX_DISTRIBUTION_PROVENANCE_SCHEMA_VERSION,
    FeedbaxRevisionError,
    assert_feedbax_revision_exact,
    resolve_feedbax_provenance,
    resolve_feedbax_revision,
)

pytestmark = pytest.mark.xdist_group(name="wheel_provenance")


def _run(
    args: list[str],
    *,
    cwd: Path,
    check: bool = True,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C",
            **(extra_env or {}),
        },
    )


def _source_bytes(source: Path) -> dict[str, str]:
    tracked = _run(["git", "ls-files", "-z"], cwd=source).stdout.split("\0")
    return {
        relative: sha256((source / relative).read_bytes()).hexdigest()
        for relative in tracked
        if relative and (source / relative).is_file()
    }


def _build(
    args: list[str],
    *,
    cwd: Path,
    cache: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return _run(
        ["uv", "build", *args],
        cwd=cwd,
        check=check,
        extra_env={"UV_CACHE_DIR": str(cache)},
    )


def _install_and_assert_wheel(
    *,
    wheel: Path,
    root: Path,
    revision_id: str,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    environment = root / "environment"
    _run(["uv", "venv", "--system-site-packages", str(environment)], cwd=root)
    python = environment / "bin" / "python"
    _run(
        ["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)],
        cwd=root,
        extra_env={"UV_CACHE_DIR": str(root / "uv-install-cache")},
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
    dependency_site = Path(pytest.__file__).resolve().parent.parent
    imported_paths = _run(
        [
            str(python),
            "-c",
            (
                "import pathlib, sys; "
                f"sys.path.append({str(dependency_site)!r}); "
                "import feedbax.analysis.exact_parents; "
                "import feedbax.models.cde as cde; "
                "import feedbax.models.feedback as feedback; "
                "print(pathlib.Path(cde.__file__).resolve()); "
                "print(pathlib.Path(feedback.__file__).resolve())"
            ),
        ],
        cwd=root,
        extra_env={"PYTHONPATH": ""},
    ).stdout.splitlines()
    assert len(imported_paths) == 2
    assert all(Path(path).is_relative_to(environment) for path in imported_paths)
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
    return package_root


@pytest.fixture(scope="module")
def installed_wheel(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, str]:
    root = tmp_path_factory.mktemp("feedbax-wheel-provenance")
    source = Path(__file__).resolve().parents[1]
    assert _run(["git", "status", "--porcelain=v1"], cwd=source).stdout == ""
    revision_id = _run(["git", "rev-parse", "HEAD"], cwd=source).stdout.strip()
    before = _source_bytes(source)
    wheel_dir = root / "wheel"
    wheel_dir.mkdir()
    _build(
        ["--wheel", "--directory", str(source), "--out-dir", str(wheel_dir)],
        cwd=root,
        cache=root / "uv-build-cache",
    )
    wheel = next(wheel_dir.glob("feedbax-*.whl"))
    package_root = _install_and_assert_wheel(
        wheel=wheel, root=root, revision_id=revision_id
    )
    assert _source_bytes(source) == before
    assert _run(["git", "status", "--porcelain=v1"], cwd=source).stdout == ""
    assert not (source / "feedbax" / "_distribution_provenance.json").exists()
    return source, package_root, revision_id


@pytest.fixture(scope="module")
def sdist_distribution(
    installed_wheel: tuple[Path, Path, str],
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, str]:
    source, _direct_package, revision_id = installed_wheel
    root = tmp_path_factory.mktemp("feedbax-sdist-provenance")
    sdist_dir = root / "sdist"
    sdist_dir.mkdir()
    _build(
        ["--sdist", "--directory", str(source), "--out-dir", str(sdist_dir)],
        cwd=root,
        cache=root / "uv-sdist-cache",
    )
    sdist = next(sdist_dir.glob("feedbax-*.tar.gz"))
    with tarfile.open(sdist, "r:gz") as archive:
        provenance_members = [
            name
            for name in archive.getnames()
            if name.endswith("feedbax/_distribution_provenance.json")
        ]
    assert len(provenance_members) == 1

    wheel_dir = root / "wheel"
    wheel_dir.mkdir()
    _build(
        ["--wheel", str(sdist), "--out-dir", str(wheel_dir)],
        cwd=root,
        cache=root / "uv-wheel-cache",
    )
    wheel = next(wheel_dir.glob("feedbax-*.whl"))
    package_root = _install_and_assert_wheel(
        wheel=wheel,
        root=root / "installed",
        revision_id=revision_id,
    )
    return sdist, package_root, revision_id


@pytest.fixture(scope="module")
def provenance_package(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    root = tmp_path_factory.mktemp("provenance-source")
    package = root / "feedbax"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    _run(["git", "init", "--quiet"], cwd=root)
    _run(["git", "add", "feedbax"], cwd=root)
    _run(
        ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
         "commit", "--quiet", "-m", "package identity"],
        cwd=root,
    )
    encoded = build_provenance_from_git(root)
    (package / "_distribution_provenance.json").write_bytes(encoded)
    return package, json.loads(encoded)["revision"]


@pytest.fixture
def wheel_package(
    provenance_package: tuple[Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, str]:
    source, revision_id = provenance_package
    package_root = tmp_path / "site-packages" / "feedbax"
    package_root.mkdir(parents=True)
    for path in source.iterdir():
        (package_root / path.name).write_bytes(path.read_bytes())
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
    assert revision.resolve_science_repo_import_revisions(()) == {}


def test_wheel_inside_science_checkout_preserves_science_ownership_and_pin(
    wheel_package: tuple[Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root, _revision_id = wheel_package
    science_package = tmp_path / "science_provider"
    science_package.mkdir()
    (science_package / "__init__.py").write_text("", encoding="utf-8")
    _run(["git", "init", "--quiet"], cwd=tmp_path)
    _run(["git", "add", "science_provider"], cwd=tmp_path)
    _run(
        ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
         "commit", "--quiet", "-m", "science identity"],
        cwd=tmp_path,
    )
    science_revision = _run(["git", "rev-parse", "HEAD"], cwd=tmp_path).stdout.strip()
    assert package_root.is_relative_to(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    resolved = revision.resolve_science_repo_import_revisions(
        (SimpleNamespace(entry_point_value="science_provider:register"),)
    )

    assert resolved == {str(tmp_path.resolve()): science_revision}
    mismatched = "a" * 40 if science_revision != "a" * 40 else "b" * 40
    with pytest.raises(FeedbaxRevisionError, match="science repo revision pin mismatch"):
        revision.assert_science_repo_revision_pin(
            primary_repo="science",
            realized_revision=mismatched,
            imported_revisions=resolved,
        )


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
    with pytest.raises(FeedbaxRevisionError, match="neither a Git-owned checkout"):
        revision.resolve_science_repo_import_revisions(())


def test_wheel_rejects_tampered_package_bytes(wheel_package: tuple[Path, str]) -> None:
    package_root, _revision_id = wheel_package
    with (package_root / "__init__.py").open("ab") as stream:
        stream.write(b"\n# tampered\n")

    with pytest.raises(FeedbaxRevisionError, match="does not match commit"):
        resolve_feedbax_revision()
    with pytest.raises(FeedbaxRevisionError, match="does not match commit"):
        revision.resolve_science_repo_import_revisions(())


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


@pytest.mark.slow
def test_wheel_build_rejects_dirty_source(
    installed_wheel: tuple[Path, Path, str],
    tmp_path: Path,
) -> None:
    clean_source, _installed_package, _revision_id = installed_wheel
    dirty_source = tmp_path / "dirty-source"
    _run(["git", "clone", "--quiet", str(clean_source), str(dirty_source)], cwd=tmp_path)
    with (dirty_source / "feedbax" / "__init__.py").open("a") as stream:
        stream.write("\n# dirty build\n")

    result = _build(
        [
            "--wheel",
            "--directory",
            str(dirty_source),
            "--out-dir",
            str(tmp_path / "wheel"),
        ],
        cwd=tmp_path,
        cache=tmp_path / "uv-dirty-cache",
        check=False,
    )

    assert result.returncode != 0
    assert "must be built from a clean source checkout" in result.stderr
    assert (
        FEEDBAX_DISTRIBUTION_PROVENANCE_SCHEMA_VERSION
        == "feedbax.distribution_provenance.v1"
    )


@pytest.mark.slow
def test_git_sdist_wheel_preserves_exact_verified_identity(
    sdist_distribution: tuple[Path, Path, str],
) -> None:
    _sdist, package_root, revision_id = sdist_distribution
    payload = json.loads(
        (package_root / "_distribution_provenance.json").read_text(encoding="utf-8")
    )

    assert payload["schema_version"] == FEEDBAX_DISTRIBUTION_PROVENANCE_SCHEMA_VERSION
    assert payload["revision"] == revision_id


@pytest.mark.slow
def test_noneditable_wheel_installs_and_verifies_public_models(installed_wheel) -> None:
    source, package_root, revision_id = installed_wheel
    assert package_root != source / "feedbax"
    assert json.loads((package_root / "_distribution_provenance.json").read_bytes())[
        "revision"
    ] == revision_id


def _rewrite_sdist(
    source: Path,
    target: Path,
    *,
    remove_provenance: bool = False,
    tamper_provenance: bool = False,
) -> None:
    unpacked = target.parent / f"{target.stem}-unpacked"
    unpacked.mkdir()
    with tarfile.open(source, "r:gz") as archive:
        archive.extractall(unpacked, filter="data")
    roots = [path for path in unpacked.iterdir() if path.is_dir()]
    assert len(roots) == 1
    provenance = roots[0] / "feedbax" / "_distribution_provenance.json"
    if remove_provenance:
        provenance.unlink()
    if tamper_provenance:
        payload = json.loads(provenance.read_text(encoding="utf-8"))
        payload["revision"] = "a" * 40
        provenance.write_text(json.dumps(payload), encoding="utf-8")
    with tarfile.open(target, "w:gz") as archive:
        archive.add(roots[0], arcname=roots[0].name)


@pytest.mark.parametrize("mutation", ["missing", "tampered"])
@pytest.mark.slow
def test_sdist_wheel_rejects_missing_or_tampered_provenance(
    sdist_distribution: tuple[Path, Path, str],
    tmp_path: Path,
    mutation: str,
) -> None:
    sdist, _package_root, _revision_id = sdist_distribution
    mutated = tmp_path / f"feedbax-{mutation}.tar.gz"
    _rewrite_sdist(
        sdist,
        mutated,
        remove_provenance=mutation == "missing",
        tamper_provenance=mutation == "tampered",
    )

    result = _build(
        ["--wheel", str(mutated), "--out-dir", str(tmp_path / "wheel")],
        cwd=tmp_path,
        cache=tmp_path / "uv-cache",
        check=False,
    )

    assert result.returncode != 0
    if mutation == "missing":
        assert "distribution provenance is missing" in result.stderr
    else:
        assert "commit identity is unverifiable" in result.stderr


@pytest.mark.slow
def test_post_initialize_failure_never_mutates_source_checkout(
    installed_wheel: tuple[Path, Path, str],
    tmp_path: Path,
) -> None:
    clean_source, _package_root, _revision_id = installed_wheel
    source = tmp_path / "source"
    _run(["git", "clone", "--quiet", str(clean_source), str(source)], cwd=tmp_path)
    hook = source / "hatch_build.py"
    hook_text = hook.read_text(encoding="utf-8")
    injection_point = "        self._temporary_directory = temporary_directory\n"
    assert injection_point in hook_text
    hook.write_text(
        hook_text.replace(
            injection_point,
            injection_point + '        raise RuntimeError("injected post-initialize failure")\n',
        ),
        encoding="utf-8",
    )
    _run(["git", "add", "hatch_build.py"], cwd=source)
    _run(["git", "commit", "--quiet", "-m", "inject build failure"], cwd=source)
    before = _source_bytes(source)

    result = _build(
        ["--wheel", "--directory", str(source), "--out-dir", str(tmp_path / "wheel")],
        cwd=tmp_path,
        cache=tmp_path / "uv-cache",
        check=False,
    )

    assert result.returncode != 0
    assert "injected post-initialize failure" in result.stderr
    assert _run(["git", "status", "--porcelain=v1"], cwd=source).stdout == ""
    assert _source_bytes(source) == before
    assert not (source / "feedbax" / "_distribution_provenance.json").exists()


@pytest.mark.slow
def test_concurrent_direct_wheel_builds_are_isolated_and_source_clean(
    installed_wheel: tuple[Path, Path, str],
    tmp_path: Path,
) -> None:
    source, _package_root, _revision_id = installed_wheel
    before = _source_bytes(source)

    def build_one(slot: int) -> subprocess.CompletedProcess[str]:
        return _build(
            [
                "--wheel",
                "--directory",
                str(source),
                "--out-dir",
                str(tmp_path / f"wheel-{slot}"),
            ],
            cwd=tmp_path,
            cache=tmp_path / f"uv-cache-{slot}",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(build_one, range(2)))

    assert all(result.returncode == 0 for result in results)
    assert all(list((tmp_path / f"wheel-{slot}").glob("feedbax-*.whl")) for slot in range(2))
    assert _run(["git", "status", "--porcelain=v1"], cwd=source).stdout == ""
    assert _source_bytes(source) == before
    assert not (source / "feedbax" / "_distribution_provenance.json").exists()
