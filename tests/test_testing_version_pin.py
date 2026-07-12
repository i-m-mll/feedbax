from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from feedbax.testing import version_pin


def _pin_file(tmp_path: Path) -> Path:
    path = tmp_path / "upstream.toml"
    path.write_text('rev = "abc123"\n', encoding="utf-8")
    return path


def test_version_pin_passes_head_and_reachability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, ...]] = []

    def run_git(root: Path, args: list[str], *, check: bool = True):  # noqa: ANN202
        calls.append(tuple(args))
        stdout = "abc123\n" if args[0] == "rev-parse" else ""
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(version_pin, "_run_git", run_git)
    report = version_pin.check_version_pin(
        package_name="upstream",
        pin_file=_pin_file(tmp_path),
        checkout_root=tmp_path,
        remote_ref="refs/remotes/origin/main",
    )

    assert report.head_revision == "abc123"
    assert not report.skipped
    assert calls == [
        ("rev-parse", "HEAD"),
        ("rev-parse", "--verify", "refs/remotes/origin/main"),
        ("merge-base", "--is-ancestor", "abc123", "refs/remotes/origin/main"),
    ]


def test_version_pin_rejects_head_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(version_pin, "_git_stdout", lambda *_args: "different")
    with pytest.raises(AssertionError, match="Bump the pin"):
        version_pin.check_version_pin(
            package_name="upstream",
            pin_file=_pin_file(tmp_path),
            checkout_root=tmp_path,
        )


def test_version_pin_escape_hatch_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run_git(root: Path, args: list[str], *, check: bool = True):  # noqa: ANN202
        if args == ["merge-base", "--is-ancestor", "abc123", "origin/main"]:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="abc123\n", stderr="")

    monkeypatch.setattr(version_pin, "_run_git", run_git)
    with pytest.warns(RuntimeWarning, match="ALLOW_LOCAL=1"):
        report = version_pin.check_version_pin(
            package_name="upstream",
            pin_file=_pin_file(tmp_path),
            checkout_root=tmp_path,
            remote_ref="origin/main",
            escape_hatch_env="ALLOW_LOCAL",
            environ={"ALLOW_LOCAL": "1"},
        )
    assert report.unpublished_allowed


def test_version_pin_noneditable_install_has_documented_skip_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(version_pin, "find_editable_git_root", lambda _path: None)
    report = version_pin.check_version_pin(
        package_name="upstream",
        pin_file=_pin_file(tmp_path),
        package_path=tmp_path / "installed" / "upstream",
    )
    assert report.skipped
    assert "not installed from a Git checkout" in (report.skip_reason or "")


def test_version_pin_missing_remote_fails_even_with_escape_hatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run_git(root: Path, args: list[str], *, check: bool = True):  # noqa: ANN202
        if args == ["rev-parse", "--verify", "origin/main"]:
            raise subprocess.CalledProcessError(1, args)
        return subprocess.CompletedProcess(args, 0, stdout="abc123\n", stderr="")

    monkeypatch.setattr(version_pin, "_run_git", run_git)
    with pytest.raises(AssertionError, match="fetch"):
        version_pin.check_version_pin(
            package_name="upstream",
            pin_file=_pin_file(tmp_path),
            checkout_root=tmp_path,
            remote_ref="origin/main",
            escape_hatch_env="ALLOW_LOCAL",
            environ={"ALLOW_LOCAL": "1"},
        )
