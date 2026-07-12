from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from feedbax.testing.suite import (
    ContractSuiteHooks,
    assert_live_family_counts,
    assert_negative_canary,
    assert_negative_canaries_collected,
    collect_contract_nodeids,
    load_suite_manifest,
)


def _write_manifest(path: Path, *, minimum: int = 1) -> None:
    path.write_text(
        f"""
manifest_version = 1
marker = "contract"

[[families]]
name = "alpha"
status = "live"
expected_collection_pattern = "tests/test_alpha.py::"
minimum_non_skipped = {minimum}
negative_canary = ["tests/test_alpha.py::test_bad"]

[[families]]
name = "later"
status = "pending"
expected_collection_pattern = "tests/test_later.py::"
minimum_non_skipped = 0
""",
        encoding="utf-8",
    )


def test_load_suite_manifest_and_meta_assertions(tmp_path: Path) -> None:
    path = tmp_path / "suite.toml"
    _write_manifest(path)

    manifest = load_suite_manifest(path, marker="contract")

    assert manifest.live_files == frozenset({"tests/test_alpha.py"})
    nodeids = ["tests/test_alpha.py::test_ok", "tests/test_alpha.py::test_bad"]
    assert_live_family_counts(manifest, nodeids)
    assert_negative_canaries_collected(manifest, nodeids)


def test_suite_meta_negative_canaries_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "suite.toml"
    _write_manifest(path, minimum=2)
    manifest = load_suite_manifest(path)

    with pytest.raises(AssertionError, match="alpha"):
        assert_live_family_counts(manifest, ["tests/test_alpha.py::test_one"])
    with pytest.raises(AssertionError, match="test_bad"):
        assert_negative_canaries_collected(manifest, ["tests/test_alpha.py::test_one"])


def test_negative_canary_helper_requires_rejection() -> None:
    error = assert_negative_canary(
        lambda: (_ for _ in ()).throw(ValueError("rejected")),
        expected_exception=ValueError,
    )
    assert str(error) == "rejected"
    with pytest.raises(AssertionError, match="accepted"):
        assert_negative_canary(lambda: None)


@pytest.mark.parametrize(
    "content, message",
    [
        ("manifest_version = 2\nmarker = 'x'\nfamilies = []\n", "manifest_version"),
        ("manifest_version = 1\nmarker = 'x'\nfamilies = {}\n", "families"),
    ],
)
def test_load_suite_manifest_rejects_malformed_schema(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "suite.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_suite_manifest(path)


def test_collect_contract_nodeids_uses_explicit_root_and_policy() -> None:
    calls: list[tuple[list[str], Path]] = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs["cwd"]))  # type: ignore[arg-type]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="tests/test_alpha.py::test_ok\n1 test collected\n",
            stderr="",
        )

    result = collect_contract_nodeids(
        rootdir=Path("/downstream"), marker="contract", runner=runner
    )

    assert result.nodeids == ("tests/test_alpha.py::test_ok",)
    assert calls[0][1] == Path("/downstream")
    assert calls[0][0][0:3] == [str(Path(__import__("sys").executable)), "-m", "pytest"]


def test_collect_contract_nodeids_rejects_collection_failure_and_skip_text() -> None:
    def failed(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="boom")

    with pytest.raises(AssertionError, match="boom"):
        collect_contract_nodeids(rootdir="/repo", marker="contract", runner=failed)

    def skipped(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="1 SKIPPED", stderr="")

    with pytest.raises(AssertionError, match="skipped"):
        collect_contract_nodeids(rootdir="/repo", marker="contract", runner=skipped)


def test_contract_suite_hooks_restrict_paths_and_enforce_marks(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    live = tests_dir / "test_alpha.py"
    live.touch()
    other = tests_dir / "test_other.py"
    other.touch()
    manifest_path = tmp_path / "suite.toml"
    _write_manifest(manifest_path)
    hooks = ContractSuiteHooks(root=tmp_path, manifest_path=manifest_path, marker="contract")

    class Option:
        markexpr = "contract"

    class Config:
        option = Option()

    assert hooks.pytest_ignore_collect(live, Config()) is False  # type: ignore[arg-type]
    assert hooks.pytest_ignore_collect(other, Config()) is True  # type: ignore[arg-type]

    class Marker:
        def __init__(self, name: str, **kwargs: object) -> None:
            self.name = name
            self.kwargs = kwargs

    class Item:
        nodeid = "tests/test_alpha.py::test_nope"

        def get_closest_marker(self, name: str) -> Marker | None:
            return Marker(name)

        def iter_markers(self) -> list[Marker]:
            return [Marker("skip")]

    with pytest.raises(pytest.UsageError, match="may not be skipped"):
        hooks.pytest_collection_modifyitems([Item()])  # type: ignore[list-item]
