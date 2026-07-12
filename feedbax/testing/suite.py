"""Manifest-driven pytest policy for downstream contract suites."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Any

import pytest


@dataclass(frozen=True)
class ContractFamily:
    """One family enrolled in a downstream contract suite."""

    name: str
    status: str
    expected_collection_pattern: str
    minimum_non_skipped: int
    negative_canaries: tuple[str, ...] = ()

    @property
    def collection_file(self) -> str:
        """Return the file portion of the expected pytest node-id pattern."""
        return self.expected_collection_pattern.split("::", maxsplit=1)[0]


@dataclass(frozen=True)
class ContractSuiteManifest:
    """Validated version-one contract-suite manifest."""

    marker: str
    families: tuple[ContractFamily, ...]
    manifest_version: int = 1

    @property
    def live_families(self) -> tuple[ContractFamily, ...]:
        """Return families whose status is ``live``."""
        return tuple(family for family in self.families if family.status == "live")

    @property
    def live_files(self) -> frozenset[str]:
        """Return files needed to collect all live families."""
        return frozenset(family.collection_file for family in self.live_families)


@dataclass(frozen=True)
class CollectionResult:
    """Result of a subprocess contract-suite collection."""

    nodeids: tuple[str, ...]
    stdout: str
    stderr: str


def load_suite_manifest(path: Path | str, *, marker: str | None = None) -> ContractSuiteManifest:
    """Load and validate a contract-suite TOML manifest from an explicit path."""
    manifest_path = Path(path)
    data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    version = data.get("manifest_version")
    if version != 1:
        raise ValueError(f"{manifest_path}: unsupported manifest_version {version!r}; expected 1")
    manifest_marker = _required_string(data, "marker", context=str(manifest_path))
    if marker is not None and manifest_marker != marker:
        raise ValueError(
            f"{manifest_path}: marker is {manifest_marker!r}, expected caller marker {marker!r}"
        )
    raw_families = data.get("families")
    if not isinstance(raw_families, list):
        raise ValueError(f"{manifest_path}: families must be an array of tables")
    families = tuple(
        _load_family(raw, context=f"{manifest_path}: families[{index}]")
        for index, raw in enumerate(raw_families)
    )
    if len({family.name for family in families}) != len(families):
        raise ValueError(f"{manifest_path}: family names must be unique")
    return ContractSuiteManifest(marker=manifest_marker, families=families)


class ContractSuiteHooks:
    """Pytest hooks suitable for thin delegation from a downstream conftest.

    A downstream binding can instantiate this class and expose its two methods::

        hooks = ContractSuiteHooks(root=ROOT, manifest_path=MANIFEST, marker="contract")
        pytest_ignore_collect = hooks.pytest_ignore_collect
        pytest_collection_modifyitems = hooks.pytest_collection_modifyitems
    """

    def __init__(self, *, root: Path | str, manifest_path: Path | str, marker: str) -> None:
        self.root = Path(root).resolve()
        self.manifest_path = Path(manifest_path)
        self.marker = marker
        self.manifest = load_suite_manifest(self.manifest_path, marker=marker)

    def pytest_ignore_collect(self, collection_path: Path, config: pytest.Config) -> bool:
        """Restrict marker-only runs to files enrolled as live in the manifest."""
        markexpr = getattr(config.option, "markexpr", "") or ""
        if self.marker not in markexpr:
            return False
        try:
            relpath = Path(collection_path).resolve().relative_to(self.root).as_posix()
        except ValueError:
            return False
        path = Path(relpath)
        live_files = self.manifest.live_files
        if Path(collection_path).is_dir():
            return not any(_path_is_under(Path(live_file), path) for live_file in live_files)
        return relpath not in live_files

    def pytest_collection_modifyitems(self, items: list[pytest.Item]) -> None:
        """Reject skips and non-strict xfails on items carrying the suite marker."""
        violations: list[str] = []
        for item in items:
            if item.get_closest_marker(self.marker) is None:
                continue
            for item_marker in item.iter_markers():
                if item_marker.name in {"skip", "skipif"}:
                    violations.append(f"{item.nodeid}: {self.marker} tests may not be skipped")
                if item_marker.name == "xfail" and item_marker.kwargs.get("strict") is not True:
                    violations.append(
                        f"{item.nodeid}: {self.marker} xfail marks must set strict=True"
                    )
        if violations:
            raise pytest.UsageError("\n".join(violations))


def assert_live_family_counts(
    manifest: ContractSuiteManifest,
    nodeids: Iterable[str],
) -> None:
    """Assert every live family meets its non-skipped collection floor."""
    live_families = manifest.live_families
    assert live_families, "contract suite manifest declares zero live families"
    collected = tuple(nodeids)
    for family in live_families:
        count = sum(family.expected_collection_pattern in nodeid for nodeid in collected)
        assert count >= family.minimum_non_skipped, (
            f"contract family {family.name!r} collected {count} tests; minimum is "
            f"{family.minimum_non_skipped} for pattern {family.expected_collection_pattern!r}"
        )


def assert_negative_canaries_collected(
    manifest: ContractSuiteManifest,
    nodeids: Iterable[str],
) -> None:
    """Assert every declared negative-canary node id was collected."""
    collected = set(nodeids)
    missing = [
        f"{family.name}: {canary}"
        for family in manifest.live_families
        for canary in family.negative_canaries
        if canary not in collected
    ]
    assert not missing, "negative canaries were not collected:\n" + "\n".join(missing)


def assert_negative_canary(
    canary: Callable[[], Any],
    *,
    expected_exception: type[BaseException] | tuple[type[BaseException], ...] = AssertionError,
) -> BaseException:
    """Assert a negative-canary callable is rejected by the expected gate error."""
    try:
        canary()
    except expected_exception as exc:
        return exc
    except BaseException as exc:
        raise AssertionError(
            f"negative canary raised unexpected {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError("negative canary was accepted")


def collect_contract_nodeids(
    *,
    rootdir: Path | str,
    marker: str,
    python_executable: Path | str = sys.executable,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    extra_args: Sequence[str] = (),
) -> CollectionResult:
    """Collect a marked suite in a subprocess rooted at an explicit directory."""
    command = [
        str(python_executable),
        "-m",
        "pytest",
        "-m",
        marker,
        "--strict-markers",
        "--collect-only",
        "-q",
        "-o",
        "xfail_strict=true",
        "-o",
        "empty_parameter_set_mark=fail_at_collect",
        *extra_args,
    ]
    result = runner(
        command,
        cwd=Path(rootdir),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(result.stdout + result.stderr)
    output = result.stdout + result.stderr
    assert "SKIPPED" not in output, "contract collection reported skipped tests\n" + output
    assert "XFAIL" not in output, "contract collection reported xfailed tests\n" + output
    nodeids = tuple(
        line.strip()
        for line in result.stdout.splitlines()
        if "::" in line and not line.startswith("<")
    )
    return CollectionResult(nodeids=nodeids, stdout=result.stdout, stderr=result.stderr)


def _load_family(raw: Any, *, context: str) -> ContractFamily:
    if not isinstance(raw, dict):
        raise ValueError(f"{context}: family must be a table")
    minimum = raw.get("minimum_non_skipped")
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 0:
        raise ValueError(f"{context}: minimum_non_skipped must be a non-negative integer")
    status = _required_string(raw, "status", context=context)
    if status not in {"live", "pending", "retired"}:
        raise ValueError(f"{context}: unsupported status {status!r}")
    raw_canaries = raw.get("negative_canary")
    if raw_canaries is None:
        canaries = ()
    elif isinstance(raw_canaries, str):
        canaries = (raw_canaries,)
    elif isinstance(raw_canaries, list) and all(isinstance(value, str) for value in raw_canaries):
        canaries = tuple(raw_canaries)
    else:
        raise ValueError(f"{context}: negative_canary must be a string or array of strings")
    return ContractFamily(
        name=_required_string(raw, "name", context=context),
        status=status,
        expected_collection_pattern=_required_string(
            raw, "expected_collection_pattern", context=context
        ),
        minimum_non_skipped=minimum,
        negative_canaries=canaries,
    )


def _required_string(data: dict[str, Any], key: str, *, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: {key} must be a non-empty string")
    return value


def _path_is_under(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents
