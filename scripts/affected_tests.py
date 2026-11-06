"""Report candidate test files against the working tree without changing the full gate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
import tomllib
from typing import Literal, TypedDict
from urllib.request import urlopen


PPO_SOURCE = "feedbax/training/rl/ppo.py"
PYTHON_ROOTS = {"feedbax", "scripts", "tests"}
STUDIO_TEST = re.compile(r"\.test\.[tj]sx?$")
PANTS_VERSION = "2.33.0"
PANTS_LAUNCHER_VERSION = "v0.13.2"
PANTS_LAUNCHERS = {
    ("Darwin", "arm64"): ("macos-aarch64", "a6f3231413ca1f793caffa621171a4b1a0158e7488cd0b5bb3e742cb99cc72a8"),
    ("Linux", "x86_64"): ("linux-x86_64", "74a1e53bc50d6ef6ce1bc67bd9f7b48e549505e0a2453ad4d5ccbc72b0bea874"),
    ("Linux", "aarch64"): ("linux-aarch64", "b40b60e50e9cb69e13029e100be995fbfdb3b3799ef1ccff60a81177f78e6b82"),
}


@dataclass(frozen=True, order=True)
class Change:
    path: str
    status: str


class FullSelection(TypedDict):
    mode: Literal["full"]
    reason: str


class CandidateSelection(TypedDict):
    mode: Literal["candidate"]
    python_tests: list[str]
    studio_unit_tests: list[str]
    studio_build_and_browser: bool


def _git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args], timeout=10)


def _changes(root: Path, base: str, *, replay: bool) -> list[Change]:
    ref = os.fsdecode(_git(root, "rev-parse", "--verify", "--end-of-options", f"{base}^{{commit}}"))
    args = ("diff", "--no-renames", "--name-status", "-z")
    if replay:
        raw = _git(root, *args, f"{ref.strip()}^", ref.strip(), "--")
    else:
        raw = _git(root, *args, ref.strip(), "--")
    parts = os.fsdecode(raw).split("\0")[:-1]
    changes = [Change(path, status) for status, path in zip(parts[::2], parts[1::2], strict=True)]
    if not replay:
        raw = _git(root, "ls-files", "--others", "--exclude-standard", "-z")
        changes.extend(Change(path, "?") for path in os.fsdecode(raw).split("\0")[:-1])
    return sorted(set(changes))


def _python_test(path: str) -> bool:
    return path.startswith("tests/") and Path(path).match("test_*.py")


def _studio_test(path: str) -> bool:
    return path.startswith("web/src/") and STUDIO_TEST.search(path) is not None


def _pants_launcher() -> str:
    if pants := shutil.which("pants"):
        return pants
    platform_key = (platform.system(), platform.machine())
    if platform_key not in PANTS_LAUNCHERS:
        raise ValueError(f"Pants launcher does not support {platform_key}")
    asset, expected_sha256 = PANTS_LAUNCHERS[platform_key]
    destination = (
        Path.home() / ".cache" / "feedbax" / "pants-launcher" / PANTS_LAUNCHER_VERSION / "pants"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == expected_sha256:
        return str(destination)
    url = (
        f"https://github.com/pantsbuild/scie-pants/releases/download/"
        f"{PANTS_LAUNCHER_VERSION}/scie-pants-{asset}"
    )
    with TemporaryDirectory(prefix="pants-launcher-", dir=destination.parent) as temporary:
        candidate = Path(temporary) / "pants"
        digest = hashlib.sha256()
        with urlopen(url, timeout=30) as source, candidate.open("wb") as output:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("downloaded Pants launcher did not match its pinned SHA-256")
        candidate.chmod(0o755)
        candidate.replace(destination)
    return str(destination)


def _python_related(root: Path, changed: set[str]) -> list[str]:
    raw = _git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    paths = {
        path
        for path in os.fsdecode(raw).split("\0")[:-1]
        if path.endswith(".py")
        and path.partition("/")[0] in PYTHON_ROOTS
        and not path.startswith("tests/fixtures/")
        and (root / path).is_file()
    }
    pants = _pants_launcher()
    with (root / "pyproject.toml").open("rb") as config:
        requires_python = tomllib.load(config)["project"]["requires-python"]
    with TemporaryDirectory(prefix="feedbax-test-graph-") as directory:
        graph = Path(directory)
        for path in paths:
            destination = graph / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / path, destination)
        (graph / "pants.toml").write_text(
            f"""[GLOBAL]
pants_version = "{PANTS_VERSION}"
backend_packages = ["pants.backend.python", "pants.backend.project_info"]
pantsd = false

[source]
root_patterns = ["/"]

[python]
interpreter_constraints = [{json.dumps(requires_python)}]

[python-infer]
init_files = "always"
unowned_dependency_behavior = "warning"
""",
            encoding="utf-8",
        )
        for package in PYTHON_ROOTS:
            build = 'python_sources(sources=["**/*.py"])\n'
            if package == "tests":
                build = (
                    'python_tests(name="tests", sources=["**/test_*.py"])\n'
                    'python_sources(name="support", sources=["**/*.py", "!**/test_*.py"])\n'
                )
            (graph / package).mkdir(exist_ok=True)
            (graph / package / "BUILD").write_text(build, encoding="utf-8")
        command = [
            pants,
            "--no-pantsrc",
            "dependents",
            "--transitive",
            "--closed",
            "--format=json",
            *sorted(changed),
        ]
        env = {key: value for key, value in os.environ.items() if not key.startswith("PANTS_")}
        result = subprocess.run(
            command, cwd=graph, env=env, capture_output=True, check=True, timeout=55
        )
    # The first-party graph intentionally omits third-party owners. Preserve local diagnostics.
    diagnostics = result.stderr.decode(errors="replace")
    for module in re.findall(r"^\s+\* ([\w.]+) \(line: \d+\)$", diagnostics, re.M):
        if module.partition(".")[0] in PYTHON_ROOTS:
            raise ValueError(f"Pants could not resolve first-party import {module}")
    if "ambiguous dependency inference" in diagnostics:
        raise ValueError("Pants reported ambiguous dependency inference")
    addresses = json.loads(result.stdout)
    if not isinstance(addresses, dict) or any(
        not isinstance(values, list) or any(not isinstance(value, str) for value in values)
        for values in addresses.values()
    ):
        raise ValueError("Pants returned an invalid dependents graph")
    owned = {address.partition(":")[0] for address in addresses}
    if not changed <= owned:
        raise ValueError("Pants did not resolve all changed Python files")
    selected: set[str] = set()
    for source in changed:
        tests = {
            path
            for address, values in addresses.items()
            if address.partition(":")[0] == source
            for value in values
            if (path := value.partition(":")[0]) in paths and _python_test(path)
        }
        if not tests:
            raise ValueError(f"no Python test witness for {source}")
        selected.update(tests)
    return sorted(selected)


def _studio_related(root: Path, paths: set[str]) -> list[str]:
    command = ["bun", "scripts/affected-tests.ts", *(str(root / path) for path in sorted(paths))]
    raw = subprocess.check_output(command, cwd=root / "web", timeout=20)
    files = json.loads(raw)
    if not isinstance(files, list) or any(
        not isinstance(path, str) or not _studio_test(path) or not (root / path).is_file()
        for path in files
    ):
        raise ValueError("Vitest returned an invalid test-file list")
    if any(_studio_test(path) and path not in files for path in paths):
        raise ValueError("Vitest did not collect a changed Studio test file")
    return sorted(set(files))


def _selection(
    root: Path, changes: list[Change], *, external: bool
) -> FullSelection | CandidateSelection:
    if external:
        return {"mode": "full", "reason": "external project or plugin source changed"}
    if any(
        change.path.startswith(
            (
                "feedbax/compiler/",
                "feedbax/component_registry/",
                "feedbax/contracts/",
                "feedbax/plugins/",
                "feedbax/runtime/",
                "web/src/generated/",
            )
        )
        for change in changes
    ):
        return {"mode": "full", "reason": "shared graph, registry or generated contract changed"}
    python, studio = set(), set()
    for path, status in ((change.path, change.status) for change in changes):
        if (
            status not in {"A", "M", "?"}
            or not (root / path).is_file()
            or (root / path).is_symlink()
        ):
            return {"mode": "full", "reason": f"missing, removed or replaced path: {path}"}
        if path == PPO_SOURCE or _python_test(path):
            python.add(path)
        elif (
            path.startswith("web/src/")
            and not path.startswith(("web/src/generated/", "web/src/test/"))
            and Path(path).suffix in {".ts", ".tsx"}
            and not path.endswith(".d.ts")
        ):
            studio.add(path)
        else:
            return {"mode": "full", "reason": f"shared or unproved dependency: {path}"}
    try:
        python_tests = _python_related(root, python) if python else []
        studio_tests = _studio_related(root, studio) if studio else []
    except (KeyError, OSError, ValueError, subprocess.SubprocessError) as exc:
        return {"mode": "full", "reason": f"dependency inference failed: {exc}"}
    if (python and not python_tests) or (studio and not studio_tests):
        return {"mode": "full", "reason": "changed source has no known test witness"}
    return {
        "mode": "candidate",
        "python_tests": python_tests,
        "studio_unit_tests": studio_tests,
        "studio_build_and_browser": bool(studio),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--base", help="Compare this Git commit to the working tree, including untracked files."
    )
    mode.add_argument(
        "--replay",
        help="Read changed paths from a historical commit and select against the current tree.",
    )
    parser.add_argument(
        "--replay-repo", type=Path, help="Repository supplying historical changed paths."
    )
    parser.add_argument(
        "--source-only", action="store_true", help="Omit changed tests from a historical replay."
    )
    args = parser.parse_args()
    if not args.replay and (args.replay_repo or args.source_only):
        parser.error("--replay-repo and --source-only require --replay")
    root = Path(__file__).resolve().parents[1]
    input_root = args.replay_repo.resolve() if args.replay_repo else root
    changes: list[Change] = []
    try:
        changes = _changes(input_root, args.replay or args.base, replay=bool(args.replay))
        if args.source_only:
            changes = [
                change
                for change in changes
                if not (_python_test(change.path) or _studio_test(change.path))
            ]
        decision = _selection(root, changes, external=input_root != root)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        decision = {"mode": "full", "reason": f"change inventory unavailable: {exc}"}
    print(
        json.dumps(
            {
                "shadow": True,
                "changes": [vars(change) for change in changes],
                **decision,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
