"""Plan independently affected checks against the working tree."""

from __future__ import annotations

import argparse
import ast
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
from typing import Literal, TypedDict
from urllib.request import urlopen


PYTHON_ROOTS = {"feedbax", "scripts", "tests"}
STUDIO_TEST = re.compile(r"\.test\.[tj]sx?$")
SHARED_STUDIO_SOURCES = (
    "web/src/api/",
    "web/src/data/components.ts",
    "web/src/features/graph/",
    "web/src/features/schema/",
    "web/src/types/graph.ts",
    "web/src/utils/canonicalJsonV2.ts",
    "web/src/utils/studioDraftHash.ts",
)
PANTS_LAUNCHER_VERSION = "v0.13.2"
PANTS_LAUNCHERS = {
    ("Darwin", "arm64"): (
        "macos-aarch64",
        "a6f3231413ca1f793caffa621171a4b1a0158e7488cd0b5bb3e742cb99cc72a8",
    ),
    ("Linux", "x86_64"): (
        "linux-x86_64",
        "74a1e53bc50d6ef6ce1bc67bd9f7b48e549505e0a2453ad4d5ccbc72b0bea874",
    ),
    ("Linux", "aarch64"): (
        "linux-aarch64",
        "b40b60e50e9cb69e13029e100be995fbfdb3b3799ef1ccff60a81177f78e6b82",
    ),
}


@dataclass(frozen=True, order=True)
class Change:
    path: str
    status: str


class TestGate(TypedDict):
    scope: Literal["none", "selected", "all"]
    tests: list[str]
    reason: str


class GatePlan(TypedDict):
    python: TestGate
    studio: TestGate
    browser: bool
    build: bool
    contracts: bool
    clean_wheel: bool
    capabilities: list[str]


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
    if (
        destination.is_file()
        and hashlib.sha256(destination.read_bytes()).hexdigest() == expected_sha256
    ):
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
    _validate_first_party_imports(root, paths)
    command = [
        pants,
        "--no-pantsrc",
        "--level=error",
        "dependents",
        "--format=json",
        "::",
    ]
    env = {key: value for key, value in os.environ.items() if not key.startswith("PANTS_")}
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, check=True, timeout=55)
    addresses = json.loads(result.stdout)
    if not isinstance(addresses, dict) or any(
        not isinstance(values, list) or any(not isinstance(value, str) for value in values)
        for values in addresses.values()
    ):
        raise ValueError("Pants returned an invalid dependents graph")
    owned = {address.partition(":")[0] for address in addresses}
    if not changed <= owned:
        raise ValueError("Pants did not resolve all changed Python files")
    file_graph = {
        source.partition(":")[0]: {
            path for value in values if (path := value.partition(":")[0]) in paths
        }
        for source, values in addresses.items()
        if source.partition(":")[0] in paths
    }
    selected: set[str] = set()
    for source in changed:
        visited = {source}
        frontier = [source]
        while frontier:
            current = frontier.pop()
            for dependent in file_graph.get(current, ()) - visited:
                visited.add(dependent)
                frontier.append(dependent)
        tests = {path for path in visited if _python_test(path)}
        if not tests:
            raise ValueError(f"changed Python source has no test witness: {source}")
        selected.update(tests)
    return sorted(selected)


def _validate_first_party_imports(root: Path, paths: set[str]) -> None:
    modules = {
        (path[:-12] if path.endswith("/__init__.py") else path[:-3]).replace("/", ".")
        for path in paths
    }
    modules.update(PYTHON_ROOTS)

    def owned(name: str) -> bool:
        return name in modules or any(module.startswith(name + ".") for module in modules)

    for path in paths:
        syntax = ast.parse((root / path).read_text(encoding="utf-8"), filename=path)
        package = path.removesuffix("/__init__.py").removesuffix(".py").replace("/", ".")
        if not path.endswith("/__init__.py"):
            package = package.rpartition(".")[0]
        for node in ast.walk(syntax):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = package.split(".")
                if node.level:
                    prefix = prefix[: len(prefix) - node.level + 1]
                else:
                    prefix = []
                names = [".".join((*prefix, node.module or "")).rstrip(".")]
            else:
                continue
            for name in names:
                if name.partition(".")[0] in PYTHON_ROOTS and not owned(name):
                    raise ValueError(
                        f"unresolved first-party import {name} in {path}:{node.lineno}"
                    )


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


def _capabilities(root: Path, tests: list[str]) -> list[str]:
    markers = {"optional_mjx", "optional_ppo"}
    required: set[str] = set()
    for path in tests:
        syntax = ast.parse((root / path).read_text(encoding="utf-8"), filename=path)
        for node in ast.walk(syntax):
            if (
                isinstance(node, ast.Attribute)
                and node.attr in markers
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "mark"
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "pytest"
            ):
                required.add(node.attr)
    return sorted(required)


def _literal_python_path(node: ast.AST, source: Path) -> Path | str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Path"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "__file__"
    ):
        return source
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        path = _literal_python_path(node.value, source)
        return path.parent if isinstance(path, Path) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        parent = _literal_python_path(node.left, source)
        child = _literal_python_path(node.right, source)
        return parent / child if isinstance(parent, Path) and isinstance(child, str) else None
    return None


def _fixture_consumers(root: Path, fixture: str) -> tuple[set[str], set[str]]:
    source = (root / fixture).resolve()
    python: set[str] = set()
    studio: set[str] = set()
    basename = source.name
    for path in (root / "tests").rglob("*.py"):
        contents = path.read_text(encoding="utf-8")
        if basename not in contents:
            continue
        syntax = ast.parse(contents, filename=str(path))
        if any(
            isinstance(resolved, Path) and resolved.resolve() == source
            for node in ast.walk(syntax)
            if (resolved := _literal_python_path(node, path)) is not None
        ):
            python.add(path.relative_to(root).as_posix())
    url_pattern = re.compile(r"new\s+URL\(\s*(['\"])([^'\"]+)\1\s*,\s*import\.meta\.url\s*\)")
    for path in (root / "web" / "src").rglob("*.ts*"):
        contents = path.read_text(encoding="utf-8")
        if basename not in contents:
            continue
        if any(
            (path.parent / relative).resolve() == source
            for _, relative in url_pattern.findall(contents)
        ):
            studio.add(path.relative_to(root).as_posix())
    return python, studio


def _selection(root: Path, changes: list[Change], *, external: bool) -> GatePlan:
    python, studio = set(), set()
    fixtures: set[str] = set()
    python_reason = ""
    studio_reason = ""
    browser = build = contracts = clean_wheel = False
    if external:
        python_reason = studio_reason = "external project or plugin source changed"
        browser = build = contracts = clean_wheel = True
    for path, status in ((change.path, change.status) for change in changes):
        if (
            status not in {"A", "M", "?"}
            or not (root / path).is_file()
            or (root / path).is_symlink()
        ):
            python_reason = studio_reason = f"missing, removed or replaced path: {path}"
            browser = build = contracts = clean_wheel = True
        elif path.startswith("feedbax/") and path.endswith(".py"):
            python.add(path)
            clean_wheel = True
            if path.startswith(
                (
                    "feedbax/compiler/",
                    "feedbax/component_registry/",
                    "feedbax/contracts/",
                    "feedbax/plugins/",
                    "feedbax/runtime/",
                )
            ):
                python_reason = studio_reason = f"shared Python contract: {path}"
                browser = build = contracts = True
        elif _python_test(path):
            python.add(path)
        elif path.startswith("tests/fixtures/") and Path(path).suffix in {".json", ".yaml", ".yml"}:
            fixtures.add(path)
        elif (
            path.startswith("web/src/")
            and not path.startswith(("web/src/generated/", "web/src/test/"))
            and Path(path).suffix in {".ts", ".tsx"}
            and not path.endswith(".d.ts")
        ):
            studio.add(path)
            browser = build = True
            if path.startswith(SHARED_STUDIO_SOURCES) and not _studio_test(path):
                python_reason = studio_reason = f"shared Studio contract: {path}"
                contracts = True
        elif path.startswith("docs/") or path.endswith(".md"):
            continue
        else:
            python_reason = studio_reason = f"shared or unproved dependency: {path}"
            browser = build = contracts = clean_wheel = True
    for fixture in fixtures:
        python_consumers, studio_consumers = _fixture_consumers(root, fixture)
        if not python_consumers:
            python_reason = f"fixture has no proven Python consumer: {fixture}"
        else:
            python.update(python_consumers)
        if not studio_consumers:
            studio_reason = f"fixture has no proven Studio consumer: {fixture}"
        else:
            studio.update(studio_consumers)
    python_tests = _python_related(root, python) if python and not python_reason else []
    studio_tests = _studio_related(root, studio) if studio and not studio_reason else []
    if python and not python_tests and not python_reason:
        python_reason = "changed Python source has no known test witness"
    if studio and not studio_tests and not studio_reason:
        studio_reason = "changed Studio source has no known test witness"
    return {
        "python": {
            "scope": "all" if python_reason else "selected" if python_tests else "none",
            "tests": python_tests,
            "reason": python_reason,
        },
        "studio": {
            "scope": "all" if studio_reason else "selected" if studio_tests else "none",
            "tests": studio_tests,
            "reason": studio_reason,
        },
        "browser": browser,
        "build": build,
        "contracts": contracts,
        "clean_wheel": clean_wheel,
        "capabilities": (
            ["optional_mjx", "optional_ppo"]
            if python_reason
            else _capabilities(root, python_tests)
            if python_tests
            else []
        ),
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
    changes = _changes(input_root, args.replay or args.base, replay=bool(args.replay))
    if args.source_only:
        changes = [
            change
            for change in changes
            if not (_python_test(change.path) or _studio_test(change.path))
        ]
    decision = _selection(root, changes, external=input_root != root)
    print(
        json.dumps(
            {
                "changes": [vars(change) for change in changes],
                **decision,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
