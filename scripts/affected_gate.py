"""Run affected checks or the complete Feedbax and Studio reference gate."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
HEAVY_LOCK = Path.home() / ".cache" / "agent-full-suite.lock"


def run(command: list[str], *, cwd: Path = ROOT, env: dict[str, str], timeout: int = 60) -> None:
    print("Running:", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd, env=env, check=False, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"command exited {result.returncode}: {' '.join(command)}")


def selection(base: str, env: dict[str, str]) -> dict:
    result = subprocess.run(
        ["uv", "run", "--no-sync", "python", "-m", "scripts.affected_tests", "--base", base],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "affected-test selection failed")
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("affected-test selection returned invalid JSON") from exc
    if not isinstance(report, dict):
        raise RuntimeError("affected-test selection returned an invalid report")
    if report.get("mode") not in {"candidate", "full"}:
        raise RuntimeError("affected-test selection returned an unknown mode")
    return report


def reference(env: dict[str, str], profile_json: Path | None, python_junit: Path | None) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        timeout=10,
    )
    if status.returncode or status.stdout:
        raise RuntimeError("complete reference requires a clean checkout for wheel provenance")
    with TemporaryDirectory(prefix="feedbax-reference-") as directory:
        run(
            [
                "uv",
                "run",
                "--no-sync",
                "python",
                "scripts/run_external_conformance.py",
                "--result-out",
                str(Path(directory) / "downstream-conformance.json"),
            ],
            env=env,
            timeout=120,
        )
        run(["uv", "run", "--no-sync", "python", "scripts/generate_studio_contracts.py"], env=env)
        run(
            ["git", "diff", "HEAD", "--exit-code", "--", "web/src/generated/studioContracts.ts"],
            env=env,
        )
        python = ["scripts/full_suite.sh", "--no-memo"]
        if profile_json is not None:
            python.extend(["--profile-json", str(profile_json)])
        if python_junit is not None:
            python.append(f"--junitxml={python_junit}")
        run(python, env=env, timeout=420)
        run(["bun", "run", "test"], cwd=ROOT / "web", env=env)
        run(["bun", "run", "test:e2e"], cwd=ROOT / "web", env=env)
        run(["bun", "run", "build"], cwd=ROOT / "web", env=env)


def affected(
    base: str, env: dict[str, str], profile_json: Path | None, python_junit: Path | None
) -> None:
    report = selection(base, env)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if report["mode"] == "full":
        reference(env, profile_json, python_junit)
        return
    python_tests = report.get("python_tests")
    studio_tests = report.get("studio_unit_tests")
    browser = report.get("studio_build_and_browser")
    if (
        not isinstance(python_tests, list)
        or not isinstance(studio_tests, list)
        or not isinstance(browser, bool)
    ):
        raise RuntimeError("candidate affected-test report has an invalid shape")
    if python_tests:
        python = ["scripts/full_suite.sh", "--affected-base", base, "--no-memo"]
        if profile_json is not None:
            python.extend(["--profile-json", str(profile_json)])
        if python_junit is not None:
            python.append(f"--junitxml={python_junit}")
        run(python, env=env, timeout=420)
    if studio_tests:
        files = [str(Path(path).relative_to("web")) for path in studio_tests]
        run(["bun", "--bun", "vitest", "run", *files], cwd=ROOT / "web", env=env)
    if browser:
        run(["bun", "run", "test:e2e"], cwd=ROOT / "web", env=env)
        run(["bun", "run", "build"], cwd=ROOT / "web", env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--base", help="Select changes since this Git commit.")
    mode.add_argument("--reference", action="store_true", help="Run every reference gate.")
    parser.add_argument("--profile-json", type=Path, help="Write the Python timing profile.")
    parser.add_argument("--python-junit", type=Path, help="Write the executed Python test cases.")
    parser.add_argument(
        "--plan", action="store_true", help="Print selection without running gates."
    )
    args = parser.parse_args()
    env = os.environ.copy()
    env.setdefault("FEEDBAX_FULL_SUITE_XDIST_WORKERS", "10")
    env.setdefault("FEEDBAX_TEST_NATIVE_THREADS", "1")
    if args.plan:
        report = (
            {"mode": "full", "reason": "complete reference requested"}
            if args.reference
            else selection(args.base, env)
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    HEAVY_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with HEAVY_LOCK.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"shared heavy-suite lock is busy: {HEAVY_LOCK}", file=sys.stderr)
            return 75
        try:
            if args.reference:
                reference(env, args.profile_json, args.python_junit)
            else:
                affected(args.base, env, args.profile_json, args.python_junit)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            print(f"affected gate failed: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
