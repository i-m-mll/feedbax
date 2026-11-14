"""Run affected checks or the complete Feedbax and Studio reference gate."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys


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
    for name in ("python", "studio"):
        gate = report.get(name)
        if (
            not isinstance(gate, dict)
            or gate.get("scope") not in {"none", "selected", "all"}
            or not isinstance(gate.get("tests"), list)
            or any(not isinstance(path, str) for path in gate["tests"])
            or not isinstance(gate.get("reason"), str)
        ):
            raise RuntimeError(f"affected-test selection returned an invalid {name} gate")
    capabilities = report.get("capabilities")
    if not isinstance(capabilities, list) or any(
        capability not in {"optional_mjx"} for capability in capabilities
    ):
        raise RuntimeError("affected-test selection returned invalid capability gates")
    if any(
        not isinstance(report.get(name), bool)
        for name in ("browser", "build", "contracts", "clean_wheel")
    ):
        raise RuntimeError("affected-test selection returned invalid auxiliary gates")
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
    base: str,
    env: dict[str, str],
    profile_json: Path | None,
    python_junit: Path | None,
    include_capabilities: bool,
) -> None:
    report = selection(base, env)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    python_gate = report["python"]
    studio_gate = report["studio"]
    if python_gate["scope"] != "none":
        python = ["scripts/full_suite.sh", "--no-memo"]
        if include_capabilities and report["capabilities"]:
            python.append("--include-optional")
        if python_gate["scope"] == "selected":
            python.extend(["--affected-tests", *python_gate["tests"]])
        if profile_json is not None:
            python.extend(["--profile-json", str(profile_json)])
        if python_junit is not None:
            python.append(f"--junitxml={python_junit}")
        run(python, env=env, timeout=420)
    if studio_gate["scope"] == "all":
        run(["bun", "run", "test"], cwd=ROOT / "web", env=env)
    elif studio_gate["scope"] == "selected":
        files = [str(Path(path).relative_to("web")) for path in studio_gate["tests"]]
        run(["bun", "--bun", "vitest", "run", *files], cwd=ROOT / "web", env=env)
    if report["contracts"]:
        run(
            [
                "uv",
                "run",
                "--no-sync",
                "python",
                "-c",
                "from scripts.generate_studio_contracts import OUTPUT, generate; "
                "import sys; sys.exit(0 if OUTPUT.read_text(encoding='utf-8') == generate() else 1)",
            ],
            env=env,
        )
    if report["browser"]:
        run(["bun", "run", "test:e2e"], cwd=ROOT / "web", env=env)
    if report["build"]:
        run(["bun", "run", "build"], cwd=ROOT / "web", env=env)
    if report["clean_wheel"]:
        print("Pending committed-candidate obligation: clean-wheel conformance", flush=True)
    if report["capabilities"] and not include_capabilities:
        print(
            "Pending optional capability checks: " + ", ".join(report["capabilities"]),
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--base", help="Select changes since this Git commit.")
    mode.add_argument("--reference", action="store_true", help="Run every reference gate.")
    parser.add_argument("--profile-json", type=Path, help="Write the Python timing profile.")
    parser.add_argument("--python-junit", type=Path, help="Write the executed Python test cases.")
    parser.add_argument(
        "--include-capabilities",
        action="store_true",
        help="Execute selected optional Python capability checks when available.",
    )
    parser.add_argument(
        "--plan", action="store_true", help="Print selection without running gates."
    )
    args = parser.parse_args()
    if args.reference and args.include_capabilities:
        parser.error(
            "--include-capabilities applies to --base; use full_suite.sh --include-optional for the complete optional suite"
        )
    env = os.environ.copy()
    env.setdefault("FEEDBAX_FULL_SUITE_XDIST_WORKERS", "10")
    env.setdefault("FEEDBAX_TEST_NATIVE_THREADS", "1")
    if args.plan:
        report = (
            {"reference": "complete", "requires_clean_checkout": True}
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
                affected(
                    args.base,
                    env,
                    args.profile_json,
                    args.python_junit,
                    args.include_capabilities,
                )
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            print(f"affected gate failed: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
