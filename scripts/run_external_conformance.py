"""Build and execute both candidate wheels from a clean temporary environment."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "external" / "feedbax_conformance_fixture"


def _run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _capture(command: list[str], *, cwd: Path, env: dict[str, str]) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout


def _check_installed_metadata(
    python_executable: Path,
    *,
    cwd: Path,
    env: dict[str, str],
) -> None:
    _run(
        ["uv", "pip", "check", "--python", str(python_executable)],
        cwd=cwd,
        env=env,
    )


def _load_installed_result(
    python_executable: Path,
    result_path: Path,
    *,
    cwd: Path,
    env: dict[str, str],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    output = _capture(
        [
            str(python_executable),
            "-c",
            (
                "import json, sys; "
                "from pathlib import Path; "
                "from feedbax_external_conformance import REQUIRED_CASE_IDS, load_result; "
                "result = load_result(json.loads(Path(sys.argv[1]).read_text())); "
                "print(json.dumps({'payload': result.model_dump(mode='json'), "
                "'required_case_ids': REQUIRED_CASE_IDS}, sort_keys=True))"
            ),
            str(result_path),
        ],
        cwd=cwd,
        env=env,
    )
    validated = json.loads(output)
    return validated["payload"], tuple(validated["required_case_ids"])


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="feedbax-external-conformance-") as raw:
        work = Path(raw)
        dist = work / "dist"
        venv = work / "venv"
        result = work / "result.json"
        requirements = work / "requirements.txt"
        env = dict(os.environ)
        env["GIT_OPTIONAL_LOCKS"] = "0"

        _run(
            [
                "uv",
                "export",
                "--quiet",
                "--frozen",
                "--no-dev",
                "--extra",
                "analysis",
                "--no-emit-project",
                "--no-hashes",
                "--output-file",
                str(requirements),
            ],
            cwd=ROOT,
            env=env,
        )
        _run(["uv", "build", "--wheel", "--out-dir", str(dist), str(ROOT)], cwd=ROOT, env=env)
        _run(
            ["uv", "build", "--wheel", "--out-dir", str(dist), str(FIXTURE)],
            cwd=ROOT,
            env=env,
        )
        _run(["uv", "venv", "--python", "3.12", str(venv)], cwd=work, env=env)
        wheels = sorted(str(path) for path in dist.glob("*.whl"))
        if len(wheels) != 2:
            raise RuntimeError(f"expected exactly two candidate wheels, found {wheels!r}")
        _run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(venv / "bin" / "python"),
                "--requirement",
                str(requirements),
            ],
            cwd=work,
            env=env,
        )
        _run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(venv / "bin" / "python"),
                "--no-deps",
                *wheels,
            ],
            cwd=work,
            env=env,
        )
        _check_installed_metadata(venv / "bin" / "python", cwd=work, env=env)

        execution_env = dict(env)
        execution_env.pop("PYTHONPATH", None)
        execution_env["UV_CACHE_DIR"] = str(work / "runtime-uv-cache")
        execution_env["XDG_CACHE_HOME"] = str(work / "cache")
        execution_env["FEEDBAX_JAX_COMPILATION_CACHE_DIR"] = str(work / "jax-cache")
        execution_env["FEEDBAX_REPO_SNAPSHOT_CACHE_DIR"] = str(work / "snapshot-cache")
        _run(
            [
                str(venv / "bin" / "python"),
                "-m",
                "feedbax_external_conformance",
                "--source-root",
                str(ROOT),
                "--result",
                str(result),
            ],
            cwd=work,
            env=execution_env,
        )
        payload, required_case_ids = _load_installed_result(
            venv / "bin" / "python",
            result,
            cwd=work,
            env=execution_env,
        )
        if payload["schema_version"] != "feedbax.external_conformance.result.v7":
            raise RuntimeError("external conformance result schema drifted")
        if payload["status"] != "pass":
            raise RuntimeError("clean-wheel external conformance did not pass")
        if payload["lifecycle"]["status"] != "pass":
            raise RuntimeError("clean-wheel external lifecycle did not pass")
        if set(payload["cases"]) != set(required_case_ids):
            raise RuntimeError("external conformance required case set drifted")
        if any(type(value) is not bool or not value for value in payload["cases"].values()):
            raise RuntimeError("one or more external conformance cases failed")
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
