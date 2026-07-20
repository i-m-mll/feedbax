"""Local subprocess orchestration driver."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from feedbax.orchestration.bundle import ResolvedAssemblyInput, RunBundle, RunRowSpec
from feedbax.orchestration.drivers.base import DriverRowProbe
from feedbax.orchestration.drivers.native_execution import (
    bind_native_execution_command,
    inject_native_execution_context,
    native_resume_checkpoint_source,
    seed_authenticated_checkpoint,
)
from feedbax.orchestration.input_materialization import (
    InputMaterializationError,
    InputProviderRootBinding,
    materialize_bundle_inputs,
)
from feedbax.orchestration.state import RunSetState
from feedbax.training import publish_directory_no_replace


class LocalDriverError(RuntimeError):
    """Raised when the local driver cannot complete a requested action."""


_DEPENDENCY_INVENTORY_SCRIPT = """
import importlib.metadata
import json
import platform
import sys

inventory = []
for distribution in importlib.metadata.distributions():
    name = distribution.metadata["Name"]
    version = distribution.version
    if not name or not version:
        raise RuntimeError("installed distribution lacks required name or version metadata")
    direct_url_text = distribution.read_text("direct_url.json")
    direct_url = json.loads(direct_url_text) if direct_url_text is not None else None
    if direct_url is not None and not isinstance(direct_url, dict):
        raise RuntimeError(f"distribution {name!r} has invalid direct_url.json")
    inventory.append({"direct_url": direct_url, "name": name, "version": version})
payload = {
    "schema_version": "feedbax.local_dependency_inventory.v1",
    "interpreter": {
        "cache_tag": sys.implementation.cache_tag,
        "executable": sys.executable,
        "implementation": sys.implementation.name,
        "version": platform.python_version(),
    },
    "distributions": inventory,
}
print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
"""


class LocalOrchestrationDriver:
    """Run orchestration rows as local subprocesses under the run-set directory."""

    def __init__(
        self,
        *,
        cwd: Path | str | None = None,
        python_executable: str | None = None,
        freeze_lines: Sequence[str] | None = None,
        input_provider_bindings: Sequence[InputProviderRootBinding] = (),
        one_update: bool = False,
    ) -> None:
        self.cwd = Path(cwd or Path.cwd())
        self.python_executable = python_executable or sys.executable
        self.freeze_lines = tuple(freeze_lines) if freeze_lines is not None else None
        self.input_provider_bindings = tuple(input_provider_bindings)
        self.one_update = one_update
        self._processes: dict[str, subprocess.Popen[bytes]] = {}

    def provision(self, bundle: RunBundle, state: RunSetState) -> Mapping[str, Any]:
        run_set_dir = bundle.run_set_dir
        for dirname in ("events", "sentinels", "rows", "collected"):
            (run_set_dir / dirname).mkdir(parents=True, exist_ok=True)
        return {"driver": "local", "run_set_dir": str(run_set_dir)}

    def realize_env(self, bundle: RunBundle, state: RunSetState) -> str:
        return compute_environment_fingerprint(
            bundle,
            cwd=self.cwd,
            python_executable=self.python_executable,
            freeze_lines=self.freeze_lines,
        )

    def stage_inputs(self, bundle: RunBundle, state: RunSetState) -> Mapping[str, Any]:
        inputs_dir = bundle.run_set_dir / "inputs"
        if os.path.lexists(inputs_dir):
            raise LocalDriverError(f"input destination already exists: {inputs_dir}")
        attempt_root = bundle.run_set_dir / ".stage-attempts" / (
            f"stage-inputs-{state.stage('STAGE_INPUTS').attempts}")
        try:
            staged_inputs = materialize_bundle_inputs(
                bundle,
                destination_root=attempt_root,
                provider_bindings=self.input_provider_bindings,
            )
        except InputMaterializationError as exc:
            raise LocalDriverError(str(exc)) from exc
        payloads: list[dict[str, str]] = []
        for row in bundle.rows:
            if row.launch.payload_routing.get("kind") != "registered-execution-payload":
                continue
            source = Path(row.execution.payload.uri or "")
            if not source.is_file():
                raise LocalDriverError(
                    f"registered execution payload is not materialized for row {row.row_id!r}"
                )
            if _sha256_file(source) != row.execution.payload.sha256:
                raise LocalDriverError(
                    f"registered execution payload digest mismatch for row {row.row_id!r}"
                )
            target = attempt_root / "inputs" / f"{row.row_id}.json"
            shutil.copy2(source, target)
            payloads.append({"row_id": row.row_id, "source": str(source),
                             "target": str(inputs_dir / target.name)})
        parent_descriptor = os.open(bundle.run_set_dir, os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            source_name = (attempt_root / "inputs").relative_to(bundle.run_set_dir).as_posix()
            publish_directory_no_replace(parent_descriptor, source_name, "inputs",
                expected_identity=os.stat(source_name, dir_fd=parent_descriptor,
                                          follow_symlinks=False))
        finally:
            os.close(parent_descriptor)
        return {
            "input_count": len(staged_inputs),
            "inputs": [
                {
                    "target_role": staged.target_role,
                    "destination": str(inputs_dir / staged.target_role),
                    "files": [asdict(item) for item in staged.files],
                }
                for staged in staged_inputs
            ],
            "inputs_dir": str(inputs_dir),
            "payload_count": len(payloads),
            "payloads": payloads,
        }

    def launch_row(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        state: RunSetState,
    ) -> Mapping[str, Any]:
        paths = _row_paths(bundle, row.row_id)
        if paths["done"].exists():
            return {"row_id": row.row_id, "status": "completed"}
        if paths["failed"].exists():
            return {"row_id": row.row_id, "status": "failed"}
        if row.row_id in self._processes and self._processes[row.row_id].poll() is None:
            return {"row_id": row.row_id, "pid": self._processes[row.row_id].pid}
        if paths["started"].exists():
            pid = _read_pid(paths["pid"])
            if pid and _pid_alive(pid):
                self._processes.pop(row.row_id, None)
                return {
                    "row_id": row.row_id,
                    "pid": pid,
                    "status": "launched",
                    "adopted": True,
                }
            discrepancy = {
                "code": "orphaned_launch",
                "message": (
                    "orphaned launch: started sentinel present, process dead, no terminal sentinel"
                ),
                "pid": pid,
            }
            paths["sentinels"].mkdir(parents=True, exist_ok=True)
            paths["failed"].write_text(discrepancy["message"] + "\n", encoding="utf-8")
            return {
                "row_id": row.row_id,
                "pid": pid,
                "status": "failed",
                "detail": discrepancy["message"],
                "event_discrepancies": [discrepancy],
            }

        paths["row_dir"].mkdir(parents=True, exist_ok=True)
        paths["sentinels"].mkdir(parents=True, exist_ok=True)
        paths["events"].mkdir(parents=True, exist_ok=True)
        checkpoint_source = native_resume_checkpoint_source(bundle, row)
        if checkpoint_source is not None:
            _seed_native_resume_checkpoint_root(
                bundle.run_set_dir / "inputs" / checkpoint_source.custody.target_role,
                paths["row_dir"],
                checkpoint_source,
            )
        paths["started"].write_text(str(time.time()), encoding="utf-8")

        env = os.environ.copy()
        env.update(
            {
                "FEEDBAX_RUN_SET_ID": bundle.run_set_id,
                "FEEDBAX_ROW_ID": row.row_id,
                "FEEDBAX_RUN_EVENTS_DIR": str(paths["events"]),
                "FEEDBAX_ENV_FINGERPRINT": state.environment_fingerprint or "",
                "FEEDBAX_ROW_DIR": str(paths["row_dir"]),
            }
        )
        env["PYTHONPATH"] = _prepend_feedbax_source_root(env.get("PYTHONPATH"))
        command, bound_row = bind_native_execution_command(
            _row_command(row, self.python_executable),
            row=row,
            payload_path=bundle.run_set_dir / "inputs" / f"{row.row_id}.json",
            collection_root=paths["row_dir"],
            one_update=self.one_update,
        )
        command = inject_native_execution_context(
            command,
            row=bound_row,
            environment_fingerprint=state.environment_fingerprint or "",
            collection_root=paths["row_dir"],
        )
        stdout = (paths["row_dir"] / "stdout.log").open("ab")
        stderr = (paths["row_dir"] / "stderr.log").open("ab")
        process = subprocess.Popen(
            command,
            cwd=self.cwd,
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        stdout.close()
        stderr.close()
        self._processes[row.row_id] = process
        paths["pid"].write_text(f"{process.pid}\n", encoding="utf-8")
        return {"row_id": row.row_id, "pid": process.pid, "command": command}

    def probe(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        state: RunSetState,
    ) -> DriverRowProbe:
        paths = _row_paths(bundle, row.row_id)
        process = self._processes.get(row.row_id)
        if process is not None:
            returncode = process.poll()
            if returncode is None:
                return DriverRowProbe(status="running", pid=process.pid)
            if returncode == 0:
                paths["done"].write_text("0\n", encoding="utf-8")
                return DriverRowProbe(status="completed", pid=process.pid)
            paths["failed"].write_text(f"{returncode}\n", encoding="utf-8")
            return DriverRowProbe(status="failed", pid=process.pid, detail=f"exit={returncode}")
        if paths["done"].exists():
            return DriverRowProbe(status="completed", pid=_read_pid(paths["pid"]))
        if paths["failed"].exists():
            return DriverRowProbe(status="failed", pid=_read_pid(paths["pid"]))
        if paths["pid"].exists():
            pid = _read_pid(paths["pid"])
            if pid and _pid_alive(pid):
                return DriverRowProbe(status="running", pid=pid)
            return DriverRowProbe(status="failed", pid=pid, detail="pid exited without sentinel")
        return DriverRowProbe(status="pending")

    def stop_row(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        state: RunSetState,
    ) -> Mapping[str, Any]:
        paths = _row_paths(bundle, row.row_id)
        process = self._processes.get(row.row_id)
        pid = process.pid if process is not None else _read_pid(paths["pid"])
        if pid and _pid_alive(pid):
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError:
                os.kill(pid, signal.SIGTERM)
        paths["failed"].write_text("stopped\n", encoding="utf-8")
        return {"row_id": row.row_id, "pid": pid, "status": "stopped"}

    def request_stop_at_checkpoint(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        state: RunSetState,
    ) -> Mapping[str, Any]:
        """Ask a local row to stop itself at its next durable checkpoint."""
        del state
        paths = _row_paths(bundle, row.row_id)
        process = self._processes.get(row.row_id)
        pid = process.pid if process is not None else _read_pid(paths["pid"])
        if pid and _pid_alive(pid):
            try:
                os.killpg(pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            except OSError:
                os.kill(pid, signal.SIGINT)
        return {"row_id": row.row_id, "pid": pid, "status": "stop_requested"}

    def collect(
        self,
        bundle: RunBundle,
        row: RunRowSpec,
        state: RunSetState,
    ) -> Mapping[str, str]:
        paths = _row_paths(bundle, row.row_id)
        dest_dir = bundle.run_set_dir / "collected" / row.row_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        collected: dict[str, str] = {}
        sources = row.launch.collect or [str(paths["event_log"])]
        for source in sources:
            source_path = Path(source)
            if not source_path.is_absolute():
                source_path = paths["row_dir"] / source_path
            if not source_path.exists():
                continue
            dest = dest_dir / source_path.name
            if source_path.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(source_path, dest)
            else:
                shutil.copy2(source_path, dest)
            collected[source_path.name] = str(dest)
        payload_sha256 = row.launch.metadata.get("payload_sha256")
        if payload_sha256 and collected:
            first = Path(next(iter(collected.values())))
            if first.is_file() and _sha256_file(first) != str(payload_sha256):
                raise LocalDriverError(f"payload sha256 mismatch for row {row.row_id!r}")
        return collected

    def teardown(self, bundle: RunBundle, state: RunSetState) -> Mapping[str, Any]:
        stopped: list[str] = []
        for row in bundle.rows:
            probe = self.probe(bundle, row, state)
            if probe.status == "running":
                self.stop_row(bundle, row, state)
                stopped.append(row.row_id)
        return {"driver": "local", "stopped_rows": stopped}


def compute_environment_fingerprint(
    bundle: RunBundle,
    *,
    cwd: Path | str,
    python_executable: str | None = None,
    freeze_lines: Sequence[str] | None = None,
) -> str:
    """Compute a deterministic local environment fingerprint."""
    root = Path(cwd)
    interpreter, discovered_packages = _probe_dependency_inventory(python_executable)
    package_lines = list(freeze_lines) if freeze_lines is not None else discovered_packages
    payload: dict[str, Any] = {
        "declared_python_version": bundle.environment.python_version,
        "dependency_inventory_source": (
            "provided" if freeze_lines is not None else "importlib.metadata"
        ),
        "interpreter": interpreter,
        "packages": sorted(package_lines),
        "repo_revisions": [],
        "lockfile_hashes": dict(sorted(bundle.environment.lockfile_hashes.items())),
        "overlay_steps": list(bundle.environment.overlay_steps),
        "image_id": bundle.environment.image_id,
    }
    for revision in bundle.environment.repo_revisions:
        repo_path = root / revision.path
        dirty = _git_dirty(repo_path)
        if dirty and not revision.dirty_allowed:
            raise LocalDriverError(f"dirty repo not allowed for fingerprint: {repo_path}")
        payload["repo_revisions"].append(
            {
                "path": revision.path,
                "revision": revision.revision,
                "dirty_allowed": revision.dirty_allowed,
                "head": _git_output(repo_path, "rev-parse", "HEAD") or None,
                "patch_sha256": _git_patch_sha256(repo_path) if dirty else None,
            }
        )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _row_command(row: RunRowSpec, python_executable: str) -> list[str]:
    if row.launch.command:
        return [str(part) for part in row.launch.command]
    assert row.launch.entry is not None
    return [python_executable, row.launch.entry]


def _prepend_feedbax_source_root(existing: str | None) -> str:
    """Make this checkout importable to rows launched from an arbitrary cwd."""
    source_root = str(Path(__file__).resolve().parents[3])
    return source_root if not existing else os.pathsep.join((source_root, existing))


def _row_paths(bundle: RunBundle, row_id: str) -> dict[str, Path]:
    run_set_dir = bundle.run_set_dir
    sentinels = run_set_dir / "sentinels"
    events = run_set_dir / "events"
    row_dir = run_set_dir / "rows" / row_id
    return {
        "sentinels": sentinels,
        "events": events,
        "row_dir": row_dir,
        "started": sentinels / f"{row_id}.started",
        "pid": sentinels / f"{row_id}.pid",
        "done": sentinels / f"{row_id}.done",
        "failed": sentinels / f"{row_id}.failed",
        "event_log": events / f"{row_id}.events.jsonl",
    }


def _probe_dependency_inventory(
    python_executable: str | None,
) -> tuple[dict[str, str | None], list[str]]:
    """Read interpreter identity and distribution provenance from the selected Python."""
    executable = python_executable or sys.executable
    try:
        result = subprocess.run(
            [executable, "-c", _DEPENDENCY_INVENTORY_SCRIPT],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", None)
        detail = str(stderr).strip() if stderr else str(exc)
        raise LocalDriverError(
            f"dependency inventory failed for interpreter {executable!r}: {detail}"
        ) from exc

    try:
        probe = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise LocalDriverError(
            f"dependency inventory from interpreter {executable!r} was not valid JSON"
        ) from exc
    if not isinstance(probe, dict) or set(probe) != {
        "distributions",
        "interpreter",
        "schema_version",
    }:
        raise LocalDriverError(
            f"dependency inventory from interpreter {executable!r} had invalid structure"
        )
    if probe["schema_version"] != "feedbax.local_dependency_inventory.v1":
        raise LocalDriverError(
            f"dependency inventory from interpreter {executable!r} had unknown schema"
        )
    interpreter = probe["interpreter"]
    if (
        not isinstance(interpreter, dict)
        or set(interpreter) != {"cache_tag", "executable", "implementation", "version"}
        or not isinstance(interpreter["executable"], str)
        or not interpreter["executable"]
        or not isinstance(interpreter["implementation"], str)
        or not interpreter["implementation"]
        or not isinstance(interpreter["version"], str)
        or not interpreter["version"]
        or (interpreter["cache_tag"] is not None and not isinstance(interpreter["cache_tag"], str))
    ):
        raise LocalDriverError(
            f"dependency inventory from interpreter {executable!r} had invalid interpreter identity"
        )
    inventory = probe["distributions"]
    return dict(interpreter), _canonicalize_dependency_inventory(inventory, executable=executable)


def _canonicalize_dependency_inventory(inventory: Any, *, executable: str) -> list[str]:
    """Validate and canonicalize importlib distribution records."""
    if not isinstance(inventory, list) or any(
        not isinstance(entry, dict)
        or set(entry) != {"direct_url", "name", "version"}
        or not isinstance(entry["name"], str)
        or not entry["name"]
        or not isinstance(entry["version"], str)
        or not entry["version"]
        or (entry["direct_url"] is not None and not isinstance(entry["direct_url"], dict))
        for entry in inventory
    ):
        raise LocalDriverError(
            f"dependency inventory from interpreter {executable!r} had invalid structure"
        )
    by_name: dict[str, dict[str, Any]] = {}
    for raw_entry in inventory:
        entry = dict(raw_entry)
        entry["name"] = re.sub(r"[-_.]+", "-", entry["name"]).lower()
        previous = by_name.get(entry["name"])
        if previous is not None and previous != entry:
            raise LocalDriverError(
                "dependency inventory contains conflicting distributions for normalized name "
                f"{entry['name']!r}"
            )
        by_name[entry["name"]] = entry
    return [
        json.dumps(by_name[name], sort_keys=True, separators=(",", ":")) for name in sorted(by_name)
    ]


def _git_output(cwd: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _git_dirty(cwd: Path) -> bool:
    return bool(_git_output(cwd, "status", "--porcelain"))


def _git_patch_sha256(cwd: Path) -> str:
    diff = _git_output(cwd, "diff", "--binary")
    untracked = _git_output(cwd, "ls-files", "--others", "--exclude-standard")
    h = hashlib.sha256()
    h.update(diff.encode("utf-8"))
    h.update(untracked.encode("utf-8"))
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _seed_native_resume_checkpoint_root(
    source: Path, row_dir: Path, resolved: ResolvedAssemblyInput
) -> None:
    """Clone one staged checkpoint tree and publish it without replacement."""

    if not source.is_dir() or source.is_symlink():
        raise LocalDriverError(f"native resume checkpoint source is not a directory: {source}")
    target = row_dir / "checkpoints"
    attempt = row_dir / ".checkpoint-seed-attempt"
    if os.path.lexists(target):
        raise LocalDriverError(f"native resume checkpoint target already exists: {target}")
    if os.path.lexists(attempt):
        raise LocalDriverError(f"native resume checkpoint attempt already exists: {attempt}")
    try:
        seed_authenticated_checkpoint(source, attempt, target, resolved)
    except LocalDriverError:
        raise
    except Exception as exc:
        raise LocalDriverError(f"native resume checkpoint seeding failed: {exc}") from exc


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
