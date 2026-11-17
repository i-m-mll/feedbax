"""Detached worker that accepts each delivery before touching its payload."""

import importlib
import json
import os
import shutil
import socket
import sys
import time
import traceback
from pathlib import Path

from .io import EVIDENCE_FILE, TOKEN_ENV, append, encode, read_worker_result, receive, write_json
from .protocol import MANAGED_ENTRYPOINTS, MAX_RPC_BYTES, VALIDITY_CHECK_ENTRYPOINT
from .source_roster import SourceRoster


def _redirect(paths: dict) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    for name, fd in (("stdout", 1), ("stderr", 2)):
        with open(paths[name], "ab", buffering=0) as stream:
            os.dup2(stream.fileno(), fd)


def _sources() -> dict[Path, tuple[int, int]]:
    """Capture loaded source, including the plugins composed during prewarm."""
    snapshots = {}
    for module in list(sys.modules.values()):
        if source := getattr(module, "__file__", None):
            path = Path(source)
            if path in snapshots:
                continue
            try:
                info = path.stat()
            except FileNotFoundError:
                continue
            snapshots[path] = (info.st_mtime_ns, info.st_size)
    return snapshots


def _sources_changed(snapshots: dict[Path, tuple[int, int]]) -> bool:
    for path, original in snapshots.items():
        try:
            info = path.stat()
        except FileNotFoundError:
            return True
        if (info.st_mtime_ns, info.st_size) != original:
            return True
    return False


def main(control_fd: str, report_fd: str, profile_path: str) -> int:
    profile = json.loads(Path(profile_path).read_text())
    control = socket.socket(fileno=int(control_fd))
    report = int(report_fd)
    if profile["warm_entrypoint"] in MANAGED_ENTRYPOINTS | {VALIDITY_CHECK_ENTRYPOINT}:
        boot_ns = time.time_ns()
        with SourceRoster(3, "Feedbax executed source") as entered:
            entered.enable()
            from feedbax.validity.recording import WorkerRecorder

            recorder = WorkerRecorder(entered.names, boot_ns=boot_ns)
            with recorder.monitor():
                return _serve(profile, control, report, recorder)
    return _serve(profile, control, report, None)


def _serve(profile: dict, control: socket.socket, report: int, recorder) -> int:
    entrypoint, import_error = None, None
    try:
        for module in profile["preload"]:
            importlib.import_module(module)
        if ref := profile["warm_entrypoint"]:
            module_name, function = ref.split(":", 1)
            entrypoint = getattr(importlib.import_module(module_name), function)
    except Exception as error:  # noqa: BLE001 -- foreign executor boundary
        import_error = (error, traceback.format_exc())
    snapshots = _sources() if recorder is None else None
    if recorder is not None:
        recorder.note_preloaded_sources()
    reusable = profile.get("light_affinity") is not None
    with control:
        while True:
            try:
                control.sendall(encode({"kind": "ready"}))
                message = receive(control, MAX_RPC_BYTES)
            except (OSError, EOFError):
                return 0
            if (recorder.changed() if recorder is not None else _sources_changed(snapshots)):
                return 0
            if not reusable:
                control.close()
            status = _execute(message, report, entrypoint, import_error, reusable=reusable, recorder=recorder)
            if not reusable:
                return status
            sys.stdout.flush()
            sys.stderr.flush()
            os.write(report, encode({"status": status}))
            if import_error is not None:
                return status
            if snapshots is not None:
                snapshots |= {path: info for path, info in _sources().items() if path not in snapshots}


def _execute(message: dict, report: int, entrypoint, import_error, *, reusable: bool, recorder) -> int:
    paths = message["paths"]
    log = Path(paths["event_log"])
    os.write(report, encode({"event_log": str(log)}))
    if not reusable:
        os.close(report)
    if not write_json(
        log, {"kind": "accepted", "data": {"token": os.environ[TOKEN_ENV]}}, exclusive=True
    ):
        return 0
    with open(message["startup_log"], "rb") as source, open(paths["stderr"], "ab") as target:
        shutil.copyfileobj(source, target)
    _redirect(paths)
    os.environ.update(
        {
            "FEEDBAX_HOST_JOB_ID": message["job_id"],
            "FEEDBAX_HOST_INPUT": paths["input"],
            "FEEDBAX_HOST_EVENT_LOG": paths["event_log"],
            "FEEDBAX_HOST_STOP_FILE": paths["stop_file"],
            "FEEDBAX_HOST_SCRATCH": paths["scratch"],
            "FEEDBAX_HOST_STDOUT": paths["stdout"],
            "FEEDBAX_HOST_STDERR": paths["stderr"],
        }
    )
    os.environ.pop("FEEDBAX_HOST_COMPLETION_BINDINGS", None)
    os.environ.pop("FEEDBAX_HOST_COMPLETION_BINDINGS_SHA256", None)
    if "completion_bindings" in paths:
        os.environ["FEEDBAX_HOST_COMPLETION_BINDINGS"] = paths["completion_bindings"]
        if digest := message.get("completion_bindings_sha256"):
            os.environ["FEEDBAX_HOST_COMPLETION_BINDINGS_SHA256"] = digest
    if "live_evaluate" in paths:
        os.environ["FEEDBAX_HOST_LIVE_EVALUATE"] = paths["live_evaluate"]
        os.environ["FEEDBAX_HOST_LIVE_EVALUATE_SHA256"] = message["live_evaluate_sha256"]
    try:
        expires_at = message.get("expires_at")
        if expires_at is not None and time.time() >= expires_at:
            append(
                log,
                "result",
                {
                    "outcome": "failed",
                    "error": {
                        "code": "session_expired",
                        "message": "The venue session expired before the worker could start.",
                    },
                },
                sync=True,
            )
            return 1
        if import_error is not None:
            error, detail = import_error
            print(detail, file=sys.stderr)
            append(
                log,
                "result",
                {
                    "outcome": "failed",
                    "error": {
                        "code": "worker_import_failed",
                        "message": str(error),
                    },
                },
                sync=True,
            )
            return 1
        argv = message["argv"]
        if entrypoint is None:
            os.execvpe(argv[0], argv, os.environ)
        if recorder is None or message.get("project_root") is None:
            return int(entrypoint(argv[1:]) or 0)
        return _record_execution(message, recorder, log, entrypoint, argv[1:])
    except Exception as error:  # noqa: BLE001 -- foreign executor boundary
        traceback.print_exc()
        append(
            log,
            "result",
            {
                "outcome": "failed",
                "error": {
                    "code": "worker_exception",
                    "message": str(error),
                },
            },
            sync=True,
        )
        return 1


def _record_execution(message, recorder, log: Path, entrypoint, argv) -> int:
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
    from feedbax.validity.environment import EnvironmentMismatch
    from feedbax.validity.evidence import release_staged_evidence, stage_evidence

    root = Path(message["store_root"])
    try:
        environment = message.get("environment_root")
        recorder.bind_project(Path(message["project_root"]), environment_root=Path(environment) if environment else None)
        with recorder.delivery(message["job_id"], root, log.parent):
            status = int(entrypoint(argv) or 0)
            if (
                message["entrypoint"] != VALIDITY_CHECK_ENTRYPOINT
                and status == 0 and (read_worker_result(log) or {}).get("outcome") == "succeeded"
            ):
                store = ImmutableArtifactBlobProvider(root)
                with store._exclusive_lock():
                    evidence = recorder.snapshot(store)
                    ref = stage_evidence(store, evidence)
                try:
                    write_json(log.with_name(EVIDENCE_FILE), ref.model_dump(mode="json"))
                except OSError:
                    release_staged_evidence(store, message["job_id"])
                    raise
            return status
    except EnvironmentMismatch as error:
        append(
            log, "result", {"outcome": "failed", "error": {"code": "environment_lock_mismatch", "message": str(error)}},
            sync=True,
        )
        return 1
