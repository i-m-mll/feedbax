"""One detached worker. It accepts one delivery before touching its payload."""

import importlib
import json
import os
import shutil
import socket
import sys
import traceback
from pathlib import Path

from .io import TOKEN_ENV, append, encode, receive, write_json
from .protocol import MAX_RPC_BYTES


def _redirect(paths: dict) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    for name, fd in (("stdout", 1), ("stderr", 2)):
        with open(paths[name], "ab", buffering=0) as stream:
            os.dup2(stream.fileno(), fd)


def _sources(entrypoint: str | None) -> dict[Path, tuple[int, int]]:
    snapshots = {}
    package = (entrypoint or "").partition(":")[0].partition(".")[0]
    for name, module in list(sys.modules.items()):
        if name.partition(".")[0] not in {"feedbax", package}:
            continue
        if source := getattr(module, "__file__", None):
            path = Path(source)
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
    entrypoint, import_error = None, None
    try:
        for module in profile["preload"]:
            importlib.import_module(module)
        if ref := profile["warm_entrypoint"]:
            module_name, function = ref.split(":", 1)
            entrypoint = getattr(importlib.import_module(module_name), function)
    except Exception as error:  # noqa: BLE001 -- foreign executor boundary
        import_error = (error, traceback.format_exc())
    snapshots = _sources(profile["warm_entrypoint"])
    try:
        with control:
            control.sendall(encode({"kind": "ready"}))
            message = receive(control, MAX_RPC_BYTES)
    except (OSError, EOFError):
        return 0
    if _sources_changed(snapshots):
        return 0
    paths = message["paths"]
    log = Path(paths["event_log"])
    os.write(report, encode({"event_log": str(log)}))
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
    if "completion_bindings" in paths:
        os.environ["FEEDBAX_HOST_COMPLETION_BINDINGS"] = paths["completion_bindings"]
    try:
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
        return int(entrypoint(argv[1:]) or 0)
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
