"""Client for the machine's sole host daemon."""

import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

from feedbax.host.io import encode, private_directory, receive, socket_path
from feedbax.host.protocol import (
    MAX_RPC_BYTES,
    VERSION,
    Admission,
    AdmissionItem,
    Edge,
    EventPage,
    HostError,
    JobsSnapshot,
    Json,
    JsonObject,
    Launch,
    StopMode,
)


def default_root() -> Path:
    return Path(
        os.environ.get("FEEDBAX_HOST_ROOT") or (Path.home() / ".feedbax" / "host")
    ).expanduser()


class HostClient:
    def __init__(self, root: Path | None = None):
        self.root = (root or default_root()).resolve()
        self.address = socket_path(self.root)
        self._start_lock = threading.Lock()
        self._start_allowed = True

    def prohibit_start(self) -> None:
        with self._start_lock:
            self._start_allowed = False

    def _connect(self, timeout: float) -> socket.socket:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect(str(self.address))
        except BaseException:
            sock.close()
            raise
        return sock

    def _connect_or_start(self, timeout: float) -> socket.socket:
        try:
            return self._connect(timeout)
        except (FileNotFoundError, ConnectionRefusedError):
            pass
        with self._start_lock:
            try:
                return self._connect(timeout)
            except (FileNotFoundError, ConnectionRefusedError):
                if not self._start_allowed:
                    raise
                private_directory(self.root)
                with (self.root / "host.log").open("ab") as log:
                    process = subprocess.Popen(
                        [sys.executable, "-m", "feedbax.host", "--root", str(self.root)],
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=log,
                        start_new_session=True,
                    )
                threading.Thread(target=process.wait, daemon=True).start()
            deadline = time.monotonic() + 5
            while True:
                try:
                    return self._connect(timeout)
                except (FileNotFoundError, ConnectionRefusedError) as error:
                    if time.monotonic() >= deadline:
                        raise HostError(
                            "unavailable", f"Host did not start. See {self.root / 'host.log'}."
                        ) from error
                    time.sleep(0.02)

    def _call(
        self, op: str, *, timeout: float = 5, start_host: bool = True, **request: Json
    ) -> Json:
        message = encode({"version": VERSION, "op": op, **request})
        if len(message) > MAX_RPC_BYTES:
            raise HostError("message_too_large", "Host message exceeds size limit.")
        try:
            connect = self._connect_or_start if start_host else self._connect
            with connect(timeout) as sock:
                sock.sendall(message)
                reply = receive(sock, MAX_RPC_BYTES)
        except (OSError, EOFError) as error:
            raise HostError("disconnected", str(error)) from error
        if "error" in reply:
            error = cast(dict[str, str], reply["error"])
            raise HostError(error["code"], error["message"])
        return reply["value"]

    def ping(self) -> JsonObject:
        return cast(JsonObject, self._call("ping"))

    def warm(self, launch: Launch, *, runtime_owner: str | None = None) -> None:
        owner = {"runtime_owner": runtime_owner} if runtime_owner is not None else {}
        self._call("warm", launch=asdict(launch), **owner)

    def admit(self, token: str, items: Sequence[AdmissionItem], *, record: JsonObject) -> Admission:
        requires_authentication = any(
            edge.authenticated_completion
            for item in items
            if item.job is not None
            for edge in item.job.edges
        )
        return cast(
            Admission,
            self._call(
                "admit_with_completion" if requires_authentication else "admit",
                token=token,
                items=[asdict(item) for item in items],
                record=record,
            ),
        )

    def jobs(
        self,
        *,
        project: str | None = None,
        request: str | None = None,
        job_id: str | None = None,
        include_launch: bool = False,
    ) -> list[JsonObject]:
        return cast(
            list[JsonObject],
            self._call(
                "jobs",
                project=project,
                request=request,
                job_id=job_id,
                include_launch=include_launch,
            ),
        )

    def jobs_snapshot(
        self,
        *,
        project: str | None = None,
        request: str | None = None,
        job_id: str | None = None,
        job_ids: Sequence[str] = (),
        include_launch: bool = False,
        start_host: bool = True,
        record_keys: Sequence[str] = (),
        checkpoint_progress_root: str | None = None,
        checkpoint_objective_root: str | None = None,
    ) -> JobsSnapshot:
        return cast(
            JobsSnapshot,
            self._call(
                "jobs_snapshot",
                start_host=start_host,
                project=project,
                request=request,
                job_id=job_id,
                job_ids=list(job_ids),
                include_launch=include_launch,
                **({"record_keys": list(record_keys)} if record_keys else {}),
                **({"checkpoint_progress_root": checkpoint_progress_root} if checkpoint_progress_root is not None else {}),
                **(
                    {"checkpoint_objective_root": checkpoint_objective_root}
                    if checkpoint_objective_root is not None else {}
                ),
            ),
        )

    def stopped_checkpoint(self, job_id: str, *, project: str, manifest_root: str) -> int | None:
        return cast(int | None, self._call(
            "stopped_checkpoint", start_host=False, job_id=job_id, project=project, manifest_root=manifest_root,
        ))

    def record_get(self, key: str) -> JsonObject | None:
        return cast(JsonObject | None, self._call("record_get", key=key))

    def record_get_many(self, keys: Sequence[str]) -> dict[str, JsonObject]:
        return cast(dict[str, JsonObject], self._call("record_get_many", keys=list(keys)))

    def record_put(self, key: str, value: JsonObject) -> None:
        self._call("record_put", key=key, value=value)

    def submissions(
        self,
        *,
        id: str | None = None,
        token: str | None = None,
        project: str | None = None,
    ) -> list[Admission]:
        return cast(list[Admission], self._call("submissions", id=id, token=token, project=project))

    def events(
        self,
        after: int = 0,
        *,
        projects: Sequence[str] = (),
        limit: int = 256,
        wait: float = 0,
        start_host: bool = True,
    ) -> EventPage:
        return cast(
            EventPage,
            self._call(
                "events",
                start_host=start_host,
                after=after,
                projects=list(projects),
                limit=limit,
                wait=wait,
                timeout=max(5, wait + 5),
            ),
        )

    def stop(self, job_ids: Sequence[str], mode: StopMode) -> list[JsonObject]:
        return cast(list[JsonObject], self._call("stop", job_ids=list(job_ids), mode=mode))

    def retire_runtime(self, runtime: str, release: str, mode: str) -> JsonObject:
        """Retire the installed runtime on the existing Host without launching one."""
        try:
            return cast(JsonObject, self._call(
                "retire_runtime", start_host=False, runtime=runtime, release=release, mode=mode,
            ))
        except HostError as error:
            if isinstance(error.__cause__, (FileNotFoundError, ConnectionRefusedError)):
                return {"status": "absent"}
            raise

    def mark_satisfied(self, job_id: str, edge: Edge) -> None:
        self._call("mark_satisfied", job_id=job_id, edge=asdict(edge))

    def policy(self, *, compute: int | None = None, light: int | None = None) -> dict[str, int]:
        return cast(dict[str, int], self._call("policy", compute=compute, light=light))
