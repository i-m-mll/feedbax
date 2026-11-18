"""The only writer of the local job store."""

import asyncio
import fcntl
import hashlib
import json
import os
import secrets
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.host import live_evaluate
from feedbax.host.checkpoint_selection import requested_checkpoint_selection
from feedbax.host.completion import CompletionFileError
from feedbax.host.io import TOKEN_ENV, encode, private_directory, socket_path, write_json
from feedbax.host.launcher import command
from feedbax.host.platform import FileChanges, local_filesystem
from feedbax.host.process import verify
from feedbax.host.protocol import (
    LIGHT_ENTRYPOINT, LIVE_STATES, MAX_RPC_BYTES, VERSION, HostError, JsonObject, SessionCloseClaim, SessionStatus,
)
from feedbax.host.session import SessionFence
from feedbax.host.store import Store, StorePublicationPending

LOG_LINE_LIMIT = 8192
EVENT_LINE_LIMIT = 2 * 1024 * 1024
POLL_SECONDS = 0.20
TERM_GRACE = 2
RESERVE_BYTES = 2 * 1024 * 1024
_SESSION_READS = {
    "ping", "events", "jobs_snapshot", "stopped_checkpoint", "checkpoint_producer", "jobs", "submissions",
    "record_get", "record_get_many",
}


@dataclass(slots=True)
class Worker:
    process: subprocess.Popen
    token: str
    control: socket.socket
    profile_key: str
    profile_dir: Path
    runtime_owner: str | None
    reusable: bool
    ready: bool = False
    retiring_since: float | None = None
    job_id: str | None = None
    data: bytes = b""


def failure(code: str, message: str, *, outcome: str = "failed") -> dict:
    return {"outcome": outcome, "error": {"code": code, "message": message}}


class Host:
    def __init__(self, root: Path, *, required_expiry_ms: int | None = None):
        self.root = private_directory(root)
        if not local_filesystem(root):
            raise OSError(f"Host state root must be on a local filesystem: {root}")
        self.lock = (root / "host.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.session = SessionFence.load(root)
        if required_expiry_ms is not None:
            if self.session is None or self.session.state.expires_at != required_expiry_ms / 1000:
                self.lock.close()
                raise HostError("session_unavailable", "The Host fence does not match the venue cutoff.")
            if self.session.expired:
                self.lock.close()
                raise HostError("session_expired", "A watchdog cannot start an expired venue Host.")
        self.address = socket_path(root)
        self.deliveries = private_directory(root / "deliveries")
        self.worker_root = private_directory(root / "workers")
        self.cache = private_directory(root / "jax")
        self.store = Store(root / "host.sqlite")
        self.stream_id: str | None = None
        if self.session is not None:
            self.stream_id = self._session_stream_id(self.session.state.session_id)
        elif self.store.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='session_stream'"
        ).fetchone() is not None:
            self.store.db.close()
            self.lock.close()
            raise HostError("session_unavailable", "The venue's Host database has no controller fence.")
        self.reserve = root / "emergency.reserve"
        if not self.reserve.exists():
            with self.reserve.open("xb") as stream:
                stream.write(bytes(RESERVE_BYTES))
                stream.flush()
                os.fsync(stream.fileno())
        self.loop = asyncio.get_running_loop()
        self.changed = asyncio.Condition()
        self.wake = asyncio.Event()
        self.file_changes = FileChanges(self.loop, self.wake.set)
        self.workers: dict[int, Worker] = {}
        self.requests: set[asyncio.Task] = set()
        self.connections: set[asyncio.StreamWriter] = set()
        self.closing = False
        self.profiles: dict[str, dict] = {}
        self.release_owner = os.environ.get("FEEDBAX_DESKTOP_RELEASE_ROOT")
        self.interpreter = os.path.realpath(sys.executable)
        self.retiring: set[str] = set()
        self.retiring_groups: dict[str, dict[int, str]] = {}
        self.exiting = False
        self.shutdown_acknowledged = False
        self.force_started: dict[str, float] = {}
        self.identity_errors: set[str] = set()
        self.publication_errors: dict[str, str] = {}
        self.power: subprocess.Popen | None = None
        self.last_clock_delta = time.time() - time.monotonic()

    def _session_stream_id(self, session_id: str) -> str:
        """Bind cursor identity to the database that stores the event stream."""
        with self.store.db:
            self.store.db.execute(
                """CREATE TABLE IF NOT EXISTS session_stream (
                    singleton INTEGER PRIMARY KEY CHECK (singleton=1), session_id TEXT NOT NULL, id TEXT NOT NULL
                )"""
            )
            self.store.db.execute(
                "INSERT OR IGNORE INTO session_stream VALUES (1, ?, ?)", (session_id, secrets.token_hex(16)),
            )
            row = self.store.db.execute("SELECT session_id, id FROM session_stream").fetchone()
            if row["session_id"] != session_id:
                raise ValueError("Venue Host event stream belongs to a different session.")
            return row["id"]

    def _paths(self, row: sqlite3.Row, *, delivery: int | None = None) -> dict[str, Path]:
        attempt = row["delivery"] if delivery is None else delivery
        directory = self.deliveries / row["id"] / str(attempt)
        return {
            "directory": directory,
            "event_log": directory / "events.jsonl",
            "input": directory / "input.json",
            "stdout": directory / "stdout",
            "stderr": directory / "stderr",
            "scratch": directory / "scratch",
            "stop_file": directory / "stop",
        }

    def _profile(self, launch: dict, slot: str, runtime_owner: str | None = None) -> tuple[str, dict]:
        interpreter = launch["argv"][0] if launch["warm_entrypoint"] else sys.executable
        profile = {
            "interpreter": interpreter,
            "cwd": launch["cwd"],
            "env": launch["env"],
            "preload": launch["preload"],
            "warm_entrypoint": launch["warm_entrypoint"],
            "slot": slot,
            "runtime_owner": runtime_owner,
        }
        if slot == "light" and launch["warm_entrypoint"] == LIGHT_ENTRYPOINT:
            payload = launch["payload"]
            profile["light_affinity"] = [payload["project_id"], payload["manifest_root"]]
        return hashlib.sha256(encode(profile)).hexdigest(), profile

    def _spawn(self, key: str, profile: dict) -> Worker:
        token = secrets.token_hex(16)
        profile_dir = private_directory(self.worker_root / token)
        profile_path = profile_dir / "profile.json"
        write_json(profile_path, profile)
        env = dict(os.environ, **profile["env"])
        env.update(
            {
                TOKEN_ENV: token,
                "PYTHONUNBUFFERED": "1",
                "JAX_COMPILATION_CACHE_DIR": str(self.cache),
                "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
                "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "0",
                "JAX_COMPILATION_CACHE_MAX_SIZE": str(1024**3),
                "JAX_RAISE_PERSISTENT_CACHE_ERRORS": "true",
            }
        )
        ours, theirs = socket.socketpair()
        try:
            with (profile_dir / "startup.log").open("ab") as startup:
                process = subprocess.Popen(
                    command(
                        profile["interpreter"], "launcher", str(theirs.fileno()), str(profile_path)
                    ),
                    cwd=profile["cwd"],
                    env=env,
                    start_new_session=True,
                    stdin=subprocess.DEVNULL,
                    stdout=startup,
                    stderr=startup,
                    pass_fds=(theirs.fileno(),),
                )
        except BaseException:
            ours.close()
            shutil.rmtree(profile_dir)
            raise
        finally:
            theirs.close()
        worker = Worker(
            process, token, ours, key, profile_dir, profile["runtime_owner"],
            reusable="light_affinity" in profile,
        )
        self.workers[process.pid] = worker
        self.loop.add_reader(ours.fileno(), self._worker_ready, worker)
        return worker

    def _worker_ready(self, worker: Worker) -> None:
        try:
            data = worker.control.recv(4096)
        except OSError:
            data = b""
        if not data:
            self._close_control(worker)
        else:
            worker.data += data
            if worker.data.endswith(b"\n"):
                worker.ready = json.loads(worker.data)["kind"] == "ready"
                worker.data = b""
        self.wake.set()

    def _close_control(self, worker: Worker) -> None:
        if worker.control.fileno() >= 0:
            self.loop.remove_reader(worker.control.fileno())
            worker.control.close()

    def warm(self, launch: dict, runtime_owner: str | None = None) -> None:
        if runtime_owner is not None and (
            not isinstance(runtime_owner, str) or runtime_owner != os.path.realpath(launch["argv"][0])
        ):
            raise HostError("invalid_request", "Warm runtime owner differs from its interpreter.")
        if runtime_owner in self.retiring or self.exiting:
            raise HostError("shutting_down", "The desktop runtime is stopping.")
        if launch["warm_entrypoint"]:
            key, profile = self._profile(launch, "compute", runtime_owner)
            self.profiles.pop(key, None)
            self.profiles[key] = profile

    def _maintain_pool(self) -> None:
        pending = {
            worker.profile_key
            for worker in self.workers.values()
            if (worker.reusable or worker.job_id is None) and worker.control.fileno() >= 0
        }
        # Keep one imported worker per recently used environment. Drop unused profiles
        # before they can grow resident memory with the number of opened projects.
        while len(self.profiles) > 4:
            expired = next(iter(self.profiles))
            del self.profiles[expired]
            for worker in self.workers.values():
                if worker.profile_key == expired and worker.job_id is None:
                    self._close_control(worker)
        for key, profile in self.profiles.items():
            if key not in pending:
                try:
                    self._spawn(key, profile)
                except OSError as error:
                    print(f"Could not prewarm project worker: {error}", file=sys.stderr, flush=True)
                    self.profiles.pop(key)
                    break

    def _retire_runtime(self, runtime: str, release: str, mode: str) -> dict:
        if (
            not Path(release).is_absolute() or not Path(runtime).is_relative_to(release)
            or mode not in ("defer", "checkpoint", "force", "interrupt")
        ):
            raise HostError("invalid_request", "Invalid installed runtime retirement.")
        owned = self.release_owner == release and self.interpreter == runtime
        if self.interpreter == runtime and self.release_owner is not None and not owned:
            raise HostError("runtime_owner", "The installed interpreter's Host has no matching release owner.")
        rows = self.store.db.execute(
            """SELECT id, state, data, worker_pid, worker_token FROM jobs
               WHERE state IN (?, ?, ?, ?) OR worker_pid IS NOT NULL""",
            LIVE_STATES,
        )
        matching, jobs = set(), []
        foreign = False
        for row in rows:
            if self._runtime(json.loads(row["data"])) == runtime:
                matching.add(row["id"])
                if row["state"] in LIVE_STATES:
                    jobs.append(row["id"])
                    if row["worker_pid"] is not None:
                        self.retiring_groups.setdefault(runtime, {})[row["worker_pid"]] = row["worker_token"]
            elif row["state"] in LIVE_STATES:
                foreign = owned
        if (mode == "defer" and jobs) or (foreign and mode != "interrupt"):
            return {"status": "deferred", "jobs": jobs, "foreign": foreign}
        self.retiring.add(runtime)
        if jobs:
            self.store.stop(jobs, mode, service=True)
        whole_host = owned and not foreign
        for key, profile in tuple(self.profiles.items()):
            if whole_host or profile.get("runtime_owner") == runtime:
                del self.profiles[key]
        remaining = [
            worker for worker in self.workers.values()
            if whole_host or worker.job_id in matching
            or (worker.job_id is None and worker.runtime_owner == runtime)
        ]
        for worker in remaining:
            if worker.job_id is None:
                self._close_control(worker)
                now = time.monotonic()
                if worker.retiring_since is None or now - worker.retiring_since >= TERM_GRACE:
                    sent, _ = self._signal_group(
                        worker.process.pid, worker.token, escalate=worker.retiring_since is not None,
                    )
                    if sent and worker.retiring_since is None:
                        worker.retiring_since = now
            elif worker.job_id not in jobs:
                self._signal_group(worker.process.pid, worker.token, escalate=True)
        groups = self.retiring_groups.get(runtime, {})
        for pid, token in tuple(groups.items()):
            verdict, _ = verify(pid, token)
            if verdict == "gone":
                del groups[pid]
            elif verdict == "verified" and not jobs:
                self._signal_group(pid, token, escalate=True)
        if jobs or remaining or groups:
            return {"status": "settling", "jobs": jobs}
        self.retiring_groups.pop(runtime, None)
        if whole_host:
            self.exiting = True
        else:
            self.retiring.discard(runtime)
        return {"status": "exiting" if whole_host else "released", "jobs": []}

    @staticmethod
    def _runtime(data: dict) -> str | None:
        runtime = data.get("runtime")
        return os.path.realpath(runtime) if isinstance(runtime, str) else None

    def _light_busy(self, key: str) -> bool:
        if any(worker.profile_key == key and worker.job_id is not None for worker in self.workers.values()):
            return True
        # A worker recovered after daemon restart has no control socket here.
        for active in self.store.active():
            if active["slot"] != "light" or active["worker_pid"] in self.workers:
                continue
            launch = json.loads(active["launch"])
            if launch["warm_entrypoint"] == LIGHT_ENTRYPOINT and self._profile(
                launch, "light", self._runtime(json.loads(active["data"])),
            )[0] == key:
                return True
        return False

    def _assign(self, row: sqlite3.Row) -> None:
        if (
            self.exiting or (self.session is not None and not self.session.accepts_work)
            or self._runtime(json.loads(row["data"])) in self.retiring
        ):
            return
        bindings = self.store.completion_bindings(row["id"])
        launch = json.loads(row["launch"])
        key, profile = self._profile(launch, row["slot"], self._runtime(json.loads(row["data"])))
        reusable = "light_affinity" in profile
        if reusable and self._light_busy(key):
            return
        origins = None
        try:
            if launch["warm_entrypoint"] == LIGHT_ENTRYPOINT:
                origins = self.store.light_origins(row)
            self.store.verify_completion_bindings(row, launch, bindings)
        except CompletionFileError as error:
            with self.store.db:
                self.store.finished(
                    row, failure("dependency_corrupt", str(error), outcome="blocked")
                )
            return
        if self.session is not None and not self.session.accepts_work:
            return
        if launch["warm_entrypoint"]:
            self.profiles.pop(key, None)
            self.profiles[key] = profile
        worker = next(
            (
                worker
                for worker in self.workers.values()
                if worker.profile_key == key
                and worker.job_id is None
                and worker.ready
                and worker.control.fileno() >= 0
            ),
            None,
        )
        if worker is None:
            worker = next(
                (
                    candidate
                    for candidate in self.workers.values()
                    if candidate.profile_key == key
                    and candidate.job_id is None
                    and candidate.control.fileno() >= 0
                ),
                None,
            )
        if worker is None:
            try:
                worker = self._spawn(key, profile)
            except OSError as error:
                with self.store.db:
                    self.store.finished(row, failure("worker_spawn", str(error)))
                return
        delivery = row["delivery"] + 1
        paths = self._paths(row, delivery=delivery)
        # A crash before the assignment transaction can leave only unclaimed files.
        # No worker has received this delivery, so those bytes can be overwritten.
        paths["scratch"].mkdir(parents=True, exist_ok=True)
        write_json(paths["input"], launch["payload"])
        completion_bindings_sha256 = None
        if bindings or origins is not None:
            paths["completion_bindings"] = paths["directory"] / "completion-bindings.json"
            delivery_bindings: JsonObject = {"bindings": bindings}
            if origins is not None:
                delivery_bindings["origins"] = origins
            completion_bindings_sha256 = hashlib.sha256(encode(delivery_bindings)).hexdigest()
            write_json(paths["completion_bindings"], delivery_bindings)
        live_evaluate_sha256 = None
        owner = live_evaluate.choose_evaluation(self.store.db, row) if row["request"].startswith("train:") else None
        if owner is not None:
            evaluation_launch = json.loads(owner["launch"])
            selection = requested_checkpoint_selection(owner["request"], evaluation_launch)
            assert selection is not None
            assignment = {
                "evaluation_job_id": owner["id"],
                "producer_job_id": row["id"],
                "producer_delivery": delivery,
                "launch": evaluation_launch["payload"],
                "updates": list(selection.coordinates),
                "stop_file": str(self._paths(owner, delivery=0)["stop_file"]),
                "placement": json.loads(owner["data"])["placement_detail"],
            }
            paths["live_evaluate"] = paths["directory"] / "live-evaluate.json"
            live_evaluate_sha256 = hashlib.sha256(encode(assignment)).hexdigest()
            write_json(paths["live_evaluate"], assignment)
        if self.session is not None and not self.session.accepts_work:
            return
        with self.store.db:
            if owner is not None:
                self.store.db.execute(
                    "INSERT INTO live_evaluations VALUES (?, ?, ?)", (owner["id"], row["id"], delivery),
                )
                self.store.append(owner, live_evaluate.DELEGATION_EVENT, {
                    "producer_job_id": row["id"], "producer_delivery": delivery,
                    "assignment_sha256": live_evaluate_sha256,
                })
            self.store.db.execute(
                """UPDATE jobs SET state='assigned', delivery=?, worker_pid=?, worker_token=?,
                   event_offset=0, stdout_offset=0, stderr_offset=0 WHERE id=?""",
                (delivery, worker.process.pid, worker.token, row["id"]),
            )
            if not worker.ready:
                data = {"phase": "importing" if launch["warm_entrypoint"] else "starting"}
                self.store.ingest(row, "phase", data)
        worker.job_id = row["id"]
        worker.ready = False
        self.wake.set()
        message = encode(
            {
                "job_id": row["id"],
                "paths": {name: str(path) for name, path in paths.items()},
                "argv": launch["argv"],
                "store_root": json.loads(row["data"]).get("manifest_root"),
                "project_root": launch.get("project_root"),
                "environment_root": launch["cwd"] if self.session is not None else None,
                "entrypoint": launch["warm_entrypoint"],
                "completion_bindings_sha256": completion_bindings_sha256,
                "live_evaluate_sha256": live_evaluate_sha256,
                "startup_log": str(worker.profile_dir / "startup.log"),
                "expires_at": self.session.state.expires_at if self.session is not None else None,
            }
        )
        try:
            worker.control.settimeout(0.05)
            worker.control.sendall(message)
        except OSError:
            self._close_control(worker)

    def _read_job_events(self, row: sqlite3.Row, paths: dict[str, Path]) -> bool:
        """Read complete worker events. Return true while final output is draining."""
        try:
            stream = paths["event_log"].open("rb")
        except FileNotFoundError:
            return False
        with stream:
            stream.seek(row["event_offset"])
            for _ in range(256):
                start = stream.tell()
                raw = stream.readline(EVENT_LINE_LIMIT + 1)
                if not raw:
                    break
                terminal = False
                if len(raw) > EVENT_LINE_LIMIT:
                    while raw and not raw.endswith(b"\n"):
                        raw = stream.readline(EVENT_LINE_LIMIT + 1)
                    if not raw:
                        break
                    self._event_error(row, "Worker event exceeds size limit.")
                elif not raw.endswith(b"\n"):
                    break
                else:
                    try:
                        event = json.loads(raw)
                        kind, data = event["kind"], event["data"]
                        if not isinstance(kind, str) or not isinstance(data, dict):
                            raise TypeError("Worker event must have a string kind and object data.")
                        if kind == "revoked":
                            break
                        if start == 0 and (
                            kind != "accepted" or data.get("token") != row["worker_token"]
                        ):
                            raise ValueError(
                                "Worker event log lacks the recorded acceptance token."
                            )
                        if kind == "publication" and (
                            not isinstance(data.get("key"), str) or not data["key"]
                        ):
                            raise ValueError("Publication event must name its key.")
                        if kind == "terminal":
                            if data.get("outcome") not in ("succeeded", "stopped", "failed", "interrupted"):
                                raise ValueError("Invalid terminal outcome.")
                            row = self.store.get(row["id"])
                            if not self._tail_output(row, paths, finished=True):
                                self.wake.set()
                                return True
                        self.store.ingest(row, kind, data)
                        terminal = kind == "terminal"
                        if terminal:
                            self.publication_errors.pop(row["id"], None)
                    except StorePublicationPending as error:
                        if self.publication_errors.get(row["id"]) != str(error):
                            self.publication_errors[row["id"]] = str(error)
                            self.store.append(row, "log", {"stream": "host", "line": f"Store publication pending: {error}"})
                        return True
                    except (KeyError, TypeError, ValueError) as error:
                        self._event_error(row, str(error))
                self.store.db.execute(
                    "UPDATE jobs SET event_offset=? WHERE id=?", (stream.tell(), row["id"])
                )
                if terminal:
                    break
            else:
                self.wake.set()
                return True
        return False

    def _event_error(self, row: sqlite3.Row, message: str) -> None:
        self.store.db.execute("UPDATE jobs SET event_error=? WHERE id=?", (message, row["id"]))
        self.store.append(row, "log", {"stream": "host", "line": message})

    def _tail_output(
        self, row: sqlite3.Row, paths: dict[str, Path], *, finished: bool = False
    ) -> bool:
        drained = True
        for name in ("stdout", "stderr"):
            try:
                stream = paths[name].open("rb")
            except FileNotFoundError:
                continue
            column = name + "_offset"
            with stream:
                stream.seek(row[column])
                for _ in range(256):
                    raw = stream.readline(LOG_LINE_LIMIT)
                    if not raw or (
                        len(raw) < LOG_LINE_LIMIT and not raw.endswith(b"\n") and not finished
                    ):
                        break
                    self.store.append(
                        row,
                        "log",
                        {
                            "stream": name,
                            "line": raw.decode(errors="replace").rstrip("\n"),
                            "truncated": len(raw) == LOG_LINE_LIMIT and not raw.endswith(b"\n"),
                        },
                    )
                    self.store.db.execute(
                        f"UPDATE jobs SET {column}=? WHERE id=?", (stream.tell(), row["id"])
                    )
                else:
                    drained = False
        return drained

    def _deliver_stop(self, row: sqlite3.Row, paths: dict[str, Path]) -> None:
        if row["stop_mode"] == "checkpoint" and not paths["stop_file"].exists():
            temp = paths["stop_file"].with_suffix(".tmp")
            with temp.open("w") as stream:
                stream.write(row["id"] + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, paths["stop_file"])
        if row["stop_mode"] not in ("force", "interrupt"):
            return
        now = time.monotonic()
        job_id = row["id"]
        started = self.force_started.get(job_id)
        if started is not None and now - started < TERM_GRACE:
            return
        sent, error = self._signal_group(row["worker_pid"], row["worker_token"], escalate=started is not None)
        if error is not None and job_id not in self.identity_errors:
            self.identity_errors.add(job_id)
            self.store.append(row, "log", {"stream": "host", "line": error})
        if sent:
            self.force_started[job_id] = now if started is None else started

    @staticmethod
    def _signal_group(pid: int, token: str, *, escalate: bool) -> tuple[bool, str | None]:
        verdict, error = verify(pid, token)
        if verdict != "verified":
            return False, error
        try:
            os.killpg(pid, signal.SIGKILL if escalate else signal.SIGTERM)
        except ProcessLookupError:
            return False, None
        return True, None

    def _lost_worker(self, row: sqlite3.Row, paths: dict[str, Path], *, recovering: bool) -> None:
        log = paths["event_log"]
        revoked = write_json(log, {"kind": "revoked", "data": {}}, exclusive=True)
        if not revoked:
            with log.open("rb") as stream:
                revoked = json.loads(stream.readline())["kind"] == "revoked"
        if revoked:
            if row["stop_mode"] == "interrupt":
                self.store.finished(row, failure(
                    "service_stopped", "The background service stopped.", outcome="interrupted",
                ))
            elif row["stop_mode"] is not None:
                self.store.finished(row, {"outcome": "stopped"})
            elif recovering or row["delivery"] < 2:
                self.store.db.execute(
                    "UPDATE jobs SET state='queued', worker_pid=NULL, worker_token=NULL, phase=NULL WHERE id=?",
                    (row["id"],),
                )
            else:
                self.store.finished(
                    row,
                    failure("worker_launch_failed", "Worker exited before accepting its delivery."),
                )
            return
        # A launcher can append its last event between our first read and the process check.
        if self._read_job_events(row, paths):
            return
        current = self.store.get(row["id"])
        if current["outcome"] is None:
            outcome = (
                {"outcome": "stopped"}
                if row["stop_mode"] == "force"
                else failure(
                    "service_stopped", "The background service stopped.", outcome="interrupted",
                )
                if row["stop_mode"] == "interrupt"
                else failure(
                    "worker_lost",
                    "Worker group disappeared without a terminal event.",
                    outcome="interrupted",
                )
            )
            if self._tail_output(current, paths, finished=True):
                self.store.finished(current, outcome)
            else:
                self.wake.set()

    def _observe(self, row: sqlite3.Row, paths: dict[str, Path], *, recovering: bool) -> None:
        with self.store.db:
            if self._read_job_events(row, paths):
                return
            current = self.store.get(row["id"])
            if current["outcome"] is not None:
                return
            self._tail_output(current, paths)
            self._deliver_stop(current, paths)
            worker = self.workers.get(row["worker_pid"])
            if worker is not None and worker.process.poll() is None and not recovering:
                return
            verdict, error = verify(row["worker_pid"], row["worker_token"])
            if recovering and not paths["event_log"].exists():
                self._lost_worker(current, paths, recovering=True)
            elif verdict == "gone":
                self._lost_worker(current, paths, recovering=recovering)
            elif verdict == "unverifiable" and row["id"] not in self.identity_errors:
                self.identity_errors.add(row["id"])
                self.store.append(row, "log", {"stream": "host", "line": error})

    def _power_assertion(self, active: bool) -> None:
        if sys.platform != "darwin":
            return
        if active and (self.power is None or self.power.poll() is not None):
            self.power = subprocess.Popen(
                ["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif not active and self.power is not None:
            if self.power.poll() is None:
                self.power.terminate()
                try:
                    self.power.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self.power.kill()
                    self.power.wait(timeout=1)
            self.power = None

    def _tick(self, *, recovering: bool = False) -> None:
        for stopped in self.store.db.execute(
            """SELECT DISTINCT j.* FROM jobs j JOIN live_evaluations d ON j.id=d.evaluation
               JOIN jobs p ON p.id=d.producer WHERE j.stop_mode IS NOT NULL
               AND p.state IN ('assigned','running','stopping')"""
        ):
            stop = self._paths(stopped, delivery=0)["stop_file"]
            if not stop.exists():
                stop.parent.mkdir(parents=True, exist_ok=True)
                write_json(stop, {"job_id": stopped["id"]})
        for pid, worker in tuple(self.workers.items()):
            if worker.process.poll() is not None:
                self._close_control(worker)
                if verify(pid, worker.token)[0] != "gone":
                    continue
                self.workers.pop(pid)
                if worker.job_id is None:
                    self.profiles.pop(worker.profile_key, None)
                shutil.rmtree(worker.profile_dir)
        active = self.store.active()
        for row in active:
            self._observe(row, self._paths(row), recovering=recovering)
        live = {row["id"] for row in self.store.active()}
        for worker in self.workers.values():
            if worker.reusable and worker.job_id is not None and worker.job_id not in live:
                worker.job_id = None
                if worker.profile_key not in self.profiles:
                    self._close_control(worker)
        self.store.resolve_completions()
        self.store.settle_blocked()
        if self.session is None or self.session.accepts_work:
            self.store.deferred.settle()
        for row in self.store.ready():
            self._assign(row)
        running = self.store.active()
        if recovering:
            for row in running:
                launch = json.loads(row["launch"])
                if launch["warm_entrypoint"]:
                    key, profile = self._profile(launch, row["slot"], self._runtime(json.loads(row["data"])))
                    self.profiles[key] = profile
        if self.session is None or self.session.accepts_work:
            self._maintain_pool()
        self.file_changes.watch(
            {
                self._paths(row)[name]
                for row in running
                for name in ("directory", "event_log", "stdout", "stderr")
            }
        )
        live_ids = {row["id"] for row in running}
        self.force_started = {
            key: value for key, value in self.force_started.items() if key in live_ids
        }
        self.identity_errors.intersection_update(live_ids)
        self._power_assertion(bool(running))
        delta = time.time() - time.monotonic()
        if delta - self.last_clock_delta > 2:
            with self.store.db:
                for row in running:
                    self.store.append(
                        row, "host_resumed", {"seconds": delta - self.last_clock_delta}
                    )
        self.last_clock_delta = delta

    async def _schedule(self) -> None:
        recovering = True
        while not self.shutdown_acknowledged:
            self.wake.clear()
            try:
                self._tick(recovering=recovering)
            except sqlite3.OperationalError as error:
                if error.sqlite_errorcode != sqlite3.SQLITE_FULL or not self.reserve.exists():
                    raise
                self.reserve.unlink()
                print(
                    "Released host emergency disk reserve after SQLite reported a full disk.",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            recovering = False
            await self._announce()
            try:
                await asyncio.wait_for(self.wake.wait(), timeout=POLL_SECONDS)
            except TimeoutError:
                pass

    async def _announce(self) -> None:
        head = self.store.db.execute("SELECT coalesce(max(cursor), 0) FROM events").fetchone()[0]
        if head != self.store.event_head:
            self.store.event_head = head
            async with self.changed:
                self.changed.notify_all()

    async def _events(self, after: int, projects: list[str], limit: int, wait: float) -> dict:
        if not 0 <= after or not 1 <= limit <= 1024 or not 0 <= wait <= 60 or len(projects) > 512:
            raise HostError(
                "invalid_request", "Invalid event cursor, limit, wait, or project filter."
            )
        deadline = self.loop.time() + wait
        async with self.changed:
            while True:
                page = self.store.events(after, projects, limit)
                if page["events"] or self.loop.time() >= deadline:
                    return page
                try:
                    await asyncio.wait_for(self.changed.wait(), max(0, deadline - self.loop.time()))
                except TimeoutError:
                    return self.store.events(after, projects, limit)

    def _collection_receipts(self, job_id: str | None = None) -> dict[str, str]:
        query = """SELECT id, state, outcome FROM jobs
                   WHERE state IN ('succeeded', 'stopped') AND outcome IS NOT NULL"""
        rows = (
            self.store.db.execute(query + " ORDER BY seq") if job_id is None
            else self.store.db.execute(query + " AND id=?", (job_id,))
        )
        receipts = {}
        for row in rows:
            outcome = json.loads(row["outcome"])
            key = "store_success" if row["state"] == "succeeded" else "store_job_entry"
            ref = outcome.get(key)
            if ref is not None:
                receipts[row["id"]] = ContentObjectRef.model_validate(ref).sha256
        return receipts

    def _session_status(self, fence: SessionFence) -> SessionStatus:
        state = fence.state
        assert self.stream_id is not None
        live = [row["id"] for row in self.store.db.execute(
            "SELECT id FROM jobs WHERE state IN (?, ?, ?, ?) ORDER BY seq", LIVE_STATES,
        )]
        idle = self.store.db.execute(
            "SELECT time FROM events WHERE kind IN ('terminal', 'blocked', 'reused') ORDER BY cursor DESC LIMIT 1",
        ).fetchone() if not live else None
        return {
            "session_id": state.session_id, "stream_id": self.stream_id,
            "operator": state.operator, "epoch": state.epoch,
            "expires_at": state.expires_at, "closing": state.closing, "expired": fence.expired,
            "live_jobs": live,
            "idle_since_ms": round(idle["time"] * 1000) if idle is not None else None,
            "pending": {
                job: digest for job, digest in self._collection_receipts().items()
                if state.collections.get(job) != digest
            },
            "collections": state.collections.copy(),
        }

    def _acknowledge_collection(self, fence: SessionFence, job_id: str, sha256: str) -> None:
        if not isinstance(job_id, str):
            raise HostError("invalid_request", "Collection acknowledgement requires a job ID.")
        digest = self._collection_receipts(job_id).get(job_id)
        if digest is None or digest != sha256:
            raise HostError("collection_conflict", "Collection evidence differs from the terminal Host receipt.")
        fence.acknowledge_collection(job_id, digest)

    def _claim_session_close(self, fence: SessionFence) -> SessionCloseClaim:
        if fence.expired:
            raise HostError("session_expired", "The venue session has expired.")
        if not fence.state.closing:
            status = self._session_status(fence)
            if status["live_jobs"] or status["pending"]:
                raise HostError("session_busy", "The venue still has live jobs or unacknowledged results.")
        return fence.claim_close()

    async def _call(self, message: dict):
        if message.pop("version") != VERSION:
            raise HostError("protocol_version", f"Host expects protocol version {VERSION}.")
        op = message.pop("op")
        access = message.pop("session", None)
        fence = self.session
        if fence is None:
            if access is not None or op in {
                "session_epoch", "adopt_session", "session_status", "acknowledge_collection", "claim_session_close",
                "recover_session_close",
            }:
                raise HostError("session_uninitialized", "This Host does not own a venue session.")
        else:
            if op == "session_epoch":
                return fence.epoch(**message)
            if op == "adopt_session":
                return fence.adopt(**message)
            if op == "recover_session_close":
                return fence.recover_close(**message)
            fence.authenticate(access)
            if op == "session_status":
                return self._session_status(fence)
            if op == "claim_session_close":
                return self._claim_session_close(fence)
            policy_read = op == "policy" and all(message.get(slot) is None for slot in ("compute", "light"))
            if op not in _SESSION_READS and not policy_read:
                fence.require_open()
            if op == "acknowledge_collection":
                return self._acknowledge_collection(fence, **message)
        if op in ("admit", "admit_with_completion") and (
            self.exiting or any(
                item.get("job") is not None and self._runtime(item["job"]["data"]) in self.retiring
                for item in message["items"]
            )
        ):
            raise HostError("shutting_down", "The desktop runtime is stopping.")
        match op:
            case "retire_runtime":
                result = self._retire_runtime(**message)
                self.wake.set()
                return result
            case "ping":
                identity = {} if self.stream_id is None else {"stream_id": self.stream_id}
                return {"version": VERSION, "pid": os.getpid(), **identity}
            case "events":
                return await self._events(**message)
            case "jobs_snapshot":
                return self.store.jobs_snapshot(**message)
            case "checkpoint_producer":
                return self.store.checkpoint_producer(**message)
            case "stopped_checkpoint":
                return self.store.stopped_checkpoint(**message)
            case "admit" | "admit_with_completion":
                result = self.store.admit(
                    **message, allow_completion=op == "admit_with_completion",
                    session_id=fence.state.session_id if fence is not None else None,
                )
                self.wake.set()
                await self._announce()
                return result
            case "warm":
                self.warm(**message)
                self.wake.set()
                return None
            case (
                "jobs"
                | "submissions"
                | "policy"
                | "stop"
                | "mark_satisfied"
                | "record_get"
                | "record_get_many"
                | "record_put"
                | "mirror_collected_train"
                | "mirror_collected_evaluate"
            ):
                result = getattr(self.store, op)(**message)
                self.wake.set()
                await self._announce()
                return result
            case _:
                raise HostError("invalid_request", f"Unknown host operation: {op}")

    def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        # Registered at the accept itself: a shutdown that begins before the handler first runs still finds it.
        self.connections.add(writer)
        return self.serve(reader, writer)

    async def serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        exiting = False
        request = asyncio.current_task()
        assert request is not None
        self.requests.add(request)
        try:
            if self.closing:
                return
            try:
                raw = await asyncio.wait_for(reader.readline(), 5)
                if not raw.endswith(b"\n"):
                    return
                message = json.loads(raw)
                value = await self._call(message)
                exiting = isinstance(value, dict) and value.get("status") == "exiting" and self.exiting
                reply = {"value": value}
            except HostError as error:
                reply = {"error": {"code": error.code, "message": str(error)}}
            except (KeyError, TypeError, ValueError) as error:
                reply = {"error": {"code": "invalid_request", "message": str(error)}}
            except sqlite3.OperationalError as error:
                if error.sqlite_errorcode != sqlite3.SQLITE_FULL:
                    raise
                reply = {
                    "error": {"code": "disk_full", "message": "The host's state disk is full."}
                }
            writer.write(encode(reply))
            await asyncio.wait_for(writer.drain(), 5)
        except (OSError, TimeoutError):
            pass
        finally:
            self.requests.discard(request)
            self.connections.discard(writer)
            if self.closing:
                writer.transport.abort()
            else:
                writer.close()
            if exiting:
                self.shutdown_acknowledged = True
                self.wake.set()

    async def run(self) -> None:
        self.address.unlink(missing_ok=True)
        server = await asyncio.start_unix_server(self._accept, str(self.address), limit=MAX_RPC_BYTES)
        try:
            async with server:
                try:
                    await self._schedule()
                finally:
                    # The server's exit waits for every open connection to close. End them instead of
                    # serving them out: cancel the requests still waiting, such as event long polls, and
                    # abort every transport, since closing one waits until a client reads its buffered reply.
                    self.closing = True
                    for request in tuple(self.requests):
                        request.cancel()
                    for connection in tuple(self.connections):
                        connection.transport.abort()
        finally:
            self.file_changes.close()
            for worker in self.workers.values():
                self._close_control(worker)
                if worker.job_id is None:
                    # An idle or still-importing pool worker holds no job; it would otherwise
                    # outlive this Host until its import finished.
                    self._signal_group(worker.process.pid, worker.token, escalate=True)
            self._power_assertion(False)
            self.address.unlink(missing_ok=True)
            self.store.db.close()
            self.lock.close()
