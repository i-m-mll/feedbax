"""Transactions for the host's jobs, submissions and durable event cursor."""

import json
import re
import sqlite3
import time
from pathlib import Path
from uuid import uuid4

from feedbax.host.protocol import LIVE_STATES, TERMINAL_STATES, HostError

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    seq INTEGER PRIMARY KEY,
    id TEXT NOT NULL UNIQUE,
    project TEXT NOT NULL,
    request TEXT NOT NULL,
    launch TEXT NOT NULL,
    slot TEXT NOT NULL,
    continues TEXT REFERENCES jobs(id),
    data TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'queued',
    stop_mode TEXT,
    phase TEXT,
    progress TEXT,
    checkpoint TEXT,
    outcome TEXT,
    event_error TEXT,
    delivery INTEGER NOT NULL DEFAULT 0,
    worker_pid INTEGER,
    worker_token TEXT,
    event_offset INTEGER NOT NULL DEFAULT 0,
    stdout_offset INTEGER NOT NULL DEFAULT 0,
    stderr_offset INTEGER NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX IF NOT EXISTS one_live_job ON jobs(project, request)
    WHERE state IN ('queued','assigned','running','stopping');
CREATE INDEX IF NOT EXISTS jobs_by_state ON jobs(state, slot, seq);
CREATE TABLE IF NOT EXISTS edges (
    job TEXT NOT NULL REFERENCES jobs(id),
    parent TEXT NOT NULL,
    publication TEXT NOT NULL,
    satisfied INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (job, parent, publication)
);
CREATE INDEX IF NOT EXISTS edges_by_parent ON edges(parent, publication);
CREATE TABLE IF NOT EXISTS publications (
    job TEXT NOT NULL REFERENCES jobs(id),
    key TEXT NOT NULL,
    PRIMARY KEY (job, key)
);
CREATE TABLE IF NOT EXISTS events (
    cursor INTEGER PRIMARY KEY AUTOINCREMENT,
    job TEXT NOT NULL REFERENCES jobs(id),
    project TEXT NOT NULL,
    kind TEXT NOT NULL,
    data TEXT NOT NULL,
    time REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS project_event_cursor ON events(project, cursor);
CREATE TABLE IF NOT EXISTS submissions (
    id TEXT PRIMARY KEY, token TEXT NOT NULL UNIQUE, response TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS submission_projects (
    submission TEXT NOT NULL REFERENCES submissions(id),
    project TEXT NOT NULL,
    PRIMARY KEY (submission, project)
);
CREATE TABLE IF NOT EXISTS records (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS policy (
    slot TEXT PRIMARY KEY,
    capacity INTEGER NOT NULL CHECK (capacity > 0)
);
INSERT OR IGNORE INTO policy VALUES ('compute', 4), ('light', 1);
"""

LIVE = "('queued','assigned','running','stopping')"
ACTIVE = "('assigned','running','stopping')"
TERMINAL = "('succeeded','stopped','failed','interrupted','blocked')"

JOB_ID = re.compile(r"[a-zA-Z0-9_-]{1,128}")
JSON_COLUMNS = ("data", "phase", "progress", "checkpoint", "outcome")


def dump(value) -> str:
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def require_text(value, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise HostError("invalid_request", f"{field} must be a nonempty string.")
    return value


class Store:
    def __init__(self, path: Path):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)
        self.event_head = self.db.execute("SELECT coalesce(max(cursor), 0) FROM events").fetchone()[
            0
        ]

    def get(self, job_id: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()

    def append(self, row: sqlite3.Row, kind: str, data: dict) -> None:
        self.db.execute(
            "INSERT INTO events(job, project, kind, data, time) VALUES (?,?,?,?,?)",
            (row["id"], row["project"], kind, dump(data), time.time()),
        )

    def finished(self, row: sqlite3.Row, outcome: dict) -> None:
        self.db.execute(
            """UPDATE jobs SET state=?, outcome=?, checkpoint=coalesce(checkpoint, ?) WHERE id=?""",
            (
                outcome["outcome"],
                dump(outcome),
                dump(outcome["checkpoint"]) if outcome.get("checkpoint") is not None else None,
                row["id"],
            ),
        )
        self.append(row, "terminal", outcome)
        if outcome["outcome"] == "succeeded":
            self.db.execute(
                "UPDATE edges SET satisfied=1 WHERE parent=? AND publication=''",
                (row["id"],),
            )

    def ingest(self, row: sqlite3.Row, kind: str, data: dict) -> None:
        job_id = row["id"]
        if kind == "accepted":
            self.db.execute(
                "UPDATE jobs SET state='running' WHERE id=? AND state='assigned'", (job_id,)
            )
            data = {}
        elif kind in ("progress", "phase"):
            self.db.execute(f"UPDATE jobs SET {kind}=? WHERE id=?", (dump(data), job_id))
        elif kind == "publication":
            key = require_text(data.get("key"), "publication key")
            self.db.execute("INSERT OR IGNORE INTO publications VALUES (?,?)", (job_id, key))
            self.db.execute(
                "UPDATE edges SET satisfied=1 WHERE parent=? AND publication=?", (job_id, key)
            )
            if data.get("checkpoint") is not None:
                self.db.execute(
                    "UPDATE jobs SET checkpoint=? WHERE id=?", (dump(data["checkpoint"]), job_id)
                )
        elif kind == "terminal":
            outcome = dict(data)
            if row["event_error"] and outcome["outcome"] == "succeeded":
                outcome = {
                    "outcome": "failed",
                    "error": {
                        "code": "worker_event_protocol",
                        "message": row["event_error"],
                    },
                }
            if row["stop_mode"] == "force" and "signal" in outcome:
                outcome["outcome"] = "stopped"
                outcome.pop("error", None)
            self.finished(row, outcome)
            return
        self.append(row, kind, data)

    def _validate_job(self, job: dict) -> None:
        if not JOB_ID.fullmatch(require_text(job.get("id"), "job id")):
            raise HostError("invalid_request", "Job id is not safe for a host path.")
        require_text(job.get("project"), "project")
        require_text(job.get("request"), "request")
        if job.get("slot") not in ("light", "compute"):
            raise HostError("invalid_request", "Invalid slot.")
        launch = job["launch"]
        argv = launch["argv"]
        if not argv or not all(isinstance(arg, str) and "\0" not in arg for arg in argv):
            raise HostError(
                "invalid_request", "Launch argv must have a command and valid arguments."
            )
        if not Path(argv[0]).is_absolute():
            raise HostError("invalid_request", "Launch command must be an absolute path.")
        for key, value in launch["env"].items():
            if not key or "=" in key or "\0" in key or not isinstance(value, str) or "\0" in value:
                raise HostError("invalid_request", "Invalid launch environment variable.")
        if launch["warm_entrypoint"] and ":" not in launch["warm_entrypoint"]:
            raise HostError("invalid_request", "Warm entrypoint must be module:function.")

    def admit(self, token: str, items: list[dict], record: dict) -> dict:
        require_text(token, "submission token")
        with self.db:
            previous = self.db.execute(
                "SELECT response FROM submissions WHERE token=?", (token,)
            ).fetchone()
            if previous:
                return json.loads(previous[0])
            results, aliases, inserted = [], {}, []
            projects = set(record.get("projects", ()))
            if record.get("project"):
                projects.add(record["project"])
            for item in items:
                job = item["job"]
                run_key = require_text(item["run_key"], "run key")
                if job is None:
                    disposition = item["disposition"]
                    if disposition not in ("reused", "rejected"):
                        raise HostError(
                            "invalid_request", "Admission item needs a job or disposition."
                        )
                    results.append(
                        {
                            "run_key": run_key,
                            "job_id": None,
                            "disposition": disposition,
                            "reason": item["reason"],
                        }
                    )
                    continue
                self._validate_job(job)
                projects.add(job["project"])
                by_id = self.get(job["id"])
                live = self.db.execute(
                    f"SELECT id, continues FROM jobs WHERE project=? AND request=? AND state IN {LIVE}",
                    (job["project"], job["request"]),
                ).fetchone()
                if by_id is not None:
                    if (by_id["project"], by_id["request"]) != (job["project"], job["request"]):
                        raise HostError(
                            "job_id_conflict", "Job id already names a different request."
                        )
                    job_id, disposition = by_id["id"], "admitted"
                elif live is not None:
                    if job["continues"] is not None and live["continues"] != job["continues"]:
                        raise HostError(
                            "invalid_resume", "Another attempt of this request is already running."
                        )
                    job_id, disposition = live["id"], "attached"
                else:
                    self._require_admission_current(job, item["checked_through"])
                    job_id, disposition = job["id"], "admitted"
                    self.db.execute(
                        "INSERT INTO jobs(id, project, request, launch, slot, data) VALUES (?,?,?,?,?,?)",
                        (
                            job_id,
                            job["project"],
                            job["request"],
                            dump(job["launch"]),
                            job["slot"],
                            dump(job["data"]),
                        ),
                    )
                    inserted.append(job)
                    self.append(self.get(job_id), "queued", {})
                aliases[job["id"]] = job_id
                results.append(
                    {
                        "run_key": run_key,
                        "job_id": job_id,
                        "disposition": disposition,
                        "reason": None,
                    }
                )
            for job in inserted:
                self._bind_edges(job, aliases)
            response = {"id": uuid4().hex, "token": token, "record": record, "items": results}
            self.db.execute(
                "INSERT INTO submissions VALUES (?,?,?)", (response["id"], token, dump(response))
            )
            self.db.executemany(
                "INSERT INTO submission_projects VALUES (?,?)",
                ((response["id"], project) for project in projects),
            )
        return response

    def _require_admission_current(self, job: dict, checked_through: int | None) -> None:
        if checked_through is None:
            return
        if type(checked_through) is not int or checked_through < 0:
            raise HostError("invalid_request", "Admission cursor must be a nonnegative integer.")
        settled = self.db.execute(
            """SELECT 1 FROM events e JOIN jobs j ON j.id=e.job
               WHERE e.project=? AND e.cursor>? AND e.kind='terminal'
               AND j.request=? AND j.state='succeeded' LIMIT 1""",
            (job["project"], checked_through, job["request"]),
        ).fetchone()
        if settled:
            raise HostError(
                "admission_changed", f"Request {job['request']} succeeded during admission."
            )

    def _bind_edges(self, job: dict, aliases: dict[str, str]) -> None:
        job_id, continues = job["id"], job["continues"]
        if continues is not None:
            source = self.get(continues)
            if (
                source is None
                or source["state"] not in TERMINAL_STATES
                or (source["project"], source["request"]) != (job["project"], job["request"])
            ):
                raise HostError(
                    "invalid_resume", "The earlier job must be settled at the same request."
                )
            self.db.execute("UPDATE jobs SET continues=? WHERE id=?", (continues, job_id))
        for edge in job["edges"]:
            parent = aliases.get(edge["job_id"], edge["job_id"])
            publication = edge["publication"] or ""
            source = self.get(parent)
            if publication:
                satisfied = (
                    self.db.execute(
                        "SELECT 1 FROM publications WHERE job=? AND key=?", (parent, publication)
                    ).fetchone()
                    is not None
                )
            else:
                satisfied = source is not None and source["state"] == "succeeded"
            cycle = self.db.execute(
                """WITH RECURSIVE ancestors(id) AS (
                       VALUES (?) UNION SELECT edges.parent FROM edges JOIN ancestors ON edges.job=ancestors.id
                   ) SELECT 1 FROM ancestors WHERE id=?""",
                (parent, job_id),
            ).fetchone()
            if cycle:
                raise HostError("dependency_cycle", f"Job {job_id} has a cyclic dependency.")
            self.db.execute(
                "INSERT OR IGNORE INTO edges VALUES (?,?,?,?)",
                (job_id, parent, publication, int(satisfied)),
            )

    def submissions(self, *, id: str | None, token: str | None, project: str | None) -> list[dict]:
        rows = self.db.execute(
            """SELECT response FROM submissions s
               WHERE (? IS NULL OR id=?) AND (? IS NULL OR token=?)
                 AND (? IS NULL OR EXISTS (SELECT 1 FROM submission_projects p
                                            WHERE p.submission=s.id AND p.project=?))
               ORDER BY rowid""",
            (id, id, token, token, project, project),
        )
        return [json.loads(row[0]) for row in rows]

    def jobs(
        self,
        *,
        project: str | None,
        request: str | None,
        job_id: str | None,
        include_launch: bool,
        job_ids: list[str] | None = None,
    ) -> list[dict]:
        selection = "AND id IN (" + ",".join("?" * len(job_ids)) + ")" if job_ids else ""
        rows = self.db.execute(
            f"""SELECT * FROM jobs WHERE (? IS NULL OR project=?) AND (? IS NULL OR request=?)
                AND (? IS NULL OR id=?) {selection} ORDER BY seq""",
            (project, project, request, request, job_id, job_id, *(job_ids or ())),
        )
        result = []
        for row in rows:
            view = {
                key: row[key]
                for key in (
                    "id",
                    "project",
                    "request",
                    "slot",
                    "continues",
                    "state",
                    "stop_mode",
                    "delivery",
                    "worker_pid",
                )
            }
            view.update(
                {
                    key: json.loads(row[key]) if row[key] is not None else None
                    for key in JSON_COLUMNS
                }
            )
            if include_launch:
                view["launch"] = json.loads(row["launch"])
            view["edges"] = [
                {
                    "job_id": edge["parent"],
                    "publication": edge["publication"] or None,
                    "satisfied": bool(edge["satisfied"]),
                }
                for edge in self.db.execute("SELECT * FROM edges WHERE job=?", (row["id"],))
            ]
            result.append(view)
        return result

    def events(self, after: int, projects: list[str], limit: int) -> dict:
        project_filter = (
            "AND e.project IN (" + ",".join("?" * len(projects)) + ")" if projects else ""
        )
        rows = self.db.execute(
            f"""SELECT e.cursor, e.job AS job_id, e.project, j.request, e.kind, e.data, e.time
                FROM events e JOIN jobs j ON j.id=e.job
                WHERE cursor>? {project_filter} ORDER BY cursor LIMIT ?""",
            (after, *projects, limit + 1),
        )
        events, remaining = [], 8 * 1024 * 1024
        for row in rows:
            remaining -= len(row["data"].encode())
            if len(events) == limit or (remaining < 0 and events):
                cursor = events[-1]["cursor"]
                break
            events.append({**row, "data": json.loads(row["data"])})
        else:
            cursor = max(after, self.event_head)
        return {"events": events, "cursor": cursor, "head": self.event_head, "oldest_cursor": 0}

    def jobs_snapshot(self, **filters) -> dict:
        return {"cursor": self.event_head, "oldest_cursor": 0, "jobs": self.jobs(**filters)}

    def record_get_many(self, keys: list[str]) -> dict:
        return {
            key: json.loads(value)
            for key, value in self.db.execute(
                "SELECT key, value FROM records WHERE key IN (" + ",".join("?" * len(keys)) + ")",
                keys,
            )
        }

    def record_get(self, key: str) -> dict | None:
        return self.record_get_many([key]).get(key)

    def record_put(self, key: str, value: dict) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO records VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, dump(value)),
            )

    def mark_satisfied(self, job_id: str, edge: dict) -> None:
        with self.db:
            updated = self.db.execute(
                "UPDATE edges SET satisfied=1 WHERE job=? AND parent=? AND publication=?",
                (job_id, edge["job_id"], edge["publication"] or ""),
            )
            if not updated.rowcount:
                raise HostError("unknown_edge", f"Job {job_id} has no such dependency.")

    def policy(self, *, compute: int | None = None, light: int | None = None) -> dict[str, int]:
        with self.db:
            for slot, value in (("compute", compute), ("light", light)):
                if value is None:
                    continue
                if type(value) is not int or not 1 <= value <= 64:
                    raise HostError("invalid_request", "Slot capacity must be from 1 to 64.")
                self.db.execute("UPDATE policy SET capacity=? WHERE slot=?", (value, slot))
        return dict(self.db.execute("SELECT slot, capacity FROM policy"))

    def stop(self, job_ids: list[str], mode: str) -> list[dict]:
        if mode not in ("checkpoint", "force"):
            raise HostError("invalid_request", "Unknown stop mode.")
        results = []
        with self.db:
            for job_id in job_ids:
                row = self.get(job_id)
                if row is None:
                    results.append(
                        {
                            "job_id": job_id,
                            "error": {"code": "unknown_job", "message": "Job does not exist."},
                        }
                    )
                    continue
                state = row["state"]
                if state in LIVE_STATES and row["stop_mode"] not in (mode, "force"):
                    self.db.execute("UPDATE jobs SET stop_mode=? WHERE id=?", (mode, job_id))
                    if state == "queued":
                        self.finished(row, {"outcome": "stopped"})
                        state = "stopped"
                    else:
                        self.db.execute("UPDATE jobs SET state='stopping' WHERE id=?", (job_id,))
                        self.append(row, "stopping", {"mode": mode})
                        state = "stopping"
                results.append({"job_id": job_id, "state": state})
        return results

    def settle_blocked(self) -> None:
        with self.db:
            rows = self.db.execute(
                f"""WITH RECURSIVE blocked(id) AS (
                        SELECT e.job FROM edges e JOIN jobs parent ON parent.id=e.parent
                        WHERE NOT e.satisfied AND parent.state IN {TERMINAL}
                        UNION SELECT e.job FROM edges e JOIN blocked b ON b.id=e.parent WHERE NOT e.satisfied
                    ) SELECT * FROM jobs WHERE state='queued' AND id IN blocked"""
            ).fetchall()
            for row in rows:
                self.finished(
                    row,
                    {
                        "outcome": "blocked",
                        "error": {
                            "code": "dependency_ended",
                            "message": "A prerequisite ended before satisfying this job.",
                        },
                    },
                )

    def ready(self) -> list[sqlite3.Row]:
        return self.db.execute(
            f"""WITH ready AS (
                   SELECT jobs.*, row_number() OVER (PARTITION BY slot ORDER BY seq) AS position
                   FROM jobs WHERE state='queued'
                   AND NOT EXISTS (SELECT 1 FROM edges WHERE job=jobs.id AND NOT satisfied)
               )
               SELECT ready.* FROM ready JOIN policy USING (slot)
               WHERE position <= capacity -
                   (SELECT count(*) FROM jobs active WHERE active.slot=ready.slot AND active.state IN {ACTIVE})
               ORDER BY seq"""
        ).fetchall()

    def active(self) -> list[sqlite3.Row]:
        return self.db.execute(
            f"SELECT * FROM jobs WHERE state IN {ACTIVE} ORDER BY seq"
        ).fetchall()
