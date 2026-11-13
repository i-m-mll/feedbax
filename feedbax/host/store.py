"""Transactions for the host's jobs, submissions and durable event cursor."""

import errno
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Literal, NamedTuple
from uuid import uuid4

from feedbax.host.checkpoint_selection import (
    CheckpointSelection, is_checkpoint_producer, requested_checkpoint_selection, selector_sha256, verify_selected_publication,
)
from feedbax.host.completion import CompletionFileError, train_source_slots, verify_completion_files
from feedbax.host.plan_history import advance_plan, plan_ref_key
from feedbax.host.protocol import LIVE_STATES, TERMINAL_STATES, HostError, can_resume_attempt


class TrainPublicationPending(Exception):
    """The accepted worker terminal event can be retried without recomputing."""


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
CREATE INDEX IF NOT EXISTS jobs_by_continues ON jobs(continues, seq);
CREATE TABLE IF NOT EXISTS edges (
    job TEXT NOT NULL REFERENCES jobs(id),
    parent TEXT NOT NULL,
    publication TEXT NOT NULL,
    satisfied INTEGER NOT NULL DEFAULT 0,
    authenticated_completion INTEGER NOT NULL DEFAULT 0,
    binding TEXT,
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
CREATE INDEX IF NOT EXISTS job_event_kind ON events(job, kind, cursor);
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
TRAIN_REQUEST_ID = re.compile(r"train:[0-9a-f]{64}")
JSON_COLUMNS = ("data", "phase", "progress", "checkpoint", "outcome")


class _CompletionRejection(NamedTuple):
    candidate_job_id: str
    stage: Literal["continuation", "completion", "checkpoint_selection"]
    message: str


def dump(value, *, sort_keys: bool = False) -> str:
    return json.dumps(value, sort_keys=sort_keys, separators=(",", ":"), allow_nan=False)


def _canonical_binding_conflicts(
    launch: dict, stored_launch: str, *, evaluate: bool, live: bool = False,
) -> bool:
    recorded = json.loads(stored_launch)
    if evaluate:
        return dump(launch, sort_keys=True) != dump(recorded, sort_keys=True)
    payload, stored = launch["payload"], recorded["payload"]
    if payload.get("canonical_request") is None and stored.get("canonical_request") is None:
        return False
    if live:
        payload = {key: value for key, value in payload.items() if key != "resume"}
        stored.pop("resume", None)
    return dump(payload, sort_keys=True) != dump(stored, sort_keys=True)


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
        if "authenticated_completion" not in {
            row["name"] for row in self.db.execute("PRAGMA table_info(edges)")
        }:
            self.db.executescript("""
                BEGIN;
                ALTER TABLE edges ADD COLUMN authenticated_completion INTEGER NOT NULL DEFAULT 0;
                ALTER TABLE edges ADD COLUMN binding TEXT;
                COMMIT;
            """)
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
                "UPDATE edges SET satisfied=1 WHERE parent=? AND publication='' AND NOT authenticated_completion",
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
            if self.get(job_id)["state"] in TERMINAL_STATES:
                return
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
            if outcome["outcome"] == "succeeded" and "model_product" in outcome:
                outcome = self._publish_train_result(row, outcome)
            self.finished(row, outcome)
            return
        self.append(row, kind, data)

    def _publish_train_result(self, row: sqlite3.Row, outcome: dict) -> dict:
        from feedbax.host.train_success import ContinuationEvidenceUnavailable, publish_train_success

        launch = json.loads(row["launch"])
        payload = launch["payload"]
        binding = payload.get("canonical_request")
        try:
            if launch["warm_entrypoint"] != "feedbax.training_host_worker:main" or binding is None:
                raise ValueError("Only canonical managed Train can publish a model product")
            self._check_completion_identity(row, outcome)
            predecessor = None
            if binding.get("checkpoint") is not None:
                (edge,) = self.completion_bindings(row["id"])
                pinned = edge["completion"]
                parent = self.get(pinned["producer_job_id"])
                if parent is None or parent["state"] != "succeeded":
                    raise ValueError("Pinned Train predecessor did not succeed")
                prior = json.loads(parent["outcome"])
                if prior["completion"] != pinned:
                    raise ValueError("Pinned Train predecessor differs from its accepted completion")
                if "store_success" not in prior:
                    return outcome | {"store_publication_error": {
                        "code": "input_unavailable", "message": "Pinned Train predecessor has no typed success receipt",
                    }}
                predecessor = pinned, prior
            ref = publish_train_success(
                Path(payload["manifest_root"]), row["id"], row["request"], payload, outcome, predecessor,
            )
        except ContinuationEvidenceUnavailable as error:
            return outcome | {"store_publication_error": {
                "code": "input_unavailable", "message": str(error),
            }}
        except (ValueError, KeyError, TypeError, OSError) as error:
            if isinstance(error, OSError) and error.errno in {
                errno.ENOSPC, errno.EDQUOT, errno.EIO, errno.EBUSY, errno.EINTR,
            }:
                raise TrainPublicationPending(str(error)) from error
            return {
                "outcome": "failed", "receipt": outcome.get("receipt"),
                "error": {"code": "publication_failed", "message": str(error)},
            }
        return outcome | {"store_success": ref}

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

    def admit(
        self, token: str, items: list[dict], record: dict, *, allow_completion: bool = False
    ) -> dict:
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
                payload = job["launch"]["payload"]
                canonical_request = payload.get("canonical_request")
                evaluate = job["request"].startswith("evaluate:") and canonical_request is not None
                input_path = "/inputs/subject" if evaluate else "/inputs/checkpoint"
                if canonical_request is not None and not isinstance(canonical_request, dict):
                    raise HostError("invalid_request", "Invalid canonical request binding.")
                if canonical_request is not None and (
                    payload.get("project_id"), payload.get("request_id")
                ) != (job["project"], job["request"]):
                    raise HostError("invalid_request", "Canonical launch identity differs from its job.")
                for edge in job["edges"]:
                    completion = edge.get("authenticated_completion", False)
                    if type(completion) is not bool or (
                        completion and (edge["publication"] is not None or not allow_completion)
                    ):
                        raise HostError("invalid_request", "Invalid authenticated completion edge.")
                try:
                    selected = requested_checkpoint_selection(job["request"], job["launch"])
                    if selected is not None and not evaluate:
                        edges = [edge for edge in job["edges"] if edge.get("authenticated_completion")]
                        if len(edges) != 1 or job["continues"] is not None:
                            raise ValueError("Private evaluation requires one exact producer edge.")
                        producer = self.get(aliases.get(edges[0]["job_id"], edges[0]["job_id"]))
                        source = json.loads(producer["launch"])["payload"] if producer is not None else {}
                        if (
                            producer is None or producer["id"] != payload["producer"]
                            or producer["project"] != job["project"]
                            or producer["request"] != payload["producer_request"]
                            or source.get("manifest_root") != payload["manifest_root"]
                            or source.get("canonical_request") is None
                            or not set(selected.coordinates) <= set(source.get("checkpoint_catalog", ()))
                        ):
                            raise ValueError("Private evaluation differs from its admitted Train catalogue.")
                except (ValueError, KeyError, TypeError) as exc:
                    raise HostError("invalid_request", str(exc)) from exc
                checkpoint = canonical_request.get("checkpoint") if canonical_request is not None else None
                checkpoint_parent = None
                if evaluate and (
                    checkpoint is None or job["continues"] is not None or len(job["edges"]) != 1
                ):
                    raise HostError("invalid_request", f"{input_path}: Evaluate requires a Train subject.")
                if checkpoint is not None and job["continues"] is None:
                    edges = [edge for edge in job["edges"] if edge.get("authenticated_completion")]
                    if len(edges) != 1 or not isinstance(checkpoint, dict):
                        raise HostError("invalid_request", f"{input_path}: requires one authenticated checkpoint edge.")
                    parent_id = edges[0]["job_id"]
                    checkpoint_parent = aliases.get(parent_id, parent_id)
                    source = self.get(checkpoint_parent)
                    if source is None:
                        results.append({
                            "run_key": run_key, "job_id": None, "disposition": "rejected",
                            "reason": f"{input_path}: the selected predecessor was not admitted",
                        })
                        continue
                    source_payload = json.loads(source["launch"])["payload"]
                    if (
                        source["project"] != job["project"]
                        or source["request"] != checkpoint.get("request_id")
                        or source_payload.get("manifest_root") != payload.get("manifest_root")
                        or source_payload.get("canonical_request") is None
                    ):
                        raise HostError("invalid_request", f"{input_path}: checkpoint edge differs from its canonical input.")
                    if (
                        selected is not None and selected.requires_catalog
                        and not set(selected.coordinates) <= set(source_payload.get("checkpoint_catalog", ()))
                    ):
                        results.append({
                            "run_key": run_key, "job_id": None, "disposition": "rejected",
                            "reason": f"{input_path}/checkpoint: admitted Train catalogue omits requested batches",
                        })
                        continue
                projects.add(job["project"])
                by_id = self.get(job["id"])
                live = self.db.execute(
                    f"SELECT id, continues, launch FROM jobs WHERE project=? AND request=? AND state IN {LIVE}",
                    (job["project"], job["request"]),
                ).fetchone()
                if by_id is not None:
                    if (by_id["project"], by_id["request"]) != (job["project"], job["request"]):
                        raise HostError(
                            "job_id_conflict", "Job id already names a different request."
                        )
                    if _canonical_binding_conflicts(job["launch"], by_id["launch"], evaluate=evaluate) or (
                        checkpoint_parent is not None
                        and self._completion_parent(by_id["id"]) != checkpoint_parent
                    ):
                        raise HostError(
                            "job_id_conflict", "Job id already names a different canonical binding."
                        )
                    job_id, disposition = by_id["id"], "admitted"
                elif live is not None:
                    if _canonical_binding_conflicts(
                        job["launch"], live["launch"], evaluate=evaluate, live=True,
                    ) or (
                        checkpoint_parent is not None
                        and self._completion_parent(live["id"]) != checkpoint_parent
                    ):
                        results.append({
                            "run_key": run_key,
                            "job_id": None,
                            "disposition": "rejected",
                            "reason": (
                                f"{input_path}: the live Evaluate has a different execution binding."
                                if evaluate else
                                "The live Train request has a different store or execution binding."
                            ),
                        })
                        continue
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
            self._advance_authored_train_plan(record, results)
            response = {"id": uuid4().hex, "token": token, "record": record, "items": results}
            self.db.execute(
                "INSERT INTO submissions VALUES (?,?,?)", (response["id"], token, dump(response))
            )
            self.db.executemany(
                "INSERT INTO submission_projects VALUES (?,?)",
                ((response["id"], project) for project in projects),
            )
        return response

    def _advance_authored_train_plan(self, record: dict, results: list[dict]) -> None:
        plan = record.get("authored_train_plan")
        if plan is None or not any(item["job_id"] is not None for item in results):
            return
        command = record.get("command") or {}
        if command.get("kind") != "canonical_train":
            raise HostError("invalid_request", "An authored Train Plan requires canonical admission.")
        project = require_text(record["project"], "project")
        root = require_text(command.get("store"), "store")
        current = plan["requests"]
        if not isinstance(current, dict) or not all(
            isinstance(key, str) and isinstance(value, str) and TRAIN_REQUEST_ID.fullmatch(value)
            for key, value in current.items()
        ):
            raise HostError("invalid_request", "Invalid authored Train Plan request map.")
        for item in results:
            if item["job_id"] is None:
                continue
            job = self.get(item["job_id"])
            payload = json.loads(job["launch"])["payload"] if job is not None else {}
            if (
                job is None
                or (job["project"], job["request"]) != (project, current.get(item["run_key"]))
                or json.loads(job["data"]).get("manifest_root") != root
                or payload.get("canonical_request") is None
                or payload.get("manifest_root") != root
            ):
                raise HostError("invalid_request", "Admitted Train does not match the authored Plan.")
        key = plan_ref_key(project, root)
        previous = self.record_get(key)
        requests = {row["request_id"] for row in previous["rows"].values()} if previous else set()
        successes = {
            request_id: {"request_id": request_id, "job_id": job_id}
            for request_id, job_id in self.db.execute(
                """SELECT j.request, j.id FROM jobs j
                   WHERE j.project=? AND j.state='succeeded'
                   AND j.request IN (SELECT value FROM json_each(?))
                   AND json_extract(j.data, '$.manifest_root')=?
                   AND json_extract(j.outcome, '$.completion.request_id')=j.request
                   AND json_extract(j.outcome, '$.completion.project_id')=j.project
                   AND json_extract(j.outcome, '$.completion.producer_job_id')=j.id
                   AND json_extract(j.outcome, '$.completion.manifest_root')=?
                   ORDER BY j.seq""",
                (project, dump(sorted(requests)), root, root),
            )
        }
        value = advance_plan(plan["object"], current, previous, successes)
        self._write_record(key, value)

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

    def _completion_parent(self, job_id: str) -> str | None:
        row = self.db.execute(
            "SELECT parent FROM edges WHERE job=? AND authenticated_completion", (job_id,)
        ).fetchone()
        return row[0] if row is not None else None

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
            self.db.execute(
                """INSERT INTO edges(job, parent, publication, satisfied, authenticated_completion, binding)
                   SELECT ?, parent, publication, satisfied, authenticated_completion, binding
                   FROM edges WHERE job=? AND authenticated_completion""",
                (job_id, continues),
            )
        for edge in job["edges"]:
            parent = aliases.get(edge["job_id"], edge["job_id"])
            publication = edge["publication"] or ""
            completion = edge.get("authenticated_completion", False)
            source = self.get(parent)
            if completion:
                if source is None or source["project"] != job["project"]:
                    raise HostError(
                        "invalid_request",
                        "Completion requires an admitted producer in the same project.",
                    )
                satisfied = False
            elif publication:
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
            existing = self.db.execute(
                "SELECT authenticated_completion FROM edges WHERE job=? AND parent=? AND publication=?",
                (job_id, parent, publication),
            ).fetchone()
            if continues is not None and completion and existing is None:
                raise HostError(
                    "invalid_resume", "A continuation cannot add new completion inputs."
                )
            if existing is not None and bool(existing[0]) != completion:
                raise HostError("invalid_request", "Conflicting dependencies on the same producer.")
            self.db.execute(
                """INSERT OR IGNORE INTO edges(job, parent, publication, satisfied, authenticated_completion)
                   VALUES (?,?,?,?,?)""",
                (job_id, parent, publication, int(satisfied), int(completion)),
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
                    **(
                        {
                            "authenticated_completion": True,
                            "binding": json.loads(edge["binding"])
                            if edge["binding"] is not None
                            else None,
                        }
                        if edge["authenticated_completion"]
                        else {}
                    ),
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

    def jobs_snapshot(self, *, record_keys: list[str] | None = None, **filters) -> dict:
        snapshot = {"cursor": self.event_head, "oldest_cursor": 0, "jobs": self.jobs(**filters)}
        if record_keys:
            snapshot["records"] = self.record_get_many(record_keys)
        return snapshot

    def stopped_checkpoint(self, *, job_id: str, project: str, manifest_root: str) -> int | None:
        """Project exact stopped subjects through the same binding check used at delivery."""
        job = self.db.execute(
            "SELECT request, data, outcome FROM jobs WHERE id=? AND project=? AND state='stopped'",
            (job_id, project),
        ).fetchone()
        if (
            job is None or not job["outcome"] or not job["request"].startswith("train:")
            or json.loads(job["data"]).get("manifest_root") != manifest_root
        ):
            return None
        checkpoint = json.loads(job["outcome"]).get("checkpoint")
        update = checkpoint.get("completed_updates") if isinstance(checkpoint, dict) else None
        if type(update) is not int or update <= 0:
            return None
        anchors = [
            row["id"] for row in self.db.execute(
                "SELECT id, launch FROM jobs WHERE project=? AND request=? AND continues IS NULL",
                (project, job["request"]),
            )
            if is_checkpoint_producer(json.loads(row["launch"]), continues=None, manifest_root=manifest_root)
        ]
        if len(anchors) != 1:
            return None
        binding, _, _ = self._completion_candidate(anchors[0], CheckpointSelection((update,), False))
        if binding is not None:
            (member,) = binding["checkpoint_selection"]["members"]
            if member["producer_job_id"] == job_id and member["checkpoint"] == checkpoint:
                return update
        return None

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
            self._write_record(key, value)

    def _write_record(self, key: str, value: dict) -> None:
        self.db.execute(
            "INSERT INTO records VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, dump(value)),
        )

    def mark_satisfied(self, job_id: str, edge: dict) -> None:
        with self.db:
            updated = self.db.execute(
                """UPDATE edges SET satisfied=1 WHERE job=? AND parent=? AND publication=?
                   AND NOT authenticated_completion""",
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

    def _completed_product(
        self, attempt: sqlite3.Row, train_slots: frozenset[str] | None = None,
    ) -> tuple[dict | None, str | None]:
        """Authenticate a successful attempt's product or report its absence."""
        outcome = json.loads(attempt["outcome"])
        completion = outcome.get("completion")
        if completion is None:
            reported = outcome.get("completion_error")
            if reported is None:
                return None, "Producer has no authenticated completion."
            if (
                not isinstance(reported, dict)
                or reported.get("code") not in ("input_unavailable", "dependency_corrupt")
                or not isinstance(reported.get("message"), str)
            ):
                raise CompletionFileError("Producer reported malformed completion evidence.")
            if reported["code"] == "dependency_corrupt":
                raise CompletionFileError(reported["message"])
            return None, reported["message"]
        self._check_completion_identity(attempt, outcome)
        verify_completion_files(completion, train_slots=train_slots)
        return completion, None

    def _check_completion_identity(self, attempt: sqlite3.Row, outcome: dict) -> None:
        completion = outcome.get("completion")
        binding_event = self.db.execute(
            """SELECT data FROM events WHERE job=? AND kind='binding' ORDER BY cursor DESC LIMIT 1""",
            (attempt["id"],),
        ).fetchone()
        launch = json.loads(attempt["launch"])["payload"]
        if (
            not isinstance(completion, dict)
            or binding_event is None
            or not isinstance(completion.get("binding_sha256"), str)
            or completion.get("binding_sha256") != json.loads(binding_event[0]).get("binding_sha256")
            or completion.get("producer_job_id") != attempt["id"]
            or completion.get("project_id") != attempt["project"]
            or completion.get("request_id") != attempt["request"]
            or completion.get("manifest_root") != launch.get("manifest_root")
            or completion.get("receipt") != outcome.get("receipt")
            or completion.get("checkpoint") != outcome.get("checkpoint")
            or not completion.get("receipt")
            or not completion.get("checkpoint")
        ):
            raise CompletionFileError(
                "Completion does not match its admitted producer and terminal outcome."
            )

    def _checkpoint_members(
        self, path: tuple[sqlite3.Row, ...], selection: CheckpointSelection,
    ) -> tuple[list[int], list[dict]]:
        """Collect one path through exact ancestor recovery pointers."""
        coordinates = selection.coordinates
        found = {}
        keys = [f"checkpoint:{coordinate}" for coordinate in coordinates]
        placeholders = ",".join("?" for _ in keys)
        for index, attempt in enumerate(path):
            boundary = (
                json.loads(path[index + 1]["launch"])["payload"]["resume"]
                if index + 1 < len(path) else None
            )
            if index + 1 < len(path) and boundary is None:
                continue
            upper = boundary["completed_updates"] if boundary is not None else None
            resume = json.loads(attempt["launch"])["payload"].get("resume")
            lower = resume["completed_updates"] if resume is not None else 0
            rows = self.db.execute(
                f"""SELECT cursor, data FROM events WHERE job=? AND kind='publication'
                    AND json_extract(data, '$.key') IN ({placeholders})
                    AND cursor < COALESCE(
                        (SELECT MIN(cursor) FROM events WHERE job=? AND kind='terminal'),
                        9223372036854775807
                    ) ORDER BY cursor""",
                (attempt["id"], *keys, attempt["id"]),
            )
            for row in rows:
                publication = json.loads(row["data"])
                if (
                    not selection.requires_catalog and not selection.allows_gaps
                    and "checkpoint_product" not in publication
                    and "checkpoint_product_error" not in publication
                ):
                    # Older recovery-only publications did not attest an Evaluate product.
                    continue
                coordinate = int(publication["key"].split(":")[1])
                if coordinate <= lower or (upper is not None and (
                    coordinate > upper
                    or (coordinate == upper and publication.get("checkpoint") != boundary)
                )):
                    continue
                previous = found.get(coordinate)
                if previous is not None:
                    previous_attempt, previous_publication, _ = previous
                    if (
                        previous_attempt["id"] != attempt["id"]
                        or previous_publication.get("checkpoint") != publication.get("checkpoint")
                        or previous_publication.get("checkpoint_product") != publication.get("checkpoint_product")
                    ):
                        raise CompletionFileError(f"Ambiguous checkpoint publication at batch {coordinate}.")
                    continue
                found[coordinate] = (attempt, publication, row["cursor"])
        missing = [coordinate for coordinate in coordinates if coordinate not in found]
        if missing and not selection.allows_gaps:
            return missing, []
        members = []
        for coordinate in coordinates:
            if coordinate not in found:
                continue
            attempt, publication, cursor = found[coordinate]
            binding = self.db.execute(
                "SELECT data FROM events WHERE job=? AND kind='binding' ORDER BY cursor LIMIT 1",
                (attempt["id"],),
            ).fetchone()
            if binding is None:
                raise CompletionFileError("Checkpoint publication has no admitted binding event.")
            launch = json.loads(attempt["launch"])["payload"]
            if "checkpoint_product_error" in publication:
                raise CompletionFileError(publication["checkpoint_product_error"])
            digest = json.loads(binding[0]).get("binding_sha256")
            if not isinstance(digest, str):
                raise CompletionFileError("Checkpoint publication lacks a valid binding digest.")
            member = verify_selected_publication(
                publication,
                attempt=attempt["id"], project=attempt["project"], request=attempt["request"],
                root=launch["manifest_root"], binding_sha256=digest,
            )
            members.append(member | {"publication_cursor": cursor})
        return missing, members

    def _series_closure(self, attempt: sqlite3.Row, path: tuple[sqlite3.Row, ...]) -> dict | None:
        if attempt["state"] == "succeeded":
            completion, _ = self._completed_product(attempt)
            return {"kind": "completion", "product": completion} if completion is not None else None
        if attempt["state"] in {"stopped", "failed", "interrupted"} and attempt["checkpoint"] is not None:
            boundary = json.loads(attempt["checkpoint"])
            selection = CheckpointSelection((boundary["completed_updates"],), False)
            _, members = self._checkpoint_members(path, selection)
            if members and members[0]["checkpoint"] == boundary:
                return {"kind": "recovery", "product": members[0]}
        return None

    def _completion_candidate(
        self, anchor: str, selection: CheckpointSelection | None = None,
        train_slots: frozenset[str] | None = None,
    ) -> tuple[dict | None, dict | None, list[_CompletionRejection]]:
        """Select an authenticated final product or exact checkpoint publications.

        An earlier attempt can still produce another child whenever M1 permits its resume.
        """
        attempts = self.db.execute(
            """WITH RECURSIVE lineage(id) AS (
                   VALUES (?) UNION ALL
                   SELECT child.id FROM jobs child JOIN lineage ON child.continues=lineage.id
               )
               SELECT jobs.* FROM jobs JOIN lineage USING (id) ORDER BY jobs.seq""",
            (anchor,),
        )
        resume_checkpoints: dict[str, str] = {}
        paths: dict[str, tuple[sqlite3.Row, ...]] = {}
        pending = False
        rejections: list[_CompletionRejection] = []
        missing = None
        settled_series = None
        completed_series = None
        for attempt in attempts:
            attempt_id = attempt["id"]
            payload = json.loads(attempt["launch"])["payload"]
            resume = payload.pop("resume", None)
            if attempt_id == anchor:
                expected_request = (attempt["project"], attempt["request"])
                expected_payload = dump(payload, sort_keys=True)
            else:
                source_checkpoint = resume_checkpoints.get(attempt["continues"])
                if source_checkpoint is None:
                    continue
                if (
                    (attempt["project"], attempt["request"]) != expected_request
                    or dump(resume, sort_keys=True) != source_checkpoint
                    or dump(payload, sort_keys=True) != expected_payload
                ):
                    rejections.append(_CompletionRejection(
                        attempt_id, "continuation",
                        "Explicit continuation changed the producer's bound input.",
                    ))
                    continue

            can_publish = True
            absent = ()
            if selection is not None:
                prior = paths.get(attempt["continues"], ()) if attempt_id != anchor else ()
                path = (*prior, attempt)
                paths[attempt_id] = path
                try:
                    absent, members = self._checkpoint_members(path, selection)
                except CompletionFileError as error:
                    rejections.append(_CompletionRejection(attempt_id, "checkpoint_selection", str(error)))
                    # An exact publication is immutable within this path, including on resume.
                    can_publish = selection.requires_catalog and not selection.allows_gaps
                else:
                    if members or selection.allows_gaps:
                        coordinates = selection.coordinates
                        binding = {
                            "dependency_job_id": anchor,
                            "checkpoint_selection": {
                                "requested": list(coordinates),
                                "selector_sha256": selector_sha256(anchor, coordinates),
                                "lineage_tip": attempt_id,
                                "lineage": [member["id"] for member in path],
                                "members": members,
                            },
                        }
                        if not absent:
                            return binding, None, rejections
                        binding["checkpoint_selection"]["gaps"] = [
                            {"update": update, "reason": "checkpoint_unavailable"} for update in absent
                        ]
                        try:
                            closure = self._series_closure(attempt, path)
                        except CompletionFileError as error:
                            rejections.append(_CompletionRejection(attempt_id, "checkpoint_selection", str(error)))
                        else:
                            if closure is not None:
                                binding["checkpoint_selection"]["closure"] = closure
                                if attempt["state"] == "succeeded" and completed_series is None:
                                    completed_series = binding
                                elif settled_series is None:
                                    settled_series = binding
                    missing = f"Producer closed without checkpoint coordinates {absent}."

            state = attempt["state"]
            if state == "succeeded" and selection is None:
                try:
                    completion, message = self._completed_product(attempt, train_slots)
                except CompletionFileError as error:
                    rejections.append(_CompletionRejection(attempt_id, "completion", str(error)))
                else:
                    if completion is not None:
                        return {"dependency_job_id": anchor, "completion": completion}, None, rejections
                    missing = missing or message
            resumable = can_resume_attempt(state, attempt["checkpoint"], attempt["delivery"])
            boundary = resume
            if resumable:
                raw = attempt["checkpoint"]
                boundary = json.loads(raw) if raw is not None else None
                resume_checkpoints[attempt_id] = dump(boundary, sort_keys=True)
            if resumable or state in LIVE_STATES:
                # A continuation starts after its recovery coordinate.
                pending |= (selection is not None and selection.allows_gaps and state in LIVE_STATES)
                pending |= can_publish and (
                    selection is None or selection.requires_catalog
                    or boundary is None
                    or any(update > boundary["completed_updates"] for update in (
                        absent if selection.allows_gaps else selection.coordinates[:1]
                    ))
                )

        if completed_series is not None:
            return completed_series, None, rejections
        if pending:
            return None, None, rejections
        if rejections:
            return None, {
                "code": "dependency_corrupt",
                "message": rejections[0].message,
            }, rejections
        if settled_series is not None:
            return settled_series, None, rejections
        return None, {
            "code": "input_unavailable",
            "message": missing or "Producer ended without a resumable completion.",
            **({"path": "/inputs/subject/checkpoint"} if selection is not None else {}),
        }, rejections

    def _record_completion_rejections(
        self, job: sqlite3.Row, anchor: str, rejections: list[_CompletionRejection]
    ) -> None:
        for rejection in rejections:
            existing = self.db.execute(
                """SELECT 1 FROM events WHERE job=? AND kind='dependency_rejected'
                   AND json_extract(data, '$.dependency_job_id')=?
                   AND json_extract(data, '$.candidate_job_id')=?
                   AND json_extract(data, '$.stage')=? LIMIT 1""",
                (job["id"], anchor, rejection.candidate_job_id, rejection.stage),
            ).fetchone()
            if existing is None:
                self.append(
                    job,
                    "dependency_rejected",
                    {
                        "dependency_job_id": anchor,
                        "candidate_job_id": rejection.candidate_job_id,
                        "stage": rejection.stage,
                        "error": {"code": "dependency_corrupt", "message": rejection.message},
                    },
                )

    def resolve_completions(self) -> None:
        with self.db:
            rows = self.db.execute(
                """SELECT e.job, e.parent FROM edges e JOIN jobs j ON j.id=e.job
                   WHERE e.authenticated_completion AND NOT e.satisfied AND j.state='queued'
                   ORDER BY j.seq, e.parent"""
            ).fetchall()
            candidates = {}
            for edge in rows:
                job = self.get(edge["job"])
                if job["state"] != "queued":
                    continue
                anchor = edge["parent"]
                launch = json.loads(job["launch"])
                selection = requested_checkpoint_selection(job["request"], launch)
                try:
                    train_slots = train_source_slots(launch)
                except CompletionFileError as error:
                    self.finished(job, {
                        "outcome": "blocked",
                        "error": {"code": "invalid_request", "message": str(error)},
                    })
                    continue
                key = anchor, selection, train_slots
                if key not in candidates:
                    candidates[key] = self._completion_candidate(anchor, selection, train_slots)
                binding, error, rejections = candidates[key]
                self._record_completion_rejections(job, anchor, rejections)
                if error is not None:
                    self.finished(job, {"outcome": "blocked", "error": error})
                elif binding is not None:
                    self.db.execute(
                        """UPDATE edges SET satisfied=1, binding=?
                           WHERE job=? AND parent=? AND authenticated_completion AND NOT satisfied""",
                        (dump(binding), edge["job"], anchor),
                    )
                    self.append(job, "dependency_bound", binding)

    def completion_bindings(self, job_id: str) -> list[dict]:
        return [
            json.loads(row[0])
            for row in self.db.execute(
                """SELECT binding FROM edges WHERE job=? AND authenticated_completion AND satisfied
                   ORDER BY parent""",
                (job_id,),
            )
        ]

    def settle_blocked(self) -> None:
        with self.db:
            rows = self.db.execute(
                f"""WITH RECURSIVE blocked(id) AS (
                        SELECT e.job FROM edges e JOIN jobs parent ON parent.id=e.parent
                        WHERE NOT e.satisfied AND NOT e.authenticated_completion AND parent.state IN {TERMINAL}
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
