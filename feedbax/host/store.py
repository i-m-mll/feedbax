"""Transactions for the host's jobs, submissions and durable event cursor."""

import errno
import json
import re
import sqlite3
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, NamedTuple
from uuid import uuid4

from feedbax.checkpoint_coordinates import finite_coordinates
from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.effective_success import (
    EmptyInputExclusion, ExcludedProduct, LightOrigins, MetricExclusion, RequestObject, TerminalTrainSubject,
    excluded_inputs,
)
from feedbax.contracts.execution_target import SessionTarget, attempt_target
from feedbax.contracts.live_evaluate import LIVE_EVALUATE_POINT
from feedbax.contracts.request_specs import DocumentTrainingRequestSpec
from feedbax.contracts.request_refs import BestRank, ConditionalProduct, ContentRef, LastMember, ProductRef
from feedbax.contracts.validation_placement import ValidationPlacement
from feedbax.host import live_evaluate
from feedbax.persistence.best_selection import BestMetricUnavailable, completion_score, ranked_requests
from feedbax.persistence.checkpoint_selection import (
    CheckpointSelection, is_checkpoint_producer, requested_checkpoint_selection, selector_sha256, verify_selected_publication,
    verify_dependency_files,
)
from feedbax.persistence.completion import CompletionFileError, train_source_slots, verify_completion_files
from feedbax.host.deferred_reuse import DEFERRED_REUSE_SCHEMA, DeferredAdmissions
from feedbax.host.plan_history import advance_plan, plan_ref_key
from feedbax.host.protocol import EVALUATE_ENTRYPOINT, LIGHT_ENTRYPOINT, LIVE_STATES, TRAINING_ENTRYPOINT, TERMINAL_STATES, HostError, can_resume_attempt
from feedbax.persistence.store_damage import inspect_without_recording


class StorePublicationPending(Exception):
    """The accepted worker terminal event can be retried without recomputing."""


_STUDIO_CONTINUATION = """EXISTS (
    SELECT 1 FROM jobs child
    WHERE child.continues=jobs.id AND json_type(child.data, '$.run_key') IS NOT NULL
)"""


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
TERMINAL = "('succeeded','reused','stopped','failed','interrupted','blocked','not_chosen')"

JOB_ID = re.compile(r"[a-zA-Z0-9_-]{1,128}")
TRAIN_REQUEST_ID = re.compile(r"train:[0-9a-f]{64}")
JSON_COLUMNS = ("data", "phase", "progress", "checkpoint", "outcome")


class _CompletionRejection(NamedTuple):
    candidate_job_id: str
    stage: Literal["continuation", "completion", "checkpoint_selection"]
    message: str


class _CompletionCandidate(NamedTuple):
    binding: dict | None
    error: dict | None
    rejections: list[_CompletionRejection]
    pending: dict | None = None
    observation: dict | None = None


def dump(value, *, sort_keys: bool = False) -> str:
    return json.dumps(value, sort_keys=sort_keys, separators=(",", ":"), allow_nan=False)


def _canonical_binding_conflicts(
    launch: dict, stored_launch: str, *, evaluate: bool, live: bool = False,
) -> bool:
    recorded = json.loads(stored_launch)
    if evaluate or LIGHT_ENTRYPOINT in (launch.get("warm_entrypoint"), recorded.get("warm_entrypoint")):
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


def _scheduled_checkpoint(update: int, spec: DocumentTrainingRequestSpec) -> bool:
    origin = spec.continuation.source_completed_updates if spec.continuation else 0
    return (
        origin < update <= spec.target_total
        and (update == spec.target_total or (update - origin) % spec.training_config.snapshot_interval == 0)
    )


def _pending_checkpoint(attempt: sqlite3.Row, boundary: dict | None, absent: Sequence[int]) -> dict:
    state = attempt["state"]
    recovering = state not in LIVE_STATES
    recovery_update = boundary["completed_updates"] if boundary is not None else None
    published = recovery_update or 0
    completed = published
    if not recovering:
        progress = json.loads(attempt["progress"] or "{}")
        latest = json.loads(attempt["checkpoint"] or "{}")
        published = max(published, latest.get("completed_updates", 0))
        completed = max(published, progress.get("completed_updates", 0))
    payload = json.loads(attempt["launch"])["payload"]
    spec = DocumentTrainingRequestSpec.model_validate(payload["spec"])
    catalogue = set(payload.get("checkpoint_catalog", ()))
    next_update = min((
        update for update in absent
        if update > published and update >= completed
        and (update in catalogue or _scheduled_checkpoint(update, spec))
    ), default=None)
    return {
        "producer_job_id": attempt["id"],
        "producer_request_id": attempt["request"],
        "state": "resumable" if recovering else state,
        "next_update": next_update,
        "recovery_update": recovery_update if recovering else None,
    }


class Store:
    def __init__(self, path: Path):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA + live_evaluate.SCHEMA + DEFERRED_REUSE_SCHEMA)
        self.deferred = DeferredAdmissions(self)
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
        updated = self.db.execute(
            """UPDATE jobs SET state=?, outcome=?, checkpoint=coalesce(checkpoint, ?)
               WHERE id=? AND outcome IS NULL""",
            (
                outcome["outcome"],
                dump(outcome),
                dump(outcome["checkpoint"]) if outcome.get("checkpoint") is not None else None,
                row["id"],
            ),
        )
        if not updated.rowcount:
            return
        reused = outcome["outcome"] == "reused"
        if not reused:
            self.deferred.discard(row["id"])
        self.append(row, "reused" if reused else "terminal", outcome)
        if outcome["outcome"] in {"succeeded", "reused"}:
            self.db.execute(
                "UPDATE edges SET satisfied=1 WHERE parent=? AND publication='' AND NOT authenticated_completion",
                (row["id"],),
            )

        elif outcome["outcome"] == "not_chosen":
            self.db.execute(
                """UPDATE edges SET satisfied=1 WHERE parent=? AND publication='' AND NOT authenticated_completion
                   AND job IN (SELECT id FROM jobs WHERE json_extract(launch,'$.warm_entrypoint')=?)""",
                (row["id"], LIGHT_ENTRYPOINT),
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
        elif kind == LIVE_EVALUATE_POINT:
            self._ingest_evaluation_point(row, data)
            return
        elif kind == "live_evaluate_error":
            self.evaluation_error(row, require_text(data.get("message"), "evaluation error"))
            return
        elif kind == "result" and json.loads(row["launch"])["warm_entrypoint"] == LIGHT_ENTRYPOINT:
            # The launcher records the independent terminal status. Draft products
            # remain in the worker log until Store publication accepts them.
            return
        elif kind == "terminal":
            if row["outcome"] is not None:
                return
            outcome = dict(data)
            outcome.pop("store_success", None)
            outcome.pop("store_job_entry", None)
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
            elif row["stop_mode"] == "interrupt" and outcome["outcome"] == "interrupted":
                outcome["error"] = {"code": "service_stopped", "message": "The background service stopped."}
            if outcome["outcome"] == "succeeded":
                try:
                    if self._is_managed_launch(row, TRAINING_ENTRYPOINT):
                        outcome = self._publish_train_result(row, outcome)
                    elif self._is_managed_launch(row, EVALUATE_ENTRYPOINT):
                        outcome = self._publish_evaluate_result(row, outcome)
                    elif json.loads(row["launch"])["warm_entrypoint"] == LIGHT_ENTRYPOINT:
                        outcome = self._publish_light_result(row, outcome)
                except (ValueError, KeyError, TypeError, OSError) as error:
                    if isinstance(error, OSError) and error.errno in {
                        errno.ENOSPC, errno.EDQUOT, errno.EIO, errno.EBUSY, errno.EINTR,
                    }:
                        raise StorePublicationPending(str(error)) from error
                    outcome = {
                        "outcome": "failed", "receipt": outcome.get("receipt"),
                        "error": {"code": "publication_failed", "message": str(error)},
                    }
            if json.loads(row["launch"])["warm_entrypoint"] == LIGHT_ENTRYPOINT:
                outcome.pop("products", None)
            outcome.pop("execution_evidence", None)
            if staged := outcome.pop("stopped_checkpoints", None):
                if outcome["outcome"] != "stopped" or not self._is_managed_launch(row, TRAINING_ENTRYPOINT):
                    raise ValueError("Only a managed Train checkpoint stop may publish a job entry")
                try:
                    ref = self._publish_stopped_train(row, outcome, staged)
                    outcome["store_job_entry"] = ref.model_dump(mode="json")
                except (OSError, ValueError, KeyError, TypeError) as error:
                    if isinstance(error, OSError) and error.errno in {
                        errno.ENOSPC, errno.EDQUOT, errno.EIO, errno.EBUSY, errno.EINTR,
                    }:
                        raise StorePublicationPending(str(error)) from error
                    outcome["store_publication_error"] = {"code": "publication_failed", "message": str(error)}
            self.finished(row, outcome)
            return
        self.append(row, kind, data)

    def _publish_train_result(self, row: sqlite3.Row, outcome: dict) -> dict:
        from feedbax.host.train_success import publish_train_success
        from feedbax.persistence.train_job import read_train_job

        payload = json.loads(row["launch"])["payload"]
        root = Path(payload["manifest_root"])
        self._check_completion_identity(row, outcome)
        collected = read_train_job(self._success_view(root).store, payload["collected_resume"]) if payload.get("collected_resume") else None
        predecessor = self._train_predecessor(row, payload) if collected is None else None
        ref = publish_train_success(
            root, row["id"], row["request"], payload, outcome, predecessor, collected=collected,
            target=attempt_target(json.loads(row["data"]).get("target")),
            execution_evidence=self._execution_evidence(outcome),
        )
        return outcome | {"store_success": ref}

    def _train_predecessor(self, row: sqlite3.Row, payload: dict) -> tuple[dict, dict] | None:
        binding = payload["canonical_request"]
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
            predecessor = pinned, prior
        return predecessor

    def _publish_stopped_train(
        self, row: sqlite3.Row, outcome: dict, staged: dict,
    ) -> ContentObjectRef:
        from pydantic import TypeAdapter

        from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
        from feedbax.host.train_success import bind_train_inputs
        from feedbax.persistence.train_job import TrainJobCheckpoint, TrainJobEntry, publish_train_job, read_train_job

        payload = json.loads(row["launch"])["payload"]
        binding = payload["canonical_request"]
        view = self._success_view(payload["manifest_root"])
        request = RequestObject.model_validate({
            "id": row["request"], "object": binding["request"], "documents": binding["documents"],
        })
        stopped = PublishedTrainingCheckpoint.model_validate(outcome["checkpoint"])
        ref = ContentObjectRef.model_validate(staged)
        checkpoints = TypeAdapter(tuple[TrainJobCheckpoint, ...]).validate_json(view.store.get_object(ref))
        if not checkpoints or checkpoints[-1].pointer != stopped.pointer or stopped.completed_updates != stopped.pointer.completed_updates:
            raise ValueError("Train Stop differs from its staged recovery publication")
        accepted = {
            PublishedTrainingCheckpoint.model_validate(data["checkpoint"]).pointer.transaction_id: data["checkpoint"]
            for (raw,) in self.db.execute("SELECT data FROM events WHERE job=? AND kind='publication'", (row["id"],))
            if (data := json.loads(raw)).get("checkpoint") is not None
        }
        if {item.pointer.transaction_id for item in checkpoints} != accepted.keys() or any(
            PublishedTrainingCheckpoint.model_validate(accepted[item.pointer.transaction_id]).pointer != item.pointer
            for item in checkpoints
        ):
            raise ValueError("Train staged progress differs from its Host publications")
        if source := payload.get("collected_resume"):
            original = read_train_job(view.store, source)
            if original.request != request:
                raise ValueError("Collected Resume differs from its original Train request")
            inputs = original.inputs
        else:
            inputs = bind_train_inputs(view, request, self._train_predecessor(row, payload))
        from feedbax.contracts.manifest import TrainingRunManifest, canonical_manifest_path, load_manifest_bytes
        from feedbax.host.job_completion import _read_exact, training_metric_file

        receipt = outcome["receipt"]
        root = Path(payload["manifest_root"])
        receipt_path = canonical_manifest_path("TrainingRunManifest", receipt["manifest_id"], root=root)
        manifest = load_manifest_bytes(_read_exact(root, receipt_path, receipt["manifest_sha256"]))
        if (
            not isinstance(manifest, TrainingRunManifest) or manifest.job_id != row["id"]
            or manifest.status != "cancelled" or manifest.completed_batches != stopped.completed_updates
        ):
            raise ValueError("Stop receipt differs from its accepted Train boundary")
        metric_file = training_metric_file(root, row["id"], manifest)
        with view.store._exclusive_lock():
            metrics = None if metric_file is None else view.store._store_file(
                root / metric_file["path"], sha256=metric_file["sha256"], size_bytes=metric_file["size_bytes"],
            )
            return publish_train_job(view.store, TrainJobEntry(
                attempt=row["id"], request=request, inputs=inputs, checkpoints=checkpoints, metrics=metrics,
            ))


    def mirror_collected_train(
        self, job: dict, state: str, receipt: dict | None, progress: dict | None, delivery: int,
        submitted_at_ms: int, ended_at_ms: int,
    ) -> None:
        """Acknowledge terminal Train custody only after it is committed to this Host."""
        from feedbax.persistence.train_job import read_train_job

        self._validate_job(job)
        payload = job["launch"]["payload"]
        data = job["data"]
        root = payload["manifest_root"]
        if (
            state not in {"succeeded", "stopped"} or not TRAIN_REQUEST_ID.fullmatch(job["request"])
            or type(delivery) is not int or delivery < 1 or job["edges"] or job["slot"] != "compute"
            or job["launch"]["warm_entrypoint"] != TRAINING_ENTRYPOINT
            or job["launch"].get("project_root") is None
            or (payload["project_id"], payload["request_id"], root)
            != (job["project"], job["request"], data.get("manifest_root"))
            or data.get("collected_train") is not True
        ):
            raise HostError("invalid_request", "Collected Train differs from its terminal attempt.")
        request = RequestObject.model_validate({
            "id": job["request"], "object": payload["canonical_request"]["request"],
            "documents": payload["canonical_request"]["documents"],
        })
        view = self._success_view(root)
        checkpoint = None
        outcome: dict = {"outcome": state}
        if state == "succeeded":
            ref = ContentObjectRef.model_validate(receipt)
            if view.attempt_ref(job["id"]) != ref:
                raise HostError("invalid_request", "Collected Train success differs from its admitted Store receipt.")
            success = view.read_owned_success(ref).receipt
            if success.request != request or success.target != attempt_target(data.get("target")):
                raise HostError("invalid_request", "Collected Train success differs from its admitted Store receipt.")
            outcome.update(store_success=receipt, receipt={"manifest_id": job["id"], "manifest_sha256": ref.sha256})
        else:
            ref = ContentObjectRef.model_validate(receipt) if receipt is not None else None
            entry = read_train_job(view.store, job["id"], expected=ref)
            if entry.request != request:
                raise HostError("invalid_request", "Collected Train Stop differs from its stored binding.")
            if ref is not None:
                outcome["store_job_entry"] = ref.model_dump(mode="json")
            checkpoint = {"completed_updates": entry.recovery.pointer.completed_updates}
        self._mirror_collected_terminal(
            job, state, outcome, progress, checkpoint, delivery, submitted_at_ms, ended_at_ms,
        )

    def mirror_collected_evaluate(
        self, job: dict, receipt: dict, progress: dict | None, delivery: int,
        submitted_at_ms: int, ended_at_ms: int,
    ) -> None:
        """Record the terminal attempt only after its exact Store receipt is admitted locally."""
        from feedbax.persistence.canonical_evaluate import (
            BoundEvaluationJobInput, decode_rollout_request, evaluation_manifest_id,
        )

        self._validate_job(job)
        payload = BoundEvaluationJobInput.model_validate(job["launch"]["payload"])
        root = str(payload.manifest_root)
        if (
            not job["request"].startswith("evaluate:") or type(delivery) is not int or delivery < 1
            or job["continues"] is not None or job["edges"] or job["slot"] != "compute"
            or job["launch"]["warm_entrypoint"] != EVALUATE_ENTRYPOINT
            or (payload.project_id, payload.request_id, root)
            != (job["project"], job["request"], job["data"].get("manifest_root"))
            or job["data"].get("collected_evaluate") is not True
        ):
            raise HostError("invalid_request", "Collected Evaluate differs from its terminal attempt.")
        ref = ContentObjectRef.model_validate(receipt)
        view = self._success_view(root)
        if view.attempt_ref(job["id"]) != ref:
            raise HostError("invalid_request", "Collected Evaluate differs from its admitted Store receipt.")
        success = view.read_owned_success(ref).receipt
        manifest = success.evaluation_manifest
        if (
            success.attempt != job["id"] or success.request.id != job["request"]
            or success.request.object != payload.canonical_request.request or manifest is None
            or success.target != attempt_target(job["data"].get("target"))
        ):
            raise HostError("invalid_request", "Collected Evaluate differs from its admitted Store receipt.")
        _, _, subject = decode_rollout_request(job["request"], view.store.get_object(success.request.object))
        if subject != payload.canonical_request.subject:
            raise HostError("invalid_request", "Collected Evaluate changed its canonical subject.")
        outcome = {
            "outcome": "succeeded",
            "store_success": receipt,
            "receipt": {"manifest_id": evaluation_manifest_id(job["id"]), "manifest_sha256": manifest.sha256},
        }
        self._mirror_collected_terminal(
            job, "succeeded", outcome, progress, None, delivery, submitted_at_ms, ended_at_ms,
        )

    def _mirror_collected_terminal(
        self, job: dict, state: str, outcome: dict, progress: dict | None, checkpoint: dict | None, delivery: int,
        submitted_at_ms: int, ended_at_ms: int,
    ) -> None:
        if any(type(value) is not int or value < 0 for value in (submitted_at_ms, ended_at_ms)):
            raise HostError("invalid_request", "Collected attempt times must be Unix epoch milliseconds.")
        data = job["data"]
        with self.db:
            if (previous := self.get(job["id"])) is not None:
                if (
                    previous["project"], previous["request"], previous["continues"], previous["state"],
                    json.loads(previous["outcome"] or "null"),
                    json.loads(previous["data"]), json.loads(previous["launch"]),
                    json.loads(previous["progress"] or "null"), json.loads(previous["checkpoint"] or "null"),
                    previous["delivery"],
                ) != (
                    job["project"], job["request"], job["continues"], state, outcome,
                    data, job["launch"], progress, checkpoint, delivery,
                ):
                    raise HostError("job_id_conflict", "Collected attempt conflicts with the existing job.")
                return
            if job["continues"] is not None:
                previous = self.get(job["continues"])
                if previous is None or (previous["project"], previous["request"]) != (job["project"], job["request"]):
                    raise HostError("invalid_resume", "Collect the exact continued Train attempt first.")
            self.db.execute(
                """INSERT INTO jobs(id, project, request, launch, slot, continues, data,
                   state, progress, checkpoint, outcome, delivery) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (job["id"], job["project"], job["request"], dump(job["launch"]), "compute", job["continues"], dump(data),
                 state, dump(progress) if progress is not None else None,
                 dump(checkpoint) if checkpoint is not None else None, dump(outcome), delivery),
            )
            recorded = self.get(job["id"])
            assert recorded is not None
            self.append(recorded, "terminal", outcome | {"venue_times": {
                "submitted_at_ms": submitted_at_ms, "ended_at_ms": ended_at_ms,
            }})

    def _is_managed_launch(self, row: sqlite3.Row, entrypoint: str) -> bool:
        launch = json.loads(row["launch"])
        return (
            launch["warm_entrypoint"] == entrypoint
            and launch["payload"].get("canonical_request") is not None
        )

    def _publish_evaluate_result(self, row: sqlite3.Row, outcome: dict) -> dict:
        from feedbax.host.evaluate_success import publish_evaluate_success

        payload = json.loads(row["launch"])["payload"]
        terminal = payload["canonical_request"].get("terminal")
        if payload["canonical_request"].get("initial_model") is not None:
            binding = {}
        elif terminal is None:
            (binding,) = self.completion_bindings(row["id"])
        else:
            binding = {"terminal": terminal}
        ref = publish_evaluate_success(
            attempt=row["id"], request_id=row["request"], project=row["project"],
            payload=payload, receipt=outcome["receipt"], binding=binding,
            target=attempt_target(json.loads(row["data"]).get("target")),
            execution_evidence=self._execution_evidence(outcome),
        )
        return outcome | {"store_success": ref.model_dump(mode="json")}

    def _publish_light_result(self, row: sqlite3.Row, outcome: dict) -> dict:
        from feedbax.host.light_jobs import publish_light_success

        if not {"outcome", "products"} <= outcome.keys() <= {"outcome", "products", "provider", "execution_evidence"}:
            raise ValueError("Managed success must supply draft products and optional provider provenance.")
        payload = json.loads(row["launch"])["payload"]
        ref = publish_light_success(
            payload, row["id"], row["request"], self.light_origins(row), outcome["products"],
            checkpoint_bindings=tuple(self.completion_bindings(row["id"])),
            provider=outcome.get("provider"),
            target=attempt_target(json.loads(row["data"]).get("target")),
            execution_evidence=self._execution_evidence(outcome),
        )
        return {
            "outcome": "succeeded",
            "receipt": {"manifest_id": row["id"], "manifest_sha256": ref["sha256"]},
            "store_success": ref,
        }

    @staticmethod
    def _execution_evidence(outcome: dict) -> ContentObjectRef:
        if "execution_evidence" not in outcome:
            raise ValueError("The managed worker did not publish its execution evidence")
        return ContentObjectRef.model_validate(outcome["execution_evidence"])

    @staticmethod
    def _light_request(launch: dict, project: str, request_id: str):
        payload = launch["payload"]
        root = Path(payload["manifest_root"])
        request = RequestObject.model_validate(payload["request"])
        if (
            payload["project_id"] != project or request.id != request_id
            or not root.is_absolute() or root.resolve() != root
        ):
            raise ValueError("Light launch differs from its admitted project, Store or request.")
        decoded = Store._success_view(root).read_request(request)
        if decoded.kind not in ("analyze", "render", "report"):
            raise ValueError("Managed light jobs require an Analyze, Render or Report request.")
        return decoded

    @staticmethod
    def _light_requests(decoded) -> frozenset[str]:
        from feedbax.contracts.effective_success import exact_model_checkpoint_input

        inputs = [member.reference for members in decoded.inputs.values() for member in members]
        if any(not isinstance(ref, (ProductRef, ContentRef)) for ref in inputs):
            raise ValueError("Light jobs require exact Store inputs or terminal products.")
        exact = exact_model_checkpoint_input(decoded)
        return frozenset(
            ref.request_id for ref in inputs
            if isinstance(ref, ProductRef) and (exact is None or ref != exact[2])
        )

    def _light_parents(self, project: str, root: str, expected: frozenset[str], parents: Sequence[str]) -> list[sqlite3.Row]:
        rows = []
        seen = set()
        for parent_id in parents:
            source = self.get(parent_id)
            if source is None or source["project"] != project or source["request"] in seen:
                raise ValueError("Light dependency is missing, repeated or outside the admitted project.")
            payload = json.loads(source["launch"])["payload"]
            if payload.get("manifest_root") != root or source["request"] not in expected:
                raise ValueError("Light dependency differs from its canonical producer or Store.")
            seen.add(source["request"])
            rows.append(source)
        if seen != expected:
            raise ValueError("Light dependencies do not cover exactly the canonical producer requests.")
        return rows

    @staticmethod
    def _success_view(root: str | Path):
        from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
        from feedbax.persistence.effective_success import SuccessView

        return SuccessView(ImmutableArtifactBlobProvider(root))

    @staticmethod
    def _light_receipt(view, source: sqlite3.Row) -> dict:
        from feedbax.contracts.artifact_custody import ContentObjectRef

        if source["state"] != "succeeded":
            raise ValueError("The exact light producer did not succeed.")
        outcome = json.loads(source["outcome"])
        ref = ContentObjectRef.model_validate(outcome.get("store_success"))
        receipt = view.read_receipt(ref)
        if receipt.attempt != source["id"] or receipt.request.id != source["request"]:
            raise ValueError("Accepted origin differs from its exact Store receipt.")
        return ref.model_dump(mode="json")

    def light_origins(self, row: sqlite3.Row) -> dict:
        launch = json.loads(row["launch"])
        payload = launch["payload"]
        root = payload["manifest_root"]
        try:
            expected = self._light_requests(self._light_request(launch, row["project"], row["request"]))
            parents = tuple(edge["parent"] for edge in self.db.execute(
                "SELECT parent FROM edges WHERE job=? AND NOT authenticated_completion ORDER BY parent", (row["id"],)
            ))
            sources = self._light_parents(row["project"], root, expected, parents)
            view = self._success_view(root)
            origins = {}
            for source in sources:
                if source["state"] == "not_chosen":
                    origins[source["request"]] = json.loads(source["outcome"])["excluded_product"]
                else:
                    origins[source["request"]] = self._light_receipt(view, self.deferred.exact_source(source))
            return origins
        except (ValueError, TypeError, KeyError, OSError) as error:
            raise CompletionFileError(str(error)) from error

    def exclude_empty_light(self, row: sqlite3.Row, origins: dict) -> bool:
        """Propagate an authenticated absence when a required light input has no chosen member."""
        if not any("cause" in origin for origin in origins.values()):
            return False
        from pydantic import TypeAdapter

        from feedbax.persistence.metric_exclusion import exclusion_objects

        launch = json.loads(row["launch"])
        payload = launch["payload"]
        try:
            decoded = self._light_request(launch, row["project"], row["request"])
            excluded = excluded_inputs(decoded, TypeAdapter(LightOrigins).validate_python(origins))
            for role, proofs in excluded.items():
                members = decoded.inputs[role]
                if len(members) != len(proofs):
                    continue
                request = RequestObject.model_validate(payload["request"])
                proof = ExcludedProduct(request=request, cause=EmptyInputExclusion(role=role, members=proofs))
                store = self._success_view(payload["manifest_root"]).store
                exclusion_objects(store, proof, request.id)
                with self.db:
                    self.finished(row, {
                        "outcome": "not_chosen", "reason": f"No chosen producer supplies input {role!r}.",
                        "excluded_product": proof.model_dump(mode="json"),
                    })
                return True
        except (ValueError, KeyError, OSError, TypeError) as error:
            raise CompletionFileError(str(error)) from error
        return False

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
        if (source_root := launch.get("project_root")) is not None and (
            not isinstance(source_root, str) or not Path(source_root).is_absolute() or "\0" in source_root
        ):
            raise HostError("invalid_request", "Managed project root must be an absolute path.")
        if launch["warm_entrypoint"] == LIGHT_ENTRYPOINT and (
            job["continues"] is not None
            or any(edge["publication"] is not None for edge in job["edges"])
        ):
            raise HostError("invalid_request", "Managed light jobs require singleton launches and terminal edges.")

    def admit(
        self, token: str, items: list[dict], record: dict, *, allow_completion: bool = False,
        session_id: str | None = None,
    ) -> dict:
        require_text(token, "submission token")
        record = dict(record)
        deferred_projection = record.pop("deferred_projection", None)
        submitted_target = record.get("target")
        try:
            target = attempt_target(submitted_target)
            if submitted_target is not None:
                venue = str(target.session_id) if isinstance(target, SessionTarget) else None
                if venue != session_id:
                    raise ValueError("Execution target differs from the authenticated Host session.")
                if isinstance(target, SessionTarget) and target.realization is None:
                    raise ValueError("Venue admission requires realized execution target evidence.")
        except (TypeError, ValueError) as error:
            raise HostError("invalid_request", str(error)) from error
        with self.db:
            previous = self.db.execute(
                "SELECT response FROM submissions WHERE token=?", (token,)
            ).fetchone()
            if previous:
                saved = json.loads(previous[0])
                if attempt_target(saved["record"].get("target")) != target:
                    raise HostError("invalid_request", "Submission token belongs to a different execution target.")
                return saved
            results, aliases, inserted = [], {}, []
            considered = set()
            pending_parents: dict[str, tuple[tuple[str, bool], ...]] = {}
            live_drivers: set[str] = set()
            projects = set(record.get("projects", ()))
            if record.get("project"):
                projects.add(record["project"])
            for item in items:
                job = item["job"]
                run_key = require_text(item["run_key"], "run key")
                if job is None:
                    disposition = item["disposition"]
                    if disposition == "reused":
                        if item.get("terminal_reuse") is not None and not allow_completion:
                            raise HostError("invalid_request", "Terminal Evaluate reuse requires authenticated admission.")
                        self._recheck_reuse(item, record)
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
                deferred_reuse = item.get("deferred_reuse")
                force_execution = item.get("force_execution", False)
                if type(force_execution) is not bool or (force_execution and deferred_reuse is not None):
                    raise HostError("invalid_request", "Invalid deferred admission policy.")
                if job["data"].get("target") != submitted_target:
                    raise HostError("invalid_request", "Job execution target differs from its submission.")
                job["data"].pop("placement", None)
                job["data"].pop("placement_detail", None)
                payload = job["launch"]["payload"]
                canonical_request = payload.get("canonical_request")
                light = job["launch"]["warm_entrypoint"] == LIGHT_ENTRYPOINT
                refused_parents = [
                    edge["job_id"] for edge in job["edges"]
                    if edge["job_id"] in considered and edge["job_id"] not in aliases
                ]
                considered.add(job["id"])
                if light and refused_parents:
                    results.append({
                        "run_key": run_key, "job_id": None, "disposition": "rejected",
                        "reason": f"The exact upstream attempt {refused_parents[0]} was rejected.",
                    })
                    continue
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
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    raise HostError("invalid_request", str(exc)) from exc
                checkpoint = canonical_request.get("checkpoint") if canonical_request is not None else None
                if light:
                    if selected is not None and selected.producer_request is not None:
                        checkpoint = {"request_id": selected.producer_request}
                        input_path = f"/inputs/{selected.role}"
                    elif any(edge.get("authenticated_completion") for edge in job["edges"]):
                        raise HostError("invalid_request", "Managed operations require an authored exact checkpoint edge.")
                checkpoint_parents = None
                best = isinstance(checkpoint, dict) and checkpoint.get("kind") == "best_rank"
                terminal = canonical_request.get("terminal") if evaluate else None
                initial = canonical_request.get("initial_model") if evaluate else None
                if initial is not None:
                    from feedbax.contracts.model_input import read_document_input
                    from feedbax.persistence.canonical_evaluate import (
                        BoundEvaluationJobInput, decode_rollout_request, initial_evaluation_references,
                        verified_imported_subject,
                    )

                    try:
                        launch = BoundEvaluationJobInput.model_validate(payload)
                        if (
                            job["continues"] is not None or job["edges"]
                            or job["launch"]["warm_entrypoint"] != EVALUATE_ENTRYPOINT
                        ):
                            raise ValueError("Initial Evaluate requires a standalone rollout launch.")
                        view = self._success_view(str(launch.manifest_root))
                        request = RequestObject(id=launch.request_id, object=launch.canonical_request.request, documents=())
                        _, _, subject = decode_rollout_request(launch.request_id, view.store.get_object(request.object))
                        if subject != launch.canonical_request.subject:
                            raise ValueError("Initial Evaluate differs from its canonical request.")
                        for content in initial_evaluation_references(view.read_request(request)).values():
                            read_document_input(content, view.store)
                        verified_imported_subject(view.store, launch.canonical_request.initial_model, verify_bytes=True)
                    except (OSError, TypeError, ValueError, KeyError) as error:
                        results.append({
                            "run_key": run_key, "job_id": None, "disposition": "rejected",
                            "reason": f"{input_path}: {error}",
                        })
                        continue
                elif terminal is not None:
                    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
                    from feedbax.persistence.evaluation_subject import read_terminal_train_subject

                    try:
                        subject = ProductRef.model_validate(checkpoint)
                        if (
                            not allow_completion or job["continues"] is not None or job["edges"]
                            or subject.role != "checkpoint" or not isinstance(subject.select, LastMember)
                            or job["launch"]["warm_entrypoint"] != EVALUATE_ENTRYPOINT
                        ):
                            raise ValueError("Store terminal subject requires one pinned last checkpoint.")
                        pin = TerminalTrainSubject.model_validate(terminal)
                        read_terminal_train_subject(
                            ImmutableArtifactBlobProvider(Path(payload["manifest_root"])), pin, subject.request_id,
                        )
                    except (OSError, TypeError, ValueError, KeyError) as error:
                        results.append({
                            "run_key": run_key, "job_id": None, "disposition": "rejected",
                            "reason": f"{input_path}: {error}",
                        })
                        continue
                elif best:
                    if not evaluate or job["continues"] is not None:
                        raise HostError("invalid_request", f"{input_path}: Best requires an Evaluate subject.")
                    checkpoint_parents = self._best_parents(job, checkpoint, aliases)
                    if checkpoint_parents is None:
                        results.append({
                            "run_key": run_key, "job_id": None, "disposition": "rejected",
                            "reason": f"{input_path}: a Best candidate was not admitted",
                        })
                        continue
                elif evaluate and (
                    checkpoint is None or job["continues"] is not None or len(job["edges"]) != 1
                ):
                    raise HostError("invalid_request", f"{input_path}: Evaluate requires a Train subject.")
                if terminal is None and not best and checkpoint is not None and job["continues"] is None:
                    edges = [edge for edge in job["edges"] if edge.get("authenticated_completion")]
                    if len(edges) != 1 or not isinstance(checkpoint, dict):
                        raise HostError("invalid_request", f"{input_path}: requires one authenticated checkpoint edge.")
                    parent_id = edges[0]["job_id"]
                    checkpoint_parent = aliases.get(parent_id, parent_id)
                    checkpoint_parents = frozenset({checkpoint_parent})
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
                continues = job["continues"]
                if (
                    continues is not None
                    and "run_key" in job["data"]
                    and continues == record.get("continues")
                ):
                    continued = self.db.execute(
                        f"SELECT 1 FROM jobs WHERE id=? AND {_STUDIO_CONTINUATION}",
                        (continues,),
                    ).fetchone()
                    if continued:
                        raise HostError("invalid_resume", "This run already has a continuation.")
                live = self.db.execute(
                    f"SELECT id, continues, launch, data FROM jobs WHERE project=? AND request=? AND state IN {LIVE}",
                    (job["project"], job["request"]),
                ).fetchone()
                matching = by_id if by_id is not None else live
                different_target = matching is not None and (
                    attempt_target(json.loads(matching["data"]).get("target")) != target
                )
                light_parents = tuple(sorted(
                    aliases.get(edge["job_id"], edge["job_id"]) for edge in job["edges"]
                )) if light else ()
                different_light_parents = (
                    light and matching is not None
                    and self._parents(matching["id"], pending_parents) != light_parents
                )
                if by_id is not None:
                    if (by_id["project"], by_id["request"]) != (job["project"], job["request"]):
                        raise HostError(
                            "job_id_conflict", "Job id already names a different request."
                        )
                    if different_target or different_light_parents or _canonical_binding_conflicts(job["launch"], by_id["launch"], evaluate=evaluate) or (
                        checkpoint_parents is not None
                        and frozenset(self._parents(by_id["id"], pending_parents, completion=True)) != checkpoint_parents
                    ):
                        raise HostError(
                            "job_id_conflict", "Job id already names a different canonical binding."
                        )
                    job_id, disposition = by_id["id"], "admitted"
                elif live is not None:
                    if different_target or different_light_parents or _canonical_binding_conflicts(
                        job["launch"], live["launch"], evaluate=evaluate, live=True,
                    ) or (
                        checkpoint_parents is not None
                        and frozenset(self._parents(live["id"], pending_parents, completion=True)) != checkpoint_parents
                    ):
                        results.append({
                            "run_key": run_key,
                            "job_id": None,
                            "disposition": "rejected",
                            "reason": (
                                "The live light request has different exact inputs."
                                if light else
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
                    if force_execution:
                        self.deferred.execute(self.get(job_id), occupied=live_drivers)
                else:
                    self._require_admission_current(job, item["checked_through"])
                    placement = None
                    if deferred_reuse is None and evaluate and selected is not None and selected.allows_gaps:
                        placement = self._place_evaluation(job, source, selected, live_drivers)
                        job["data"].update(placement=placement["location"], placement_detail=placement)
                    job_id, disposition = job["id"], "admitted"
                    row = self.insert_queued(job)
                    if deferred_reuse is not None:
                        self.deferred.reserve(row, job, deferred_reuse, deferred_projection)
                    inserted.append((job, item.get("light_definition")))
                    pending_parents[job_id] = tuple(
                        (aliases.get(edge["job_id"], edge["job_id"]), edge.get("authenticated_completion", False))
                        for edge in job["edges"]
                    )
                    if placement is not None:
                        self.append(row, live_evaluate.PLACEMENT_EVENT, placement)
                aliases[job["id"]] = job_id
                results.append(
                    {
                        "run_key": run_key,
                        "job_id": job_id,
                        "disposition": disposition,
                        "reason": None,
                    }
                )
            for job, definition in inserted:
                self._bind_edges(job, aliases, definition)
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

    def insert_queued(self, job: dict) -> sqlite3.Row:
        self.db.execute(
            "INSERT INTO jobs(id, project, request, launch, slot, data) VALUES (?,?,?,?,?,?)",
            (job["id"], job["project"], job["request"], dump(job["launch"]), job["slot"], dump(job["data"])),
        )
        row = self.get(job["id"])
        assert row is not None
        self.append(row, "queued", {})
        return row

    def _recheck_reuse(self, item: dict, record: dict) -> None:
        cursor = item.get("checked_through")
        if cursor is None:
            return
        run = record.get("runs", {}).get(item["run_key"], {})
        reused = run.get("reused_job_id")
        row = self.get(reused) if isinstance(reused, str) else None
        if row is None:
            raise HostError("admission_changed", "The reused result is unavailable.")
        launch = json.loads(row["launch"])
        root = launch["payload"].get("manifest_root")
        if (
            row["state"] != "succeeded" or row["project"] != record.get("project")
            or row["request"] != run.get("request_id") or root != (record.get("command") or {}).get("store")
        ):
            raise HostError("invalid_request", "Reused result differs from its admitted run.")
        if launch["warm_entrypoint"] == LIGHT_ENTRYPOINT:
            try:
                self._light_receipt(self._success_view(root), row)
            except (ValueError, TypeError, KeyError, OSError) as error:
                raise HostError("admission_changed", str(error)) from error
        if (terminal := item.get("terminal_reuse")) is not None:
            self._recheck_terminal_reuse(row, root, terminal)
        if row["request"].startswith(("analyze:", "evaluate:")):
            from feedbax.host.current_checkpoints import CurrentCheckpoints

            view = self._success_view(root)
            if (ref := view.attempt_ref(row["id"])) is not None:
                difference = CurrentCheckpoints(self, view, row["project"]).compare(view.read_receipt(ref), launch)
                if difference is not None:
                    raise HostError("admission_changed", difference.detail)
        self._require_admission_current(row, cursor)

    def _recheck_terminal_reuse(self, row: sqlite3.Row, root: str, expected: dict) -> None:
        from feedbax.contracts.effective_success import EvaluationSubjectValue
        from feedbax.persistence.effective_success import terminal_train_subject

        try:
            view = self._success_view(root)
            attempt = view.read_attempt(row["id"])
            if attempt is None:
                raise ValueError("The reused Evaluate receipt is unavailable.")
            (bound,) = attempt.receipt.inputs["subject"]
            source = bound.evaluation_subject
            receipt = ContentObjectRef.model_validate(expected["receipt"])
            if source is None or source.terminal_receipt is None:
                raise ValueError("The reused Evaluate does not bind a terminal Train product.")
            if (anchor := expected["anchor"]) is None:
                pin = terminal_train_subject(view.store, source.producer.id)
            else:
                producer = self.get(anchor)
                if (
                    producer is None or producer["project"] != row["project"]
                    or producer["request"] != source.producer.id
                    or json.loads(producer["data"]).get("manifest_root") != root
                ):
                    raise ValueError("The reused Evaluate names a different Train lineage.")
                pin = self._terminal_train_for_anchor(anchor)
            if pin is None or pin.receipt != receipt:
                raise ValueError("The terminal Train receipt changed during admission.")
            if not EvaluationSubjectValue.model_validate(bound.descriptor).matches_terminal_model(pin.model.content):
                raise ValueError("The reused Evaluate binds a different Train model.")
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise HostError("admission_changed", str(error)) from error

    def _advance_authored_train_plan(self, record: dict, results: list[dict]) -> None:
        plan = record.get("authored_train_plan")
        if plan is None:
            return
        admitted = [
            (item, self.get(item["job_id"])) for item in results if item["job_id"] is not None
        ]
        trains = [(item, job) for item, job in admitted if job["request"].startswith("train:")]
        if not trains:
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
        for item, job in trains:
            payload = json.loads(job["launch"])["payload"]
            if (
                (job["project"], job["request"]) != (project, current.get(item["run_key"]))
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
                   AND json_type(j.outcome, '$.store_success')='object'
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

    def _require_admission_current(self, job: dict | sqlite3.Row, checked_through: int | None) -> None:
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

    def _parents(
        self, job_id: str, pending: Mapping[str, tuple[tuple[str, bool], ...]], *, completion: bool = False,
    ) -> tuple[str, ...]:
        """The sorted parents of a job, or only its authenticated completion parents.

        A job admitted earlier in the same submission has its edges in `pending`: they are bound after every item.
        """
        if (edges := pending.get(job_id)) is not None:
            return tuple(sorted(parent for parent, authenticated in edges if authenticated or not completion))
        return tuple(row[0] for row in self.db.execute(
            "SELECT parent FROM edges WHERE job=?" + (" AND authenticated_completion" if completion else "")
            + " ORDER BY parent",
            (job_id,),
        ))

    def _best_parents(self, job: dict, checkpoint: dict, aliases: dict[str, str]) -> frozenset[str] | None:
        path = "/inputs/subject"
        try:
            rule = BestRank.model_validate(checkpoint)
        except ValueError as error:
            raise HostError("invalid_request", f"{path}: invalid Best subject: {error}") from error
        expected = {candidate.reference.request_id for candidate in rule.candidates}
        root = job["launch"]["payload"].get("manifest_root")
        if not isinstance(root, str) or not Path(root).is_absolute():
            raise HostError("invalid_request", f"{path}: Best requires an explicit project store.")
        edges = job["edges"]
        if len(edges) != len(expected) or any(not edge.get("authenticated_completion") for edge in edges):
            raise HostError("invalid_request", f"{path}: requires one authenticated edge per Best candidate request.")
        roots = set()
        missing = False
        for edge in edges:
            parent_id = aliases.get(edge["job_id"], edge["job_id"])
            parent = self.get(parent_id)
            if parent is None:
                missing = True
                continue
            if (
                parent["project"] != job["project"] or parent["request"] not in expected
                or not TRAIN_REQUEST_ID.fullmatch(parent["request"])
                or not is_checkpoint_producer(
                    json.loads(parent["launch"]), continues=parent["continues"], manifest_root=root,
                    collected=json.loads(parent["data"]).get("collected_train", False),
                )
            ):
                raise HostError("invalid_request", f"{path}: Best candidate edge differs from its canonical input.")
            expected.remove(parent["request"])
            roots.add(parent_id)
        return None if missing else frozenset(roots)

    def _bind_edges(self, job: dict, aliases: dict[str, str], light_definition: dict | None = None) -> None:
        job_id, continues = job["id"], job["continues"]
        payload = job["launch"]["payload"]
        collected = payload.get("collected_resume")
        if collected is not None and (collected != continues or job["launch"]["warm_entrypoint"] != TRAINING_ENTRYPOINT):
            raise HostError("invalid_resume", "Collected recovery must name the continued Train attempt.")
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
            if "run_key" in job["data"]:
                self.append(source, "continued", {})
            if collected is not None:
                from feedbax.persistence.train_job import read_train_job

                try:
                    recovery = read_train_job(self._success_view(payload["manifest_root"]).store, collected)
                    if recovery.request.id != job["request"]:
                        raise ValueError("Collected recovery differs from the original Train request")
                except (OSError, ValueError, KeyError, TypeError) as error:
                    raise HostError("invalid_resume", str(error)) from error
            else:
                self.db.execute(
                    """INSERT INTO edges(job, parent, publication, satisfied, authenticated_completion, binding)
                       SELECT ?, parent, publication, satisfied, authenticated_completion, binding
                       FROM edges WHERE job=? AND authenticated_completion""",
                    (job_id, continues),
                )
        if job["launch"]["warm_entrypoint"] == LIGHT_ENTRYPOINT:
            from feedbax.contracts.operation import OperationDefinition
            from feedbax.host.light_jobs import validate_light_requirements

            payload = job["launch"]["payload"]
            try:
                decoded = self._light_request(job["launch"], job["project"], job["request"])
                definition = OperationDefinition.model_validate(light_definition) if light_definition is not None else None
                validate_light_requirements(decoded, definition)
                numerical = definition.numerical if definition is not None else None
                if job["slot"] != ("compute" if numerical else "light") or (
                    numerical and any(
                        job["launch"]["env"].get(key) != value for key, value in numerical.environment().items()
                    )
                ):
                    raise ValueError("Managed delivery worker does not match the declared numerical requirements")
                self._light_parents(
                    job["project"], payload["manifest_root"], self._light_requests(decoded),
                    tuple(
                        aliases.get(edge["job_id"], edge["job_id"])
                        for edge in job["edges"] if not edge.get("authenticated_completion")
                    ),
                )
            except (ValueError, KeyError, TypeError, OSError) as error:
                raise HostError("invalid_request", str(error)) from error
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
                satisfied = source is not None and (
                    source["state"] in ("succeeded", "reused") or (
                        source["state"] == "not_chosen" and job["launch"]["warm_entrypoint"] == LIGHT_ENTRYPOINT
                    )
                )
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
        """Read job rows and their lifecycle times. Older venue mirrors recorded only local custody time."""
        selection = "AND jobs.id IN (" + ",".join("?" * len(job_ids)) + ")" if job_ids else ""
        rows = self.db.execute(
            f"""SELECT jobs.*, {_STUDIO_CONTINUATION} AS continued,
                       coalesce(
                           CAST(queued.time * 1000 AS INTEGER),
                           json_extract(terminal.data, '$.venue_times.submitted_at_ms'),
                           CAST(terminal.time * 1000 AS INTEGER)
                       ) AS submitted_at_ms,
                       coalesce(
                           CASE WHEN queued.cursor IS NULL
                               THEN json_extract(terminal.data, '$.venue_times.ended_at_ms') END,
                           CAST(terminal.time * 1000 AS INTEGER)
                       ) AS ended_at_ms
                FROM jobs
                LEFT JOIN events queued ON queued.cursor = (
                    SELECT cursor FROM events WHERE job=jobs.id AND kind='queued' ORDER BY cursor LIMIT 1
                )
                LEFT JOIN events terminal ON jobs.state IN {TERMINAL} AND terminal.cursor = (
                    SELECT cursor FROM events WHERE job=jobs.id
                    AND kind=CASE WHEN jobs.state='reused' THEN 'reused' ELSE 'terminal' END
                    ORDER BY cursor LIMIT 1
                )
                WHERE (? IS NULL OR jobs.project=?) AND (? IS NULL OR jobs.request=?)
                AND (? IS NULL OR jobs.id=?) {selection} ORDER BY jobs.seq""",
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
                    "submitted_at_ms",
                    "ended_at_ms",
                )
            }
            view.update(
                {
                    key: json.loads(row[key]) if row[key] is not None else None
                    for key in JSON_COLUMNS
                }
            )
            view["continued"] = bool(row["continued"])
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

    @inspect_without_recording()
    def checkpoint_producer(self, *, project: str, manifest_root: str, request: str) -> str | None:
        from feedbax.host.current_checkpoints import CurrentCheckpoints

        return CurrentCheckpoints(self, self._success_view(manifest_root), project).native_anchor(request)

    @inspect_without_recording()
    def _project_checkpoints(
        self, rows: list[dict], project: str, root: str, requests: list[str], *, include_producers: bool,
    ) -> tuple[dict, dict, dict]:
        from feedbax.host.current_checkpoints import CurrentCheckpoints

        view = self._success_view(root)
        selection = CurrentCheckpoints(self, view, project)
        wanted = set(requests)
        producers = {}
        held = {}
        if include_producers:
            for request in {job["request"] for job in rows if job["request"].startswith("train:")}:
                if (anchor := selection.native_anchor(request)) is not None:
                    producers[request] = anchor
                    if (coordinates := self._finished_checkpoints(anchor)) is not None:
                        held[request] = list(coordinates)
        differences = {}
        for job in rows:
            if (
                job["request"] not in wanted or job["state"] != "succeeded"
                or job["data"].get("manifest_root") != root
                or not job["request"].startswith(("analyze:", "evaluate:"))
                or (ref := view.attempt_ref(job["id"])) is None
            ):
                continue
            try:
                receipt = view.read_receipt(ref)
            except (OSError, ValueError):
                continue
            launch = job.get("launch")
            if launch is None:
                launch = json.loads(self.get(job["id"])["launch"])
            if (difference := selection.compare(receipt, launch)) is not None:
                differences[job["id"]] = difference.model_dump(mode="json")
        return differences, producers, held

    def _finished_checkpoints(self, anchor: str) -> tuple[int, ...] | None:
        """The checkpoint coordinates a Train lineage published on its way to success; None while no attempt of it
        has succeeded or one is still live, since it may yet publish more."""
        lineage = {
            row["id"]: row for row in self.db.execute(
                """WITH RECURSIVE lineage(id) AS (
                       VALUES (?) UNION ALL
                       SELECT child.id FROM jobs child JOIN lineage ON child.continues=lineage.id
                   )
                   SELECT jobs.* FROM jobs JOIN lineage USING (id) ORDER BY jobs.seq""",
                (anchor,),
            )
        }
        if any(row["state"] in LIVE_STATES for row in lineage.values()):
            return None
        tip = next((row for row in reversed(lineage.values()) if row["state"] == "succeeded"), None)
        path = []
        while tip is not None:
            path.append(tip)
            tip = lineage.get(tip["continues"]) if tip["continues"] is not None else None
        if not path:
            return None
        held = set()
        upper = None
        for attempt in path:
            resume = json.loads(attempt["launch"])["payload"].get("resume")
            lower = resume["completed_updates"] if resume is not None else 0
            for (key,) in self.db.execute(
                "SELECT key FROM publications WHERE job=? AND key LIKE 'checkpoint:%'", (attempt["id"],),
            ):
                coordinate = int(key.split(":")[1])
                if lower < coordinate and (upper is None or coordinate <= upper):
                    held.add(coordinate)
            upper = lower
        return tuple(sorted(held))

    def jobs_snapshot(
        self, *, record_keys: list[str] | None = None, checkpoint_progress_root: str | None = None,
        checkpoint_objective_root: str | None = None, terminal_train: tuple[str, list[str]] | None = None,
        checkpoint_admission: tuple[str, list[str]] | None = None,
        attempt_addresses: tuple[str, list[str]] | None = None,
        **filters,
    ) -> dict:
        snapshot = {"cursor": self.event_head, "oldest_cursor": 0, "jobs": self.jobs(**filters)}
        if checkpoint_objective_root is not None and not filters.get("job_id"):
            raise HostError("invalid_request", "A checkpoint objective read needs one Evaluate attempt.")
        root = checkpoint_objective_root or checkpoint_progress_root
        if root is not None and filters.get("include_launch"):
            for job in snapshot["jobs"]:
                if (
                    job["state"] in LIVE_STATES and job["request"].startswith("evaluate:")
                    and job["data"].get("manifest_root") == root
                    and job["data"].get("placement_detail") is not None
                ):
                    self._checkpoint_progress(job, include_objective=checkpoint_objective_root is not None)
        if record_keys:
            snapshot["records"] = self.record_get_many(record_keys)
        if terminal_train is not None:
            root, requests = terminal_train
            requested = frozenset(requests)
            receipts = {}
            for job in snapshot["jobs"]:
                if (
                    job["request"] in requested and job["continues"] is None
                    and job["data"].get("manifest_root") == root and not job["data"].get("collected_train")
                    and (pin := self._terminal_train_for_anchor(job["id"])) is not None
                ):
                    receipts[job["id"]] = pin.receipt.model_dump(mode="json")
            snapshot["terminal_train_receipts"] = receipts
        if checkpoint_admission is not None:
            project = filters.get("project")
            if project is None:
                raise HostError("invalid_request", "Checkpoint admission requires one project.")
            differences, producers, held = self._project_checkpoints(
                snapshot["jobs"], project, *checkpoint_admission,
                include_producers=bool(filters.get("include_launch")),
            )
            snapshot.update(
                checkpoint_differences=differences, checkpoint_producers=producers, finished_checkpoints=held,
            )
        if attempt_addresses is not None:
            if (project := filters.get("project")) is None:
                raise HostError("invalid_request", "Attempt addresses require one project.")
            snapshot["attempt_addresses"] = self._attempt_addresses(project, *attempt_addresses)
        return snapshot

    def _attempt_addresses(self, project: str, root: str, addresses: list[str]) -> dict[str, list[str]]:
        """Read which authored rows were given each exact attempt, including attached aliases."""
        result: dict[str, list[str]] = {}
        for job, address in self.db.execute(
            """
            SELECT DISTINCT j.id, run.key
            FROM submission_projects p
            JOIN submissions s ON s.id=p.submission
            JOIN json_each(s.response, '$.record.runs') run
            JOIN json_each(s.response, '$.items') item ON json_extract(item.value, '$.run_key')=run.key
            JOIN jobs j ON j.id=json_extract(item.value, '$.job_id')
            WHERE p.project=? AND j.project=p.project
              AND json_extract(s.response, '$.record.project')=p.project
              AND json_extract(s.response, '$.record.command.store')=?
              AND json_extract(j.data, '$.manifest_root')=?
              AND j.request=json_extract(run.value, '$.request_id')
              AND run.key IN (SELECT value FROM json_each(?))
            """,
            (project, root, root, dump(addresses)),
        ):
            result.setdefault(job, []).append(address)
        return result

    @inspect_without_recording()
    def _terminal_train_for_anchor(self, anchor: str) -> TerminalTrainSubject | None:
        binding = self._completion_candidate(anchor, terminal_model=True, current=True).binding
        return TerminalTrainSubject.model_validate(binding["terminal"]) if binding is not None else None

    @inspect_without_recording()
    def _checkpoint_progress(self, job: dict, *, include_objective: bool) -> None:
        """Read the adjudicated Train path and its live point progress at the snapshot cursor."""
        edge = next(edge for edge in job["edges"] if edge.get("authenticated_completion"))
        binding = edge["binding"]
        pending = None
        points: Sequence[dict] = ()
        is_live = job["data"]["placement"] == "live"
        observed = None
        if binding is not None:
            points = binding.get("live_evaluation_points", ())
            observed = binding["checkpoint_selection"]
        else:
            launch = job["launch"]
            anchor = edge["job_id"]
            selection = requested_checkpoint_selection(job["request"], launch)
            tip = live_evaluate.pinned_tip(self.db, job["id"]) if is_live else None
            if is_live and tip is None:
                # Before delegation only the admitted root can receive the live owner.
                tip = self.get(anchor)
                if tip["state"] in TERMINAL_STATES:
                    return
            tip_id = tip["id"] if tip is not None else None
            candidate = self._completion_candidate(anchor, selection, tip=tip_id, observe=True, current=not is_live)
            if include_objective and is_live and candidate.rejections:
                job["checkpoint_objective_error"] = True
            pending = candidate.pending
            if tip is not None and (observed := candidate.observation) is not None:
                members = observed["members"]
                points = live_evaluate.selected_points(self.db, job["id"], members)
                if pending is not None and tip["state"] in LIVE_STATES:
                    reported = {point["publication_cursor"] for point in points}
                    scoring = min((
                        member["checkpoint"]["completed_updates"] for member in members
                        if member["producer_job_id"] == tip["id"] and member["publication_cursor"] not in reported
                    ), default=None)
                    if scoring is not None:
                        pending = pending | {"next_update": scoring}
        subject = job["launch"]["payload"]["canonical_request"]["checkpoint"]
        job["checkpoint_progress"] = {
            "pending": pending,
            "points": [
                {
                    "job_id": point["checkpoint"]["pointer"]["run_id"],
                    "request_id": subject["request_id"],
                    "completed_updates": point["checkpoint"]["completed_updates"],
                    "publication_cursor": point["publication_cursor"],
                    "transaction_root_sha256": point["checkpoint"]["pointer"]["transaction_root_sha256"],
                    "status": point["result"]["status"],
                }
                for point in points
            ],
        }
        if include_objective and is_live and observed is not None:
            publications = {member["publication_cursor"]: member for member in observed["members"]}
            selected = []
            for point in points:
                member = publications.get(point["publication_cursor"])
                if member is None or point["checkpoint"] != member["checkpoint"]:
                    job["checkpoint_objective_error"] = True
                    return
                selected.append({
                    "point": point,
                    "producer_job_id": member["producer_job_id"],
                    "producer_request_id": member["request_id"],
                    "binding_sha256": member["binding_sha256"],
                })
            job["checkpoint_objective"] = {
                "lineage": observed["lineage"],
                "gaps": observed.get("gaps", ()),
                "points": selected,
            }

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
                "SELECT id, launch, data FROM jobs WHERE project=? AND request=? AND continues IS NULL",
                (project, job["request"]),
            )
            if is_checkpoint_producer(
                json.loads(row["launch"]), continues=None, manifest_root=manifest_root,
                collected=json.loads(row["data"]).get("collected_train", False),
            )
        ]
        if len(anchors) != 1:
            return None
        binding = self._completion_candidate(anchors[0], CheckpointSelection((update,), False)).binding
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
            row = self.get(job_id)
            if row is not None and json.loads(row["launch"])["warm_entrypoint"] == LIGHT_ENTRYPOINT:
                raise HostError("invalid_request", "Managed light edges require the exact producer to succeed.")
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

    def stop(self, job_ids: list[str], mode: str, *, service: bool = False) -> list[dict]:
        if mode not in ("checkpoint", "force") and not (service and mode == "interrupt"):
            raise HostError("invalid_request", "Unknown stop mode.")
        strength = {None: 0, "checkpoint": 1, "interrupt": 2, "force": 3}
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
                if state in LIVE_STATES and strength[mode] > strength[row["stop_mode"]]:
                    mode_for_job = "checkpoint" if state == "queued" and mode == "interrupt" else mode
                    self.db.execute("UPDATE jobs SET stop_mode=? WHERE id=?", (mode_for_job, job_id))
                    if state == "queued":
                        self.finished(row, {"outcome": "stopped"})
                        state = "stopped"
                    else:
                        self.db.execute("UPDATE jobs SET state='stopping' WHERE id=?", (job_id,))
                        self.append(row, "stopping", {"mode": "force" if mode == "interrupt" else mode})
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
            return None, "Producer has no authenticated completion."
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

    def checkpoint_for_export(self, *, job_id: str, project: str, manifest_root: str, update: int) -> dict | None:
        """Authenticate a checkpoint on the named attempt's exact recovery lineage."""
        coordinates = finite_coordinates((update,))
        attempt = self.get(job_id)
        if (
            attempt is None or attempt["project"] != project or not attempt["request"].startswith("train:")
            or json.loads(attempt["data"]).get("manifest_root") != manifest_root
        ):
            return None
        path = [attempt]
        while path[-1]["continues"] is not None:
            ancestor = self.get(path[-1]["continues"])
            if ancestor is None or (ancestor["project"], ancestor["request"]) != (project, attempt["request"]):
                raise HostError("checkpoint_unavailable", "The selected Train lineage is incomplete.")
            path.append(ancestor)
        read = self._completion_candidate(
            path[-1]["id"], CheckpointSelection(coordinates, requires_catalog=False), tip=job_id, observe=True,
        )
        if read.binding is not None:
            return read.binding["checkpoint_selection"]["members"][0]["checkpoint"]
        if read.rejections:
            raise HostError("checkpoint_unavailable", read.rejections[-1].message)
        return None

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

    @inspect_without_recording()
    def _current_completion_tip(self, anchor: str) -> str | None:
        from feedbax.host.current_checkpoints import CurrentCheckpoints

        producer = self.get(anchor)
        if producer is None or not producer["request"].startswith("train:"):
            return None
        root = json.loads(producer["data"]).get("manifest_root")
        if root is None:
            return None
        return CurrentCheckpoints(self, self._success_view(root), producer["project"]).tip_for_anchor(producer)

    def _completion_candidate(
        self, anchor: str, selection: CheckpointSelection | None = None, *,
        train_slots: frozenset[str] | None = None, tip: str | None = None, observe: bool = False,
        terminal_model: bool = False, current: bool = False,
    ) -> _CompletionCandidate:
        """Select an authenticated final product or exact checkpoint publications.

        An earlier attempt can still produce another child whenever M1 permits its resume.
        """
        try:
            if (source := self.get(anchor)) is not None and source["state"] == "reused":
                retained = self.deferred.exact_source(source)
                tip = retained["id"]
                while retained["continues"] is not None:
                    retained = self.get(retained["continues"])
                    assert retained is not None
                anchor = retained["id"]
            elif current and tip is None:
                tip = self._current_completion_tip(anchor)
        except (KeyError, OSError, ValueError) as error:
            return _CompletionCandidate(None, {"code": "dependency_corrupt", "message": str(error)}, [])
        attempts = self.db.execute(
            """WITH RECURSIVE lineage(id) AS (
                   VALUES (?) UNION ALL
                   SELECT child.id FROM jobs child JOIN lineage ON child.continues=lineage.id
               )
               SELECT jobs.* FROM jobs JOIN lineage USING (id) ORDER BY jobs.seq""",
            (anchor,),
        )
        pinned_tip = tip
        if tip is not None:
            by_id = {attempt["id"]: attempt for attempt in attempts}
            pinned = []
            while tip is not None:
                attempt = by_id[tip]
                pinned.append(attempt)
                if tip == anchor:
                    break
                tip = attempt["continues"]
            attempts = reversed(pinned)
        resume_checkpoints: dict[str, str] = {}
        paths: dict[str, tuple[sqlite3.Row, ...]] = {}
        pending_paths: dict[str, tuple[dict, dict | None]] = {}
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
            observation = None
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
                        if observe:
                            observation = binding["checkpoint_selection"].copy()
                        if not absent and (not observe or pinned_tip is None or attempt_id == pinned_tip):
                            waiting = _pending_checkpoint(attempt, resume, ()) if (
                                observe and pinned_tip is not None and attempt["state"] in LIVE_STATES
                            ) else None
                            return _CompletionCandidate(
                                binding, None, rejections, waiting, observation if observe else None,
                            )
                        binding["checkpoint_selection"]["gaps"] = [
                            {"update": update, "reason": "checkpoint_unavailable"} for update in absent
                        ]
                        if pinned_tip is None or attempt_id == pinned_tip:
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
                    if terminal_model:
                        from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
                        from feedbax.persistence.evaluation_subject import terminal_train_subject_at

                        outcome = json.loads(attempt["outcome"] or "{}")
                        self._check_completion_identity(attempt, outcome)
                        ref = ContentObjectRef.model_validate(outcome.get("store_success"))
                        pin = terminal_train_subject_at(
                            ImmutableArtifactBlobProvider(Path(payload["manifest_root"])),
                            ref, attempt["request"], attempt=attempt_id,
                        )
                        return _CompletionCandidate(
                            {"dependency_job_id": anchor, "terminal": pin.model_dump(mode="json")}, None, rejections,
                        )
                    completion, message = self._completed_product(attempt, train_slots)
                except (CompletionFileError, OSError, KeyError, ValueError) as error:
                    rejections.append(_CompletionRejection(attempt_id, "completion", str(error)))
                else:
                    if completion is not None:
                        return _CompletionCandidate(
                            {"dependency_job_id": anchor, "completion": completion}, None, rejections,
                        )
                    missing = missing or message
            resumable = can_resume_attempt(state, attempt["checkpoint"], attempt["delivery"])
            boundary = resume
            if resumable:
                raw = attempt["checkpoint"]
                boundary = json.loads(raw) if raw is not None else None
                resume_checkpoints[attempt_id] = dump(boundary, sort_keys=True)
            if (pinned_tip is None or attempt_id == pinned_tip) and (resumable or state in LIVE_STATES):
                # A continuation starts after its recovery coordinate.
                path_pending = (selection is not None and selection.allows_gaps and state in LIVE_STATES) or (
                    can_publish and (
                        selection is None or selection.requires_catalog
                        or boundary is None
                        or any(update > boundary["completed_updates"] for update in (
                            absent if selection.allows_gaps else selection.coordinates[:1]
                        ))
                    )
                )
                pending |= path_pending
                if observe and path_pending and selection is not None and selection.allows_gaps:
                    for ancestor in path[:-1]:
                        pending_paths.pop(ancestor["id"], None)
                    pending_paths[attempt_id] = (
                        _pending_checkpoint(attempt, boundary, absent), observation,
                    )

        if completed_series is not None:
            return _CompletionCandidate(
                completed_series, None, rejections,
                observation=completed_series["checkpoint_selection"] if observe else None,
            )
        if pending:
            if len(pending_paths) == 1:
                waiting, observation = next(iter(pending_paths.values()))
                return _CompletionCandidate(None, None, rejections, waiting, observation)
            return _CompletionCandidate(None, None, rejections)
        if rejections:
            return _CompletionCandidate(None, {
                "code": "dependency_corrupt",
                "message": rejections[0].message,
            }, rejections)
        if settled_series is not None:
            return _CompletionCandidate(
                settled_series, None, rejections,
                observation=settled_series["checkpoint_selection"] if observe else None,
            )
        return _CompletionCandidate(None, {
            "code": "input_unavailable",
            "message": missing or "Producer ended without a resumable completion.",
            **({"path": f"/inputs/{selection.role}/checkpoint"} if selection is not None else {}),
        }, rejections)

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

    def _ingest_evaluation_point(self, producer: sqlite3.Row, data: dict) -> None:
        owner = live_evaluate.assigned_evaluation(self.db, producer, producer["delivery"])
        if owner is None or owner["state"] != "queued":
            return
        if owner["id"] != data.get("evaluation_job_id"):
            self.evaluation_error(producer, "Live Evaluate point names a different owning attempt.")
            return
        try:
            point, cursor = live_evaluate.authenticate_point(self.db, producer, owner, data)
            data = point.model_dump(mode="json") | {"publication_cursor": cursor}
            existing = self.db.execute(
                """SELECT data FROM events WHERE job=? AND kind=?
                   AND json_extract(data,'$.publication_cursor')=? LIMIT 1""",
                (owner["id"], LIVE_EVALUATE_POINT, cursor),
            ).fetchone()
            if existing is not None:
                if json.loads(existing[0]) != data:
                    raise CompletionFileError("Conflicting live Evaluate points for one checkpoint.")
                return
        except (OSError, KeyError, TypeError, ValueError) as error:
            self.evaluation_error(producer, str(error))
            return
        self.append(owner, LIVE_EVALUATE_POINT, data)

    def evaluation_error(self, producer: sqlite3.Row, message: str) -> None:
        owner = live_evaluate.assigned_evaluation(self.db, producer, producer["delivery"])
        if owner is not None and owner["state"] == "queued":
            self.finished(owner, {"outcome": "failed", "error": {
                "code": "live_evaluation_protocol", "message": message,
            }})

    def _place_evaluation(self, job: dict, producer: sqlite3.Row, selection: CheckpointSelection, occupied: set[str]) -> dict:
        from feedbax.training.validation_placement import (
            ValidationQualification, place_validation, validation_runtime,
        )

        source = json.loads(producer["launch"])
        payload = source["payload"]
        spec = DocumentTrainingRequestSpec.model_validate(payload["spec"])
        # Live dispatch never changes the Train publication schedule.
        supported = (
            source.get("warm_entrypoint") == TRAINING_ENTRYPOINT
            and producer["continues"] is None
            and spec.checkpoint_initialization is None
            and all(_scheduled_checkpoint(update, spec) for update in selection.coordinates)
        )
        qualification = None
        if raw := job["data"].pop("validation_qualification", None):
            qualification = ValidationQualification.model_validate(raw)
            if (
                qualification.train_request != producer["request"]
                or qualification.evaluate_request != job["request"]
            ):
                raise HostError("invalid_request", "Validation qualification differs from the admitted requests.")
            if qualification.runtime != validation_runtime(source):
                qualification = None
        placement = place_validation(
            selection.coordinates,
            producer_started=producer["state"] != "queued" or producer["delivery"] != 0,
            live_products_supported=supported,
            budget=qualification.budget if qualification is not None else None,
        )
        if placement.location == "live":
            if producer["id"] in occupied or live_evaluate.choose_evaluation(self.db, producer) is not None:
                placement = ValidationPlacement(location="ordinary", reason="driver_occupied")
            else:
                occupied.add(producer["id"])
        return placement.model_dump(mode="json", exclude_none=True) | (
            {"evidence": qualification.evidence} if qualification is not None else {}
        )

    def _resolve_best_completion(self, job: sqlite3.Row, rule: BestRank, candidates: dict) -> None:
        roots = {
            row["request"]: row["parent"] for row in self.db.execute(
                """SELECT e.parent, parent.request FROM edges e JOIN jobs parent ON parent.id=e.parent
                   WHERE e.job=? AND e.authenticated_completion""",
                (job["id"],),
            )
        }
        completed = {}
        pending = False
        for request, anchor in roots.items():
            key = anchor, None, None, None
            if key not in candidates:
                candidates[key] = self._completion_candidate(anchor, current=True)
            candidate = candidates[key]
            binding, error, rejections = candidate.binding, candidate.error, candidate.rejections
            self._record_completion_rejections(job, anchor, rejections)
            if error is not None:
                self.finished(job, {"outcome": "blocked", "error": error | {"path": "/inputs/subject"}})
                return
            if binding is None:
                pending = True
            else:
                completed[request] = binding
        if pending:
            return
        try:
            scored = {
                request: completion_score(binding["completion"], rule.metric)
                for request, binding in completed.items()
            }
        except (BestMetricUnavailable, CompletionFileError) as error:
            code = "input_unavailable" if isinstance(error, BestMetricUnavailable) else "dependency_corrupt"
            self.finished(job, {"outcome": "blocked", "error": {
                "code": code, "message": str(error), "path": "/inputs/subject",
            }})
            return
        ordered, selected = ranked_requests(rule, scored)
        winner = completed[selected]
        binding = winner | {"best_rank": {
            "key": rule.key, "rank": rule.rank, "candidates": [
                {"completion": completed[request]["completion"], "score": scored[request]}
                for request in ordered
            ],
        }}
        self.db.execute(
            """UPDATE edges SET satisfied=1, binding=CASE WHEN parent=? THEN ? END
               WHERE job=? AND authenticated_completion""",
            (roots[selected], dump(binding), job["id"]),
        )
        self.append(job, "dependency_bound", binding)

    def _condition_pin(
        self, job: sqlite3.Row, anchor: str, checkpoint: dict, candidates: dict,
    ) -> tuple[str, TerminalTrainSubject] | None:
        from feedbax.persistence.evaluation_subject import read_terminal_train_subject
        from feedbax.persistence.train_metrics import passes_final_metric_condition
        from feedbax.contracts.run_rule import METRIC_CONDITION_FALSE

        key = anchor, None, None, None, True
        if key not in candidates:
            candidates[key] = self._completion_candidate(anchor, terminal_model=True, current=True)
        candidate = candidates[key]
        self._record_completion_rejections(job, anchor, candidate.rejections)
        if candidate.error is not None:
            self.finished(job, {"outcome": "blocked", "error": candidate.error})
        if candidate.binding is None:
            return None
        launch = json.loads(job["launch"])
        try:
            reference = ConditionalProduct.model_validate(checkpoint)
            pin = TerminalTrainSubject.model_validate(candidate.binding["terminal"])
            store = self._success_view(launch["payload"]["manifest_root"]).store
            receipt, _ = read_terminal_train_subject(store, pin, reference.request_id)
            if passes_final_metric_condition(store, receipt, reference):
                return receipt.attempt, pin
        except (OSError, TypeError, KeyError, ValueError) as error:
            self.finished(job, {"outcome": "blocked", "error": {
                "code": "input_unavailable", "message": str(error),
            }})
        else:
            canonical = launch["payload"]["canonical_request"]
            proof = ExcludedProduct(
                request=RequestObject.model_validate({
                    "id": job["request"], "object": canonical["request"], "documents": canonical.get("documents", ()),
                }),
                cause=MetricExclusion(train=pin.receipt),
            )
            self.finished(job, {
                "outcome": "not_chosen", "reason": METRIC_CONDITION_FALSE,
                "excluded_product": proof.model_dump(mode="json"),
            })
        return None

    def resolve_completions(self) -> None:
        with self.db:
            rows = self.db.execute(
                """SELECT e.job, e.parent FROM edges e JOIN jobs j ON j.id=e.job
                   WHERE e.authenticated_completion AND NOT e.satisfied AND j.state='queued'
                   ORDER BY j.seq, e.parent"""
            ).fetchall()
            candidates = {}
            best_processed = set()
            for edge in rows:
                job = self.get(edge["job"])
                if job["state"] != "queued":
                    continue
                anchor = edge["parent"]
                launch = json.loads(job["launch"])
                checkpoint = (launch["payload"].get("canonical_request") or {}).get("checkpoint")
                if isinstance(checkpoint, dict) and checkpoint.get("kind") == "best_rank":
                    if job["id"] not in best_processed:
                        self._resolve_best_completion(job, BestRank.model_validate(checkpoint), candidates)
                        best_processed.add(job["id"])
                    continue
                try:
                    selection = requested_checkpoint_selection(job["request"], launch)
                    train_slots = train_source_slots(launch)
                except (CompletionFileError, OSError, KeyError, TypeError, ValueError) as error:
                    self.finished(job, {
                        "outcome": "blocked",
                        "error": {"code": "invalid_request", "message": str(error)},
                    })
                    continue
                condition = None
                if isinstance(checkpoint, dict) and checkpoint.get("kind") == "conditional":
                    condition = self._condition_pin(job, anchor, checkpoint, candidates)
                    if condition is None:
                        continue
                is_live = json.loads(job["data"]).get("placement") == "live"
                tip = live_evaluate.pinned_tip(self.db, job["id"]) if is_live else None
                if is_live and tip is None:
                    producer = self.get(anchor)
                    if producer["state"] in TERMINAL_STATES:
                        self.finished(job, {"outcome": "interrupted", "error": {
                            "code": "producer_unassigned", "message": "Train ended before live Evaluate was assigned.",
                        }})
                    continue
                if tip is not None and tip["state"] in LIVE_STATES:
                    continue
                tip_id = condition[0] if condition is not None else tip["id"] if tip is not None else None
                terminal_model = (
                    job["request"].startswith("evaluate:") and selection is None
                    and launch["warm_entrypoint"] == EVALUATE_ENTRYPOINT
                    and isinstance(checkpoint, dict) and (checkpoint.get("select") or {}).get("kind") == "last"
                )
                key = anchor, selection, train_slots, tip_id, terminal_model
                if condition is not None and terminal_model:
                    candidate = candidates[anchor, None, None, None, True]
                elif key in candidates:
                    candidate = candidates[key]
                else:
                    candidate = candidates[key] = self._completion_candidate(
                        anchor, selection, train_slots=train_slots, tip=tip_id, terminal_model=terminal_model,
                        current=not is_live,
                    )
                binding, error, rejections = candidate.binding, candidate.error, candidate.rejections
                self._record_completion_rejections(job, anchor, rejections)
                if error is not None:
                    outcome = "interrupted" if is_live and error["code"] == "input_unavailable" else "blocked"
                    self.finished(job, {"outcome": outcome, "error": error})
                elif binding is not None:
                    if condition is not None and "terminal" not in binding:
                        binding = binding | {"metric_producer": condition[1].receipt.model_dump(mode="json")}
                    if is_live:
                        binding = binding | {"live_evaluation_points": live_evaluate.selected_points(
                            self.db, job["id"], binding["checkpoint_selection"]["members"],
                        )}
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
                """SELECT binding FROM edges WHERE job=? AND authenticated_completion AND satisfied AND binding IS NOT NULL
                   ORDER BY parent""",
                (job_id,),
            )
        ]

    def verify_completion_bindings(self, row: sqlite3.Row, launch: dict, bindings: list[dict]) -> None:
        from feedbax.persistence.train_metrics import passes_final_metric_condition

        train_slots = train_source_slots(launch)
        payload = launch["payload"]
        request = payload.get("canonical_request") or {}
        terminal = request.get("terminal")
        for binding in bindings if terminal is None else ({"terminal": terminal},):
            checkpoint = request.get("checkpoint")
            if isinstance(checkpoint, dict) and checkpoint.get("kind") == "conditional":
                view = self._success_view(payload["manifest_root"])
                try:
                    evidence = (
                        TerminalTrainSubject.model_validate(binding["terminal"]).receipt
                        if "terminal" in binding else ContentObjectRef.model_validate(binding["metric_producer"])
                    )
                    producer = view.read_receipt(evidence)
                    if not passes_final_metric_condition(view.store, producer, ConditionalProduct.model_validate(checkpoint)):
                        raise ValueError("The producer's final metrics do not meet the rule.")
                except (OSError, TypeError, KeyError, ValueError) as error:
                    raise CompletionFileError(str(error)) from error
            if "terminal" in binding:
                from feedbax.persistence.evaluation_subject import read_terminal_train_subject

                try:
                    read_terminal_train_subject(
                        self._success_view(payload["manifest_root"]).store,
                        TerminalTrainSubject.model_validate(binding["terminal"]), request["checkpoint"]["request_id"],
                    )
                except (OSError, TypeError, ValueError, KeyError) as error:
                    raise CompletionFileError(str(error)) from error
            elif train_slots is None:
                verify_dependency_files(binding)
            elif row["continues"] is None:
                verify_completion_files(binding["completion"], train_slots=train_slots)

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
                   AND NOT EXISTS (SELECT 1 FROM deferred_reuses WHERE job=jobs.id)
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
