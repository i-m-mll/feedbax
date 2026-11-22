"""Resolve queued reuse reservations against the exact admitted inputs."""

from __future__ import annotations

from hashlib import sha256
import json
import sqlite3
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.effective_success import TerminalTrainSubject
from feedbax.contracts.execution_target import attempt_target
from feedbax.contracts.request_refs import BestRank
from feedbax.persistence.checkpoint_selection import requested_checkpoint_selection
from feedbax.host.live_evaluate import PLACEMENT_EVENT
from feedbax.host.protocol import DeferredReuse, HostError, LIGHT_ENTRYPOINT, MANAGED_ENTRYPOINTS
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import PublishedSuccess, SuccessView
from feedbax.persistence.store_damage import inspect_without_recording
from feedbax.validity.admission_projection import AdmissionProjection
from feedbax.validity.project_admission import (
    ProjectValidity, current_input_differences, is_quick, receipt_traces, reuse_disposition,
)

if TYPE_CHECKING:
    from feedbax.host.store import Store


DEFERRED_REUSE_SCHEMA = """
CREATE TABLE IF NOT EXISTS deferred_projections (
    id TEXT PRIMARY KEY,
    observation TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS deferred_reuses (
    job TEXT PRIMARY KEY REFERENCES jobs(id),
    attempt TEXT NOT NULL REFERENCES jobs(id),
    receipt TEXT NOT NULL,
    policy TEXT NOT NULL CHECK (policy IN ('selected','automatic','prerequisite')),
    quick_seconds REAL NOT NULL CHECK (quick_seconds >= 0),
    projection TEXT REFERENCES deferred_projections(id)
);
CREATE INDEX IF NOT EXISTS reuse_projection ON deferred_reuses(projection);
"""

_RESERVATION = TypeAdapter(DeferredReuse)
_UNAVAILABLE = (KeyError, OSError, TypeError, ValueError)
_RETRY_MESSAGE = "The result changed during admission. Check the current result and retry Run."


class DeferredAdmissions:
    def __init__(self, store: Store):
        self.store = store
        self.db = store.db

    def _reservation(self, job: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM deferred_reuses WHERE job=?", (job,)).fetchone()

    def reserve(self, row: sqlite3.Row, job: dict, value: dict, observation: dict | None) -> None:
        try:
            pending = _RESERVATION.validate_python(value)
            projection = AdmissionProjection.model_validate(observation)
            ref = ContentObjectRef.model_validate(pending.receipt)
            seconds = timedelta(seconds=pending.quick_seconds).total_seconds()
            if (
                not job["launch"].get("project_root") or job["launch"]["warm_entrypoint"] not in MANAGED_ENTRYPOINTS
                or job["continues"] is not None or not job["edges"] or any(edge["publication"] for edge in job["edges"])
                or seconds < 0 or "run_key" not in job["data"]
                or projection.roots.get("project") != Path(job["launch"]["project_root"]).resolve()
            ):
                raise ValueError("Deferred reuse requires a managed queued request with exact producer edges.")
            source = self.store.get(pending.attempt)
            if not self._same_request(row, source) or source["state"] != "succeeded":
                raise ValueError("Deferred reuse must name a successful attempt of the same request.")
        except (*_UNAVAILABLE, OverflowError) as error:
            raise HostError("invalid_request", str(error)) from error
        expected = projection.model_dump_json()
        projection_id = sha256(expected.encode()).hexdigest()
        self.db.execute("INSERT OR IGNORE INTO deferred_projections VALUES (?, ?)", (projection_id, expected))
        self.db.execute(
            """INSERT INTO deferred_reuses(job, attempt, receipt, policy, quick_seconds, projection)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (row["id"], pending.attempt, ref.model_dump_json(), pending.policy, seconds, projection_id),
        )

    @staticmethod
    def _same_request(row: sqlite3.Row, source: sqlite3.Row | None) -> bool:
        return source is not None and (
            source["project"], source["request"], json.loads(source["data"]).get("manifest_root"),
        ) == (row["project"], row["request"], json.loads(row["data"]).get("manifest_root"))

    def exact_source(self, row: sqlite3.Row) -> sqlite3.Row:
        if row["state"] != "reused":
            return row
        reservation = self._reservation(row["id"])
        if reservation is None:
            raise ValueError("The retained admission has no exact result reference.")
        view = self.store._success_view(json.loads(row["data"])["manifest_root"])
        return self._source(row, reservation, view)

    def _source(self, row: sqlite3.Row, pending: sqlite3.Row, view: SuccessView) -> sqlite3.Row:
        source = self.store.get(pending["attempt"])
        ref = ContentObjectRef.model_validate_json(pending["receipt"])
        if (
            not self._same_request(row, source) or source["state"] != "succeeded"
            or view.attempt_ref(pending["attempt"]) != ref
            or ContentObjectRef.model_validate(json.loads(source["outcome"])["store_success"]) != ref
        ):
            raise ValueError("The retained admission no longer has its exact successful attempt.")
        return source

    def _published(self, row: sqlite3.Row, pending: sqlite3.Row, view: SuccessView) -> PublishedSuccess:
        source = self._source(row, pending, view)
        ref = ContentObjectRef.model_validate_json(pending["receipt"])
        if view.authored_ref(row["request"]) != ref:
            raise ValueError("The retained receipt is no longer the authored success.")
        success = view.read_owned_success(ref)
        if (success.receipt.attempt, success.receipt.request.id, success.receipt.target) != (
            source["id"], source["request"], attempt_target(json.loads(source["data"]).get("target")),
        ):
            raise ValueError("The retained receipt names a different attempt.")
        return success

    def discard(self, job: str) -> bool:
        pending = self.db.execute("DELETE FROM deferred_reuses WHERE job=? RETURNING projection", (job,)).fetchone()
        if pending is None:
            return False
        self._drop_projection(pending["projection"])
        return True

    def _drop_projection(self, projection: str | None) -> None:
        if projection is not None:
            self.db.execute(
                """DELETE FROM deferred_projections WHERE id=?
                   AND NOT EXISTS (SELECT 1 FROM deferred_reuses WHERE projection=?)""",
                (projection, projection),
            )

    def execute(self, row: sqlite3.Row, *, occupied: set[str] | None = None) -> None:
        if not self.discard(row["id"]):
            return
        launch = json.loads(row["launch"])
        selection = requested_checkpoint_selection(row["request"], launch)
        if row["request"].startswith("evaluate:") and selection is not None and selection.allows_gaps:
            (anchor,) = self.store._parents(row["id"], {}, completion=True)
            source = self.store.get(anchor)
            assert source is not None
            data = json.loads(row["data"])
            job = {"launch": launch, "data": data, "request": row["request"]}
            placement = self.store._place_evaluation(job, source, selection, occupied if occupied is not None else set())
            data.update(placement=placement["location"], placement_detail=placement)
            self.db.execute("UPDATE jobs SET data=? WHERE id=?", (json.dumps(data), row["id"]))
            self.store.append(row, PLACEMENT_EVENT, placement)

    @inspect_without_recording()
    def settle(self) -> None:
        rows = self.db.execute(
            """SELECT jobs.* FROM jobs JOIN deferred_reuses d ON d.job=jobs.id
               WHERE state='queued' AND delivery=0 AND stop_mode IS NULL
               AND NOT EXISTS (SELECT 1 FROM edges WHERE job=jobs.id AND NOT satisfied)
               ORDER BY seq"""
        ).fetchall()
        for row in rows:
            launch = json.loads(row["launch"])
            store = ImmutableArtifactBlobProvider(Path(launch["payload"]["manifest_root"]))
            with store._exclusive_lock(), self.db:
                self._settle(row, launch, SuccessView(store))

    def _settle(self, row: sqlite3.Row, launch: dict, view: SuccessView) -> None:
        pending = self._reservation(row["id"])
        assert pending is not None
        try:
            published = self._published(row, pending, view)
        except _UNAVAILABLE:
            self._execute_or_block(row, pending, quick=pending["policy"] != "automatic")
            return
        threshold = timedelta(seconds=pending["quick_seconds"])

        def quick() -> bool:
            return is_quick(published.receipt, threshold, view.store)

        try:
            origins, bindings = self._input_authority(row, launch, view)
            equal = self._equal_inputs(row, published, view, origins, bindings)
        except _UNAVAILABLE as error:
            self._block(row, "dependency_corrupt", str(error))
            return
        if not equal:
            self._execute_or_block(row, pending, quick=pending["policy"] != "automatic" or quick())
            return
        projection_json = self.db.execute(
            "SELECT observation FROM deferred_projections WHERE id=?", (pending["projection"],),
        ).fetchone()
        assert projection_json is not None
        projection = AdmissionProjection.model_validate_json(projection_json[0])
        root = Path(launch["project_root"])
        try:
            traces = receipt_traces(view.store, published.receipt)
            if projection.current(traces):
                observer = ProjectValidity(root, projection, quick_duration=threshold)
                validity = observer.inspect(view, published, origins=origins).decision
                disposition = reuse_disposition(validity, pending["policy"], quick=quick)
            else:
                disposition = "unavailable"
        except _UNAVAILABLE:
            disposition = "unavailable"
        match disposition:
            case "reuse":
                updated = self.db.execute(
                    """UPDATE deferred_reuses SET projection=NULL WHERE job=? AND projection=?
                       AND EXISTS (SELECT 1 FROM jobs WHERE id=? AND state='queued' AND delivery=0
                                   AND stop_mode IS NULL AND outcome IS NULL)""",
                    (row["id"], pending["projection"], row["id"]),
                )
                if not updated.rowcount:
                    return
                self._drop_projection(pending["projection"])
                self.store.finished(row, {"outcome": "reused", "retained_attempt": pending["attempt"]})
            case "execute":
                self.execute(row)
            case "check":
                self._block(row, "admission_changed", _RETRY_MESSAGE)
            case "explicit_run":
                self._block(row, "explicit_run_required", "This request requires explicit Run.")
            case "unavailable":
                self._execute_or_block(row, pending, quick=pending["policy"] != "automatic" or quick())

    def _input_authority(
        self, row: sqlite3.Row, launch: dict, view: SuccessView,
    ) -> tuple[dict[str, ContentObjectRef], list[dict]]:
        origins = (
            {request: ContentObjectRef.model_validate(value) for request, value in self.store.light_origins(row).items()}
            if launch["warm_entrypoint"] == LIGHT_ENTRYPOINT else {}
        )
        bindings = self.store.completion_bindings(row["id"])
        self.store.verify_completion_bindings(row, launch, bindings)
        for binding in bindings:
            if "best_rank" in binding:
                from feedbax.persistence.canonical_evaluate import best_bound_rank

                rank = BestRank.model_validate(launch["payload"]["canonical_request"]["checkpoint"])
                best_bound_rank(rank, binding, verify_metrics=True)
            elif "terminal" in binding:
                pin = TerminalTrainSubject.model_validate(binding["terminal"])
                predecessor = launch["payload"]["canonical_request"]["checkpoint"]["request_id"]
                origins[predecessor] = pin.receipt
            elif row["request"].startswith("train:"):
                completion = binding["completion"]
                source = self.store.get(completion["producer_job_id"])
                if source is None:
                    raise ValueError("The pinned Train producer is unavailable.")
                origins[source["request"]] = ContentObjectRef.model_validate(self.store._light_receipt(view, source))
        return origins, bindings

    @staticmethod
    def _equal_inputs(
        row: sqlite3.Row, published: PublishedSuccess, view: SuccessView,
        origins: dict[str, ContentObjectRef], bindings: list[dict],
    ) -> bool:
        if bindings and row["request"].startswith(("analyze:", "evaluate:")):
            from feedbax.host.current_checkpoints import compare_pinned_checkpoint

            (binding,) = bindings
            if compare_pinned_checkpoint(published.receipt, binding) is not None:
                return False
        return not current_input_differences(
            view, published.receipt, origins=origins, authenticate_origins=True,
        )

    def _execute_or_block(self, row: sqlite3.Row, pending: sqlite3.Row, *, quick: bool) -> None:
        if pending["policy"] == "automatic" and not quick:
            self._block(row, "explicit_run_required", "This request requires explicit Run.")
        else:
            self.execute(row)

    def _block(self, row: sqlite3.Row, code: str, message: str) -> None:
        self.store.finished(row, {"outcome": "blocked", "error": {"code": code, "message": message}})
