"""Host ownership and verified event delivery for delegated Evaluate scoring."""

from collections.abc import Sequence
import json
from pathlib import Path
import sqlite3

from feedbax.contracts.live_evaluate import LIVE_EVALUATE_POINT, LiveEvaluationPoint, ScoredEvaluationPoint
from feedbax.host.checkpoint_selection import requested_checkpoint_selection, verify_selected_publication
from feedbax.host.completion import CompletionFileError

PLACEMENT_EVENT = "evaluation_placement"
DELEGATION_EVENT = "evaluation_delegated"
SCHEMA = """
CREATE TABLE IF NOT EXISTS live_evaluations (
    evaluation TEXT NOT NULL REFERENCES jobs(id),
    producer TEXT NOT NULL REFERENCES jobs(id),
    delivery INTEGER NOT NULL,
    PRIMARY KEY (evaluation, producer, delivery),
    UNIQUE (producer, delivery)
);
"""


def assigned_evaluation(db: sqlite3.Connection, producer: sqlite3.Row, delivery: int) -> sqlite3.Row | None:
    return db.execute(
        """SELECT j.* FROM live_evaluations d JOIN jobs j ON j.id=d.evaluation
           WHERE d.producer=? AND d.delivery=?""",
        (producer["id"], delivery),
    ).fetchone()


def choose_evaluation(db: sqlite3.Connection, producer: sqlite3.Row) -> sqlite3.Row | None:
    """Select one queued owner, or inherit only the exact selected parent's owner."""
    previous = db.execute(
        """SELECT d.*, j.checkpoint, j.launch FROM live_evaluations d JOIN jobs j ON j.id=d.producer
           WHERE d.producer IN (?, ?) ORDER BY j.seq DESC, d.delivery DESC LIMIT 1""",
        (producer["id"], producer["continues"]),
    ).fetchone()
    if previous is None:
        return db.execute(
            """SELECT j.* FROM edges e JOIN jobs j ON j.id=e.job
               WHERE e.parent=? AND e.authenticated_completion AND NOT e.satisfied AND j.state='queued'
               AND json_extract(j.data,'$.placement')='live'
               AND NOT EXISTS (SELECT 1 FROM live_evaluations d WHERE d.evaluation=j.id)
               ORDER BY j.seq LIMIT 1""",
            (producer["id"],),
        ).fetchone()
    owner = db.execute("SELECT * FROM jobs WHERE id=? AND state='queued'", (previous["evaluation"],)).fetchone()
    if owner is None:
        return None
    if previous["producer"] == producer["id"]:
        return owner
    parent = json.loads(previous["launch"])
    launch = json.loads(producer["launch"])
    parent_payload = parent.pop("payload")
    payload = launch.pop("payload")
    resume = payload.pop("resume", None)
    parent_payload.pop("resume", None)
    if (
        resume is None or previous["checkpoint"] is None or resume != json.loads(previous["checkpoint"])
        or launch != parent or payload != parent_payload
    ):
        return None
    newer = db.execute(
        """SELECT 1 FROM live_evaluations d JOIN jobs j ON j.id=d.producer
           WHERE d.evaluation=? AND j.seq > (SELECT seq FROM jobs WHERE id=?)""",
        (owner["id"], previous["producer"]),
    ).fetchone()
    return owner if newer is None else None


def pinned_tip(db: sqlite3.Connection, evaluation: str) -> sqlite3.Row | None:
    return db.execute(
        """SELECT j.* FROM live_evaluations d JOIN jobs j ON j.id=d.producer
           WHERE d.evaluation=? ORDER BY j.seq DESC, d.delivery DESC LIMIT 1""",
        (evaluation,),
    ).fetchone()


def authenticate_point(
    db: sqlite3.Connection, producer: sqlite3.Row, owner: sqlite3.Row, data: dict,
) -> tuple[LiveEvaluationPoint, int]:
    point = LiveEvaluationPoint.model_validate(data)
    update = point.checkpoint.completed_updates
    selection = requested_checkpoint_selection(owner["request"], json.loads(owner["launch"]))
    if selection is None or update not in selection.coordinates or point.checkpoint.pointer.run_id != producer["id"]:
        raise CompletionFileError("Live Evaluate point differs from its assignment.")
    publication = db.execute(
        """SELECT cursor, data FROM events WHERE job=? AND kind='publication'
           AND json_extract(data,'$.key')=? ORDER BY cursor LIMIT 1""",
        (producer["id"], f"checkpoint:{update}"),
    ).fetchone()
    accepted_binding = db.execute(
        "SELECT data FROM events WHERE job=? AND kind='binding' ORDER BY cursor DESC LIMIT 1",
        (producer["id"],),
    ).fetchone()
    if publication is None or accepted_binding is None:
        raise CompletionFileError("Live Evaluate point has no admitted checkpoint publication.")
    root = json.loads(owner["launch"])["payload"]["manifest_root"]
    product = verify_selected_publication(
        json.loads(publication["data"]), attempt=producer["id"], project=owner["project"],
        request=producer["request"], root=root, binding_sha256=json.loads(accepted_binding["data"])["binding_sha256"],
    )
    if product["checkpoint"] != point.checkpoint.model_dump(mode="json"):
        raise CompletionFileError("Live Evaluate point differs from its exact checkpoint.")
    if isinstance(point.result, ScoredEvaluationPoint):
        if point.result.member.coordinate != update:
            raise CompletionFileError("Live Evaluate product differs from its update.")
        from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
        from feedbax.persistence.content_retention import require_active_root

        store = ImmutableArtifactBlobProvider(Path(root))
        ref = point.result.states.object_ref
        custody = require_active_root(store, point.root_id)
        if len(custody.targets) != 1 or custody.targets[0].object != ref:
            raise CompletionFileError("Live Evaluate point lacks its exact durable object.")
    return point, publication["cursor"]


def selected_points(db: sqlite3.Connection, evaluation: str, members: Sequence[dict]) -> list[dict]:
    accepted = {
        item["publication_cursor"]: item
        for row in db.execute(
            "SELECT data FROM events WHERE job=? AND kind=? ORDER BY cursor",
            (evaluation, LIVE_EVALUATE_POINT),
        )
        if (item := json.loads(row[0]))
    }
    return [accepted[member["publication_cursor"]] for member in members if member["publication_cursor"] in accepted]
