"""Host ownership and verified event delivery for delegated Evaluate scoring."""

from collections.abc import Sequence
import json
from pathlib import Path
import sqlite3

from feedbax.contracts.live_evaluate import LIVE_EVALUATE_POINT, LiveEvaluationObservation, LiveEvaluationPoint, ScoredEvaluationPoint
from feedbax.persistence.checkpoint_selection import requested_checkpoint_selection, verify_selected_publication
from feedbax.persistence.completion import CompletionFileError
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.jobs import JobReceipt, StudioEvaluationCheckpoint, StudioEvaluationObjectiveSeries
from feedbax.contracts.objective_document import ObjectiveDocument
from feedbax.contracts.rollout import RolloutParameters, rollout_objective_sha256, rollout_semantics
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.canonical_evaluate import _objective_points
from feedbax.persistence.canonical_train import BoundStudioTrainingJobInput

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
        from feedbax.persistence.live_evaluate import read_live_evaluation_evidence

        try:
            read_live_evaluation_evidence(ImmutableArtifactBlobProvider(Path(root)), point)
        except (OSError, ValueError) as error:
            raise CompletionFileError(str(error)) from error
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


def live_objective_series(
    store: ImmutableArtifactBlobProvider, *, project_id: str, job_id: str, request_id: str,
    producer_request: str, params: RolloutParameters, requested: list[int],
    objective: ObjectiveDocument, observation: LiveEvaluationObservation,
) -> StudioEvaluationObjectiveSeries:
    """Authenticate a live objective from the exact datasets accepted by the Host."""
    from feedbax.host.job_completion import read_training_binding
    from feedbax.persistence.content_retention import read_active_roots
    from feedbax.persistence.dataset import open_stored_dataset
    from feedbax.persistence.live_evaluate import read_live_evaluation_evidence
    from feedbax.host.method_checkpoint import read_method_checkpoint_manifest

    objective_hash = rollout_objective_sha256(objective)
    condition = params.condition_id()
    if params.variation or len(set(observation.lineage)) != len(observation.lineage):
        raise ValueError("Live objectives require one unchanged scientific lineage.")
    points = []
    scored = []
    seen_updates = set()
    for selected in observation.points:
        point = selected.point
        checkpoint = point.checkpoint
        update = checkpoint.completed_updates
        if (
            point.evaluation_job_id != job_id or selected.producer_request_id != producer_request
            or selected.producer_job_id != checkpoint.pointer.run_id
            or selected.producer_job_id not in observation.lineage
            or checkpoint.checkpoint_root != store.root / "checkpoints" / selected.producer_job_id
            or update not in requested or update in seen_updates
        ):
            raise ValueError("Live objective differs from its selected publication.")
        seen_updates.add(update)
        if isinstance(result := point.result, ScoredEvaluationPoint):
            scored.append((selected, result))
    if seen_updates.intersection(gap.update for gap in observation.gaps):
        raise ValueError("Live objective gaps overlap accepted observations.")

    launches: dict[str, BoundStudioTrainingJobInput] = {}
    semantics: dict[bytes, tuple[str, str]] = {}
    roots = read_active_roots(store, tuple(selected.point.root_id for selected, _ in scored)) if scored else ()
    for (selected, result), root in zip(scored, roots, strict=True):
        point = selected.point
        checkpoint = point.checkpoint
        pointer = checkpoint.pointer
        update = checkpoint.completed_updates
        read_live_evaluation_evidence(store, point, active_root=root)
        if selected.binding_sha256 not in launches:
            launches[selected.binding_sha256] = read_training_binding(
                store.root, selected.producer_job_id, selected.binding_sha256,
            )
        source = launches[selected.binding_sha256]
        if (
            source.project_id != project_id or source.request_id != producer_request
            or source.manifest_root != store.root or source.spec.trial_documents is None
            or source.spec.trial_documents.objective != objective
        ):
            raise ValueError("Live objective differs from its bound Train request.")
        identity = canonical_json_v2_bytes(source.model_dump(mode="json", exclude={"resume"}))
        if identity not in semantics:
            semantics[identity] = _unchanged_rollout_semantics(source)
        model, task = semantics[identity]
        member = result.member
        if (
            (member.model, member.task, result.objective) != (model, task, objective_hash)
            or member.model_tree != read_method_checkpoint_manifest(
                checkpoint.checkpoint_root, pointer,
            ).model.tree.descriptor.sha256
            or member.coordinate != update or member.trials != params.trials
        ):
            raise ValueError("Live objective differs from its resolved scientific inputs.")
        subject = StudioEvaluationCheckpoint(
            job_id=selected.producer_job_id, request_id=selected.producer_request_id,
            completed_updates=update, publication_cursor=point.publication_cursor,
            transaction=JobReceipt(manifest_id=pointer.transaction_id, manifest_sha256=pointer.manifest_sha256),
            transaction_root_sha256=pointer.transaction_root_sha256, binding_sha256=selected.binding_sha256,
        )
        with open_stored_dataset(store, result.states, verify=True) as dataset:
            if (
                dataset.descriptor.schema_id != params.states_schema
                or dataset.data("update").tolist() != [update]
                or dataset.data("condition").tolist() != [condition]
                or dataset.data("trial").tolist() != list(params.trials)
            ):
                raise ValueError("Live objective dataset differs from its authored axes.")
            (numeric,) = _objective_points(dataset, [subject], objective, len(params.trials))
            if numeric.total != result.total:
                raise ValueError("Live objective total differs from its accepted dataset.")
            points.append(numeric)
    return StudioEvaluationObjectiveSeries(
        job_id=job_id, request_id=request_id, receipt=None, lineage=observation.lineage,
        condition=condition if points else None, trials=list(params.trials), seed=params.seed,
        objective=objective_hash, requested=requested, points=points, gaps=list(observation.gaps),
    )


def _unchanged_rollout_semantics(source: BoundStudioTrainingJobInput) -> tuple[str, str]:
    from feedbax.compiler.graph import resolve_graph_schedule
    from feedbax.application.composition import load_application
    from feedbax.persistence.canonical_train import resolve_training_request_binding

    registry = load_application().bundle
    trial = resolve_training_request_binding(source, registries=registry).trial
    graph = trial.graph.graph
    _, key_schedule = resolve_graph_schedule(graph)
    return rollout_semantics(graph, key_schedule, trial.task_source, trial.execution.input_bindings, {})
