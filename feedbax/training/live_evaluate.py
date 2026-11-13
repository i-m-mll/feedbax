"""Run the shared Evaluate scorer after a managed checkpoint is durable."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import jax

from feedbax.contracts.artifact_custody import ContentRetentionTarget, ContentRootRecord
from feedbax.contracts.live_evaluate import (
    LIVE_EVALUATE_POINT, FailedEvaluationPoint, LiveEvaluationPoint, ScoredEvaluationPoint,
)
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import publish_dataset
from feedbax.studio.canonical_evaluate import BoundEvaluationJobInput, decode_rollout_request
from feedbax.studio.canonical_train import BoundStudioTrainingJobInput
from feedbax.training.checkpoint_custody import open_published_model_slot
from feedbax.training.evaluate_worker import prepare_rollout, rollout_dataset
from feedbax.training_host_worker import _Events, _error

if TYPE_CHECKING:
    from feedbax.runtime.graph import Graph


def live_evaluation_callback(
    launch: BoundStudioTrainingJobInput, events: _Events,
) -> Callable[[PublishedTrainingCheckpoint, Graph], None] | None:
    sidecar = os.environ.get("FEEDBAX_HOST_LIVE_EVALUATE")
    if sidecar is None:
        return None
    raw = Path(sidecar).read_bytes()
    if sha256(raw).hexdigest() != os.environ.get("FEEDBAX_HOST_LIVE_EVALUATE_SHA256"):
        raise ValueError("Live Evaluate assignment changed after delivery.")
    assignment = json.loads(raw)
    request = BoundEvaluationJobInput.model_validate(assignment["launch"])
    store = ImmutableArtifactBlobProvider(launch.manifest_root)
    _, params, subject = decode_rollout_request(
        request.request_id, store.get_object(request.canonical_request.request),
    )
    if (
        request.project_id != launch.project_id or request.manifest_root != launch.manifest_root
        or subject != request.canonical_request.checkpoint or subject.request_id != launch.request_id
        or params.variation or assignment["producer_job_id"] != os.environ["FEEDBAX_HOST_JOB_ID"]
    ):
        raise ValueError("Live Evaluate assignment differs from its admitted Train.")
    coordinates = frozenset(assignment["updates"])
    train_stop = Path(os.environ["FEEDBAX_HOST_STOP_FILE"])
    evaluation_stop = Path(assignment["stop_file"])

    prepared = None
    compiled = None

    def score(checkpoint: PublishedTrainingCheckpoint, model: Graph) -> None:
        nonlocal prepared, compiled
        if checkpoint.completed_updates not in coordinates or train_stop.exists() or evaluation_stop.exists():
            return
        try:
            if compiled is None:
                prepared = prepare_rollout(launch, checkpoint, params)
                compiled = prepared.scorer.lower(model, capture_state=prepared.layout.capture).compile()
                slot = prepared.slot
            else:
                slot = open_published_model_slot(checkpoint.checkpoint_root, checkpoint.pointer)
                if slot is None or slot.graph != prepared.slot.graph:
                    raise ValueError("Live Evaluate model layout changed after preparation.")
            if train_stop.exists() or evaluation_stop.exists():
                return
            evaluated = jax.block_until_ready(compiled(model))
            if train_stop.exists() or evaluation_stop.exists():
                return
            member = prepared.member.model_copy(update={
                "coordinate": checkpoint.completed_updates, "model_tree": slot.tree.descriptor.sha256,
            })
            bound = replace(prepared, graph=model, slot=slot, member=member)
            dataset = rollout_dataset(bound, evaluated, params, update_axis="update")
            with store._exclusive_lock():
                states = publish_dataset(store, dataset)
                point = LiveEvaluationPoint(
                    evaluation_job_id=assignment["evaluation_job_id"], checkpoint=checkpoint,
                    result=ScoredEvaluationPoint(
                        states=states, member=member, objective=bound.objective,
                        total=float(evaluated.objective.loss),
                    ),
                )
                store.register_root(ContentRootRecord(
                    root_id=point.root_id, root_kind="run",
                    targets=(ContentRetentionTarget(ref_kind="object", object=states.object_ref),),
                ))
        except Exception as error:
            point = LiveEvaluationPoint(
                evaluation_job_id=assignment["evaluation_job_id"], checkpoint=checkpoint,
                result=FailedEvaluationPoint(error=_error(error, "live_evaluate")),
            )
        events.append(LIVE_EVALUATE_POINT, point.model_dump(mode="json"), sync=True)

    return score
