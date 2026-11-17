"""Run the shared Evaluate scorer after a managed checkpoint is durable."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from feedbax.contracts.artifact_custody import ContentRootRecord
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.live_evaluate import (
    LIVE_EVALUATE_POINT, FailedEvaluationPoint, LiveEvaluationPoint, ScoredEvaluationPoint,
)
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import publish_dataset
from feedbax.persistence.live_evaluate import live_evaluation_targets
from feedbax.studio.canonical_evaluate import BoundEvaluationJobInput, decode_rollout_request
from feedbax.studio.canonical_train import BoundStudioTrainingJobInput
from feedbax.training.method_checkpoint import read_method_checkpoint
from feedbax.training.evaluate_computation import EvaluationComputation, RolloutExecutables
from feedbax.training.evaluate_worker import _prepare_model_rollout, _rollouts
from feedbax.training_host_worker import _Events, _error
from feedbax.validity.recording import (
    execution_dependencies, inherit_execution_dependencies, nested_execution, record_computation,
    snapshot_execution,
)

if TYPE_CHECKING:
    from feedbax.runtime.graph import Graph
    from feedbax.plugins.application import ApplicationRegistryBundle


def live_evaluation_callback(
    launch: BoundStudioTrainingJobInput, events: _Events, *, registry: ApplicationRegistryBundle,
) -> Callable[[PublishedTrainingCheckpoint, tuple[Graph, ...]], None] | None:
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

    executables = RolloutExecutables()
    prepared = None
    dependencies = None
    kernel = None

    def score(checkpoint: PublishedTrainingCheckpoint, models: tuple[Graph, ...]) -> None:
        if checkpoint.completed_updates not in coordinates or stopped():
            return
        try:
            with nested_execution(assignment["evaluation_job_id"]):
                point = evaluate_point(checkpoint, models)
        except Exception as error:
            point = LiveEvaluationPoint(
                evaluation_job_id=assignment["evaluation_job_id"], checkpoint=checkpoint,
                result=FailedEvaluationPoint(error=_error(error, "live_evaluate")),
            )
        if point is not None:
            events.append(LIVE_EVALUATE_POINT, point.model_dump(mode="json"), sync=True)

    def stopped() -> bool:
        return train_stop.exists() or evaluation_stop.exists()

    def evaluate_point(checkpoint, models):
        nonlocal prepared, dependencies, kernel
        slot = read_method_checkpoint(
            checkpoint.checkpoint_root, checkpoint.pointer, components=registry.components,
        ).model
        if prepared is None:
            initial = _prepare_model_rollout(
                launch.spec.trial_documents, params, slot, checkpoint.completed_updates,
                seed=launch.spec.seed or 0, registry=registry,
            )
            captured = execution_dependencies()
            scoring_kernel = initial.scorer.kernel(capture_state=initial.layout.capture)
        else:
            if slot.graph != prepared.slot.graph or slot.replica_ids != prepared.slot.replica_ids:
                raise ValueError("Live Evaluate model layout changed after preparation.")
            inherit_execution_dependencies(dependencies)
            initial, captured, scoring_kernel = prepared, dependencies, kernel
        member = initial.member.model_copy(update={
            "coordinate": checkpoint.completed_updates, "model_tree": slot.tree.descriptor.sha256,
        })
        replicas = tuple(zip(slot.replica_ids, models, strict=True))
        bound = replace(
            initial, graph=models[0], slot=slot, member=member, replicas=replicas if len(replicas) > 1 else (),
        )
        computation = EvaluationComputation(store)
        record_computation(computation.snapshot)
        result = _rollouts(
            bound, params, stopped, update_axis="update", computation=computation, executables=executables,
            kernel=scoring_kernel,
        )
        if result is None:
            return None
        dataset, total, member, objective = result
        if total is None or objective is None:
            raise ValueError("Live Evaluate requires objective-bearing results.")
        with store._exclusive_lock():
            states = publish_dataset(store, dataset)
            evidence = snapshot_execution(store)
            if evidence is None:
                raise ValueError("Live Evaluate completed outside managed execution.")
            evidence_ref = store.store_object(canonical_json_v2_bytes(evidence.model_dump(mode="json")))
            point = LiveEvaluationPoint(
                evaluation_job_id=assignment["evaluation_job_id"], checkpoint=checkpoint,
                result=ScoredEvaluationPoint(
                    states=states, member=member, objective=objective, total=total, evidence=evidence_ref,
                ),
            )
            _, targets = live_evaluation_targets(store, point)
            store.register_root(ContentRootRecord(root_id=point.root_id, root_kind="run", targets=targets))
        prepared, dependencies, kernel = initial, captured, scoring_kernel
        return point

    return score
