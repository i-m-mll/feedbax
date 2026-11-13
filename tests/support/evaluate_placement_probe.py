"""Measure the same resolved Evaluate rollout at a managed checkpoint callback."""

from hashlib import sha256
import os
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import jax

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import publish_dataset
from feedbax.studio.canonical_evaluate import decode_rollout_request
from feedbax.studio.canonical_train import BoundStudioTrainingJobInput
from feedbax.training import executor
from feedbax.training.evaluate_worker import prepare_rollout, rollout_dataset
from feedbax.training_host_worker import _Events, main as train


def main(argv):
    launch = BoundStudioTrainingJobInput.model_validate_json(
        Path(os.environ["FEEDBAX_HOST_INPUT"]).read_bytes(),
    )
    store = ImmutableArtifactBlobProvider(Path(os.environ["EVALUATE_PLACEMENT_STORE"]))
    raw = store.get_object(ContentObjectRef.model_validate_json(os.environ["EVALUATE_PLACEMENT_REQUEST"]))
    request_id = os.environ["EVALUATE_PLACEMENT_ID"]
    _, params, subject = decode_rollout_request(request_id, raw)
    assert subject.request_id == launch.request_id
    events = _Events(Path(os.environ["FEEDBAX_HOST_EVENT_LOG"]))
    execute = executor.execute_managed_training_run

    def measured(spec, **kwargs):
        committed = kwargs["committed_update_callback"]
        published = kwargs["checkpoint_published"]
        checkpoint_update = subject.select.updates[-1]
        observe = os.environ["EVALUATE_PLACEMENT"] == "live"
        last = None
        start = None

        def on_update(coordinate, update, slots):
            nonlocal last, start
            if update == 2:
                start = perf_counter()
            committed(coordinate, update, slots)
            last = (update, slots["model"])

        def on_checkpoint(write):
            published(write)
            update = write.latest_pointer.completed_training_batches
            if not observe or update != checkpoint_update:
                return
            assert last is not None and last[0] == update
            checkpoint = PublishedTrainingCheckpoint(
                checkpoint_root=write.root, completed_updates=update, pointer=write.latest_pointer,
            )
            started = perf_counter()
            prepared = prepare_rollout(launch, checkpoint, params)
            preparation_ms = 1000 * (perf_counter() - started)

            started = perf_counter()
            lowered = prepared.scorer.lower(last[1], capture_state=prepared.layout.capture)
            compiled = lowered.compile()
            compilation_ms = 1000 * (perf_counter() - started)
            lowering_sha256 = sha256(lowered.as_text().encode()).hexdigest()
            samples = []
            for _ in range(3):
                started = perf_counter()
                result = jax.block_until_ready(compiled(last[1]))
                score_done = perf_counter()
                dataset = rollout_dataset(prepared, result, params, update_axis="update")
                with store._exclusive_lock():
                    product = publish_dataset(store, dataset)
                    ref = store.store_rooted_object(
                        store.get_object(product.object_ref), namespace="qualification-evaluate-dataset",
                    )
                events.append("qualification_evaluate_point", {
                    "request": request_id, "checkpoint": checkpoint.model_dump(mode="json"),
                    "product": product.model_dump(mode="json"), "ref": ref.model_dump(mode="json"),
                }, sync=True)
                samples.append({
                    "score_ms": 1000 * (score_done - started),
                    "durable_ms": 1000 * (perf_counter() - started),
                })
            events.append("qualification_evaluate_timing", {
                "preparation_ms": preparation_ms, "compilation_ms": compilation_ms,
                "lowering_sha256": lowering_sha256, "samples": samples,
                "trials": len(params.trials), "transitions": prepared.paired.execution.task.n_steps,
            }, sync=True)

        result = execute(spec, **(kwargs | {
            "committed_update_callback": on_update, "checkpoint_published": on_checkpoint,
        }))
        events.append("qualification_driver_timing", {"window_ms": 1000 * (perf_counter() - start)})
        return result

    with patch.object(executor, "execute_managed_training_run", measured):
        return train(argv)
