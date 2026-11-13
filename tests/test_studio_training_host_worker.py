"""Numerical continuation and live metrics through the real Studio host worker."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import replace
from pathlib import Path
from shutil import copytree
import subprocess
import sys
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest

from feedbax.contracts.checkpoints import CheckpointLatestPointer
from feedbax.contracts.graph import GraphSpec, STUDIO_TASK_TIMELINE_SCHEMA_VERSION
from feedbax.contracts.manifest import canonical_manifest_path, load_manifest, training_run_manifest_id
from feedbax.contracts.request_specs import DocumentTrainingRequestSpec
from feedbax.contracts.spec_storage import canonical_training_run_spec_sha256
from feedbax.contracts.studio_training import StudioGraphMethodPayload
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.runtime._graph_key_schedule import derive_key
from feedbax.runtime.parameter_constraints import apply_parameter_constraints
from feedbax.studio.canonical_train import lower_canonical_train
from feedbax.tasks.document import TaskParameters
from feedbax.training.checkpoint_custody import (
    authenticate_published_checkpoint_custody,
    load_checkpoint_custody_documents,
    materialize_checkpoint_custody_archive,
    open_published_model_slot,
    produce_checkpoint_custody_archive,
    require_checkpoint_semantic_state_equal,
)
from feedbax.training.model_checkpoint import MODEL_TREE_MEDIA_TYPE
from feedbax.training.optimizers import build_optimizer
from feedbax.training.standard_train_checkpoint import STANDARD_TRAIN_STATE_MEDIA_TYPE
from feedbax.training.studio_executor import StudioUpdateKernel, studio_training_key
from tests.support.managed_train_fixture import COMMAND_ADDRESS, managed_train_payload
from web.e2e.studio.fixtures.reaching_project import reaching_project, reaching_train_request


_STATE = ("model", "optimizer", "prng", "batch_counter")
_SCIENTIFIC_METRICS = ("train_loss", "loss_terms", "grad_norm")
_PROCESS_ENV = (
    "PATH", "TMPDIR", "LANG", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
    "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)
_SOURCE = Path(__file__).absolute().parents[1]


def _array_digests(tree):
    return [
        hashlib.sha256(array.tobytes()).hexdigest()
        for array in map(np.asarray, filter(eqx.is_array, jax.tree.leaves(tree)))
    ]


def _readout(graph):
    return graph.nodes["readout"].layer.weight


class _TrialEndpoint(NamedTuple):
    predicted: jax.Array
    target: jax.Array
    hold_steps: jax.Array


def _trial_endpoints(execution, seed):
    run_key = studio_training_key(seed, "task-run")

    @eqx.filter_jit
    def scalar_trial(model, key):
        trial = execution.task.ahead(TaskParameters(), run_key, key)
        retained, _ = execution.rollout(model, trial, derive_key(key, "trial-model"))
        command = retained[COMMAND_ADDRESS][-1]
        return _TrialEndpoint(
            command, jnp.zeros_like(command), trial.epochs.ends[0] - trial.epochs.starts[0],
        )

    return scalar_trial


def _reference_trials(batch_size, scalar_trial, slots, result, *, hold_bounds):
    next_key, step_key = jr.split(slots["prng"])
    np.testing.assert_array_equal(result["prng"], next_key)
    assert int(result["batch_counter"]) == int(slots["batch_counter"]) + 1
    keys = [step_key]
    keys.extend(
        derive_key(step_key, "studio_training", "trial", index=i)
        for i in range(1, batch_size)
    )
    values = [scalar_trial(slots["model"], key) for key in keys]
    predicted, targets, hold_steps = map(np.asarray, zip(*values, strict=True))
    minimum_hold, maximum_hold = hold_bounds
    assert np.all((hold_steps >= minimum_hold) & (hold_steps <= maximum_hold))
    assert len(set(hold_steps)) > 1
    endpoint_losses = np.sum(np.square(predicted.astype(np.float64) - targets), axis=1)
    reference_mean = math.fsum(endpoint_losses) / len(keys)
    assert result["train_loss"] == pytest.approx(reference_mean, rel=1e-5, abs=1e-8)
    (term,) = result["loss_terms"].values()
    assert term == pytest.approx(reference_mean, rel=1e-5, abs=1e-8)
    return keys, {
        "trial_keys": [np.asarray(jr.key_data(key)).tolist() for key in keys],
        "hold_steps": hold_steps.tolist(),
        "endpoint_predictions": predicted.tolist(),
        "endpoint_targets": targets.tolist(),
        "per_trial_endpoint_losses": endpoint_losses.tolist(),
        "independent_mean": reference_mean,
        "mean_absolute_error": abs(reference_mean - result["train_loss"]),
    }


def _reference_gradient(trainable_filter, method, scalar_trial, slots, result, keys, grad_clip):
    trainable, frozen = eqx.partition(slots["model"], trainable_filter)

    def endpoint_error(candidate, key):
        result = scalar_trial(eqx.combine(candidate, frozen), key)
        return jnp.sum(jnp.square(result.predicted - result.target))

    gradient = eqx.filter_jit(eqx.filter_grad(endpoint_error))
    trial_grads = [gradient(trainable, key) for key in keys]
    grads = jax.tree.map(lambda *values: sum(values) / len(keys), *trial_grads)
    grad_norm = float(optax.global_norm(grads))
    assert result["grad_norm"] == pytest.approx(grad_norm, rel=1e-4, abs=1e-8)

    optimizer = build_optimizer(
        method.optimizer, schedule_origin_step=0, current_step=0,
        gradient_clip=grad_clip,
    )
    updates, opt_state = optimizer.update(grads, slots["optimizer"], trainable)
    model = apply_parameter_constraints(
        eqx.combine(eqx.apply_updates(trainable, updates), frozen)
    )
    errors = {}
    for name, expected in (("model", model), ("optimizer", opt_state)):
        actual, reference = jax.tree.map(
            lambda value: np.asarray(value) if eqx.is_array(value) else value,
            (result[name], expected),
        )
        assert jax.tree.structure(actual) == jax.tree.structure(reference)
        differences = []
        for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(reference), strict=True):
            if isinstance(left, np.ndarray):
                np.testing.assert_allclose(left, right, rtol=1e-4, atol=2e-7)
                if left.size:
                    differences.append(float(np.max(np.abs(left - right))))
        errors[name] = max(differences)
    return {"grad_norm": grad_norm, "max_absolute_state_error": errors}


def _observe_worker(
    audit_path: Path, stop_after: int, distinct_preparation: bool, checkpoint_profile: str,
) -> int:
    from feedbax import training_host_worker
    from feedbax.compiler.trainability import compile_trainability
    from feedbax.compiler.trial_execution import PreparedTrialExecution, compile_trial_execution
    from feedbax.training import checkpoint_custody, executor, managed_studio

    prepare_run = managed_studio.prepare_managed_training
    update = StudioUpdateKernel.reported_update
    execute = executor.execute_managed_training_run
    prepared_model = trainable_filter = scalar_trial = method = config = None
    pending_evidence = None
    slot_reads = {}
    read_blob = checkpoint_custody._read_blob

    def record_slot_read(slot, path, *, read_bytes=Path.read_bytes):
        slot_reads[slot.slot] = slot.media_type
        return read_blob(slot, path, read_bytes=read_bytes)

    def prepare(spec, registries, *, model_assets=None):
        nonlocal prepared_model, trainable_filter, scalar_trial, method, config
        preparation = prepare_run(spec, registries, model_assets=model_assets)
        documents = spec.trial_documents
        assert documents is not None
        config = spec.training_config
        method = StudioGraphMethodPayload.model_validate(spec.method_payload.payload)
        prepared_model = preparation.initial_slots["model"]
        graph = GraphSpec.model_validate(spec.graph.inline)
        paths, trainable_filter = compile_trainability(graph, prepared_model, registries.components)
        assert paths == ("cell",)
        if distinct_preparation:
            prepared_model = eqx.tree_at(_readout, prepared_model, _readout(prepared_model) + 0.25)
            preparation = replace(preparation, initial_slots=dict(preparation.initial_slots) | {"model": prepared_model})
        if config.batch_size > 1:
            pair = compile_trial_execution(
                documents.model_name, documents.model, documents.task_source,
                registries.components, registries.expression_functions,
                python_tasks=registries.python_tasks, objective=documents.objective,
                input_overrides=documents.input_overrides, artifact_provider=model_assets,
            )
            assert isinstance(pair, PreparedTrialExecution), pair
            scalar_trial = _trial_endpoints(pair.execution, method.seed)
        return preparation

    def observe(kernel, slots):
        nonlocal pending_evidence
        model = slots["model"]
        result = update(kernel, slots)
        assert np.any(np.asarray(_readout(model)) != 0)
        assert _array_digests(model.nodes["cell"]) != _array_digests(result["model"].nodes["cell"])
        evidence = {
            "pid": os.getpid(),
            "update": int(result["batch_counter"]),
            "before": {name: _array_digests(slots[name]) for name in _STATE},
            "after": {name: _array_digests(result[name]) for name in _STATE},
            "prepared_readout": _array_digests(_readout(prepared_model)),
            "input_readout": _array_digests(_readout(model)),
            "output_readout": _array_digests(_readout(result["model"])),
            "slot_reads": dict(slot_reads),
            "cursor_before": int(slots["batch_counter"]),
            "optimizer_count_before": int(slots["optimizer"][1][0].count),
            "metrics": {
                name: result[name] for name in (*_SCIENTIFIC_METRICS, "step_time_ms")
            },
        }
        if scalar_trial is not None:
            keys, reference = _reference_trials(
                config.batch_size, scalar_trial, slots, result, hold_bounds=(5, 14),
            )
            evidence["trial_reference"] = reference
            if int(slots["batch_counter"]) == 0 and not stop_after:
                evidence["gradient_reference"] = _reference_gradient(
                    trainable_filter, method, scalar_trial, slots, result, keys, config.grad_clip,
                )
        assert pending_evidence is None
        pending_evidence = evidence
        return result

    def stop_at_commit(*args, **kwargs):
        callback = kwargs["committed_update_callback"]

        def observe_commit(coordinate, completed, slots):
            nonlocal pending_evidence
            callback(coordinate, completed, slots)
            assert pending_evidence["update"] == completed
            assert pending_evidence["after"] == {
                name: _array_digests(slots[name]) for name in _STATE
            }
            with audit_path.open("a") as stream:
                stream.write(json.dumps(pending_evidence, allow_nan=False) + "\n")
            pending_evidence = None
            if completed == stop_after:
                Path(os.environ["FEEDBAX_HOST_STOP_FILE"]).touch()

        kwargs["committed_update_callback"] = observe_commit
        if checkpoint_profile != "standard_train":
            kwargs["train_state_codec"] = "legacy"
        if checkpoint_profile == "legacy":
            kwargs.pop("semantic_model")
        return execute(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(managed_studio, "prepare_managed_training", prepare)
        patch.setattr(StudioUpdateKernel, "reported_update", observe)
        patch.setattr(executor, "execute_managed_training_run", stop_at_commit)
        patch.setattr(checkpoint_custody, "_read_blob", record_slot_read)
        # Force an immediate first emission and one final partial interval.
        patch.setattr(training_host_worker._Events, "elapsed", lambda _events: 0.1)
        return training_host_worker.main([])


def _authored_request(*, batches=8, batch_size=1, sampled_hold=False):
    project = reaching_project(batches=batches)
    project.graph.nodes["readout"].params["trainable"] = False
    scenario = project.workspace.scenarios["scenario:train"]
    scenario.training_spec["batch_size"] = batch_size
    if sampled_hold:
        scenario.task_spec["timeline"] = {
            "schema_version": STUDIO_TASK_TIMELINE_SCHEMA_VERSION,
            "epochs": [
                {
                    "id": "hold", "label": "Hold", "index": 0,
                    "length": {
                        "mode": "distribution",
                        "distribution": {
                            "family": "uniform", "parameters": {"min": 1, "max": 81},
                        },
                    },
                },
                {
                    "id": "reach", "label": "Reach", "index": 1,
                    "length": {"mode": "constant"},
                },
            ],
            "signals": [
                {
                    "id": "reach_target", "label": "Reach target", "kind": "signal",
                    "path": "inputs.effector_target.pos",
                },
            ],
            "epoch_value_specs": [
                {
                    "target_id": "reach_target", "epoch_id": "hold",
                    "value_spec": {"mode": "constant", "value": [0.0, 0.0]},
                },
            ],
        }
    return reaching_train_request(project)


def _authored_spec(**options):
    request = _authored_request(**options)
    registry = new_application_registry_bundle()
    resolved = registry.training_programs.resolve_execution(request.method_ref, request.method_payload)
    from feedbax.contracts.training import TrainingRunSpec, WorkerExecutionSpec

    return TrainingRunSpec(
        graph={"inline": request.graph.model_dump(mode="json", exclude_none=True)},
        task=request.task, objective=request.objective, task_binding_spec=request.task_binding_spec,
        method_ref=request.method_ref, method_payload=request.method_payload,
        training_config=request.training_config,
        worker_execution=WorkerExecutionSpec(method_contract=resolved.contract, effective_phase=resolved.effective_phase),
        checkpoint_progress={"checkpoint_interval": request.training_config.snapshot_interval, "progress_interval": 1},
    )


def _managed_spec(*, batches=8, batch_size=1, sampled_hold=False):
    scenario = "delayed-reaches" if sampled_hold else "simple-reaches"
    payload = managed_train_payload(batches=batches, scenario=scenario)
    payload["training_config"]["batch_size"] = batch_size
    documents = payload["trial_documents"]
    documents["model"]["interior"]["components"]["readout"]["fields"]["trainable"] = False
    request = DocumentTrainingRequestSpec.model_validate(payload)
    registries = new_application_registry_bundle()
    return lower_canonical_train(request, registries.training_programs, registries=registries)


def test_retained_reaching_batch_matches_independent_endpoint_reduction():
    from feedbax.plugins.retained import new_retained_registry_bundle
    from feedbax.training.invocation import prepare_training_invocation
    from feedbax.web.worker import execution

    spec = _authored_spec(batches=4, batch_size=3, sampled_hold=True)
    registries = new_retained_registry_bundle()
    compile_run = execution.compile_training_run
    compiled = None

    def capture(*args, **kwargs):
        nonlocal compiled
        compiled = compile_run(*args, **kwargs)
        return compiled

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(execution, "compile_training_run", capture)
        preparation = prepare_training_invocation(
            spec, registries, run_id="retained-endpoint-control", resume=False,
        )
    assert compiled is not None

    @eqx.filter_jit
    def scalar_trial(model, key):
        rollout = execution.rollout_graph(model, compiled, key=key)
        targets = rollout["task_data"]["reach_target"]
        return _TrialEndpoint(
            rollout["trace"]["path:plant.effector.pos"][-1],
            targets[-1],
            jnp.count_nonzero(jnp.all(targets == 0, axis=-1)),
        )

    slots = preparation.initial_slots
    result = preparation.kernel_context["studio_graph_update"].reported_update(slots)
    method = StudioGraphMethodPayload.model_validate(spec.method_payload.payload)
    keys, reference = _reference_trials(3, scalar_trial, slots, result, hold_bounds=(1, 80))
    assert reference["independent_mean"] != pytest.approx(reference["per_trial_endpoint_losses"][0])
    assert reference["independent_mean"] != pytest.approx(sum(reference["per_trial_endpoint_losses"]))
    _reference_gradient(compiled.trainable_filter, method, scalar_trial, slots, result, keys, spec.training_config.grad_clip)


def _launch(
    root, job, spec, *, stop_after=0, resume=None, source=_SOURCE, fail=False,
    checkpoint_profile="standard_train",
):
    control = root / job
    control.mkdir()
    log = control / "events.jsonl"
    audit = control / "updates.jsonl"
    launch_input = control / "input.json"
    launch_input.write_text(json.dumps({
        "spec": spec.model_dump(mode="json"),
        "manifest_root": str(root / "runs"),
        "project_id": "frozen-readout",
        "request_id": canonical_training_run_spec_sha256(spec),
        "resume": resume,
    }))
    env = {name: os.environ[name] for name in _PROCESS_ENV if name in os.environ}
    env.update(
        PYTHONPATH=str(source),
        FEEDBAX_HOST_JOB_ID=job,
        FEEDBAX_HOST_INPUT=str(launch_input),
        FEEDBAX_HOST_EVENT_LOG=str(log),
        FEEDBAX_HOST_STOP_FILE=str(control / "stop"),
        FEEDBAX_HOST_SCRATCH=str(control),
        JAX_PLATFORMS="cpu",
        JAX_COMPILATION_CACHE_DIR=str(root / "jax"),
        JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0",
        JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="0",
        JAX_COMPILATION_CACHE_MAX_SIZE=str(32 * 1024 * 1024),
    )
    child = subprocess.run(
        [
            sys.executable, __file__, str(audit), str(stop_after), str(int(resume is not None)),
            checkpoint_profile,
        ],
        cwd=source, env=env, capture_output=True, text=True, timeout=30, check=False,
    )
    events = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    if child.returncode != int(fail):
        receipt_path = canonical_manifest_path("TrainingRunManifest", training_run_manifest_id(job), root=root / "runs")
        manifest = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        fields = {key: manifest.get(key) for key in ("id", "job_id", "status")}
        pytest.fail(
            f"{job}: worker exited {child.returncode}\n{child.stdout}{child.stderr}\n"
            f"Terminal event: {events[-1] if events else None}\nManifest: {fields}",
            pytrace=False,
        )
    assert events[-1]["kind"] == "result"
    result = events[-1]["data"]
    if fail:
        assert result["outcome"] == "failed"
        return result
    assert result["outcome"] == ("stopped" if stop_after else "succeeded")
    binding, = [event["data"] for event in events if event["kind"] == "binding"]
    assert binding["resume"] == resume
    checkpoint = result["checkpoint"]
    custody = authenticate_published_checkpoint_custody(checkpoint["checkpoint_root"])
    pointer = load_checkpoint_custody_documents(checkpoint["checkpoint_root"]).latest_pointer.document
    assert pointer.model_dump(mode="json") == checkpoint["pointer"]
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    receipt = canonical_manifest_path(
        "TrainingRunManifest", result["receipt"]["manifest_id"], root=root / "runs",
    )
    assert hashlib.sha256(receipt.read_bytes()).hexdigest() == result["receipt"]["manifest_sha256"]
    manifest = load_manifest(receipt)
    assert manifest.status == ("cancelled" if stop_after else "completed")
    assert manifest.completed_batches == checkpoint["completed_updates"]
    assert manifest.terminal_checkpoint.id == pointer.transaction_id
    assert manifest.terminal_checkpoint.metadata["manifest_sha256"] == pointer.manifest_sha256
    return checkpoint, custody.manifest, rows, events, receipt


def _require_reductions(reductions, samples):
    assert set(reductions) == set(samples[0])
    for name, reduction in reductions.items():
        values = [sample[name] for sample in samples]
        if isinstance(values[0], dict):
            _require_reductions(reduction, values)
        else:
            assert reduction == {
                "count": len(values),
                "mean": pytest.approx(math.fsum(values) / len(values), rel=1e-14),
                "min": min(values),
                "max": max(values),
                "last": values[-1],
            }


def _require_progress(events, rows, start):
    initial, first, final = [event["data"] for event in events if event["kind"] == "progress"]
    assert initial["completed_updates"] == start
    assert initial["metrics"] == {}
    assert "metric_interval" not in initial
    for event, updates in ((first, rows[:1]), (final, rows[1:])):
        interval = event["metric_interval"]
        assert interval["start_exclusive"] == updates[0]["update"] - 1
        assert interval["end_inclusive"] == event["completed_updates"] == updates[-1]["update"]
        assert event["metrics"] == updates[-1]["metrics"]
        _require_reductions(interval["metrics"], [row["metrics"] for row in updates])


def test_reaching_continues_checkpoint_frozen_readout_across_two_stops(tmp_path):
    from feedbax.contracts.studio_training import STUDIO_INDEPENDENT_KEY_SCHEDULE

    row = _managed_spec()
    assert row.method_payload.payload["key_schedule"] == STUDIO_INDEPENDENT_KEY_SCHEDULE
    legacy = row.model_copy(deep=True)
    del legacy.method_payload.payload["key_schedule"]
    assert canonical_training_run_spec_sha256(legacy) != canonical_training_run_spec_sha256(row)
    assert len({
        np.asarray(studio_training_key(0, stream)).tobytes()
        for stream in ("model", "update", "observation")
    } | {np.asarray(jr.PRNGKey(0)).tobytes()}) == 4

    _, reference, whole, whole_events, _ = _launch(tmp_path, "whole", row)
    first, first_manifest, prefix, prefix_events, first_receipt = _launch(
        tmp_path, "first", row, stop_after=3,
    )
    first_bytes = first_receipt.read_bytes()
    second, second_manifest, middle, middle_events, second_receipt = _launch(
        tmp_path, "second", row, resume=first, stop_after=6,
    )
    second_bytes = second_receipt.read_bytes()
    _, last_manifest, suffix, suffix_events, _ = _launch(tmp_path, "last", row, resume=second)

    assert [entry["update"] for entry in whole] == list(range(1, 9))
    assert whole[0]["before"]["prng"] == _array_digests(studio_training_key(0, "update"))
    assert [entry["update"] for entry in prefix] == [1, 2, 3]
    assert [entry["update"] for entry in middle] == [4, 5, 6]
    assert [entry["update"] for entry in suffix] == [7, 8]
    assert len({part[0]["pid"] for part in (whole, prefix, middle, suffix)}) == 4
    for expected, actual in zip(whole, prefix + middle + suffix, strict=True):
        assert actual["input_readout"] == expected["prepared_readout"]
        assert actual["output_readout"] == expected["prepared_readout"]
        assert actual["before"] == expected["before"]
        assert actual["after"] == expected["after"]
        assert actual["metrics"]["grad_norm"] > 0
        for name in _SCIENTIFIC_METRICS:
            assert actual["metrics"][name] == expected["metrics"][name]
    for updates in (middle, suffix):
        assert all(entry["prepared_readout"] != entry["input_readout"] for entry in updates)
        assert all(
            {
                name: media for name, media in entry["slot_reads"].items() if name != "model"
            } == dict.fromkeys(("optimizer", "prng", "batch_counter"), STANDARD_TRAIN_STATE_MEDIA_TYPE)
            for entry in updates
        )
        assert all(entry["cursor_before"] == entry["optimizer_count_before"] == entry["update"] - 1
                   for entry in updates)
        assert all("application/x-python-pickle" not in entry["slot_reads"].values() for entry in updates)

    assert second_manifest.parent_lineage[0].transaction_id == first_manifest.transaction_id
    assert last_manifest.parent_lineage[0].transaction_id == second_manifest.transaction_id
    for name in _STATE:
        require_checkpoint_semantic_state_equal(reference, last_manifest, name)
    model_refs = [
        next(slot for slot in manifest.slots if slot.slot == "model")
        for manifest in (first_manifest, reference, last_manifest)
    ]
    assert all(slot.media_type == MODEL_TREE_MEDIA_TYPE for slot in model_refs)
    assert model_refs[1].sha256 == model_refs[2].sha256
    assert reference.transaction_id != last_manifest.transaction_id
    assert model_refs[1].metadata == model_refs[2].metadata
    for manifest in (first_manifest, second_manifest, reference, last_manifest):
        assert all(
            slot.media_type == STANDARD_TRAIN_STATE_MEDIA_TYPE
            for slot in manifest.slots if slot.slot != "model"
        )

    model_only = copytree(first["checkpoint_root"], tmp_path / "model-only")
    pointer = CheckpointLatestPointer.model_validate(first["pointer"])
    transaction = (model_only / pointer.manifest_relative_path).parent
    for slot in first_manifest.slots:
        if slot.slot != "model":
            (transaction / slot.relative_path).unlink()
    (model_only / "latest.json").unlink()
    selected = open_published_model_slot(model_only, pointer)
    assert selected is not None
    assert _array_digests(selected.restore()) == prefix[-1]["after"]["model"]
    archive_store = ImmutableArtifactBlobProvider(tmp_path / "archive-store")
    first_root = Path(first["checkpoint_root"])
    parent_ref = authenticate_published_checkpoint_custody(first_root).parent_ref
    archive = produce_checkpoint_custody_archive(
        parent_ref, allowed_root=first_root, artifact_provider=archive_store,
    )
    unpacked = materialize_checkpoint_custody_archive(
        archive_store, archive.artifact_ref, tmp_path / "archived-stop",
        expected_parent_ref=parent_ref,
        expected_transaction_root_sha256=pointer.transaction_root_sha256,
    )
    assert {
        name: _array_digests(value) for name, value in unpacked.resolved_transaction.slots.items()
    } == prefix[-1]["after"]
    assert first_receipt.read_bytes() == first_bytes
    assert second_receipt.read_bytes() == second_bytes
    for events, updates, start in (
        (whole_events, whole, 0), (prefix_events, prefix, 0),
        (middle_events, middle, 3), (suffix_events, suffix, 6),
    ):
        _require_progress(events, updates, start)

    rejected = _launch(tmp_path, "wrong-binding", legacy, resume=first, fail=True)
    assert rejected["error"]["code"] == "incompatible_checkpoint"
    rejected = _launch(tmp_path, "old-seed-zero-start", legacy, fail=True)
    assert rejected["error"]["code"] == "invalid_request"
    assert "Create a new Studio request" in rejected["error"]["message"]

    for profile, legacy_slots in (
        ("hybrid", {"optimizer", "prng", "batch_counter"}), ("legacy", set(_STATE)),
    ):
        old, old_manifest, _, _, _ = _launch(
            tmp_path, f"{profile}-writer", row, stop_after=6, checkpoint_profile=profile,
        )
        assert {
            slot.slot for slot in old_manifest.slots
            if slot.media_type == "application/x-python-pickle"
        } == legacy_slots
        _, migrated, after_legacy, _, _ = _launch(
            tmp_path, f"{profile}-reader", row, resume=old,
        )
        assert {
            name for name, media in after_legacy[0]["slot_reads"].items()
            if media == "application/x-python-pickle"
        } == legacy_slots
        assert after_legacy[0]["before"] == whole[6]["before"]
        assert after_legacy[0]["after"] == whole[6]["after"]
        assert all(
            slot.media_type == (MODEL_TREE_MEDIA_TYPE if slot.slot == "model" else STANDARD_TRAIN_STATE_MEDIA_TYPE)
            for slot in migrated.slots
        )
        for name in _STATE:
            require_checkpoint_semantic_state_equal(reference, migrated, name)


def test_reaching_trial_batch_matches_independent_endpoint_reduction_and_resumes(tmp_path):
    row = _managed_spec(batches=4, batch_size=3, sampled_hold=True)
    assert row.training_config.batch_size == 3
    _, uninterrupted, whole, _, _ = _launch(tmp_path, "batch-whole", row)
    stopped, paused, prefix, _, receipt = _launch(
        tmp_path, "batch-paused", row, stop_after=2,
    )
    original_receipt = receipt.read_bytes()
    _, restarted, suffix, _, _ = _launch(tmp_path, "batch-resumed", row, resume=stopped)

    assert [entry["update"] for entry in prefix] == [1, 2]
    assert [entry["update"] for entry in suffix] == [3, 4]
    assert len({part[0]["pid"] for part in (whole, prefix, suffix)}) == 3
    for reference, actual in zip(whole, prefix + suffix, strict=True):
        for field in ("before", "after", "trial_reference"):
            assert actual[field] == reference[field]
        for name in _SCIENTIFIC_METRICS:
            assert actual["metrics"][name] == reference["metrics"][name]
    first = whole[0]["trial_reference"]
    assert len(first["trial_keys"]) == 3
    assert len({tuple(key) for key in first["trial_keys"]}) == 3
    assert first["independent_mean"] != pytest.approx(first["per_trial_endpoint_losses"][0])
    assert first["independent_mean"] != pytest.approx(sum(first["per_trial_endpoint_losses"]))
    assert "gradient_reference" in whole[0]
    assert all(entry["prepared_readout"] != entry["input_readout"] for entry in suffix)

    assert stopped["completed_updates"] == paused.completed_training_batches == 2
    assert restarted.parent_lineage[0].transaction_id == paused.transaction_id
    assert restarted.completed_training_batches == uninterrupted.completed_training_batches == 4
    for name in _STATE:
        require_checkpoint_semantic_state_equal(uninterrupted, restarted, name)
    assert receipt.read_bytes() == original_receipt

    scalar = _managed_spec(batches=4, sampled_hold=True)
    assert canonical_training_run_spec_sha256(scalar) != canonical_training_run_spec_sha256(row)
    rejected = _launch(tmp_path, "different-batch-size", scalar, resume=stopped, fail=True)
    assert rejected["error"]["code"] == "incompatible_checkpoint"


if __name__ == "__main__":
    raise SystemExit(_observe_worker(
        Path(sys.argv[1]), int(sys.argv[2]), bool(int(sys.argv[3])), sys.argv[4],
    ))
