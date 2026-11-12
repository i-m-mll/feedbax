"""Bounded current-format nominal and adaptive runs using the native executor."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import equinox as eqx
import jax
import numpy as np
import optax

from feedbax.contracts.base import sha256_file
from feedbax.contracts.checkpoints import (
    CheckpointContinuationRequest,
    CheckpointForkBarrierMapping,
    CheckpointForkHistoryPolicy,
    CheckpointForkPlan,
    CheckpointForkSourcePreparation,
    CheckpointForkTarget,
    CheckpointForkTransformRecord,
    CheckpointForkTransformStep,
    CheckpointLatestPointer,
)
from feedbax.contracts.manifest import (
    ParentRef,
    canonical_manifest_path,
    load_manifest,
    training_run_manifest_id,
)
from feedbax.contracts.run_matrix import AuthoredTrainingRow
from feedbax.contracts.spec_storage import training_spec_sha256
from feedbax.contracts.training import TrainingRunSpec
from feedbax.contracts.worker import ProgressCoordinate
from feedbax.plugins.composition import compose_application
from feedbax.training.authoring import compile_training_method_authoring
from feedbax.training.checkpoint_custody import (
    CheckpointCompatibilityError,
    CheckpointForkPlanBindings,
    CheckpointForkTransformRegistry,
    CheckpointReferenceResolutionError,
    derive_checkpoint_fork_compatibility_projection,
    fork_checkpoint_plan,
    load_checkpoint_custody_documents,
    require_checkpoint_semantic_state_equal,
    resolve_checkpoint_custody_ref,
)
from feedbax.training.executor import TrainingRunExecutorError, execute_training_run_spec
from feedbax.training.invocation import prepare_training_invocation
from feedbax.training.preparation import _thaw_runtime_value

from scripts.qualify_adaptive_phase_seam import governed_adaptive_rows


def _bounded(payload: dict, *, batches: int, hidden_dim: int = 180) -> dict:
    bounded = copy.deepcopy(payload)
    controller = bounded["task"]["controller"]
    if hidden_dim != controller["hidden_dim"]:
        controller["hidden_dim"] = hidden_dim
        bounded["task"]["topology"]["hidden_dim"] = hidden_dim
        for scale in ("input_scale", "recurrent_scale", "readout_scale"):
            controller[scale] = 1 / math.sqrt(hidden_dim)
    bounded["training"].update(
        n_batches=batches, batch_size=2, progress_interval=1, checkpoint_interval=1,
    )
    return bounded


def _compile(registry, method, row_id, payload, continuation=None):
    return compile_training_method_authoring(
        AuthoredTrainingRow(
            row_id=row_id, row_index=0, payload=payload,
            payload_hash=training_spec_sha256(payload), axis_coordinates={},
        ),
        method_ref=method, continuation=continuation, registry=registry,
    ).run_spec


def _successful_receipt(path, pointer, completed_batches) -> dict:
    receipt = load_manifest(path)
    assert receipt.status == "completed" and receipt.completed_batches == completed_batches
    terminal = receipt.terminal_checkpoint
    assert isinstance(terminal, ParentRef)
    assert terminal.id == pointer.transaction_id
    assert terminal.metadata["manifest_sha256"] == pointer.manifest_sha256
    return {"id": receipt.id, "sha256": sha256_file(path)}


def _receipt(result) -> dict:
    return _successful_receipt(
        result.manifest_path, result.checkpoint_writes[-1].latest_pointer, result.end_completed_batches,
    )


def _checkpoint_ref(write):
    pointer = write.latest_pointer
    return ParentRef(
        kind="TrainingCheckpointTransactionManifest", id=pointer.transaction_id,
        role="training_checkpoint_custody",
        uri=pointer.manifest_relative_path, metadata={"manifest_sha256": pointer.manifest_sha256},
    )


def _authenticated_slots(write):
    return resolve_checkpoint_custody_ref(_checkpoint_ref(write), allowed_root=write.root).slots


def _require_equal(expected, actual):
    assert jax.tree.structure(expected) == jax.tree.structure(actual)
    for left, right in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        lhs, rhs = np.asarray(left), np.asarray(right)
        assert lhs.dtype == rhs.dtype and lhs.shape == rhs.shape and lhs.tobytes() == rhs.tobytes()


def _adam_state(optimizer):
    def is_adam(value):
        return isinstance(value, optax.ScaleByAdamState)

    state, = filter(is_adam, jax.tree.leaves(optimizer, is_leaf=is_adam))
    return state


def _fork(spec, payload, prepared, source, destination):
    from rlrmp2.training.fork import (
        ADAPTIVE_BARRIER_MAPPING_IDENTITY,
        ADAPTIVE_BOOTSTRAP_BARRIER,
        ADAPTIVE_BOOTSTRAP_PHASE,
        MAPPED_ADAPTIVE_STATE_SLOT_IDENTITY,
        NOMINAL_TO_ADAPTIVE_TRANSFORM_ID,
        MappedNominalToAdaptiveTransformParameters,
        nominal_to_adaptive_transform_authority,
        register_checkpoint_fork_transforms,
    )

    phase = spec.worker_execution.method_contract.phase_program
    barrier = phase.checkpoint_barriers[0]
    assert barrier.name == ADAPTIVE_BOOTSTRAP_BARRIER
    prepared_templates = {slot.slot: prepared.initial_slots[slot.slot] for slot in barrier.slots}
    # Match the executor's mapping normalization at the public fork boundary.
    runtime_slots = _thaw_runtime_value(prepared.initial_slots)
    templates = {slot.slot: runtime_slots[slot.slot] for slot in barrier.slots}
    compatibility = derive_checkpoint_fork_compatibility_projection(spec, phase, prepared_templates)
    assert derive_checkpoint_fork_compatibility_projection(spec, phase, templates) == compatibility
    continuation = spec.checkpoint_progress.continuation
    parameters = MappedNominalToAdaptiveTransformParameters(
        initial_lambda=payload.initial_lambda, clamp=payload.clamp,
        replica_count=payload.training.mapped_axis.size,
        state_dtype=str(prepared.initial_slots["adaptive_state"].lambda_value.dtype),
    )
    source_manifest = source.manifest
    plan = CheckpointForkPlan(
        source=CheckpointForkSourcePreparation(
            checkpoint_root_ref="source",
            expected_transaction_id=source_manifest.transaction_id,
            expected_transaction_root_sha256=(
                source_manifest.content_integrity_digest.transaction_root_sha256
            ),
        ),
        targets=[CheckpointForkTarget(
            target_id="successor", checkpoint_root_ref="destination",
            run_spec_ref="run", slot_template_ref="slots",
            compatibility=compatibility,
            history_policy=CheckpointForkHistoryPolicy(
                mode="prepare_continuation", continuation_request=continuation,
            ),
            barrier_mapping=CheckpointForkBarrierMapping(
                source_barrier=source_manifest.barrier,
                target_barrier=ADAPTIVE_BOOTSTRAP_BARRIER,
                target_coordinate=ProgressCoordinate(
                    run_id=destination.name, phase=ADAPTIVE_BOOTSTRAP_PHASE,
                    completed_barrier=ADAPTIVE_BOOTSTRAP_BARRIER, program_step=0,
                    completed_batches=continuation.source_completed_batches,
                    schedule_origin_step=continuation.source_completed_batches,
                ),
                coordinate_mapping={"identity": ADAPTIVE_BARRIER_MAPPING_IDENTITY},
            ),
            transforms=[CheckpointForkTransformStep(
                step_id="add-adaptive-state", stage="target_post",
                target_only_slots={"adaptive_state": {
                    "identity": MAPPED_ADAPTIVE_STATE_SLOT_IDENTITY, "method_ref": spec.method_ref.key,
                }},
                records=[CheckpointForkTransformRecord(
                    slot="adaptive_state", identity=NOMINAL_TO_ADAPTIVE_TRANSFORM_ID,
                    parameters=parameters.model_dump(mode="json"), **nominal_to_adaptive_transform_authority(),
                )],
            )],
        )],
    )
    transforms = CheckpointForkTransformRegistry()
    register_checkpoint_fork_transforms(transforms)
    bindings = CheckpointForkPlanBindings(
        checkpoint_roots={"source": source.root, "destination": destination},
        run_specs={"run": spec}, slot_templates={"slots": templates},
    )
    return fork_checkpoint_plan(plan, bindings, transform_registry=transforms)["successor"]


def _committed_rows():
    evidence = []

    def observe(coordinate, completed, slots: Mapping):
        indices = np.asarray(slots["batch_indices"])
        assert indices.ndim == 2 and indices.shape[-1] == 2
        assert np.isfinite(np.asarray(slots["train_loss"])).all()
        evidence.append({
            "completed": completed,
            "schedule_origin": coordinate.schedule_origin_step,
            "adam_count": np.asarray(_adam_state(slots["optimizer"]).count).tolist(),
            "training_batch_indices": indices.tolist(),
            "applied_epsilon_shape": list(slots["applied_epsilon"].shape),
            "train_loss": np.asarray(slots["train_loss"]).tolist(),
            "learning_rate": np.asarray(slots["learning_rate"]).tolist(),
            "adaptive_lambda": np.asarray(slots["adaptive_state"].lambda_value).tolist(),
            "prng": np.asarray(slots["prng"]).tolist(),
        })

    return evidence, observe


def _registries():
    return asyncio.run(compose_application(modules=("rlrmp2.plugin",), entry_points=())).bundle


def _execute(registries, scratch, spec, run_id, **kwargs):
    return execute_training_run_spec(
        spec, run_id=run_id, registry=registries.training_programs,
        preparation=prepare_training_invocation(
            spec, registries, run_id=run_id, resume=kwargs.get("resume", False),
        ),
        manifest_root=scratch / "runs", checkpoint_root=scratch / run_id, **kwargs,
    )


def _restart(path: Path) -> None:
    request = json.loads(path.read_bytes())
    scratch = path.parent
    rows, observe = _committed_rows()
    try:
        result = _execute(
            _registries(), scratch, TrainingRunSpec.model_validate(request["spec"]), request["run_id"],
            resume=True, checkpoint_source_root=Path(request["checkpoint_root"]),
            checkpoint_source_pointer=CheckpointLatestPointer.model_validate(request["pointer"]),
            committed_update_callback=observe,
        )
    except (CheckpointCompatibilityError, TrainingRunExecutorError) as error:
        print(json.dumps({"refusal": str(error), "type": type(error).__name__, "updates": rows}))
        return
    print(json.dumps({"receipt": _receipt(result), "updates": rows}))


def qualify(
    downstream: Path, parent_bytes: bytes, scratch: Path, *, hidden_dim: int,
    prefix_root: Path | None, suffixes: list[str],
) -> dict:
    from rlrmp2.training.method_payloads import MappedAdaptiveLambdaMethodPayload
    from rlrmp2.training.nominal import NOMINAL_BASELINE_METHOD_REF
    from rlrmp2.training.programs import mapped_adaptive_lambda_training_program

    registries = _registries()
    programs = registries.training_programs
    _, _, rows = governed_adaptive_rows(downstream, parent_bytes)
    authored = {row.row_id: row.payload for row in rows.rows}
    source_batch = 2
    successor_batch = 2
    nominal = _bounded(json.loads(parent_bytes), batches=source_batch, hidden_dim=hidden_dim)
    # The independent variant explicitly trains the same encoder that mapped adaptive uses.
    nominal["task"]["task_variant"] = {"kind": "hold_free"}
    # The new prefix authors the suffix's float zero before creating optimizer state.
    source_optimizer = nominal["training"]["optimizer"]
    flat_optimizer = authored["flat_3e-5"]["training"]["optimizer"]
    rewarm_optimizer = authored["rewarm_3e-3"]["training"]["optimizer"]
    assert source_optimizer["type"] == flat_optimizer["type"] == rewarm_optimizer["type"]
    assert flat_optimizer["params"] == rewarm_optimizer["params"]
    source_optimizer["params"] = flat_optimizer["params"].copy()
    prefix = _compile(programs, NOMINAL_BASELINE_METHOD_REF, "current-prefix", nominal)

    if prefix_root is None:
        prefix_result = _execute(registries, scratch, prefix, "current-prefix")
        source = prefix_result.checkpoint_writes[-1]
        receipt = _receipt(prefix_result)
    else:
        root = prefix_root / "current-prefix"
        documents = load_checkpoint_custody_documents(root)
        source = SimpleNamespace(
            root=root, latest_pointer=documents.latest_pointer.document,
            manifest=documents.manifest.document,
        )
        path = canonical_manifest_path(
            "TrainingRunManifest", training_run_manifest_id("current-prefix"), root=prefix_root / "runs",
        )
        loaded = load_manifest(path)
        assert TrainingRunSpec.model_validate(loaded.training_spec.inline) == prefix
        receipt = _successful_receipt(path, source.latest_pointer, source_batch)
    assert source.manifest.completed_training_batches == source_batch
    source_slots = _authenticated_slots(source)
    wrong_ref = _checkpoint_ref(source).model_copy(update={"metadata": {"manifest_sha256": "0" * 64}})
    try:
        resolve_checkpoint_custody_ref(wrong_ref, allowed_root=source.root)
    except CheckpointReferenceResolutionError as error:
        substitution_refusal = str(error)
    else:
        raise AssertionError("checkpoint accepted under a substituted manifest hash")
    assert set(source_slots) == {"model", "optimizer", "prng", "batch_counter"}
    assert "h0_encoder" in source_slots["model"].nodes
    adam = _adam_state(source_slots["optimizer"])
    assert np.all(np.asarray(adam.count) == source_batch)
    assert any(np.count_nonzero(leaf) for leaf in jax.tree.leaves(adam.mu))

    continuation = CheckpointContinuationRequest(
        source_completed_batches=source_batch, additional_batches=successor_batch,
    )
    reports = {}
    for name in suffixes:
        row_id = "bounded-" + name
        raw = _bounded(authored[name], batches=successor_batch, hidden_dim=hidden_dim)
        payload = MappedAdaptiveLambdaMethodPayload.model_validate(raw)
        spec = _compile(
            programs, mapped_adaptive_lambda_training_program().method_ref, row_id, raw, continuation,
        )
        prepared = prepare_training_invocation(spec, registries, run_id=row_id, resume=True)
        bootstrap = _fork(spec, payload, prepared, source, scratch / f"{row_id}-bootstrap")
        boot_slots = _authenticated_slots(bootstrap)
        assert set(boot_slots) == set(source_slots) | {"adaptive_state"}
        for name_in_source in source_slots:
            require_checkpoint_semantic_state_equal(source.manifest, bootstrap.manifest, name_in_source)
        _require_equal(boot_slots["adaptive_state"], prepared.initial_slots["adaptive_state"])
        assert bootstrap.manifest.completed_training_batches == source_batch
        exact_source = {
            "resume": True, "checkpoint_source_root": bootstrap.root,
            "checkpoint_source_pointer": bootstrap.latest_pointer,
        }
        whole_rows, observe = _committed_rows()
        whole = _execute(
            registries, scratch, spec, row_id + "-whole",
            committed_update_callback=observe, **exact_source,
        )
        whole_receipt = _receipt(whole)
        assert [entry["completed"] for entry in whole_rows] == [source_batch + 1, source_batch + 2]
        for entry in whole_rows:
            assert entry["schedule_origin"] == source_batch
            assert set(entry["adam_count"]) == {entry["completed"]}
            assert entry["applied_epsilon_shape"][1] == 2
        assert any(len(set(indices)) == 2 for row in whole_rows for indices in row["training_batch_indices"])
        assert np.asarray(whole_rows[0]["prng"]).tolist() == np.asarray(
            jax.vmap(lambda key: jax.random.split(key)[0])(boot_slots["prng"])
        ).tolist()
        final_slots = _authenticated_slots(whole.checkpoint_writes[-1])
        for changed in ("model", "optimizer", "prng", "adaptive_state"):
            assert not bool(eqx.tree_equal(boot_slots[changed], final_slots[changed]))

        paused_rows, observe = _committed_rows()
        paused = _execute(
            registries, scratch, spec, row_id + "-paused",
            stop_before_update=lambda: len(paused_rows) == 1,
            committed_update_callback=observe, **exact_source,
        )
        paused_checkpoint = paused.checkpoint_writes[-1]
        assert paused.end_completed_batches == source_batch + 1
        assert paused.status == "cancelled"
        paused_receipt_hash = sha256_file(paused.manifest_path)
        restart_id = row_id + "-restart"
        restart_request = scratch / f"{restart_id}.json"
        restart_request.write_text(json.dumps({
            "spec": spec.model_dump(mode="json"), "run_id": restart_id,
            "checkpoint_root": str(paused_checkpoint.root),
            "pointer": paused_checkpoint.latest_pointer.model_dump(mode="json"),
        }))
        completed = subprocess.run(
            [sys.executable, __file__, "--restart", str(restart_request)],
            check=True, capture_output=True, text=True, timeout=30,
        )
        restarted = json.loads(completed.stdout)
        assert sha256_file(paused.manifest_path) == paused_receipt_hash
        equal_restart_slots = []
        if "refusal" in restarted:
            assert not restarted["updates"]
            assert not (scratch / restart_id / "latest.json").exists()
        else:
            assert paused_rows + restarted["updates"] == whole_rows
            restarted_manifest = load_checkpoint_custody_documents(scratch / restart_id).manifest.document
            for slot in boot_slots:
                require_checkpoint_semantic_state_equal(
                    whole.checkpoint_writes[-1].manifest, restarted_manifest, slot,
                )
                equal_restart_slots.append(slot)
        reports[name] = {
            "request_sha256": training_spec_sha256(spec.model_dump(mode="json")),
            "prefix_transaction": source.manifest.transaction_id,
            "bootstrap_transaction": bootstrap.manifest.transaction_id,
            "initialized_lambda": np.asarray(boot_slots["adaptive_state"].lambda_value).tolist(),
            "whole": whole_receipt, "restart": restarted,
            "pause_completed_batches": paused.end_completed_batches,
            "equal_restart_slots": sorted(equal_restart_slots), "updates": whole_rows,
        }
    assert len({report["request_sha256"] for report in reports.values()}) == len(reports)
    return {
        "current_prefix": receipt,
        "prefix_request_sha256": training_spec_sha256(prefix.model_dump(mode="json")),
        "prefix_completed_batches": source_batch,
        "prefix_checkpoint_manifest_substitution": substitution_refusal,
        "prefix_adam_count": np.asarray(adam.count).tolist(),
        "prefix_optimizer_params": source_optimizer["params"],
        "bounded_trial_batch": 2,
        "bounded_hidden_dim": hidden_dim,
        "successors": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downstream", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--hidden-dim", type=int, default=180)
    parser.add_argument("--prefix-root", type=Path)
    parser.add_argument("--suffix", choices=("flat_3e-5", "rewarm_3e-3"), action="append")
    parser.add_argument("--restart", type=Path)
    args = parser.parse_args()
    if args.restart is not None:
        _restart(args.restart)
        return
    if any(value is None for value in (args.downstream, args.parent, args.scratch)):
        parser.error("qualification requires --downstream, --parent and --scratch")
    evidence = qualify(
        args.downstream.resolve(), args.parent.read_bytes(), args.scratch.resolve(),
        hidden_dim=args.hidden_dim,
        prefix_root=args.prefix_root.resolve() if args.prefix_root else None,
        suffixes=args.suffix or ["flat_3e-5", "rewarm_3e-3"],
    )
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
