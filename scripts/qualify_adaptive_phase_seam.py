"""Reproduce the authored adaptive suffixes and the current public archive refusal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from feedbax.contracts.checkpoints import CheckpointContinuationRequest
from feedbax.contracts.experiment_envelope import ExperimentEnvelopeParentAuthority
from feedbax.contracts.manifest import ParentRef
from feedbax.contracts.project_experiment import load_project_declaration
from feedbax.contracts.run_composition import ResolvedOutputParent
from feedbax.contracts.run_matrix import AuthoredTrainingRow, TrainingRowParentProvenance
from feedbax.contracts.spec_storage import training_spec_sha256
from feedbax.contracts.training import TrainingProgramRegistry
from feedbax.envelope.entrypoint import kernel_for
from feedbax.persistence import ImmutableArtifactBlobProvider
from feedbax.training import (
    CheckpointReferenceResolutionError,
    materialize_checkpoint_custody_archive,
)
from feedbax.training.authoring import compile_training_method_authoring
from feedbax.training.optimizers import learning_rate_at_step
from feedbax.training.row_lowering import GovernedTrainingRowParent, TrainingRowLoweringContext
from feedbax.training.run_matrix import materialize_adapted_run_matrix


def _authored_rows(downstream: Path, parent_bytes: bytes, source_batch: int) -> dict:
    from rlrmp2.training.method_payloads import MappedAdaptiveLambdaMethodPayload
    from rlrmp2.training.nominal import (
        NOMINAL_BASELINE_PAYLOAD_SCHEMA_ID,
        NOMINAL_BASELINE_PAYLOAD_SCHEMA_VERSION,
    )
    from rlrmp2.training.programs import mapped_adaptive_lambda_training_program

    envelopes = downstream / "specs/experiment"
    mapped = json.loads((envelopes / "mapped-fork-continuation.envelope.json").read_bytes())
    parent = ResolvedOutputParent.model_validate(mapped["training"]["root"]["parent"])
    provenance = TrainingRowParentProvenance(
        role="mapped_fork_training_parent",
        parent_kind=parent.kind,
        ref=parent.ref,
        semantic_hash=parent.resolved_root_hash,
        artifact_id=parent.ref,
        artifact_sha256=parent.resolved_root_hash,
        schema_id=NOMINAL_BASELINE_PAYLOAD_SCHEMA_ID,
        schema_version=NOMINAL_BASELINE_PAYLOAD_SCHEMA_VERSION,
    )
    authority = ExperimentEnvelopeParentAuthority(
        parent=parent, provenance=provenance, payload_bytes=parent_bytes
    )
    compiler = kernel_for(load_project_declaration(downstream), parent_authorities=(authority,))
    outcome = compiler.compile_envelope_file(
        envelopes / "applied-damage-ema-continuations.envelope.json", repo_root=downstream
    )
    rows = materialize_adapted_run_matrix(
        outcome.document,
        repo_root=downstream,
        # Materialize authored rows here. The program compiler below validates executable specs.
        row_validator=lambda _payload, _row_id: None,
        row_lowering_context=TrainingRowLoweringContext(
            (
                GovernedTrainingRowParent(
                    parent=parent, provenance=provenance, payload_bytes=parent_bytes
                ),
            )
        ),
    )
    registry = TrainingProgramRegistry()
    program = mapped_adaptive_lambda_training_program()
    registry.register_program(program)
    selected = {}
    for index, row in enumerate(rows.rows):
        if row.row_id not in {"flat_3e-5", "rewarm_3e-3"}:
            continue
        payload = MappedAdaptiveLambdaMethodPayload.model_validate(row.payload)
        assert row.seed in (None, payload.training.batch_seed)
        assert payload.clamp.measurement_basis == "training_batch"
        assert payload.training.training_diagnostics.measurement_basis == "training_batch"
        compiled = compile_training_method_authoring(
            AuthoredTrainingRow(
                row_id=row.row_id,
                row_index=index,
                payload=row.payload,
                payload_hash=training_spec_sha256(row.payload),
                seed=row.seed,
                axis_coordinates={},
            ),
            method_ref=program.method_ref,
            continuation=CheckpointContinuationRequest(
                source_completed_batches=source_batch,
                additional_batches=payload.training.n_batches,
            ),
            registry=registry,
        ).run_spec
        continuation = compiled.checkpoint_progress.continuation
        assert continuation is not None and continuation.source_completed_batches == source_batch
        schedule = payload.training.optimizer.lr_schedule
        assert schedule is not None and schedule.origin.kind == "segment_start"
        sample_batches = (source_batch, source_batch + 1000, source_batch + 3500)
        selected[row.row_id] = {
            "authored_payload_sha256": training_spec_sha256(row.payload),
            "executable_spec_sha256": training_spec_sha256(compiled.model_dump(mode="json")),
            "continuation": continuation.model_dump(mode="json"),
            "schedule": schedule.model_dump(mode="json", exclude_none=True),
            "authored_schedule_origin_batch": source_batch,
            "projected_learning_rates": {
                str(batch): float(
                    learning_rate_at_step(
                        schedule, current_step=batch, schedule_origin_step=source_batch
                    )
                )
                for batch in sample_batches
            },
            "batch_seed": payload.training.batch_seed,
            "authored_trial_batch_size": payload.training.batch_size,
            "authored_replica_count": payload.training.mapped_axis.size,
            "measurement_basis": payload.clamp.measurement_basis,
        }
    assert selected.keys() == {"flat_3e-5", "rewarm_3e-3"}
    flat, rewarm = selected["flat_3e-5"], selected["rewarm_3e-3"]
    assert flat["executable_spec_sha256"] != rewarm["executable_spec_sha256"]
    assert flat["schedule"]["kind"] == "constant"
    assert rewarm["schedule"]["kind"] == "warmup_cosine"
    return {
        "governed_parent_sha256": parent.resolved_root_hash,
        "authored_target_only_transform": mapped["training"]["root"]["selected_checkpoint"][
            "slot_transforms"
        ][0],
        "suffixes": selected,
    }


def _archive_refusal(archive: Path, source: dict, scratch: Path) -> dict:
    parent = ParentRef.model_validate(source["parent_ref"])
    provider = ImmutableArtifactBlobProvider(scratch / "provider")
    artifact = provider.store_bytes(
        archive.read_bytes(),
        role="training_checkpoint_custody_archive",
        logical_name=f"{parent.id}.checkpoint-custody.tar.gz",
        media_type="application/vnd.feedbax.training-checkpoint-custody.v1+tar+gzip",
        metadata={
            "schema_id": "feedbax.archive.training_checkpoint_custody",
            "schema_version": "feedbax.archive.training_checkpoint_custody.v1",
            "transaction_root_sha256": source["transaction_root_sha256"],
        },
    )
    assert artifact.sha256 == source["archive_sha256"] and artifact.size_bytes == 5_728_278
    destination = scratch / "source"
    try:
        materialize_checkpoint_custody_archive(
            provider,
            artifact,
            destination,
            expected_parent_ref=parent,
            expected_transaction_root_sha256=source["transaction_root_sha256"],
        )
    except CheckpointReferenceResolutionError as error:
        detail = str(error)
        assert "slot 'model' uses unsupported serializer" in detail, error
        assert "pickle.v1" in detail and "pickle.v3" in detail, error
        assert not destination.exists()
        return {
            "archive_sha256": artifact.sha256,
            "archive_size_bytes": artifact.size_bytes,
            "expected_source_transaction": parent.id,
            "expected_source_manifest_sha256": parent.metadata["manifest_sha256"],
            "expected_source_transaction_root_sha256": source["transaction_root_sha256"],
            "expected_source_batch": source["completed_batches"],
            "refusal_type": type(error).__name__,
            "refusal_detail": detail,
            "source_destination_published": False,
        }
    raise AssertionError("the exact legacy source unexpectedly materialized on this producer")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downstream", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    downstream = args.downstream.resolve()
    receipt = (
        downstream / "results/training/mapped_fork_continuation.checkpoint_custody_receipt.json"
    )
    source = json.loads(receipt.read_bytes())["source_checkpoint"]
    with TemporaryDirectory(prefix="feedbax-adaptive-phase-seam-") as scratch:
        evidence = _authored_rows(downstream, args.parent.read_bytes(), source["completed_batches"])
        evidence["source_archive"] = _archive_refusal(args.archive, source, Path(scratch).resolve())
    print(json.dumps(evidence, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
