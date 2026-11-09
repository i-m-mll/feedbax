"""Prepare an exact staged Studio training row for the local orchestration engine."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from platform import python_version

from feedbax.contracts.artifact_custody import ImmutableArtifactBlobProviderSpec
from feedbax.contracts.checkpoints import CheckpointContinuationRequest
from feedbax.contracts.execution_identity import ImmutableInputIdentity
from feedbax.contracts.manifest import (
    ParentRef,
    SpecPayload,
    TrainingRunAxisCoordinate,
    TrainingRunManifest,
    write_manifest,
)
from feedbax.contracts.run_matrix import TrainingRowProvenance
from feedbax.contracts.spec_storage import (
    canonical_training_run_spec_projection,
    store_canonical_json_artifact,
    training_spec_sha256,
)
from feedbax.contracts.training import TrainingRunSpec
from feedbax.orchestration.assembly import AssemblyContext, persist_compiled_row
from feedbax.orchestration.bundle import (
    BudgetPolicy,
    CheckpointCustodyArchiveMaterializer,
    DeploymentPolicy,
    EnvironmentDeclaration,
    ImmutableInputArtifactRef,
    InputCustodySource,
    InputFormatIdentity,
    ResolvedAssemblyInput,
    RunBundle,
    RunRowSpec,
    SchemaArtifactRef,
    StagedTrainingBinding,
    default_orchestration_root,
    mint_run_set_id,
)
from feedbax.orchestration.drivers.capabilities import DriverConstructionContext
from feedbax.orchestration.input_materialization import InputProviderRootBinding
from feedbax.orchestration.revision import resolve_feedbax_revision
from feedbax.orchestration.staged_training import (
    StagedTrainingConflict,
    load_staged_training_receipt,
    require_unclaimed_training_receipt,
    staging_generation,
)
from feedbax.orchestration.stages import StageEngine
from feedbax.plugins.application import ApplicationRegistryBundle
from feedbax.plugins.bootstrap import PluginProvenance
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.training.checkpoint_custody import (
    authenticate_published_checkpoint_custody,
    produce_checkpoint_custody_archive,
)
from feedbax.training.interruption import CancellationDecision
from feedbax.training.spec_storage import TrainingRunIdentityAdapter, compile_native_training_row


STAGED_NATIVE_ROW_KEY = "staged_native_row"
CHECKPOINT_PROVIDER = "studio.checkpoint.inputs"


def prepare_native_training_dispatch(
    manifest: TrainingRunManifest,
    *,
    root: Path,
    generation: str,
    registry_bundle: ApplicationRegistryBundle,
    plugin_registrations: tuple[PluginProvenance, ...] = (),
    interruption_probe: Callable[[], CancellationDecision | None] | None = None,
    checkpoint: ResolvedAssemblyInput | None = None,
) -> StageEngine:
    """Build one local StageEngine whose PROVISION must claim the exact receipt."""
    require_unclaimed_training_receipt(manifest)
    if staging_generation(manifest.metadata) != generation:
        raise StagedTrainingConflict("Training staging generation changed. Refresh the queue.")
    if manifest.metadata.get("planned") is not True:
        raise ValueError("Launch requires a staged Studio training receipt.")
    if manifest.metadata.get("execution_target") != "local":
        raise ValueError("Only local staged training runs can be launched from Studio.")
    staged_spec = manifest.training_spec
    if staged_spec is None or staged_spec.kind != "TrainingRunSpec":
        raise ValueError("Launch requires a native TrainingRunSpec matrix row.")
    raw_native_row = manifest.metadata.get(STAGED_NATIVE_ROW_KEY)
    if raw_native_row is None:
        raise ValueError("Restage this matrix to capture native row provenance.")
    provenance = TrainingRowProvenance.model_validate(raw_native_row)
    coordinate = TrainingRunAxisCoordinate.model_validate(provenance.axis_coordinates)
    if (
        provenance.planned_run_id != manifest.id
        or coordinate.run_id != manifest.id
        or coordinate.index != provenance.row_index
        or provenance.lowered_execution_payload_hash != training_spec_sha256(staged_spec.inline)
    ):
        raise StagedTrainingConflict("Staged training row provenance does not match its receipt.")

    spec = TrainingRunSpec.model_validate(staged_spec.inline)
    registry_bundle.training_programs.validate_payload(
        spec.method_ref, spec.method_payload, path="/method_payload"
    )
    if (spec.checkpoint_progress.continuation is None) != (checkpoint is None):
        raise ValueError("Native continuation and its authenticated checkpoint input must agree")
    payload = canonical_training_run_spec_projection(spec)
    run_set_id = mint_run_set_id()
    revision = resolve_feedbax_revision()
    run_set_root = default_orchestration_root(run_set_id).resolve()
    custody_root = run_set_root.parent / "custody"
    row = compile_native_training_row(
        payload,
        provenance=provenance,
        coordinate=coordinate,
        overrides=provenance.overrides,
        run_set_id=run_set_id,
        immutable_inputs=(checkpoint.identity,) if checkpoint is not None else (),
        plugin_modules=tuple(
            source.entry_point_name
            for source in plugin_registrations
            if source.distribution is None
            and source.entry_point_value == f"{source.entry_point_name}:PLUGIN_REGISTRATION"
        ),
    )
    authored = store_canonical_json_artifact(
        payload,
        root=custody_root,
        role="authored_intent",
        logical_name=f"{row.row_id}.staged.json",
    )
    stored = persist_compiled_row(
        row,
        authored=payload,
        identity_adapter=TrainingRunIdentityAdapter(),
        context=AssemblyContext(
            custody_root=custody_root,
            materializer_commit=revision,
            authored_ref=SchemaArtifactRef(
                schema_id=spec.schema_id,
                schema_version=spec.schema_version,
                artifact_id=authored.artifact_id,
                sha256=authored.sha256,
                uri=authored.uri,
            ),
        ),
    )
    bundle = RunBundle(
        run_set_id=run_set_id,
        feedbax_revision=revision,
        deployment_policy=DeploymentPolicy(
            driver="local",
            venue="local",
            cloud_authorized=False,
            review_required=False,
            review_authorized=False,
        ),
        environment=EnvironmentDeclaration(python_version=python_version()),
        budget=BudgetPolicy(max_wall_clock_seconds=3600),
        rows=[RunRowSpec(row_id=stored.row_id, execution=stored.execution, launch=stored.launch)],
        resolved_inputs=[checkpoint] if checkpoint is not None else [],
        orchestration_root=str(run_set_root.parent),
        staged_training=StagedTrainingBinding(
            manifest_root=str(root.resolve()),
            manifest_id=manifest.id,
            generation=generation,
        ),
    )
    return native_training_engine(
        bundle,
        registry_bundle=registry_bundle,
        plugin_registrations=plugin_registrations,
        interruption_probe=interruption_probe,
    )


def stage_native_checkpoint_resume(
    bundle: RunBundle, receipt: TrainingRunManifest, terminal: TrainingRunManifest,
    checkpoints: Path,
) -> tuple[TrainingRunManifest, ResolvedAssemblyInput | None]:
    """Reuse one continuation receipt for a stopped attempt and its exact checkpoint."""
    assert bundle.staged_training is not None and receipt.training_spec is not None
    root = Path(bundle.staged_training.manifest_root)
    checkpoint = authenticate_published_checkpoint_custody(checkpoints)
    parent = terminal.terminal_checkpoint
    completed = checkpoint.manifest.completed_training_batches
    if not isinstance(parent, ParentRef) or parent != checkpoint.parent_ref or completed != terminal.completed_batches:
        raise ValueError("Stopped training result does not identify its authenticated checkpoint")
    spec = TrainingRunSpec.model_validate(receipt.training_spec.inline)
    source = spec.checkpoint_progress.continuation
    target = (source.source_completed_batches if source is not None else 0) + spec.training_config.n_batches
    if completed >= target:
        raise ValueError("Training has no batches remaining after this checkpoint")
    continuation = CheckpointContinuationRequest(
        source_completed_batches=completed, additional_batches=target - completed,
    )
    spec.training_config.n_batches = target - completed
    spec.checkpoint_progress.continuation = continuation
    payload = canonical_training_run_spec_projection(spec)
    digest = training_spec_sha256(payload)
    generation = sha256(f"{bundle.run_set_id}:{parent.id}:{digest}".encode()).hexdigest()
    manifest_id = f"training-resume-{generation}"
    try:
        successor, _ = load_staged_training_receipt(root, manifest_id)
    except FileNotFoundError:
        successor = None
    if successor is not None and successor.metadata.get("staged_attempt") is not None:
        return successor, None
    archive = produce_checkpoint_custody_archive(
        parent, allowed_root=checkpoints,
        artifact_provider=ImmutableArtifactBlobProvider(_checkpoint_provider_root(bundle)),
    )
    artifact, evidence = archive.artifact_ref, archive.evidence
    resolved = ResolvedAssemblyInput(
        identity=ImmutableInputIdentity(
            role="checkpoint", kind="checkpoint-custody-archive", identifier=parent.id,
            digest={"value": artifact.sha256},
        ),
        custody=InputCustodySource(
            target_role="checkpoint",
            provider=ImmutableArtifactBlobProviderSpec(),
            provider_binding=CHECKPOINT_PROVIDER,
            artifact=ImmutableInputArtifactRef(
                **artifact.model_dump(include={
                    "artifact_id", "sha256", "size_bytes", "media_type", "storage_backend",
                }),
            ),
            format=InputFormatIdentity(
                format_id=evidence.schema_id, format_version=evidence.schema_version,
                media_type=evidence.media_type,
            ),
            materializer=CheckpointCustodyArchiveMaterializer(
                expected_parent_ref=parent,
                expected_transaction_root_sha256=evidence.transaction_root_sha256,
            ),
        ),
    )
    if successor is None:
        old = TrainingRowProvenance.model_validate(receipt.metadata[STAGED_NATIVE_ROW_KEY])
        provenance = old.model_copy(update={
            "row_id": manifest_id,
            "planned_run_id": manifest_id,
            "axis_coordinates": old.axis_coordinates | {"run_id": manifest_id},
            "authored_payload_hash": digest,
            "lowered_execution_payload_hash": digest,
        })
        successor = TrainingRunManifest(
            id=manifest_id,
            status="pending",
            graph_spec=receipt.graph_spec,
            training_spec=SpecPayload(kind="TrainingRunSpec", inline=payload),
            task_spec=receipt.task_spec,
            task_binding_spec=receipt.task_binding_spec,
            metadata={
                "name": f"Resume {receipt.metadata.get('name') or receipt.id}",
                "planned": True,
                "staging_generation": generation,
                "execution_target": "local",
                "resumed_from": {
                    "manifest_id": receipt.id, "run_set_id": bundle.run_set_id,
                    "checkpoint": parent.model_dump(mode="json"),
                },
                STAGED_NATIVE_ROW_KEY: provenance.model_dump(mode="json"),
            },
        )
        write_manifest(successor, root=root)
    return successor, resolved


def _checkpoint_provider_root(bundle: RunBundle) -> Path:
    return bundle.run_set_dir.parent / "custody" / "checkpoint-inputs"


def native_training_engine(
    bundle: RunBundle,
    *,
    registry_bundle: ApplicationRegistryBundle,
    plugin_registrations: tuple[PluginProvenance, ...],
    interruption_probe: Callable[[], CancellationDecision | None] | None,
) -> StageEngine:
    """Construct supervision for an existing exact local native bundle."""
    driver = registry_bundle.drivers.construct(
        bundle.deployment_policy.driver,
        DriverConstructionContext(
            configuration={"bundle": bundle, "preserve_owned_resources": bundle.keep_alive},
            runtime_bindings={
                "input_provider_bindings": (
                    InputProviderRootBinding(CHECKPOINT_PROVIDER, _checkpoint_provider_root(bundle)),
                ),
            } if bundle.resolved_inputs else None,
        ),
    )
    return StageEngine(
        bundle=bundle,
        driver=driver,
        training_method_registry=registry_bundle.training_programs,
        conformance_registry=registry_bundle.conformance_checks,
        plugin_provenance=plugin_registrations,
        interruption_probe=interruption_probe,
    )
