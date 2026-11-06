"""Prepare an exact staged Studio training row for the local orchestration engine."""

from __future__ import annotations

from pathlib import Path
from platform import python_version

from feedbax.contracts.manifest import TrainingRunAxisCoordinate, TrainingRunManifest
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
    DeploymentPolicy,
    EnvironmentDeclaration,
    RunBundle,
    RunRowSpec,
    SchemaArtifactRef,
    StagedTrainingBinding,
    default_orchestration_root,
    mint_run_set_id,
)
from feedbax.orchestration.drivers.capabilities import DriverConstructionContext
from feedbax.orchestration.revision import resolve_feedbax_revision
from feedbax.orchestration.staged_training import (
    StagedTrainingConflict,
    require_unclaimed_training_receipt,
    staging_generation,
)
from feedbax.orchestration.stages import StageEngine
from feedbax.plugins.application import ApplicationRegistryBundle
from feedbax.plugins.bootstrap import PluginProvenance
from feedbax.training.spec_storage import TrainingRunIdentityAdapter, compile_native_training_row


STAGED_NATIVE_ROW_KEY = "staged_native_row"


def prepare_native_training_dispatch(
    manifest: TrainingRunManifest,
    *,
    root: Path,
    generation: str,
    registry_bundle: ApplicationRegistryBundle,
    plugin_registrations: tuple[PluginProvenance, ...] = (),
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
    if spec.checkpoint_progress.continuation is not None:
        raise ValueError(
            "Studio launch requires a fresh row. Native resume requires an explicit "
            "immutable checkpoint input from orchestration assembly."
        )
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
        orchestration_root=str(run_set_root.parent),
        staged_training=StagedTrainingBinding(
            manifest_root=str(root.resolve()),
            manifest_id=manifest.id,
            generation=generation,
        ),
    )
    driver = registry_bundle.drivers.construct(
        bundle.deployment_policy.driver,
        DriverConstructionContext(
            configuration={"bundle": bundle, "preserve_owned_resources": bundle.keep_alive},
        ),
    )
    return StageEngine(
        bundle=bundle,
        driver=driver,
        training_method_registry=registry_bundle.training_programs,
        conformance_registry=registry_bundle.conformance_checks,
        plugin_provenance=plugin_registrations,
    )
