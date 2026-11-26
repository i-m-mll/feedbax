"""Publish fitted native weights and export retained model products."""

from feedbax.contracts.artifact_custody import ArtifactBlobCustodyError, ContentTreeRef
from feedbax.contracts.effective_success import success_content_ref
from feedbax.contracts.loaded_model import LOADED_MODEL_SCHEMA_ID, ModelFitReport
from feedbax.contracts.stored_model import STORED_MODEL_DOCUMENT_SCHEMA_ID, StoredModelDocument
from feedbax.contracts.model_checkpoint import SemanticModelEncoding
from feedbax.contracts.project import LoadedModelDocument
from feedbax.contracts.project_inputs import DocumentReference
from feedbax.contracts.project_resolution import ProjectDiagnostic, compose_document, composed_loaded_model_projection
from feedbax.contracts.studio_loaded_model import (
    StudioLoadedModelDoesNotFit, StudioLoadedModelExportRequest, StudioLoadedModelImported,
    StudioLoadedModelImportQuery, StudioLoadedModelImportResult, StudioLoadedModelUnreadableArchive,
)
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.host.protocol import HostError
from feedbax.jobs import plan_attempt_key
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView
from feedbax.persistence.method_checkpoint import read_method_checkpoint_manifest
from feedbax.persistence.model_tree import MODEL_CHECKPOINT_STORE
from feedbax.planner.project_plan import RunKey
from feedbax.project_folder import ProjectFileExists, ProjectFolderInvalid, ProjectFolderNotFound
from feedbax.project_model_assets import ModelAssetError, read_pinned_project_model, stage_project_archive
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanService


def import_loaded_model(
    folders: ProjectFolderService, project: str, query: StudioLoadedModelImportQuery, raw: bytes,
) -> StudioLoadedModelImportResult:
    from feedbax.host.model_archive import UnreadableModelArchive

    authored = folders.read_authored_snapshot(project, revision=query.revision)
    read = authored.project.documents.get
    reference = DocumentReference(document=query.model)
    if read(query.name) is not None:
        saved = compose_document(read, query.name)
        if isinstance(saved, ProjectDiagnostic) or saved.schema_id != LOADED_MODEL_SCHEMA_ID:
            raise ProjectFileExists(f"documents/{query.name}.json")
        existing = LoadedModelDocument.model_validate(saved.payload)
        if existing.model != reference:
            raise ProjectFileExists(f"documents/{query.name}.json")
        try:
            model = read_pinned_project_model(
                authored.root, authored.store_path, existing.lower(),
                lambda: composed_loaded_model_projection(existing, read),
            )
        except (ModelAssetError, ArtifactBlobCustodyError) as error:
            raise ProjectFolderInvalid(str(error)) from error
    else:
        existing = None
        composed = compose_document(read, query.model)
        if isinstance(composed, ProjectDiagnostic):
            raise ProjectFolderInvalid(composed.detail)
        if composed.schema_id != STORED_MODEL_DOCUMENT_SCHEMA_ID:
            raise ProjectFolderInvalid("Choose a Model document to fit the archive.")
        model = StoredModelDocument.model_validate(composed.payload).lower().with_keyed_root_ports()
    assert folders.components is not None
    with stage_project_archive(raw, model, folders.components) as (staging, staged):
        if isinstance(staged, ModelFitReport):
            if existing is not None:
                raise ProjectFileExists(f"documents/{query.name}.json")
            return StudioLoadedModelImportResult(StudioLoadedModelDoesNotFit(status="does_not_fit", report=staged))
        if isinstance(staged, UnreadableModelArchive):
            return StudioLoadedModelImportResult(StudioLoadedModelUnreadableArchive(
                status="unreadable_archive", path=staged.path, detail=staged.detail,
            ))
        payload = LoadedModelDocument.from_source(reference, staged)
        snapshot = folders.publish_loaded_model(
            project, revision=query.revision, name=query.name, label=query.label, model=payload, staging=staging, raw_archive=raw,
        )
        return StudioLoadedModelImportResult(StudioLoadedModelImported(
            status="imported", document=query.name, snapshot=snapshot,
        ))


def export_loaded_model(
    plans: ProjectPlanService, project: str, body: StudioLoadedModelExportRequest,
) -> bytes:
    key = RunKey(body.key.variant, body.key.step, body.key.row)
    view = plans.read(project, revision=body.revision, source_runs=(key,))
    run = next((row for row in view.settled.plan.runs if row.key == key), None)
    if run is not None and run.kind == "synthesize":
        if body.completed_updates is not None:
            raise ProjectFolderInvalid("A Synthesize model has no completed-update coordinate.")
        attempts = (*view.attempts_by_run.get(key, ()), *view.jobs.by_address.get(plan_attempt_key(key), ()))
        attempt = next((item for item in attempts if item.job_id == body.job_id and item.light_success is not None), None)
        if attempt is None or attempt.light_success.request.id.partition(":")[0] != "synthesize":
            raise ProjectFolderNotFound("The selected attempt does not belong to this Synthesize row.")
        model, = attempt.light_success.products["model"]
        tree = success_content_ref(model.content)
        assert isinstance(tree, ContentTreeRef)
        store = ImmutableArtifactBlobProvider(view.settled.store_path)
        return _archive_bytes(store, tree, SemanticModelEncoding.model_validate(model.descriptor))
    if body.completed_updates is None:
        raise ProjectFolderInvalid("Choose a completed Train update to export.")
    if run is not None and run.kind != "train":
        run = None
    attempts = () if run is None else (*view.attempts_by_run.get(key, ()), *view.previous_train_attempts(run))
    attempt = next((item for item in attempts if item.job_id == body.job_id), None)
    if attempt is None:
        raise ProjectFolderNotFound("The selected attempt does not belong to this Train row.")
    if attempt.successful_receipt is not None:
        store = ImmutableArtifactBlobProvider(view.settled.store_path)
        published = SuccessView(store).read_attempt(attempt.job_id)
        if (
            published is not None and published.receipt.request.id == attempt.request_id
            and (terminal := published.receipt.continuation) is not None
            and terminal.clock.completed_updates == body.completed_updates
        ):
            (model,) = published.receipt.products["checkpoint"]
            tree = success_content_ref(model.content)
            assert isinstance(tree, ContentTreeRef)
            return _archive_bytes(store, tree, SemanticModelEncoding.model_validate(model.descriptor))
    try:
        checkpoint = plans.jobs.host.checkpoint_for_export(
            job_id=body.job_id, project=view.settled.project_id,
            manifest_root=str(view.settled.store_path), update=body.completed_updates,
        )
    except HostError as error:
        raise ProjectFolderInvalid(str(error)) from error
    if checkpoint is None:
        raise ProjectFolderNotFound("The selected Train checkpoint is unavailable.")
    published = PublishedTrainingCheckpoint.model_validate(checkpoint)
    manifest = read_method_checkpoint_manifest(published.checkpoint_root, published.pointer)
    store = ImmutableArtifactBlobProvider(
        (published.checkpoint_root / published.pointer.manifest_relative_path).parent / MODEL_CHECKPOINT_STORE,
    )
    return _archive_bytes(store, manifest.model.tree, manifest.model.encoding)


def _archive_bytes(
    store: ImmutableArtifactBlobProvider, tree: ContentTreeRef, encoding: SemanticModelEncoding,
) -> bytes:
    from feedbax.host.model_archive import UnreadableModelArchive, export_model_archive

    archive = export_model_archive(store, tree, encoding)
    if isinstance(archive, UnreadableModelArchive):
        raise ProjectFolderInvalid(f"{archive.path}: {archive.detail}")
    return archive
