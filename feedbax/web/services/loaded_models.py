"""Publish fitted native weights and export retained model products."""

from uuid import uuid4

from feedbax.contracts.artifact_custody import ContentRetentionTarget, ContentRootRecord, ContentTreeRef
from feedbax.contracts.effective_success import success_content_ref
from feedbax.contracts.loaded_model import LOADED_MODEL_SCHEMA_ID, LOADED_MODEL_SCHEMA_VERSION, ModelFitReport
from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID, ModelDocument
from feedbax.contracts.model_checkpoint import SemanticModelEncoding
from feedbax.contracts.project import LoadedModelDocument, RootDocument
from feedbax.contracts.project_inputs import DocumentReference
from feedbax.contracts.project_resolution import ProjectDiagnostic, compose_document
from feedbax.contracts.studio_loaded_model import (
    StudioLoadedModelDoesNotFit, StudioLoadedModelExportRequest, StudioLoadedModelImported,
    StudioLoadedModelImportQuery, StudioLoadedModelImportResult, StudioLoadedModelUnreadableArchive,
)
from feedbax.contracts.studio_project import CreateNamedDocument, StudioProjectCreateFile
from feedbax.contracts.studio_training import PublishedTrainingCheckpoint
from feedbax.host.protocol import HostError
from feedbax.jobs import plan_attempt_key
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView
from feedbax.persistence.method_checkpoint import read_method_checkpoint_manifest
from feedbax.persistence.model_tree import MODEL_CHECKPOINT_STORE
from feedbax.planner.project_plan import RunKey
from feedbax.project_folder import ProjectFileExists, ProjectFolderInvalid, ProjectFolderNotFound, ProjectRevisionConflict
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanService


def import_loaded_model(
    folders: ProjectFolderService, project: str, query: StudioLoadedModelImportQuery, raw: bytes,
) -> StudioLoadedModelImportResult:
    from feedbax.host.model_archive import UnreadableModelArchive, stage_model_archive

    authored = folders.read_authored_snapshot(project, revision=query.revision)
    composed = compose_document(authored.project.documents.get, query.model)
    if isinstance(composed, ProjectDiagnostic):
        raise ProjectFolderInvalid(composed.detail)
    if composed.schema_id != MODEL_DOCUMENT_SCHEMA_ID:
        raise ProjectFolderInvalid("Choose a Model document to fit the archive.")
    model = ModelDocument.model_validate(composed.payload)
    store = ImmutableArtifactBlobProvider(authored.store_path)
    assert folders.components is not None
    with stage_model_archive(raw, model, store, components=folders.components) as staged:
        if isinstance(staged, ModelFitReport):
            return StudioLoadedModelImportResult(StudioLoadedModelDoesNotFit(status="does_not_fit", report=staged))
        if isinstance(staged, UnreadableModelArchive):
            return StudioLoadedModelImportResult(StudioLoadedModelUnreadableArchive(
                status="unreadable_archive", path=staged.path, detail=staged.detail,
            ))
        payload = LoadedModelDocument(
            schema_version=LOADED_MODEL_SCHEMA_VERSION, model=DocumentReference(document=query.model), source=staged,
        )
        root = store.register_root(ContentRootRecord(
            root_id=f"loaded-model-import:{uuid4().hex}", root_kind="import",
            targets=(
                ContentRetentionTarget(ref_kind="object", object=staged.model),
                ContentRetentionTarget(ref_kind="tree", tree=staged.weights),
            ),
        ))
        try:
            created = folders.create_file(project, StudioProjectCreateFile(
                revision=query.revision,
                file=CreateNamedDocument(
                    kind="document", id=query.name,
                    document=RootDocument(schema=LOADED_MODEL_SCHEMA_ID, payload=payload.model_dump(mode="json")),
                ),
            ))
        except (ProjectRevisionConflict, ProjectFileExists):
            store.release_root(root.root_id)
            raise
        return StudioLoadedModelImportResult(StudioLoadedModelImported(
            status="imported", document=query.name, snapshot=created.snapshot,
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
