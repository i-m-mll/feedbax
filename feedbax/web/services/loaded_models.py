"""Publish fitted native weights beside their Model and export retained Train members."""

from uuid import uuid4

from feedbax.contracts.artifact_custody import ContentRetentionTarget, ContentRootRecord
from feedbax.contracts.loaded_model import LOADED_MODEL_SCHEMA_ID, LOADED_MODEL_SCHEMA_VERSION, ModelFitReport
from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID, ModelDocument
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
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
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
    from feedbax.host.model_archive import UnreadableModelArchive, export_model_archive

    key = RunKey(body.key.variant, body.key.step, body.key.row)
    view = plans.read(project, revision=body.revision, source_runs=(key,))
    run = next((row for row in view.settled.plan.runs if row.key == key and row.kind == "train"), None)
    if run is None or not any(
        item.job_id == body.job_id
        for item in (*view.attempts_by_run.get(key, ()), *view.previous_train_attempts(run))
    ):
        raise ProjectFolderNotFound("The selected attempt does not belong to this Train row.")
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
    archive = export_model_archive(store, manifest.model.tree, manifest.model.encoding)
    if isinstance(archive, UnreadableModelArchive):
        raise ProjectFolderInvalid(f"{archive.path}: {archive.detail}")
    return archive
