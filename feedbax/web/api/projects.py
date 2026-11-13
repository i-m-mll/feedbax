"""Studio's project-folder document routes."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, HTTPException, Request, Response

from feedbax.contracts.model_edit import (
    EvaluateAttachmentEdit, EvaluateAttachmentReadRequest, EvaluateAttachmentRequest, EvaluateAttachmentState,
    ModelConnectionReview, ModelConnectionReviewRequest,
    ModelProjection, ModelProjectionRequest,
    ModelEditRequest, ModelEditResponse, ModelUndoRequest,
)
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.studio_analyze import StudioAnalyzeAuthoring, StudioAnalyzeRequest
from feedbax.contracts.studio_authoring import (
    StudioAuthoringCatalog,
    StudioAuthoringSurface,
    StudioDocumentAuthoring,
    StudioDocumentAuthoringRequest,
    StudioTrainAuthoring,
    StudioTrainAuthoringRequest,
)
from feedbax.contracts.studio_project import (
    StudioPresentationWrite,
    StudioProjectCreateFile,
    StudioProjectDocument,
    StudioProjectPatch,
    StudioProjectPatchBatch,
    StudioProjectRename,
    StudioProjectRevision,
    StudioProjectSnapshot,
)
from feedbax.contracts.studio_trials import StudioTrialAuthoring, StudioTrialAuthoringRequest
from feedbax.contracts.studio_scene import StudioPlanarSceneRequest, StudioPlanarSceneResponse
from feedbax.contracts.studio_field import ModelFieldRequest, ModelFieldResponse, TrainFieldRequest, TrainFieldResponse
from feedbax.web.services.project_folder import (
    ProjectFileExists,
    ProjectFolderInvalid,
    ProjectFolderNotFound,
    ProjectFolderService,
    ProjectRevisionConflict,
)


router = APIRouter(prefix="/api/projects", tags=["projects"])
_NO_STORE = {"Cache-Control": "no-store"}


def _project_response[Result](response: Response, operation: Callable[[], Result]) -> Result:
    try:
        snapshot = operation()
    except ProjectFolderNotFound as exc:
        raise HTTPException(404, detail=str(exc), headers=_NO_STORE) from exc
    except ProjectFolderInvalid as exc:
        raise HTTPException(422, detail=str(exc), headers=_NO_STORE) from exc
    except ProjectRevisionConflict as exc:
        raise HTTPException(
            409,
            detail={"message": str(exc), "current": exc.current, "resource": exc.resource},
            headers=_NO_STORE,
        ) from exc
    except ProjectFileExists as exc:
        raise HTTPException(
            409,
            detail={"code": "authored_file_exists", "message": f"An authored file already exists at {exc}."},
            headers=_NO_STORE,
        ) from exc
    except OSError as exc:
        raise HTTPException(
            503,
            detail={
                "code": "project_file_unavailable",
                "message": "Project file operation could not be confirmed.",
            },
            headers=_NO_STORE,
        ) from exc
    response.headers.update(_NO_STORE)
    return snapshot


@router.get("/authoring/schema")
def read_authoring_surface(request: Request, response: Response) -> StudioAuthoringSurface:
    from feedbax.studio.authoring import authoring_surface

    return _project_response(response, lambda: authoring_surface(request.app.state.bootstrap_state))


@router.get("/authoring/catalog")
def read_authoring_catalog(request: Request, response: Response) -> StudioAuthoringCatalog:
    from feedbax.studio.authoring import authoring_surface

    return _project_response(response, lambda: authoring_surface(
        request.app.state.bootstrap_state, catalog=True,
    ))


@router.post("/{project_id}/authoring/train/{step_id}")
def read_train_authoring(
    project_id: ProjectName,
    step_id: ProjectName,
    body: StudioTrainAuthoringRequest,
    request: Request,
    response: Response,
) -> StudioTrainAuthoring:
    from feedbax.studio.authoring import project_train_fields

    def project_fields() -> StudioTrainAuthoring:
        folders = request.app.state.project_folders
        snapshot = folders.read(project_id, revision=body.revision)
        bundle = request.app.state.bootstrap_state.bundle
        try:
            fields, patches, documents, reach_target = project_train_fields(
                snapshot.project, step_id, bundle.training_programs, body.change, components=bundle.components,
                read_plan=lambda keys: folders.read_plan(project_id, revision=snapshot.revision, source_runs=keys).plan,
            )
        except ValueError as error:
            raise ProjectFolderInvalid(str(error)) from error
        return StudioTrainAuthoring(
            project_id=project_id, step_id=step_id, revision=snapshot.revision,
            fields=fields, patches=patches, documents=documents, reach_target=reach_target,
        )

    return _project_response(response, project_fields)


@router.post("/{project_id}/authoring/train/{step_id}/fields")
def project_train_numeric_fields(
    project_id: ProjectName, step_id: ProjectName, body: TrainFieldRequest, request: Request, response: Response,
) -> TrainFieldResponse:
    from feedbax.studio.authoring import project_training_numeric_fields

    def project_fields() -> TrainFieldResponse:
        folders = request.app.state.project_folders
        snapshot = folders.read(project_id, revision=body.revision)
        bundle = request.app.state.bootstrap_state.bundle
        try:
            return project_training_numeric_fields(
                snapshot.project, step_id, bundle.training_programs, bundle.expression_functions, body,
                lambda keys: folders.read_plan(project_id, revision=body.revision, source_runs=keys).plan,
            )
        except ValueError as error:
            raise ProjectFolderInvalid(str(error)) from error

    return _project_response(response, project_fields)


@router.post("/{project_id}/authoring/document", response_model_exclude_unset=True)
def read_document_authoring(
    project_id: ProjectName, body: StudioDocumentAuthoringRequest, request: Request, response: Response,
) -> StudioDocumentAuthoring:
    from feedbax.web.services.document_authoring import document_authoring

    return _project_response(
        response, lambda: document_authoring(
            request.app.state.project_folders, project_id, body,
            components=request.app.state.bootstrap_state.bundle.components,
        ),
    )


@router.post("/{project_id}/authoring/trials")
def read_trial_authoring(
    project_id: ProjectName, body: StudioTrialAuthoringRequest, request: Request, response: Response,
) -> StudioTrialAuthoring:
    from feedbax.web.services.trial_authoring import trial_authoring

    return _project_response(response, lambda: trial_authoring(
        request.app.state.project_folders, project_id, body,
        request.app.state.bootstrap_state.bundle,
    ))


@router.post("/{project_id}/authoring/analyze")
def read_analyze_authoring(
    project_id: ProjectName, body: StudioAnalyzeRequest, request: Request, response: Response,
) -> StudioAnalyzeAuthoring:
    from feedbax.web.services.analyze_authoring import analyze_authoring

    return _project_response(
        response, lambda: analyze_authoring(request.app.state.project_plan, project_id, body),
    )


@router.post("/{project_id}/scene/planar")
def read_planar_scene(
    project_id: ProjectName, body: StudioPlanarSceneRequest, request: Request, response: Response,
) -> StudioPlanarSceneResponse:
    from feedbax.studio.scene_geometry import prepare_project_scene

    def prepare() -> StudioPlanarSceneResponse:
        snapshot = request.app.state.project_folders.read(project_id)
        if snapshot.revision != body.revision:
            raise ProjectRevisionConflict("project", snapshot.revision)
        model, diagnostics = prepare_project_scene(
            snapshot.project, body.document_id, body.node_path,
            request.app.state.bootstrap_state.bundle.components,
        )
        return StudioPlanarSceneResponse(
            project_id=project_id, **body.model_dump(), model=model, diagnostics=diagnostics,
        )

    return _project_response(response, prepare)


@router.get("/{project_id}", response_model_exclude_unset=True)
def read_project(
    project_id: ProjectName, request: Request, response: Response
) -> StudioProjectSnapshot:
    return _project_response(response, lambda: request.app.state.project_folders.read(project_id))


@router.patch("/{project_id}", response_model_exclude_unset=True)
def patch_project(
    project_id: ProjectName,
    body: StudioProjectPatch | StudioProjectPatchBatch,
    request: Request,
    response: Response,
) -> StudioProjectSnapshot:
    return _project_response(
        response, lambda: request.app.state.project_folders.patch(project_id, body)
    )


@router.post("/{project_id}/model/connections")
def review_model_connections(
    project_id: ProjectName, body: ModelConnectionReviewRequest, request: Request, response: Response,
) -> ModelConnectionReview:
    components = request.app.state.bootstrap_state.bundle.components
    return _project_response(
        response, lambda: request.app.state.project_folders.review_model_connections(project_id, body, components)
    )


@router.post("/{project_id}/model/projection", response_model_exclude_unset=True)
def read_model_projection(
    project_id: ProjectName, body: ModelProjectionRequest, request: Request, response: Response,
) -> ModelProjection:
    from feedbax.web.services.model_projection import model_projection

    components = request.app.state.bootstrap_state.bundle.components
    return _project_response(
        response, lambda: model_projection(request.app.state.project_folders, project_id, body, components)
    )


@router.post("/{project_id}/model/edit", response_model_exclude_unset=True)
def edit_model(
    project_id: ProjectName, body: ModelEditRequest, request: Request, response: Response,
) -> ModelEditResponse:
    components = request.app.state.bootstrap_state.bundle.components
    return _project_response(
        response, lambda: request.app.state.project_folders.edit_model(project_id, body, components)
    )


@router.post("/{project_id}/model/fields")
def project_model_fields(
    project_id: ProjectName, body: ModelFieldRequest, request: Request, response: Response,
) -> ModelFieldResponse:
    from feedbax.web.services.model_field import model_fields

    bundle = request.app.state.bootstrap_state.bundle
    return _project_response(
        response, lambda: model_fields(
            request.app.state.project_folders, project_id, body, bundle.components, bundle.expression_functions,
        ),
    )


@router.post("/{project_id}/model/undo", response_model_exclude_unset=True)
def undo_model(
    project_id: ProjectName, body: ModelUndoRequest, request: Request, response: Response,
) -> ModelEditResponse:
    return _project_response(
        response, lambda: request.app.state.project_folders.undo_model(project_id, body)
    )


@router.post("/{project_id}/model/evaluate-attachment")
def evaluate_attachment(
    project_id: ProjectName, body: EvaluateAttachmentRequest, request: Request, response: Response,
) -> EvaluateAttachmentEdit:
    from feedbax.web.services.evaluate_attachment import EvaluateAttachmentRefused, prepare_evaluate_attachment

    try:
        return _project_response(
            response, lambda: prepare_evaluate_attachment(
                request.app.state.project_folders, project_id, body,
                request.app.state.bootstrap_state.bundle.components,
            ),
        )
    except EvaluateAttachmentRefused as error:
        raise HTTPException(422, detail=error.problem.model_dump(mode="json"), headers=_NO_STORE) from error


@router.post("/{project_id}/model/evaluate-attachments", response_model_exclude_unset=True)
def evaluate_attachments(
    project_id: ProjectName, body: EvaluateAttachmentReadRequest, request: Request, response: Response,
) -> EvaluateAttachmentState:
    from feedbax.web.services.evaluate_attachment import read_evaluate_attachments

    try:
        return _project_response(
            response, lambda: read_evaluate_attachments(
                request.app.state.project_folders, project_id, body, request.app.state.bootstrap_state.bundle.components,
            ),
        )
    except (KeyError, ValueError) as error:
        raise HTTPException(422, detail=str(error), headers=_NO_STORE) from error


@router.put("/{project_id}/presentation", response_model_exclude_unset=True)
def put_project_presentation(
    project_id: ProjectName,
    body: StudioPresentationWrite,
    request: Request,
    response: Response,
) -> StudioProjectSnapshot:
    return _project_response(
        response, lambda: request.app.state.project_folders.write_presentation(project_id, body)
    )


@router.post("", status_code=201, response_model_exclude_unset=True)
def new_project(request: Request, response: Response) -> StudioProjectDocument:
    document = _project_response(response, request.app.state.project_folders.new)
    response.headers["Location"] = request.url_for(
        "open_project", project_id=document.snapshot.project_id
    ).path
    return document


@router.get("/{project_id}/open", response_model_exclude_unset=True)
def open_project(project_id: ProjectName, request: Request, response: Response) -> StudioProjectDocument:
    return _project_response(response, lambda: request.app.state.project_folders.open(project_id))


@router.post("/{project_id}/rename", response_model_exclude_unset=True)
def rename_project(
    project_id: ProjectName, body: StudioProjectRename, request: Request, response: Response
) -> StudioProjectDocument:
    return _project_response(
        response, lambda: request.app.state.project_folders.rename(project_id, body)
    )


@router.post("/{project_id}/files", status_code=201, response_model_exclude_unset=True)
def create_project_file(
    project_id: ProjectName, body: StudioProjectCreateFile, request: Request, response: Response
) -> StudioProjectDocument:
    return _project_response(
        response, lambda: request.app.state.project_folders.create_file(project_id, body)
    )


class ProjectRevisionWatch:
    """Reconcile one connection's folder revisions against current file observations."""

    def __init__(self, service: ProjectFolderService, folders: tuple[str, ...]) -> None:
        self._service = service
        self._folders = folders
        self._fingerprints: dict[str, object] = {}
        self._revisions: dict[str, StudioProjectRevision] = {}

    def changed(self) -> list[StudioProjectRevision]:
        revisions = []
        for folder in self._folders:
            try:
                fingerprint = self._service.fingerprint(folder)
                if fingerprint == self._fingerprints.get(folder):
                    continue
                revision = self._service.revision(folder)
            except (ProjectFolderInvalid, OSError):
                self._fingerprints.pop(folder, None)
                continue
            self._fingerprints[folder] = fingerprint
            if revision != self._revisions.get(folder):
                self._revisions[folder] = revision
                revisions.append(revision)
        return revisions
