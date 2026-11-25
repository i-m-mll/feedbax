"""Studio's project-folder document routes."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, HTTPException, Request, Response

from feedbax.contracts.model_edit import (
    EvaluateModelRequest, EvaluateModelState,
    ModelConnectionReview, ModelConnectionReviewRequest,
    ModelProjection, ModelProjectionRequest,
    ModelEditRequest, ModelEditResponse, ModelUndoRequest,
)
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.studio_analyze import StudioAnalyzeAuthoring, StudioAnalyzeRequest
from feedbax.contracts.studio_removed_runs import StudioRemovedRuns, StudioRemovedRunsRequest
from feedbax.contracts.studio_authoring import (
    StudioAuthoringCatalog,
    StudioAuthoringSurface,
    StudioTrainAuthoring,
    StudioTrainAuthoringRequest,
    StudioTrainStartEdit,
    StudioTrainStartRequest,
)
from feedbax.contracts.studio_project import (
    StudioMemberCommand,
    StudioMemberCommandResult,
    StudioPresentationWrite,
    StudioProjectCreateFile,
    StudioProjectCreateFiles,
    StudioProjectDocument,
    StudioProjectExample,
    StudioProjectExamples,
    StudioProjectList,
    StudioProjectNew,
    StudioProjectPatch,
    StudioProjectPatchBatch,
    StudioProjectRename,
    StudioProjectReplay,
    StudioProjectRevision,
    StudioProjectSnapshot,
)
from feedbax.contracts.studio_sets import StudioSetEdit, StudioSetEditRequest, StudioSetsOffered, StudioSetsOfferedRequest
from feedbax.contracts.studio_trials import StudioTrialAuthoring, StudioTrialAuthoringRequest
from feedbax.contracts.studio_checkpoint_rule import StudioCheckpointRuleEdit, StudioCheckpointRuleRequest
from feedbax.contracts.studio_subset import (
    StudioRunRuleCheck, StudioRunRuleRequest, StudioSubsetToRowsEdit, StudioSubsetToRowsRequest,
)
from feedbax.contracts.studio_scene import (
    StudioSceneGeometryRequest,
    StudioSceneGeometryResponse,
)
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
    from feedbax.web.services.authored_scope import preview_plan, train_edit_store
    from feedbax.web.services.trial_scope import preview_stored, train_selection
    from feedbax.contracts.jobs import StudioProjectWindowMapProblem
    from feedbax.contracts.project_resolution import ResolutionCode

    def project_fields() -> StudioTrainAuthoring:
        folders = request.app.state.project_folders
        snapshot = folders.read(project_id, revision=body.revision)
        bundle = request.app.state.bootstrap_state.bundle
        selection = train_selection(
            folders, project_id, snapshot.revision, snapshot.project, step_id, body.run, body.rows, body.axes,
            body.chosen_steps, defer_scope=True,
        )
        try:
            projection = project_train_fields(
                snapshot.project, step_id, bundle.training_methods, body.change, selection,
                read_plan=lambda keys: folders.read_plan(project_id, revision=snapshot.revision, source_runs=keys).plan,
                class_sweep=body.class_sweep, reset=body.reset,
            )
        except ValueError as error:
            raise ProjectFolderInvalid(str(error)) from error
        stored = train_edit_store(folders, project_id, snapshot.project, step_id, body, projection.stored)
        settled = folders.read_plan(project_id, revision=snapshot.revision)
        plan = settled.plan
        if not isinstance(stored, list) and (stored.files or stored.created) and any(
            step_id == step and planned.sweep is not None for (_, step), planned in plan.steps.items()
        ):
            plan = preview_plan(folders, settled, preview_stored(snapshot.project, stored))
        sweep_problems = [
            StudioProjectWindowMapProblem.from_diagnostic(item, declaration="")
            for item in plan.diagnostics if item.step == step_id and item.code == ResolutionCode.INVALID_SWEEP
        ]
        return StudioTrainAuthoring(
            project_id=project_id, step_id=step_id, revision=snapshot.revision, fields=projection.fields,
            settings=projection.settings, mixed=projection.mixed, files=[] if isinstance(stored, list) else stored.files,
            created=[] if isinstance(stored, list) else list(stored.created),
            store_choices=stored if isinstance(stored, list) else None,
            documents=projection.documents, classes=projection.classes, sweep_problems=sweep_problems,
            removed_values=projection.removed, changed=projection.changed,
        )

    return _project_response(response, project_fields)


@router.post("/{project_id}/authoring/train/{step_id}/start")
def start_train_step(
    project_id: ProjectName, step_id: ProjectName, body: StudioTrainStartRequest, request: Request, response: Response,
) -> StudioTrainStartEdit:
    from feedbax.web.services.train_start import prepare_train_start

    return _project_response(response, lambda: prepare_train_start(
        request.app.state.project_folders, project_id, step_id, body,
    ))


@router.post("/{project_id}/authoring/train/{step_id}/fields")
def project_train_numeric_fields(
    project_id: ProjectName, step_id: ProjectName, body: TrainFieldRequest, request: Request, response: Response,
) -> TrainFieldResponse:
    from feedbax.studio.authoring import project_numeric_fields
    from feedbax.studio.selection import StoredEdit
    from feedbax.web.services.authored_scope import train_edit_store
    from feedbax.web.services.trial_scope import train_selection

    def project_fields() -> TrainFieldResponse:
        folders = request.app.state.project_folders
        snapshot = folders.read(project_id, revision=body.revision)
        bundle = request.app.state.bootstrap_state.bundle
        selection = train_selection(
            folders, project_id, body.revision, snapshot.project, step_id, body.run, body.rows, body.axes,
            body.chosen_steps, defer_scope=True,
        )
        try:
            projection = project_numeric_fields(
                snapshot.project, step_id, bundle.training_methods, bundle.expression_functions, body, selection,
                lambda keys: folders.read_plan(project_id, revision=body.revision, source_runs=keys).plan,
            )
        except ValueError as error:
            raise ProjectFolderInvalid(str(error)) from error
        stored = train_edit_store(
            folders, project_id, snapshot.project, step_id, body, StoredEdit(projection.files, tuple(projection.created)),
            new_rows_for_selection=selection.values is not None,
        )
        if isinstance(stored, list):
            return projection.model_copy(update={"files": [], "created": [], "store_choices": stored})
        return projection.model_copy(update={"files": stored.files, "created": list(stored.created)})

    return _project_response(response, project_fields)


@router.post("/{project_id}/authoring/checkpoint-rule", response_model_exclude_unset=True)
def edit_checkpoint_rule(
    project_id: ProjectName, body: StudioCheckpointRuleRequest, request: Request, response: Response,
) -> StudioCheckpointRuleEdit:
    from feedbax.web.services.checkpoint_rule import prepare_checkpoint_rule_edit

    return _project_response(response, lambda: prepare_checkpoint_rule_edit(
        request.app.state.project_folders, project_id, body,
    ))


@router.post("/{project_id}/authoring/run-rule")
def check_run_rule(
    project_id: ProjectName, body: StudioRunRuleRequest, request: Request, response: Response,
) -> StudioRunRuleCheck:
    return _project_response(response, lambda: request.app.state.project_plan.check_run_rule(project_id, body))


@router.post("/{project_id}/authoring/subset-to-rows", response_model_exclude_unset=True)
def prepare_subset_to_rows(
    project_id: ProjectName, body: StudioSubsetToRowsRequest, request: Request, response: Response,
) -> StudioSubsetToRowsEdit:
    return _project_response(response, lambda: request.app.state.project_plan.subset_to_rows(project_id, body))


@router.post("/{project_id}/authoring/sets", response_model_exclude_unset=True)
def prepare_set_edit(
    project_id: ProjectName, body: StudioSetEditRequest, request: Request, response: Response,
) -> StudioSetEdit:
    from feedbax.web.services.set_authoring import set_edit

    return _project_response(response, lambda: set_edit(request.app.state.project_folders, project_id, body))


@router.post("/{project_id}/authoring/sets-offered")
def read_sets_offered(
    project_id: ProjectName, body: StudioSetsOfferedRequest, request: Request, response: Response,
) -> StudioSetsOffered:
    from feedbax.web.services.set_authoring import sets_offered_to

    return _project_response(response, lambda: sets_offered_to(request.app.state.project_folders, project_id, body))


@router.post("/{project_id}/authoring/removed-runs")
def review_removed_runs(
    project_id: ProjectName, body: StudioRemovedRunsRequest, request: Request, response: Response,
) -> StudioRemovedRuns:
    from feedbax.web.services.removed_runs import removed_runs

    return _project_response(response, lambda: removed_runs(request.app.state.project_folders, project_id, body))


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


@router.post("/{project_id}/scene/geometry")
def read_scene_geometry(
    project_id: ProjectName, body: StudioSceneGeometryRequest, request: Request, response: Response,
) -> StudioSceneGeometryResponse:
    from feedbax.studio.scene_geometry import prepare_scene_geometry

    def prepare() -> StudioSceneGeometryResponse:
        snapshot = request.app.state.project_folders.read(project_id)
        if snapshot.revision != body.revision:
            raise ProjectRevisionConflict("project", snapshot.revision)
        frames = prepare_scene_geometry(
            snapshot.project, body.document_id, body.poses,
            request.app.state.bootstrap_state.bundle.components,
        )
        return StudioSceneGeometryResponse(
            project_id=project_id, revision=body.revision, document_id=body.document_id, frames=frames,
        )

    return _project_response(response, prepare)


@router.get("/examples")
def list_project_examples(response: Response) -> StudioProjectExamples:
    from feedbax.contracts.project import member_display_name
    from feedbax.examples import load_shipped_projects

    return _project_response(response, lambda: StudioProjectExamples(examples=[
        StudioProjectExample(id=key, name=member_display_name(key))
        for key in load_shipped_projects()
    ]))


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


@router.post("/{project_id}/history/replay", response_model_exclude_unset=True)
def replay_project(
    project_id: ProjectName, body: StudioProjectReplay, request: Request, response: Response,
) -> StudioProjectSnapshot:
    return _project_response(
        response, lambda: request.app.state.project_folders.replay(project_id, body)
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


@router.post("/{project_id}/model/evaluate-model", response_model_exclude_unset=True)
def read_evaluate_model(
    project_id: ProjectName, body: EvaluateModelRequest, request: Request, response: Response,
) -> EvaluateModelState:
    from feedbax.web.services.evaluate_sources import read_evaluate_model

    try:
        return _project_response(
            response, lambda: read_evaluate_model(
                request.app.state.project_folders, project_id, body, request.app.state.bootstrap_state.bundle.components,
            ).state,
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


@router.get("")
def list_projects(request: Request, response: Response) -> StudioProjectList:
    return _project_response(response, request.app.state.project_folders.list_projects)


@router.post("", status_code=201, response_model_exclude_unset=True)
def new_project(body: StudioProjectNew, request: Request, response: Response) -> StudioProjectDocument:
    document = _project_response(response, lambda: request.app.state.project_folders.new(body.name, example=body.example))
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
    project_id: ProjectName, body: StudioProjectCreateFile | StudioProjectCreateFiles, request: Request, response: Response,
) -> StudioProjectDocument:
    return _project_response(
        response, lambda: request.app.state.project_folders.create_file(project_id, body)
    )


@router.post("/{project_id}/members/command", response_model_exclude_unset=True)
def command_project_member(
    project_id: ProjectName, body: StudioMemberCommand, request: Request, response: Response
) -> StudioMemberCommandResult:
    return _project_response(
        response, lambda: request.app.state.project_folders.member_command(project_id, body)
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
