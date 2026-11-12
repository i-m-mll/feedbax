"""Studio's project-folder document routes."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from fastapi import APIRouter, HTTPException, Request, Response

from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.studio_project import (
    StudioPresentationWrite,
    StudioProjectPatch,
    StudioProjectRevision,
    StudioProjectSnapshot,
)
from feedbax.contracts.studio_scene import StudioPlanarSceneRequest, StudioPlanarSceneResponse
from feedbax.web.services.project_folder import (
    ProjectFolderInvalid,
    ProjectFolderNotFound,
    ProjectFolderService,
    ProjectRevisionConflict,
)


router = APIRouter(prefix="/api/projects", tags=["projects"])
_NO_STORE = {"Cache-Control": "no-store"}
_Result = TypeVar("_Result")


def _project_response(response: Response, operation: Callable[[], _Result]) -> _Result:
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


@router.get("/{project_id}")
def read_project(
    project_id: ProjectName, request: Request, response: Response
) -> StudioProjectSnapshot:
    return _project_response(response, lambda: request.app.state.project_folders.read(project_id))


@router.patch("/{project_id}")
def patch_project(
    project_id: ProjectName,
    body: StudioProjectPatch,
    request: Request,
    response: Response,
) -> StudioProjectSnapshot:
    return _project_response(
        response, lambda: request.app.state.project_folders.patch(project_id, body)
    )


@router.put("/{project_id}/presentation")
def put_project_presentation(
    project_id: ProjectName,
    body: StudioPresentationWrite,
    request: Request,
    response: Response,
) -> StudioProjectSnapshot:
    return _project_response(
        response, lambda: request.app.state.project_folders.write_presentation(project_id, body)
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
