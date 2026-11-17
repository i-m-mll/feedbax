"""Bind saved Jobs on the venue that will execute them."""

from __future__ import annotations

import argparse
from base64 import b64decode
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path, PurePosixPath
import sys
from tempfile import TemporaryDirectory

from pydantic import TypeAdapter, ValidationError

from feedbax._atomic import sync_parent_directories, write_bytes_atomic
from feedbax._secure_fs import directory_open_flags, rename_no_replace
from feedbax.contracts.execution_target import SessionTarget
from feedbax.contracts.jobs import StudioJobSubmission
from feedbax.contracts.project_inputs import ProjectName
from feedbax.contracts.project_io import (
    PROJECT_DECLARATION_SUFFIX,
    PROJECT_FILE_FAMILIES,
    TextProjectExternalInputs,
    parse_text_project_files,
    project_semantic_root,
    read_text_project_snapshot,
)
from feedbax.contracts.project_resolution import BaseReaders
from feedbax.host.client import HostClient
from feedbax.host.io import private_directory
from feedbax.host.protocol import HostError, SessionAccess
from feedbax.jobs import JobService
from feedbax.orchestration.session import Nonempty, SessionError, SessionModel
from feedbax.orchestration.session_jobs import Delegation, admit_saved_delegation
from feedbax.persistence.ssh_store import RemoteStore, SSHStoreConnection
from feedbax.plugins.application import ApplicationRegistryBundle
from feedbax.plugins.composition import load_application
from feedbax.project_folder import (
    ProjectFolderLocation, SavedPlanMemo, SettledProjectPlan, _folder_lock, compile_saved_project,
)


_PROJECT_NAME = TypeAdapter(ProjectName)
_WORKER_ENV = {"JAX_PLATFORMS", "JAX_ENABLE_X64", "XLA_FLAGS"}


class _Admission(SessionModel):
    delegation: Delegation
    token: Nonempty
    access: SessionAccess
    placement: SessionTarget


@dataclass(frozen=True)
class VenueJobs:
    """Deliver the reviewed snapshot to a venue using its actual interpreter and Store."""

    connection: SSHStoreConnection
    python: str | PurePosixPath = field(kw_only=True)
    cwd: str | PurePosixPath = field(kw_only=True)
    host_root: str | PurePosixPath = field(kw_only=True)
    projects_root: str | PurePosixPath = field(kw_only=True)
    stores_root: str | PurePosixPath = field(kw_only=True)
    worker_environment: dict[str, str] | None = None

    def __post_init__(self) -> None:
        for value in (self.python, self.cwd, self.host_root, self.projects_root, self.stores_root):
            path = PurePosixPath(value)
            if not path.is_absolute() or ".." in path.parts or "\0" in str(path):
                raise ValueError("Venue admission requires absolute POSIX paths.")
        if self.worker_environment is not None and any(
            key not in _WORKER_ENV or not isinstance(value, str) or "\0" in value
            for key, value in self.worker_environment.items()
        ):
            raise ValueError("Invalid venue worker environment.")

    def store(self, project_id: str) -> RemoteStore:
        root = PurePosixPath(self.stores_root) / _PROJECT_NAME.validate_python(project_id)
        return RemoteStore(root, self.connection, python=str(self.python))

    def admit(
        self, request: Delegation, *, token: str, access: SessionAccess, placement: SessionTarget,
    ) -> StudioJobSubmission:
        argv = [
            "env", *(f"{key}={value}" for key, value in (self.worker_environment or {}).items()),
            str(self.python), "-B", "-m", "feedbax.orchestration.venue_jobs", str(self.cwd),
            str(self.host_root), str(self.projects_root), str(self.stores_root),
        ]
        call = _Admission(delegation=request, token=token, access=access, placement=placement)
        reply = json.loads(self.connection.run(argv, call.model_dump_json().encode()))
        if "error" in reply:
            error = reply["error"]
            if error["code"] == "invalid_admission":
                raise SessionError(error["message"])
            raise HostError(error["code"], error["message"])
        return StudioJobSubmission.model_validate(reply["value"])


def _stage_snapshot(root: Path, request: Delegation) -> ProjectFolderLocation:
    if request.files is None or request.request_ids is None:
        raise SessionError("Venue admission requires an authored snapshot and selected request IDs.")
    files = {}
    for relative, data in request.files.items():
        path = PurePosixPath(relative)
        declaration = len(path.parts) == 1 and path.suffix == PROJECT_DECLARATION_SUFFIX
        member = len(path.parts) == 2 and path.parts[0] in PROJECT_FILE_FAMILIES and path.suffix == ".json"
        if path.as_posix() != relative or not (declaration or member):
            raise SessionError("Invalid authored venue path.")
        if member:
            _PROJECT_NAME.validate_python(path.stem)
        files[relative] = b64decode(data, validate=True)

    destination_parent = private_directory(private_directory(root) / request.revision)
    destination = destination_parent / request.project
    text = parse_text_project_files(destination, files)
    if text.revision != request.revision or project_semantic_root(text.project) != request.semantic_root:
        raise SessionError("Authored snapshot differs from the saved delegation.")
    with TemporaryDirectory(prefix=".stage-", dir=destination_parent) as staged:
        stage = Path(staged)
        for relative, data in files.items():
            write_bytes_atomic(stage / relative, data, replace=False)
        parent = os.open(destination_parent, directory_open_flags(error_factory=OSError))
        try:
            try:
                rename_no_replace(
                    stage.name, destination.name, source_dir_fd=parent, destination_dir_fd=parent,
                    error_factory=OSError, context="venue project snapshot",
                )
            except FileExistsError:
                pass
        finally:
            os.close(parent)
    location = ProjectFolderLocation.selected(destination)
    if read_text_project_snapshot(destination).files != files:
        raise SessionError("The venue already has different bytes for this snapshot.")
    sync_parent_directories(destination)
    return location


@dataclass(frozen=True)
class _DelegatedProjectPlans:
    request: Delegation
    location: ProjectFolderLocation
    store: Path
    bundle: ApplicationRegistryBundle
    memo: SavedPlanMemo = field(default_factory=SavedPlanMemo, repr=False, compare=False)

    @contextmanager
    def _review(self, project_id: str, *, write: bool) -> Iterator[SettledProjectPlan]:
        if project_id != self.request.project:
            raise SessionError("The venue project differs from the authenticated delegation.")
        with _folder_lock(self.location, write=write) as folder:
            text = read_text_project_snapshot(folder.path)
            if text.revision != self.request.revision or project_semantic_root(text.project) != self.request.semantic_root:
                raise SessionError("Authored snapshot differs from the saved delegation.")
            read_pin = TextProjectExternalInputs(text, authority=self.location.path.parent).read
            yield compile_saved_project(
                text, project_id=self.request.project, store=self.store, readers=BaseReaders(figure_external=read_pin),
                operations=self.bundle.operations, components=self.bundle.components,
                functions=self.bundle.expression_functions, python_tasks=self.bundle.python_tasks,
                training_methods=self.bundle.training_methods,
                figure_templates=self.bundle.figure_templates, memo=self.memo,
            )

    def read_plan(self, project_id: str, *, revision: str | None) -> SettledProjectPlan:
        if revision != self.request.revision:
            raise SessionError("The requested revision differs from the authenticated delegation.")
        with self._review(project_id, write=False) as plan:
            return plan

    def admission_plan(self, project_id: str) -> AbstractContextManager[SettledProjectPlan]:
        return self._review(project_id, write=True)


def _serve(args: argparse.Namespace, call: _Admission) -> StudioJobSubmission:
    if str(call.placement.session_id) != call.access.session_id or call.placement.realization is None:
        raise SessionError("Venue admission requires a realized target matching the controller fence.")
    os.chdir(args.cwd)
    host = HostClient(args.host_root, session=call.access)
    host.prohibit_start()
    state = host.session_status()
    if state["expired"] or state["closing"]:
        raise SessionError("The venue session cannot accept a saved delegation.")
    bundle = load_application().bundle
    location = _stage_snapshot(args.projects_root, call.delegation)
    folders = _DelegatedProjectPlans(
        call.delegation, location, private_directory(args.stores_root) / call.delegation.project, bundle,
    )
    return admit_saved_delegation(
        JobService(host, bundle), folders, call.delegation, token=call.token,
        placement=call.placement,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cwd", "host_root", "projects_root", "stores_root"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    try:
        call = _Admission.model_validate_json(sys.stdin.buffer.read())
        result = _serve(args, call)
        reply = {"value": result.model_dump(mode="json")}
    except HostError as error:
        reply = {"error": {"code": error.code, "message": str(error)}}
    except ValidationError:
        reply = {"error": {"code": "invalid_admission", "message": "Invalid venue admission request."}}
    except (SessionError, OSError, ValueError) as error:
        reply = {"error": {"code": "invalid_admission", "message": str(error)}}
    print(json.dumps(reply))


if __name__ == "__main__":
    main()
