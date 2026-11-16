"""Bind saved Jobs on the venue that will execute them."""

from __future__ import annotations

import argparse
import asyncio
from base64 import b64decode
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
    PROJECT_FILE_FAMILIES,
    parse_text_project_files,
    project_declaration_name,
    project_semantic_root,
    read_text_project_snapshot,
)
from feedbax.host.client import HostClient
from feedbax.host.io import private_directory
from feedbax.host.protocol import HostError, SessionAccess
from feedbax.jobs import JobService
from feedbax.orchestration.session import Nonempty, SessionError, SessionModel
from feedbax.orchestration.session_jobs import Delegation, admit_saved_delegation
from feedbax.persistence.ssh_store import RemoteStore, SSHStoreConnection
from feedbax.plugins.composition import compose_application
from feedbax.project_folder import ProjectFolder, ProjectFolderLocation


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
    bootstrap_modules: tuple[str, ...] = ()
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
            str(self.python), "-m", "feedbax.orchestration.venue_jobs", str(self.cwd),
            str(self.host_root), str(self.projects_root), str(self.stores_root),
        ]
        for module in self.bootstrap_modules:
            argv.extend(("--bootstrap", module))
        call = _Admission(delegation=request, token=token, access=access, placement=placement)
        reply = json.loads(self.connection.run(argv, call.model_dump_json().encode()))
        if "error" in reply:
            error = reply["error"]
            if error["code"] == "invalid_admission":
                raise SessionError(error["message"])
            raise HostError(error["code"], error["message"])
        return StudioJobSubmission.model_validate(reply["value"])


def _stage_snapshot(root: Path, request: Delegation) -> Path:
    if request.files is None or request.request_ids is None:
        raise SessionError("Venue admission requires an authored snapshot and selected request IDs.")
    files = {}
    declaration = project_declaration_name(request.project)
    for relative, data in request.files.items():
        path = PurePosixPath(relative)
        if relative != declaration and (
            path.as_posix() != relative or len(path.parts) != 2 or path.parts[0] not in PROJECT_FILE_FAMILIES
            or path.suffix != ".json"
        ):
            raise SessionError("Invalid authored venue path.")
        if relative != declaration:
            _PROJECT_NAME.validate_python(path.stem)
        files[relative] = b64decode(data, validate=True)

    destination_parent = private_directory(private_directory(root) / request.revision)
    destination = destination_parent / request.project
    text = parse_text_project_files(destination, files)
    if (
        text.configuration.project != request.project or text.revision != request.revision
        or project_semantic_root(text.project) != request.semantic_root
    ):
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
    ProjectFolderLocation.selected(destination)
    if read_text_project_snapshot(destination).files != files:
        raise SessionError("The venue already has different bytes for this snapshot.")
    sync_parent_directories(destination)
    return destination


def _serve(args: argparse.Namespace, call: _Admission) -> StudioJobSubmission:
    if str(call.placement.session_id) != call.access.session_id or call.placement.realization is None:
        raise SessionError("Venue admission requires a realized target matching the controller fence.")
    os.chdir(args.cwd)
    host = HostClient(args.host_root, session=call.access)
    host.prohibit_start()
    state = host.session_status()
    if state["expired"] or state["closing"]:
        raise SessionError("The venue session cannot accept a saved delegation.")
    bootstrap = tuple(args.bootstrap)
    bundle = asyncio.run(compose_application(modules=bootstrap)).bundle
    destination = _stage_snapshot(args.projects_root, call.delegation)
    folders = ProjectFolder(
        destination.parent, stores_root=private_directory(args.stores_root),
        operations=bundle.operations, components=bundle.components, functions=bundle.expression_functions,
        python_tasks=bundle.python_tasks,
    )
    return admit_saved_delegation(
        JobService(host, bundle, bootstrap_modules=bootstrap), folders, call.delegation, token=call.token,
        placement=call.placement,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cwd", "host_root", "projects_root", "stores_root"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--bootstrap", action="append", default=[])
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
