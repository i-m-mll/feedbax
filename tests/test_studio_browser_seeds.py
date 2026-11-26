"""Check browser seeds with Studio's native file creation and registered writer."""

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import re

import pytest
from fastapi.testclient import TestClient

from feedbax.application.project_writer import dump_text_project
from feedbax.application.stored_regions import StoredRegionRegistries
from feedbax.contracts.analysis_document import ANALYSIS_SCHEMA
from feedbax.contracts.project import Project
from feedbax.contracts.project_io import (
    STUDIO_PRESENTATION_FILENAME,
    ProjectConfiguration,
    TextProject,
    parse_text_project_files,
    read_text_project_snapshot,
)
from feedbax.contracts.project_storage import write_project_file
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.strict_json import strict_json_loads
from feedbax.contracts.studio_project import CreateNamedDocument, PresentationDocument, StudioProjectCreateFiles
from feedbax.desktop.release import AppPaths
from feedbax.host import HostClient
from feedbax.web.app import create_app
from tests.support.browser_seed_actions import FileSeed, FolderSeed, HomeSeed, NonSeed


_REPO = Path(__file__).resolve().parents[1]
_STUDIO = _REPO / "web" / "e2e" / "studio"
_FIXTURES = _STUDIO / "fixtures"
_MODULE_PREFIX = "web.e2e.studio.fixtures."
_PYTHON_MODULE = re.compile(r"""["']((?:web\.e2e\.studio\.fixtures|tests\.fixtures)\.\w+)["']""")


def _fixture_modules() -> set[str]:
    local = {
        _MODULE_PREFIX + ".".join(path.relative_to(_FIXTURES).with_suffix("").parts)
        for path in _FIXTURES.rglob("*.py") if path.name != "__init__.py"
    }
    referenced = {
        match.group(1) for path in _STUDIO.rglob("*.ts") for match in _PYTHON_MODULE.finditer(path.read_text())
    }
    return local | referenced


def _action_cases():
    for name in sorted(_fixture_modules()):
        module = import_module(name)
        actions = getattr(module, "ACTIONS", None)
        assert isinstance(actions, dict) and actions, f"{name} must declare its fixture actions."
        for key, action in actions.items():
            id = f"{name.rsplit('.', 1)[-1]}:{key}"
            if isinstance(action, NonSeed):
                assert action.reason.strip(), f"{id} must explain why it is not a seed."
                continue
            assert isinstance(action, FileSeed | HomeSeed | FolderSeed), f"Unclassified fixture action {id}."
            assert action.cases, f"{id} needs an offline case."
            for arguments in action.cases:
                suffix = " ".join(arguments)
                yield pytest.param(action, arguments, id=f"{id} {suffix}".rstrip())


@pytest.fixture(scope="module")
def registries():
    return StoredRegionRegistries()


@pytest.fixture(autouse=True)
def offline_seed(monkeypatch):
    def refuse_host(*args, **kwargs):
        pytest.fail("Browser seed actions must run without a Host.")

    monkeypatch.setattr(HostClient, "_exchange", refuse_host)


def _write(registries: StoredRegionRegistries, text: TextProject) -> dict[str, bytes]:
    return dump_text_project(text.project, text.configuration, registries=registries)


def _member_bytes(files: dict[str, bytes]) -> dict[str, bytes]:
    return {name: data for name, data in files.items() if name.startswith(("documents/", "steps/"))}


def _create_files(root: Path, registries: StoredRegionRegistries, files: object) -> dict[str, bytes]:
    request = StudioProjectCreateFiles.model_validate({"revision": "0" * 64, "files": files})
    documents = {file.id: file.document for file in request.files if isinstance(file, CreateNamedDocument)}
    steps = {file.id: file.step for file in request.files if not isinstance(file, CreateNamedDocument)}
    assert len(documents) + len(steps) == len(request.files), "The seed creates the same file twice."
    written = dump_text_project(
        Project(documents=documents, steps=steps), ProjectConfiguration(project="Fixture", store="results"),
        registries=registries,
    )
    reopened = parse_text_project_files(root, written)
    assert written == _write(registries, reopened)
    return _member_bytes(written)


def _file_envelopes(text: TextProject) -> list[object]:
    files = []
    for relative, data in text.files.items():
        path = Path(relative)
        kind = {"documents": "document", "steps": "step"}.get(path.parts[0])
        if kind:
            files.append({"kind": kind, "id": path.stem, kind: strict_json_loads(data, ref=relative)})
    return files


def _verify_folder(registries: StoredRegionRegistries, root: Path) -> None:
    authored = read_text_project_snapshot(root)
    canonical = _write(registries, authored)
    reopened = parse_text_project_files(root, canonical)
    assert canonical == _write(registries, reopened)
    assert _member_bytes(canonical) == _create_files(root, registries, _file_envelopes(authored))
    presentation = root / STUDIO_PRESENTATION_FILENAME
    if presentation.is_file():
        model = PresentationDocument.model_validate_json(presentation.read_bytes())
        written = write_project_file(model, source=STUDIO_PRESENTATION_FILENAME)
        assert written == write_project_file(
            PresentationDocument.model_validate_json(written), source=STUDIO_PRESENTATION_FILENAME,
        )


@pytest.mark.parametrize(("action", "arguments"), list(_action_cases()))
def test_browser_python_seed(action, arguments, registries, tmp_path):
    if isinstance(action, FileSeed):
        _create_files(tmp_path, registries, action.command(*arguments))
        return
    if isinstance(action, FolderSeed):
        root = tmp_path / "Folder"
        action.command(root, *arguments)
    else:
        home = tmp_path / "home"
        name = action.fixed_project or "Folder"
        arguments = (*(() if action.fixed_project else (name,)), *arguments)
        action.command(str(home), *arguments)
        root = AppPaths.resolve(instance_root=home).web / "projects" / name
    _verify_folder(registries, root)


@dataclass(frozen=True)
class _DocumentPayload:
    schema: str


@dataclass(frozen=True)
class _StepPayload:
    pass


_JSON_PAYLOAD_ROLES: dict[str, _DocumentPayload | _StepPayload] = {
    "velocity-analysis.json": _DocumentPayload(ANALYSIS_SCHEMA),
    "velocity-summary.report.json": _DocumentPayload("feedbax.report_document.v1"),
    "velocity-report-step.json": _StepPayload(),
}
_JSON_FIXTURES = tuple(sorted(_FIXTURES.rglob("*.json")))


@pytest.mark.parametrize("path", _JSON_FIXTURES, ids=lambda path: str(path.relative_to(_FIXTURES)))
def test_browser_json_seed(path, registries, tmp_path):
    relative = str(path.relative_to(_FIXTURES))
    value = strict_json_loads(path.read_bytes(), ref=relative)
    match _JSON_PAYLOAD_ROLES.get(relative):
        case _DocumentPayload(schema):
            files = [{
                "kind": "document", "id": "fixture",
                "document": {"schema": schema, "label": "Fixture", "payload": value},
            }]
        case _StepPayload():
            files = [{"kind": "step", "id": "fixture", "step": value}]
        case None:
            files = value if isinstance(value, list) else [value]
    _create_files(tmp_path, registries, files)


def test_browser_json_payload_roles_point_to_fixtures():
    assert _JSON_PAYLOAD_ROLES.keys() <= {str(path.relative_to(_FIXTURES)) for path in _JSON_FIXTURES}


def test_browser_evaluate_expression_argument_survives_repair(tmp_path):
    from web.e2e.studio.fixtures.evaluate_authoring import TARGET_Y, files

    app = create_app(projects_dir=tmp_path)
    with TestClient(app) as client:
        created = client.post("/api/projects", json={"name": "evaluate-fixture"})
        assert created.status_code == 201, created.text
        snapshot = created.json()["snapshot"]
        project_id = snapshot["project_id"]
        route = f"/api/projects/{project_id}"
        for file in files():
            reply = client.post(f"{route}/files", json={"revision": snapshot["revision"], "file": file})
            assert reply.status_code == 201, reply.text
            snapshot = reply.json()["snapshot"]

        folders = app.state.project_folders
        plan = folders.read_plan(project_id, revision=snapshot["revision"]).plan
        assert [(problem.code, problem.row) for problem in plan.diagnostics] == [
            ("authored_graph_variation", "seed=control"),
        ]
        before = {run.key: run.request_id for run in plan.runs}
        edit = client.patch(route, json={"revision": snapshot["revision"], "files": [{
            "address": {"kind": "step", "id": "conditions"},
            "patches": [{"op": "remove", "path": "/sweep/exceptions/raw-control/patches/params.variation/value/raw-mass"}],
        }]})
        assert edit.status_code == 200, edit.text
        snapshot = edit.json()
        conditions = snapshot["project"]["steps"]["conditions"]
        patch = conditions["sweep"]["exceptions"]["raw-control"]["patches"]["params.variation"]
        assert patch["value"] == {"row-y": {"op": "replace", "path": TARGET_Y, "value": 0.04}}

        plan = folders.read_plan(project_id, revision=snapshot["revision"]).plan
        assert not plan.diagnostics
        (control,) = (run for run in plan.runs if run.key.row == "seed=control")
        assert {run.key: run.request_id for run in plan.runs if run.key != control.key} == before
        request = decode_request(plan.requests[control.request_id], plan.documents.__getitem__)
        (task,) = (edit["value"] for edit in request.spec.params["variation"] if edit["path"] == "task")
        (target,) = (variable["value"] for variable in task["variables"] if variable["id"] == "target")
        assert [(part["value"], part["unit"]) for part in target["args"]] == [(0.18, "m"), (0.04, "m")]
