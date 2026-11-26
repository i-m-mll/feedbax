from copy import deepcopy

from fastapi.testclient import TestClient

from feedbax import project_folder
from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.planner.project_plan import compile_project
from feedbax.project_folder import ProjectFolderLocation
from feedbax.web.app import create_app
from feedbax.web.services import trial_authoring
from tests.test_document_train_pair import authored_documents, document_project


def _project_with_choice(folder, count, *, alternatives=()):
    raw = document_project(authored_documents())
    raw["documents"]["task"]["payload"]["variables"]["width"] = {
        "label": "Width", "type": {"base": "real"},
        "value": "choice(" + ", ".join(map(str, range(count))) + ")",
    }
    for key in alternatives:
        candidate = deepcopy(raw["documents"]["task"])
        candidate["label"] = key.title()
        candidate["payload"]["id"] = key
        del candidate["payload"]["signals"]["target_pos"]
        raw["documents"][key] = candidate
    files = dump_text_project(Project.model_validate(raw), ProjectConfiguration(project=folder.name, store="results"))
    for relative, data in files.items():
        path = folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _check_tables(payload):
    assert payload["fields"]
    sites = payload["sites"]
    lists = payload["function_lists"]
    assert set(sites) == {field["site"] for field in payload["fields"]}
    assert set(lists) == {site["functions"] for site in sites.values()}
    assert len(lists) == len({tuple(info["id"] for info in functions) for functions in lists.values()})
    for site in sites.values():
        for name in site["names"]:
            assert "level" not in name
            assert "dimensions" not in name["type"]
    for functions in lists.values():
        for info in functions:
            assert not info.keys() & {"implicit_names", "level", "requires_period", "random", "stateful", "reset"}


def test_field_sites_keep_task_response_growth_under_the_per_field_budget(tmp_path):
    """An in-place Task view states its scope and function menus once as editable numeric arguments are added."""
    replies = []
    for count in (8, 48):
        folder = tmp_path / f"choice-{count}"
        _project_with_choice(folder, count)
        project = ProjectFolderLocation.selected(folder).project_id
        app = create_app(projects_dir=tmp_path / f"home-{count}", open_folder=folder, automatic_run="off")
        with TestClient(app) as client:
            url = f"/api/projects/{project}"
            revision = app.state.project_folders.read(project).revision
            body = {"revision": revision, "run": {"variant": "default", "step": "train", "row": "default"}}
            response = client.post(url + "/authoring/trials", json=body)
            assert response.status_code == 200, response.text
            task = response.json()
            _check_tables(task)
            arguments = [field for field in task["fields"] if field["path"].startswith("/task/variables/width/value/args/")]
            assert len(arguments) == count
            assert len({field["site"] for field in arguments}) == 1
            assert any(site["names"] for site in task["sites"].values())
            assert any(task["function_lists"].values())
            replies.append((len(response.content), len(task["fields"])))

            if count == 8:
                width = next(field for field in task["fields"] if field["path"] == "/task/variables/width/value")
                site = task["sites"][width["site"]]
                functions = {info["id"]: info for info in task["function_lists"][site["functions"]]}
                assert functions["loguniform"]["label"] == "Log-uniform"
                assert functions["truncexp"]["label"] == "Truncated exponential"
                assert "label" not in functions["add"]

                train = client.post(url + "/authoring/train/train/fields", json=body)
                assert train.status_code == 200, train.text
                _check_tables(train.json())
                model = client.post(url + "/model/fields", json={
                    "revision": revision, "document": "reach-model",
                    "owner": {"composites": [], "element": {"kind": "graph"}},
                })
                assert model.status_code == 200, model.text
                _check_tables(model.json())

    (small_bytes, small_fields), (large_bytes, large_fields) = replies
    assert large_fields - small_fields == 40
    assert (large_bytes - small_bytes) / (large_fields - small_fields) < 1_500, replies


def test_a_warm_task_view_reuses_choices_and_compiles_one_field_edit(tmp_path, monkeypatch):
    """The choices keep their problems without re-pairing every Task or recompiling the current Plan."""
    folder = tmp_path / "choices"
    _project_with_choice(folder, 16, alternatives=("alternative-a", "alternative-b"))
    project = ProjectFolderLocation.selected(folder).project_id
    app = create_app(projects_dir=tmp_path / "home", open_folder=folder, automatic_run="off")

    compiles = []
    choices = []
    sweep_sources = []
    project_view = trial_authoring._project
    read_sweep_source = trial_authoring._sweep_source

    def counted_compile(*args, **kwargs):
        compiles.append(1)
        return compile_project(*args, **kwargs)

    def counted_project(documents, bundle, **kwargs):
        if not kwargs.get("show_objective", True):
            choices.append(documents.task_source.id)
        return project_view(documents, bundle, **kwargs)

    def counted_sweep_source(*args):
        source = read_sweep_source(*args)
        if source is not None:
            sweep_sources.append(source.document)
        return source

    monkeypatch.setattr(project_folder, "compile_project", counted_compile)
    monkeypatch.setattr(trial_authoring, "compile_project", counted_compile)
    monkeypatch.setattr(trial_authoring, "_project", counted_project)
    monkeypatch.setattr(trial_authoring, "_sweep_source", counted_sweep_source)
    with TestClient(app) as client:
        revision = app.state.project_folders.read(project).revision
        body = {"revision": revision, "run": {"variant": "default", "step": "train", "row": "default"}}
        url = f"/api/projects/{project}/authoring/trials"

        def view(**extra):
            response = client.post(url, json=body | extra)
            assert response.status_code == 200, response.text
            return response.json()

        cold = view()
        assert choices == ["alternative-a", "alternative-b", cold["task"]["id"]]
        indices = {choice["document"]: choice["problem_indices"] for choice in cold["tasks"]}
        assert cold["task_problems"]
        assert indices["alternative-a"] == indices["alternative-b"]
        assert indices["alternative-a"] == list(range(len(cold["task_problems"])))
        assert not indices["task"]

        compiles.clear()
        choices.clear()
        sweep_sources.clear()
        warm = view()
        assert warm["tasks"] == cold["tasks"] and warm["task_problems"] == cold["task_problems"]
        assert not compiles and not choices
        assert sorted(sweep_sources) == ["objective", "task"]

        edited = view(change={
            "kind": "expression", "path": "/task/variables/width/value/args/2",
            "edit": {"kind": "text", "text": "2.5"},
        })
        assert edited["files"] and not edited["problems"]
        assert compiles == [1]
        assert choices == [cold["task"]["id"]]
