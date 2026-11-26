from fastapi.testclient import TestClient

from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.project_folder import ProjectFolderLocation
from feedbax.web.app import create_app
from tests.test_document_train_pair import authored_documents, document_project


def _project_with_choice(folder, count):
    raw = document_project(authored_documents())
    raw["documents"]["task"]["payload"]["variables"]["width"] = {
        "label": "Width", "type": {"base": "real"},
        "value": {
            "kind": "draw", "family": "choice", "per": "trial",
            "args": [{"kind": "literal", "value": float(index)} for index in range(count)],
        },
    }
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
            assert not info.keys() & {"label", "implicit_names", "level", "requires_period", "random", "stateful", "reset"}


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
