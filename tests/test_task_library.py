"""New offers the task documents every installed package ships, read whole from their library folders."""

import json
import os
import sys

from feedbax.application.composition import load_application
from feedbax.contracts.task_document import BLANK_TASK, TASK_DOCUMENT_SCHEMA_ID
from feedbax.studio.authoring import authoring_surface


def test_new_lists_an_installed_package_s_task_documents_whole_beside_feedbax_s(tmp_path, monkeypatch):
    plugins = tmp_path / "plugins"
    documents = plugins / "reach_library" / "examples" / "curved" / "documents"
    documents.mkdir(parents=True)
    (plugins / "reach_library" / "__init__.py").write_text("")
    task = BLANK_TASK.model_copy(update={"id": "curved"}).model_dump(mode="json")
    (documents / "curved-task.json").write_text(json.dumps({"label": "Document", "schema": TASK_DOCUMENT_SCHEMA_ID, "payload": task}))
    (documents / "longer-task.json").write_text(json.dumps({
        "label": "Document",
        "schema": TASK_DOCUMENT_SCHEMA_ID, "base": {"document": "curved-task"},
        "differences": {"/fixed_steps": {"value": 40}},
    }))
    (documents / "model.json").write_text(json.dumps({"label": "Document", "schema": "feedbax.spec.model", "payload": {"step_seconds": 0.01, "ports": {}, "interior": {}}}))
    metadata = plugins / "reach_library-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: reach-library\nVersion: 1.0\n")
    (metadata / "entry_points.txt").write_text("[feedbax.plugins]\nreach_library = reach_library\n")
    monkeypatch.syspath_prepend(str(plugins))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, (str(plugins), os.environ.get("PYTHONPATH")))))
    try:
        catalog = authoring_surface(load_application(), catalog=True).tasks
    finally:
        sys.modules.pop("reach_library", None)
    offered = {(entry.distribution, entry.library, entry.name): entry.document for entry in catalog.documents}
    assert ("feedbax", "point-mass-reaching", "simple-reaches-task") in offered
    assert {key for key in offered if key[0] == "reach-library"} == {
        ("reach-library", "curved", "curved-task"), ("reach-library", "curved", "longer-task"),
    }
    longer = offered["reach-library", "curved", "longer-task"]
    assert longer.payload == task | {"fixed_steps": 40}
    assert catalog.blank == BLANK_TASK
