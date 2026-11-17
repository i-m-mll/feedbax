"""Verify authoring discovery against the composed registries and Studio route."""

import asyncio
import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from feedbax.cli import main
from feedbax.contracts.analysis_document import AnalysisDocument
from feedbax.contracts.figure_document import FIGURE_DOCUMENT_SCHEMA, FigureDocument
from feedbax.contracts.project import ExternalBase, Step
from feedbax.contracts.report_document import ReportDocument
from feedbax.contracts.request_identity import REQUEST_KINDS
from feedbax.plugins import bootstrap
from feedbax.plugins.composition import compose_application
from feedbax.web.api.projects import router
from tests.support import training_fixtures


def test_cli_authoring_catalog_matches_studio_and_registered_declarations(
    tmp_path,
    monkeypatch,
    capsys,
):
    plugin = training_fixtures.PLUGIN_REGISTRATION
    entry_point = SimpleNamespace(
        name="authoring-fixture",
        value="tests.support.training_fixtures:PLUGIN_REGISTRATION",
        dist=SimpleNamespace(name="feedbax-authoring-fixture", version="1.0"),
        load=lambda: plugin,
    )
    monkeypatch.setattr(bootstrap, "_installed_entry_points", lambda group: (entry_point,))
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(tmp_path / "host"))
    state = asyncio.run(compose_application())
    app = FastAPI()
    app.include_router(router)
    app.state.bootstrap_state = state
    client = TestClient(app)
    legacy = client.get("/api/projects/authoring/schema")
    response = client.get("/api/projects/authoring/catalog")
    assert legacy.status_code == response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    catalog = response.json()

    assert main(["describe"]) == 0
    assert json.loads(capsys.readouterr().out) == catalog
    assert not (tmp_path / "host").exists()

    steps = catalog["step_payloads"]
    assert catalog["step_grammar"] == Step.model_json_schema(by_alias=True)
    assert steps == {
        "train": REQUEST_KINDS["train"].spec_type.model_json_schema(),
        "evaluate": REQUEST_KINDS["evaluate"].spec_type.model_json_schema(),
        "analyze": AnalysisDocument.model_json_schema(),
        "report": ReportDocument.model_json_schema(),
    }
    families = catalog["document_families"]
    assert families["figure"] == {
        "schema_id": FIGURE_DOCUMENT_SCHEMA,
        "payload_schema": TypeAdapter(FigureDocument | ExternalBase).json_schema(),
    }
    assert families["analysis"]["payload_schema"] == steps["analyze"]
    assert families["report"]["payload_schema"] == steps["report"]

    registries = catalog["registries"]
    components = registries["components"]["entries"]
    assert [entry["name"] for entry in components] == state.bundle.components.names()
    assert components == [
        entry.model_dump(mode="json")
        for entry in sorted(state.bundle.components.list_all(), key=lambda entry: entry.name)
    ]
    operations = registries["operations"]["entries"]
    assert [entry["key"] for entry in operations] == list(state.bundle.operations.keys())
    assert operations == [
        definition.model_dump(mode="json") for definition in state.bundle.operations.definitions()
    ]
    methods = registries["training_methods"]["entries"]
    training = state.bundle.training_programs
    assert [entry["key"] for entry in methods] == list(training.available_keys())
    for entry in methods:
        method = training.resolve(entry["key"], path="/method_ref")
        assert entry["payload_schema"] == method.payload_model.model_json_schema()
    assert registries["training_methods"]["origins"][training_fixtures.CLI_METHOD_REF] == {
        "plugin": plugin.declaration.plugin_id,
        "distribution": entry_point.dist.name,
    }

    assert legacy.json() == {
        "project_grammar": catalog["project_grammar"],
        "step_payloads": {"train": steps["train"]},
        "document_schemas": {
            families[name]["schema_id"]: families[name]["payload_schema"]
            for name in ("model", "task", "objective")
        },
        "training_methods": methods,
    }
    for selector, expected in (
        ("family:figure", families["figure"]),
        (FIGURE_DOCUMENT_SCHEMA, families["figure"]),
        ("report", steps["report"]),
        ("registry:operations", registries["operations"]),
    ):
        assert main(["describe", selector]) == 0
        assert json.loads(capsys.readouterr().out) == expected

    assert main(["describe", "receipt"]) == 2
    rejected = capsys.readouterr()
    assert not rejected.out
    assert "unknown authoring selector 'receipt'" in rejected.err
    assert "step:report" in rejected.err and "family:report" in rejected.err
