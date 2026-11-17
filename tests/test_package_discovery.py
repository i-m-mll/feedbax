"""Exercise installed declarations through the Studio backend and managed worker."""

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest

from feedbax.cli import main
from feedbax.contracts.dataset import DATASET_MEDIA_TYPE, DatasetProductRef
from feedbax.contracts.effective_success import Product
from feedbax.contracts.request_identity import encode_request
from feedbax.contracts.request_refs import InputMember, ProductRef
from feedbax.contracts.request_specs import AnalysisRequestSpec
from feedbax.dataset import Dataset, Variable
from feedbax.light_jobs import LightDelivery, publish_light_success
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import open_stored_dataset, publish_dataset
from feedbax.persistence.effective_success import SuccessView, store_request
from feedbax.web.api.projects import router
from feedbax.plugins.composition import load_application
from tests.support.installed_package import install_discovery_fixture

_ANALYSIS = "entry_point_provider.shift_series"


@pytest.fixture(scope="module")
def installed_site(tmp_path_factory):
    return install_discovery_fixture(tmp_path_factory.mktemp("installed-declarations"))


def _child_environment(installed_site: Path) -> dict[str, str]:
    paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    return os.environ | {"PYTHONPATH": os.pathsep.join((str(installed_site), *filter(None, paths)))}


def test_installed_package_appears_in_cli_and_studio_catalogs(installed_site, monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(installed_site))
    state = load_application()
    app = FastAPI()
    app.include_router(router)
    app.state.bootstrap_state = state

    response = TestClient(app).get("/api/projects/authoring/catalog")
    assert response.status_code == 200
    assert main(["describe"]) == 0
    catalog = response.json()
    assert json.loads(capsys.readouterr().out) == catalog

    entries = catalog["registries"]
    for family, key in {
        "components": "entry_point_provider.ExternalGain",
        "python_tasks": "entry_point_provider.ExternalTrial",
        "operations": _ANALYSIS,
        "training_methods": "entry_point_provider.ExternalMethod",
        "figure_templates": "entry_point_provider.external_plot",
    }.items():
        registry = entries[family]
        (entry,) = [item for item in registry["entries"] if item.get("key", item.get("name")) == key]
        assert registry["origins"][key] == {
            "package": "entry_point_provider.declarations",
            "distribution": "feedbax-discovery-fixture",
            "version": "2.7.1",
        }
        if family == "operations":
            assert entry["static_parameters"] == ["unit"]
            assert "default" not in entry["parameter_json_schema"]["properties"]["increments"]


def test_managed_worker_runs_the_installed_operation(installed_site, monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(installed_site))
    store = ImmutableArtifactBlobProvider(tmp_path / "store")
    dataset = Dataset.from_arrays(
        {"value": Variable(np.array([2, 4]), ("step",))}, schema_id="fixture.series.v1",
    )
    source = publish_dataset(store, dataset)
    producer = store_request(store, encode_request(AnalysisRequestSpec(analysis_type="fixture.source"), {}))
    published = SuccessView(store).publish(
        producer, attempt="fixture-source", products={
            "source": (Product(
                name="source", content=source.content.model_copy(update={"media_type": DATASET_MEDIA_TYPE}),
                descriptor={"descriptor_sha256": source.descriptor_sha256},
            ),),
        },
    )
    request = store_request(store, encode_request(AnalysisRequestSpec(analysis_type=_ANALYSIS), {
        "source": (InputMember(key="source", reference=ProductRef(request_id=producer.id, role="source")),),
    }))
    delivery = LightDelivery(project_id="fixture", manifest_root=store.root, request=request)
    payload = tmp_path / "delivery.json"
    payload.write_text(delivery.model_dump_json())
    origins = {producer.id: published.ref.model_dump(mode="json")}
    bindings = tmp_path / "bindings.json"
    raw_bindings = json.dumps({"bindings": [], "origins": origins}).encode()
    bindings.write_bytes(raw_bindings)
    log = tmp_path / "events.jsonl"
    log.touch()
    worker = subprocess.run(
        [sys.executable, "-c", "from feedbax.light_host_worker import main; raise SystemExit(main([]))"],
        env=_child_environment(installed_site) | {
            "FEEDBAX_HOST_JOB_ID": "test-attempt",
            "FEEDBAX_HOST_EVENT_LOG": str(log),
            "FEEDBAX_HOST_INPUT": str(payload),
            "FEEDBAX_HOST_COMPLETION_BINDINGS": str(bindings),
            "FEEDBAX_HOST_COMPLETION_BINDINGS_SHA256": sha256(raw_bindings).hexdigest(),
        },
        timeout=45, text=True, capture_output=True,
    )
    assert worker.returncode == 0, worker.stderr
    result = json.loads(log.read_text())["data"]
    assert result["outcome"] == "succeeded"
    publish_light_success(delivery.model_dump(mode="json"), "test-attempt", request.id, origins, result["products"])
    (shifted,) = SuccessView(store).read_attempt("test-attempt").receipt.products["shifted"]
    with open_stored_dataset(store, DatasetProductRef(
        content=shifted.content, descriptor_sha256=shifted.descriptor["descriptor_sha256"],
    )) as data:
        np.testing.assert_array_equal(data.data("value"), [7, 9])


def test_describe_fails_if_an_installed_package_cannot_import(installed_site, tmp_path):
    broken_site = shutil.copytree(installed_site, tmp_path / "broken-site")
    (broken_site / "entry_point_provider" / "declarations" / "broken.py").write_text(
        "raise ImportError('fixture import failed')\n",
    )
    result = subprocess.run(
        [sys.executable, "-m", "feedbax", "describe"],
        env=_child_environment(broken_site),
        timeout=45, text=True, capture_output=True,
    )
    assert result.returncode == 2, result.stderr
    assert not result.stdout
    assert "entry_point_provider.declarations.broken" in result.stderr
    assert "feedbax-discovery-fixture" in result.stderr
