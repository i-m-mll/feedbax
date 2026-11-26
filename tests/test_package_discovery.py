"""Exercise installed declarations through the Studio backend and managed worker."""

from dataclasses import replace
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
from feedbax.host.light_jobs import AdmittedDeclaration, LightDelivery, publish_light_success
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import open_stored_dataset, publish_dataset
from feedbax.persistence.effective_success import SuccessView, store_request
from feedbax.web.api.figures import router as figure_router
from feedbax.web.api.projects import router
from feedbax.application.composition import load_application
from tests.support.installed_package import install_discovery_fixture
from tests.support.hang import HANG_SECONDS

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
    app.include_router(figure_router, prefix="/api/figures")
    app.state.bootstrap_state = state

    client = TestClient(app)
    response = client.get("/api/projects/authoring/catalog")
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
        if family != "components":
            assert "revision" not in entry
        if family == "operations":
            assert entry["static_parameters"] == ["unit"]
            assert "default" not in entry["parameter_json_schema"]["properties"]["increments"]

    templates = client.get("/api/figures/templates")
    assert templates.status_code == 200
    (external_plot,) = [
        item for item in templates.json() if item["name"] == "entry_point_provider.external_plot"
    ]
    metadata = external_plot["metadata"]
    assert metadata["distribution"] == "feedbax-discovery-fixture"
    assert metadata["version"] == "2.7.1"
    increments = metadata["settings_schema"]["properties"]["increments"]
    assert (increments["type"], increments["items"]["type"]) == ("array", "integer")

    resolved = state.bundle.operations.resolve("entry_point_provider.offset_series")
    from entry_point_provider.declarations.examples import shift_series

    assert resolved.key == _ANALYSIS
    assert resolved.source.distribution == "feedbax-discovery-fixture"
    assert resolved.source.version == "2.7.1"
    assert resolved.defining_object is shift_series


def _shift_delivery(installed_site, tmp_path, declaration=None):
    """An Analyze delivery of the installed operation over a stored source series, and how to run its worker."""
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
    spec = AnalysisRequestSpec(analysis_type=_ANALYSIS, params={"increments": [3, 4], "unit": "test"})
    request = store_request(store, encode_request(spec, {
        "source": (InputMember(key="source", reference=ProductRef(request_id=producer.id, role="source")),),
    }))
    declaration = declaration or AdmittedDeclaration.of(load_application().bundle.operations.resolve(_ANALYSIS))
    delivery = LightDelivery(project_id="fixture", manifest_root=store.root, request=request, declaration=declaration)
    payload = tmp_path / "delivery.json"
    payload.write_text(delivery.model_dump_json())
    origins = {producer.id: published.ref.model_dump(mode="json")}
    bindings = tmp_path / "bindings.json"
    raw_bindings = json.dumps({"bindings": [], "origins": origins}).encode()
    bindings.write_bytes(raw_bindings)
    log = tmp_path / "events.jsonl"
    log.touch()

    def run_worker():
        return subprocess.run(
            [sys.executable, "-c", "from feedbax.host.light_worker import main; raise SystemExit(main([]))"],
            env=_child_environment(installed_site) | {
                "FEEDBAX_HOST_JOB_ID": "test-attempt",
                "FEEDBAX_HOST_EVENT_LOG": str(log),
                "FEEDBAX_HOST_INPUT": str(payload),
                "FEEDBAX_HOST_COMPLETION_BINDINGS": str(bindings),
                "FEEDBAX_HOST_COMPLETION_BINDINGS_SHA256": sha256(raw_bindings).hexdigest(),
            },
            timeout=HANG_SECONDS, text=True, capture_output=True,
        )

    def publish(provider):
        result = json.loads(log.read_text())["data"]
        return publish_light_success(
            delivery.model_dump(mode="json"), "test-attempt", request.id, origins, result["products"],
            provider=provider,
        )

    return store, run_worker, publish, log


def test_managed_worker_runs_the_installed_operation(installed_site, monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(installed_site))
    store, run_worker, publish, log = _shift_delivery(installed_site, tmp_path)
    worker = run_worker()
    assert worker.returncode == 0, worker.stderr
    result = json.loads(log.read_text())["data"]
    assert result["outcome"] == "succeeded"
    publish(result["provider"])
    receipt = SuccessView(store).read_attempt("test-attempt").receipt
    provider = receipt.provider
    assert provider is not None
    assert (provider.package, provider.distribution, provider.version) == (
        "entry_point_provider.declarations", "feedbax-discovery-fixture", "2.7.1",
    )
    (shifted,) = receipt.products["shifted"]
    with open_stored_dataset(store, DatasetProductRef(
        content=shifted.content, descriptor_sha256=shifted.descriptor["descriptor_sha256"],
    )) as data:
        np.testing.assert_array_equal(data.data("value"), [9, 11])


def test_a_worker_refuses_a_declaration_that_changed_after_planning(installed_site, monkeypatch, tmp_path):
    """Caught: a result computed by code other than the code that planned the step, committed as if it were."""
    monkeypatch.syspath_prepend(str(installed_site))
    admitted = AdmittedDeclaration.of(load_application().bundle.operations.resolve(_ANALYSIS))
    planned = admitted.model_copy(update={"provider": replace(admitted.provider, version="2.7.0")})
    store, run_worker, _, log = _shift_delivery(installed_site, tmp_path, planned)
    assert run_worker().returncode != 0
    assert not log.read_text(), "the worker computed under a declaration it was not admitted with"


def test_the_host_refuses_a_result_from_another_provider(installed_site, monkeypatch, tmp_path):
    """Caught: the Host committing a result whose provider is not the one the job was admitted with."""
    monkeypatch.syspath_prepend(str(installed_site))
    store, run_worker, publish, log = _shift_delivery(installed_site, tmp_path)
    worker = run_worker()
    assert worker.returncode == 0, worker.stderr
    provider = json.loads(log.read_text())["data"]["provider"]
    with pytest.raises(ValueError):
        publish(provider | {"version": "9.9.9"})
    with pytest.raises(ValueError):
        publish(None)
    assert SuccessView(store).attempt_ref("test-attempt") is None


def test_describe_fails_if_an_installed_package_cannot_import(installed_site, tmp_path):
    broken_site = shutil.copytree(installed_site, tmp_path / "broken-site")
    (broken_site / "entry_point_provider" / "declarations" / "broken.py").write_text(
        "raise ImportError('fixture import failed')\n",
    )
    result = subprocess.run(
        [sys.executable, "-m", "feedbax", "describe"],
        env=_child_environment(broken_site),
        timeout=HANG_SECONDS, text=True, capture_output=True,
    )
    assert result.returncode == 2, result.stderr
    assert not result.stdout
    assert "entry_point_provider.declarations.broken" in result.stderr
    assert "feedbax-discovery-fixture" in result.stderr
