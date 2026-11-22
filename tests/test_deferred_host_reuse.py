"""Exercise pending reuse with a reopened scheduler and exact retained products."""

from dataclasses import asdict, replace
from datetime import UTC, datetime
from hashlib import sha256
import json
import sys

import numpy as np
import pytest

from feedbax.contracts.code_validity import AttemptProvenance, CodeTrace, FileSnapshot, ReceiptEvidence, SourcePath
from feedbax.contracts.dataset import DATASET_MEDIA_TYPE
from feedbax.contracts.effective_success import Product
from feedbax.contracts.figure_document import FigureDocument, PLOTLY_MEDIA_TYPE
from feedbax.contracts.operation import DatasetOutputPort, OperationDefinition
from feedbax.contracts.report_document import ReportDocument, ReportFigure
from feedbax.contracts.request_identity import encode_request
from feedbax.contracts.request_refs import ContentRef, InputMember, ProductRef
from feedbax.contracts.request_specs import AnalysisRequestSpec, RenderRequestSpec, ReportRequestSpec
from feedbax.dataset import Dataset, Variable
from feedbax.host.protocol import AdmissionItem, DeferredReuse, Edge, Job, Launch, LIGHT_ENTRYPOINT
from feedbax.host.store import Store
from feedbax.light_jobs import LightDelivery
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import publish_dataset
from feedbax.persistence.effective_success import SuccessView, store_request
from feedbax.validity.admission_projection import AdmissionProjection
from feedbax.persistence.evidence import stage_evidence


@pytest.mark.parametrize("action", ["reuse", "stop", "rerun", "producer_stopped", "product_missing", "file_changed"])
def test_deferred_reuse_keeps_exact_result_across_reopening(tmp_path, action):
    path = tmp_path / "jobs.sqlite3"
    store = Store(path)
    blobs = ImmutableArtifactBlobProvider(tmp_path / "products")
    schema = "fixture.series.v1"
    definition = OperationDefinition(
        key="fixture.series", owner="fixture", owner_version="1", label="Series", description="Test input",
        param_schema=[], parameter_json_schema={},
        static_parameters=(), inputs={}, outputs={"series": DatasetOutputPort(schema_ids=(schema,))},
    ).model_dump(mode="json")
    series = store_request(blobs, encode_request(AnalysisRequestSpec(analysis_type="fixture.series", params={}), {}))
    figures = FigureDocument.model_validate({
        "data": ["series"], "layout": [{"name": "main", "row": 1, "col": 1}],
        "layers": [{"name": "series", "role": "series", "panel": "main", "mark": "line",
                    "encode": {"x": "time", "y": "value"}}],
    })
    plot = store_request(blobs, encode_request(RenderRequestSpec(document=figures), {
        "series": (InputMember(key="data", reference=ProductRef(request_id=series.id, role="series")),),
    }))
    report = store_request(blobs, encode_request(
        ReportRequestSpec(document=ReportDocument(title="Report", blocks=(ReportFigure(figure="plot"),))),
        {"plot": (InputMember(key="figure", reference=ProductRef(request_id=plot.id, role="figure")),)},
    ))
    data = publish_dataset(blobs, Dataset.from_arrays(
        {"value": Variable(np.array([2.0, 3.0]), ("time",))},
        coords={"time": Variable(np.array([0.0, 1.0]), ("time",))},
        schema_id=schema,
    ))
    series_product = Product(
        name="series", content=data.content.model_copy(update={"media_type": DATASET_MEDIA_TYPE}),
        descriptor={"descriptor_sha256": data.descriptor_sha256},
    )
    figure = blobs.store_object(b'{"data":[{"x":[0,1],"y":[2,3]}]}')
    figure_product = Product(
        name="figure", content=ContentRef(sha256=figure.sha256, size=figure.size_bytes, media_type=PLOTLY_MEDIA_TYPE),
    )
    configuration = tmp_path / "ticks.txt"
    configuration.write_text("two")
    tracked = FileSnapshot(
        path=SourcePath(scope="project", path=configuration.name),
        sha256=sha256(configuration.read_bytes()).hexdigest(),
    )

    def job(attempt, request, parent=None):
        return Job(
            attempt, "project", request.id,
            Launch((sys.executable,), payload=LightDelivery(
                project_id="project", manifest_root=blobs.root, request=request,
            ).model_dump(mode="json"), warm_entrypoint=LIGHT_ENTRYPOINT, project_root=str(tmp_path)),
            "light", edges=(Edge(parent),) if parent else (),
            data={"manifest_root": str(blobs.root), "run_key": attempt},
        )

    def publish(attempt, request, product, origins):
        now = datetime.now(UTC)
        evidence = stage_evidence(blobs, ReceiptEvidence(
            attempt=attempt,
            trace=CodeTrace(
                sources=(), registry=(), files=(tracked,) if request == plot else (),
                environment={}, indeterminate=(),
            ),
            provenance=AttemptProvenance(
                feedbax_commit="fixture", feedbax_dirty=False, lock_sha256=None, platform="fixture",
                started_at=now, ended_at=now,
            ),
        ))
        view = SuccessView(blobs)
        _, bound = view.bind(request, origins=origins)
        result = view.publish_bound(
            request, attempt=attempt, products={product.name: (product,)}, bindings=bound, execution_evidence=evidence,
        )
        with store.db:
            store.finished(store.get(attempt), {
                "outcome": "succeeded", "store_success": result.ref.model_dump(mode="json"),
                "receipt": {"manifest_id": attempt, "manifest_sha256": result.ref.sha256},
            })
        return result.ref

    old_series, old_plot = job("old-series", series), job("old-plot", plot, "old-series")
    store.admit("initial", [
        asdict(AdmissionItem("series", old_series, light_definition=definition)),
        asdict(AdmissionItem("plot", old_plot)),
    ], {})
    upstream = publish(old_series.id, series, series_product, {})
    retained = publish(old_plot.id, plot, figure_product, {series.id: upstream})

    new_series, new_plot = job("new-series", series), job("new-plot", plot, "new-series")
    dependent = job("report", report, new_plot.id)
    items = [
        asdict(AdmissionItem("series", new_series, light_definition=definition)),
        asdict(AdmissionItem("plot", new_plot, deferred_reuse=DeferredReuse(
            old_plot.id, retained.model_dump(mode="json"), "selected", 2.0,
        ))),
        asdict(AdmissionItem("report", dependent)),
    ]
    record = {"project": "project", "deferred_projection": AdmissionProjection.capture(
        {"project": tmp_path}, (), {},
    ).model_dump(mode="json")}
    acknowledged = store.admit("joint", items, record)
    assert [row["id"] for row in store.ready()] == [new_series.id]
    store.db.close()
    store = Store(path)
    if action == "stop":
        store.stop([new_plot.id], "force")
    elif action == "rerun":
        attached = store.admit("force", [
            asdict(AdmissionItem("plot", replace(new_plot, id="forced-plot"), force_execution=True)),
        ], {})
        assert attached["items"][0]["job_id"] == new_plot.id
    elif action == "file_changed":
        configuration.write_text("three")
    if action == "producer_stopped":
        store.stop([new_series.id], "force")
    else:
        publish(new_series.id, series, series_product, {})
    if action == "product_missing":
        blobs.object_path(data.object_ref).unlink()
    store.settle_blocked()
    store.deferred.settle()
    store.settle_blocked()

    reserved = store.get(new_plot.id)
    view = SuccessView(blobs)
    assert reserved["delivery"] == 0 and view.attempt_ref(new_plot.id) is None
    assert view.authored_ref(plot.id) == retained
    assert store.admit("joint", items, record) == acknowledged
    if action == "reuse":
        assert reserved["state"] == "reused"
        assert json.loads(reserved["outcome"])["retained_attempt"] == old_plot.id
        assert [row["id"] for row in store.ready()] == [dependent.id]
        assert store.light_origins(store.get(dependent.id)) == {plot.id: retained.model_dump(mode="json")}
    elif action in {"rerun", "file_changed"}:
        assert [row["id"] for row in store.ready()] == [new_plot.id]
    else:
        assert reserved["state"] == ("stopped" if action == "stop" else "blocked")
        if action == "product_missing":
            assert json.loads(reserved["outcome"])["error"]["code"] == "dependency_corrupt"
        assert store.get(dependent.id)["state"] == "blocked"
        assert store.ready() == []
    store.db.close()
