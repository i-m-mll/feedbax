"""Read ordered retained producer evidence for the mounted Analyze result."""

import json
from pathlib import Path
import sys

import numpy as np

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.request_specs import EvaluationRequestSpec
from feedbax.desktop.release import AppPaths
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.host import HostClient
from feedbax.host.light_jobs import dataset_member
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.dataset import open_stored_dataset
from feedbax.persistence.effective_success import SuccessView


def figure_samples(view: SuccessView, figure: str) -> list[dict]:
    published = view.read_attempt(figure)
    assert published is not None
    (summary,) = published.receipt.inputs["summary"]
    with open_stored_dataset(view.store, dataset_member(summary), verify=True) as dataset:
        time = dataset.data("state_time").tolist()
        dims = list(dataset.descriptor.variables["mean"].dims)
        mean = np.take(np.asarray(dataset.data("mean")), 0, axis=dims.index("checkpoint"))
        dims.remove("checkpoint")
        rows = np.moveaxis(mean, dims.index("state_time"), -1).reshape(-1, len(time))
        conditions = np.asarray(dataset.data("condition")).reshape(-1).tolist()
        return [
            {"name": condition, "x": time, "y": row.tolist()}
            for condition, row in zip(conditions, rows, strict=True)
        ]


def main(instance: str, project: str, alignment: str, figure: str) -> None:
    paths = AppPaths.resolve(instance_root=Path(instance))
    _, location = ProjectFolderService(paths.web / "projects").open_native_id(project)
    store = location.path / "results"
    view = SuccessView(ImmutableArtifactBlobProvider(store))
    jobs = HostClient(paths.host).jobs_snapshot(project=project, start_host=False)["jobs"]
    job = next(job for job in jobs if job["id"] == alignment)
    ref = ContentObjectRef.model_validate(job["outcome"]["store_success"])
    receipt = view.read_success(ref).receipt
    members = []
    conditions = []
    for item in receipt.inputs["states"]:
        selected = item.selection.receipt
        producer = view.read_success(selected).receipt
        assert producer.evaluation_manifest is not None
        assert view.attempt_ref(producer.attempt) == selected
        request = view.read_request(producer.request).spec
        assert isinstance(request, EvaluationRequestSpec)
        conditions.append(request.params)
        members.append({
            "key": item.key,
            "job_id": producer.attempt,
            "request_id": producer.request.id,
            "evaluation_manifest_sha256": producer.evaluation_manifest.sha256,
            "store_success_sha256": selected.sha256,
        })
    print(json.dumps({
        "members": members,
        "producers": [edge["job_id"] for edge in job["edges"]],
        "condition_inputs": conditions,
        "figure_samples": figure_samples(view, figure),
    }, allow_nan=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
