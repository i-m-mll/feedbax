"""Read ordered retained producer evidence for the mounted Analyze result."""

import json
from pathlib import Path
import sys

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.desktop.release import AppPaths
from feedbax.host import HostClient
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView


def main(instance: str, project: str, alignment: str) -> None:
    paths = AppPaths.resolve(instance_root=Path(instance))
    store = paths.web / "projects" / project / "results"
    view = SuccessView(ImmutableArtifactBlobProvider(store))
    jobs = HostClient(paths.host).jobs_snapshot(project=project, start_host=False)["jobs"]
    job = next(job for job in jobs if job["id"] == alignment)
    ref = ContentObjectRef.model_validate(job["outcome"]["store_success"])
    receipt = view.read_success(ref).receipt
    members = []
    for item in receipt.inputs["states"]:
        producer = view.read_success(item.selection.receipt).receipt
        members.append({
            "key": item.key,
            "job_id": producer.attempt,
            "request_id": producer.request.id,
            "receipt": item.selection.receipt.sha256,
        })
    print(json.dumps({"members": members, "producers": [edge["job_id"] for edge in job["edges"]]}))


if __name__ == "__main__":
    main(*sys.argv[1:])
