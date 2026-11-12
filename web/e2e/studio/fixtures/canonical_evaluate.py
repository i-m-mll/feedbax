"""Author the signed Evaluate fixture and observe one selected Analysis variable."""

import json
from pathlib import Path
import sys
from unittest.mock import patch

from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.desktop.release import AppPaths
from feedbax.host import HostClient
from feedbax.jobs import JobService
from feedbax.persistence.array_container import ArrayContainerReader
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.studio.canonical_evaluate import state_key
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanService
from tests.test_canonical_evaluate_admission import authored_evaluation_project


def seed(home: str, project_id: str):
    paths = AppPaths.resolve(instance_root=Path(home))
    folders = ProjectFolderService(paths.web / "projects")
    project, _ = authored_evaluation_project()
    project.steps.pop("unsupported-condition")
    for name, data in dump_text_project(
        project, ProjectConfiguration(project=project_id, store="results"),
    ).items():
        path = folders.root / project_id / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return project_id


def observe(home: str, project: str, revision: str, trial: str):
    paths = AppPaths.resolve(instance_root=Path(home))
    folders = ProjectFolderService(paths.web / "projects")
    jobs = JobService(HostClient(paths.host), new_application_registry_bundle(), enable_evaluation_candidate=True)
    view = ProjectPlanService(folders, jobs).read(project, revision=revision)
    variable = state_key("plant", "effector/position")
    reads = []
    original = ArrayContainerReader.read_tensor

    def read_tensor(reader, name):
        reads.append(name)
        return original(reader, name)

    observed = {}
    for key, row in view.evaluations.items():
        if row.result is None:
            continue
        result = jobs.read_evaluation_result(
            project=project, store=view.settled.store_path, job_id=row.result.job_id,
        )
        assert result.receipt == row.result.receipt
        assert result.size_bytes == row.result.size_bytes
        reads.clear()
        with patch.object(ArrayContainerReader, "read_tensor", read_tensor), result.open_states() as dataset:
            samples = dataset.select(variable, labels={"trial": trial})[0, 0, :].tolist()
        assert reads == [variable]
        observed[row.result.job_id] = {
            "key": [key.variant, key.step, key.row],
            "request_id": row.request_id,
            "receipt": result.receipt.model_dump(mode="json"),
            "checkpoint_source": row.result.checkpoint_source.model_dump(mode="json"),
            "size_bytes": result.size_bytes,
            "trial": trial,
            "tensor_reads": list(reads),
            "samples": samples,
        }
    return observed


if __name__ == "__main__":
    action, *args = sys.argv[1:]
    print(json.dumps({"seed": seed, "observe": observe}[action](*args)))
