"""Author the signed Evaluate fixture and observe one selected Analysis variable."""

import json
from pathlib import Path
import sys
from unittest.mock import patch

from feedbax.contracts.project import Project, Step
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.desktop.release import AppPaths
from feedbax.host import HostClient
from feedbax.jobs import JobService
from feedbax.persistence.array_container import ArrayContainerReader
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.studio.canonical_evaluate import RAW_STATES_SCHEMA, state_key
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanService
from tests.test_canonical_evaluate_admission import authored_evaluation_project


def seed(home: str, project_id: str):
    paths = AppPaths.resolve(instance_root=Path(home))
    folders = ProjectFolderService(paths.web / "projects")
    project, _ = authored_evaluation_project()
    project.steps.pop("unsupported-condition")
    train = project.steps["train"].model_dump(mode="json", exclude_unset=True)
    train["base"]["inline"]["training_config"]["snapshot_interval"] = 1
    project.steps["train"] = Step.model_validate(train)
    finite = project.steps["baseline"].model_dump(mode="json", exclude_unset=True)
    finite["inputs"]["subject"]["checkpoint"] = {"batches": [2, 1]}
    project.steps["finite"] = Step.model_validate(finite)
    for name, data in dump_text_project(
        project, ProjectConfiguration(project=project_id, store="results"),
    ).items():
        path = folders.root / project_id / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return project_id


def walkthrough_files():
    project, _ = authored_evaluation_project()
    train = project.steps["train"].model_dump(mode="json", exclude_unset=True)
    baseline = project.steps["baseline"].model_dump(mode="json", exclude_unset=True)
    heavy = project.steps["baseline"].model_dump(mode="json", exclude_unset=True)
    heavy["base"]["inline"]["params"]["variation"] = [
        {"path": "nodes.plant.params.mass", "op": "replace", "value": 2},
    ]
    return {
        "train": [{"kind": "step", "id": "train", "step": train}],
        "conditions": [{"kind": "step", "id": name, "step": step}
                       for name, step in (("baseline", baseline), ("heavy", heavy))],
    }


def save_walkthrough_conditions(home: str, project_id: str):
    folders = ProjectFolderService(AppPaths.resolve(instance_root=Path(home)).web / "projects")
    current = folders.open(project_id)
    authored = walkthrough_files()["conditions"]
    steps = {entry["id"]: entry["step"] for entry in authored}
    files = dump_text_project(Project.model_validate({"steps": steps}), current.configuration)
    for name, data in files.items():
        if name.startswith("steps/"):
            with (folders.root / project_id / name).open("xb") as stream:
                stream.write(data)
    return folders.open(project_id).snapshot.revision


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
        with result.open_states() as dataset:
            coordinates = dataset.data("checkpoint").tolist()
            descriptor = dataset.descriptor
            assert descriptor.schema_id == RAW_STATES_SCHEMA
            assert descriptor.dimensions["state_time"] == 101
            assert descriptor.dimensions["checkpoint"] == len(coordinates)
            with patch.object(ArrayContainerReader, "read_tensor", read_tensor):
                if row.result.checkpoint_sources:
                    samples = [
                        dataset.select(variable, labels={"checkpoint": coordinate, "trial": trial})[0].tolist()
                        for coordinate in coordinates
                    ]
                    expected_reads = ["checkpoint", variable] * len(coordinates)
                else:
                    samples = [dataset.select(variable, labels={"trial": trial})[0, 0].tolist()]
                    expected_reads = [variable]
        assert reads == expected_reads, reads
        facts = {
            "key": [key.variant, key.step, key.row],
            "request_id": row.request_id,
            "receipt": result.receipt.model_dump(mode="json"),
            "size_bytes": result.size_bytes,
            "trial": trial,
            "tensor_reads": [name for name in reads if name != "checkpoint"],
        }
        if row.result.checkpoint_sources:
            facts["checkpoint_coordinate_reads"] = reads.count("checkpoint")
            facts["checkpoint_sources"] = [
                source.model_dump(mode="json") for source in row.result.checkpoint_sources
            ]
            facts["checkpoints"] = [
                {"coordinate": coordinate, "samples": selected}
                for coordinate, selected in zip(coordinates, samples, strict=True)
            ]
        else:
            assert row.result.checkpoint_source is not None
            facts["checkpoint_source"] = row.result.checkpoint_source.model_dump(mode="json")
            facts["samples"] = samples[0]
        observed[row.result.job_id] = facts
    return observed


if __name__ == "__main__":
    action, *args = sys.argv[1:]
    print(json.dumps({
        "seed": seed, "observe": observe, "walkthrough-files": walkthrough_files,
        "save-walkthrough-conditions": save_walkthrough_conditions,
    }[action](*args)))
