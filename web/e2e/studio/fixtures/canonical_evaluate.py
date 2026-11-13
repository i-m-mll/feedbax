"""Author the signed Evaluate fixture and observe one selected Analysis variable."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.contracts.rollout import ROLLOUT_TYPE
from feedbax.desktop.release import AppPaths
from feedbax.host import HostClient
from feedbax.jobs import JobService
from feedbax.persistence.array_container import ArrayContainerReader
from feedbax.plugins.application import new_application_registry_bundle
from feedbax.studio.canonical_evaluate import RAW_STATES_SCHEMA, state_key
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanService
from tests.test_document_train_pair import authored_documents, document_project


def _mass_variation(mass: int):
    return [{"path": "graph.nodes.plant.params.mass", "op": "replace", "value": mass}]


def _evaluation_step(*, mass: int | None = None):
    return {
        "kind": "evaluate",
        "base": {"inline": {
            "evaluation_type": ROLLOUT_TYPE,
            "params": {
                "seed": 731,
                "trial_set": {"kind": "every_condition"},
                "trial_policy": "paired",
                "variation": [] if mass is None else _mass_variation(mass),
            },
        }},
        "inputs": {"subject": {"step": "train", "checkpoint": "last"}},
    }


def _evaluation_project():
    raw = document_project(authored_documents())
    raw["steps"]["train"]["base"]["inline"]["training_config"]["snapshot_interval"] = 1
    baseline = _evaluation_step()
    raw["steps"].update({
        "baseline": baseline,
        "conditions": baseline | {"sweep": {"axes": {"mass": {"values": {
            name: {"value": mass, "patches": {"params.variation": {"value": _mass_variation(mass)}}}
            for name, mass in (("nominal", 1), ("heavy", 2))
        }}}}},
        "unsupported": baseline | {
            "inputs": {"subject": {"step": "train", "checkpoint": {"best_by": "loss"}}},
        },
    })
    finite = deepcopy(baseline)
    finite["inputs"]["subject"]["checkpoint"] = {"batches": [2, 1]}
    raw["steps"]["finite"] = finite
    return raw


def seed(home: str, project_id: str):
    paths = AppPaths.resolve(instance_root=Path(home))
    folders = ProjectFolderService(paths.web / "projects")
    project = Project.model_validate(_evaluation_project())
    for name, data in dump_text_project(
        project, ProjectConfiguration(project=project_id, store="results"),
    ).items():
        path = folders.root / project_id / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return project_id


def walkthrough_files(batches: str | None = None):
    project = document_project(authored_documents())
    if batches is not None:
        train = project["steps"]["train"]["base"]["inline"]
        train["training_config"]["n_batches"] = int(batches)
        train["method_payload"]["payload"]["target_total"] = int(batches)
    documents = [
        {"kind": "document", "id": name, "document": document}
        for name, document in project["documents"].items()
    ]
    return [*documents, {"kind": "step", "id": "train", "step": project["steps"]["train"]}]


def series_files():
    files = walkthrough_files("4")
    files[-1]["step"]["base"]["inline"]["training_config"]["snapshot_interval"] = 2
    return files


def scoped_series_files():
    evaluation = _evaluation_step()
    evaluation["inputs"]["subject"]["checkpoint"] = {"every": 2}
    evaluation["sweep"] = {"axes": {
        "subject": {"input": "subject"},
        "replicate": {"values": {
            "left": {"patches": {"params.seed": {"value": 31}}},
            "right": {"patches": {"params.seed": {"value": 32}}},
        }},
    }}
    return [*series_files(), {"kind": "step", "id": "evaluate", "step": evaluation}]


def seed_series_variants(home: str):
    *documents, train = series_files()
    evaluation = _evaluation_step()
    evaluation["sweep"] = {"axes": {"subject": {"input": "subject"}}}
    project = Project.model_validate({
        "documents": {file["id"]: file["document"] for file in documents},
        "steps": {"train": train["step"], "evaluate": evaluation},
        "sweep": {"rows": {"base": {}, "comparison": {"from": "base"}}},
    })
    project_id = "series-variants"
    root = AppPaths.resolve(instance_root=Path(home)).web / "projects" / project_id
    for name, data in dump_text_project(
        project, ProjectConfiguration(project=project_id, store="results"),
    ).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return project_id


def observe(home: str, project: str, revision: str, trial_index: str):
    paths = AppPaths.resolve(instance_root=Path(home))
    folders = ProjectFolderService(paths.web / "projects")
    jobs = JobService(HostClient(paths.host), new_application_registry_bundle(), enable_evaluation_candidate=True)
    view = ProjectPlanService(folders, jobs).read(project, revision=revision)
    variable = state_key("plant", "effector/position")
    trial = int(trial_index)
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
        with result.open_states() as dataset:
            coordinates = dataset.data("checkpoint").tolist()
            descriptor = dataset.descriptor
            assert descriptor.schema_id == RAW_STATES_SCHEMA
            assert descriptor.dimensions["state_time"] == 81
            assert descriptor.dimensions["checkpoint"] == len(coordinates)
            assert dataset.data("trial").tolist() == row.trials
            assert len(row.trials) == 28
            label = row.trials[trial]
            assert np.isfinite(dataset.data("objective/total")).all()
            assert np.all(dataset.data("objective/coverage/effort") == 28)
            reads.clear()
            with patch.object(ArrayContainerReader, "read_tensor", read_tensor):
                if row.result.checkpoint_sources:
                    samples = [
                        dataset.select(variable, labels={"checkpoint": coordinate, "trial": label})[0].tolist()
                        for coordinate in coordinates
                    ]
                    expected_reads = ["checkpoint", variable] * len(coordinates)
                else:
                    samples = [dataset.select(variable, labels={"trial": label})[0, 0].tolist()]
                    expected_reads = [variable]
        assert reads == expected_reads, reads
        facts = {
            "key": [key.variant, key.step, key.row],
            "request_id": row.request_id,
            "receipt": result.receipt.model_dump(mode="json"),
            "size_bytes": result.size_bytes,
            "trial_index": trial,
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
        "series-files": series_files, "seed-series-variants": seed_series_variants,
        "scoped-series-files": scoped_series_files,
    }[action](*args)))
