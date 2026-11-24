"""Saved named documents for the Evaluate authoring browser witness."""

import json
from pathlib import Path
import sys

from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.contracts.rollout import ROLLOUT_TYPE
from feedbax.desktop.release import AppPaths
from feedbax.web.services.project_folder import ProjectFolderService
from tests.support.managed_train_fixture import managed_train_project


TARGET_Y = "task.variables.target.value.args.1.value"
"""The Evaluate variation path of the y of the task's target."""


def _train_definition():
    project = managed_train_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    documents = project["documents"]
    train = project["steps"]["train"]
    inline = train["base"]["inline"]
    trial = inline["trial_documents"]
    task = documents[trial["task"]["document"]]["payload"]
    task["fixed_steps"] = 12
    task["epochs"][0]["end"]["length"]["value"] = 12
    del task["validation_trials"]
    (target,) = (variable for variable in task["variables"] if variable["id"] == "target")
    target["value"] = {"kind": "call", "function": "vector", "args": [
        {"kind": "literal", "value": coordinate, "unit": "m"} for coordinate in (0.18, 0.05)
    ]}
    model = documents[trial["model"]["document"]]["payload"]
    model["interior"]["components"]["plant"]["fields"]["mass"] = 1.25
    inline["training_config"]["snapshot_interval"] = 1
    return project


def seed_train_variants(home: str, project_id: str):
    definition = _train_definition()
    definition["sweep"] = {"rows": {"base": {}, "compare": {"from": "base"}}}
    root = AppPaths.resolve(instance_root=Path(home)).web / "projects" / project_id
    project = Project.model_validate(definition)
    for name, data in dump_text_project(project, ProjectConfiguration(project=project_id, store="results")).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return ProjectFolderService(root.parent).open(project_id).snapshot.project_id


def files():
    project = _train_definition()
    documents = project["documents"]
    train = project["steps"]["train"]
    rows = (("control", 731, 0.04), ("comparison", 732, 0.06), ("unselected", 733, 0.08))
    evaluate = {
        "kind": "evaluate",
        "base": {"inline": {"evaluation_type": ROLLOUT_TYPE, "params": {
            "seed": 731, "trial_set": {"kind": "sample", "count": 1},
            "trial_policy": "paired", "variation": {},
        }}},
        "inputs": {"subject": {"step": "train", "checkpoint": "last"}},
        "sweep": {"axes": {"seed": {"values": {
            name: {"value": seed, "patches": {
                "params.seed": {"value": seed},
                "params.variation.row-y": {
                    "op": "add", "value": {"op": "replace", "path": TARGET_Y, "value": y},
                },
            }}
            for name, seed, y in rows
        }}}},
    }
    control = evaluate["sweep"]["axes"]["seed"]["values"]["control"]["patches"]
    parameter = control.pop("params.variation.row-y")["value"]
    evaluate["sweep"]["exceptions"] = {
        "raw-control": {"where": {"rows": ["seed=control"]}, "patches": {
            "params.variation": {"op": "replace", "value": {
                "raw-mass": {"op": "replace", "path": "graph.nodes.plant.params.mass", "value": 2},
                "row-y": parameter,
            }},
        }},
    }
    return [
        *({"kind": "document", "id": name, "document": document} for name, document in documents.items()),
        {"kind": "step", "id": "train", "step": train},
        {"kind": "step", "id": "conditions", "step": evaluate},
    ]


def own_sweep_files():
    """A Train of two seeds and an Evaluate of each of its runs, sampling one trial."""
    project = _train_definition()
    train = project["steps"]["train"]
    train["sweep"] = {"axes": {"seed": {"values": {
        f"s{seed}": {"value": seed, "patches": {"seed": {"op": "add", "value": seed}}} for seed in (0, 1)
    }}}}
    evaluate = {
        "kind": "evaluate",
        "base": {"inline": {"evaluation_type": ROLLOUT_TYPE, "params": {
            "seed": 731, "trial_set": {"kind": "sample", "count": 1}, "trial_policy": "paired", "variation": {},
        }}},
        "inputs": {"subject": {"step": "train", "checkpoint": "last"}},
        "sweep": {"axes": {"trained": {"input": "subject"}}},
    }
    return [
        *({"kind": "document", "id": name, "document": document} for name, document in project["documents"].items()),
        {"kind": "step", "id": "train", "step": train},
        {"kind": "step", "id": "probe", "step": evaluate},
    ]


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments == ["own-sweep-files"]:
        print(json.dumps(own_sweep_files()))
    else:
        print(json.dumps(seed_train_variants(*arguments) if arguments else files()))
