"""Saved named documents for the Evaluate authoring browser witness."""

import json
from pathlib import Path
import sys

from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.contracts.rollout import ROLLOUT_TYPE
from feedbax.desktop.release import AppPaths
from tests.support.managed_train_fixture import managed_train_project


def _train_definition():
    project = managed_train_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    documents = project["documents"]
    train = project["steps"]["train"]
    inline = train["base"]["inline"]
    trial = inline["trial_documents"]
    task = trial.pop("task")
    trial["python_task"] = task
    documents[task["document"]] = {
        "schema": "feedbax.spec.python_task",
        "payload": {"type": "feedbax.task.reaching.v1", "params": {"n_steps": 12, "x": 0.15, "y": 0.05}},
    }
    model = documents[trial["model"]["document"]]["payload"]
    model["ports"]["inputs"][0]["name"] = "reach_target"
    train["document_overrides"] = {
        trial["model"]["document"]: {"/interior/components/plant/fields/mass": 1.25},
        task["document"]: {"/params/x": 0.18},
    }
    inline["training_config"].update(n_reach_steps=12, snapshot_interval=1)
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
    return project_id


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
                    "op": "add", "value": {"op": "replace", "path": "task.params.y", "value": y},
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


if __name__ == "__main__":
    print(json.dumps(seed_train_variants(*sys.argv[1:]) if sys.argv[1:] else files()))
