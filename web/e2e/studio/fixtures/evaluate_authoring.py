"""Saved named documents for the Evaluate authoring browser witness."""

import json

from feedbax.contracts.rollout import ROLLOUT_TYPE
from tests.support.managed_train_fixture import managed_train_project


def files():
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
    print(json.dumps(files()))
