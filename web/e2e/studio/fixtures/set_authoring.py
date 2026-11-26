"""Seed projects whose steps read Sets."""

import json
import sys
from copy import deepcopy

from web.e2e.studio.fixtures.canonical_evaluate import _seed


def seed(home: str, project_id: str):
    """An Evaluate step over two Train steps' models, and a third Train step that reads the same Model."""
    from tests.test_saved_light_run import _saved_light_definition

    raw = _saved_light_definition(paired=True)
    steps = raw["steps"]
    for name in ("baseline", "report", "analyze"):
        steps.pop(name, None)
    steps["train_b"] = deepcopy(steps["train"]) | {"label": "Train b"}
    steps["train_b"]["base"]["inline"]["settings"]["optimizer"]["learning_rate"]["value"] = 0.003
    steps["other"] = deepcopy(steps["train"]) | {"label": "Other"}
    evaluate = steps.pop("conditions")
    evaluate["label"] = "Pair"
    evaluate["sweep"] = {"axes": {"subject": {"input": "subject"}}}
    evaluate["inputs"]["subject"] = [{"step": "train", "checkpoint": "last"}, {"step": "train_b", "checkpoint": "last"}]
    steps["pair"] = evaluate
    for name in ("velocity-analysis", "velocity-figure", "velocity-report"):
        raw["documents"].pop(name, None)
    return _seed(home, project_id, raw)


def seed_entries(home: str, project_id: str):
    """A Train step whose required sweep reads a Set of two entries with no differences."""
    from tests.test_saved_light_run import _saved_light_definition

    raw = _saved_light_definition(paired=True)
    raw["steps"] = {"train": raw["steps"]["train"]}
    for name in ("velocity-analysis", "velocity-figure", "velocity-report"):
        raw["documents"].pop(name, None)
    raw["documents"]["set"] = {
        "schema": "feedbax.set_document.v1", "label": "Set", "payload": {"entries": {"light": {"label": "Light"}, "heavy": {"label": "Heavy"}}},
    }
    raw["steps"]["train"]["sweep"] = {"axes": {"set": {"set": "set"}}}
    return _seed(home, project_id, raw)


def seed_window(home: str, project_id: str):
    """A Model force pulse whose onset is declared by one Set entry."""
    from tests.support.managed_train_fixture import managed_train_project

    raw = managed_train_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    raw["documents"]["pulses"] = {
        "schema": "feedbax.set_document.v1",
        "label": "Pulses",
        "payload": {"entries": {
            "nominal": {},
            "force": {
                "task_variables": [{
                    "id": "onset", "name": "Onset",
                    "type": {"base": "real", "dimensions": {"time": 1}},
                    "value": {"kind": "literal", "value": 0.1, "unit": "s"},
                }],
                "differences": {"/interior/wires/drive/attachments": {"value": [{
                    "kind": "intervention", "id": "pulse", "operation": "add", "operands": {"value": {"kind": "value", "value": [0.5, 0.0]}},
                    "window": {"spans": [{
                        "kind": "interval",
                        "start": {"anchor": {"kind": "trial", "edge": "start"},
                                  "offset": {"kind": "name", "reference": "onset"}},
                        "end": {"offset": {"value": 5.0, "unit": "st"}},
                    }]},
                }]}},
            },
        }},
    }
    raw["steps"]["train"]["sweep"] = {"axes": {"pulse": {"set": "pulses"}}}
    return _seed(home, project_id, raw)


if __name__ == "__main__":
    action, *args = sys.argv[1:]
    print(json.dumps({"seed": seed, "seed_entries": seed_entries, "seed_window": seed_window}[action](*args)))
