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
    steps["train_b"] = deepcopy(steps["train"])
    steps["train_b"]["base"]["inline"]["settings"]["optimizer"]["learning_rate"]["value"] = 0.003
    steps["other"] = deepcopy(steps["train"])
    evaluate = steps.pop("conditions")
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
    raw["documents"]["set"] = {"schema": "feedbax.set_document.v1", "payload": {"entries": {"light": {}, "heavy": {}}}}
    raw["steps"]["train"]["sweep"] = {"axes": {"set": {"set": "set"}}}
    return _seed(home, project_id, raw)


if __name__ == "__main__":
    action, *args = sys.argv[1:]
    print(json.dumps({"seed": seed, "seed_entries": seed_entries}[action](*args)))
