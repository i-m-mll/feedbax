"""Seed a Train step with one run, a Train step over two learning rates, and an Evaluate step over the second."""

import json
import sys
from copy import deepcopy

from web.e2e.studio.fixtures.canonical_evaluate import _seed


def seed(home: str, project_id: str):
    from tests.test_saved_light_run import _saved_light_definition

    raw = _saved_light_definition(paired=True)
    steps = raw["steps"]
    for name in ("baseline", "report", "analyze"):
        steps.pop(name, None)
    steps["train_b"] = deepcopy(steps["train"]) | {"sweep": {"axes": {"rate": {
        "path": "settings.optimizer.learning_rate.value", "values": [0.001, 0.003],
    }}}}
    evaluate = steps.pop("conditions")
    evaluate.pop("sweep")
    evaluate["inputs"]["subject"] = {"step": "train_b", "checkpoint": "last"}
    steps["pair"] = evaluate
    for name in ("velocity-analysis", "velocity-figure", "velocity-report"):
        raw["documents"].pop(name, None)
    return _seed(home, project_id, raw)


if __name__ == "__main__":
    action, *args = sys.argv[1:]
    print(json.dumps({"seed": seed}[action](*args)))
