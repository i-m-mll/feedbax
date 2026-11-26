"""Seed a Train step whose Model holds a Linear and a Conv2d without their required parameters, a sound 8000-batch
Train step, an Evaluate of the first, and an Analyze of the Evaluate."""

import json
import sys
from copy import deepcopy

from web.e2e.studio.fixtures.canonical_evaluate import _seed


def seed(home: str, project_id: str):
    from tests.test_saved_light_run import _saved_light_definition

    raw = _saved_light_definition(paired=True)
    steps = raw["steps"]
    for name in ("baseline", "report"):
        steps.pop(name)
    steps["analyze"]["inputs"]["states"] = {"step": "conditions"}
    documents = raw["documents"]
    documents["sound-model"] = deepcopy(documents["reach-model"]) | {"label": "Sound model"}
    components = documents["reach-model"]["payload"]["interior"]["components"]
    components["linear"] = {"kind": "leaf", "name": "Linear", "type": "feedbax.equinox.Linear", "fields": {}}
    components["conv"] = {"kind": "leaf", "name": "Conv", "type": "feedbax.Conv2d", "fields": {}}
    train = steps["train"]
    train["base"]["inline"]["training_config"]["n_batches"] = 10000
    train["sweep"] = {"axes": {"rate": {
        "path": "settings.optimizer.learning_rate.value", "values": [0.001, 0.003],
    }}}
    steps["train_b"] = deepcopy(train) | {"label": "Train b"}
    steps["train_b"].pop("sweep")
    steps["train_b"]["base"]["inline"]["training_config"]["n_batches"] = 8000
    steps["train_b"]["base"]["inline"]["trial_documents"]["model"] = {"document": "sound-model"}
    conditions = steps["conditions"]
    conditions["sweep"] = {"axes": {"subject": {"input": "subject"}}}
    conditions["inputs"]["subject"] = {"step": "train", "checkpoint": "last"}
    return _seed(home, project_id, raw)


if __name__ == "__main__":
    action, *args = sys.argv[1:]
    print(json.dumps({"seed": seed}[action](*args)))
