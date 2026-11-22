"""Seed a Train step whose Model lacks required parameters, a sound 8000-batch Train step, and an Evaluate of the first."""

import json
import sys
from copy import deepcopy

from web.e2e.studio.fixtures.canonical_evaluate import _seed


def seed(home: str, project_id: str):
    from tests.test_saved_light_run import _saved_light_definition

    raw = _saved_light_definition(paired=True)
    steps = raw["steps"]
    for name in ("baseline", "analyze", "report"):
        steps.pop(name)
    documents = raw["documents"]
    sound = deepcopy(documents["reach-model"])
    controller = documents["reach-model"]["payload"]["interior"]["components"]["wrapper"]["interior"]["components"][
        "controller"]["interior"]["components"]
    del controller["cell"]["fields"]["hidden_size"]
    del controller["readout"]["fields"]["output_size"]
    documents["sound-model"] = sound
    train = steps["train"]
    train["base"]["inline"]["training_config"]["n_batches"] = 10000
    train["sweep"] = {"axes": {"rate": {
        "path": "settings.optimizer.learning_rate.value", "values": [0.001, 0.003],
    }}}
    steps["train_b"] = deepcopy(train)
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
