"""Seed two Train steps, two Evaluate steps that read the first, and an Analyze step over one of them."""

from tests.support.browser_seed_actions import HomeSeed, print_action
from copy import deepcopy

from web.e2e.studio.fixtures.canonical_evaluate import _seed


def seed(home: str, project_id: str):
    from tests.test_saved_light_run import _saved_light_definition

    raw = _saved_light_definition(paired=True)
    steps = raw["steps"]
    for name in ("baseline", "report"):
        steps.pop(name, None)
    steps["train_b"] = deepcopy(steps["train"]) | {
        "label": "Train b",
        "sweep": {"axes": {"rate": {"path": "settings.optimizer.learning_rate.value", "values": [0.001, 0.003]}}},
    }
    conditions = steps["conditions"]
    conditions["inputs"]["subject"] = {"step": "train", "checkpoint": "last"}
    steps["conditions_b"] = deepcopy(conditions) | {"label": "Conditions b"}
    masses = steps["conditions_b"]["sweep"]["axes"]["mass"]["values"]
    for old, (new, mass) in {"nominal": ("light", 0.5), "heavy": ("heavier", 3)}.items():
        value = masses.pop(old)
        value["value"] = mass
        for patch in value["patches"].values():
            patch["value"] = mass
        masses[new] = value
    steps["analyze"]["inputs"]["states"] = {"step": "conditions"}
    layer = raw["documents"]["velocity-figure"]["payload"]["layers"]["bounds"]
    layer["group"] = ["source", "model", "rate", "mass"]
    layer["style"] = {"label": "{source} {model} {rate} {mass}"}
    return _seed(home, project_id, raw)


ACTIONS = {"seed": HomeSeed(seed)}


if __name__ == "__main__":
    print_action(ACTIONS)
