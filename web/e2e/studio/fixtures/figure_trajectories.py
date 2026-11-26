"""The shipped reaching example with a swept-mass rollout, a baseline rollout and two Analyze steps whose figures the
trajectory spec authors in the figure inspector."""

from itertools import islice
from tests.support.browser_seed_actions import HomeSeed, print_action

from feedbax.contracts.analysis_document import ANALYSIS_SCHEMA
from feedbax.contracts.semantic_values import semantic_key
from feedbax.examples import load_shipped_projects
from web.e2e.studio.fixtures.canonical_evaluate import _MASS, _seed

_POSITION = "/v1/model/state/plant/effector~1position"


def _rollout(label: str, sweep: dict | None = None):
    step = {
        "kind": "evaluate",
        "label": label,
        "base": {"inline": {"evaluation_type": "feedbax.rollout", "params": {
            "seed": 7, "trial_policy": "paired", "variation": [], "trial_set": {"kind": "every_condition"},
        }}},
        "inputs": {"subject": {"step": "train", "checkpoint": "last"}},
    }
    return step | ({"sweep": sweep} if sweep else {})


def _view(label: str, role: str, source: dict, panel: str, layer: dict):
    """An authored figure of one line layer, which the spec replaces by trajectories."""
    return {
        "label": label,
        "inputs": {role: source},
        "figure": {
            "data": [role],
            "layout": {panel: {"row": 1, "col": 1, "axes_labels": {"x": "Time (s)", "y": "Position (m)"}}},
            "layers": {"line": {"role": role, "panel": panel, "mark": "line"} | layer},
        },
    }


def seed(home: str, project_id: str):
    project = load_shipped_projects()["point-mass-reaching"].model_dump(mode="json", by_alias=True, exclude_unset=True)
    task = project["documents"]["simple-reaches-task"]["payload"]
    task["fixed_steps"] = 8
    task["epochs"]["move"]["end"]["length"] = "8 st"
    task["validation_trials"] = dict(islice(task["validation_trials"].items(), 2))
    train = project["steps"]["simple-reaches"]
    train["base"]["inline"]["training_config"].update(batch_size=1, replicates=1)
    first = {"checkpoint": {"index": 0}}
    project["steps"] = {
        "train": train | {"label": "Train"},
        "masses": _rollout("Masses", {"axes": {"mass": {"values": {
            name: {"value": mass, "patches": {_MASS: {"value": mass}}} for name, mass in (("light", 0.5), ("heavy", 2.0))
        }}}}),
        "baseline": _rollout("Baseline"),
        "hands": {
            "kind": "analyze",
            "label": "Hands",
            "base": {"document": "hands"},
            "inputs": {"states": {"step": "masses"}},
        },
        "dynamics": {
            "kind": "analyze",
            "label": "Dynamics",
            "base": {"document": "dynamics"},
            "inputs": {"states": {"step": "baseline"}},
        },
    }
    project["documents"] |= {
        "hands": {"schema": ANALYSIS_SCHEMA, "label": "Hands", "payload": {
            "inputs": {"states": {"kind": "dataset"}},
            "views": {"hand-paths": _view("Hand paths", "trials", {"input": "states"}, "hand", {
                "select": first, "group": ["member", "condition", "trial"],
                "encode": {"x": "state_time", "y": {"variable": _POSITION, "select": {"xy": "x"}}},
            })},
        }},
        "dynamics": {"schema": ANALYSIS_SCHEMA, "label": "Dynamics", "payload": {
            "inputs": {"states": {"kind": "dataset"}},
            "operations": {"pca": {
                "label": "Pca",
                "operation": "feedbax.hidden_state_pca",
                "inputs": {"states": {"input": "states"}},
                "params": {"state": semantic_key("model", "state", ("cell",), "hidden")},
                }},
                "views": {"hidden": _view("Hidden", "scores", {"element": "pca", "product": "pca"}, "pc", {
                    "select": first, "group": ["condition", "trial"],
                    "encode": {"x": "state_time", "y": {"variable": "scores", "select": {"component": 1}}},
                })},
        }},
    }
    return _seed(home, project_id, project)


ACTIONS = {"seed": HomeSeed(seed)}


if __name__ == "__main__":
    print_action(ACTIONS)
