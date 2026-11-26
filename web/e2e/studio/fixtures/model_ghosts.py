"""Seed the ordinary Train step swept over a set whose `drop` entry removes the Point mass, in two variants."""

from tests.support.browser_seed_actions import HomeSeed, print_action

from feedbax.contracts.project import Project
from web.e2e.studio.fixtures.canonical_train import ordinary_files, write_project


def seed(home: str, project_id: str):
    *documents, train = ordinary_files("2")
    train["step"]["sweep"] = {"axes": {"arm": {"set": "arms"}}}
    arms = {"schema": "feedbax.set_document.v2", "label": "Arms", "payload": {"entries": {
        "keep": {"label": "Keep"},
        "drop": {"label": "Drop", "differences": {"/interior/components/plant": {"op": "remove"}, "/interior/wires/drive": {"op": "remove"}}},
    }}}
    project = Project.model_validate({
        "documents": {**{file["id"]: file["document"] for file in documents}, "arms": arms},
        "steps": {"train": train["step"]},
        "sweep": {"rows": {"base": {"label": "Baseline"}, "control": {"from": "base", "label": "Control"}}},
    })
    return write_project(home, project_id, project)


ACTIONS = {"seed": HomeSeed(seed)}


if __name__ == "__main__":
    print_action(ACTIONS)
