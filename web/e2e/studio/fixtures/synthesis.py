"""Seed the shipped point-mass project with its simple-reaches Train step alone."""

from tests.support.browser_seed_actions import HomeSeed, print_action

from web.e2e.studio.fixtures.canonical_evaluate import _seed


def seed(home: str, project_id: str):
    from feedbax.examples import load_shipped_projects

    raw = load_shipped_projects()["point-mass-reaching"].model_dump(mode="json", by_alias=True, exclude_unset=True)
    raw["steps"] = {"simple-reaches": raw["steps"]["simple-reaches"]}
    raw.pop("sweep", None)
    return _seed(home, project_id, raw)


ACTIONS = {"seed": HomeSeed(seed)}


if __name__ == "__main__":
    print_action(ACTIONS)
