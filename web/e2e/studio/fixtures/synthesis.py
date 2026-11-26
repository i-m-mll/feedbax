"""Seed the shipped point-mass project with its simple-reaches Train step alone."""

import json
import sys

from web.e2e.studio.fixtures.canonical_evaluate import _seed


def seed(home: str, project_id: str):
    from feedbax.examples import load_shipped_projects

    raw = load_shipped_projects()["point-mass-reaching"].model_dump(mode="json", by_alias=True, exclude_unset=True)
    raw["steps"] = {"simple-reaches": raw["steps"]["simple-reaches"]}
    raw.pop("sweep", None)
    return _seed(home, project_id, raw)


if __name__ == "__main__":
    action, *args = sys.argv[1:]
    print(json.dumps({"seed": seed}[action](*args)))
