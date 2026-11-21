"""Seed one Evaluate over the best two runs of each of two Train steps and a named subset of the first."""

import json
import sys

from web.e2e.studio.fixtures.canonical_evaluate import _seed


def seed(home: str, project_id: str):
    from tests.test_list_subsets import listed_best

    return _seed(home, project_id, listed_best())


if __name__ == "__main__":
    action, *args = sys.argv[1:]
    print(json.dumps({"seed": seed}[action](*args)))
