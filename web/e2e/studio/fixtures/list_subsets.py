"""Seed one Evaluate over the best two runs of each of two Train steps and a named subset of the first."""

from tests.support.browser_seed_actions import HomeSeed, print_action

from web.e2e.studio.fixtures.canonical_evaluate import _seed


def seed(home: str, project_id: str):
    from tests.test_list_subsets import listed_best

    return _seed(home, project_id, listed_best())


ACTIONS = {"seed": HomeSeed(seed)}


if __name__ == "__main__":
    print_action(ACTIONS)
