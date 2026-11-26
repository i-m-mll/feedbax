"""Seed an Analyze card of two operations and a figure, with a Report over its figure, for Split and Merge."""

from tests.support.browser_seed_actions import HomeSeed, NonSeed, print_action
from web.e2e.studio.fixtures.canonical_evaluate import _seed
from web.e2e.studio.fixtures.managed_report import requests


def seed(home: str, project_id: str):
    from tests.test_project_report_plan import _report_project

    return _seed(home, project_id, _report_project().model_dump(mode="json", by_alias=True, exclude_unset=True))


ACTIONS = {
    "seed": HomeSeed(seed),
    "requests": NonSeed(requests, "measures Plan identities after editing the seed"),
}


if __name__ == "__main__":
    print_action(ACTIONS)
