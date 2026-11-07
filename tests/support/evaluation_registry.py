import pytest

from feedbax.analysis.evaluation import EvaluationRecipeRegistry


@pytest.fixture(name="evaluation_registry")
def evaluation_registry_fixture() -> EvaluationRecipeRegistry:
    return EvaluationRecipeRegistry()
