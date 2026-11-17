"""Fresh execution recomputes the selected analysis and its dependencies."""

from pathlib import Path

import equinox as eqx
import pytest

from feedbax.analysis.analysis import AbstractAnalysis, AnalysisRef
from feedbax.analysis.specs import (
    AnalysisRecipeRegistry,
    AnalysisRecipeResult,
    execute_analysis_run_spec,
)
from feedbax.contracts.manifest import AnalysisRunSpec
from tests.support.analysis_fixtures import ToyAnalysis, build_toy_analysis_data
from tests.support.application_registry import registry_bundle_fixture as registry_bundle_fixture


@pytest.fixture
def cached_graph():
    calls = {"source": 0, "selected": 0}
    computation = {"value": 3}

    class Source(AbstractAnalysis):

        def compute(self, data, **kwargs):
            calls["source"] += 1
            return computation["value"]

    class Selected(ToyAnalysis):

        def compute(self, data, source, **kwargs):
            calls["selected"] += 1
            return {"value": source + 1}

        def make_figs(self, data, *, result, **kwargs):
            return super().make_figs(data, result=result, **kwargs)

    selected = eqx.tree_at(
        lambda analysis: analysis._extra_inputs,
        Selected(variant="toy"),
        {"source": AnalysisRef("source")},
    )

    def recipe(*_args):
        return AnalysisRecipeResult(
            analyses={"selected": selected},
            custom_dependencies={"source": Source(variant="toy")},
            data=build_toy_analysis_data(),
        )

    spec = AnalysisRunSpec(analysis_type="test.cached_graph")
    registry = AnalysisRecipeRegistry()
    registry.register(spec.analysis_type, recipe)
    return spec, registry, calls, computation


@pytest.mark.parametrize("fresh", [{"use_cache": False}, {"force": True}])
def test_fresh_execution_bypasses_receipt_reuse(
    tmp_path: Path, application_registry_bundle, cached_graph, fresh
) -> None:
    spec, registry, calls, computation = cached_graph

    def execute(**kwargs):
        return execute_analysis_run_spec(
            spec,
            registry=registry,
            evaluation_registry=application_registry_bundle.evaluation_recipes,
            experiment_registry=application_registry_bundle.experiment_packages,
            root=tmp_path,
            fig_dump_formats=("json",),
            **kwargs,
        )

    first, path = execute()
    assert calls == {"source": 1, "selected": 1}

    reused, _ = execute()
    assert reused == first
    assert calls == {"source": 1, "selected": 1}

    path.unlink()
    executed, _ = execute()
    assert executed.id == first.id
    assert calls == {"source": 2, "selected": 2}

    computation["value"] = 9
    changed, _ = execute(**fresh)
    assert changed.id == first.id
    assert calls == {"source": 3, "selected": 3}
    assert changed.artifacts[0].sha256 != first.artifacts[0].sha256
