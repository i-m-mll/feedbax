"""Fresh execution must bypass cached computations throughout the analysis graph."""

from pathlib import Path
from typing import ClassVar

import equinox as eqx
import pytest

from feedbax.analysis.analysis import AbstractAnalysis, AnalysisRef
from feedbax.analysis.context import AnalysisRunContext
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
        computation_revision: ClassVar[str] = "1"

        def compute(self, data, **kwargs):
            calls["source"] += 1
            return computation["value"]

    class Selected(ToyAnalysis):
        computation_revision: ClassVar[str] = "1"

        def compute(self, data, source, **kwargs):
            calls["selected"] += 1
            return {"value": source + 1}

        def make_figs(self, data, *, result, **kwargs):
            return super().make_figs(data, result=result, **kwargs)

    selected = eqx.tree_at(
        lambda analysis: analysis._extra_inputs,
        Selected(variant="toy", cache_result=True),
        {"source": AnalysisRef("source")},
    )

    def recipe(*_args):
        return AnalysisRecipeResult(
            analyses={"selected": selected},
            custom_dependencies={"source": Source(variant="toy", cache_result=True)},
            data=build_toy_analysis_data(),
        )

    spec = AnalysisRunSpec(analysis_type="test.cached_graph")
    registry = AnalysisRecipeRegistry()
    registry.register(spec.analysis_type, recipe, computation_revision="1")
    return spec, registry, calls, computation, recipe


@pytest.mark.parametrize("fresh", [{"use_cache": False}, {"force": True}])
def test_fresh_execution_bypasses_selected_and_transitive_result_caches(
    tmp_path: Path, application_registry_bundle, cached_graph, fresh
) -> None:
    spec, registry, calls, computation, _recipe = cached_graph

    def execute(root=tmp_path, **kwargs):
        return execute_analysis_run_spec(
            spec,
            registry=registry,
            evaluation_registry=application_registry_bundle.evaluation_recipes,
            experiment_registry=application_registry_bundle.experiment_packages,
            root=root,
            fig_dump_formats=("json",),
            **kwargs,
        )

    _manifest, path = execute()
    assert calls == {"source": 1, "selected": 1}
    cache_dir = AnalysisRunContext(
        spec=spec, root=tmp_path, computation=registry.computation(spec.analysis_type)
    ).results_cache_dir
    def cache_snapshot():
        return {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in cache_dir.glob("*.pkl")}

    cached_bytes = cache_snapshot()
    assert len(cached_bytes) == 2

    execute()
    assert calls == {"source": 1, "selected": 1}
    path.unlink()
    execute()
    assert calls == {"source": 1, "selected": 1}

    execute(**fresh)
    assert calls == {"source": 2, "selected": 2}
    assert cache_snapshot() == cached_bytes

    computation["value"] = 9
    changed, _path = execute(**fresh)
    assert calls == {"source": 3, "selected": 3}
    assert changed.artifacts[0].sha256 != _manifest.artifacts[0].sha256
    assert cache_snapshot() == cached_bytes

    uncached_root = tmp_path / "uncached"
    execute(root=uncached_root, **fresh)
    assert not (uncached_root / "cache" / "analysis_results").exists()
