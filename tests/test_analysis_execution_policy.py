"""Fresh execution must bypass cached computations throughout the analysis graph."""

from pathlib import Path

import equinox as eqx
import pytest

from feedbax.analysis.analysis import AbstractAnalysis, AnalysisRef
from feedbax.analysis.context import AnalysisRunContext
from feedbax.analysis.fulfillment_adapters import (
    AnalysisNodeRequest,
    FulfillmentEnvironment,
    fulfill_node,
)
from feedbax.analysis.fulfillment_custody import rebuild_node
from feedbax.analysis.specs import (
    AnalysisRecipeRegistry,
    AnalysisRecipeResult,
    execute_analysis_run_spec,
)
from feedbax.contracts.manifest import AnalysisRunSpec
from tests.support.analysis_fixtures import ToyAnalysis, build_toy_analysis_data


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
    registry.register(spec.analysis_type, recipe)
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
    cache_dir = AnalysisRunContext(spec=spec, root=tmp_path).results_cache_dir
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
    assert not AnalysisRunContext(spec=spec, root=uncached_root).results_cache_dir.exists()


def test_rebuild_detects_changed_computation_despite_mirrored_result_caches(
    tmp_path: Path, application_registry_bundle, cached_graph
) -> None:
    spec, _registry, calls, computation, recipe = cached_graph
    application_registry_bundle.analysis_recipes.register(spec.analysis_type, recipe)
    environment = FulfillmentEnvironment(root=tmp_path, registries=application_registry_bundle)
    node = AnalysisNodeRequest(node_key="cached", spec=spec)
    execute_analysis_run_spec(
        spec,
        registry=application_registry_bundle.analysis_recipes,
        evaluation_registry=application_registry_bundle.evaluation_recipes,
        experiment_registry=application_registry_bundle.experiment_packages,
        root=tmp_path,
    )
    cache_dir = AnalysisRunContext(spec=spec, root=tmp_path).results_cache_dir
    cached_bytes = {p: p.read_bytes() for p in cache_dir.glob("*.pkl")}
    assert len(cached_bytes) == 2
    receipt = fulfill_node(node, environment=environment).receipt
    original = receipt.path.read_bytes()
    artifact_bytes = {
        Path(a.uri): Path(a.uri).read_bytes() for a in receipt.manifest.artifacts
    }

    assert rebuild_node(node, environment=environment).matched
    assert calls == {"source": 2, "selected": 2}
    computation["value"] = 9
    outcome = rebuild_node(node, environment=environment)

    assert calls == {"source": 3, "selected": 3}
    assert not outcome.matched
    assert "artifacts[0].sha256" in outcome.field_paths
    assert receipt.path.read_bytes() == original
    assert all(path.read_bytes() == data for path, data in artifact_bytes.items())
    assert {p: p.read_bytes() for p in cache_dir.glob("*.pkl")} == cached_bytes
