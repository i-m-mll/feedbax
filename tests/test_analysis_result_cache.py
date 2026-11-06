"""Behavior of the scientific result cache through analysis execution."""

from dataclasses import field
from pathlib import Path
from typing import ClassVar

import dill
import jax.numpy as jnp
import numpy as np
import pytest

from feedbax import _atomic
from feedbax._vendor.jax_cookbook import LDict
from feedbax.analysis.analysis import AbstractAnalysis
from feedbax.analysis.context import AnalysisRunContext
from feedbax.analysis.execution import run_analyses_with_context
from feedbax.contracts.manifest import AnalysisRunSpec
from tests.support.analysis_fixtures import build_toy_analysis_data


@pytest.fixture
def context(tmp_path):
    return AnalysisRunContext(
        spec=AnalysisRunSpec(analysis_type="test.result_cache"), root=tmp_path
    )


@pytest.fixture
def constant_analysis():
    calls = []

    class Constant(AbstractAnalysis):
        computation_revision: ClassVar[str] = "1"

        def compute(self, data):
            calls.append(None)
            return 7

    return Constant(cache_result=True), calls


def _run(analysis, context, *, data=None, **inputs):
    _, results, _ = run_analyses_with_context(
        {"probe": analysis},
        build_toy_analysis_data() if data is None else data,
        AnalysisRunContext(spec=context.spec, root=context.root_path),
        **inputs,
    )
    return results["probe"]


def test_cache_reuses_only_the_declared_computation_revision(
    context, constant_analysis, monkeypatch
):
    original, calls = constant_analysis
    analysis_type = type(original)

    assert _run(original, context) == 7
    assert _run(analysis_type(cache_result=True), context) == 7
    assert len(calls) == 1

    def revised_compute(self, data):
        calls.append(None)
        return 8

    monkeypatch.setattr(analysis_type, "compute", revised_compute)
    monkeypatch.setattr(analysis_type, "computation_revision", "2")
    revised = analysis_type(cache_result=True)

    assert revised.md5_str == original.md5_str
    assert _run(revised, context) == 8
    assert _run(analysis_type(cache_result=True), context) == 8
    assert len(calls) == 2


def test_distinct_final_callbacks_share_only_the_raw_result(context, constant_analysis):
    analysis, calls = constant_analysis
    doubled = analysis.then_transform_result(lambda result: 2 * result)
    tripled = analysis.then_transform_result(lambda result: 3 * result)

    _, results, _ = run_analyses_with_context(
        {"doubled": doubled, "tripled": tripled}, build_toy_analysis_data(), context
    )

    assert results == {"doubled": 14, "tripled": 21}
    assert len(calls) == 1


def test_framework_selections_deduplicate_by_key(context):
    calls = []

    class ReadStates(AbstractAnalysis):
        def compute(self, data):
            calls.append(None)
            return data.states

    def select(key):
        return ReadStates().after_getitem_at_level("condition", key)

    data = build_toy_analysis_data(LDict.of("condition")({"a": 14, "b": 21}))
    _, results, _ = run_analyses_with_context(
        {"a": select("a"), "b": select("b"), "also_a": select("a")}, data, context
    )

    assert results == {"a": {"value": 14}, "b": {"value": 21}, "also_a": {"value": 14}}
    assert len(calls) == 2


def test_inherited_revision_does_not_authorize_cache(context, monkeypatch):
    calls = []

    class Parent(AbstractAnalysis):
        computation_revision: ClassVar[str] = "1"

        def compute(self, data):
            calls.append(None)
            return len(calls)

    class Child(Parent):
        computation_revision: ClassVar[str] = "1"

    assert _run(Child(cache_result=True), context) == 1
    (entry,) = context.results_cache_dir.glob("*.pkl")
    original_bytes = entry.read_bytes()
    monkeypatch.delattr(Child, "computation_revision")

    assert _run(Child(cache_result=True), context) == 2
    assert _run(Child(cache_result=True), context) == 3
    assert list(context.results_cache_dir.glob("*.pkl")) == [entry]
    assert entry.read_bytes() == original_bytes


def test_one_graph_cannot_alias_two_computation_revisions(context):
    def analysis(revision):
        class Report(AbstractAnalysis):
            computation_revision: ClassVar[str] = revision

            def compute(self, data):
                return revision

        return Report()

    first, second = analysis("1"), analysis("2")
    assert first.md5_str == second.md5_str
    with pytest.raises(ValueError, match="conflicting computation revisions"):
        run_analyses_with_context(
            {"first": first, "second": second}, build_toy_analysis_data(), context
        )


def test_cache_distinguishes_array_shapes_and_reuses_equal_input(context):
    calls = []

    class Shape(AbstractAnalysis):
        computation_revision: ClassVar[str] = "1"

        def compute(self, data, *, value):
            calls.append(value.shape)
            return value.sum(axis=0)

    def run(shape):
        return _run(Shape(cache_result=True), context, value=jnp.arange(6).reshape(shape))

    np.testing.assert_array_equal(run((2, 3)), [3, 5, 7])
    np.testing.assert_array_equal(run((3, 2)), [6, 9])
    np.testing.assert_array_equal(run((2, 3)), [3, 5, 7])
    assert calls == [(2, 3), (3, 2)]


def test_cache_distinguishes_scalar_leaf_boundaries(context):
    class PairSum(AbstractAnalysis):
        computation_revision: ClassVar[str] = "1"

        def compute(self, data, *, value):
            return sum(value)

    analysis = PairSum(cache_result=True)
    assert _run(analysis, context, value=(1, 23)) == 24
    assert _run(analysis, context, value=(12, 3)) == 15


def test_cache_distinguishes_full_analysis_parameter_arrays(context):
    class Coefficient(AbstractAnalysis):
        computation_revision: ClassVar[str] = "1"
        weights: np.ndarray = field(kw_only=True)

        def compute(self, data):
            return int(self.weights[600])

    first = np.arange(1200)
    second = first.copy()
    second[600] = 1

    assert _run(Coefficient(weights=first, cache_result=True), context) == 600
    assert _run(Coefficient(weights=second, cache_result=True), context) == 1


def test_cache_reuses_vmapped_dependency(context):
    calls = []

    class Double(AbstractAnalysis):
        computation_revision: ClassVar[str] = "1"

        def compute(self, data, *, value):
            calls.append(None)
            return 2 * value

        def emit_artifacts(self, context, data, *, result, value):
            return result + value

    analysis = Double(cache_result=True).vmap({"value": 0})
    data = build_toy_analysis_data(jnp.asarray([0]))
    for _ in range(2):
        np.testing.assert_array_equal(
            _run(analysis, context, data=data, value=jnp.arange(3)), [0, 3, 6]
        )
    assert len(calls) == 1


def test_opaque_closure_inputs_compute_without_cache(context):
    class ApplyMeasure(AbstractAnalysis):
        computation_revision: ClassVar[str] = "1"

        def compute(self, data, *, measure, value):
            return measure(value)

    def measure_at_offset(offset):
        return lambda value: value + offset

    analysis = ApplyMeasure(cache_result=True)
    assert _run(analysis, context, measure=measure_at_offset(2), value=5) == 7
    assert _run(analysis, context, measure=measure_at_offset(3), value=5) == 8
    assert list(context.results_cache_dir.glob("*.pkl")) == []


def test_cache_replaces_unsafe_payload_version(context, constant_analysis):
    analysis, calls = constant_analysis
    assert _run(analysis, context) == 7
    (entry,) = context.results_cache_dir.glob("*.pkl")
    payload = dill.loads(entry.read_bytes())
    payload.update(schema_version="feedbax.analysis.result-cache.v3", result=-1)
    entry.write_bytes(dill.dumps(payload))

    assert _run(analysis, context) == 7
    assert _run(analysis, context) == 7
    assert len(calls) == 2


def test_cache_failed_publication_leaves_no_partial_entry(context, constant_analysis, monkeypatch):
    analysis, calls = constant_analysis
    replace = _atomic.os.replace
    publication_attempts = []

    def fail_replace(staging: Path, destination: Path):
        if destination.parent == context.results_cache_dir:
            publication_attempts.append((staging.exists(), destination.exists()))
            raise OSError("injected cache publication failure")
        return replace(staging, destination)

    with monkeypatch.context() as patch:
        patch.setattr(_atomic.os, "replace", fail_replace)
        assert _run(analysis, context) == 7

    assert publication_attempts == [(True, False)]
    assert list(context.results_cache_dir.iterdir()) == []
    assert _run(analysis, context) == 7
    assert _run(analysis, context) == 7
    assert len(calls) == 2
