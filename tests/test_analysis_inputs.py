from collections import OrderedDict

import pytest

from feedbax._vendor.jax_cookbook import LDict
from feedbax.analysis.analysis import IdentityNode, SinglePort
from feedbax.analysis.context import AnalysisRunContext
from feedbax.analysis.execution import run_analyses_with_context
from feedbax.analysis.inputs import CallWithDeps, LiteralInput
from feedbax.contracts.manifest import AnalysisRunSpec
from tests.support.analysis_fixtures import build_toy_analysis_data


def test_call_with_deps_allocates_ports_per_instance() -> None:
    first = CallWithDeps("a", scale="b")(lambda x, *, scale: x * scale)
    second = CallWithDeps("c")(lambda x: x)

    first_ports = set(first._ports)
    second_ports = set(second._ports)

    assert first_ports
    assert second_ports
    assert first_ports.isdisjoint(second_ports)
    first_values = {
        port: {"a": 2, "b": 3}[spec] for port, spec in first._extra_inputs.items()
    }
    assert first(**first_values) == 6
    assert second(**{port: 5 for port in second._ports}) == 5
    assert not hasattr(CallWithDeps, "_counter")


def test_call_with_deps_allocates_ports_per_call_on_reused_builder() -> None:
    builder = CallWithDeps("value")
    increment = builder(lambda value: value + 1)
    double = builder(lambda value: value * 2)

    assert set(increment._ports).isdisjoint(double._ports)
    assert increment(**{increment._ports[0]: 4}) == 5
    assert double(**{double._ports[0]: 4}) == 8


@pytest.mark.parametrize(
    ("first", "second", "equivalent", "expected_first", "expected_second"),
    [
        pytest.param(
            {1: LiteralInput(10), 2: LiteralInput(20)},
            {"1": LiteralInput(10), "2": LiteralInput(20)},
            {2: LiteralInput(20), 1: LiteralInput(10)},
            ((1, 10), (2, 20)),
            (("1", 10), ("2", 20)),
            id="typed-keys",
        ),
        pytest.param(
            OrderedDict(a=LiteralInput(10), b=LiteralInput(20)),
            OrderedDict(b=LiteralInput(20), a=LiteralInput(10)),
            OrderedDict(a=LiteralInput(10), b=LiteralInput(20)),
            (("a", 10), ("b", 20)),
            (("b", 20), ("a", 10)),
            id="ordered-keys",
        ),
        pytest.param(
            LDict("training", {"a": LiteralInput(10)}),
            LDict("evaluation", {"a": LiteralInput(10)}),
            LDict("training", {"a": LiteralInput(10)}),
            "training",
            "evaluation",
            id="mapping-label",
        ),
    ],
)
def test_analysis_run_distinguishes_mapping_inputs_and_reuses_equivalent_nodes(
    tmp_path, first, second, equivalent, expected_first, expected_second
) -> None:
    calls = []

    class ObserveMapping(IdentityNode):
        def compute(self, data, *, input, **kwargs):
            observed = input.label if isinstance(input, LDict) else tuple(input.items())
            calls.append(observed)
            return observed

    analyses = {
        "first": ObserveMapping(inputs=SinglePort(input=first)),
        "second": ObserveMapping(inputs=SinglePort(input=second)),
        "equivalent": ObserveMapping(inputs=SinglePort(input=equivalent)),
    }
    context = AnalysisRunContext(spec=AnalysisRunSpec(analysis_type="observe_mapping"), root=tmp_path)

    _, results, _ = run_analyses_with_context(analyses, build_toy_analysis_data(), context)

    assert results == {
        "first": expected_first,
        "second": expected_second,
        "equivalent": expected_first,
    }
    assert calls == [expected_first, expected_second]
