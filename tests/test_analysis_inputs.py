from feedbax.analysis.inputs import CallWithDeps


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
