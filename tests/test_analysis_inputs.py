from feedbax.analysis.inputs import CallWithDeps


def test_call_with_deps_allocates_ports_per_instance() -> None:
    first = CallWithDeps("a", scale="b")(lambda x, *, scale: x * scale)
    second = CallWithDeps("c")(lambda x: x)

    first_ports = set(first._ports)
    second_ports = set(second._ports)

    assert first_ports
    assert second_ports
    assert first_ports.isdisjoint(second_ports)
    assert not hasattr(CallWithDeps, "_counter")
