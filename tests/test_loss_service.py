"""Tests for the loss service."""

import pytest
from feedbax.web.services.loss_service import LossService, ProbeInfo
from feedbax.web.models.graph import (
    GraphSpec,
    ComponentSpec,
    WireSpec,
    BarnacleSpec,
    TapSpec,
)
from feedbax.web.models.training import LossTermSpec, TimeAggregationSpec


@pytest.fixture
def loss_service():
    return LossService()


@pytest.fixture
def sample_graph():
    """Create a sample graph with nodes, barnacles, and taps."""
    return GraphSpec(
        nodes={
            "network": ComponentSpec(
                type="Network",
                params={},
                input_ports=["input"],
                output_ports=["output", "hidden"],
            ),
            "effector": ComponentSpec(
                type="Effector",
                params={},
                input_ports=["command"],
                output_ports=["position", "velocity"],
            ),
        },
        wires=[
            WireSpec(
                source_node="network",
                source_port="output",
                target_node="effector",
                target_port="command",
            ),
        ],
        input_ports=["input"],
        output_ports=["position", "velocity"],
        input_bindings={"input": ("network", "input")},
        output_bindings={
            "position": ("effector", "position"),
            "velocity": ("effector", "velocity"),
        },
        barnacles={
            "effector": [
                BarnacleSpec(
                    id="effector_pos",
                    kind="probe",
                    timing="output",
                    label="Effector Position",
                    read_paths=["state.position"],
                    write_paths=[],
                    transform="",
                ),
            ],
        },
        taps=[
            TapSpec(
                id="hidden_activity",
                type="probe",
                position={"afterNode": "network"},
                paths={"hidden": "state.hidden"},
            ),
        ],
    )


class TestGetAvailableProbes:
    """Tests for get_available_probes."""

    def test_extracts_barnacle_probes(self, loss_service, sample_graph):
        probes = loss_service.get_available_probes(sample_graph)
        barnacle_probes = [p for p in probes if p.selector == "probe:effector_pos"]
        assert len(barnacle_probes) == 1
        probe = barnacle_probes[0]
        assert probe.id == "effector_pos"
        assert probe.label == "Effector Position"
        assert probe.node == "effector"
        assert probe.timing == "output"

    def test_extracts_tap_probes(self, loss_service, sample_graph):
        probes = loss_service.get_available_probes(sample_graph)
        tap_probes = [p for p in probes if p.selector == "probe:hidden_activity"]
        assert len(tap_probes) == 1
        probe = tap_probes[0]
        assert probe.id == "hidden_activity"
        assert probe.node == "network"

    def test_extracts_implicit_port_probes(self, loss_service, sample_graph):
        probes = loss_service.get_available_probes(sample_graph)
        port_probes = [p for p in probes if p.selector.startswith("port:")]
        # network: output, hidden
        # effector: position, velocity
        assert len(port_probes) == 4
        selectors = {p.selector for p in port_probes}
        assert "port:network.output" in selectors
        assert "port:network.hidden" in selectors
        assert "port:effector.position" in selectors
        assert "port:effector.velocity" in selectors

    def test_handles_empty_graph(self, loss_service):
        empty_graph = GraphSpec()
        probes = loss_service.get_available_probes(empty_graph)
        assert probes == []


class TestResolveProbSelector:
    """Tests for resolve_probe_selector."""

    def test_resolves_barnacle_probe(self, loss_service, sample_graph):
        result = loss_service.resolve_probe_selector("probe:effector_pos", sample_graph)
        assert result is not None
        assert result["type"] == "barnacle"
        assert result["node"] == "effector"
        assert result["barnacle_id"] == "effector_pos"

    def test_resolves_tap_probe(self, loss_service, sample_graph):
        result = loss_service.resolve_probe_selector(
            "probe:hidden_activity", sample_graph
        )
        assert result is not None
        assert result["type"] == "tap"
        assert result["tap_id"] == "hidden_activity"

    def test_resolves_port_selector(self, loss_service, sample_graph):
        result = loss_service.resolve_probe_selector(
            "port:effector.position", sample_graph
        )
        assert result is not None
        assert result["type"] == "port"
        assert result["node"] == "effector"
        assert result["port"] == "position"

    def test_resolves_path_selector(self, loss_service, sample_graph):
        result = loss_service.resolve_probe_selector(
            "path:state.hidden.output", sample_graph
        )
        assert result is not None
        assert result["type"] == "path"
        assert result["path"] == "state.hidden.output"

    def test_returns_none_for_unknown_probe(self, loss_service, sample_graph):
        result = loss_service.resolve_probe_selector("probe:unknown", sample_graph)
        assert result is None

    def test_returns_none_for_unknown_port(self, loss_service, sample_graph):
        result = loss_service.resolve_probe_selector(
            "port:unknown.port", sample_graph
        )
        assert result is None

    def test_returns_none_for_empty_selector(self, loss_service, sample_graph):
        result = loss_service.resolve_probe_selector("", sample_graph)
        assert result is None


class TestBuildTimeAggregation:
    """Tests for build_time_aggregation."""

    def test_default_to_all_mode(self, loss_service):
        result = loss_service.build_time_aggregation(None)
        assert result.mode == "all"

    def test_all_mode(self, loss_service):
        time_agg = TimeAggregationSpec(mode="all")
        result = loss_service.build_time_aggregation(time_agg)
        assert result.mode == "all"

    def test_final_mode(self, loss_service):
        time_agg = TimeAggregationSpec(mode="final")
        result = loss_service.build_time_aggregation(time_agg)
        assert result.mode == "final"

    def test_range_mode(self, loss_service):
        time_agg = TimeAggregationSpec(mode="range", start=10, end=50)
        result = loss_service.build_time_aggregation(time_agg)
        assert result.mode == "range"
        assert result.time_range is not None
        assert result.time_range.start == 10
        assert result.time_range.end == 50

    def test_segment_mode(self, loss_service):
        time_agg = TimeAggregationSpec(mode="segment", segment_name="movement")
        result = loss_service.build_time_aggregation(time_agg)
        assert result.mode == "segment"
        assert result.segment_name == "movement"

    def test_custom_mode(self, loss_service):
        time_agg = TimeAggregationSpec(mode="custom", time_idxs=[0, 10, 50, 100])
        result = loss_service.build_time_aggregation(time_agg)
        assert result.mode == "custom"
        assert result.time_idxs == [0, 10, 50, 100]

    def test_power_discount(self, loss_service):
        time_agg = TimeAggregationSpec(
            mode="all", discount="power", discount_exp=6.0
        )
        result = loss_service.build_time_aggregation(time_agg)
        assert result.discount_type == "power"
        assert result.discount_exp == 6.0

    def test_linear_discount(self, loss_service):
        time_agg = TimeAggregationSpec(mode="all", discount="linear")
        result = loss_service.build_time_aggregation(time_agg)
        assert result.discount_type == "linear"


class TestGetNormFunction:
    """Tests for get_norm_function."""

    def test_squared_l2(self, loss_service):
        result = loss_service.get_norm_function("squared_l2")
        assert result == "feedbax.loss.norms.squared_l2"

    def test_l2(self, loss_service):
        result = loss_service.get_norm_function("l2")
        assert result == "feedbax.loss.norms.l2"

    def test_l1(self, loss_service):
        result = loss_service.get_norm_function("l1")
        assert result == "feedbax.loss.norms.l1"

    def test_huber(self, loss_service):
        result = loss_service.get_norm_function("huber")
        assert result == "feedbax.loss.norms.huber"

    def test_unknown_norm(self, loss_service):
        result = loss_service.get_norm_function("unknown")
        assert result is None

    def test_none_norm(self, loss_service):
        result = loss_service.get_norm_function(None)
        assert result is None


class TestValidateLossSpec:
    """Tests for validate_loss_spec."""

    def test_valid_simple_loss(self, loss_service, sample_graph):
        spec = LossTermSpec(
            type="TargetStateLoss",
            label="Position Error",
            weight=1.0,
            selector="port:effector.position",
            norm="squared_l2",
            time_agg=TimeAggregationSpec(mode="all"),
        )
        errors = loss_service.validate_loss_spec(spec, sample_graph)
        assert len(errors) == 0

    def test_invalid_selector(self, loss_service, sample_graph):
        spec = LossTermSpec(
            type="TargetStateLoss",
            label="Test",
            weight=1.0,
            selector="probe:nonexistent",
        )
        errors = loss_service.validate_loss_spec(spec, sample_graph)
        assert len(errors) == 1
        assert errors[0]["field"] == "selector"

    def test_invalid_range_time_agg(self, loss_service, sample_graph):
        spec = LossTermSpec(
            type="TargetStateLoss",
            label="Test",
            weight=1.0,
            selector="port:effector.position",
            time_agg=TimeAggregationSpec(mode="range"),  # Missing start/end
        )
        errors = loss_service.validate_loss_spec(spec, sample_graph)
        assert any(e["field"] == "time_agg" for e in errors)

    def test_negative_weight(self, loss_service, sample_graph):
        spec = LossTermSpec(
            type="TargetStateLoss",
            label="Test",
            weight=-1.0,
            selector="port:effector.position",
        )
        errors = loss_service.validate_loss_spec(spec, sample_graph)
        assert any(e["field"] == "weight" for e in errors)

    def test_validates_children(self, loss_service, sample_graph):
        spec = LossTermSpec(
            type="Composite",
            label="Combined",
            weight=1.0,
            children={
                "valid": LossTermSpec(
                    type="TargetStateLoss",
                    label="Valid",
                    weight=1.0,
                    selector="port:effector.position",
                ),
                "invalid": LossTermSpec(
                    type="TargetStateLoss",
                    label="Invalid",
                    weight=1.0,
                    selector="probe:nonexistent",
                ),
            },
        )
        errors = loss_service.validate_loss_spec(spec, sample_graph)
        assert len(errors) == 1
        assert errors[0]["path"] == ["invalid"]


class TestSpecToLossConfig:
    """Tests for spec_to_loss_config."""

    def test_simple_loss_config(self, loss_service, sample_graph):
        spec = LossTermSpec(
            type="TargetStateLoss",
            label="Position Error",
            weight=1.0,
            selector="port:effector.position",
            norm="squared_l2",
            time_agg=TimeAggregationSpec(mode="final"),
        )
        config = loss_service.spec_to_loss_config(spec, sample_graph)
        assert config["type"] == "TargetStateLoss"
        assert config["label"] == "Position Error"
        assert config["weight"] == 1.0
        assert config["norm"] == "feedbax.loss.norms.squared_l2"
        assert config["time_aggregation"]["mode"] == "final"
        assert "probe" in config

    def test_composite_loss_config(self, loss_service, sample_graph):
        spec = LossTermSpec(
            type="Composite",
            label="Combined",
            weight=1.0,
            children={
                "position": LossTermSpec(
                    type="TargetStateLoss",
                    label="Position",
                    weight=1.0,
                    selector="port:effector.position",
                ),
                "velocity": LossTermSpec(
                    type="TargetStateLoss",
                    label="Velocity",
                    weight=0.5,
                    selector="port:effector.velocity",
                ),
            },
        )
        config = loss_service.spec_to_loss_config(spec, sample_graph)
        assert config["type"] == "Composite"
        assert "children" in config
        assert "position" in config["children"]
        assert "velocity" in config["children"]
        assert config["children"]["velocity"]["weight"] == 0.5
