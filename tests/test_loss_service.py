"""Tests for the loss service."""

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from feedbax.contracts.graph import (
    BarnacleSpec,
    ComponentSpec,
    GraphSpec,
    TapSpec,
    WireSpec,
)
from feedbax.contracts.training import (
    LossTermSpec,
    TimeAggregationSpec,
    standard_supervised_method_contract,
)
from feedbax.contracts.worker import AxisReducerSpec, toy_adaptive_curriculum_method_contract
from feedbax.objectives.service import (
    LossService,
    ObjectiveLoweringError,
    loss_term_spec_to_objective_spec,
)
from feedbax.objectives.spec import (
    EpochMaskSpec,
    MatrixPayloadSpec,
    MatrixQuadraticLossSpec,
    MovementEpochRampScheduleSpec,
    ObjectiveSpec,
    PowerLawScheduleSpec,
    ReductionSpec,
    SelectorAddressSpec,
    TargetStateLossSpec,
    TargetValueSpec,
    TaskTimelineSpec,
    TimelineEpochSpec,
)
from feedbax.training.worker_validation import (
    WorkerContractValidationError,
    validate_worker_contract,
)

pytestmark = [pytest.mark.feedbax_contract]


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

    def test_matrix_quadratic_loss_config_preserves_matrix_payload(
        self, loss_service, sample_graph
    ):
        spec = LossTermSpec(
            type="MatrixQuadraticLoss",
            label="Quadratic Effort",
            weight=0.5,
            selector="port:effector.velocity",
            matrix=[[1.0, 0.0], [0.0, 2.0]],
            matrix_kind="dense",
            time_agg=TimeAggregationSpec(mode="sum"),
        )
        config = loss_service.spec_to_loss_config(spec, sample_graph)
        assert config["type"] == "MatrixQuadraticLoss"
        assert config["matrix"] == [[1.0, 0.0], [0.0, 2.0]]
        assert config["matrix_kind"] == "dense"
        assert config["time_aggregation"]["mode"] == "sum"

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


def _runtime_state() -> SimpleNamespace:
    return SimpleNamespace(
        output=jnp.asarray(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]],
            ]
        ),
        control=jnp.asarray(
            [
                [[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]],
                [[-1.0, 1.0], [-2.0, 2.0], [-3.0, 3.0]],
            ]
        ),
    )


def _timeline() -> TaskTimelineSpec:
    return TaskTimelineSpec(
        n_steps=3,
        epochs=[
            TimelineEpochSpec(name="hold", index=0, length_range=(1, 1)),
            TimelineEpochSpec(name="movement", index=1, length_range=(2, 2)),
        ],
    )


def _state_selector(selector: str) -> SelectorAddressSpec:
    return SelectorAddressSpec(
        selector=selector,
        kind="state",
        temporal_axis="time",
        feature_axis="coordinate",
    )


def _legacy_expected_target_state(norm: str, time_mode: str) -> jnp.ndarray:
    diff = _runtime_state().output - jnp.asarray([1.0, 0.0])
    if norm == "squared_l2":
        values = jnp.sum(jnp.square(diff), axis=-1)
    elif norm == "l1":
        values = jnp.sum(jnp.abs(diff), axis=-1)
    elif norm == "l2":
        values = jnp.sqrt(jnp.sum(jnp.square(diff), axis=-1))
    elif norm == "huber":
        abs_diff = jnp.abs(diff)
        values = jnp.sum(
            jnp.where(abs_diff <= 1.0, 0.5 * jnp.square(diff), abs_diff - 0.5),
            axis=-1,
        )
    else:
        raise AssertionError(f"unexpected norm {norm!r}")

    if time_mode in {"all", "mean"}:
        values = jnp.mean(values, axis=1)
    elif time_mode == "sum":
        values = jnp.sum(values, axis=1)
    elif time_mode == "final":
        values = jnp.take(values, -1, axis=1)
    else:
        raise AssertionError(f"unexpected time mode {time_mode!r}")
    return jnp.mean(values)


class TestExecutableLowering:
    def test_loss_term_lowers_to_executable_loss_matching_hand_reduction(self) -> None:
        spec = LossTermSpec(
            type="target_state",
            label="output",
            selector="state.output",
            target_value=[1.0, 0.0],
            norm="squared_l2",
            time_agg=TimeAggregationSpec(mode="mean"),
        )

        lowered = LossService().lower_loss_term_spec(spec)
        value = lowered.loss(_runtime_state(), SimpleNamespace(), None).total

        diff = _runtime_state().output - jnp.asarray([1.0, 0.0])
        expected = jnp.mean(jnp.mean(jnp.sum(jnp.square(diff), axis=-1), axis=1))
        assert jnp.allclose(value, expected)
        assert lowered.requirements.requires_axes == ["batch"]
        assert lowered.requirements.aggregation_semantics == {"batch": "mean"}

    def test_loss_term_l2_uses_euclidean_feature_norm(self) -> None:
        spec = LossTermSpec(
            type="target_state",
            label="output",
            selector="state.output",
            target_value=[1.0, 0.0],
            norm="l2",
            time_agg=TimeAggregationSpec(mode="sum"),
        )

        lowered = LossService().lower_loss_term_spec(spec)
        value = lowered.loss(_runtime_state(), SimpleNamespace(), None).total

        diff = _runtime_state().output - jnp.asarray([1.0, 0.0])
        expected = jnp.mean(jnp.sum(jnp.sqrt(jnp.sum(jnp.square(diff), axis=-1)), axis=1))
        assert jnp.allclose(value, expected)

    @pytest.mark.parametrize("norm", ["squared_l2", "l1", "l2", "huber"])
    @pytest.mark.parametrize("time_mode", ["all", "mean", "sum", "final"])
    def test_loss_term_adapter_preserves_legacy_norm_and_time_values(
        self,
        norm: str,
        time_mode: str,
    ) -> None:
        spec = LossTermSpec(
            type="target_state",
            label="output",
            selector="state.output",
            target_value=[1.0, 0.0],
            norm=norm,
            time_agg=TimeAggregationSpec(mode=time_mode),
        )

        objective = loss_term_spec_to_objective_spec(spec)
        assert objective.terms[0].reduction == ReductionSpec(
            time="mean" if time_mode == "all" else time_mode,
            trial="mean",
            feature="sum",
        )
        if norm == "huber":
            assert objective.terms[0].metric.huber_delta == 1.0

        lowered = LossService().lower_loss_term_spec(spec)
        value = lowered.loss(_runtime_state(), SimpleNamespace(), None).total

        assert jnp.allclose(value, _legacy_expected_target_state(norm, time_mode))

    def test_loss_term_adapter_rejects_range_time_aggregation(self) -> None:
        spec = LossTermSpec(
            type="target_state",
            label="output",
            selector="state.output",
            target_value=[1.0, 0.0],
            time_agg=TimeAggregationSpec(mode="range", start=0, end=2),
        )

        with pytest.raises(ObjectiveLoweringError, match="no ObjectiveSpec equivalent"):
            LossService().lower_loss_term_spec(spec)

    def test_loss_term_adapter_rejects_inert_legacy_discount(self) -> None:
        spec = LossTermSpec(
            type="target_state",
            label="output",
            selector="state.output",
            target_value=[1.0, 0.0],
            time_agg=TimeAggregationSpec(mode="mean", discount="power", discount_exp=2.0),
        )

        with pytest.raises(ObjectiveLoweringError, match="time_agg.discount"):
            LossService().lower_loss_term_spec(spec)

    def test_modern_objective_huber_delta_controls_threshold(self) -> None:
        spec = ObjectiveSpec(
            terms=[
                TargetStateLossSpec(
                    label="output_huber",
                    selector=_state_selector("state.output"),
                    target=TargetValueSpec(kind="constant", value=[1.0, 0.0]),
                    metric={"kind": "huber", "huber_delta": 0.5},
                    reduction=ReductionSpec(time="mean", trial="mean", feature="sum"),
                )
            ],
        )

        lowered = LossService().lower_objective_spec(spec)
        value = lowered.loss(_runtime_state(), SimpleNamespace(), None).total

        diff = _runtime_state().output - jnp.asarray([1.0, 0.0])
        abs_diff = jnp.abs(diff)
        expected_terms = jnp.where(abs_diff <= 0.5, 0.5 * jnp.square(diff), 0.5 * (abs_diff - 0.25))
        expected = jnp.mean(jnp.mean(jnp.sum(expected_terms, axis=-1), axis=1))
        assert jnp.allclose(value, expected)

    def test_objective_selector_value_dtype_casts_selected_value(self) -> None:
        spec = ObjectiveSpec(
            terms=[
                TargetStateLossSpec(
                    label="output_dtype",
                    selector=SelectorAddressSpec(
                        selector="state.output",
                        kind="state",
                        value_dtype="float16",
                        temporal_axis="time",
                        feature_axis="coordinate",
                    ),
                    target=TargetValueSpec(kind="constant", value=[1.0, 0.0]),
                    reduction=ReductionSpec(time="mean", trial="mean", feature="sum"),
                )
            ],
        )

        lowered = LossService().lower_objective_spec(spec)
        value = lowered.loss(_runtime_state(), SimpleNamespace(), None).total

        diff = _runtime_state().output.astype(jnp.float16) - jnp.asarray([1.0, 0.0])
        expected = jnp.mean(jnp.mean(jnp.sum(jnp.square(diff), axis=-1), axis=1))
        assert jnp.allclose(value, expected)

    def test_objective_spec_l2_uses_euclidean_feature_norm(self) -> None:
        spec = ObjectiveSpec(
            terms=[
                TargetStateLossSpec(
                    label="output_l2",
                    selector=_state_selector("state.output"),
                    target=TargetValueSpec(kind="constant", value=[1.0, 0.0]),
                    metric={"kind": "l2"},
                    reduction=ReductionSpec(time="sum", trial="mean", feature="sum"),
                )
            ],
        )

        lowered = LossService().lower_objective_spec(spec)
        value = lowered.loss(_runtime_state(), SimpleNamespace(), None).total

        diff = _runtime_state().output - jnp.asarray([1.0, 0.0])
        expected = jnp.mean(jnp.sum(jnp.sqrt(jnp.sum(jnp.square(diff), axis=-1)), axis=1))
        assert jnp.allclose(value, expected)

    def test_objective_spec_lowers_matrix_mask_and_schedule_terms(self) -> None:
        spec = ObjectiveSpec(
            timeline=_timeline(),
            terms=[
                TargetStateLossSpec(
                    label="masked_output",
                    selector=_state_selector("state.output"),
                    target=TargetValueSpec(kind="constant", value=[1.0, 0.0]),
                    mask=EpochMaskSpec(epochs=["movement"]),
                    schedule=MovementEpochRampScheduleSpec(
                        anchor_epoch="movement",
                        duration_steps=2,
                        start_value=0.0,
                        end_value=1.0,
                    ),
                    reduction=ReductionSpec(time="sum", trial="mean", feature="sum"),
                ),
                MatrixQuadraticLossSpec(
                    label="control_cost",
                    selector=_state_selector("state.control"),
                    target=TargetValueSpec(kind="constant", value=[0.0, 0.0]),
                    matrix=MatrixPayloadSpec(kind="diagonal", value=[1.0, 2.0]),
                    reduction=ReductionSpec(time="sum", trial="mean", feature="sum"),
                    weight=0.5,
                ),
            ],
        )

        lowered = LossService().lower_objective_spec(spec)
        value = lowered.loss(_runtime_state(), SimpleNamespace(), None).total

        output_diff = _runtime_state().output - jnp.asarray([1.0, 0.0])
        ramp = jnp.asarray([0.0, 0.0, 0.5])
        mask = jnp.asarray([0.0, 1.0, 1.0])
        output_expected = jnp.mean(
            jnp.sum(jnp.sum(jnp.square(output_diff), axis=-1) * ramp * mask, axis=1)
        )
        control_expected = jnp.mean(
            jnp.sum(
                jnp.sum(jnp.square(_runtime_state().control) * jnp.asarray([1.0, 2.0]), axis=-1),
                axis=1,
            )
        )
        assert jnp.allclose(value, output_expected + 0.5 * control_expected)

    def test_tail_reduction_emits_risk_axis_requirement(self) -> None:
        spec = ObjectiveSpec(
            terms=[
                TargetStateLossSpec(
                    label="tail_output",
                    selector=_state_selector("state.output"),
                    target=TargetValueSpec(kind="constant", value=[0.0, 0.0]),
                    reduction=ReductionSpec(time="sum", trial="tail", feature="sum"),
                )
            ],
        )

        lowered = LossService().lower_objective_spec(spec, trial_axis="realization")

        assert lowered.requirements.requires_axes == ["realization"]
        assert lowered.requirements.aggregation_semantics == {"realization": "tail"}

    def test_invalid_selector_error_names_path(self) -> None:
        spec = LossTermSpec(type="target_state", label="bad", selector="probe:missing")

        with pytest.raises(ObjectiveLoweringError) as excinfo:
            LossService().lower_loss_term_spec(spec, graph=GraphSpec())

        assert "/loss/selector" in str(excinfo.value)
        assert "probe:missing" in str(excinfo.value)

    def test_unknown_loss_term_error_names_path(self) -> None:
        spec = LossTermSpec(type="mystery", label="bad", selector="state.output")

        with pytest.raises(ObjectiveLoweringError, match="/loss/type"):
            LossService().lower_loss_term_spec(spec)

    def test_loss_term_missing_target_error_names_path(self) -> None:
        spec = LossTermSpec(type="target_state", label="bad", selector="state.output")

        with pytest.raises(ObjectiveLoweringError) as excinfo:
            LossService().lower_loss_term_spec(spec)

        assert "/loss" in str(excinfo.value)
        assert "requires either target_selector or target_value" in str(excinfo.value)

    def test_matrix_shape_error_names_offending_path(self) -> None:
        spec = LossTermSpec(
            type="MatrixQuadraticLoss",
            label="bad_matrix",
            selector="state.output",
            matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            matrix_kind="dense",
        )
        lowered = LossService().lower_loss_term_spec(spec)

        with pytest.raises(ObjectiveLoweringError) as excinfo:
            lowered.loss(_runtime_state(), SimpleNamespace(), None).total

        assert "/loss/terms/0/matrix" in str(excinfo.value)
        assert "feature dimension" in str(excinfo.value)

    def test_unsupported_epoch_schedule_error_names_path(self) -> None:
        spec = ObjectiveSpec(
            timeline=_timeline(),
            terms=[
                TargetStateLossSpec(
                    label="bad_schedule",
                    selector=_state_selector("state.output"),
                    target=TargetValueSpec(kind="constant", value=[0.0, 0.0]),
                    schedule=PowerLawScheduleSpec(
                        exponent=2.0,
                        window="epoch",
                        epoch="movement",
                    ),
                )
            ],
        )

        with pytest.raises(ObjectiveLoweringError, match="/objective/terms/0/schedule"):
            LossService().lower_objective_spec(spec)

    def test_worker_validation_rejects_required_tail_when_objective_is_mean(self) -> None:
        spec = ObjectiveSpec(
            terms=[
                TargetStateLossSpec(
                    label="mean_output",
                    selector=_state_selector("state.output"),
                    target=TargetValueSpec(kind="constant", value=[0.0, 0.0]),
                    reduction=ReductionSpec(time="sum", trial="mean", feature="sum"),
                )
            ],
        )
        lowered = LossService().lower_objective_spec(spec, trial_axis="realization")
        contract = toy_adaptive_curriculum_method_contract()
        contract.metadata["required_objective_aggregation"] = {"realization": "tail"}

        with pytest.raises(WorkerContractValidationError) as excinfo:
            validate_worker_contract(contract, objective_requirements=lowered.requirements)

        assert "/objective_execution_requirements/aggregation_semantics/realization" in str(
            excinfo.value
        )
        assert "expected 'tail', found 'mean'" in str(excinfo.value)

    def test_worker_validation_rejects_objective_and_worker_double_reducer(self) -> None:
        spec = ObjectiveSpec(
            terms=[
                TargetStateLossSpec(
                    label="mean_output",
                    selector=_state_selector("state.output"),
                    target=TargetValueSpec(kind="constant", value=[0.0, 0.0]),
                    reduction=ReductionSpec(time="sum", trial="mean", feature="sum"),
                )
            ],
        )
        lowered = LossService().lower_objective_spec(spec)
        contract = standard_supervised_method_contract()
        contract.axes[0].reducer = AxisReducerSpec(
            owner="worker",
            reduction="mean",
            path="/axes/0/reducer",
        )

        with pytest.raises(WorkerContractValidationError, match="more than one reducer"):
            validate_worker_contract(contract, objective_requirements=lowered.requirements)
