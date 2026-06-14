"""Tests for retained-observable lowering and loss evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from feedbax.contracts.manifest import Provenance, load_manifest, write_training_run_manifest
from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.integrations.provider import provider_manifest, validate_graph_spec, validate_training_spec
from feedbax.contracts.retention_artifact_schema import (
    RETAINED_OBSERVABLES_ARTIFACT_SCHEMA_ID,
    RETAINED_OBSERVABLES_ARTIFACT_SCHEMA_VERSION,
    RETENTION_PLAN_SCHEMA_ID,
    RETENTION_PLAN_SCHEMA_VERSION,
)
from feedbax.runtime.retained_observables import (
    RetentionPlanError,
    evaluate_loss_plan,
    lower_retention_plan,
    normalize_selector_ref,
    retention_plan_to_json,
)
from feedbax.contracts.graph import (
    AnalysisInputConsumerSpec,
    AnalysisInputRequirement,
    ComponentSpec,
    GraphSpec,
    RetainedObservableSpec,
    RetainedObservableTargetSpec,
    RetentionPolicySpec,
    WireSpec,
)
from feedbax.contracts.training import (
    LossTermSpec,
    OptimizerSpec,
    TimeAggregationSpec,
    TrainingSpec,
)


def _graph() -> GraphSpec:
    return GraphSpec(
        nodes={
            "network": ComponentSpec(
                type="Network",
                params={},
                input_ports=["input"],
                output_ports=["output", "hidden"],
            ),
            "plant": ComponentSpec(
                type="PointMass",
                params={},
                input_ports=["force"],
                output_ports=["effector"],
            ),
        },
        wires=[
            WireSpec(
                source_node="network",
                source_port="output",
                target_node="plant",
                target_port="force",
            ),
            WireSpec(
                source_node="network",
                source_port="hidden",
                target_node="network",
                target_port="input",
                temporality="recurrent",
            ),
        ],
        output_ports=["effector"],
        output_bindings={"effector": ("plant", "effector")},
    )


def _provider_graph() -> GraphSpec:
    return GraphSpec(
        nodes={
            "gain": ComponentSpec(
                type="Gain",
                params={"gain": 2.0},
                input_ports=["input"],
                output_ports=["output"],
            )
        },
        output_ports=["output"],
        output_bindings={"output": ("gain", "output")},
    )


def _training(loss: LossTermSpec) -> TrainingSpec:
    return TrainingSpec(
        optimizer=OptimizerSpec(type="adamw", params={"learning_rate": 0.001}),
        loss=loss,
        n_batches=1,
        batch_size=2,
    )


def test_normalize_selector_refs_from_strings_and_dicts() -> None:
    port = normalize_selector_ref("port:network.output")
    task = normalize_selector_ref({"namespace": "task_data", "path": "targets.effector"})
    state = normalize_selector_ref(
        {"compact": "path:states.mechanics.effector.pos", "namespace": "state_path"}
    )

    assert port.kind == "port"
    assert port.node_id == "network"
    assert port.port == "output"
    assert task.selector == "task_data:targets.effector"
    assert task.kind == "task_data"
    assert state.path == "states.mechanics.effector.pos"


def test_lowering_merges_explicit_and_loss_retention_requirements() -> None:
    graph = _graph()
    graph.retained_observables = [
        RetainedObservableSpec(
            id="obs:network-output",
            label="Network output",
            target=RetainedObservableTargetSpec(
                kind="port",
                selector="port:network.output",
                node_id="network",
                port="output",
            ),
            retention=RetentionPolicySpec(mode="stream", reason="probe"),
        )
    ]
    training = _training(
        LossTermSpec(
            type="TargetStateLoss",
            label="Output tracking",
            selector="port:network.output",
            target_selector="task_data:targets.effector",
            retention=RetentionPolicySpec(mode="trajectory", reason="loss"),
            norm="squared_l2",
        )
    )

    plan = lower_retention_plan(graph, training)
    by_selector = plan.by_selector

    assert set(by_selector) == {"port:network.output", "task_data:targets.effector"}
    output = by_selector["port:network.output"]
    assert output.id == "obs:network-output"
    assert output.explicit is True
    assert output.retention.mode == "trajectory"
    assert output.retention.reasons == ("probe", "loss")
    assert plan.loss_terms[0].source.selector == "port:network.output"
    assert plan.loss_terms[0].target is not None
    assert plan.loss_terms[0].target.selector == "task_data:targets.effector"


def test_analysis_input_requirements_lower_as_implicit_observables() -> None:
    graph = _graph()

    plan = lower_retention_plan(
        graph,
        analysis_input_requirements=[
            AnalysisInputRequirement(
                label="Hidden activity",
                selector="port:network.hidden",
                retention=RetentionPolicySpec(mode="trajectory"),
                value_schema={"dtype": "float32", "shape": ["time", "hidden"]},
                consumer=AnalysisInputConsumerSpec(
                    page_id="page:activity",
                    node_id="analysis-node:hidden",
                    input_port="activity",
                ),
            )
        ],
    )

    observable = plan.by_selector["port:network.hidden"]
    assert observable.explicit is False
    assert observable.retention.reasons == ("analysis_input",)
    assert observable.sources == ("/analysis/input_requirements/0",)
    assert observable.value_schema == {"dtype": "float32", "shape": ["time", "hidden"]}
    assert observable.metadata["source"] == "analysis_input"
    assert observable.metadata["consumer"]["page_id"] == "page:activity"
    assert observable.metadata["consumer"]["node_id"] == "analysis-node:hidden"
    assert observable.metadata["consumer"]["input_port"] == "activity"


def test_analysis_inputs_do_not_require_explicit_retained_observables() -> None:
    graph = _graph()

    plan = lower_retention_plan(
        graph,
        analysis_input_requirements=[
            {
                "selector": "graph_output:effector",
                "retention": {"mode": "trajectory"},
            }
        ],
    )

    observable = plan.by_selector["graph_output:effector"]
    assert observable.explicit is False
    assert observable.retention.mode == "trajectory"


def test_analysis_input_requirement_merges_with_explicit_capture_without_becoming_explicit_source() -> None:
    graph = _graph()
    graph.retained_observables = [
        RetainedObservableSpec(
            selector="port:network.output",
            retention=RetentionPolicySpec(mode="stream", reason="explicit_capture"),
        )
    ]

    plan = lower_retention_plan(
        graph,
        analysis_input_requirements=[
            AnalysisInputRequirement(
                selector="port:network.output",
                retention=RetentionPolicySpec(mode="trajectory"),
                consumer=AnalysisInputConsumerSpec(page_id="page:analysis"),
            )
        ],
    )

    observable = plan.by_selector["port:network.output"]
    assert observable.explicit is True
    assert observable.retention.mode == "trajectory"
    assert observable.retention.reasons == ("explicit_capture", "analysis_input")
    assert observable.sources == ("/graph/retained_observables/0", "/analysis/input_requirements/0")


def test_lowering_supports_structural_selector_kinds() -> None:
    graph = _graph()
    graph.retained_observables = [
        RetainedObservableSpec(
            selector="edge:network.output->plant.force",
            retention=RetentionPolicySpec(mode="trajectory"),
        ),
        RetainedObservableSpec(
            target=RetainedObservableTargetSpec(
                kind="recurrent_carry",
                selector="edge:network.hidden->network.input",
            ),
            retention=RetentionPolicySpec(mode="trajectory"),
        ),
        RetainedObservableSpec(selector="graph_output:effector"),
        RetainedObservableSpec(selector="path:states.plant.effector.pos"),
        RetainedObservableSpec(selector="task_data:targets.effector"),
    ]

    plan = lower_retention_plan(graph)

    assert {
        observable.selector.kind for observable in plan.observables
    } == {"edge", "recurrent_carry", "graph_output", "state_path", "task_data"}


@pytest.mark.parametrize("window_size", [None, 0, -1])
def test_window_retention_requires_positive_window_size(window_size: int | None) -> None:
    graph = _graph()
    graph.retained_observables = [
        RetainedObservableSpec(
            selector="graph_output:effector",
            retention=RetentionPolicySpec(mode="window", window_size=window_size),
        )
    ]

    with pytest.raises(RetentionPlanError) as exc_info:
        lower_retention_plan(graph)

    assert exc_info.value.path == "/graph/retained_observables/0/retention/window_size"
    assert "positive window_size" in str(exc_info.value)


@pytest.mark.parametrize("mode", ["stream", "window"])
def test_non_trajectory_retention_fails_pathfully_until_executable(mode: str) -> None:
    graph = _graph()
    retention = RetentionPolicySpec(mode=mode)
    if mode == "window":
        retention.window_size = 2
    graph.retained_observables = [
        RetainedObservableSpec(selector="graph_output:effector", retention=retention)
    ]

    with pytest.raises(RetentionPlanError) as exc_info:
        lower_retention_plan(graph)

    assert exc_info.value.path == "/graph/retained_observables/0/retention/mode"
    assert exc_info.value.selector == "graph_output:effector"
    assert "not supported by the current graph worker" in str(exc_info.value)


def test_lowering_reports_pathful_errors_for_unknown_selectors() -> None:
    graph = _graph()
    graph.retained_observables = [RetainedObservableSpec(selector="port:missing.output")]

    with pytest.raises(RetentionPlanError) as exc_info:
        lower_retention_plan(graph)

    assert exc_info.value.selector == "port:missing.output"
    assert exc_info.value.path == "/graph"
    assert "Unknown node" in str(exc_info.value)


def test_loss_target_value_norms_and_time_aggregation() -> None:
    graph = _graph()
    training = _training(
        LossTermSpec(
            type="TargetStateLoss",
            label="Final target",
            selector="graph_output:effector",
            target_value=[1.0, 1.0],
            norm="l1",
            time_agg=TimeAggregationSpec(mode="final"),
        )
    )
    plan = lower_retention_plan(graph, training)

    total, terms = evaluate_loss_plan(
        plan.loss_terms,
        {"graph_output:effector": jnp.asarray([[0.0, 1.0], [2.0, 3.0]])},
    )

    assert float(total) == pytest.approx(3.0)
    assert float(terms["loss"]) == pytest.approx(3.0)


def test_loss_target_selector_norms_and_range_aggregation() -> None:
    graph = _graph()
    training = _training(
        LossTermSpec(
            type="TargetStateLoss",
            label="Range target",
            selector="port:network.output",
            target_selector="task_data:targets.effector",
            norm="squared_l2",
            time_agg=TimeAggregationSpec(mode="range", start=1, end=3),
        )
    )
    plan = lower_retention_plan(graph, training)

    total, _terms = evaluate_loss_plan(
        plan.loss_terms,
        {
            "port:network.output": jnp.asarray(
                [[0.0, 0.0], [1.0, 1.0], [3.0, 2.0], [10.0, 10.0]]
            ),
            "task_data:targets.effector": jnp.asarray(
                [[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [10.0, 10.0]]
            ),
        },
    )

    # Selected timesteps are [1, 3): squared errors are 1 and 4; range uses mean.
    assert float(total) == pytest.approx(2.5)


def test_matrix_quadratic_loss_uses_target_centering_and_final_aggregation() -> None:
    graph = _graph()
    training = _training(
        LossTermSpec(
            type="MatrixQuadraticLoss",
            label="Terminal quadratic",
            selector="graph_output:effector",
            target_selector="task_data:targets.effector",
            matrix=[[2.0, 0.5], [0.5, 4.0]],
            matrix_kind="dense",
            time_agg=TimeAggregationSpec(mode="final"),
        )
    )
    plan = lower_retention_plan(graph, training)

    total, terms = evaluate_loss_plan(
        plan.loss_terms,
        {
            "graph_output:effector": jnp.asarray([[0.0, 0.0], [3.0, 5.0]]),
            "task_data:targets.effector": jnp.asarray([[0.0, 0.0], [1.0, 2.0]]),
        },
    )

    assert plan.loss_terms[0].matrix_kind == "dense"
    assert float(total) == pytest.approx(50.0)
    assert float(terms["loss"]) == pytest.approx(50.0)
    assert retention_plan_to_json(plan)["loss_terms"][0]["matrix"] == [
        [2.0, 0.5],
        [0.5, 4.0],
    ]


def test_matrix_quadratic_loss_rejects_mismatched_matrix_shape() -> None:
    term = LossTermSpec(
        type="MatrixQuadraticLoss",
        label="Bad quadratic",
        selector="graph_output:effector",
        matrix=[[1.0]],
    )
    plan = lower_retention_plan(_graph(), _training(term))

    with pytest.raises(RetentionPlanError, match="Dense matrix shape"):
        evaluate_loss_plan(
            plan.loss_terms,
            {"graph_output:effector": jnp.asarray([[1.0, 2.0]])},
        )


def test_matrix_quadratic_loss_requires_matrix_during_lowering() -> None:
    term = LossTermSpec(
        type="MatrixQuadraticLoss",
        label="Missing quadratic",
        selector="graph_output:effector",
    )

    with pytest.raises(RetentionPlanError, match="requires a matrix payload"):
        lower_retention_plan(_graph(), _training(term))


def test_loss_rejects_both_target_selector_and_target_value() -> None:
    graph = _graph()
    training = _training(
        LossTermSpec(
            type="TargetStateLoss",
            label="Invalid target",
            selector="graph_output:effector",
            target_selector="task_data:targets.effector",
            target_value=[0.0, 0.0],
        )
    )

    with pytest.raises(RetentionPlanError) as exc_info:
        lower_retention_plan(graph, training)

    assert exc_info.value.path == "/loss"
    assert "cannot specify both" in str(exc_info.value)


def test_segment_time_aggregation_rejected_during_lowering() -> None:
    graph = _graph()
    training = _training(
        LossTermSpec(
            type="TargetStateLoss",
            label="Segment target",
            selector="graph_output:effector",
            target_value=[0.0, 0.0],
            time_agg=TimeAggregationSpec(mode="segment", segment_name="movement"),
        )
    )

    with pytest.raises(RetentionPlanError) as exc_info:
        lower_retention_plan(graph, training)

    assert exc_info.value.path == "/task_spec/timeline"
    assert "timeline mask" in str(exc_info.value)


def _fixed_task_timeline() -> dict:
    return {
        "type": "DelayedReaches",
        "params": {"n_steps": 5},
        "timeline": {
            "schema_version": "feedbax.spec.studio.task_timeline.v1",
            "epochs": [
                {
                    "id": "epoch:0",
                    "label": "hold",
                    "index": 0,
                    "length": {
                        "schema_version": "feedbax.spec.studio.value.v1",
                        "mode": "constant",
                        "value": {"steps": 1},
                        "metadata": {"scope": "trial"},
                    },
                    "metadata": {},
                },
                {
                    "id": "epoch:1",
                    "label": "target_on",
                    "index": 1,
                    "length": {
                        "schema_version": "feedbax.spec.studio.value.v1",
                        "mode": "constant",
                        "value": {"steps": 2},
                        "metadata": {"scope": "trial"},
                    },
                    "metadata": {},
                },
                {
                    "id": "epoch:2",
                    "label": "movement",
                    "index": 2,
                    "length": {
                        "schema_version": "feedbax.spec.studio.value.v1",
                        "mode": "constant",
                        "value": None,
                        "metadata": {"inferred_from_remaining_steps": True},
                    },
                    "metadata": {},
                },
            ],
            "segments": [
                {"id": "cue_window", "label": "cue_window", "epoch_ids": ["epoch:0", "epoch:1"]},
            ],
            "metadata": {"n_steps": 5},
        },
    }


def test_segment_time_aggregation_lowers_fixed_task_timeline_mask() -> None:
    graph = _graph()
    training = _training(
        LossTermSpec(
            type="TargetStateLoss",
            label="Movement only",
            selector="graph_output:effector",
            target_value=[0.0, 0.0],
            time_agg=TimeAggregationSpec(mode="segment", segment_name="movement"),
        )
    )

    plan = lower_retention_plan(graph, training, task_spec=_fixed_task_timeline())
    assert plan.loss_terms[0].metadata["time_mask"]["epoch_ids"] == ["epoch:2"]

    total, _terms = evaluate_loss_plan(
        plan.loss_terms,
        {
            "graph_output:effector": jnp.asarray(
                [[1.0, 0.0], [10.0, 0.0], [20.0, 0.0], [3.0, 0.0], [4.0, 0.0]]
            ),
        },
    )

    assert float(total) == pytest.approx((9.0 + 16.0) / 2.0)


def test_segment_time_aggregation_supports_named_segment_groups() -> None:
    graph = _graph()
    training = _training(
        LossTermSpec(
            type="TargetStateLoss",
            label="Cue window",
            selector="graph_output:effector",
            target_value=[0.0, 0.0],
            time_agg=TimeAggregationSpec(mode="segment", segment_name="cue_window"),
        )
    )

    plan = lower_retention_plan(graph, training, task_spec=_fixed_task_timeline())
    total, _terms = evaluate_loss_plan(
        plan.loss_terms,
        {
            "graph_output:effector": jnp.asarray(
                [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [30.0, 0.0], [40.0, 0.0]]
            ),
        },
    )

    assert plan.loss_terms[0].metadata["time_mask"]["epoch_ids"] == ["epoch:0", "epoch:1"]
    assert float(total) == pytest.approx((1.0 + 4.0 + 9.0) / 3.0)


def test_segment_time_aggregation_rejects_unknown_segment_names() -> None:
    graph = _graph()
    training = _training(
        LossTermSpec(
            type="TargetStateLoss",
            label="Missing segment",
            selector="graph_output:effector",
            target_value=[0.0, 0.0],
            time_agg=TimeAggregationSpec(mode="segment", segment_name="post_go"),
        )
    )

    with pytest.raises(RetentionPlanError) as exc_info:
        lower_retention_plan(graph, training, task_spec=_fixed_task_timeline())

    assert exc_info.value.path == "/task_spec/timeline/segment_name"
    assert "available segments" in str(exc_info.value)
    assert "movement" in str(exc_info.value)


def test_segment_time_aggregation_rejects_variable_timeline_without_resolved_bounds() -> None:
    graph = _graph()
    training = _training(
        LossTermSpec(
            type="TargetStateLoss",
            label="Movement only",
            selector="graph_output:effector",
            target_value=[0.0, 0.0],
            time_agg=TimeAggregationSpec(mode="segment", segment_name="movement"),
        )
    )
    task = _fixed_task_timeline()
    task["timeline"]["epochs"][1]["length"]["value"] = {"min": 1, "max": 4}

    with pytest.raises(RetentionPlanError) as exc_info:
        lower_retention_plan(graph, training, task_spec=task)

    assert exc_info.value.path == "/task_spec/timeline/epochs/1/length"
    assert "Variable-length timeline epochs" in str(exc_info.value)


def test_loss_supports_l2_huber_and_sum_modes() -> None:
    graph = _graph()
    training = _training(
        LossTermSpec(
            type="Composite",
            label="Composite",
            children={
                "l2": LossTermSpec(
                    type="TargetStateLoss",
                    label="L2",
                    selector="graph_output:effector",
                    target_value=[0.0, 0.0],
                    norm="l2",
                    time_agg=TimeAggregationSpec(mode="sum"),
                ),
                "huber": LossTermSpec(
                    type="TargetStateLoss",
                    label="Huber",
                    selector="port:network.output",
                    target_value=[0.0, 0.0],
                    norm="huber",
                    time_agg=TimeAggregationSpec(mode="mean"),
                ),
            },
        )
    )
    plan = lower_retention_plan(graph, training)

    total, terms = evaluate_loss_plan(
        plan.loss_terms,
        {
            "graph_output:effector": jnp.asarray([[3.0, 4.0], [0.0, 5.0]]),
            "port:network.output": jnp.asarray([[0.5, 2.0], [0.0, 0.0]]),
        },
    )

    assert float(terms["loss.children.l2"]) == pytest.approx(10.0)
    assert float(terms["loss.children.huber"]) == pytest.approx((0.125 + 1.5) / 2)
    assert float(total) == pytest.approx(10.8125)


def test_provider_validation_uses_retention_lowering_for_loss_selectors() -> None:
    graph = _provider_graph()
    training = _training(
        LossTermSpec(
            type="TargetStateLoss",
            label="Output",
            selector="graph_output:missing",
            target_value=0.0,
        )
    )

    result = validate_training_spec(training, graph_spec=graph)

    assert result.valid is False
    assert result.errors[0].type == "loss_graph_mismatch"
    assert result.errors[0].location is not None
    assert result.errors[0].location["selector"] == "graph_output:missing"


def test_provider_validation_accepts_retained_observables_and_exposes_artifact_roles() -> None:
    graph = _provider_graph()
    graph.retained_observables = [RetainedObservableSpec(selector="graph_output:output")]

    result = validate_graph_spec(graph)
    manifest = provider_manifest()

    assert result.valid is True
    assert "retention_plan" in manifest.artifact_roles
    assert "retained_observables" in manifest.artifact_roles


def test_training_manifest_stores_retention_plan_and_observable_artifacts(tmp_path: Path) -> None:
    graph = _graph()
    training = _training(
        LossTermSpec(
            type="TargetStateLoss",
            label="Output",
            selector="graph_output:effector",
            target_value=0.0,
        )
    )
    plan = lower_retention_plan(graph, training)
    payload = retention_plan_to_json(plan)

    assert payload["schema_id"] == RETENTION_PLAN_SCHEMA_ID
    assert payload["schema_version"] == RETENTION_PLAN_SCHEMA_VERSION

    _manifest, path = write_training_run_manifest(
        job_id="job-retained",
        total_batches=1,
        training_spec=training.model_dump(mode="json", exclude_none=True),
        graph_spec=graph.model_dump(mode="json", exclude_none=True),
        retention_plan=payload,
        retained_observables={"graph_output:effector": [[0.0, 1.0]]},
        root=tmp_path / "runs",
        provenance=Provenance(source_commit="abc123", dirty=False),
    )

    loaded = load_manifest(path)
    roles = {artifact.role for artifact in loaded.artifacts}
    assert {"retention_plan", "retained_observables"}.issubset(roles)
    for artifact in loaded.artifacts:
        if artifact.role in {"retention_plan", "retained_observables"}:
            assert artifact.media_type == "application/json"
            assert artifact.uri is not None
            assert Path(artifact.uri).exists()
            assert isinstance(artifact.metadata, dict)
            stored = json.loads(Path(artifact.uri).read_text(encoding="utf-8"))
            if artifact.role == "retention_plan":
                assert artifact.metadata["schema_id"] == RETENTION_PLAN_SCHEMA_ID
                assert artifact.metadata["schema_version"] == RETENTION_PLAN_SCHEMA_VERSION
                assert stored["schema_id"] == RETENTION_PLAN_SCHEMA_ID
                assert stored["schema_version"] == RETENTION_PLAN_SCHEMA_VERSION
                assert "observables" in stored
                assert "loss_terms" in stored
            else:
                assert artifact.metadata["schema_id"] == RETAINED_OBSERVABLES_ARTIFACT_SCHEMA_ID
                assert (
                    artifact.metadata["schema_version"]
                    == RETAINED_OBSERVABLES_ARTIFACT_SCHEMA_VERSION
                )
                assert stored["schema_id"] == RETAINED_OBSERVABLES_ARTIFACT_SCHEMA_ID
                assert stored["schema_version"] == RETAINED_OBSERVABLES_ARTIFACT_SCHEMA_VERSION
                assert stored["observables"] == {"graph_output:effector": [[0.0, 1.0]]}


def test_training_manifest_rejects_unsupported_retention_artifact_version(
    tmp_path: Path,
) -> None:
    for version in (
        "feedbax.manifest.training.retention_plan.v1",
        "feedbax.manifest.training.retention_plan.v0",
    ):
        with pytest.raises(UnsupportedSpecVersion) as exc_info:
            write_training_run_manifest(
                job_id="job-old-retention",
                total_batches=1,
                retention_plan={
                    "schema_id": RETENTION_PLAN_SCHEMA_ID,
                    "schema_version": version,
                    "observables": [],
                    "loss_terms": [],
                },
                root=tmp_path / "runs",
                provenance=Provenance(source_commit="abc123", dirty=False),
            )

        message = str(exc_info.value)
        assert "RetentionPlan" in message
        assert version in message
        assert "migration_intentionally_absent=yes" in message


def test_load_manifest_rejects_unsupported_retention_artifact_ref_metadata(
    tmp_path: Path,
) -> None:
    manifest, path = write_training_run_manifest(
        job_id="job-retained-old-ref",
        total_batches=1,
        retention_plan={
            "observables": [],
            "loss_terms": [],
        },
        root=tmp_path / "runs",
        provenance=Provenance(source_commit="abc123", dirty=False),
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    artifact = next(item for item in data["artifacts"] if item["role"] == "retention_plan")
    artifact["metadata"]["schema_version"] = "feedbax.manifest.training.retention_plan.v0"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(UnsupportedSpecVersion) as exc_info:
        load_manifest(path)

    assert "artifacts/0/metadata" in str(exc_info.value)
    assert manifest.id in data["id"]


def test_retention_artifacts_are_registered_with_manifest_policy() -> None:
    families = {
        family.kind: family
        for family in default_spec_registry.families()
        if family.kind
        in {
            "RetentionPlan",
            "RetainedObservablePlan",
            "RetentionPolicyPlan",
            "LossTermPlan",
            "RetainedObservablesArtifact",
            "RetainedObservableSpec",
        }
    }

    assert families["RetainedObservableSpec"].identity == "feedbax.spec.graph.retained_observable"
    for kind in (
        "RetentionPlan",
        "RetainedObservablePlan",
        "RetentionPolicyPlan",
        "LossTermPlan",
        "RetainedObservablesArtifact",
    ):
        family = families[kind]
        assert family.namespace is not None
        assert family.namespace.value == "manifest"
        assert family.policy is not None
        assert family.policy.required_tests == ("tests/test_retained_observables.py",)
