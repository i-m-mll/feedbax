from __future__ import annotations

from dataclasses import replace

import pytest

from feedbax.contracts.checkpoints import (
    CheckpointContinuationRequest,
    CheckpointForkCompatibilityProjection,
    CheckpointForkPlan,
    CheckpointForkSourcePreparation,
    CheckpointForkTarget,
    CheckpointSegmentLineage,
    CheckpointTransactionManifest,
    ContinuationScheduleDiscontinuityExemption,
    RunContractBinding,
    ScheduleStateWindowSpec,
)
from feedbax.contracts.training import (
    GovernedScheduleProjection,
    LossTermSpec,
    LrScheduleSpec,
    MethodPayloadEnvelope,
    ObjectiveSlotSpec,
    OptimizerSpec,
    ScheduleProjection,
    ScheduleProjectionSample,
    RunControlSpec,
    TaskSpec,
    TrainingConfig,
    TrainingMethodScheduleProjector,
    TrainingRunSpec,
    WorkerExecutionSpec,
    standard_supervised_effective_phase_spec,
    standard_supervised_method_contract,
    standard_supervised_method_payload,
    standard_supervised_method_ref,
)
from feedbax.orchestration.schedule_eval import (
    compare_continuation_schedule_projections,
    project_training_schedules,
)
from feedbax.training.checkpoint_custody import (
    CheckpointCompatibilityError,
    authenticated_run_contract_source_projection,
)


RAMP_ID = "rlrmp.applied_damage"


def _minimal_graph() -> dict[str, object]:
    return {
        "nodes": {
            "gain": {
                "type": "Gain",
                "params": {"gain": 1.0},
                "input_ports": ["input"],
                "output_ports": ["output"],
            }
        },
        "wires": [],
        "input_ports": ["input"],
        "output_ports": ["output"],
        "input_bindings": {"input": ("gain", "input")},
        "output_bindings": {"output": ("gain", "output")},
    }


def _run_spec(
    *,
    continuation: CheckpointContinuationRequest | None = None,
    lr_schedule: LrScheduleSpec | None = None,
) -> TrainingRunSpec:
    payload = standard_supervised_method_payload().model_dump(mode="json")
    optimizer = OptimizerSpec(
        type="adamw",
        params={} if lr_schedule is not None else {"learning_rate": 1e-3},
        lr_schedule=lr_schedule,
    )
    payload["payload"]["optimizer"] = optimizer.model_dump(mode="json")
    return TrainingRunSpec(
        graph={"inline": _minimal_graph()},
        task=TaskSpec(type="ReachingTask", params={"n_steps": 4}),
        training_config=TrainingConfig(n_batches=4, batch_size=3),
        objective=ObjectiveSlotSpec(
            loss=LossTermSpec(type="target_state", label="target", selector="output")
        ),
        method_ref=standard_supervised_method_ref(),
        method_payload=MethodPayloadEnvelope.model_validate(payload),
        worker_execution=WorkerExecutionSpec(
            method_contract=standard_supervised_method_contract(),
            effective_phase=standard_supervised_effective_phase_spec(),
        ),
        checkpoint_progress={"continuation": continuation},
    )


def _with_method_schedule(
    run_spec: TrainingRunSpec,
    values: tuple[float, float, float] | None,
) -> TrainingRunSpec:
    resolved = run_spec.resolved_method
    assert resolved.descriptor is not None
    if values is None:
        projector = None
    else:
        projector = TrainingMethodScheduleProjector(
            projector_id="tests.applied_damage_projection",
            projector_version="v1",
            projector=lambda _payload, coordinates: ScheduleProjection(
                schedules=(
                    {
                        RAMP_ID: GovernedScheduleProjection(
                            origin={"kind": "run_start"},
                            samples=[
                                ScheduleProjectionSample(coordinate=coordinate, value=value)
                                for coordinate, value in zip(coordinates, values, strict=True)
                            ],
                        )
                    }
                    if values
                    else {}
                )
            ),
        )
    descriptor = replace(resolved.descriptor, schedule_projector=projector)
    run_spec._resolved_method = replace(resolved, descriptor=descriptor)
    run_spec._resolved_method_cache_key = run_spec._method_resolution_cache_key()
    return run_spec


def _source_manifest(*, boundary: int = 10, algorithm_version: str = "v4") -> CheckpointTransactionManifest:
    return CheckpointTransactionManifest.model_construct(
        transaction_id="tx-source",
        segment_lineage=CheckpointSegmentLineage(
            start_batch=0,
            segment_batch_count=boundary,
        ),
        run_contract_binding=RunContractBinding.model_construct(
            algorithm_version=algorithm_version,
            canonical_projection=None,
            canonical_projection_sha256=None,
        ),
    )


def _continuation(
    exemptions: list[ContinuationScheduleDiscontinuityExemption] | None = None,
) -> CheckpointContinuationRequest:
    return CheckpointContinuationRequest(
        source_completed_batches=10,
        additional_batches=4,
        schedule_discontinuity_exemptions=exemptions or [],
    )


def _compare(
    source_values: tuple[float, float, float],
    target_values: tuple[float, float, float],
    *,
    continuation: CheckpointContinuationRequest | None = None,
) -> tuple[list[str], dict[str, object]]:
    request = continuation or _continuation()
    return compare_continuation_schedule_projections(
        source_run_spec=_with_method_schedule(_run_spec(), source_values),
        target_run_spec=_with_method_schedule(
            _run_spec(continuation=request), target_values
        ),
        source_manifest=_source_manifest(),
        continuation=request,
    )


def test_incident_ramp_discontinuity_fails_with_id_and_values() -> None:
    failures, _ = _compare((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))

    assert len(failures) == 1
    assert RAMP_ID in failures[0]
    assert "source=[1.0, 1.0, 1.0]" in failures[0]
    assert "target=[0.0, 0.0, 0.0]" in failures[0]


def test_corrected_hold_at_one_projection_passes() -> None:
    failures, observed = _compare((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))

    assert failures == []
    assert observed["boundary_side"] == "pre_update"
    assert observed["coordinates"] == [10, 11, 12]


def test_source_history_projection_accepts_continuation_bearing_run_spec() -> None:
    request = _continuation()
    failures, observed = compare_continuation_schedule_projections(
        source_run_spec=_with_method_schedule(
            _run_spec(continuation=request), (1.0, 1.0, 1.0)
        ),
        target_run_spec=_with_method_schedule(
            _run_spec(continuation=request), (1.0, 1.0, 1.0)
        ),
        source_manifest=_source_manifest(),
        continuation=request,
    )

    assert failures == []
    assert observed["coordinates"] == [10, 11, 12]


def test_lineage_projector_uses_segment_origin_for_boundary_and_later_call() -> None:
    request = _continuation()
    run_spec = _run_spec(continuation=request)
    resolved = run_spec.resolved_method
    assert resolved.descriptor is not None

    def project_segment_schedule(_payload, coordinates, lineage):
        return ScheduleProjection(schedules={
            RAMP_ID: GovernedScheduleProjection(
                origin={"kind": "segment_start"},
                samples=[ScheduleProjectionSample(
                    coordinate=coordinate,
                    value=float(coordinate - lineage.start_batch),
                ) for coordinate in coordinates],
            )
        })

    descriptor = replace(
        resolved.descriptor,
        schedule_projector=TrainingMethodScheduleProjector(
            projector_id="tests.segment_local_projection",
            projector_version="v1",
            lineage_projector=project_segment_schedule,
        ),
    )
    run_spec._resolved_method = replace(resolved, descriptor=descriptor)
    run_spec._resolved_method_cache_key = run_spec._method_resolution_cache_key()
    lineage = CheckpointSegmentLineage(
        parent_transaction_id="tx-source",
        start_batch=10,
        segment_batch_count=4,
    )

    boundary = project_training_schedules(
        run_spec, coordinates=(10, 11, 12), lineage=lineage
    )
    later = project_training_schedules(run_spec, coordinates=(12,), lineage=lineage)

    assert [sample.coordinate for sample in boundary.schedules[RAMP_ID].samples] == [10, 11, 12]
    assert [sample.value for sample in boundary.schedules[RAMP_ID].samples] == [0.0, 1.0, 2.0]
    assert later.schedules[RAMP_ID].samples[0].value == 2.0


def test_existing_global_projector_remains_two_argument_compatible() -> None:
    projector = TrainingMethodScheduleProjector(
        projector_id="tests.global_projection",
        projector_version="v1",
        projector=lambda _payload, coordinates: ScheduleProjection(),
    )

    assert projector.project(standard_supervised_method_payload().payload, (12,)).schedules == {}


def test_lineage_projector_rejects_missing_or_contradictory_origin() -> None:
    projector = TrainingMethodScheduleProjector(
        projector_id="tests.missing_lineage_projection",
        projector_version="v1",
        lineage_projector=lambda _payload, _coordinates, _lineage: ScheduleProjection(),
    )
    with pytest.raises(ValueError, match="requires CheckpointSegmentLineage"):
        projector.project(standard_supervised_method_payload().payload, (10,))

    run_spec = _with_method_schedule(
        _run_spec(continuation=_continuation()), (1.0, 1.0, 1.0)
    )
    with pytest.raises(ValueError, match="lineage contradicts continuation"):
        project_training_schedules(
            run_spec,
            coordinates=(10, 11, 12),
            lineage=CheckpointSegmentLineage(
                parent_transaction_id="tx-source",
                start_batch=9,
                segment_batch_count=4,
            ),
        )


def test_divergence_only_at_first_update_fails() -> None:
    failures, _ = _compare((1.0, 1.0, 1.0), (1.0, 0.0, 1.0))

    assert len(failures) == 1
    assert RAMP_ID in failures[0]


def test_learning_rate_origin_mismatch_fails() -> None:
    request = _continuation()
    source = _with_method_schedule(
        _run_spec(
            lr_schedule=LrScheduleSpec(
                origin={"kind": "run_start"},
                kind="delayed_cosine",
                learning_rate_0=1e-3,
                total_steps=20,
            )
        ),
        (),
    )
    target = _with_method_schedule(
        _run_spec(
            continuation=request,
            lr_schedule=LrScheduleSpec(
                origin={"kind": "segment_start"},
                kind="delayed_cosine",
                learning_rate_0=1e-3,
                total_steps=20,
            ),
        ),
        (),
    )

    failures, _ = compare_continuation_schedule_projections(
        source_run_spec=source,
        target_run_spec=target,
        source_manifest=_source_manifest(),
        continuation=request,
    )

    assert len(failures) == 1
    assert "feedbax.learning_rate" in failures[0]
    assert "source_origin" in failures[0]


def test_matching_typed_exemption_passes() -> None:
    exemption = ContinuationScheduleDiscontinuityExemption(
        schedule_id=RAMP_ID,
        boundary_batch=10,
        expected_source_state=ScheduleStateWindowSpec(
            boundary=1.0, first_update=1.0, second_update=1.0
        ),
        expected_target_state=ScheduleStateWindowSpec(
            boundary=0.0, first_update=0.5, second_update=1.0
        ),
        intended_first_update_behavior="increase",
        reason="deliberate longer anneal",
    )

    failures, observed = _compare(
        (1.0, 1.0, 1.0),
        (0.0, 0.5, 1.0),
        continuation=_continuation([exemption]),
    )

    assert failures == []
    assert observed["used_exemptions"] == [RAMP_ID]


def test_typed_exemption_matches_values_within_schedule_tolerance() -> None:
    exemption = ContinuationScheduleDiscontinuityExemption(
        schedule_id=RAMP_ID,
        boundary_batch=10,
        expected_source_state=ScheduleStateWindowSpec(
            boundary=1.0 + 1e-12, first_update=1.0 + 1e-12, second_update=1.0 + 1e-12
        ),
        expected_target_state=ScheduleStateWindowSpec(
            boundary=0.0, first_update=0.5 + 5e-13, second_update=1.0 + 1e-12
        ),
        intended_first_update_behavior="increase",
        reason="deliberate longer anneal",
    )

    failures, observed = _compare(
        (1.0, 1.0, 1.0),
        (0.0, 0.5, 1.0),
        continuation=_continuation([exemption]),
    )

    assert failures == []
    assert observed["used_exemptions"] == [RAMP_ID]


def test_unused_exemption_fails() -> None:
    exemption = ContinuationScheduleDiscontinuityExemption(
        schedule_id=RAMP_ID,
        boundary_batch=10,
        expected_source_state={"boundary": 1.0, "first_update": 1.0, "second_update": 1.0},
        expected_target_state={"boundary": 1.0, "first_update": 1.0, "second_update": 1.0},
        intended_first_update_behavior="hold",
        reason="should not be needed",
    )

    failures, _ = _compare(
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        continuation=_continuation([exemption]),
    )

    assert failures == [f"unused schedule discontinuity exemptions: ['{RAMP_ID}']"]


def test_method_descriptor_without_projector_fails_for_continuation() -> None:
    request = _continuation()

    with pytest.raises(ValueError, match="no complete schedule projector"):
        compare_continuation_schedule_projections(
            source_run_spec=_with_method_schedule(_run_spec(), (1.0, 1.0, 1.0)),
            target_run_spec=_with_method_schedule(
                _run_spec(continuation=request), None
            ),
            source_manifest=_source_manifest(),
            continuation=request,
        )


def test_pre_v4_source_projection_fails_closed() -> None:
    manifest = _source_manifest(algorithm_version="feedbax.training_checkpoint.run_contract_binding.v3")

    with pytest.raises(CheckpointCompatibilityError, match="algorithm v4"):
        authenticated_run_contract_source_projection(manifest)


def test_continuation_v2_migrates_to_empty_exemption_inventory() -> None:
    request = CheckpointContinuationRequest.model_validate(
        {
            "schema_id": "feedbax.spec.training_checkpoint_continuation",
            "schema_version": "feedbax.spec.training_checkpoint_continuation.v2",
            "source_completed_batches": 10,
            "additional_batches": 4,
            "self_contained": False,
        }
    )

    assert request.schema_version == "feedbax.spec.training_checkpoint_continuation.v3"
    assert request.schedule_discontinuity_exemptions == []


def test_embedded_v2_continuations_migrate_in_run_control_and_fork_plan() -> None:
    legacy = {
        "schema_id": "feedbax.spec.training_checkpoint_continuation",
        "schema_version": "feedbax.spec.training_checkpoint_continuation.v2",
        "source_completed_batches": 10,
        "additional_batches": 4,
        "self_contained": False,
    }
    control = RunControlSpec(n_batches=4, batch_size=3, continuation=legacy)
    plan = CheckpointForkPlan(
        source=CheckpointForkSourcePreparation(checkpoint_root_ref="source"),
        targets=[
            CheckpointForkTarget(
                target_id="target",
                checkpoint_root_ref="target",
                run_spec_ref="run",
                slot_template_ref="slots",
                compatibility=CheckpointForkCompatibilityProjection(
                    run_contract_algorithm_version="v4",
                    run_contract_hash_domain="tests",
                    run_contract_projection_sha256="a" * 64,
                    slot_structural_abi_sha256={"model": "b" * 64},
                ),
                history_policy={
                    "mode": "prepare_continuation",
                    "continuation_request": legacy,
                },
            )
        ],
    )

    assert control.continuation is not None
    assert control.continuation.schema_version.endswith(".v3")
    nested = plan.targets[0].history_policy.continuation_request
    assert nested is not None
    assert nested.schema_version.endswith(".v3")
