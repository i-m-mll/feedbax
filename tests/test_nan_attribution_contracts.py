from __future__ import annotations

import pytest
import jax.numpy as jnp
from pydantic import ValidationError

from feedbax.contracts.nan_attribution import (
    MAX_SLOT_LEAF_EXEMPLARS,
    AxisNonFiniteSummary,
    MetricNonFiniteSummary,
    NanAttributionDetection,
    NanAttributionRestorationOutcome,
    NanGuardCoordinate,
    RestoredCheckpointIdentity,
    SlotLeafNonFiniteExemplar,
    SlotNonFiniteSummary,
)
from feedbax.contracts.training import (
    BatchScheduleOriginSpec,
    GovernedScheduleProjection,
    ScheduleProjection,
    ScheduleProjectionSample,
)
from feedbax.contracts.worker import MaterializedSlotAxisBinding, ProgressCoordinate
from feedbax.training import executor as training_executor
from feedbax.training.diagnostics import (
    TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V2,
    TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V3,
    TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V4,
    TrainingDiagnostics,
)


def _shared(*, nan_count: int = 1, inf_count: int = 0) -> AxisNonFiniteSummary:
    return AxisNonFiniteSummary(
        mode="shared",
        nan_count=nan_count,
        inf_count=inf_count,
    )


def _mapped() -> AxisNonFiniteSummary:
    return AxisNonFiniteSummary(
        mode="mapped",
        axis_role="replicate",
        array_axis=0,
        axis_size=3,
        axis_indices=(0, 1, 2),
        nan_count=2,
        inf_count=1,
        nan_mask=(False, True, False),
        inf_mask=(False, False, True),
        nan_counts_by_axis=(0, 2, 0),
        inf_counts_by_axis=(0, 0, 1),
    )


def _schedule_state() -> ScheduleProjection:
    return ScheduleProjection(
        schedules={
            "method.temperature": GovernedScheduleProjection(
                origin=BatchScheduleOriginSpec(kind="run_start"),
                samples=[ScheduleProjectionSample(coordinate=2, value=0.5)],
            )
        }
    )


def _detection(**updates: object) -> NanAttributionDetection:
    values: dict[str, object] = {
        "run_id": "run-1",
        "on_nan": "raise",
        "coordinate": NanGuardCoordinate(
            completed_batches_as_observed=None,
            completed_batches_observation="unavailable",
            completed_batches_observation_error="not available in contract fixture",
            program_step=2,
            outer_step=1,
            inner_step=0,
        ),
        "metrics": (MetricNonFiniteSummary(metric_name="loss", non_finite=_mapped()),),
        "total_offending_metrics": 1,
        "metrics_truncated": False,
        "slots": (),
        "total_offending_slots": 0,
        "slots_truncated": False,
        "schedule_state": _schedule_state(),
        "schedule_coordinate_source": "segment_start_fallback",
        "total_schedules": 1,
        "schedule_state_truncated": False,
    }
    values.update(updates)
    return NanAttributionDetection.model_validate(values)


def test_detection_represents_shared_and_mapped_metrics_without_raw_values() -> None:
    detection = _detection(
        on_nan="halt_restore_checkpoint",
        metrics=(
            MetricNonFiniteSummary(metric_name="mapped_loss", non_finite=_mapped()),
            MetricNonFiniteSummary(metric_name="shared_loss", non_finite=_shared()),
        ),
        total_offending_metrics=2,
    )

    payload = detection.model_dump(mode="json")
    assert payload["metrics"][0]["non_finite"]["nan_mask"] == [False, True, False]
    assert payload["metrics"][0]["non_finite"]["axis_role"] == "replicate"
    assert payload["metrics"][1]["non_finite"]["nan_count"] == 1
    assert payload["metrics"][1]["non_finite"]["nan_mask"] is None
    assert all("value" not in metric for metric in payload["metrics"])
    assert all("value" not in slot for slot in payload["slots"])
    assert payload["coordinate"]["provenance"] == "guard_pre_authoritative_extraction"


def test_mapped_summary_rejects_inconsistent_axis_masks_and_counts() -> None:
    with pytest.raises(ValidationError, match="nan_mask"):
        _mapped().model_copy(update={"nan_mask": (True, False, False)}).__class__.model_validate(
            {
                **_mapped().model_dump(),
                "nan_mask": (True, False, False),
            }
        )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (jnp.asarray([1.0, 2.0]), "expected axis 'ensemble' size 3"),
        (1.0, "does not have declared array_axis 0; leaf rank is 0"),
    ],
)
def test_mapped_attribution_rejects_leaves_that_do_not_conform_to_axis(
    value: object,
    message: str,
) -> None:
    binding = MaterializedSlotAxisBinding(
        axis="ensemble",
        role="replicate",
        size=3,
        level=0,
        mode="mapped",
        array_axis=0,
    )

    with pytest.raises(training_executor.TrainingRunExecutorError, match=message):
        training_executor._project_nonfinite_leaves(value, (binding,))


def test_slot_leaf_inventory_is_bounded_and_reports_truncation() -> None:
    exemplar = SlotLeafNonFiniteExemplar(
        leaf_path="predicate.validity[1]",
        nan_count=1,
        inf_count=0,
        non_finite=_shared(),
    )
    summary = SlotNonFiniteSummary(
        slot_name="predicate_components",
        total_leaf_count=20,
        offending_leaf_count=17,
        nonfinite_leaf_count=17,
        predicate_false_leaf_count=0,
        nan_count=17,
        inf_count=0,
        exemplars=(exemplar,) * MAX_SLOT_LEAF_EXEMPLARS,
        truncated=True,
    )
    detection = _detection(
        slots=(summary,),
        total_offending_slots=2,
        slots_truncated=True,
    )

    assert detection.slots[0].truncated is True
    assert detection.slots_truncated is True
    with pytest.raises(ValidationError):
        SlotNonFiniteSummary(
            slot_name="slot",
            total_leaf_count=20,
            offending_leaf_count=17,
            nonfinite_leaf_count=17,
            predicate_false_leaf_count=0,
            nan_count=17,
            inf_count=0,
            exemplars=(exemplar,) * (MAX_SLOT_LEAF_EXEMPLARS + 1),
            truncated=False,
        )


def test_restoration_outcome_supports_restored_failed_and_not_attempted() -> None:
    digest = "a" * 64
    restored = NanAttributionRestorationOutcome(
        run_id="run-1",
        detection_artifact_sha256=digest,
        status="restored",
        restored_transaction=RestoredCheckpointIdentity(
            transaction_id="transaction-1",
            completed_batches=1,
            coordinate=ProgressCoordinate(run_id="run-1", phase="train", program_step=1),
        ),
    )
    failed = NanAttributionRestorationOutcome(
        run_id="run-1",
        detection_artifact_sha256=digest,
        status="failed",
        restore_failure="checkpoint unavailable",
    )
    not_attempted = NanAttributionRestorationOutcome(
        run_id="run-1",
        detection_artifact_sha256=digest,
        status="not_attempted",
        not_attempted_reason="on_nan=raise",
    )

    assert restored.restored_transaction is not None
    assert failed.restore_failure == "checkpoint unavailable"
    assert not_attempted.not_attempted_reason == "on_nan=raise"


@pytest.mark.parametrize(
    "schema_version",
    [TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V2, TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V3],
)
def test_healthy_legacy_diagnostics_remain_valid_and_byte_stable(schema_version: str) -> None:
    diagnostics = TrainingDiagnostics(
        schema_version=schema_version,
        manifest_id="manifest",
        run_id="run",
        terminal_status="completed",
        completed_batches=0,
        segment_completed_batches=0,
        cumulative_completed_batches=0,
    )

    assert diagnostics.failure_kind is None
    assert "failure_kind" not in diagnostics.model_dump_json()


def test_failed_diagnostics_require_v4_nan_guard_failure_kind() -> None:
    diagnostics = TrainingDiagnostics(
        schema_version=TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V4,
        manifest_id="manifest",
        run_id="run",
        terminal_status="failed",
        failure_kind="nan_guard",
        completed_batches=0,
        segment_completed_batches=0,
        cumulative_completed_batches=0,
    )
    assert diagnostics.failure_kind == "nan_guard"

    with pytest.raises(ValidationError, match="failure_kind='nan_guard'"):
        TrainingDiagnostics(
            schema_version=TRAINING_DIAGNOSTICS_SCHEMA_VERSION_V4,
            manifest_id="manifest",
            run_id="run",
            terminal_status="failed",
            completed_batches=0,
            segment_completed_batches=0,
            cumulative_completed_batches=0,
        )
