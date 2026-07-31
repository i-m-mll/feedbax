"""Shared learning-rate schedule evaluation for orchestration checks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from feedbax.contracts.checkpoints import (
    CheckpointContinuationRequest,
    CheckpointSegmentLineage,
    CheckpointTransactionManifest,
    ScheduleStateWindowSpec,
)
from feedbax.contracts.training import (
    BatchScheduleOriginSpec,
    GovernedScheduleProjection,
    OptimizerSpec,
    ResolvedTrainingMethod,
    ScheduleProjection,
    ScheduleProjectionSample,
    TrainingRunSpec,
)


LEARNING_RATE_SCHEDULE_ID = "feedbax.learning_rate"
SCHEDULE_REL_TOL = 1e-9


@dataclass(frozen=True)
class ScheduleEvalContext:
    """Concrete inputs passed to ``build_optimizer`` schedule construction."""

    schedule_origin_step: int
    current_step: int
    optimizer_count_at_current_step: int

    def model_dump(self) -> dict[str, int]:
        return {
            "schedule_origin_step": self.schedule_origin_step,
            "current_step": self.current_step,
            "optimizer_count_at_current_step": self.optimizer_count_at_current_step,
        }


class MissingScheduleContext(ValueError):
    """Raised when scheduled optimizer evaluation lacks resume/build context."""


def learning_rate_from_build_optimizer(
    optimizer_spec: OptimizerSpec,
    *,
    sample_step: int,
    schedule_origin_step: int,
    current_step: int,
    optimizer_count_at_current_step: int,
) -> float:
    """Evaluate the live scheduled LR through the same optimizer builder used by runs."""
    import jax.numpy as jnp
    import jax.tree as jt

    from feedbax.training.optimizers import build_optimizer

    if sample_step < current_step:
        raise ValueError(f"lr_trace sample_step={sample_step} precedes current_step={current_step}")
    if optimizer_spec.lr_schedule is None:
        learning_rate = optimizer_spec.params.get("learning_rate")
        if learning_rate is None:
            raise ValueError("/params/learning_rate is required when lr_schedule is absent")
        return float(learning_rate)
    optimizer = build_optimizer(
        optimizer_spec,
        schedule_origin_step=schedule_origin_step,
        current_step=current_step,
        optimizer_count_at_current_step=optimizer_count_at_current_step,
    )
    params = {"w": jnp.asarray(1.0)}
    state = optimizer.init(params)
    target_count = optimizer_count_at_current_step + (sample_step - current_step)
    state = _with_injected_count(state, target_count)
    grads = jt.map(jnp.zeros_like, params)
    _updates, next_state = optimizer.update(grads, state, params)
    return _scheduled_learning_rate(next_state)


def extract_resume_context(
    bundle_row_spec: Mapping[str, Any] | None,
    training_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract declared resume/fork schedule context from row artifacts.

    Top-level row declarations take precedence, followed by diagnostics and
    finally governed row metadata.
    """
    raw = _first_present(
        _path(bundle_row_spec, "resume_context"),
        _path(training_diagnostics, "resume_context"),
        {},
    )
    metadata_raw = _path(bundle_row_spec, "metadata", "resume_context")
    return {
        "schedule_origin_step": _first_present(
            _path(raw, "schedule_origin_step"),
            _path(bundle_row_spec, "schedule_origin_step"),
            _path(training_diagnostics, "schedule_origin_step"),
            _path(metadata_raw, "schedule_origin_step"),
            _path(bundle_row_spec, "metadata", "schedule_origin_step"),
        ),
        "current_step": _first_present(
            _path(raw, "current_step"),
            _path(bundle_row_spec, "current_step"),
            _path(training_diagnostics, "current_step"),
            _path(metadata_raw, "current_step"),
            _path(bundle_row_spec, "metadata", "current_step"),
        ),
        "optimizer_count_at_current_step": _first_present(
            _path(raw, "optimizer_count_at_current_step"),
            _path(bundle_row_spec, "optimizer_count_at_current_step"),
            _path(training_diagnostics, "optimizer_count_at_current_step"),
            _path(metadata_raw, "optimizer_count_at_current_step"),
            _path(bundle_row_spec, "metadata", "optimizer_count_at_current_step"),
        ),
    }


def extract_optimizer_build_context(
    bundle_row_spec: Mapping[str, Any] | None,
    training_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract the explicit context the run path declares for optimizer construction.

    The executor's construction context must be recorded separately from the
    declared resume context. Falling back to the declaration would turn this
    check into a self-comparison and conceal dropped or mis-wired context.
    Existing row and diagnostics declarations take precedence over governed
    row metadata.
    """
    raw = _first_present(
        _path(bundle_row_spec, "optimizer_build_context"),
        _path(bundle_row_spec, "build_optimizer_context"),
        _path(bundle_row_spec, "executor_optimizer_context"),
        _path(bundle_row_spec, "training", "optimizer_build_context"),
        _path(bundle_row_spec, "method_payload", "payload", "optimizer_build_context"),
        _path(training_diagnostics, "optimizer_build_context"),
        _path(bundle_row_spec, "metadata", "optimizer_build_context"),
        _MISSING,
    )
    if raw is _MISSING:
        return {
            "schedule_origin_step": _MISSING,
            "current_step": _MISSING,
            "optimizer_count_at_current_step": _MISSING,
        }
    return {
        "schedule_origin_step": _path(raw, "schedule_origin_step"),
        "current_step": _path(raw, "current_step"),
        "optimizer_count_at_current_step": _path(raw, "optimizer_count_at_current_step"),
    }


def require_schedule_context(context: Mapping[str, Any], *, label: str) -> ScheduleEvalContext:
    """Validate and coerce schedule context for optimizer schedule evaluation."""
    missing = [
        key
        for key in ("schedule_origin_step", "current_step", "optimizer_count_at_current_step")
        if context.get(key) is _MISSING
    ]
    if missing:
        raise MissingScheduleContext(f"{label} missing {', '.join(missing)}")
    return ScheduleEvalContext(
        schedule_origin_step=int(context["schedule_origin_step"]),
        current_step=int(context["current_step"]),
        optimizer_count_at_current_step=int(context["optimizer_count_at_current_step"]),
    )


def schedule_sample_steps(
    optimizer_spec: OptimizerSpec,
    context: ScheduleEvalContext,
    *,
    run_end_step: int | None = None,
    minimum: int = 4,
) -> tuple[int, ...]:
    """Choose schedule-relevant global sample steps at or after the current step.

    ``run_end_step`` is the post-completion coordinate: one beyond the final
    applied update. When it extends past a cosine endpoint, the samples prove
    the terminal rate at the endpoint, the first post-terminal coordinate, the
    final applied update, and the reported completion coordinate.
    """
    schedule = optimizer_spec.lr_schedule
    local_positions = {0, 1, 2, 3}
    if schedule is not None and schedule.total_steps is not None:
        terminal = int(schedule.total_steps)
        warmup_or_hold = int(schedule.constant_lr_iterations)
        mid_decay = warmup_or_hold + max((terminal - warmup_or_hold) // 2, 1)
        local_positions = {0, warmup_or_hold, mid_decay, terminal}
        if run_end_step is not None:
            terminal_step = context.schedule_origin_step + terminal
            if run_end_step > terminal_step:
                local_positions.add(terminal + 1)
            for sample_step in (run_end_step - 1, run_end_step):
                local_position = sample_step - context.schedule_origin_step
                if local_position >= 0:
                    local_positions.add(local_position)
        cursor = terminal - 1
        while len(local_positions) < minimum and cursor > 0:
            local_positions.add(cursor)
            cursor -= 1
    cursor = 0
    while len(local_positions) < minimum:
        local_positions.add(cursor)
        cursor += 1
    sample_steps = {
        context.schedule_origin_step + position
        for position in local_positions
        if context.schedule_origin_step + position >= context.current_step
    }
    cursor = context.current_step
    while len(sample_steps) < minimum:
        sample_steps.add(cursor)
        cursor += 1
    return tuple(sorted(sample_steps))


def evaluate_schedule_samples(
    optimizer_spec: OptimizerSpec,
    context: ScheduleEvalContext,
    sample_steps: Sequence[int],
) -> dict[int, float]:
    """Evaluate scheduled LR values for sample steps using ``build_optimizer``."""
    return {
        int(step): learning_rate_from_build_optimizer(
            optimizer_spec,
            sample_step=int(step),
            schedule_origin_step=context.schedule_origin_step,
            current_step=context.current_step,
            optimizer_count_at_current_step=context.optimizer_count_at_current_step,
        )
        for step in sample_steps
    }


def compare_schedule_samples(
    optimizer_spec: OptimizerSpec,
    *,
    expected_context: ScheduleEvalContext,
    observed_context: ScheduleEvalContext,
    run_end_step: int | None = None,
    rel_tol: float = 1e-9,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return sampled LR evidence and the subset whose values differ."""
    sample_steps = schedule_sample_steps(
        optimizer_spec,
        expected_context,
        run_end_step=run_end_step,
    )
    expected = evaluate_schedule_samples(optimizer_spec, expected_context, sample_steps)
    observed = evaluate_schedule_samples(optimizer_spec, observed_context, sample_steps)
    samples = [
        {
            "sample_step": step,
            "schedule_position": step - expected_context.schedule_origin_step,
            "expected": expected[step],
            "observed": observed[step],
        }
        for step in sample_steps
    ]
    mismatches = [
        sample
        for sample in samples
        if not math.isclose(
            float(sample["observed"]),
            float(sample["expected"]),
            rel_tol=rel_tol,
            abs_tol=0.0,
        )
    ]
    return samples, mismatches


def project_training_schedules(
    run_spec: TrainingRunSpec,
    *,
    coordinates: Sequence[int],
    lineage: CheckpointSegmentLineage,
    resolved_method: ResolvedTrainingMethod[Any],
    validate_continuation_lineage: bool = True,
) -> ScheduleProjection:
    """Build the complete sampled schedule table consumed by runtime and preflight.

    Coordinates use the pre-update side of the global training-batch boundary:
    coordinate ``N`` is the schedule value applied by update ``N``.
    """
    normalized_coordinates = tuple(int(coordinate) for coordinate in coordinates)
    if not normalized_coordinates or normalized_coordinates != tuple(
        sorted(set(normalized_coordinates))
    ):
        raise ValueError("schedule projection coordinates must be non-empty, unique, and sorted")
    continuation = run_spec.checkpoint_progress.continuation
    if validate_continuation_lineage and continuation is not None:
        expected_lineage = (
            continuation.source_completed_batches,
            continuation.additional_batches,
        )
        actual_lineage = (lineage.start_batch, lineage.segment_batch_count)
        if actual_lineage != expected_lineage:
            raise ValueError(
                "schedule projection lineage contradicts continuation; "
                f"lineage={actual_lineage!r}, continuation={expected_lineage!r}"
            )
    descriptor = resolved_method.descriptor
    if descriptor is None or descriptor.schedule_projector is None:
        raise ValueError(
            f"training method {run_spec.method_ref.key!r} has no complete schedule projector"
        )
    method_projection = descriptor.schedule_projector.project(
        resolved_method.payload,
        normalized_coordinates,
        lineage=lineage,
    )
    if descriptor.optimizer_spec_projector is None:
        raise ValueError(
            f"training method {run_spec.method_ref.key!r} has no optimizer spec projector"
        )
    optimizer = OptimizerSpec.model_validate(
        descriptor.optimizer_spec_projector(resolved_method.payload)
    )
    if LEARNING_RATE_SCHEDULE_ID in method_projection.schedules:
        raise ValueError(
            f"method schedule projector must not emit reserved ID {LEARNING_RATE_SCHEDULE_ID!r}"
        )
    schedule = optimizer.lr_schedule
    if schedule is None:
        origin = BatchScheduleOriginSpec(kind="run_start")
        learning_rate = optimizer.params.get("learning_rate")
        if learning_rate is None:
            return method_projection
        lr_values = {
            coordinate: _normalize_schedule_value(learning_rate)
            for coordinate in normalized_coordinates
        }
    else:
        from feedbax.training.schedule_clocks import resolve_schedule_window

        origin = schedule.origin
        window = resolve_schedule_window(
            origin,
            lineage=lineage,
            duration=schedule.total_steps,
            allow_inert=schedule.allow_inert,
        )
        context = ScheduleEvalContext(
            schedule_origin_step=window.start_batch,
            current_step=normalized_coordinates[0],
            optimizer_count_at_current_step=normalized_coordinates[0],
        )
        lr_values = {
            coordinate: _normalize_schedule_value(value)
            for coordinate, value in evaluate_schedule_samples(
                optimizer, context, normalized_coordinates
            ).items()
        }
    schedules = dict(method_projection.schedules)
    schedules[LEARNING_RATE_SCHEDULE_ID] = GovernedScheduleProjection(
        origin=origin,
        samples=[
            ScheduleProjectionSample(coordinate=coordinate, value=lr_values[coordinate])
            for coordinate in normalized_coordinates
        ],
    )
    return ScheduleProjection(schedules=dict(sorted(schedules.items())))


def compare_continuation_schedule_projections(
    *,
    source_run_spec: TrainingRunSpec,
    target_run_spec: TrainingRunSpec,
    source_manifest: CheckpointTransactionManifest,
    continuation: CheckpointContinuationRequest,
    source_resolved_method: ResolvedTrainingMethod[Any],
    target_resolved_method: ResolvedTrainingMethod[Any],
) -> tuple[list[str], dict[str, Any]]:
    """Compare source-realized and target-declared schedules over ``N..N+2``."""
    boundary = continuation.source_completed_batches
    source_segment_end = (
        source_manifest.segment_lineage.start_batch
        + source_manifest.segment_lineage.segment_batch_count
    )
    if source_segment_end != boundary:
        raise ValueError(
            "source checkpoint lineage contradicts continuation boundary; "
            f"source_segment_end={source_segment_end}, boundary={boundary}"
        )
    if target_run_spec.checkpoint_progress.continuation != continuation:
        raise ValueError("target run continuation contradicts requested continuation")
    coordinates = (boundary, boundary + 1, boundary + 2)
    source = project_training_schedules(
        source_run_spec,
        coordinates=coordinates,
        lineage=source_manifest.segment_lineage,
        resolved_method=source_resolved_method,
        validate_continuation_lineage=False,
    )
    target = project_training_schedules(
        target_run_spec,
        coordinates=coordinates,
        lineage=CheckpointSegmentLineage(
            parent_transaction_id=source_manifest.transaction_id,
            start_batch=boundary,
            segment_batch_count=continuation.additional_batches or 0,
        ),
        resolved_method=target_resolved_method,
    )
    exemptions = {
        exemption.schedule_id: exemption
        for exemption in continuation.schedule_discontinuity_exemptions
    }
    used_exemptions: set[str] = set()
    failures: list[str] = []
    comparisons: list[dict[str, Any]] = []
    for schedule_id in sorted(set(source.schedules) | set(target.schedules)):
        source_schedule = source.schedules.get(schedule_id)
        target_schedule = target.schedules.get(schedule_id)
        record = _schedule_comparison_record(
            schedule_id,
            coordinates,
            source_schedule,
            target_schedule,
        )
        comparisons.append(record)
        if not record["mismatch"]:
            continue
        exemption = exemptions.get(schedule_id)
        if exemption is not None:
            if _exemption_matches_record(
                exemption.expected_source_state, record["source_values"]
            ) and _exemption_matches_record(
                exemption.expected_target_state, record["target_values"]
            ):
                record["exempted"] = True
                used_exemptions.add(schedule_id)
                continue
            failures.append(
                f"schedule {schedule_id!r} discontinuity exemption expected values do not "
                f"match reality; source={record['source_values']!r} "
                f"target={record['target_values']!r}"
            )
            used_exemptions.add(schedule_id)
            continue
        failures.append(
            f"schedule {schedule_id!r} diverges at coordinates {coordinates!r}; "
            f"source={record['source_values']!r} target={record['target_values']!r}; "
            f"values_by_coordinate={record['values_by_coordinate']!r}; "
            f"source_origin={record['source_origin']!r} "
            f"target_origin={record['target_origin']!r}"
        )
    unused = sorted(set(exemptions) - used_exemptions)
    if unused:
        failures.append(f"unused schedule discontinuity exemptions: {unused!r}")
    return failures, {
        "boundary_side": "pre_update",
        "dtype": "float64",
        "relative_tolerance": SCHEDULE_REL_TOL,
        "coordinates": list(coordinates),
        "source_projection": source.model_dump(mode="json"),
        "target_projection": target.model_dump(mode="json"),
        "comparisons": comparisons,
        "used_exemptions": sorted(used_exemptions),
    }


def _schedule_comparison_record(
    schedule_id: str,
    coordinates: Sequence[int],
    source: GovernedScheduleProjection | None,
    target: GovernedScheduleProjection | None,
) -> dict[str, Any]:
    source_values = _projection_values(source, coordinates)
    target_values = _projection_values(target, coordinates)
    source_origin = None if source is None else source.origin.model_dump(mode="json")
    target_origin = None if target is None else target.origin.model_dump(mode="json")
    values_match = (
        source_values is not None
        and target_values is not None
        and all(
            math.isclose(source_value, target_value, rel_tol=SCHEDULE_REL_TOL, abs_tol=0.0)
            for source_value, target_value in zip(source_values, target_values, strict=True)
        )
    )
    return {
        "schedule_id": schedule_id,
        "source_origin": source_origin,
        "target_origin": target_origin,
        "source_values": source_values,
        "target_values": target_values,
        "values_by_coordinate": [
            {
                "coordinate": coordinate,
                "source": None if source_values is None else source_values[index],
                "target": None if target_values is None else target_values[index],
            }
            for index, coordinate in enumerate(coordinates)
        ],
        "mismatch": not values_match or source_origin != target_origin,
        "exempted": False,
    }


def _projection_values(
    projection: GovernedScheduleProjection | None,
    coordinates: Sequence[int],
) -> list[float] | None:
    if projection is None:
        return None
    values = {
        sample.coordinate: _normalize_schedule_value(sample.value) for sample in projection.samples
    }
    if set(values) != set(coordinates):
        raise ValueError(
            f"schedule projection coordinate inventory differs; "
            f"expected={list(coordinates)!r} observed={sorted(values)!r}"
        )
    return [values[coordinate] for coordinate in coordinates]


def _exemption_matches_record(
    expected: ScheduleStateWindowSpec,
    observed: list[float] | None,
) -> bool:
    if observed is None:
        return False
    expected_values = [
        _normalize_schedule_value(expected.boundary),
        _normalize_schedule_value(expected.first_update),
        _normalize_schedule_value(expected.second_update),
    ]
    return all(
        math.isclose(expected_value, observed_value, rel_tol=SCHEDULE_REL_TOL, abs_tol=0.0)
        for expected_value, observed_value in zip(expected_values, observed, strict=True)
    )


def _normalize_schedule_value(value: Any) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"schedule value must be finite; observed={normalized!r}")
    return normalized


def _with_injected_count(value: Any, count: int) -> Any:
    import jax.numpy as jnp

    if _is_injected_hyperparams_state(value):
        hyperparams_states = dict(value.hyperparams_states)
        patched_states = {
            key: state._replace(count=jnp.asarray(count, dtype=jnp.int32))
            for key, state in hyperparams_states.items()
            if hasattr(state, "_replace") and hasattr(state, "count")
        }
        hyperparams_states.update(patched_states)
        return value._replace(
            count=jnp.asarray(count, dtype=jnp.int32),
            hyperparams_states=hyperparams_states,
        )
    if isinstance(value, tuple):
        return type(value)(_with_injected_count(item, count) for item in value)
    if isinstance(value, list):
        return [_with_injected_count(item, count) for item in value]
    if isinstance(value, dict):
        return {key: _with_injected_count(item, count) for key, item in value.items()}
    return value


def _scheduled_learning_rate(value: Any) -> float:
    import jax.tree as jt

    leaves = jt.leaves(value, is_leaf=_is_injected_hyperparams_state)
    for leaf in leaves:
        if _is_injected_hyperparams_state(leaf):
            return float(leaf.hyperparams["learning_rate"])
    raise ValueError("scheduled optimizer state not found")


def _is_injected_hyperparams_state(value: Any) -> bool:
    fields = getattr(value, "_fields", ())
    return {"hyperparams", "inner_state"}.issubset(set(fields))


class _Missing:
    pass


_MISSING = _Missing()


def _path(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if current is None:
            return _MISSING
        if isinstance(current, Mapping):
            if key not in current:
                return _MISSING
            current = current[key]
            continue
        current = getattr(current, key, _MISSING)
        if current is _MISSING:
            return _MISSING
    return current


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not _MISSING:
            return value
    return _MISSING
