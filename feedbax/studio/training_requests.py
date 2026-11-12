"""Lower authored Studio train runs to executable requests."""

from __future__ import annotations

import json
import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

from feedbax.contracts.graph import GraphSpec, StudioTaskBindingSpec, StudioWorkspaceSpec
from feedbax.contracts.spec_storage import canonical_training_run_spec_sha256
from feedbax.contracts.studio_training import (
    STUDIO_GRAPH_METHOD,
    STUDIO_GRAPH_METHOD_PAYLOAD_ID,
    STUDIO_GRAPH_METHOD_PAYLOAD_VERSION,
    STUDIO_INDEPENDENT_KEY_SCHEDULE,
    StudioGraphMethodPayload,
)
from feedbax.contracts.training import (
    MethodPayloadEnvelope,
    ObjectiveSlotSpec,
    TaskSpec,
    TrainingConfig,
    TrainingRunSpec,
    TrainingSpec,
    validate_training_run_spec_semantics,
)
from feedbax.integrations.provider import validate_task_spec, validate_training_spec
from feedbax.studio.protocol import infer_task_n_steps
from feedbax.studio.schema import validate_task_binding_schema
from feedbax.studio.sweep_matrix import materialize_sweep_matrix, matrix_spec_from_selection

if TYPE_CHECKING:
    from feedbax.plugins.application import ApplicationRegistryBundle


ModelT = TypeVar("ModelT", bound=BaseModel)


class StudioTrainingLoweringError(ValueError):
    """An authored training input could not be lowered."""

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


@dataclass(frozen=True)
class StudioTrainingRun:
    """One executable request and its attribution to a Studio row."""

    run_key: str
    request_id: str
    spec: TrainingRunSpec
    stage_id: str
    scenario_id: str
    label: str
    coordinates: Mapping[str, Any]


def studio_training_run_key(stage_id: str, row_id: str | None = None) -> str:
    """Return the logical row address within the project."""
    parts = [stage_id] if row_id is None else [stage_id, row_id]
    return json.dumps(parts, ensure_ascii=False, separators=(",", ":"))


def lower_studio_training_runs(
    workspace: StudioWorkspaceSpec,
    graph: GraphSpec,
    *,
    stage_id: str,
    registry_bundle: ApplicationRegistryBundle,
    repo_root: Path | None = None,
    run_keys: Collection[str] | None = None,
) -> tuple[StudioTrainingRun, ...]:
    """Compile an ordinary scenario or selected rows of a sweep without staging."""
    stage = next((candidate for candidate in workspace.stages if candidate.id == stage_id), None)
    if stage is None or stage.kind != "train":
        raise StudioTrainingLoweringError("/stage_id", f"No train step with id {stage_id!r}")
    scenario = workspace.scenarios.get(stage.scenario_id) if stage.scenario_id else None
    if scenario is None:
        raise StudioTrainingLoweringError("/scenario_id", "Train step requires a scenario")

    requested = None if run_keys is None else set(run_keys)
    try:
        matrix = matrix_spec_from_selection(stage.selection_spec)
    except ValueError as exc:
        raise StudioTrainingLoweringError("/selection_spec", str(exc)) from exc

    graph_payload = graph.model_dump(mode="json", exclude_none=True)
    training = scenario.training_spec
    task = scenario.task_spec
    binding = (
        scenario.task_binding_spec.model_dump(mode="json", exclude_none=True)
        if scenario.task_binding_spec is not None
        else None
    )
    rows: list[StudioTrainingRun] = []

    def append(
        run_key: str,
        spec: TrainingRunSpec,
        label: str,
        coordinates: Mapping[str, Any],
    ) -> None:
        rows.append(
            StudioTrainingRun(
                run_key=run_key,
                request_id=canonical_training_run_spec_sha256(spec),
                spec=spec,
                stage_id=stage.id,
                scenario_id=scenario.id,
                label=label,
                coordinates=coordinates,
            )
        )

    if matrix is None:
        run_key = studio_training_run_key(stage.id)
        if requested is None or run_key in requested:
            spec = _lower_scenario(graph_payload, training, task, binding, registry_bundle)
            append(run_key, spec, stage.label, {})
    else:
        try:
            materialized = materialize_sweep_matrix(
                matrix,
                graph_spec=graph_payload,
                training_spec=training or {},
                task_spec=task or {},
                task_binding_spec=binding,
                default_name=stage.label,
                method_registry=registry_bundle.training_programs,
                row_lowerer_registry=registry_bundle.row_lowerers,
                repo_root=repo_root,
            )
        except ValueError as exc:
            raise StudioTrainingLoweringError("/selection_spec/matrix", str(exc)) from exc
        for row in materialized.rows:
            run_key = studio_training_run_key(stage.id, row.row_id)
            if requested is not None and run_key not in requested:
                continue
            if row.spec is None:
                spec = _lower_scenario(
                    row.payload["graph_spec"],
                    row.payload["training_spec"],
                    row.payload["task_spec"],
                    row.payload.get("task_binding_spec"),
                    registry_bundle,
                )
            else:
                spec = row.spec
                try:
                    validate_training_run_spec_semantics(spec, registry_bundle.training_programs)
                except ValueError as exc:
                    raise StudioTrainingLoweringError("/selection_spec/matrix", str(exc)) from exc
            coordinate = row.coordinate
            append(
                run_key,
                spec,
                stage.label if coordinate is None else coordinate.label or stage.label,
                {} if coordinate is None else coordinate.values,
            )

    return tuple(rows)


def _lower_scenario(
    graph: dict[str, Any],
    training: dict[str, Any] | None,
    task: dict[str, Any] | None,
    binding: dict[str, Any] | None,
    registry_bundle: ApplicationRegistryBundle,
) -> TrainingRunSpec:
    if training is None:
        raise StudioTrainingLoweringError("/training_spec", "Scenario requires training settings")
    if task is None:
        raise StudioTrainingLoweringError("/task_spec", "Scenario requires a task")
    if binding is None:
        raise StudioTrainingLoweringError("/task_binding_spec", "Scenario requires task bindings")

    graph_spec = _validate(GraphSpec, graph, "/graph")
    training_spec = _validate(TrainingSpec, training, "/training_spec")
    task_spec = _validate(TaskSpec, task, "/task_spec")
    task_binding_spec = _validate(StudioTaskBindingSpec, binding, "/task_binding_spec")

    training_validation = validate_training_spec(
        training_spec,
        graph_spec=graph_spec,
        task_spec=task_spec,
        component_registry=registry_bundle.components,
    )
    task_validation = validate_task_spec(task_spec)
    for path, validation in (
        ("/training_spec", training_validation),
        ("/task_spec", task_validation),
    ):
        if validation.errors:
            error = validation.errors[0]
            raise StudioTrainingLoweringError(
                path + (error.location or {}).get("path", ""), error.message
            )
    binding_errors = validate_task_binding_schema(
        task_binding_spec,
        graph_spec,
        "/task_binding_spec",
        component_registry=registry_bundle.components,
    )
    if error := next((issue for issue in binding_errors if issue.severity == "error"), None):
        raise StudioTrainingLoweringError(
            (error.location or {}).get("path", "/task_binding_spec"), error.message
        )

    n_steps = infer_task_n_steps(task_spec)
    if n_steps is None:
        raise StudioTrainingLoweringError(
            "/task_spec/params/n_steps", "Task requires a rollout length"
        )
    settings = studio_graph_training_settings(training_spec, training, n_steps)
    method = registry_bundle.training_programs.resolve_execution(STUDIO_GRAPH_METHOD, settings.method_payload)
    return TrainingRunSpec(
        graph={
            "inline": graph_spec.model_dump(mode="json", exclude_none=True, exclude={"metadata"})
        },
        task=task_spec,
        task_binding_spec=task_binding_spec,
        training_config=settings.config,
        objective=settings.objective,
        method_ref=STUDIO_GRAPH_METHOD,
        method_payload=settings.method_payload,
        worker_execution={
            "method_contract": method.contract,
            "effective_phase": method.effective_phase,
        },
        checkpoint_progress={
            "checkpoint_interval": settings.config.snapshot_interval,
            "progress_interval": 1,
        },
    )


@dataclass(frozen=True)
class StudioGraphTrainingSettings:
    config: TrainingConfig
    objective: ObjectiveSlotSpec
    method_payload: MethodPayloadEnvelope


def studio_graph_training_settings(
    training_spec: TrainingSpec, training: Mapping[str, Any], n_steps: int | None,
) -> StudioGraphTrainingSettings:
    """Translate the saved Studio settings into the shared authored Train fields."""
    checkpoint_interval = training_spec.checkpoint_interval
    if checkpoint_interval is not None and checkpoint_interval <= 0:
        raise StudioTrainingLoweringError(
            "/training_spec/checkpoint_interval", "Checkpoint interval must be positive"
        )
    if checkpoint_interval is None:
        checkpoint_interval = 100

    optimizer = training_spec.optimizer
    if optimizer.lr_schedule is not None:
        raise StudioTrainingLoweringError(
            "/training_spec/optimizer/lr_schedule", "Studio graph training requires a constant rate"
        )
    rate = optimizer.params.get("learning_rate")
    if type(rate) not in (float, int) or not math.isfinite(rate):
        raise StudioTrainingLoweringError(
            "/training_spec/optimizer/params/learning_rate", "Learning rate must be a finite number"
        )
    payload_fields: dict[str, Any] = {
        "optimizer": optimizer,
        "target_total": training_spec.n_batches,
        "key_schedule": STUDIO_INDEPENDENT_KEY_SCHEDULE,
    }
    if "seed" in training:
        payload_fields["seed"] = training["seed"]
    payload = _validate(StudioGraphMethodPayload, payload_fields, "/training_spec")
    envelope = MethodPayloadEnvelope(
        schema_id=STUDIO_GRAPH_METHOD_PAYLOAD_ID,
        schema_version=STUDIO_GRAPH_METHOD_PAYLOAD_VERSION,
        payload=payload.model_dump(mode="json", exclude_defaults=True),
    )
    horizon = {"n_reach_steps": n_steps} if n_steps is not None else {}
    return StudioGraphTrainingSettings(
        TrainingConfig(
            n_batches=training_spec.n_batches,
            batch_size=training_spec.batch_size,
            learning_rate=rate,
            snapshot_interval=checkpoint_interval,
            **horizon,
        ),
        ObjectiveSlotSpec(loss=training_spec.loss),
        envelope,
    )


def _validate(model: type[ModelT], value: Any, path: str) -> ModelT:
    try:
        return model.model_validate(value)
    except ValidationError as exc:
        error = exc.errors()[0]
        parts = (str(part).replace("~", "~0").replace("/", "~1") for part in error["loc"])
        suffix = "".join(f"/{part}" for part in parts)
        raise StudioTrainingLoweringError(path + suffix, error["msg"]) from exc
