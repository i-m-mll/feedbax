"""Training settings read by the saved Studio workspace converter."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from feedbax.contracts.studio_training import (
    STUDIO_GRAPH_METHOD_PAYLOAD_ID,
    STUDIO_GRAPH_METHOD_PAYLOAD_VERSION,
    STUDIO_INDEPENDENT_KEY_SCHEDULE,
    StudioGraphMethodPayload,
)
from feedbax.contracts.training import (
    MethodPayloadEnvelope,
    ObjectiveSlotSpec,
    TrainingConfig,
    TrainingSpec,
)


ModelT = TypeVar("ModelT", bound=BaseModel)


class StudioTrainingLoweringError(ValueError):
    """An authored training input could not be lowered."""

    def __init__(self, path: str, message: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


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
