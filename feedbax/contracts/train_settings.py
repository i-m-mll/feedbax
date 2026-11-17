"""Resolve and normalize the settings of one installed Train method."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel

    from feedbax.contracts.request_specs import DocumentTrainingRequestSpec
    from feedbax.training.methods import RegisteredTrainingMethod, TrainingMethodRegistry


def normalize_train_method(
    request: DocumentTrainingRequestSpec, registry: TrainingMethodRegistry,
) -> tuple[DocumentTrainingRequestSpec, RegisteredTrainingMethod, BaseModel]:
    try:
        method = registry.resolve(request.method)
    except KeyError as error:
        raise ValueError(f"/method: {error.args[0]}") from error
    settings = method.settings_model.model_validate(request.settings, extra="forbid")
    return request.model_copy(update={
        "method": method.key,
        "method_revision": method.revision,
        "settings": settings.model_dump(mode="json"),
    }), method, settings
