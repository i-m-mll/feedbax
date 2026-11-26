"""The declared types of authored step bodies and registry-owned values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from typing import Annotated, Any, Protocol

from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from pydantic_core import to_jsonable_python

from feedbax.contracts.analysis_document import StoredAnalysisPayload
from feedbax.contracts.base import StrictModel
from feedbax.contracts.objective_document import StoredObjectiveDocument
from feedbax.contracts.project_inputs import DocumentReference
from feedbax.contracts.report_document import ReportDocument
from feedbax.contracts.request_specs import (
    AnalysisRequestSpec, DocumentTrainingRequestSpec, EvaluationRequestSpec, ReportRequestSpec,
)
from feedbax.contracts.rollout import ROLLOUT_TYPE, EveryCondition, RolloutSettings, SampleTrials, SelectedConditions
from feedbax.contracts.stored_model import StoredModelDocument
from feedbax.contracts.synthesis import SynthesisStepPayload
from feedbax.contracts.task_document import PythonTaskSelection, StoredTaskDocument


class _TrialSlots(StrictModel):
    model: DocumentReference | StoredModelDocument
    task: DocumentReference | StoredTaskDocument | None
    objective: DocumentReference | StoredObjectiveDocument
    input_bindings: dict[str, str | None] = Field(default_factory=dict)
    python_task: DocumentReference | PythonTaskSelection | None = None


class StoredTrainInline(DocumentTrainingRequestSpec):
    trial_documents: _TrialSlots


class _EvaluateSlots(StrictModel):
    model: DocumentReference | StoredModelDocument | None = None
    task: DocumentReference | StoredTaskDocument | PythonTaskSelection | None = None


class StoredEvaluateInline(EvaluationRequestSpec):
    trial_documents: _EvaluateSlots | None = None


class StoredRolloutSettings(RolloutSettings):
    trial_set: Annotated[EveryCondition | SelectedConditions | SampleTrials, Field(discriminator="kind")] | None = None


_STEP_MODELS = {
    "train": StoredTrainInline,
    "synthesize": SynthesisStepPayload,
    "evaluate": StoredEvaluateInline,
    "analyze": StoredAnalysisPayload,
    "report": ReportDocument,
}


def step_inline_model(kind: str, raw: Mapping) -> type[BaseModel]:
    if kind == "analyze" and "analysis_type" in raw:
        return AnalysisRequestSpec
    if kind == "report" and "document" in raw:
        return ReportRequestSpec
    return _STEP_MODELS[kind]


@dataclass(frozen=True)
class StoredRegion:
    """A registry-owned values object, whose declaration may be unavailable."""

    model: type[BaseModel] | None
    component: bool = False
    contextual: frozenset[str] = frozenset()
    lowered_expressions: Mapping[str, Any] = field(default_factory=dict)
    authored_expressions: Mapping[str, Any] = field(default_factory=dict)

    def authored_values(self, values: dict) -> dict:
        if not self.authored_expressions:
            return values
        return {key: self.authored_expressions.get(key, value) for key, value in values.items()}

    def validate(self, values: Mapping) -> BaseModel:
        assert self.model is not None
        if self.component:
            from feedbax.contracts.portable import validate_authoring

            return validate_authoring(self.model, values)
        return self.model.model_validate(values, extra="forbid")


class StoredRegionResolver(Protocol):
    """Resolve open authored fields against declarations supplied by the caller."""

    def region(self, owner: BaseModel, field: str) -> StoredRegion | None: ...


def declared_stored_region(owner: BaseModel, field: str) -> StoredRegion | None:
    if isinstance(owner, StoredEvaluateInline) and field == "params" and (
        getattr(owner, "evaluation_type", None) == ROLLOUT_TYPE
    ):
        return StoredRegion(StoredRolloutSettings)
    return None


def inline_models() -> tuple[type[BaseModel], ...]:
    return (*_STEP_MODELS.values(), StoredRolloutSettings, AnalysisRequestSpec, ReportRequestSpec)


def _deferred(value: Any) -> bool:
    if isinstance(value, dict):
        return (
            value.get("kind") == "unresolved"
            or isinstance(value.get("document"), str) and value.keys() <= {"document", "pointer", "unresolved"}
            or any(_deferred(child) for child in value.values())
        )
    return isinstance(value, list) and any(_deferred(child) for child in value)


@cache
def _field_adapter(model: type[BaseModel], name: str) -> TypeAdapter:
    return TypeAdapter(model.model_fields[name].rebuild_annotation())


def typed_region(
    region: StoredRegion, raw: dict[str, Any], *, inline: bool = False,
) -> tuple[BaseModel, dict[str, Any], bool]:
    """Validate local values without completing a step fragment or resolving deferred authored values."""
    model = region.model
    assert model is not None
    if region.lowered_expressions:
        raw = {key: region.lowered_expressions.get(key, value) for key, value in raw.items()}
    try:
        typed = region.validate(raw)
    except ValidationError as error:
        if not inline and not _deferred(raw) and any(item["type"] != "missing" for item in error.errors()):
            raise
    else:
        return typed, typed.model_dump(mode="json", by_alias=True), True

    # Missing fields may be supplied by a delta or a sweep, and a document pointer's value is only available in Plan.
    names = {
        alias: name for name, field in model.model_fields.items()
        for alias in (name, field.alias, field.serialization_alias) if alias is not None
    }
    values, encoded = {}, {}
    for key, value in raw.items():
        name = names.get(key)
        if name is None:
            if inline:
                encoded[key] = value
                continue
            raise ValueError(f"{model.__name__} has no field {key!r}.")
        try:
            parsed = _field_adapter(model, name).validate_python(value)
        except ValidationError:
            if not _deferred(value):
                raise
            parsed = value
        values[name] = parsed
        field = model.model_fields[name]
        encoded[field.serialization_alias or field.alias or name] = to_jsonable_python(parsed, by_alias=True)
    return model.model_construct(_fields_set=set(values), **values), encoded, False
