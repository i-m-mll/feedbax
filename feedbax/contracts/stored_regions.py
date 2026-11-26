"""The declared types of authored step bodies and registry-owned values."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any, Annotated, TYPE_CHECKING

from pydantic import BaseModel, Field, JsonValue, TypeAdapter, ValidationError
from pydantic_core import to_jsonable_python

from feedbax.contracts.analysis_document import AnalysisPayload, OperationElement
from feedbax.contracts.base import StrictModel
from feedbax.contracts.objective_document import StoredObjectiveDocument
from feedbax.contracts.project_inputs import DocumentReference
from feedbax.contracts.report_document import ReportDocument
from feedbax.contracts.request_specs import (
    AnalysisRequestSpec, DocumentTrainingRequestSpec, EvaluationRequestSpec, ReportRequestSpec,
)
from feedbax.contracts.rollout import ROLLOUT_TYPE, EveryCondition, RolloutSettings, SampleTrials, SelectedConditions
from feedbax.contracts.stored_model import StoredLeafComponent, StoredModelDocument
from feedbax.contracts.synthesis import SynthesisStepPayload
from feedbax.contracts.task_document import PythonTaskSelection, StoredTaskDocument
from feedbax.contracts.training_document import TrainingDocument


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
    "analyze": AnalysisPayload[JsonValue],
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

    def validate(self, values: Mapping) -> BaseModel:
        assert self.model is not None
        if self.component:
            from feedbax.contracts.portable import validate_authoring

            return validate_authoring(self.model, values)
        return self.model.model_validate(values, extra="forbid")


if TYPE_CHECKING:
    from feedbax.analysis.operations import OperationRegistry
    from feedbax.component_registry import ComponentRegistry
    from feedbax.synthesis.methods import SynthesisMethodRegistry
    from feedbax.tasks.registry import PythonTaskRegistry
    from feedbax.training.methods import TrainingMethodRegistry


@dataclass(frozen=True)
class StoredRegionRegistries:
    """The installed schemas supplied by the same registries that make the plan."""

    training_methods: TrainingMethodRegistry | None = None
    synthesis_methods: SynthesisMethodRegistry | None = None
    operations: OperationRegistry | None = None
    python_tasks: PythonTaskRegistry | None = None
    components: ComponentRegistry | None = None

    def _registry(self, family: str):
        supplied = getattr(self, family)
        if supplied is not None:
            return supplied
        from feedbax.application.composition import load_application

        return getattr(load_application().bundle, family)

    def region(self, owner: BaseModel, field: str) -> StoredRegion | None:
        """Find the declaration owning an open value field, if there is one."""
        try:
            if isinstance(owner, (StoredTrainInline, TrainingDocument)) and field == "settings":
                method = getattr(owner, "method", None)
                return StoredRegion(self._registry("training_methods").resolve(method).settings_model
                                    if isinstance(method, str) else None)
            if isinstance(owner, SynthesisStepPayload) and field == "settings":
                method = getattr(owner, "method", None)
                return StoredRegion(self._registry("synthesis_methods").resolve(method).declaration.settings_model
                                    if isinstance(method, str) else None)
            if isinstance(owner, OperationElement) and field == "params":
                operation = getattr(owner, "operation", None)
                return StoredRegion(self._registry("operations").resolve(operation).value.parameter_model
                                    if isinstance(operation, str) else None)
            if isinstance(owner, PythonTaskSelection) and field == "params":
                task = self._registry("python_tasks").resolve(owner.type)
                return StoredRegion(task.parameters if task is not None else None)
            if isinstance(owner, StoredLeafComponent) and field == "fields":
                declaration = self._registry("components").get(owner.type)
                if declaration is not None and owner.revision in (None, declaration.revision):
                    from feedbax.compiler.model_document import model_tau_seconds, split_model_field_context

                    _, context = split_model_field_context(declaration, owner.fields)
                    if declaration.name == "feedbax.VanillaRNN" and "tau" in context:
                        model_tau_seconds(context["tau"], 1.0)
                    return StoredRegion(declaration.params.model, component=True, contextual=frozenset(context))
                return StoredRegion(None)
            if isinstance(owner, StoredEvaluateInline) and field == "params" and (
                getattr(owner, "evaluation_type", None) == ROLLOUT_TYPE
            ):
                return StoredRegion(StoredRolloutSettings)
        except KeyError:
            return StoredRegion(None)
        return None

    def feedbax_models(self) -> Iterable[type[BaseModel]]:
        """Enumerate the declared defaults that belong in Feedbax's manifest."""
        for family in ("training_methods", "synthesis_methods", "operations", "python_tasks", "components"):
            registry = self._registry(family)
            for key in registry.names() if family == "components" else registry.keys():
                record = registry.resolve(key) if family == "operations" else registry.declaration_record(key)
                if record.source.distribution != "feedbax":
                    continue
                value = record.value
                match family:
                    case "training_methods":
                        yield value.settings_model
                    case "synthesis_methods":
                        yield value.declaration.settings_model
                    case "operations":
                        yield value.parameter_model
                    case "python_tasks":
                        yield value.parameters
                    case "components":
                        yield value.params.model


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
) -> tuple[BaseModel, dict[str, Any]]:
    """Validate local values without completing a step fragment or resolving deferred authored values."""
    model = region.model
    assert model is not None
    try:
        typed = region.validate(raw)
    except ValidationError as error:
        if not inline and not _deferred(raw) and any(item["type"] != "missing" for item in error.errors()):
            raise
    else:
        return typed, typed.model_dump(mode="json", by_alias=True)

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
    return model.model_construct(_fields_set=set(values), **values), encoded

