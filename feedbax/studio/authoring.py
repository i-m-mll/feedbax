"""The authored project schema and the controls supported for direct Train steps."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, JsonValue, ValidationError

from feedbax.contracts.graph import GraphSpec, StudioTaskBindingSpec
from feedbax.contracts.migrations import default_spec_registry
from feedbax.contracts.project import DocumentReference, InlineBase, Project
from feedbax.contracts.project_resolution import ResolvedStep, StepTemplate, compose_step
from feedbax.contracts.project_sources import ValueSource, pointer
from feedbax.contracts.request_identity import REQUEST_KINDS
from feedbax.contracts.studio_authoring import (
    StudioAuthoringChange,
    StudioAuthoringChoice,
    StudioAuthoringField,
    StudioAuthoringMethod,
    StudioAuthoringNumber,
    StudioAuthoringOption,
    StudioAuthoringSource,
    StudioAuthoringSurface,
)
from feedbax.contracts.studio_project import SetProjectValue
from feedbax.contracts.studio_training import (
    STUDIO_GRAPH_METHOD,
    STUDIO_INDEPENDENT_KEY_SCHEDULE,
    StudioGraphMethodPayload,
)
from feedbax.contracts.training import (
    LossTermSpec,
    MethodPayloadEnvelope,
    MethodRefSpec,
    ObjectiveSlotSpec,
    OptimizerSpec,
    TaskSpec,
    TimeAggregationSpec,
    TrainingConfig,
    TrainingProgramRegistry,
)
from feedbax.objectives.spec import TargetStateLossSpec
from feedbax.plugins.application import TRAINING_PROGRAMS
from feedbax.plugins.bootstrap import BootstrapState
from feedbax.studio.protocol import infer_task_n_steps
from feedbax.studio.schema import retained_observable_selector
from feedbax.training.optimizers import optimizer_types


_TRAIN = REQUEST_KINDS["train"].spec_type
_DOCUMENT_FIELDS: dict[str, tuple[type[BaseModel], str]] = {
    "graph": (GraphSpec, "Model"),
    "task": (TaskSpec, "Task"),
}
_TARGET_STATE = TargetStateLossSpec.model_fields["type"].default
_GRAPH_BATCH_SIZE = 1
_DEFAULT_OPTIMIZER = "adam"
type _Path = tuple[str, ...]
type _Scalar = str | int | float


def _document_schema(model: type[BaseModel]) -> str:
    return default_spec_registry.resolve(model.__name__).identity


def authoring_surface(state: BootstrapState) -> StudioAuthoringSurface:
    """Describe authored shapes and method declarations without consulting execution facets."""
    registry = state.bundle.training_programs
    plugins = {
        key: origin.plugin_id
        for origin in state.provenance
        for key in origin.registered_keys.get(TRAINING_PROGRAMS.family, ())
    }
    methods = [
        StudioAuthoringMethod(
            key=key,
            schema_id=method.payload_schema_id,
            schema_version=method.payload_schema_version,
            owner=method.owner,
            package=method.package,
            plugin=plugins.get(key),
            payload_schema=method.payload_model.model_json_schema(),
        )
        for key in registry.available_keys()
        for method in (registry.resolve(key, path="/method_ref"),)
    ]
    return StudioAuthoringSurface(
        project_grammar=Project.model_json_schema(by_alias=True),
        step_payloads={"train": _TRAIN.model_json_schema()},
        document_schemas={
            _document_schema(model): model.model_json_schema()
            for model, _ in _DOCUMENT_FIELDS.values()
        },
        training_methods=methods,
    )


@dataclass(frozen=True)
class _Field:
    view: StudioAuthoringField
    write: Callable[[_Scalar], list[SetProjectValue]]
    documents: tuple[str, ...]


def _read[Model: BaseModel](model: type[Model], value: Any) -> Model | None:
    try:
        return model.model_validate(value)
    except ValidationError:
        return None


def _options(values: dict[str, str]) -> StudioAuthoringChoice:
    return StudioAuthoringChoice(
        kind="choice",
        options=[
            StudioAuthoringOption(value=value, label=label) for value, label in values.items()
        ],
    )


def _get(payload: JsonValue, path: _Path) -> JsonValue:
    for name in path:
        if not isinstance(payload, dict):
            return None
        payload = payload.get(name)
    return payload


def _source(source: ValueSource | None) -> StudioAuthoringSource | None:
    if source is None or source.kind not in ("document", "step"):
        return None
    return StudioAuthoringSource(kind=source.kind, name=source.name, path=source.path)


def _loss_choices(
    graph: GraphSpec | None, binding: StudioTaskBindingSpec | None
) -> dict[str, tuple[str, str]]:
    if graph is None or binding is None:
        return {}
    choices = {}
    for observable in graph.retained_observables or ():
        selector = retained_observable_selector(observable)
        schema = observable.value_schema
        if not selector.startswith("path:") or schema is None or schema.shape is None:
            continue
        for target in binding.exposed_data:
            shape = target.expected_shape
            if shape is None or shape[:1] != ["time"] or shape[1:] != schema.shape:
                continue
            if schema.units and target.units and schema.units != target.units:
                continue
            if schema.frame and target.frame and schema.frame != target.frame:
                continue
            key = json.dumps([selector, f"task_data:{target.id}"])
            choices[key] = (observable.label or schema.label, target.label)
    return choices


class _TrainFields:
    def __init__(self, project: Project, step_id: str, registry: TrainingProgramRegistry) -> None:
        step = project.steps.get(step_id)
        if step is None or step.kind != "train":
            raise ValueError(f"Unknown Train step {step_id!r}")
        if (
            not isinstance(step.base, InlineBase)
            or step.sweep is not None
            or step.deltas
            or step.inputs
            or project.sweep is not None
        ):
            raise ValueError("These Train controls require a direct Train step without a sweep")
        self.project = project
        self.payload = step.base.inline
        self.registry = registry
        template = compose_step(project, step_id, track_sources=True)
        resolved = (
            template.resolve(include_sources=True) if isinstance(template, StepTemplate) else None
        )
        self.sources = (resolved.sources or {}) if isinstance(resolved, ResolvedStep) else {}
        self.fields: dict[str, _Field] = {}

    def set(self, path: _Path, value: JsonValue) -> SetProjectValue:
        parent = _get(self.payload, path[:-1])
        exists = isinstance(parent, dict) and path[-1] in parent
        return SetProjectValue(
            op="replace" if exists else "add",
            path=pointer(("base", "inline", *path)),
            value=value,
        )

    def field(
        self,
        path: _Path,
        label: str,
        value: _Scalar | None,
        control: StudioAuthoringChoice | StudioAuthoringNumber | None,
        write: Callable[[_Scalar], list[SetProjectValue]],
        *,
        source: ValueSource | None = None,
        documents: tuple[str, ...] = (),
    ) -> None:
        field_id = pointer(path)
        self.fields[field_id] = _Field(
            StudioAuthoringField(
                id=field_id,
                label=label,
                value=value,
                source=_source(source or self.sources.get(field_id)),
                control=control,
            ),
            write,
            documents,
        )

    def named_documents(self, field: str) -> tuple[str, ...]:
        reference = _read(DocumentReference, self.payload.get(field))
        return (reference.document,) if reference else ()

    def document[Model: BaseModel](self, name: str, model: type[Model], label: str) -> Model | None:
        schema = _document_schema(model)
        documents = {
            key: parsed
            for key, document in self.project.documents.items()
            if document.schema_id == schema
            if (parsed := _read(model, document.payload)) is not None
        }
        raw = self.payload.get(name)
        ref = _read(DocumentReference, raw)
        path = (name,)
        can_choose = (raw is None or (ref is not None and not ref.pointer)) and bool(documents)
        source = (
            ValueSource("document", ref.document, ref.pointer or "")
            if ref and ref.document in documents
            else None
        )
        self.field(
            path,
            label,
            ref.document if ref else "Inline" if raw is not None else None,
            _options({key: key for key in documents}) if can_choose else None,
            lambda key: [self.set(path, {"document": key})],
            source=source,
            documents=self.named_documents(name),
        )
        if ref is not None and not ref.pointer:
            return documents.get(ref.document)
        return _read(model, raw)

    def objective(self, graph: GraphSpec | None) -> None:
        binding = _read(StudioTaskBindingSpec, self.payload.get("task_binding_spec"))
        choices = _loss_choices(graph, binding)
        objective = _read(ObjectiveSlotSpec, self.payload.get("objective"))
        loss = objective.loss if objective is not None and objective.kind == "loss_term" else None
        key = json.dumps([loss.selector, loss.target_selector]) if loss else None
        supported = (
            loss is not None
            and loss.type in (_TARGET_STATE, "TargetStateLoss")
            and loss.time_agg is not None
            and loss.time_agg.mode == "final"
            and loss.target_value is None
            and not loss.children
            and key in choices
            and isinstance(_get(self.payload, ("objective", "loss")), dict)
        )
        path = ("objective", "loss", "selector")

        def write(value: _Scalar) -> list[SetProjectValue]:
            selector, target_selector = json.loads(str(value))
            if supported:
                return [
                    self.set(path, selector),
                    self.set(("objective", "loss", "target_selector"), target_selector),
                ]
            spec = ObjectiveSlotSpec(
                loss=LossTermSpec(
                    type=_TARGET_STATE,
                    label=choices[str(value)][0],
                    selector=selector,
                    target_selector=target_selector,
                    time_agg=TimeAggregationSpec(mode="final"),
                ),
            )
            return [
                self.set(
                    ("objective",),
                    spec.model_dump(mode="json", exclude_none=True, exclude_unset=True),
                )
            ]

        self.field(
            path,
            "Final-state objective",
            key
            if supported
            else loss.label
            if loss
            else "Authored objective"
            if "objective" in self.payload
            else None,
            _options({key: f"{source} → {target}" for key, (source, target) in choices.items()})
            if choices and (supported or "objective" not in self.payload)
            else None,
            write,
            documents=self.named_documents("graph"),
        )

    def training(self, task: TaskSpec | None) -> None:
        raw_config = self.payload.get("training_config")
        config = None if _read(DocumentReference, raw_config) else _read(TrainingConfig, raw_config)
        n_steps = infer_task_n_steps(task)
        defaults = TrainingConfig(batch_size=_GRAPH_BATCH_SIZE)
        if n_steps is not None:
            defaults.n_reach_steps = n_steps
        method_ref = _read(MethodRefSpec, self.payload.get("method_ref"))
        envelope = _read(MethodPayloadEnvelope, self.payload.get("method_payload"))
        registration = self.registry.resolve(STUDIO_GRAPH_METHOD, path="/method_ref")
        method = None
        if method_ref == STUDIO_GRAPH_METHOD and envelope is not None:
            try:
                method = self.registry.validate_payload(
                    method_ref, envelope, path="/method_payload"
                )
            except ValueError:
                pass
        if not isinstance(method, StudioGraphMethodPayload):
            method = None

        def method_change(_value: _Scalar) -> list[SetProjectValue]:
            settings = config or defaults
            payload = StudioGraphMethodPayload(
                optimizer=OptimizerSpec(
                    type=_DEFAULT_OPTIMIZER, params={"learning_rate": settings.learning_rate}
                ),
                target_total=settings.n_batches,
                key_schedule=STUDIO_INDEPENDENT_KEY_SCHEDULE,
            )
            candidate = MethodPayloadEnvelope(
                schema_id=registration.payload_schema_id,
                schema_version=registration.payload_schema_version,
                payload=payload.model_dump(mode="json", exclude_defaults=True),
            )
            self.registry.validate_payload(STUDIO_GRAPH_METHOD, candidate, path="/method_payload")
            return [
                self.set(("method_ref",), STUDIO_GRAPH_METHOD.model_dump(mode="json")),
                self.set(("method_payload",), candidate.model_dump(mode="json")),
            ]

        can_define = "method_payload" not in self.payload and (
            method_ref == STUDIO_GRAPH_METHOD or "method_ref" not in self.payload
        )
        self.field(
            ("method_ref", "name"),
            "Training method",
            None if can_define else method_ref.key if method_ref else "Authored method",
            _options({registration.method_ref: "Studio graph"})
            if can_define or method is not None
            else None,
            method_change,
        )

        if method is not None:
            optimizer_path = ("method_payload", "payload", "optimizer", "type")
            choices = {name: name for name in optimizer_types()}
            self.field(
                optimizer_path,
                "Optimizer",
                method.optimizer.type,
                _options(choices)
                if method.optimizer.type in choices
                and method.optimizer.params.keys() <= {"learning_rate"}
                else None,
                lambda value: [self.set(optimizer_path, value)],
            )
            seed_path = ("method_payload", "payload", "seed")
            seed_schema = registration.payload_model.model_json_schema()["properties"]["seed"]
            self.field(
                seed_path,
                seed_schema["title"],
                _get(self.payload, seed_path),
                StudioAuthoringNumber(
                    kind="number",
                    min=seed_schema["minimum"],
                    max=seed_schema["exclusiveMaximum"] - 1,
                    step=1,
                ),
                lambda value: [self.set(seed_path, value)],
            )

        rate_path = ("method_payload", "payload", "optimizer", "params")
        raw_params = _get(self.payload, rate_path)
        params_ref = _read(DocumentReference, raw_params)
        if not (can_define or method is not None) or (
            config is None and (n_steps is None or "training_config" in self.payload or params_ref)
        ):
            return

        initial_config = defaults.model_dump(
            mode="json",
            include={"batch_size", "n_reach_steps"},
        )
        if method is not None:
            initial_config["n_batches"] = method.target_total
            rate = method.optimizer.params.get("learning_rate")
            if type(rate) in (int, float) and 0 < rate <= sys.float_info.max:
                initial_config["learning_rate"] = rate

        def settings_change(
            name: str, value: _Scalar, mirror: _Path | None
        ) -> list[SetProjectValue]:
            path = ("training_config",)
            patches = (
                [self.set((*path, name), value)]
                if config is not None
                else [self.set(path, {**initial_config, name: value})]
            )
            if method is not None and mirror is not None:
                if name == "learning_rate" and raw_params is None:
                    patches.append(self.set(rate_path, {"learning_rate": value}))
                else:
                    patches.append(self.set(("method_payload", "payload", *mirror), value))
            return patches

        settings = TrainingConfig.model_json_schema()["properties"]
        for name, mirror in {
            "n_batches": ("target_total",),
            "learning_rate": ("optimizer", "params", "learning_rate"),
            "snapshot_interval": None,
        }.items():
            schema = settings[name]
            path = ("training_config", name)
            self.field(
                path,
                schema["title"],
                _get(self.payload, path),
                StudioAuthoringNumber(
                    kind="number",
                    min=1 if schema["type"] == "integer" else 0,
                    step=1 if schema["type"] == "integer" else None,
                )
                if name != "learning_rate" or params_ref is None
                else None,
                lambda value, name=name, mirror=mirror: settings_change(name, value, mirror),
                documents=self.named_documents("task") if config is None else (),
            )


def project_train_fields(
    project: Project,
    step_id: str,
    registry: TrainingProgramRegistry,
    change: StudioAuthoringChange | None,
) -> tuple[list[StudioAuthoringField], list[SetProjectValue], list[str]]:
    """Project the supported fields and plan one atomic, step-local edit."""
    definition = _TrainFields(project, step_id, registry)
    graph = definition.document("graph", GraphSpec, _DOCUMENT_FIELDS["graph"][1])
    task = definition.document("task", TaskSpec, _DOCUMENT_FIELDS["task"][1])
    definition.objective(graph)
    definition.training(task)
    patches = []
    documents: set[str] = set()
    if change is not None:
        field = definition.fields.get(change.field)
        control = field.view.control if field is not None else None
        if field is None or control is None:
            raise ValueError(f"Field {change.field!r} has no supported edit")
        value = change.value
        if isinstance(control, StudioAuthoringChoice):
            if value not in {option.value for option in control.options}:
                raise ValueError(f"{field.view.label} requires a declared choice")
        elif (
            type(value) not in (int, float)
            or (type(value) is int and abs(value) > 2**53 - 1)
            or not math.isfinite(value)
            or (control.step == 1 and (value != int(value) or abs(value) > 2**53 - 1))
            or (control.min is not None and value < control.min)
            or (control.max is not None and value > control.max)
            or (control.step is None and value <= 0)
        ):
            raise ValueError(f"{field.view.label} requires a number in the supported range")
        if isinstance(control, StudioAuthoringNumber) and control.step == 1:
            value = int(value)
        if value != field.view.value:
            patches = field.write(value)
            documents.update(field.documents)
            documents.update(
                ref.document
                for patch in patches
                if (ref := _read(DocumentReference, patch.value)) is not None
            )
    return [field.view for field in definition.fields.values()], patches, sorted(documents)
