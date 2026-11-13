"""The authored project schema and the controls supported for Train steps."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, get_args, overload

from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

from feedbax.component_registry.registry import ComponentRegistry
from feedbax.contracts.migrations import default_spec_registry
from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID, ModelDocument
from feedbax.contracts.task_document import TASK_DOCUMENT_SCHEMA_ID, TaskDocument
from feedbax.contracts.objective_document import OBJECTIVE_DOCUMENT_SCHEMA_ID, ObjectiveDocument
from feedbax.contracts.analysis_document import AnalysisDocument
from feedbax.contracts.checkpoints import CheckpointContinuationRequest
from feedbax.contracts.figure_document import FIGURE_DOCUMENT_SCHEMA, FigureDocument
from feedbax.contracts.figures import COMPARISON_POLICY_SCHEMA_ID, ComparisonPolicySpec
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.project import (
    DocumentReference, ExternalBase, InlineBase, Project, Step, StepBase, StepKind, json_pointer_parts,
)
from feedbax.contracts.project_inputs import StepReference, SubsetTerms
from feedbax.contracts.project_plan import ProjectPlan, RunKey
from feedbax.contracts.project_resolution import ResolvedStep, StepTemplate, compose_step
from feedbax.contracts.project_sources import ValueSource, pointer
from feedbax.contracts.report_document import ReportDocument
from feedbax.contracts.request_identity import REQUEST_KINDS
from feedbax.contracts.request_refs import LastMember, ProductRef
from feedbax.contracts.run_matrix import MatrixCompositionDelta
from feedbax.contracts.studio_authoring import (
    StudioAuthoredDocumentFamily,
    StudioAuthoringCatalog,
    StudioAuthoringChange,
    StudioAuthoringChoice,
    StudioAuthoringField,
    StudioAuthoringMethod,
    StudioAuthoringNumber,
    StudioAuthoringOption,
    StudioAuthoringOrigin,
    StudioAuthoringRegistry,
    StudioAuthoringSource,
    StudioAuthoringSurface,
    StudioAuthoringSweepChange,
    StudioReachTargetAuthoring,
    StudioReachTargetEdit,
)
from feedbax.contracts.studio_project import SetProjectValue
from feedbax.contracts.studio_training import (
    STUDIO_GRAPH_METHOD,
    STUDIO_INDEPENDENT_KEY_SCHEDULE,
    StudioGraphMethodPayload,
    StudioTrainContinuation,
)
from feedbax.contracts.training import (
    MethodPayloadEnvelope,
    MethodRefSpec,
    OptimizerSpec,
    TrainingConfig,
    TrainingProgramRegistry,
)
from feedbax.plugins.application import COMPONENTS, EXPRESSION_FUNCTIONS, OPERATIONS, TRAINING_PROGRAMS
from feedbax.plugins.bootstrap import BootstrapState
from feedbax.studio.authoring_sweeps import SweepTarget, TrainFieldSweeps
from feedbax.training.optimizers import optimizer_types


_TRAIN = REQUEST_KINDS["train"].spec_type
_TRAIN_DOCUMENT_MODELS = {
    "model": (MODEL_DOCUMENT_SCHEMA_ID, ModelDocument),
    "task": (TASK_DOCUMENT_SCHEMA_ID, TaskDocument),
    "objective": (OBJECTIVE_DOCUMENT_SCHEMA_ID, ObjectiveDocument),
}
_STEP_DOCUMENT_FAMILY = {"analyze": "analysis", "report": "report"}
_GRAPH_BATCH_SIZE = 1
_DEFAULT_OPTIMIZER = "adam"
type _Path = tuple[str, ...]
type _Scalar = str | int | float
type _ReadPlan = Callable[[tuple[RunKey, ...]], ProjectPlan]
_METHOD_PAYLOAD = ("method_payload", "payload")
_CHECKPOINT_POLICY = (*_METHOD_PAYLOAD, "checkpoint_initialization")
_METHOD_TOTAL = (*_METHOD_PAYLOAD, "target_total")
_CHECKPOINT_CHOICES = {"initialize_from": "Initialize model only", "continue_from": "Carry training state"}


def _document_schema(model: type[BaseModel]) -> str:
    return default_spec_registry.resolve(model.__name__).identity


def _origins(state: BootstrapState, family: str) -> dict[str, StudioAuthoringOrigin]:
    return {
        key: StudioAuthoringOrigin(plugin=origin.plugin_id, distribution=origin.distribution)
        for origin in state.provenance
        for key in origin.registered_keys.get(family, ())
    }


def _family(
    model: type[BaseModel], schema_id: str | None = None,
) -> StudioAuthoredDocumentFamily:
    return StudioAuthoredDocumentFamily(schema_id=schema_id, payload_schema=model.model_json_schema())


@overload
def authoring_surface(state: BootstrapState, *, catalog: Literal[False] = False) -> StudioAuthoringSurface: ...


@overload
def authoring_surface(state: BootstrapState, *, catalog: Literal[True]) -> StudioAuthoringCatalog: ...


def authoring_surface(
    state: BootstrapState, *, catalog: bool = False,
) -> StudioAuthoringSurface | StudioAuthoringCatalog:
    """Describe authored shapes and registry declarations from one bootstrap state."""
    registry = state.bundle.training_programs
    method_origins = _origins(state, TRAINING_PROGRAMS.family)
    methods = [
        StudioAuthoringMethod(
            key=key,
            schema_id=method.payload_schema_id,
            schema_version=method.payload_schema_version,
            owner=method.owner,
            package=method.package,
            plugin=origin.plugin if (origin := method_origins.get(key)) else None,
            payload_schema=method.payload_model.model_json_schema(),
        )
        for key in registry.available_keys()
        for method in (registry.resolve(key, path="/method_ref"),)
    ]
    project_grammar = Project.model_json_schema(by_alias=True)
    train_documents = {
        name: (schema, model.model_json_schema())
        for name, (schema, model) in _TRAIN_DOCUMENT_MODELS.items()
    }
    if not catalog:
        return StudioAuthoringSurface(
            project_grammar=project_grammar,
            step_payloads={"train": _TRAIN.model_json_schema()},
            document_schemas=dict(train_documents.values()),
            training_methods=methods,
        )

    documents = {
        name: StudioAuthoredDocumentFamily(schema_id=schema_id, payload_schema=payload)
        for name, (schema_id, payload) in train_documents.items()
    } | {
        "training-settings": _family(TrainingConfig),
        "analysis": _family(AnalysisDocument, AnalysisDocument.model_fields["schema_version"].default),
        "figure": StudioAuthoredDocumentFamily(
            schema_id=FIGURE_DOCUMENT_SCHEMA,
            payload_schema=TypeAdapter(FigureDocument | ExternalBase).json_schema(),
        ),
        "report": _family(ReportDocument, ReportDocument.model_fields["schema_version"].default),
        "comparison-policy": _family(ComparisonPolicySpec, COMPARISON_POLICY_SCHEMA_ID),
        "subset": _family(SubsetTerms, "subset"),
    }
    return StudioAuthoringCatalog(
        project_grammar=project_grammar,
        step_grammar=Step.model_json_schema(by_alias=True),
        step_payloads={
            kind: (
                documents[_STEP_DOCUMENT_FAMILY[kind]].payload_schema
                if kind in _STEP_DOCUMENT_FAMILY
                else REQUEST_KINDS[kind].spec_type.model_json_schema()
            )
            for kind in get_args(StepKind.__value__)
        },
        document_families=documents,
        registries={
            COMPONENTS.family: StudioAuthoringRegistry(
                entries=sorted(state.bundle.components.list_all(), key=lambda definition: definition.name),
                origins=_origins(state, COMPONENTS.family),
            ),
            OPERATIONS.family: StudioAuthoringRegistry(
                entries=list(state.bundle.operations.definitions()),
                origins=_origins(state, OPERATIONS.family),
            ),
            EXPRESSION_FUNCTIONS.family: StudioAuthoringRegistry(
                entries=list(state.bundle.expression_functions.definitions()),
                origins=_origins(state, EXPRESSION_FUNCTIONS.family),
            ),
            "training_methods": StudioAuthoringRegistry(entries=methods, origins=method_origins),
        },
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


class _TrainFields:
    def __init__(self, project: Project, step_id: str, registry: TrainingProgramRegistry) -> None:
        step = project.steps.get(step_id)
        if step is None or step.kind != "train":
            raise ValueError(f"Unknown Train step {step_id!r}")
        direct = isinstance(step.base, InlineBase) and not (
            step.deltas or step.inputs
            or (step.sweep is not None and (step.sweep.rows or step.sweep.exceptions))
        )
        checkpoint = step.inputs.get("checkpoint")
        linked = (
            isinstance(step.base, StepBase) and step.base.row is None
            and isinstance(checkpoint, StepReference) and checkpoint.step == step.base.step
            and checkpoint.checkpoint == "last"
        )
        if project.sweep is not None or not (direct or linked):
            raise ValueError("These Train controls require an independent Train or a direct checkpoint continuation")
        self.project = project
        self.step_id = step_id
        self.step = step
        self.linked = linked
        self.direct_unswept = direct and step.sweep is None and not step.document_overrides
        self.registry = registry
        template = compose_step(project, step_id, track_sources=True)
        resolved = (
            template.resolve(include_sources=True) if isinstance(template, StepTemplate) else None
        )
        if linked:
            if not isinstance(template, StepTemplate) or not isinstance(resolved, ResolvedStep):
                raise ValueError("The checkpoint continuation's definition could not be resolved")
            self.payload = template.authored_payload()
        else:
            assert isinstance(step.base, InlineBase)
            self.payload = step.base.inline
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

    def authored_edits(self, field: str, edits: list[SetProjectValue]) -> list[SetProjectValue]:
        if not self.linked or not edits:
            return edits
        patches = [
            OverridePatch(op=edit.op, path=".".join(json_pointer_parts(edit.path)[2:]), value=edit.value)
            for edit in edits
        ]
        layer = MatrixCompositionDelta(
            layer_id=f"studio-{field.replace('/', '-').strip('-')}-{len(self.step.deltas) + 1}",
            patches=patches,
            acknowledges_ancestor_paths=[patch.path for patch in patches],
        ).model_dump(mode="json", exclude_unset=True)
        has_deltas = "deltas" in self.step.model_fields_set
        return [SetProjectValue(
            op="add",
            path=f"/deltas/{len(self.step.deltas)}" if has_deltas else "/deltas",
            value=layer if has_deltas else [layer],
        )]

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

    def inline_component_fields(self, graph: GraphSpec, components: ComponentRegistry) -> None:
        for node, component in graph.nodes.items():
            meta = components.get(component.type)
            if meta is None or (component.revision is not None and component.revision != meta.revision):
                continue
            for schema in meta.param_schema:
                path = ("graph", "nodes", node, "params", schema.name)
                value = _get(self.payload, path)
                if type(value) not in (int, float) or schema.type not in ("int", "float"):
                    continue
                self.field(
                    path, schema.name.replace("_", " ").capitalize(), value,
                    StudioAuthoringNumber(
                        kind="number", min=schema.min, max=schema.max,
                        step=1 if schema.type == "int" else None,
                    ),
                    lambda value, path=path: [self.set(path, value)],
                )

    def studio_method(self, payload: dict[str, JsonValue]) -> StudioGraphMethodPayload | None:
        method_ref = _read(MethodRefSpec, payload.get("method_ref"))
        envelope = _read(MethodPayloadEnvelope, payload.get("method_payload"))
        if method_ref != STUDIO_GRAPH_METHOD or envelope is None:
            return None
        try:
            method = self.registry.validate_payload(method_ref, envelope, path="/method_payload")
        except ValueError:
            return None
        return method if isinstance(method, StudioGraphMethodPayload) else None

    def checkpoint_boundary(
        self, read_plan: _ReadPlan, method: StudioGraphMethodPayload, batches: int,
    ) -> tuple[int, tuple[str, ...]] | None:
        checkpoint = self.step.inputs["checkpoint"]
        assert isinstance(checkpoint, StepReference)
        plan = read_plan(())
        runs = {run.key: run for run in plan.runs}
        pairs = {}
        for row in plan.rows:
            if row.key.step != self.step_id:
                continue
            run = runs.get(row.key)
            if run is None or run.admission_problem is not None or run.inputs.keys() != {"checkpoint"}:
                return None
            binding = run.inputs["checkpoint"]
            source = runs.get(binding.producer)
            reference = binding.member.reference
            if (
                source is None or source.kind != "train" or source.admission_problem is not None
                or (source.key.variant, source.key.step) != (run.key.variant, checkpoint.step)
                or not isinstance(reference, ProductRef) or not isinstance(reference.select, LastMember)
                or reference.role != "checkpoint" or reference.request_id != source.request_id
            ):
                return None
            pairs[run.key] = source.key
        if not pairs:
            return None

        definitions = read_plan(tuple(dict.fromkeys((*pairs, *pairs.values())))).selected_definitions
        counts: set[int] = set()
        documents: set[str] = set()
        editable_paths = tuple("/payload" + pointer(path) for path in (
            _CHECKPOINT_POLICY, _METHOD_TOTAL, ("training_config", "n_batches"),
        ))
        for key, source_key in pairs.items():
            target = definitions.get(key)
            source = definitions.get(source_key)
            if target is None or source is None:
                return None
            current = self.studio_method(target.payload)
            predecessor = self.studio_method(source.payload)
            target_config = _read(TrainingConfig, target.payload.get("training_config"))
            source_config = _read(TrainingConfig, source.payload.get("training_config"))
            if (
                current is None or predecessor is None or target_config is None or source_config is None
                or current.checkpoint_initialization != method.checkpoint_initialization
                or current.target_total != method.target_total or target_config.n_batches != batches
                or current.optimizer != predecessor.optimizer
                or target_config.grad_clip != source_config.grad_clip
                or (current.optimizer.lr_schedule is not None and current.optimizer.lr_schedule.kind != "constant")
                or any(
                    (origin.kind != "step" or origin.path.startswith("/sweep/")) and any(
                        path == field or path.startswith(field + "/") for field in editable_paths
                    ) for path, origin in target.sources.items()
                )
            ):
                return None
            counts.add(predecessor.target_total)
            documents.update(target.document_uses | source.document_uses)
            subset = plan.subsets.get((key.variant, key.step, "checkpoint"))
            if subset is not None and subset.document is not None:
                documents.add(subset.document)
        return (counts.pop(), tuple(sorted(documents))) if len(counts) == 1 else None

    def checkpoint_carry(self, method: StudioGraphMethodPayload, read_plan: _ReadPlan | None) -> None:
        policy = method.checkpoint_initialization
        stateful = isinstance(policy, StudioTrainContinuation)
        value = policy.mode if stateful else policy
        batches = policy.progress.additional_batches if stateful else method.target_total
        preset = policy == "initialize_from" or (
            stateful
            and (policy.optimizer, policy.schedule, policy.rng, policy.cursor) == (
                "preserve", "continue", "carry", "carry",
            )
            and not policy.progress.self_contained and not policy.progress.schedule_discontinuity_exemptions
        )
        scope = self.checkpoint_boundary(read_plan, method, batches) if preset and read_plan and batches else None
        if not preset:
            value = "Authored policy" if value is not None else None
        elif scope is None and value is not None:
            value = _CHECKPOINT_CHOICES[value]
        elif stateful and scope is not None and policy.progress.source_completed_batches != scope[0]:
            value = None

        def change(selected: _Scalar) -> list[SetProjectValue]:
            assert scope is not None and batches is not None
            if selected == "initialize_from":
                return [self.set(_CHECKPOINT_POLICY, "initialize_from"), self.set(_METHOD_TOTAL, batches)]
            continuation = StudioTrainContinuation(
                mode="continue_from", optimizer="preserve", schedule="continue", rng="carry", cursor="carry",
                progress=CheckpointContinuationRequest(source_completed_batches=scope[0], additional_batches=batches),
            )
            return [
                self.set(_CHECKPOINT_POLICY, continuation.model_dump(mode="json", exclude_unset=True)),
                self.set(_METHOD_TOTAL, continuation.progress.target_total),
            ]

        self.field(
            _CHECKPOINT_POLICY, "Checkpoint carryover", value,
            _options(_CHECKPOINT_CHOICES) if scope is not None else None, change,
            source=self.sources.get(pointer((*_CHECKPOINT_POLICY, "mode"))),
            documents=scope[1] if scope is not None else (),
        )

    def training(self, n_steps: int | None, read_plan: _ReadPlan | None) -> None:
        raw_config = self.payload.get("training_config")
        config = None if _read(DocumentReference, raw_config) else _read(TrainingConfig, raw_config)
        defaults = TrainingConfig(batch_size=_GRAPH_BATCH_SIZE)
        if n_steps is not None:
            defaults.n_reach_steps = n_steps
        method_ref = _read(MethodRefSpec, self.payload.get("method_ref"))
        registration = self.registry.resolve(STUDIO_GRAPH_METHOD, path="/method_ref")
        method = self.studio_method(self.payload)

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
            if self.linked:
                self.checkpoint_carry(method, read_plan)
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
                method.seed,
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
            initial_config["n_batches"] = (
                method.continuation.progress.additional_batches
                if method.continuation is not None else method.target_total
            )
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
                elif name == "n_batches" and method.continuation is not None:
                    assert type(value) is int
                    patches.extend((
                        self.set(_METHOD_TOTAL, method.continuation.progress.source_completed_batches + value),
                        self.set((*_CHECKPOINT_POLICY, "progress", "additional_batches"), value),
                    ))
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
    change: StudioAuthoringChange | StudioAuthoringSweepChange | StudioReachTargetEdit | None,
    *,
    components: ComponentRegistry | None = None,
    read_plan: _ReadPlan | None = None,
) -> tuple[
    list[StudioAuthoringField], list[SetProjectValue], list[str], StudioReachTargetAuthoring | None,
]:
    """Project the supported fields and plan one atomic, step-local edit."""
    definition = _TrainFields(project, step_id, registry)
    reach_target = None
    reach_edit = change if isinstance(change, StudioReachTargetEdit) else None
    task = _read(TaskDocument, _get(definition.payload, ("trial_documents", "task")))
    definition.training(task.fixed_steps if task is not None else None, read_plan)

    sweeps = {}
    for field_id, field in definition.fields.items():
        target = SweepTarget.from_field(field.view.value, field.view.control, field.write)
        if target is not None and not definition.linked:
            sweep = TrainFieldSweeps(definition.step, target, field.view.label)
            sweeps[field_id] = sweep
            field.view.sweep = sweep.view()
            field.view.can_sweep = sweep.can_sweep
        if not definition.linked and definition.step.sweep is not None and (
            target is None
            or not sweeps[field_id].can_sweep
        ):
            # Keep the typed editor for the sweep's own values.
            if field.view.sweep is None:
                field.view.control = None

    patches = []
    documents: set[str] = set()
    if reach_edit is not None:
        raise ValueError("Edit the target in the authored Task document.")
    elif isinstance(change, StudioAuthoringSweepChange):
        sweep = sweeps.get(change.field)
        if sweep is None:
            raise ValueError(f"Field {change.field!r} has no supported sweep edit")
        field = definition.fields[change.field]
        patches = sweep.edit(change, lambda value: _number(field.view, value))
        documents.update(field.documents)
    elif change is not None:
        field = definition.fields.get(change.field)
        control = field.view.control if field is not None else None
        if field is None or control is None or field.view.sweep is not None:
            raise ValueError(f"Field {change.field!r} has no supported edit")
        value = change.value
        if isinstance(control, StudioAuthoringChoice):
            if value not in {option.value for option in control.options}:
                raise ValueError(f"{field.view.label} requires a declared choice")
        else:
            value = _number(field.view, value)
        if value != field.view.value:
            patches = field.write(value)
            documents.update(field.documents)
            documents.update(
                ref.document
                for patch in patches
                if (ref := _read(DocumentReference, patch.value)) is not None
            )
    if isinstance(change, (StudioAuthoringChange, StudioAuthoringSweepChange)):
        patches = definition.authored_edits(change.field, patches)
    return [field.view for field in definition.fields.values()], patches, sorted(documents), reach_target


def _number(field: StudioAuthoringField, value: _Scalar) -> int | float:
    control = field.control
    if not isinstance(control, StudioAuthoringNumber) or (
        type(value) not in (int, float)
        or (type(value) is int and abs(value) > 2**53 - 1)
        or not math.isfinite(value)
        or (control.step == 1 and (value != int(value) or abs(value) > 2**53 - 1))
        or (control.min is not None and value < control.min)
        or (control.max is not None and value > control.max)
    ):
        raise ValueError(f"{field.label} requires a number in the supported range")
    if field.id == "/training_config/learning_rate" and value <= 0:
        raise ValueError(f"{field.label} requires a positive number")
    return int(value) if control.step == 1 else value
