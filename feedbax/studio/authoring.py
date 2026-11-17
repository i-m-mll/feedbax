"""The authored project schema and the controls supported for Train steps."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, get_args, overload

from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

from feedbax.contracts.migrations import default_spec_registry
from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID, ModelDocument
from feedbax.contracts.task_document import TASK_DOCUMENT_SCHEMA_ID, TaskDocument
from feedbax.contracts.objective_document import OBJECTIVE_DOCUMENT_SCHEMA_ID, ObjectiveDocument
from feedbax.contracts.analysis_document import AnalysisDocument
from feedbax.contracts.expression_values import Level, LiteralValue, Problem, ProblemCode, Site, ValueBase, ValueType
from feedbax.contracts.figure_document import FIGURE_DOCUMENT_SCHEMA, FigureSource
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.project import (
    DocumentReference, ExternalBase, InlineBase, Project, Step, StepBase, StepKind, json_pointer_parts,
)
from feedbax.contracts.project_inputs import StepReference, SubsetTerms
from feedbax.contracts.project_plan import ProjectPlan, RunKey
from feedbax.contracts.project_resolution import ResolvedStep, StepTemplate, compose_step
from feedbax.contracts.project_sources import ValueSource, pointer
from feedbax.contracts.report_document import ReportDocument
from feedbax.contracts.request_identity import REQUEST_KINDS, decode_request
from feedbax.contracts.request_specs import DocumentTrainingConfig, DocumentTrainingRequestSpec, TrainContinuation
from feedbax.contracts.request_refs import LastMember, ProductRef
from feedbax.contracts.run_matrix import MatrixCompositionDelta
from feedbax.contracts.studio_authoring import (
    StudioAuthoredDocumentFamily,
    StudioAuthoringCatalog,
    StudioAuthoringChange,
    StudioAuthoringChoice,
    StudioAuthoringDeclaration,
    StudioAuthoringField,
    StudioAuthoringNumber,
    StudioAuthoringOption,
    StudioAuthoringOrigin,
    StudioAuthoringRegistry,
    StudioAuthoringSource,
    StudioAuthoringSurface,
    StudioAuthoringSweepChange,
)
from feedbax.contracts.studio_project import SetProjectValue
from feedbax.contracts.studio_field import TrainFieldRequest, TrainFieldResponse, TrainFieldView
from feedbax.contracts.studio_project import AuthoredFileAddress, StudioProjectFilePatch
from feedbax.expression_registry import FunctionRegistry
from feedbax.plugins.application import (
    COMPONENTS, EXPRESSION_FUNCTIONS, FIGURE_TEMPLATES, OPERATIONS, PYTHON_TASKS, TRAINING_METHODS,
)
from feedbax.plugins.bootstrap import BootstrapState
from feedbax.plugins.declarations import ResolvedDeclaration
from feedbax.studio.authoring_sweeps import SweepTarget, FieldSweeps
from feedbax.studio.numeric_fields import NumericField, prepare_numeric_field, project_numeric_field
from feedbax.training.methods import TrainingMethodRegistry


_TRAIN = REQUEST_KINDS["train"].spec_type
_TRAIN_DOCUMENT_MODELS = {
    "model": (MODEL_DOCUMENT_SCHEMA_ID, ModelDocument),
    "task": (TASK_DOCUMENT_SCHEMA_ID, TaskDocument),
    "objective": (OBJECTIVE_DOCUMENT_SCHEMA_ID, ObjectiveDocument),
}
_STEP_DOCUMENT_FAMILY = {"analyze": "analysis", "report": "report"}
_GRAPH_BATCH_SIZE = 1
_OPTIMIZER_RATE = ("settings", "optimizer", "params", "learning_rate")
type _Path = tuple[str, ...]
type _Scalar = str | int | float
type _ReadPlan = Callable[[tuple[RunKey, ...]], ProjectPlan]
_CHECKPOINT_POLICY = ("checkpoint_initialization",)
_CHECKPOINT_CHOICES = {"initialize_from": "Initialize model only", "continue_from": "Carry training state"}


def _document_schema(model: type[BaseModel]) -> str:
    return default_spec_registry.resolve(model.__name__).identity


def _origins[T](entries: Iterable[ResolvedDeclaration[T]]) -> dict[str, StudioAuthoringOrigin]:
    return {
        entry.key: StudioAuthoringOrigin(
            package=entry.source.package,
            distribution=entry.source.distribution,
            version=entry.source.version,
        )
        for entry in entries
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
    bundle = state.bundle
    method_records = [bundle.training_methods.declaration_record(key) for key in bundle.training_methods.keys()]
    methods = [
        StudioAuthoringDeclaration(
            key=method.key,
            settings_schema=method.settings_model.model_json_schema(),
            label=method.label or method.update.__name__,
            renamed_from=method.renamed_from,
        )
        for record in method_records
        for method in (record.value,)
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
        "training-settings": _family(DocumentTrainingConfig),
        "analysis": _family(AnalysisDocument, AnalysisDocument.model_fields["schema_version"].default),
        "figure": StudioAuthoredDocumentFamily(
            schema_id=FIGURE_DOCUMENT_SCHEMA,
            payload_schema=TypeAdapter(FigureSource | ExternalBase).json_schema(),
        ),
        "report": _family(ReportDocument, ReportDocument.model_fields["schema_version"].default),
        "subset": _family(SubsetTerms, "subset"),
    }
    task_records = [bundle.python_tasks.declaration_record(key) for key in bundle.python_tasks.keys()]
    figure_records = [bundle.figure_templates.declaration_record(key) for key in bundle.figure_templates.keys()]
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
                entries=sorted(bundle.components.list_all(), key=lambda definition: definition.name),
                origins=_origins(bundle.components.declaration_record(key) for key in bundle.components.names()),
            ),
            OPERATIONS.family: StudioAuthoringRegistry(
                entries=list(bundle.operations.definitions()),
                origins=_origins(bundle.operations.resolve(key) for key in bundle.operations.keys()),
            ),
            EXPRESSION_FUNCTIONS.family: StudioAuthoringRegistry(
                entries=list(bundle.expression_functions.definitions()),
                origins={},
            ),
            TRAINING_METHODS.family: StudioAuthoringRegistry(entries=methods, origins=_origins(method_records)),
            PYTHON_TASKS.family: StudioAuthoringRegistry(
                entries=[
                    StudioAuthoringDeclaration(
                        key=task.key, label=task.label,
                        renamed_from=task.renamed_from, settings_schema=task.parameters.model_json_schema(),
                    )
                    for record in task_records
                    for task in (record.value,)
                ],
                origins=_origins(task_records),
            ),
            FIGURE_TEMPLATES.family: StudioAuthoringRegistry(
                entries=[
                    StudioAuthoringDeclaration(
                        key=entry.key, label=entry.label,
                        renamed_from=entry.renamed_from, settings_schema=entry.settings_model.model_json_schema(),
                    )
                    for record in figure_records
                    for entry in (record.value,)
                ],
                origins=_origins(figure_records),
            ),
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
    def __init__(self, project: Project, step_id: str, registry: TrainingMethodRegistry, *, field_editor: bool = False) -> None:
        step = project.steps.get(step_id)
        if step is None or step.kind != "train":
            raise ValueError(f"Unknown Train step {step_id!r}")
        direct = isinstance(step.base, InlineBase) and not (
            step.deltas or step.inputs
            or (not field_editor and step.sweep is not None and (step.sweep.rows or step.sweep.exceptions))
        )
        checkpoint = step.inputs.get("checkpoint")
        linked = (
            isinstance(step.base, StepBase) and step.base.row is None
            and isinstance(checkpoint, StepReference) and checkpoint.step == step.base.step
            and checkpoint.checkpoint == "last"
        )
        if not (direct or linked):
            raise ValueError("These Train controls require an independent Train or a direct checkpoint continuation")
        self.project = project
        self.step_id = step_id
        self.step = step
        self.linked = linked
        self.field_editor = field_editor
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

    def checkpoint_boundary(self, read_plan: _ReadPlan, policy) -> tuple[int, tuple[str, ...]] | None:
        checkpoint = self.step.inputs["checkpoint"]
        assert isinstance(checkpoint, StepReference)
        plan = read_plan(())
        runs = {run.key: run for run in plan.runs}
        pairs = {}
        counts = set()
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
            target_spec = decode_request(plan.requests[run.request_id], plan.documents.__getitem__).spec
            source_spec = decode_request(plan.requests[source.request_id], plan.documents.__getitem__).spec
            if (
                not isinstance(target_spec, DocumentTrainingRequestSpec)
                or not isinstance(source_spec, DocumentTrainingRequestSpec)
                or target_spec.checkpoint_initialization != policy
                or (target_spec.method, target_spec.settings) != (source_spec.method, source_spec.settings)
            ):
                return None
            counts.add(source_spec.target_total)
            pairs[run.key] = source.key
        if not pairs or len(counts) != 1:
            return None
        definitions = read_plan(tuple(dict.fromkeys((*pairs, *pairs.values())))).selected_definitions
        documents: set[str] = set()
        path = "/payload" + pointer(_CHECKPOINT_POLICY)
        for key, source_key in pairs.items():
            target, source = definitions.get(key), definitions.get(source_key)
            if target is None or source is None or any(
                (origin.kind != "step" or origin.path.startswith("/sweep/"))
                and (site == path or site.startswith(path + "/"))
                for site, origin in target.sources.items()
            ):
                return None
            documents.update(target.document_uses | source.document_uses)
            subset = plan.subsets.get((key.variant, key.step, "checkpoint"))
            if subset is not None and subset.document is not None:
                documents.add(subset.document)
        return counts.pop(), tuple(sorted(documents))

    def checkpoint_carry(self, read_plan: _ReadPlan | None) -> None:
        raw = self.payload.get("checkpoint_initialization")
        policy = _read(TrainContinuation, raw) if isinstance(raw, dict) else raw
        preset = policy == "initialize_from" or (
            isinstance(policy, TrainContinuation)
            and (policy.state, policy.schedule, policy.rng) == ("preserve", "continue", "carry")
        )
        scope = self.checkpoint_boundary(read_plan, policy) if preset and read_plan else None
        value = policy.mode if isinstance(policy, TrainContinuation) else policy
        if not preset:
            value = "Authored policy" if raw is not None else None
        elif scope is None:
            value = _CHECKPOINT_CHOICES[value]

        def change(selected: _Scalar) -> list[SetProjectValue]:
            assert scope is not None
            continuation = TrainContinuation(
                source_completed_updates=scope[0], state="preserve", schedule="continue", rng="carry",
            )
            return [self.set(_CHECKPOINT_POLICY, selected if selected == "initialize_from" else continuation.model_dump(mode="json"))]

        self.field(
            _CHECKPOINT_POLICY, "Checkpoint carryover", value,
            _options(_CHECKPOINT_CHOICES) if scope is not None else None, change,
            source=self.sources.get(pointer((*_CHECKPOINT_POLICY, "mode"))),
            documents=scope[1] if scope is not None else (),
        )

    def training(self, read_plan: _ReadPlan | None) -> None:
        key = self.payload.get("method")
        registrations = {name: self.registry.resolve(name) for name in self.registry.keys()}
        registration = registrations.get(key) if isinstance(key, str) else None

        def method_change(selected: _Scalar) -> list[SetProjectValue]:
            model = registrations[selected].settings_model
            try:
                defaults = model.model_validate({}).model_dump(mode="json")
            except ValidationError:
                defaults = {}
            return [self.set(("method",), selected), self.set(("settings",), defaults)]

        self.field(
            ("method",), "Training method", key if isinstance(key, str) else None,
            _options({name: method.label or name for name, method in registrations.items()}),
            method_change,
        )
        if self.linked:
            self.checkpoint_carry(read_plan)
        self.field(
            ("seed",), "Seed", self.payload.get("seed", 0),
            StudioAuthoringNumber(kind="number", min=0, max=2**32 - 1, step=1),
            lambda value: [self.set(("seed",), value)],
        )
        raw = self.payload.get("training_config")
        config = _read(DocumentTrainingConfig, raw)
        if config is not None or raw is None:
            defaults = DocumentTrainingConfig(batch_size=_GRAPH_BATCH_SIZE)
            for name, schema in DocumentTrainingConfig.model_json_schema()["properties"].items():
                path = ("training_config", name)

                def write(value: _Scalar, *, name=name, path=path) -> list[SetProjectValue]:
                    if config is not None:
                        return [self.set(path, value)]
                    return [self.set(("training_config",), defaults.model_dump(mode="json") | {name: value})]

                self.field(
                    path, schema["title"], getattr(config or defaults, name),
                    StudioAuthoringNumber(kind="number", min=1, step=1), write,
                )
        if registration is None:
            return
        raw = self.payload.get("settings")
        if not isinstance(raw, dict):
            return
        schema = registration.settings_model.model_json_schema()

        def fields(path: _Path, values: dict, shape: dict) -> None:
            for name, leaf in values.items():
                props = shape.get("properties", {}).get(name, {})
                if "$ref" in props:
                    props = schema["$defs"][props["$ref"].rsplit("/", 1)[-1]]
                if "anyOf" in props:
                    props = props | next((member for member in props["anyOf"] if member.get("type") != "null"), {})
                site = (*path, name)
                if isinstance(leaf, dict):
                    fields(site, leaf, props)
                elif type(leaf) in (int, float):
                    minimum = props.get("minimum")
                    if site == _OPTIMIZER_RATE:
                        minimum = 0
                    self.field(
                        site, props.get("title", name.replace("_", " ").title()), leaf,
                        StudioAuthoringNumber(
                            kind="number", min=minimum, max=props.get("maximum"),
                            step=1 if props.get("type") == "integer" else None,
                        ),
                        lambda value, site=site: [self.set(site, value)],
                    )

        fields(("settings",), raw, schema)


def project_train_fields(
    project: Project,
    step_id: str,
    registry: TrainingMethodRegistry,
    change: StudioAuthoringChange | StudioAuthoringSweepChange | None,
    *,
    read_plan: _ReadPlan | None = None,
) -> tuple[
    list[StudioAuthoringField], list[SetProjectValue], list[str],
]:
    """Project the supported fields and plan one atomic, step-local edit."""
    definition = _TrainFields(project, step_id, registry)
    definition.training(read_plan)

    sweeps = {}
    for field_id, field in definition.fields.items():
        target = SweepTarget.from_field(field.view.value, field.view.control, field.write)
        if target is not None and not definition.linked:
            sweep = FieldSweeps(definition.step, target, field.view.label, integer=field.view.control.step == 1)
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
    if isinstance(change, StudioAuthoringSweepChange):
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
    return [field.view for field in definition.fields.values()], patches, sorted(documents)


def project_training_numeric_fields(
    project: Project, step_id: str, registry: TrainingMethodRegistry, functions: FunctionRegistry,
    body: TrainFieldRequest, read_plan: _ReadPlan,
) -> TrainFieldResponse:
    definition = _TrainFields(project, step_id, registry, field_editor=True)
    definition.training(read_plan)
    selected = effective = None
    if body.run is not None:
        key = RunKey(body.run.variant, body.run.step, body.run.row)
        if key.step != step_id:
            raise ValueError("Choose a run of this Train step.")
        plan = read_plan((key,))
        selected = next((row for row in plan.rows if row.key == key), None)
        effective = plan.selected_definitions.get(key)
        available = {row.key.row for row in plan.rows if (row.key.variant, row.key.step) == (key.variant, step_id)}
        if not set(body.rows) <= available or len(set(body.rows)) != len(body.rows):
            raise ValueError("The selected rows no longer belong to this Train step.")
    elif body.rows:
        raise ValueError("Choose a run of this Train step.")
    fields = []
    targets = {}
    for field_id, field in definition.fields.items():
        view = field.view
        if not isinstance(view.control, StudioAuthoringNumber) or type(view.value) not in (float, int):
            continue
        path = json_pointer_parts(field_id)
        target = SweepTarget.from_field(view.value, view.control, field.write) if not definition.linked else None
        axis = FieldSweeps(definition.step, target, view.label, integer=view.control.step == 1) if target is not None else None
        numeric = NumericField(
            path, view.label, view.value,
            Site(expected=ValueType(base=ValueBase.INTEGER if view.control.step == 1 else ValueBase.REAL),
                 finest=Level.RUN, structural=True),
            target is not None, lambda value, view=view: _number(view, value),
        )
        fields.append(project_numeric_field(
            TrainFieldView, numeric, axis, functions, selected,
            _get(effective.payload, path) if effective else None, id=field_id,
        ))
        targets[field_id] = field, numeric, axis
    if body.edit is None:
        return TrainFieldResponse(fields=fields)
    if body.field not in targets:
        raise ValueError("Choose a supported numeric Train field.")
    field, numeric, axis = targets[body.field]
    prepared = prepare_numeric_field(
        numeric, axis, body.edit, functions, rows=tuple(body.rows), keep_value=field.write,
    )
    result = prepared.result
    if isinstance(result, tuple):
        return TrainFieldResponse(problems=list(result))
    if isinstance(result, list):
        patches = result
    elif definition.step.sweep and axis is None:
        return TrainFieldResponse(problems=[Problem(
            code=ProblemCode.STRUCTURE, reason="This Train field has no writable sweep destination.",
        )])
    else:
        assert not isinstance(result, LiteralValue)
        patches = definition.authored_edits(body.field, field.write(result))
    return TrainFieldResponse(
        files=[StudioProjectFilePatch(address=AuthoredFileAddress(kind="step", id=step_id), patches=patches)] if patches else [],
        dependencies=list(field.documents),
        removed_values=prepared.removed,
    )


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
    if field.id in {"/settings/grad_clip", pointer(_OPTIMIZER_RATE)} and value <= 0:
        raise ValueError(f"{field.label} requires a positive number")
    return int(value) if control.step == 1 else value
