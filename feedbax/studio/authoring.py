"""The authored project schema and the controls supported for Train steps."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin, overload

import optax
from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError
from pydantic.fields import FieldInfo

from feedbax.compiler.model_templates import with_model_placement_reason
from feedbax.contracts.train_settings import setting_schema_node
from feedbax.contracts.migrations import default_spec_registry
from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID, ModelDocument
from feedbax.contracts.task_document import BLANK_TASK, TASK_DOCUMENT_SCHEMA_ID, VALUE_TYPES, TaskDocument
from feedbax.studio.library import installed_library_sources
from feedbax.contracts.objective_document import OBJECTIVE_DOCUMENT_SCHEMA_ID, ObjectiveDocument
from feedbax.contracts.training_document import TRAINING_DOCUMENT_KEYS, TRAINING_DOCUMENT_SCHEMA_ID, TrainingDocument
from feedbax.contracts.analysis_document import AnalysisDocument
from feedbax.contracts.expression_values import Level, LiteralValue, Problem, ProblemCode, Site, ValueBase, ValueType
from feedbax.contracts.figure_document import FIGURE_DOCUMENT_SCHEMA, FigureSource
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.matrix_core import apply_override_patches
from feedbax.contracts.project import (
    BasedDocument, DocumentReference, ExternalBase, InlineBase, Project, RunBase, Step, StepBase, StepKind,
    json_pointer_parts,
)
from feedbax.contracts.project_sweep import SweepCoordinate
from feedbax.contracts.project_inputs import StepReference, SubsetTerms
from feedbax.planner.project_plan import ProjectPlan, RunKey, analyze_input_roles, step_input_roles
from feedbax.contracts.project_resolution import (
    ProjectDiagnostic, ResolvedStep, StepTemplate, compose_document, compose_step, document_write_target,
)
from feedbax.contracts.project_sources import ValueSource, pointer
from feedbax.contracts.report_document import ReportDocument
from feedbax.contracts.request_identity import REQUEST_KINDS, decode_request
from feedbax.contracts.request_specs import DocumentTrainingConfig, DocumentTrainingRequestSpec, TrainContinuation
from feedbax.contracts.request_refs import LastMember, ProductRef
from feedbax.contracts.studio_authoring import (
    StudioAuthoredDocumentFamily,
    StudioLibraryDocument,
    StudioNamedValueType,
    StudioTaskCatalog,
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
    StudioClassField,
    StudioClassSweepChange,
)
from feedbax.contracts.synthesis import SynthesisStepPayload
from feedbax.contracts.studio_project import SetProjectValue
from feedbax.contracts.studio_field import FieldResetEdit, TrainFieldRequest, TrainFieldResponse, TrainFieldView
from feedbax.contracts.studio_project import AuthoredFileAddress, StudioProjectFilePatch
from feedbax.contracts.expression_registry import FunctionRegistry
from feedbax.application.bundle import (
    COMPONENTS, EXPRESSION_FUNCTIONS, FIGURE_TEMPLATES, OPERATIONS, PYTHON_TASKS, SYNTHESIS_METHODS, TRAINING_METHODS,
)
from feedbax.application.bundle import BootstrapState
from feedbax.plugins.declarations import ResolvedDeclaration
from feedbax.studio.authoring_sweeps import SelectedValues, SweepTarget, FieldSweeps
from feedbax.studio.class_fields import ClassField, edit_class_sweep
from feedbax.studio.numeric_fields import NumericField, prepare_numeric_field, project_numeric_field
from feedbax.studio.selection import StoredEdit, TrainSelection
from feedbax.training.methods import RegisteredTrainingMethod, TrainingMethodRegistry
from feedbax.training.optimizers import OPTIMIZERS, SCHEDULES


_TRAIN = REQUEST_KINDS["train"].spec_type
_TRAIN_DOCUMENT_MODELS = {
    "model": (MODEL_DOCUMENT_SCHEMA_ID, ModelDocument),
    "task": (TASK_DOCUMENT_SCHEMA_ID, TaskDocument),
    "objective": (OBJECTIVE_DOCUMENT_SCHEMA_ID, ObjectiveDocument),
    "training": (TRAINING_DOCUMENT_SCHEMA_ID, TrainingDocument),
}
_STEP_DOCUMENT_FAMILY = {"analyze": "analysis", "report": "report"}
_GRAPH_BATCH_SIZE = 1
type _Path = tuple[str, ...]
type _Scalar = str | int | float
type _ReadPlan = Callable[[tuple[RunKey, ...]], ProjectPlan]
_CHECKPOINT_POLICY = ("checkpoint_initialization",)
_OPTAX_ORIGIN = StudioAuthoringOrigin(package="optax", distribution="optax", version=optax.__version__)
_MISSING = object()
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
            label=method.label or method.key,
            renamed_from=method.renamed_from,
        )
        for record in method_records
        for method in (record.value,)
    ]
    synthesis_records = [bundle.synthesis_methods.declaration_record(key) for key in bundle.synthesis_methods.keys()]
    synthesis_methods = [
        StudioAuthoringDeclaration(
            key=entry.key, label=entry.declaration.label or entry.key,
            renamed_from=entry.declaration.renamed_from,
            settings_schema=entry.declaration.settings_model.model_json_schema(),
            outputs=dict(entry.declaration.outputs),
        )
        for record in synthesis_records
        for entry in (record.value,)
    ]
    project_grammar = Project.model_json_schema(by_alias=True)
    train_documents = {
        name: (schema, model.model_json_schema())
        for name, (schema, model) in _TRAIN_DOCUMENT_MODELS.items()
    }
    if not catalog:
        return StudioAuthoringSurface(
            project_grammar=project_grammar,
            step_payloads={"train": _TRAIN.model_json_schema(), "synthesize": SynthesisStepPayload.model_json_schema()},
            document_schemas=dict(train_documents.values()),
            training_methods=methods, synthesis_methods=synthesis_methods,
        )

    documents = {
        name: StudioAuthoredDocumentFamily(schema_id=schema_id, payload_schema=payload)
        for name, (schema_id, payload) in train_documents.items()
    } | {
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
            for kind in get_args(StepKind.__value__) if kind != "synthesize"
        } | {"synthesize": SynthesisStepPayload.model_json_schema()},
        document_families=documents,
        step_inputs=step_input_roles(),
        analyze_inputs=analyze_input_roles(),
        tasks=_task_catalog(),
        registries={
            COMPONENTS.family: StudioAuthoringRegistry(
                entries=[
                    with_model_placement_reason(definition, bundle.components)
                    for definition in sorted(bundle.components.list_all(), key=lambda definition: definition.name)
                ],
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
            SYNTHESIS_METHODS.family: StudioAuthoringRegistry(
                entries=synthesis_methods, origins=_origins(synthesis_records),
            ),
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
            **{
                family: StudioAuthoringRegistry(
                    entries=[
                        StudioAuthoringDeclaration(
                            key=entry.key, label=entry.label, renamed_from=(),
                            settings_schema=entry.model.model_json_schema(),
                        )
                        for entry in entries.values()
                    ],
                    origins={key: _OPTAX_ORIGIN for key in entries},
                )
                for family, entries in (("optimizers", OPTIMIZERS), ("schedules", SCHEDULES))
            },
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


def _task_catalog() -> StudioTaskCatalog:
    documents = []
    for source in installed_library_sources():
        for name, document in source.documents().items():
            if isinstance(document, ProjectDiagnostic):
                raise ValueError(f"{source.distribution} ships {source.name}/{name}, which does not compose: {document.detail}")
            if document.schema_id == TASK_DOCUMENT_SCHEMA_ID:
                documents.append(StudioLibraryDocument(
                    distribution=source.distribution, library=source.name, name=name, document=document,
                ))
    return StudioTaskCatalog(
        blank=BLANK_TASK, documents=documents,
        value_types=[StudioNamedValueType(name=name, type=value_type) for name, value_type in VALUE_TYPES],
    )


type _Edit = list[OverridePatch]


@dataclass(frozen=True)
class TrainProjection:
    fields: list[StudioAuthoringField]
    settings: dict[str, JsonValue] | None
    mixed: list[str]
    stored: StoredEdit
    documents: list[str]
    classes: list[StudioClassField]
    removed: dict[str, list[str]]
    changed: list[str]


@dataclass(frozen=True)
class _Field:
    view: StudioAuthoringField
    write: Callable[[_Scalar], _Edit]
    documents: tuple[str, ...]
    name: _Path

    def number_write(self, value: int | float) -> list[OverridePatch]:
        """A number field's write, which no sweep refuses."""
        patches = self.write(value)
        assert isinstance(patches, list), f"{self.view.label} refused a number"
        return patches


def _step_writes(writes: list[OverridePatch]) -> list[SetProjectValue]:
    """Writes in a step's definition as edits of the step's own inline definition."""
    return [
        SetProjectValue(op=write.op, path=pointer(("base", "inline", *write.path.split("."))), value=write.value)
        for write in writes
    ]


def _changes(path: _Path, old: JsonValue, new: JsonValue) -> list[OverridePatch]:
    """The writes that turn ``old``, present at ``path``, into ``new``, each entry that stays untouched.

    A kept entry is not written, so a sweep's patch at it is not overlapped by a patch for the selected runs.
    """
    if old == new:
        return []
    if not isinstance(old, dict) or not isinstance(new, dict):
        return [OverridePatch(op="replace", path=".".join(path), value=new)]
    return [
        *(write for key, value in new.items()
          for write in (_changes((*path, key), old[key], value) if key in old
                        else [OverridePatch(op="add", path=".".join((*path, key)), value=value)])),
        *(OverridePatch(op="remove", path=".".join((*path, key))) for key in old if key not in new),
    ]


def _effective(payload: dict[str, JsonValue], registry: TrainingMethodRegistry) -> dict[str, JsonValue]:
    """A run's Train definition with its method's defaults filled in, so runs compare by the values they train with."""
    effective = {key: value for key, value in payload.items() if key != "trial_documents"}
    method = payload.get("method")
    if isinstance(method, str) and method in registry.keys():
        settings = _read(registry.resolve(method).settings_model, payload.get("settings"))
        if settings is not None:
            effective["settings"] = settings.model_dump(mode="json")
    config = payload.get("training_config")
    if config is None or isinstance(config, dict):
        effective["training_config"] = DocumentTrainingConfig(batch_size=_GRAPH_BATCH_SIZE).model_dump(mode="json") | (config or {})
    return effective


def _differences(values: list[JsonValue], at: _Path = ()) -> set[str]:
    """Pointers below ``at`` where ``values`` differ: each differing leaf, and each entry some of them lack."""
    if all(value == values[0] for value in values):
        return set()
    if not all(isinstance(value, dict) for value in values):
        return {pointer(at)}
    keys = {key for value in values for key in value}
    return {
        found for key in keys
        for found in (_differences([value[key] for value in values], (*at, key))
                      if all(key in value for value in values) else {pointer((*at, key))})
    }


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


def _base_source(step: Step) -> str:
    base = step.base
    if isinstance(base, StepBase):
        return f"row {base.row!r} of step {base.step!r}"
    if isinstance(base, RunBase):
        return f"run {base.run!r}"
    if isinstance(base, ExternalBase):
        return f"file {base.path!r}"
    if isinstance(base, DocumentReference):
        return f"document {base.document!r}"
    return "its own definition with differences written as text"


class _StepFields:
    """A Train or Evaluate step's own fields as the selected runs have them.

    ``current`` is the definition a write is made against, the step's own when absent. ``selected`` are the selected
    runs' definitions and ``shown`` the one fields are drawn from; a class-valued field draws the parameters every
    selected run's class has.
    """

    def __init__(
        self, project: Project, step_id: str, registry: TrainingMethodRegistry, current: dict[str, JsonValue] | None = None,
        *, shown: dict[str, JsonValue] | None = None, selected: tuple[dict[str, JsonValue], ...] = (),
        runs: tuple[Mapping[str, SweepCoordinate], ...] | None = None,
    ) -> None:
        step = project.steps.get(step_id)
        if step is None or step.kind not in ("train", "evaluate"):
            raise ValueError(f"Unknown Train or Evaluate step {step_id!r}")
        direct = isinstance(step.base, InlineBase) and not (step.deltas or step.kind == "train" and step.inputs)
        inherits = isinstance(step.base, StepBase) and step.base.row is None
        if not (direct or inherits):
            raise ValueError(f"Train step {step_id!r} takes its definition from {_base_source(step)}")
        checkpoint = step.inputs.get("checkpoint")
        self.project = project
        self.step_id = step_id
        self.step = step
        self.inherits = inherits
        self.continues = (
            inherits and isinstance(checkpoint, StepReference) and checkpoint.step == step.base.step
            and checkpoint.checkpoint == "last"
        )
        self.registry = registry
        template = compose_step(project, step_id, track_sources=True)
        resolved = (
            template.resolve(include_sources=True) if isinstance(template, StepTemplate) else None
        )
        if inherits:
            if not isinstance(template, StepTemplate) or not isinstance(resolved, ResolvedStep):
                raise ValueError(f"Train step {step_id!r}'s definition could not be resolved")
            self.authored = template.authored_payload()
        else:
            assert isinstance(step.base, InlineBase)
            self.authored = step.base.inline
        # The definition as the step's runs read it: each named document it refers to read in place.
        self.payload = resolved.payload if isinstance(resolved, ResolvedStep) else self.authored
        self.current = self.payload if current is None else current
        self.shown = shown if shown is not None else selected[0] if selected else self.payload
        self.selected = selected or (self.shown,)
        self.runs = runs
        self.sources = (resolved.sources or {}) if isinstance(resolved, ResolvedStep) else {}
        self.fields: dict[str, _Field] = {}
        self.classes: dict[str, ClassField] = {}
        self.method: RegisteredTrainingMethod | None = None

    def set(self, path: _Path, value: JsonValue) -> OverridePatch:
        """The write of ``value`` at ``path`` in the definition, made against the selected runs' definition."""
        parent = _get(self.current, path[:-1])
        exists = isinstance(parent, dict) and path[-1] in parent
        return OverridePatch(op="replace" if exists else "add", path=".".join(path), value=value)

    def choose_class(self, path: _Path, model: type[BaseModel]) -> _Edit:
        """Replace the value at ``path`` with ``model``'s."""
        current = _get(self.current, path)
        value = replacement(model, current)
        return _changes(path, current, value) if isinstance(current, dict) else [self.set(path, value)]

    def field_label(self, path: _Path) -> str:
        field = self.fields.get(pointer(path))
        return field.view.label if field is not None else _title({}, path[-1])

    def field(
        self,
        path: _Path,
        label: str,
        value: _Scalar | None,
        control: StudioAuthoringChoice | StudioAuthoringNumber | None,
        write: Callable[[_Scalar], _Edit],
        *,
        source: ValueSource | None = None,
        documents: tuple[str, ...] = (),
        name: _Path | None = None,
    ) -> None:
        field_id = pointer(path)
        if not documents and (target := document_write_target(self.authored, ".".join(path))) is not None:
            documents = (target[0],)
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
            name or path[-1:],
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

        def change(selected: _Scalar) -> list[OverridePatch]:
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

    def effective(self, payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """A run's definition as it runs: a Train's with its method's defaults filled in."""
        return _effective(payload, self.registry) if self.step.kind == "train" else payload

    def own(self, read_plan: _ReadPlan | None) -> None:
        """Record the fields of the step's own definition."""
        if self.step.kind == "evaluate":
            self.evaluation()
        else:
            self.training(read_plan)

    def evaluation(self) -> None:
        params = self.payload.get("params")
        if not isinstance(params, dict):
            return
        seed = ("params", "seed")
        self.field(
            seed, "Seed", params.get("seed", 0), StudioAuthoringNumber(kind="number", min=0, max=2**32 - 1, step=1),
            lambda value: [self.set(seed, value)],
        )
        rule = params.get("trial_set")
        if isinstance(rule, dict) and rule.get("kind") == "sample":
            count = ("params", "trial_set", "count")
            self.field(
                count, "Trials", rule.get("count"), StudioAuthoringNumber(kind="number", min=1, max=10_000, step=1),
                lambda value: [self.set(count, value)], name=("trials",),
            )

    def training(self, read_plan: _ReadPlan | None) -> None:
        key = self.payload.get("method")
        registrations = {name: self.registry.resolve(name) for name in self.registry.keys()}
        registration = registrations.get(key) if isinstance(key, str) else None
        self.method = registration

        def method_change(selected: _Scalar) -> _Edit:
            chosen = registrations[selected]
            patches = self.choose_class(("settings",), chosen.settings_model)
            return [self.set(("method",), selected), *patches]

        self.field(
            ("method",), "Method", key if isinstance(key, str) else None,
            _options({name: method.label or name for name, method in registrations.items()}),
            method_change,
        )
        if self.continues:
            self.checkpoint_carry(read_plan)
        self.field(
            ("seed",), "Seed", self.payload.get("seed", 0),
            StudioAuthoringNumber(kind="number", min=0, max=2**32 - 1, step=1),
            lambda value: [self.set(("seed",), value)],
        )
        raw = self.payload.get("training_config")
        if raw is None or isinstance(raw, dict):
            # An invalid value is still drawn in its field, so the problem that names it has a place to go.
            defaults = DocumentTrainingConfig(batch_size=_GRAPH_BATCH_SIZE).model_dump(mode="json")
            for name, schema in DocumentTrainingConfig.model_json_schema()["properties"].items():
                path = ("training_config", name)

                def write(value: _Scalar, *, name=name, path=path) -> list[OverridePatch]:
                    if raw is not None:
                        return [self.set(path, value)]
                    return [self.set(("training_config",), defaults | {name: value})]

                self.field(
                    path, schema["title"], (raw or defaults).get(name, defaults[name]),
                    StudioAuthoringNumber(kind="number", min=1, step=1), write,
                )
        if registration is None:
            return
        raw = self.payload.get("settings")
        if not isinstance(raw, dict):
            return
        schema = registration.settings_model.model_json_schema()
        seen = self.shown.get("settings")

        def fields(
            path: _Path, values: dict, held: dict, shape: dict, anchor: tuple[_Path, dict] | None,
            owner: tuple[str, str, bool] | None, drawn: set[str] | None,
        ) -> None:
            """Draw the fields below ``path``: ``values`` the step's record there, ``held`` the shown run's."""
            for name, declared in shape.get("properties", {}).items():
                if drawn is not None and name not in drawn:
                    continue
                site = (*path, name)
                present = name in values
                leaf = values[name] if present else held.get(name, declared.get("default"))
                record = held.get(name, leaf)
                props = setting_schema_node(schema, declared, record)
                inner = anchor if present or anchor is not None else (site, leaf)
                if isinstance(record, dict):
                    # A chosen class's parameter is named by the field holding the class, and by its own
                    # name only when the class has several: a constant rate is the rate.
                    tag = declared.get("discriminator", {}).get("propertyName")
                    single = tag is not None and len(set(props.get("properties", {})) - {tag}) == 1
                    shared = None
                    if tag is not None:
                        self.class_field(site, _title(declared, name), tag, leaf if present else None)
                        shared = set(props.get("properties", {})).intersection(*(
                            node.get("properties", {}) for payload in self.selected
                            if (node := setting_schema_node(schema, declared, _get(payload, site) or record))
                        ))
                    fields(
                        site, leaf if present and isinstance(leaf, dict) else {}, record, props, inner,
                        (name, _title(declared, name), single) if tag is not None else None, shared,
                    )
                elif type(leaf) in (int, float) and props.get("type") in ("number", "integer"):
                    def write(value: _Scalar, site=site, inner=inner) -> list[OverridePatch]:
                        if inner is None:
                            return [self.set(site, value)]
                        root, default = inner
                        return [self.set(root, _set_in(default, site[len(root):], value))]

                    self.field(
                        site, owner[1] if owner is not None and owner[2] else _title(props, name), leaf,
                        StudioAuthoringNumber(
                            kind="number", min=props.get("minimum"), max=props.get("maximum"),
                            step=1 if props.get("type") == "integer" else None,
                        ),
                        write,
                        name=None if owner is None else (owner[0],) if owner[2] else (owner[0], name),
                    )

        fields(("settings",), raw, seen if isinstance(seen, dict) else raw, schema, None, None, None)

    def class_field(self, site: _Path, label: str, tag: str, stored: JsonValue) -> None:
        """Record the class-valued field at ``site``, ``stored`` the step's record of it."""
        assert self.method is not None
        choice = _class_choice_at(self.method.settings_model, self.shown.get("settings"), site[1:])
        if choice is None:
            return
        _, _, classes = choice
        key = stored.get(tag) if isinstance(stored, dict) else None
        sweeps = None
        if not self.inherits and isinstance(key, str):
            def writes(value: JsonValue) -> list[SetProjectValue]:
                return _step_writes([OverridePatch(op="replace", path=".".join((*site, tag)), value=value)])

            sweeps = FieldSweeps(self.step, SweepTarget(key, writes), label, site[-1:], integer=False, runs=self.runs)
        self.classes[pointer(site)] = ClassField(
            site, label, tag, {key: model for key, model in classes.items() if isinstance(key, str)}, sweeps,
        )


def _title(node: dict, name: str) -> str:
    return node.get("title", name.replace("_", " ").title())


def _classes(annotation: Any) -> list[type[BaseModel]]:
    """The classes a field's annotation admits: itself, a union's members, an optional's."""
    origin = get_origin(annotation)
    if origin is Annotated:
        return _classes(get_args(annotation)[0])
    if origin in (Union, UnionType):
        return [member for argument in get_args(annotation) for member in _classes(argument)]
    return [annotation] if isinstance(annotation, type) and issubclass(annotation, BaseModel) else []


def _class_choices(info: FieldInfo) -> tuple[str, dict[JsonValue, type[BaseModel]]] | None:
    """A class-valued field's tag, and its classes by the key their tag fixes."""
    members = _classes(info.annotation)
    tagged = [
        {name for name, field in member.model_fields.items() if get_origin(field.annotation) is Literal}
        for member in members
    ]
    tags = set.intersection(*tagged) if len(members) > 1 else set()
    if len(tags) != 1:
        return None
    (tag,) = tags
    return tag, {member.model_fields[tag].default: member for member in members}


def _class_at(model: type[BaseModel], value: JsonValue, path: _Path) -> type[BaseModel] | None:
    """The class that governs the record at ``path`` in ``value``, a value of ``model``."""
    for name in path:
        info = model.model_fields.get(name)
        value = value.get(name) if isinstance(value, dict) else None
        if info is None:
            return None
        choices = _class_choices(info)
        if choices is not None:
            tag, members = choices
            chosen = next((member for key, member in members.items() if isinstance(value, dict) and value.get(tag) == key), None)
        else:
            members = _classes(info.annotation)
            chosen = members[0] if len(members) == 1 else None
        if chosen is None:
            return None
        model = chosen
    return model


def _fits(info: FieldInfo, value: JsonValue) -> bool:
    try:
        TypeAdapter(Annotated[info.annotation, info]).validate_python(value)
    except ValidationError:
        return False
    return True


def replacement(model: type[BaseModel], current: JsonValue) -> dict[str, JsonValue]:
    """``model``'s value in place of ``current``: each entry of ``current`` that ``model`` declares and accepts kept,
    ``model``'s defaults for the rest."""
    record = current if isinstance(current, dict) else {}
    value: dict[str, JsonValue] = {}
    for name, info in model.model_fields.items():
        if name in record and get_origin(info.annotation) is not Literal and _fits(info, record[name]):
            value[name] = record[name]
        elif not info.is_required():
            value[name] = TypeAdapter(info.annotation).dump_python(info.get_default(call_default_factory=True), mode="json")
    return value


def _set_in(value: JsonValue, path: _Path, leaf: JsonValue) -> JsonValue:
    if not path:
        return leaf
    record = value if isinstance(value, dict) else {}
    return record | {path[0]: _set_in(record.get(path[0]), path[1:], leaf)}


def project_train_fields(
    project: Project,
    step_id: str,
    registry: TrainingMethodRegistry,
    change: StudioAuthoringChange | None,
    selection: TrainSelection,
    *,
    read_plan: _ReadPlan | None = None,
    class_sweep: StudioClassSweepChange | None = None,
    reset: str | None = None,
) -> TrainProjection:
    """Project the Train step's declared fields and the selected runs' settings, and store one atomic edit for the
    selected runs."""
    payloads = selection.payloads
    definition = _StepFields(project, step_id, registry, selection.current, selected=payloads, runs=selection.coordinates)
    definition.own(read_plan)
    settings = definition.shown.get("settings")
    mixed = _differences([definition.effective(payload) for payload in payloads]) if payloads else set()
    stored = StoredEdit([])
    removed: dict[str, list[str]] = {}
    documents: set[str] = set()
    if sum(item is not None for item in (change, class_sweep, reset)) > 1:
        raise ValueError("An edit is a change, a sweep edit or a reset, one at a time.")
    if class_sweep is not None:
        made = _class_sweep_edit(definition, class_sweep, selection)
        stored, removed = made.stored, made.removed
    if reset is not None:
        stored = selection.store(reset_writes(project, step_id, reset))
    if change is not None:
        field = definition.fields.get(change.field)
        patches: _Edit = []
        if field is None:
            patches = _settings_edit(definition, change)
        else:
            control = field.view.control
            if control is None:
                raise ValueError(f"{field.view.label} of step {step_id!r} is read-only")
            value = change.value
            if isinstance(control, StudioAuthoringChoice):
                if value not in {option.value for option in control.options}:
                    raise ValueError(f"{field.view.label} has no choice {value!r}")
            else:
                value = _number(field.view, value)
            held = _get(definition.current, tuple(json_pointer_parts(change.field))) if selection.scoped else field.view.value
            if value != held:
                patches = field.write(value)
                documents.update(field.documents)
                documents.update(
                    ref.document
                    for patch in patches
                    if (ref := _read(DocumentReference, patch.value)) is not None
                )
        if patches:
            stored = selection.store(patches)
    coordinates = selection.row.coordinates if selection.row is not None else {}
    return TrainProjection(
        [field.view for field in definition.fields.values()],
        settings if isinstance(settings, dict) else None,
        sorted(mixed),
        stored,
        sorted(documents),
        [
            field.view(
                list(dict.fromkeys(
                    key for payload in definition.selected
                    if isinstance(key := _get(payload, (*field.site, field.tag)), str)
                )),
                coordinate.value_id if (members := field.members()) is not None
                and (coordinate := coordinates.get(members[0])) is not None else None,
            )
            for field in definition.classes.values()
        ],
        removed,
        sorted(_changed_definition(project, step_id, definition, settings)),
    )


def _changed_definition(project: Project, step_id: str, definition: _StepFields, settings: Any) -> set[str]:
    """The pointers of the Train definition whose value a difference sets: its fields, its settings at every depth,
    and its classes, a class being changed where its tag is."""
    tags = {f"{at}/{field.tag}": at for at, field in definition.classes.items()}
    changed = changed_fields(project, step_id, [
        *definition.fields, *definition.classes, *tags,
        *(f"/settings{pointer}" for pointer in _pointers(settings)),
    ])
    return changed | {tags[at] for at in changed & tags.keys()}


def _pointers(value: Any, at: str = "") -> Iterator[str]:
    """``value``'s pointer and those of every value inside it."""
    yield at
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _pointers(child, f"{at}/{str(key).replace('~', '~0').replace('/', '~1')}")


def _class_sweep_edit(definition: _StepFields, change: StudioClassSweepChange, selection: TrainSelection):
    field = definition.classes.get(change.field)
    if field is None:
        raise ValueError(f"Train step {definition.step_id!r} has no class field {change.field!r}")
    stored = _get(definition.payload, field.site)
    return edit_class_sweep(
        field, change, stored if isinstance(stored, dict) else {}, scoped=selection.scoped,
        store=selection.store,
        step_file=lambda patches: StudioProjectFilePatch(
            address=AuthoredFileAddress(kind="step", id=definition.step_id), patches=patches,
        ),
    )


def _class_choice_at(
    model: type[BaseModel], value: JsonValue, path: _Path,
) -> tuple[_Path, str, dict[JsonValue, type[BaseModel]]] | None:
    """The class-valued field that ``path`` names, or whose tag it names: its path, tag and classes."""
    owner = _class_at(model, value, path[:-1])
    info = owner.model_fields.get(path[-1]) if owner is not None else None
    if info is not None and (choices := _class_choices(info)) is not None:
        return path, *choices
    if len(path) < 2:
        return None
    holder = _class_at(model, value, path[:-2])
    info = holder.model_fields.get(path[-2]) if holder is not None else None
    choices = _class_choices(info) if info is not None else None
    if choices is not None and choices[0] == path[-1]:
        return path[:-1], *choices
    return None


def _settings_edit(definition: _StepFields, change: StudioAuthoringChange) -> _Edit:
    """Write one value under ``/settings`` when the method's settings model accepts the result.

    At a class-valued field, or at a class's tag, the value is the key of the class chosen in place of the current one.
    """
    path = tuple(json_pointer_parts(change.field))
    settings = definition.current.get("settings")
    method = definition.method
    missing = ValueError(f"Train step {definition.step_id!r} has no field {change.field!r}")
    if len(path) < 2 or path[0] != "settings" or method is None or not isinstance(settings, dict):
        raise missing
    owner = _class_at(method.settings_model, settings, path[1:-1])
    if owner is None or path[-1] not in owner.model_fields:
        raise missing
    choice = _class_choice_at(method.settings_model, settings, path[1:])
    if choice is not None:
        below, tag, members = choice
        chosen = next((member for key, member in members.items() if key == change.value), None)
        if chosen is None:
            raise ValueError(f"{definition.field_label(('settings', *below))} has no class {change.value!r}")
        if _get(settings, (*below, tag)) == change.value:
            return []
        swept = definition.classes.get(pointer(("settings", *below)))
        if swept is not None and swept.covered:
            raise ValueError(f"{swept.label} is swept; its classes are the sweep's members")
        edit = definition.choose_class(("settings", *below), chosen)
    else:
        parent = _get(settings, path[1:-1])
        if isinstance(parent, dict) and parent.get(path[-1], _MISSING) == change.value:
            return []
        edit = [definition.set(path, change.value)]
    if isinstance(edit, list):
        before = {_error_key(error) for error in _setting_errors(method.settings_model, settings)}
        edited = apply_override_patches({"settings": settings}, edit)["settings"]
        refused = next((
            error for error in _setting_errors(method.settings_model, edited)
            if error["type"] != "missing" and _error_key(error) not in before
        ), None)
        if refused is not None:
            location = ".".join(str(part) for part in refused["loc"]) or path[-1]
            raise ValueError(f"{location}: {refused['msg']}")
    return edit


def _error_key(error) -> tuple:
    return tuple(error["loc"]), error["type"], error["msg"]


def _setting_errors(model: type[BaseModel], settings: JsonValue) -> list:
    """The validation errors of ``settings``; a required field without a value is the plan's to report."""
    try:
        model.model_validate(settings)
    except ValidationError as error:
        return error.errors()
    return []


def project_numeric_fields(
    project: Project, step_id: str, registry: TrainingMethodRegistry, functions: FunctionRegistry,
    body: TrainFieldRequest, selection: TrainSelection, read_plan: _ReadPlan,
) -> TrainFieldResponse:
    """The number fields of a Train or Evaluate step's own definition, each with its sweep, and the files that store
    an edit of one."""
    definition = _StepFields(
        project, step_id, registry, shown=selection.current, selected=selection.payloads, runs=selection.coordinates,
    )
    definition.own(read_plan)
    effective = [definition.effective(payload) for payload in selection.payloads]
    selected = (
        SelectedValues(selection.values, selection.current)
        if selection.values is not None and selection.current is not None else None
    )
    fields = []
    targets = {}
    for field_id, field in definition.fields.items():
        view = field.view
        if not isinstance(view.control, StudioAuthoringNumber) or type(view.value) not in (float, int):
            continue
        path = json_pointer_parts(field_id)
        def step_writes(value, field=field):
            return _step_writes(field.number_write(value))
        target = SweepTarget.from_field(view.value, view.control, step_writes) if not definition.inherits else None
        axis = FieldSweeps(
            definition.step, target, view.label, field.name, integer=view.control.step == 1, selected=selected,
            runs=selection.coordinates,
        ) if target is not None else None
        numeric = NumericField(
            path, view.label, view.value,
            Site(expected=ValueType(base=ValueBase.INTEGER if view.control.step == 1 else ValueBase.REAL),
                 finest=Level.RUN, structural=True),
            target is not None, lambda value, view=view: _number(view, value),
        )
        values = [json.loads(value) for value in dict.fromkeys(json.dumps(_get(payload, path)) for payload in effective)]
        fields.append(project_numeric_field(
            TrainFieldView, numeric, axis, functions, selection.row,
            values[0] if len(values) == 1 else None, held=values,
            scoped=selection.scoped, id=field_id,
        ))
        targets[field_id] = field, numeric, axis
    changed = changed_fields(project, step_id, [view.id for view in fields])
    fields = [view.model_copy(update={"changed": view.id in changed}) for view in fields]
    if body.edit is None:
        return TrainFieldResponse(fields=fields)
    if body.field not in targets:
        raise ValueError(f"Step {step_id!r} has no number field {body.field!r}")
    field, numeric, axis = targets[body.field]
    if isinstance(body.edit, FieldResetEdit):
        stored = selection.store(reset_writes(project, step_id, body.field))
        return TrainFieldResponse(files=stored.files, created=list(stored.created), dependencies=list(field.documents))
    prepared = prepare_numeric_field(numeric, axis, body.edit, functions, scoped=selection.scoped)
    result = prepared.result
    if isinstance(result, tuple):
        return TrainFieldResponse(problems=list(result))
    if isinstance(result, list):
        stored = (selection.store(prepared.keep) if prepared.keep else StoredEdit([])).then(
            [StudioProjectFilePatch(address=AuthoredFileAddress(kind="step", id=step_id), patches=result)] if result else [],
        )
    elif definition.step.sweep and axis is None:
        return TrainFieldResponse(problems=[Problem(
            code=ProblemCode.STRUCTURE, reason="This field has no writable sweep destination.",
        )])
    else:
        assert not isinstance(result, LiteralValue)
        definition.current = selection.current if selection.current is not None else definition.payload
        stored = selection.store(field.number_write(result))
    return TrainFieldResponse(
        files=stored.files, created=list(stored.created), dependencies=list(field.documents),
        removed_values=prepared.removed,
    )


def changed_fields(project: Project, step_id: str, fields: Iterable[str]) -> set[str]:
    """Those of ``fields``, pointers in the Train definition, whose value is a difference of the document based on
    another that they are read from."""
    template = compose_step(project, step_id)
    if isinstance(template, ProjectDiagnostic):
        return set()
    payload = template.authored_payload()
    changed = set()
    for field in fields:
        target = document_write_target(payload, ".".join(json_pointer_parts(field)))
        document = project.documents.get(target[0]) if target is not None else None
        if not isinstance(document, BasedDocument):
            continue
        parts = tuple(target[1].split("."))
        if any(parts[:len(path)] == path for path in map(json_pointer_parts, document.differences)):
            changed.add(field)
    return changed


def reset_writes(project: Project, step_id: str, field: str) -> list[OverridePatch]:
    """The write that gives ``field``, a pointer in the Train definition, the value of the base of the document it
    passes through."""
    template = compose_step(project, step_id)
    if isinstance(template, ProjectDiagnostic):
        raise ValueError(template.detail)
    parts = json_pointer_parts(field)
    target = document_write_target(template.authored_payload(), ".".join(parts))
    document = project.documents.get(target[0]) if target is not None else None
    if not isinstance(document, BasedDocument):
        raise ValueError(f"{field} is not read from a document based on another.")
    base = compose_document(project.documents.get, document.base.document)
    if isinstance(base, ProjectDiagnostic):
        raise ValueError(base.detail)
    value: Any = base.payload
    for part in target[1].split("."):
        if not isinstance(value, dict) or part not in value:
            return [OverridePatch(op="remove", path=".".join(parts))]
        value = value[part]
    return [OverridePatch(op="replace", path=".".join(parts), value=value)]


def training_document_choice(
    project: Project, step_id: str, registry: TrainingMethodRegistry, document: str,
) -> list[OverridePatch]:
    """The writes that point the Train step at the Training document ``document``."""
    entry = compose_document(project.documents.get, document)
    if (
        isinstance(entry, ProjectDiagnostic) or entry.schema_id != TRAINING_DOCUMENT_SCHEMA_ID
        or not isinstance(entry.payload, dict)
    ):
        raise ValueError(f"{document} is not a Training document.")
    definition = _StepFields(project, step_id, registry)
    definition.training(None)
    return [
        OverridePatch(op="replace" if key in definition.authored else "add", path=key,
                      value={"document": document, "pointer": f"/{key}"})
        for key in TRAINING_DOCUMENT_KEYS
    ]


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
        raise ValueError(f"{value!r} is outside the range of {field.label}")
    if field.id == "/settings/grad_clip" and value <= 0:
        raise ValueError(f"{field.label} is {value!r}, not positive")
    return int(value) if control.step == 1 else value
