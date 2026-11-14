"""Compile authored project requests without result access."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import cache, cached_property
from threading import Lock
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

from pydantic import ValidationError
from pydantic_core import to_json

from feedbax.contracts import request_identity
from feedbax.contracts._request_codec import same_typed_value
from feedbax.contracts.base import sha256_bytes
from feedbax.contracts.expression_values import Problem, ValueBase
from feedbax.contracts.figure_document import PLOTLY_FIGURE_ROLE, FigureDocument, PlotlyFigurePort
from feedbax.contracts.intent_diff import first_request_change
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.model_address import ComponentElement, FieldElement, ModelAddress
from feedbax.contracts.model_attachment import MODEL_ATTACHMENTS, MODEL_DIFFERENCE_BATCHES, ModelDifference, SetField
from feedbax.contracts.matrix_core import apply_override_patch_in_place, apply_value_patch_in_place
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.model_values import model_owner_problem
from feedbax.contracts.operation import DatasetPort
from feedbax.contracts.project import Document, InlineBase, Payload, Project, Step, StepBase, StepKind, json_pointer_parts
from feedbax.contracts.project_inputs import (
    BarrierCheckpoint,
    BatchCheckpoint,
    BestCheckpoint,
    CheckpointRule,
    CheckpointBatches,
    CheckpointCadence,
    InputBinding,
    NamedSubset,
    OmittedReference,
    SelectedRow,
    StepReference,
    SubsetSelectionError,
    SubsetTerms,
    producer_member_key,
    select_step_rows,
)
from feedbax.contracts.project_resolution import (
    BaseReaders,
    ProjectDiagnostic,
    ResolutionCode,
    ResolvedStep,
    StepComposition,
    StepTemplate,
    _TrialResolution,
    compose_step,
    _select,
)
from feedbax.contracts.rollout import ROLLOUT_TYPE, RolloutParameters, authored_rollout_settings, resolve_trial_ids
from feedbax.contracts.project_sweep import (
    CHECKPOINT_RULE_TARGET,
    CheckpointRuleAxis,
    DEFAULT_SWEEP_ROW,
    ExpandedRow,
    InputAxis,
    Sweep,
    SweepCoordinate,
    SweepExpansionError,
    SweepInputMember,
    checkpoint_rule_write,
    expand_sweep,
    model_field_target,
    sweep_patch,
)
from feedbax.contracts.project_sources import ValueSource, changed_sources, patch_source, pointer, source_shape
from feedbax.contracts.report_document import ReportDocument
from feedbax.contracts.request_identity import EncodedRequest, RequestDocumentMemo
from feedbax.contracts.request_refs import (
    BarrierMember,
    BatchMember,
    BatchSetMember,
    BestMember,
    BestCandidate,
    BestRank,
    InputMember,
    LastMember,
    ProductRef,
    ProductSelector,
    RequestId,
    RequestKind,
    UpdateCadence,
    UpdateSeriesMember,
)
from feedbax.contracts.request_specs import (
    DocumentTrainingRequestSpec,
    ReportRequestSpec,
)
from feedbax.contracts.studio_training import STUDIO_GRAPH_METHOD, StudioGraphMethodPayload

if TYPE_CHECKING:
    from feedbax.analysis.operations import OperationRegistry
    from feedbax.contracts.request_specs import TrialDocuments
    from feedbax.tasks.registry import PythonTaskRegistry
    from feedbax.component_registry import ComponentRegistry
    from feedbax.expression_registry import FunctionRegistry


@cache
def _builtin_model_registry() -> ComponentRegistry:
    from feedbax.component_registry import ComponentRegistry

    registry = ComponentRegistry()
    registry.seal()
    return registry


@dataclass(frozen=True)
class RunKey:
    variant: str
    step: str
    row: str


@dataclass(frozen=True)
class ElementKey:
    run: RunKey
    name: str

    def payload_path(self, kind: RequestKind) -> str:
        family = "operations" if kind == "analyze" else "views"
        return f"payload.{family}.{self.name}"


@dataclass(frozen=True)
class PlannedRankGroup:
    coordinates: Mapping[str, str]
    ranks: tuple[str, ...]


@dataclass(frozen=True)
class SubsetSource:
    """The authored selector and producer step shared by a step input's runs."""

    origin: ValueSource
    producer_step: str
    rows: tuple[SelectedRow, ...] = ()


@dataclass(frozen=True)
class PlannedSubset:
    terms: SubsetTerms
    candidates: int
    members: tuple[RunKey, ...]
    document: str | None
    rank_groups: tuple[PlannedRankGroup, ...] = ()
    source: SubsetSource | None = None

    @property
    def ranks(self) -> tuple[str, ...]:
        return tuple(rank for group in self.rank_groups for rank in group.ranks)


@dataclass(frozen=True)
class PlannedInput:
    producer: RunKey | None
    member: InputMember
    producer_element: str | None = None
    candidates: tuple[RunKey, ...] = ()

    @property
    def sources(self) -> tuple[RunKey, ...]:
        return (self.producer,) if self.producer is not None else self.candidates

    @property
    def element(self) -> ElementKey | None:
        return ElementKey(self.producer, self.producer_element) if self.producer and self.producer_element else None


def _upstream_source(binding: PlannedInput) -> ValueSource | None:
    producer = binding.producer
    if producer is None:
        return None
    path = ""
    if binding.producer_element is not None and isinstance(binding.member.reference, ProductRef):
        path = pointer(("elements", binding.producer_element, "products", binding.member.reference.role))
    return ValueSource(
        "upstream", producer.step, path, step=producer.step, variant=producer.variant, row=producer.row,
    )


@dataclass(frozen=True)
class PlannedRow:
    """An expanded row, including a row whose request validation fails."""

    key: RunKey
    kind: StepKind
    label: str | None
    coordinates: Mapping[str, SweepCoordinate]
    document_ids: Mapping[str, str] = field(default_factory=dict, kw_only=True)
    document_uses: frozenset[str] = field(default=frozenset(), kw_only=True)


@dataclass(frozen=True)
class RunProblem:
    path: str
    detail: str
    model_address: ModelAddress | None = None


@dataclass(frozen=True)
class PlannedRun(PlannedRow):
    request_id: RequestId
    inputs: Mapping[str, PlannedInput]
    omissions: Mapping[str, str]
    requires: tuple[str, ...]
    admission_problem: RunProblem | None = None


@dataclass(frozen=True)
class PlannedElement:
    key: ElementKey
    kind: Literal["analyze", "render"]
    request_id: RequestId
    inputs: Mapping[str, PlannedInput]
    outputs: Mapping[str, DatasetPort | PlotlyFigurePort]

    @property
    def datasets(self) -> Mapping[str, DatasetPort]:
        return MappingProxyType({
            role: port for role, port in self.outputs.items() if isinstance(port, DatasetPort)
        })

    @property
    def path(self) -> str:
        return self.key.payload_path(self.kind)


@dataclass(frozen=True)
class PlannedAnalyzeRun(PlannedRow):
    elements: Mapping[str, PlannedElement]
    requires: tuple[str, ...]
    states: PlannedInput
    figures: Mapping[str, FigureDocument]


@dataclass(frozen=True)
class ElementChange:
    """An authored change to a singleton run or element request."""

    state: Literal["unchanged", "changed", "added", "removed"]
    path: str | None = None
    upstream: ElementKey | None = None
    before_source: ValueSource | None = None
    after_source: ValueSource | None = None


@dataclass(frozen=True)
class PlannedStep:
    """Local inputs, resolved bindings, and the sweep projected for authoring."""

    kind: StepKind
    group: str | None
    authored_inputs: Mapping[str, InputBinding | None]
    inputs: Mapping[str, InputBinding] | None = None
    sweep: Sweep | None = field(default=None, compare=False)


@dataclass(frozen=True)
class SelectedRunDefinition:
    payload: Payload
    sources: Mapping[str, ValueSource]
    document_ids: Mapping[str, str]
    document_uses: frozenset[str]
    document_pointers: Mapping[str, str] = field(default_factory=dict)
    document_problems: tuple[ProjectDiagnostic, ...] = ()


@dataclass(frozen=True)
class _TrainEvaluationSource:
    request: bytes
    documents: tuple[tuple[str, bytes], ...]
    task: tuple[str, ...] | bytes


@dataclass(frozen=True)
class SourceSweep:
    inputs: dict[str, tuple[SweepInputMember, ...]] = field(default_factory=dict)
    integer_sites: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceSnapshot:
    project: Project
    variant_sites: dict[str, bool]
    sweeps: dict[tuple[str, str], SourceSweep]


@dataclass(frozen=True)
class PlanSourceIndex:
    snapshot: SourceSnapshot
    producers: Mapping[RunKey | ElementKey, Mapping[str, ValueSource]] = field(default_factory=dict)
    externals: Mapping[str, bytes] = field(default_factory=dict)

    @cached_property
    def _resolver(self) -> _Compiler:
        readers = BaseReaders(external=self.externals.get, figure_external=self.externals.get)
        return _Compiler(self.snapshot.project, readers, ())

    @cached_property
    def _templates(self) -> Callable[[str, str], tuple[StepTemplate, dict[str, ExpandedRow]]]:
        @cache
        def template(variant: str, step: str):
            compiler = self._resolver
            if variant not in compiler.variants:
                compiler.source_variants = frozenset({variant})
                compiler.expand_variants(self.snapshot.variant_sites)
            project = compiler.variants[variant]
            template = compose_step(
                project, step, readers=compiler.readers, track_sources=True,
                variant_sources=compiler.variant_sources.get(variant),
            )
            if isinstance(template, ProjectDiagnostic):
                raise ValueError(f"Cannot resolve saved source for {variant}/{step}: {template.detail}")
            sweep = self.snapshot.sweeps.get((variant, step), SourceSweep())
            rows = expand_sweep(
                template.sweep, inputs=sweep.inputs, integer_site=sweep.integer_sites.__getitem__,
                checkpoint_subject=project.steps[step].kind == "evaluate",
            )
            return template, {row.id: row for row in rows}
        return template

    def source(self, key: RunKey | ElementKey, parts: tuple[str, ...]) -> ValueSource | None:
        if not parts:
            return None
        if parts[0] == "inputs":
            return self.producers.get(key, {}).get(parts[1]) if len(parts) == 2 else None
        run = key.run if isinstance(key, ElementKey) else key
        try:
            template, rows = self._templates(run.variant, run.step)
            row = rows[run.row]
            if (
                parts[:3] == ("payload", "trial_documents", "model")
                and self._resolver.variants[run.variant].steps[run.step].kind == "train"
                and template.has_model_differences(row)
            ):
                return self._model_field_source(template, run, row, parts)
            return template.source_leaf(row, parts[1:])
        except (KeyError, ValueError) as error:
            raise ValueError(f"Invalid saved source for {run.variant}/{run.step}/{run.row}: {error}") from error

    def _model_field_source(
        self, template: StepTemplate, run: RunKey, row: ExpandedRow, parts: tuple[str, ...],
    ) -> ValueSource | None:
        definition = self._resolver.project_definition(template.composition, run, row, {})
        if definition is None:
            raise ValueError("Cannot resolve the managed Model sources")
        payload, model = self._resolver.lower_train_model(
            run, definition.payload, definition.document_ids, definition.document_pointers,
        )
        if isinstance(payload, ProjectDiagnostic):
            raise ValueError(payload.detail)
        if model is not None:
            definition = _train_model_definition(definition, model)
        return definition.sources.get(pointer(parts))


@dataclass(frozen=True)
class ProjectPlan:
    """Named runs and deduplicated objects encoded by the shared request codec."""

    variants: tuple[str, ...]
    steps: Mapping[tuple[str, str], PlannedStep]
    rows: tuple[PlannedRow, ...]
    runs: tuple[PlannedRun, ...]
    requests: Mapping[RequestId, bytes]
    documents: Mapping[str, bytes]
    diagnostics: tuple[ProjectDiagnostic, ...]
    subsets: Mapping[tuple[str, str, str], PlannedSubset]
    selected_definitions: Mapping[RunKey, SelectedRunDefinition]
    _train_evaluation_sources: Mapping[RequestId, _TrainEvaluationSource] = field(
        default_factory=lambda: MappingProxyType({}), repr=False,
    )
    source_snapshot: SourceSnapshot | None = field(default=None, repr=False, compare=False)
    exception_matches: Mapping[tuple[str, str], Mapping[str, tuple[str, ...]]] = field(default_factory=dict)

    def train_evaluation_source(self, request_id: RequestId) -> tuple[str, ...] | bytes | None:
        """Read the compiled Task facts only for their exact request and documents."""
        source = self._train_evaluation_sources.get(request_id)
        if source is not None and self.requests.get(request_id) == source.request and all(
            self.documents.get(digest) == data for digest, data in source.documents
        ):
            return source.task
        return None

    @property
    def elements(self) -> Mapping[ElementKey, PlannedElement]:
        return MappingProxyType({
            element.key: element
            for row in self.rows if isinstance(row, PlannedAnalyzeRun)
            for element in row.elements.values()
        })

    def document_users(self, variant: str, document: str) -> frozenset[str]:
        return frozenset(
            row.key.step for row in self.rows if row.key.variant == variant and document in row.document_uses
        )

    @property
    def sources(self) -> Mapping[RunKey, Mapping[str, ValueSource]]:
        return MappingProxyType(
            {key: definition.sources for key, definition in self.selected_definitions.items()}
        )


@dataclass(frozen=True)
class _Role:
    name: str
    required: bool
    producer_kind: StepKind
    product_role: str


_ROLES = {
    "train": _Role("checkpoint", False, "train", "checkpoint"),
    "evaluate": _Role("subject", True, "train", "checkpoint"),
    "analyze": _Role("states", True, "evaluate", "states"),
}

type _CompiledRun = PlannedRun | PlannedAnalyzeRun
type _Members = tuple[tuple[PlannedInput, _CompiledRun], ...]


def _checkpoint(rule: CheckpointRule | None, *, horizon: int | None = None) -> ProductSelector | None:
    match rule:
        case None:
            return None
        case "last":
            return LastMember()
        case BestCheckpoint(best_by=metric):
            return BestMember(metric=metric)
        case BatchCheckpoint(at=batch):
            return BatchMember(batch=batch)
        case CheckpointBatches(batches=batches):
            return BatchSetMember(batches=batches)
        case CheckpointCadence():
            return BatchSetMember(batches=rule.coordinates(), cadence=rule)
        case tuple():
            return UpdateSeriesMember(rule=rule, updates=tuple(point.at for point in rule))
        case UpdateCadence():
            if horizon is None:
                raise ValueError("an update cadence requires a planned Train definition")
            return UpdateSeriesMember(rule=rule, updates=rule.coordinates(horizon))
        case BarrierCheckpoint(barrier=name):
            return BarrierMember(name=name)


def _train_checkpoint_horizon(train: object) -> int | None:
    if not isinstance(train, DocumentTrainingRequestSpec):
        return None
    if train.method_ref == STUDIO_GRAPH_METHOD:
        return StudioGraphMethodPayload.model_validate(train.method_payload.payload).target_total
    return train.training_config.n_batches


def _paired_checkpoint_step(step: Step, reference: InputBinding | None) -> str | None:
    if (
        step.kind == "train"
        and isinstance(step.base, StepBase)
        and step.base.row is None
        and isinstance(reference, StepReference)
        and reference.step == step.base.step
        and reference.row is None
        and reference.variants is None
    ):
        return reference.step
    return None


type _DocumentStamp = tuple[str, str | None]


class _AuthoredInputs:
    def __init__(self, project: Project):
        self.project = project
        self._documents: dict[str, str | None] = {}
        self._steps: dict[str, str] = {}

    def document_stamp(self, name: str) -> str | None:
        if name not in self._documents:
            document = self.project.documents.get(name)
            self._documents[name] = (
                sha256_bytes(document.model_dump_json(exclude_unset=True).encode())
                if document is not None else None
            )
        return self._documents[name]

    def step_chain(self, name: str) -> tuple[tuple[str, str | None], ...]:
        chain = {}
        while name not in chain:
            step = self.project.steps.get(name)
            if step is None:
                chain[name] = None
                break
            if name not in self._steps:
                self._steps[name] = sha256_bytes(
                    step.model_dump_json(exclude_unset=True).encode()
                )
            chain[name] = self._steps[name]
            if not isinstance(step.base, StepBase):
                break
            name = step.base.step
        return tuple(chain.items())


@dataclass(frozen=True)
class _WorkEntry[T]:
    stamp: tuple[object, ...]
    documents: tuple[_DocumentStamp, ...]
    result: T


def _work[K, T](
    previous: Mapping[K, _WorkEntry[T]],
    current: dict[K, _WorkEntry[T]],
    key: K,
    stamp: tuple[object, ...],
    inputs: _AuthoredInputs,
    compute: Callable[[Callable[[str], Document | None], T | None], T],
) -> T:
    entry = previous.get(key)
    if entry is not None and entry.stamp == stamp and all(
        inputs.document_stamp(name) == digest for name, digest in entry.documents
    ):
        current[key] = entry
        return entry.result
    dependencies = {}

    def read_document(name: str) -> Document | None:
        dependencies[name] = inputs.document_stamp(name)
        return inputs.project.documents.get(name)

    result = compute(read_document, entry.result if entry is not None else None)
    current[key] = _WorkEntry(stamp, tuple(dependencies.items()), result)
    return result


@dataclass(frozen=True)
class _ModelSource:
    document: str
    model: ModelDocument
    stamp: str
    evaluation_problem: ProjectDiagnostic | None = None
    evaluation_graph: GraphSpec | None = None


def _model_differences(payload: Payload) -> tuple[Payload, tuple[ModelDifference, ...]]:
    if MODEL_ATTACHMENTS not in payload:
        return payload, ()
    batches = MODEL_DIFFERENCE_BATCHES.validate_python(payload[MODEL_ATTACHMENTS])
    return (
        {name: value for name, value in payload.items() if name != MODEL_ATTACHMENTS},
        tuple(item for batch in batches.values() for item in batch),
    )


def _authored_variation_problem(problem: ProjectDiagnostic, names: tuple[str, ...]) -> ProjectDiagnostic:
    parts = json_pointer_parts(problem.path) if problem.path.startswith("/") else ()
    if len(parts) >= 3 and parts[:2] == ("params", "variation") and parts[2].isdigit():
        index = int(parts[2])
        if index < len(names):
            return replace(problem, path=pointer((*parts[:2], names[index], *parts[3:])))
    return problem


def _raw_graph_target(value: object) -> bool:
    return isinstance(value, str) and (value == "graph" or value.startswith("graph."))


def _authored_graph_variation_problem(key: RunKey, payload: Payload) -> ProjectDiagnostic | None:
    params = payload.get("params")
    variation = params.get("variation") if isinstance(params, dict) else None
    entries = (
        sorted(variation.items()) if isinstance(variation, dict)
        else enumerate(variation) if isinstance(variation, list) else ()
    )
    for name, edit in entries:
        if isinstance(edit, dict) and _raw_graph_target(edit.get("path")):
            return ProjectDiagnostic(
                ResolutionCode.AUTHORED_GRAPH_VARIATION,
                pointer(("params", "variation", str(name), "path")),
                "Use an addressed Model operation for this saved Evaluate edit.",
                step=key.step, row=key.row, variant=key.variant,
            )
    return None


def _with_train_model(
    payload: Payload, model: _ModelSource, document_ids: Mapping[str, str],
    document_pointers: Mapping[str, str],
) -> Payload:
    data = model.model.model_dump(mode="json")
    draft = deepcopy(payload)
    main = ("trial_documents", "model")
    apply_value_patch_in_place(draft, main, "replace", data, path=pointer(main))
    for slot, name in document_ids.items():
        if name != model.document or not slot.startswith("payload."):
            continue
        path = tuple(slot.removeprefix("payload.").split("."))
        if path != main:
            selected = _select(data, json_pointer_parts(document_pointers.get(slot)))
            apply_value_patch_in_place(draft, path, "replace", selected, path=slot)
    return draft


def _train_model_definition(definition: SelectedRunDefinition, model: _ModelSource) -> SelectedRunDefinition:
    payload = _with_train_model(definition.payload, model, definition.document_ids, definition.document_pointers)
    prefix = ("trial_documents", "model")
    sources = dict(definition.sources)
    batches = definition.payload.get(MODEL_ATTACHMENTS, {})
    for batch, differences in batches.items():
        for index, difference in enumerate(differences):
            if difference["op"] != "set_field":
                continue
            address = ModelAddress.model_validate(difference["address"])
            source = sources.get(pointer(("payload", MODEL_ATTACHMENTS, batch, str(index), "value")))
            if source is None or address.element.parent.kind == "attachment":
                continue
            field = address.stored_path()
            for slot, name in definition.document_ids.items():
                if name != model.document or not slot.startswith("payload."):
                    continue
                root = json_pointer_parts(definition.document_pointers.get(slot))
                target = tuple(slot.split("."))
                if field[:len(root)] == root:
                    path = pointer((*target, *field[len(root):]))
                    sources[path] = source
                    if path + "/value" in sources:
                        sources[path + "/value"] = source
                elif root == (*field, "value"):
                    sources[pointer(target)] = source
            if not definition.document_ids.get("payload." + ".".join(prefix)):
                sources[pointer(("payload", *prefix, *field))] = source
    return replace(definition, payload=payload, sources=MappingProxyType(sources))


def _model_difference_problem(
    key: RunKey, address: ModelAddress | None, detail: str, *, path: str | None = None,
) -> ProjectDiagnostic:
    return ProjectDiagnostic(
        ResolutionCode.INVALID_PAYLOAD, path or f"payload.{MODEL_ATTACHMENTS}", detail, key.step,
        row=key.row, variant=key.variant, document=address.document if address else None, model_address=address,
    )


def _sweep_problem(
    error: SweepExpansionError, *, step: str | None = None, variant: str | None = None,
) -> ProjectDiagnostic:
    address = None
    if error.target is not None:
        try:
            target = model_field_target(error.target)
            address = target.address if target is not None else None
        except ValueError:
            pass  # The sweep diagnostic already reports the malformed target.
    return ProjectDiagnostic(
        ResolutionCode.INVALID_SWEEP, error.path, error.detail, step=step, variant=variant,
        document=address.document if address is not None else None, model_address=address,
    )


def _integer_model_site(
    document: str, model: ModelDocument, address: ModelAddress, components: ComponentRegistry,
) -> bool:
    from feedbax.contracts.model_document import CompositeComponent, LeafComponent

    element = address.element
    if document != address.document or not isinstance(element, FieldElement) or not isinstance(element.parent, ComponentElement):
        return False
    graph = model
    for name in address.composites:
        component = graph.interior.components.get(name)
        if not isinstance(component, CompositeComponent):
            return False
        graph = component
    component = graph.interior.components.get(element.parent.id)
    path = element.path
    if isinstance(component, CompositeComponent):
        return path == ("substeps",)
    if not isinstance(component, LeafComponent) or len(path) != 2 or path[0] != "fields":
        return False
    declaration = components.get(component.type)
    return declaration is not None and any(
        parameter.name == path[1] and parameter.type == "int" for parameter in declaration.params.param_schema
    )


def _integer_train_site(path: str, payload: Payload, python_tasks: PythonTaskRegistry | None) -> bool:
    from feedbax.tasks.reaching import builtin_python_tasks
    from feedbax.tasks.registry import python_task_parameter_site

    documents = payload.get("trial_documents")
    site = python_task_parameter_site(path, documents, python_tasks or builtin_python_tasks())
    if site is not None:
        return site.expected.base is ValueBase.INTEGER and not site.expected.shape
    annotation = DocumentTrainingRequestSpec
    for segment in path.split("."):
        fields = getattr(annotation, "model_fields", {})
        if segment not in fields:
            return False
        annotation = fields[segment].annotation
        args = get_args(annotation)
        if len(args) == 2 and type(None) in args:
            annotation = next(arg for arg in args if arg is not type(None))
    return annotation is int


@dataclass(frozen=True)
class _LoweredRow:
    request: EncodedRequest | None
    diagnostics: tuple[ProjectDiagnostic, ...]
    encoding_stamp: tuple[StepKind, str, bytes] | None = None
    admission_problem: RunProblem | None = None
    document_ids: Mapping[str, str] = field(default_factory=dict)
    document_uses: frozenset[str] = frozenset()
    model_source: _ModelSource | None = None
    evaluation_source: tuple[str, ...] | bytes | None = None


@dataclass(frozen=True)
class _TrainPreparation:
    checkpoint_problem: RunProblem | None
    evaluation: GraphSpec | ProjectDiagnostic
    condition_ids: tuple[str, ...]


@dataclass(frozen=True)
class _PreparedTrainRow:
    documents: TrialDocuments
    preparation: _TrainPreparation
    model: _ModelSource


@dataclass
class _TrainRows:
    resolution: _TrialResolution = field(default_factory=_TrialResolution)
    prepared: tuple[Payload, _PreparedTrainRow] | None = None


@cache
def _builtin_expression_registry():
    from feedbax.expression_builtins import register_builtins
    from feedbax.expression_registry import FunctionRegistry

    registry = FunctionRegistry()
    register_builtins(registry)
    registry.seal()
    return registry


def _encode_row(
    key: RunKey,
    kind: StepKind,
    payload: Payload,
    inputs: Mapping[str, PlannedInput],
    encoding_stamp: tuple[StepKind, str, bytes] | None,
    document_memo: RequestDocumentMemo,
    prepare_trial: Callable[[TrialDocuments], _TrainPreparation | tuple[Problem, ...]],
    python_tasks: PythonTaskRegistry | None = None,
    model_source: _ModelSource | None = None,
    prepared_train: _PreparedTrainRow | None = None,
    remember_train: Callable[[_PreparedTrainRow], None] | None = None,
) -> _LoweredRow:
    checkpoint_problem = None
    condition_ids = None
    if prepared_train is not None:
        payload = payload | {"trial_documents": prepared_train.documents}
        checkpoint_problem = prepared_train.preparation.checkpoint_problem
        condition_ids = prepared_train.preparation.condition_ids
        model_source = prepared_train.model
    elif kind == "train":
        from feedbax.compiler.trial_execution import validate_python_task
        from feedbax.tasks.reaching import builtin_python_tasks
        from feedbax.contracts.request_specs import TrialDocuments

        try:
            authored = TrialDocuments.model_validate(payload.get("trial_documents"))
            if authored.python_task is not None:
                resolved = validate_python_task(authored.python_task, python_tasks or builtin_python_tasks())
                if isinstance(resolved[0], Problem):
                    return _LoweredRow(None, tuple(ProjectDiagnostic(
                        ResolutionCode.INVALID_PAYLOAD,
                        ".".join(("payload.trial_documents", *map(str, problem.path))),
                        problem.reason, key.step, row=key.row, variant=key.variant,
                    ) for problem in resolved), encoding_stamp)
                _, params = resolved
                authored = authored.model_copy(update={"python_task": authored.python_task.model_copy(
                    update={"params": params.model_dump(mode="json")},
                )})
                payload = payload | {"trial_documents": authored.model_dump(mode="json")}
            preparation = prepare_trial(authored)
        except ValueError as error:
            return _LoweredRow(None, (ProjectDiagnostic(
                ResolutionCode.INVALID_PAYLOAD, "payload.trial_documents", str(error), key.step,
                row=key.row, variant=key.variant,
            ),), encoding_stamp)
        if isinstance(preparation, tuple):
            return _LoweredRow(None, tuple(ProjectDiagnostic(
                ResolutionCode.INVALID_PAYLOAD,
                ".".join(("payload.trial_documents", *map(str, problem.path))),
                problem.reason, key.step, row=key.row, variant=key.variant,
                document=problem.model_address.document if problem.model_address else None,
                model_address=problem.model_address,
            ) for problem in preparation), encoding_stamp)
        checkpoint_problem = preparation.checkpoint_problem
        condition_ids = preparation.condition_ids
        assert model_source is not None
        if isinstance(preparation.evaluation, ProjectDiagnostic):
            model_source = replace(model_source, evaluation_problem=preparation.evaluation)
        else:
            model_source = replace(model_source, evaluation_graph=preparation.evaluation)
    try:
        if kind == "report":
            document = ReportDocument.model_validate(payload)
            expected = set(document.figure_roles)
            invalid = [
                (role, "required figure input is missing") for role in sorted(expected - inputs.keys())
            ] + [
                (role, "unknown Report figure role") for role in sorted(inputs.keys() - expected)
            ]
            if invalid:
                return _LoweredRow(
                    None,
                    tuple(ProjectDiagnostic(
                        ResolutionCode.BAD_REFERENCE, f"inputs.{role}", detail, key.step,
                        row=key.row, variant=key.variant,
                    ) for role, detail in invalid),
                    encoding_stamp,
                )
            spec = ReportRequestSpec(report_type=None, document=document)
        else:
            spec = request_identity.REQUEST_KINDS[kind].spec_type.model_validate(payload)
        encoded = request_identity.encode_request(
            spec,
            {role: (binding.member,) for role, binding in inputs.items()},
            document_memo=document_memo,
            capture_train_task=kind == "train",
        )
    except ValidationError as error:
        problems = tuple(
            (".".join(map(str, failure["loc"])), failure["msg"])
            for failure in error.errors(include_url=False, include_input=False)
        )
    except ValueError as error:
        problems = (("", str(error)),)
    else:
        if remember_train is not None:
            assert isinstance(spec, DocumentTrainingRequestSpec) and model_source is not None
            remember_train(_PreparedTrainRow(spec.trial_documents, preparation, model_source))
        evaluation_source = None
        problem = (
            RunProblem("/trial_documents/model/interior/components", "Model has no components.")
            if isinstance(spec, DocumentTrainingRequestSpec) and not spec.trial_documents.model.interior.components else
            RunProblem("/trial_documents/objective/terms", "Objective has no enabled terms.")
            if isinstance(spec, DocumentTrainingRequestSpec) and not any(
                term.enabled for term in spec.trial_documents.objective.terms
            ) else checkpoint_problem
        )
        if problem is None and encoded.train_task is not None:
            selection = encoded.train_task.python_selection
            evaluation_source = selection if selection is not None else condition_ids
        return _LoweredRow(
            encoded, (), encoding_stamp, admission_problem=problem,
            model_source=model_source, evaluation_source=evaluation_source,
        )
    return _LoweredRow(
        None,
        tuple(
            ProjectDiagnostic(
                ResolutionCode.INVALID_PAYLOAD,
                f"payload.{path}" if path else "payload",
                message,
                key.step,
                row=key.row,
                variant=key.variant,
            )
            for path, message in problems
        ),
        encoding_stamp,
    )


@dataclass
class _PlanWork:
    compositions: dict[
        tuple[str, str, bool], _WorkEntry[StepComposition | ProjectDiagnostic]
    ] = field(default_factory=dict)
    sweeps: dict[
        tuple[str, str], _WorkEntry[tuple[tuple[ExpandedRow, ...] | ProjectDiagnostic, dict[str, tuple[str, ...]]]]
    ] = field(default_factory=dict)
    rows: dict[RunKey, _WorkEntry[_LoweredRow]] = field(default_factory=dict)
    definitions: dict[RunKey, _WorkEntry[SelectedRunDefinition | None]] = field(default_factory=dict)
    inline_origins: OrderedDict[ValueSource, tuple[str, Any]] = field(default_factory=OrderedDict)
    request_documents: RequestDocumentMemo = field(default_factory=RequestDocumentMemo)


class ProjectPlanMemo:
    """Process-local compiler work for one project's most recent read."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._work = _PlanWork()
        self._authority: tuple[object, ...] = ()
        self._started = 0
        self._published = 0

    def _begin(self, authority: tuple[object, ...]) -> tuple[int, _PlanWork]:
        with self._lock:
            self._started += 1
            return self._started, self._work if self._authority == authority else _PlanWork()

    def _publish(self, ticket: int, authority: tuple[object, ...], work: _PlanWork) -> None:
        with self._lock:
            if ticket > self._published:
                self._published = ticket
                self._authority = authority
                self._work = work


def _compiler_authority(
    components: ComponentRegistry | None, operations: OperationRegistry | None,
    functions: FunctionRegistry | None, python_tasks: PythonTaskRegistry | None,
) -> tuple[object, ...]:
    return (
        compose_step, StepTemplate.resolve, expand_sweep, source_shape, tuple(_ROLES.items()),
        request_identity.request_codec_authority(), components, operations, functions, python_tasks,
    )


class _Compiler:
    def __init__(
        self,
        project: Project,
        readers: BaseReaders,
        source_runs: Iterable[RunKey],
        previous: _PlanWork | None = None,
        operations: OperationRegistry | None = None,
        components: ComponentRegistry | None = None,
        functions: FunctionRegistry | None = None,
        python_tasks: PythonTaskRegistry | None = None,
        *,
        capture_sources: bool = False,
    ):
        self.project = project
        self.readers = readers
        self.previous = previous
        self.operations = operations
        self.components = components
        self.functions = functions
        self.python_tasks = python_tasks
        self.source_sweeps: dict[tuple[str, str], SourceSweep] | None = {} if capture_sources else None
        self.variant_sites: dict[str, bool] = {}
        self.source_variants: frozenset[str] = frozenset()
        self.trial_preparations: dict[bytes, _TrainPreparation | tuple[Problem, ...]] = {}
        self.work = _PlanWork()
        if previous is not None:
            self.work.inline_origins.update(previous.inline_origins)
            self.work.request_documents = previous.request_documents.fork()
        self.wanted_sources: dict[tuple[str, str], set[str]] = {}
        for key in source_runs:
            self.wanted_sources.setdefault((key.variant, key.step), set()).add(key.row)
        self.variants: dict[str, Project] = {}
        self.variant_order: tuple[str, ...] = ()
        self.variant_sources: dict[str, Any] = {}
        self.variant_rows: dict[str, ExpandedRow] = {}
        self.variant_stamps: dict[str, str] = {}
        self.authored_inputs: dict[str, _AuthoredInputs] = {}
        self.steps: dict[tuple[str, str], PlannedStep] = {}
        self.rows: dict[RunKey, PlannedRow] = {}
        self.excluded_checkpoint_rows: set[RunKey] = set()
        self.runs: dict[tuple[str, str], tuple[_CompiledRun, ...]] = {}
        self.swept_step_axes: dict[tuple[str, str], tuple[str, ...]] = {}
        self.requests: dict[RequestId, bytes] = {}
        self.documents: dict[str, bytes] = {}
        self.diagnostics: list[ProjectDiagnostic] = []
        self.subsets: dict[tuple[str, str, str], PlannedSubset] = {}
        self.exception_matches: dict[tuple[str, str], Mapping[str, tuple[str, ...]]] = {}
        self.selected_definitions: dict[RunKey, SelectedRunDefinition] = {}
        self.train_evaluation_sources: dict[RequestId, _TrainEvaluationSource] = {}
        self.model_sources: dict[RunKey, _ModelSource] = {}
        self.checkpoint_horizons: dict[RequestId, int | None] = {}

    def expand_variants(self, known_sites: Mapping[str, bool] | None = None) -> None:
        def integer_site(path: str) -> bool:
            if (target := model_field_target(path)) is not None:
                from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID

                address = target.address
                if target.step is not None:
                    step = self.project.steps.get(target.step)
                    if step is None or step.kind != "train":
                        return False
                    template = compose_step(self.project, target.step, readers=self.readers)
                    if isinstance(template, ProjectDiagnostic):
                        return False
                    resolved = template.resolve()
                    if isinstance(resolved, ProjectDiagnostic):
                        return False
                    _, source = self.lower_train_model(
                        RunKey(DEFAULT_SWEEP_ROW, target.step, DEFAULT_SWEEP_ROW),
                        resolved.payload, resolved.document_ids, resolved.document_pointers,
                    )
                    return source is not None and _integer_model_site(
                        source.document, source.model, address, self.components or _builtin_model_registry(),
                    )
                entry = self.project.documents.get(address.document)
                return entry is not None and entry.schema_id == MODEL_DOCUMENT_SCHEMA_ID and _integer_model_site(
                    address.document, ModelDocument.model_validate(entry.payload), address,
                    self.components or _builtin_model_registry(),
                )
            parts = path.split(".", 4)
            if len(parts) == 5 and parts[0] == "steps" and parts[2:4] == ["base", "inline"]:
                step = self.project.steps.get(parts[1])
                if step is not None and step.kind == "train" and isinstance(step.base, InlineBase):
                    return _integer_train_site(parts[4], step.base.inline, self.python_tasks)
            return False

        def capture_site(path: str) -> bool:
            self.variant_sites[path] = answer = integer_site(path)
            return answer

        try:
            variants = expand_sweep(
                self.project.sweep,
                integer_site=known_sites.__getitem__ if known_sites is not None
                else capture_site if self.source_sweeps is not None else integer_site,
            )
        except SweepExpansionError as error:
            self.diagnostics.append(_sweep_problem(error))
            return
        self.variant_order = tuple(row.id for row in variants)
        for row in variants:
            if self.source_variants and row.id not in self.source_variants:
                continue
            self.variant_rows[row.id] = row
            if not row.patches:
                self.variants[row.id] = self.project
                continue
            data = self.project.model_dump(mode="json", by_alias=True, exclude_unset=True)
            data.pop("sweep", None)
            data.setdefault("documents", {})
            data.setdefault("steps", {})
            track = row.id in self.source_variants or any(variant == row.id for variant, _ in self.wanted_sources)
            if track and self.previous is not None:
                self.variant_stamps[row.id] = sha256_bytes(to_json(tuple(row.patches.items())))
            origins = source_shape(data, None) if track else None
            path = "sweep"
            address = None
            try:
                from feedbax.compiler.model_attachment import resolve_effective_model
                from feedbax.compiler.model_document import ModelCompilationProblem
                from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID

                for path, write in row.patches.items():
                    address = None
                    patch = sweep_patch(path, write)
                    if isinstance(patch, SetField):
                        address = patch.address
                        target = model_field_target(path)
                        if target.step is not None:
                            step = data["steps"].get(target.step)
                            if step is None or step["kind"] not in {"train", "evaluate"}:
                                raise ValueError("The addressed Train or Evaluate step does not exist")
                            batches = step.setdefault(MODEL_ATTACHMENTS, {})
                            batch = "variant-fields"
                            while batch in batches:
                                batch += "-"
                            records = [patch.model_dump(mode="json")]
                            batches[batch] = records
                            if origins is not None:
                                step_path = f"steps.{target.step}.{MODEL_ATTACHMENTS}"
                                if len(batches) == 1:
                                    origins = patch_source(origins, OverridePatch(
                                        path=step_path, op="add", value={},
                                    ), source_shape({}, None))
                                source = ValueSource("variant", row.id, pointer(("sweep", path)), variant=row.id)
                                origins = patch_source(origins, OverridePatch(
                                    path=f"{step_path}.{batch}", op="add", value=records,
                                ), source_shape(records, source))
                            continue
                        entry = data["documents"].get(address.document)
                        if entry is None or entry["schema"] != MODEL_DOCUMENT_SCHEMA_ID:
                            raise ValueError("The addressed Model document does not exist")
                        before = entry["payload"]
                        entry["payload"] = resolve_effective_model(
                            address.document, ModelDocument.model_validate(before),
                            self.components or _builtin_model_registry(), (patch,), functions=self.functions,
                        ).model_dump(mode="json")
                        if origins is not None:
                            source = ValueSource("variant", row.id, pointer(("sweep", path)), variant=row.id)
                            authored = origins["documents"][address.document]
                            authored["payload"] = changed_sources(before, entry["payload"], authored["payload"], source)
                        continue
                    parts = tuple(path.split("."))
                    if len(parts) < 2 or parts[0] not in {"documents", "steps"}:
                        raise ValueError("a variant must patch a named document or step")
                    if len(parts) >= 4 and parts[0] == "steps" and parts[2] == MODEL_ATTACHMENTS:
                        step = data["steps"].get(parts[1])
                        if step is not None and MODEL_ATTACHMENTS not in step:
                            step[MODEL_ATTACHMENTS] = {}
                            if origins is not None:
                                origins = patch_source(origins, OverridePatch(
                                    path=".".join(parts[:3]), op="add", value={},
                                ), source_shape({}, None))
                    apply_override_patch_in_place(data, patch)
                    if origins is not None:
                        source = ValueSource("variant", row.id, pointer(parts), variant=row.id)
                        origins = patch_source(origins, patch, source_shape(patch.value, source))
                self.variants[row.id] = Project.model_validate(data)
                self.variant_sources[row.id] = origins
            except ValueError as error:
                if isinstance(error, ModelCompilationProblem):
                    address = error.address
                self.diagnostics.append(
                    ProjectDiagnostic(
                        ResolutionCode.INVALID_SWEEP, path, str(error), variant=row.id,
                        document=address.document if address else None, model_address=address,
                    )
                )

    def plan(self) -> ProjectPlan:
        self.expand_variants()
        if self.previous is not None:
            self.authored_inputs.update(
                (variant, _AuthoredInputs(project)) for variant, project in self.variants.items()
            )
        for variant, project in self.variants.items():
            for name in sorted(project.steps):
                self.step(variant, name, ())
        return ProjectPlan(
            self.variant_order,
            MappingProxyType(self.steps),
            tuple(self.rows.values()),
            tuple(run for runs in self.runs.values() for run in runs if isinstance(run, PlannedRun)),
            MappingProxyType(self.requests),
            MappingProxyType(self.documents),
            tuple(self.diagnostics),
            MappingProxyType(self.subsets),
            MappingProxyType(self.selected_definitions),
            MappingProxyType(self.train_evaluation_sources),
            SourceSnapshot(self.project.model_copy(deep=True), self.variant_sites, self.source_sweeps)
            if self.source_sweeps is not None else None,
            MappingProxyType(self.exception_matches),
        )

    def step(
        self, variant: str, name: str, active: tuple[tuple[str, str], ...]
    ) -> tuple[_CompiledRun, ...]:
        key = variant, name
        if key not in self.runs:
            self.runs[key] = self.compile_step(variant, name, (*active, key))
        return self.runs[key]

    def inline_origin(self, payload: Payload, source: ValueSource) -> Any:
        if self.previous is None:
            return source_shape(payload, source)
        cache = self.work.inline_origins
        stamp = sha256_bytes(to_json(payload))
        entry = cache.get(source)
        if entry is None or entry[0] != stamp:
            entry = stamp, source_shape(payload, source)
            cache[source] = entry
        cache.move_to_end(source)
        if len(cache) > 16:
            cache.popitem(last=False)
        return entry[1]

    def compose(self, variant: str, name: str, track: bool) -> StepComposition | ProjectDiagnostic:
        project = self.variants[variant]

        def compose(read_document=None, _previous=None):
            template = compose_step(
                project,
                name,
                readers=self.readers,
                track_sources=track,
                variant_sources=self.variant_sources.get(variant),
                read_document=read_document,
                inline_origin=self.inline_origin,
            )
            return template if isinstance(template, ProjectDiagnostic) else template.composition

        if self.previous is None:
            return compose()
        inputs = self.authored_inputs[variant]
        other_mode = (variant, name, not track)
        if (entry := self.previous.compositions.get(other_mode)) is not None:
            self.work.compositions[other_mode] = entry
        return _work(
            self.previous.compositions,
            self.work.compositions,
            (variant, name, track),
            (inputs.step_chain(name), self.variant_stamps.get(variant) if track else None),
            inputs,
            compose,
        )

    def expand_rows(
        self,
        variant: str,
        name: str,
        composition: StepComposition,
        members: Mapping[str, _Members],
    ) -> tuple[ExpandedRow, ...] | ProjectDiagnostic:
        kind = self.variants[variant].steps[name].kind
        checkpoint_axis = kind == "evaluate" and composition.sweep is not None and any(
            isinstance(axis, CheckpointRuleAxis) for axis in composition.sweep.axes.values()
        )
        fanout = {
            role: tuple(
                SweepInputMember(
                    item.member.key,
                    producer.coordinates if item.producer is not None else {
                        axis: producer.coordinates[axis] for axis in item.member.reference.group
                    } if isinstance(item.member.reference, BestRank) else {},
                    producer.label if item.producer is not None else item.member.key,
                    self.checkpoint_horizon(producer.request_id)
                    if checkpoint_axis and role == "subject" and isinstance(producer, PlannedRun) else None,
                )
                for item, producer in bindings
            )
            for role, bindings in members.items()
        }
        subjects = tuple(
            self.model_sources.get(key)
            for item, _ in members.get("subject", ())
            for key in item.sources
        ) if kind == "evaluate" else ()

        def expand(read_document=None, _previous=None):
            @cache
            def train_payload():
                return self.resolve_row(
                    composition, RunKey(variant, name, DEFAULT_SWEEP_ROW),
                    ExpandedRow(DEFAULT_SWEEP_ROW, None, {}, {}, {}), read_document,
                )

            @cache
            def train_model():
                resolved = train_payload()
                if isinstance(resolved, ProjectDiagnostic):
                    return None
                _, model = self.lower_train_model(
                    RunKey(variant, name, DEFAULT_SWEEP_ROW), resolved.payload, resolved.document_ids, resolved.document_pointers,
                )
                return model

            def integer_site(path: str) -> bool:
                target = model_field_target(path)
                if target is None:
                    if kind != "train":
                        return False
                    resolved = train_payload()
                    return isinstance(resolved, ResolvedStep) and _integer_train_site(
                        path, resolved.payload, self.python_tasks,
                    )
                if target.step is not None:
                    raise ValueError("A step-scoped Model target belongs in the project sweep")
                sources = subjects if kind == "evaluate" else (train_model(),) if kind == "train" else ()
                types = {
                    _integer_model_site(
                        source.document, source.model, target.address, self.components or _builtin_model_registry(),
                    )
                    for source in sources if source is not None
                }
                if len(types) > 1:
                    raise ValueError("The addressed range has different site types in the selected Train subjects")
                return True in types

            sites: dict[str, bool] = {}

            def capture_site(path: str) -> bool:
                sites[path] = answer = integer_site(path)
                return answer

            try:
                exception_matches: dict[str, tuple[str, ...]] = {}
                rows = expand_sweep(
                    composition.sweep, inputs=fanout,
                    integer_site=capture_site if self.source_sweeps is not None else integer_site,
                    checkpoint_subject=kind == "evaluate",
                    exception_matches=exception_matches,
                )
                if self.source_sweeps is not None:
                    axes = composition.sweep.axes if composition.sweep is not None else {}
                    roles = {axis.input for axis in axes.values() if isinstance(axis, InputAxis)}
                    if checkpoint_axis:
                        roles.add("subject")
                    source = SourceSweep({
                        role: tuple(
                            replace(member, coordinates=deepcopy(dict(member.coordinates)))
                            for member in members
                        )
                        for role, members in fanout.items() if role in roles
                    }, sites)
                    if source.inputs or source.integer_sites:
                        self.source_sweeps[variant, name] = source
                return rows, exception_matches
            except SweepExpansionError as error:
                return _sweep_problem(error, step=name, variant=variant), {}

        if self.previous is None:
            rows, matches = expand()
        else:
            rows, matches = _work(
                self.previous.sweeps,
                self.work.sweeps,
                (variant, name),
                (
                    composition,
                    tuple(subject.stamp if subject is not None else None for subject in subjects),
                    to_json(tuple(
                        (role, tuple(
                            (member.key, tuple(member.coordinates.items()), member.label, member.checkpoint_horizon)
                            for member in bindings
                        ))
                        for role, bindings in sorted(fanout.items())
                    )),
                ),
                self.authored_inputs[variant],
                expand,
            )
        if matches:
            self.exception_matches[variant, name] = MappingProxyType(matches)
        return rows

    def checkpoint_horizon(self, request: RequestId) -> int | None:
        if request not in self.checkpoint_horizons:
            self.checkpoint_horizons[request] = _train_checkpoint_horizon(request_identity.decode_request(
                self.requests[request], self.documents.__getitem__,
            ).spec)
        return self.checkpoint_horizons[request]

    def resolve_row(
        self,
        composition: StepComposition,
        key: RunKey,
        row: ExpandedRow,
        reader: Callable[[str], Document | None] | None,
        *,
        sources: bool = False,
        trial_resolution: _TrialResolution | None = None,
    ) -> ResolvedStep | ProjectDiagnostic:
        project = self.variants[key.variant]
        return composition.bind(
            project,
            readers=self.readers,
            variant_sources=self.variant_sources.get(key.variant),
            read_document=reader,
        ).resolve(
            row, include_sources=sources, partial_figures=project.steps[key.step].kind == "analyze",
            trial_resolution=trial_resolution,
        )

    def project_definition(
        self,
        composition: StepComposition,
        key: RunKey,
        row: ExpandedRow,
        inputs: Mapping[str, PlannedInput],
    ) -> SelectedRunDefinition | None:
        def project(reader=None, _previous=None):
            resolved = self.resolve_row(composition, key, row, reader, sources=True)
            if isinstance(resolved, ProjectDiagnostic) or resolved.sources is None:
                return None
            sources = {f"/payload{path}": source for path, source in resolved.sources.items()}
            for role, binding in inputs.items():
                subset = self.subsets.get((key.variant, key.step, role))
                if subset is not None and subset.source is not None:
                    sources[pointer(("inputs", role, "where"))] = subset.source.origin
                if (source := _upstream_source(binding)) is not None:
                    sources[pointer(("inputs", role))] = source
            return SelectedRunDefinition(
                resolved.payload, MappingProxyType(sources), MappingProxyType(resolved.document_ids), resolved.document_uses,
                MappingProxyType(resolved.document_pointers),
                resolved.document_problems,
            )

        if self.previous is None:
            return project()
        return _work(
            self.previous.definitions,
            self.work.definitions,
            key,
            (
                composition,
                to_json((
                    tuple(row.patches.items()),
                    tuple(row.patch_origins.items()),
                    tuple(row.patch_authors.items()),
                    tuple(row.coordinates.items()),
                )),
                tuple(sorted(inputs.items())),
                tuple(
                    subset.source.origin
                    for role in inputs
                    if (subset := self.subsets.get((key.variant, key.step, role))) is not None
                    and subset.source is not None
                ),
            ),
            self.authored_inputs[key.variant],
            project,
        )

    def lower_row(
        self,
        composition: StepComposition,
        key: RunKey,
        row: ExpandedRow,
        kind: StepKind,
        inputs: Mapping[str, PlannedInput],
        basis: str | None,
        definition: SelectedRunDefinition | None,
        train_rows: _TrainRows | None = None,
    ) -> _LoweredRow:
        member = to_json(tuple(
            (role, binding.member.model_dump(mode="json")) for role, binding in sorted(inputs.items())
        ))
        subject = inputs.get("subject") if kind == "evaluate" else None
        model_source = self.model_sources.get(subject.producer) if subject is not None and subject.producer else None
        source_stamp = model_source.stamp if model_source is not None else None
        source_problem = model_source.evaluation_problem if model_source is not None else None

        def lower(reader=None, previous=None):
            if definition is None:
                resolved = self.resolve_row(
                    composition, key, row, reader,
                    trial_resolution=train_rows.resolution if train_rows is not None else None,
                )
                if isinstance(resolved, ProjectDiagnostic):
                    return _LoweredRow(None, (replace(resolved, variant=key.variant),))
                payload = resolved.payload
                document_ids = MappingProxyType(resolved.document_ids)
                document_uses = resolved.document_uses
                document_pointers = resolved.document_pointers
            else:
                if reader is not None:
                    for name, _ in self.work.definitions[key].documents:
                        reader(name)
                payload = definition.payload
                document_ids = definition.document_ids
                document_uses = definition.document_uses
                document_pointers = definition.document_pointers
            authored_model = None
            prepared_train = None
            remember_train = None
            has_differences = MODEL_ATTACHMENTS in payload
            encoding_stamp = (
                (kind, sha256_bytes(to_json((payload, document_ids.get("payload.graph"), source_stamp))), member)
                if basis is not None else None
            )
            if previous is not None and previous.encoding_stamp == encoding_stamp:
                return replace(
                    previous, document_ids=document_ids, document_uses=document_uses,
                )
            if kind == "evaluate" and payload.get("evaluation_type") == ROLLOUT_TYPE:
                params = payload.get("params")
                managed_subject = model_source is not None and (
                    model_source.evaluation_graph is not None or source_problem is not None
                )
                if managed_subject or (isinstance(params, dict) and "trial_set" in params):
                    if problem := _authored_graph_variation_problem(key, payload):
                        return _LoweredRow(
                            None, (problem,), encoding_stamp, document_ids=document_ids, document_uses=document_uses,
                        )
            if source_problem is not None and payload.get("evaluation_type") == ROLLOUT_TYPE:
                return _LoweredRow(
                    None, (replace(source_problem, step=key.step, row=key.row, variant=key.variant),),
                    encoding_stamp, document_ids=document_ids, document_uses=document_uses,
                )
            if kind == "train":
                trial = payload.get("trial_documents")
                if (
                    train_rows is not None and isinstance(trial, dict)
                    and train_rows.resolution.contains(trial) and not has_differences
                ):
                    if (entry := train_rows.prepared) is not None and entry[0] is trial:
                        prepared_train = entry[1]
                    else:
                        def remember_train(prepared):
                            train_rows.prepared = trial, prepared
                if prepared_train is None:
                    payload, authored_model = self.lower_train_model(key, payload, document_ids, document_pointers)
            elif kind == "evaluate":
                reference = subject.member.reference if subject is not None else None
                if isinstance(reference, BestRank):
                    assert subject is not None
                    payload = self.lower_best_evaluate(key, payload, subject, reference, has_differences)
                else:
                    payload = self.lower_evaluate_payload(key, payload, model_source, reference, has_differences)
            if isinstance(payload, ProjectDiagnostic):
                return _LoweredRow(None, (payload,), encoding_stamp, document_ids=document_ids, document_uses=document_uses)
            lowered = _encode_row(
                key, kind, payload, inputs, encoding_stamp,
                self.work.request_documents, self.prepare_train_trial,
                self.python_tasks, model_source=authored_model,
                prepared_train=prepared_train, remember_train=remember_train,
            )
            return replace(
                lowered,
                document_ids=document_ids,
                document_uses=document_uses,
                model_source=lowered.model_source or authored_model,
            )

        if self.previous is None:
            return lower()
        return _work(
            self.previous.rows,
            self.work.rows,
            key,
            (kind, basis, to_json(tuple(row.patches.items())), tuple(sorted(inputs.items())), source_stamp),
            self.authored_inputs[key.variant],
            lower,
        )

    def prepare_train_trial(self, authored: TrialDocuments) -> _TrainPreparation | tuple[Problem, ...]:
        from feedbax.compiler.trial_execution import compile_trial_execution

        # Names and document order can change which diagnostic is reported.
        identity = to_json(
            authored.model_dump(exclude_unset=True, round_trip=True), inf_nan_mode="constants",
        )
        if (prepared := self.trial_preparations.get(identity)) is not None:
            return prepared
        paired = compile_trial_execution(
            authored.model_name, authored.model, authored.task_source,
            self.components or _builtin_model_registry(), self.functions or _builtin_expression_registry(),
            python_tasks=self.python_tasks, objective=authored.objective, input_overrides=authored.input_overrides,
        )
        result: _TrainPreparation | tuple[Problem, ...]
        if isinstance(paired, tuple):
            result = paired
        else:
            graph = paired.executable.resolved.graph
            checkpoint_problem = None
            if problem := model_owner_problem(graph, "checkpoint"):
                address = paired.lowered.source_address(pointer(problem.path))
                path = ("trial_documents", "model", *address.stored_path()) if address is not None else problem.path
                checkpoint_problem = RunProblem(pointer(path), problem.reason, address)
            evaluation: GraphSpec | ProjectDiagnostic = graph
            if problem := model_owner_problem(graph, "evaluate"):
                address = paired.lowered.source_address(pointer(problem.path))
                evaluation = ProjectDiagnostic(
                    ResolutionCode.INVALID_PAYLOAD, pointer(("inputs", "subject", *problem.path)), problem.reason,
                    document=address.document if address is not None else None, model_address=address,
                )
            result = _TrainPreparation(checkpoint_problem, evaluation, paired.execution.task.condition_ids)
        self.trial_preparations[identity] = result
        return result

    def lower_train_model(
        self, key: RunKey, payload: Payload, document_ids: Mapping[str, str], document_pointers: Mapping[str, str],
    ) -> tuple[Payload | ProjectDiagnostic, _ModelSource | None]:
        from feedbax.compiler.model_attachment import resolve_effective_model
        from feedbax.compiler.model_document import ModelCompilationProblem

        raw = payload.get("trial_documents")
        model_payload = raw.get("model") if isinstance(raw, dict) else None
        document = raw.get("model_name") if isinstance(raw, dict) else None
        address = None
        has_differences = MODEL_ATTACHMENTS in payload
        try:
            payload, differences = _model_differences(payload)
            if not isinstance(model_payload, dict) or "interior" not in model_payload:
                if differences:
                    address = differences[0].address
                    raise ValueError("Model differences require an authored Train Model")
                return payload, None
            if not isinstance(document, str):
                if differences:
                    address = differences[0].address
                    raise ValueError("Model differences require a named Train Model")
                return payload, None
            model = ModelDocument.model_validate(model_payload)
            if differences:
                model = resolve_effective_model(
                    document, model, self.components or _builtin_model_registry(), differences,
                    functions=self.functions,
                )
            model_source = _ModelSource(
                document, model, sha256_bytes(to_json((document, model.model_dump(mode="json")))),
            )
            if differences:
                payload = _with_train_model(payload, model_source, document_ids, document_pointers)
            return payload, model_source
        except ModelCompilationProblem as error:
            address, detail = error.address, str(error)
        except (KeyError, ValueError) as error:
            if not has_differences:
                return payload, None  # The ordinary Train validator reports invalid model documents.
            detail = str(error)
        return _model_difference_problem(key, address, detail), None

    def lower_evaluate_payload(
        self, key: RunKey, payload: Payload, model_source: _ModelSource | None,
        reference: object, has_differences: bool,
    ) -> Payload | ProjectDiagnostic:
        if not has_differences:
            return self.lower_evaluate_trials(key, payload, reference, model_source)
        original = payload.get("params")
        lowered = self.lower_evaluate_model(key, payload, model_source, reference)
        if isinstance(lowered, ProjectDiagnostic):
            return lowered
        result = self.lower_evaluate_trials(key, lowered, reference, model_source)
        # Derived Model patches append to an ordered copy. Report a Task
        # refusal at the original named edit without changing request encoding.
        if isinstance(result, ProjectDiagnostic) and isinstance(original, dict):
            named = original.get("variation")
            params = lowered.get("params")
            if isinstance(named, dict) and isinstance(params, dict) and isinstance(params.get("variation"), list):
                return _authored_variation_problem(result, tuple(sorted(named)))
        return result

    def lower_best_evaluate(
        self, key: RunKey, payload: Payload, subject: PlannedInput, rank: BestRank, has_differences: bool,
    ) -> Payload | ProjectDiagnostic:
        def problem(detail: str) -> ProjectDiagnostic:
            return ProjectDiagnostic(
                ResolutionCode.BAD_REFERENCE, "inputs.subject.where.best", detail, key.step,
                row=key.row, variant=key.variant,
            )

        if payload.get("evaluation_type") != ROLLOUT_TYPE:
            return problem("Best requires managed rollout Evaluate.")
        common = None
        for producer, candidate in zip(subject.candidates, rank.candidates, strict=True):
            model = self.model_sources.get(producer)
            name = f"{producer.variant}/{producer.step}/{producer.row}"
            if model is None or model.evaluation_graph is None:
                message = model.evaluation_problem.detail if model and model.evaluation_problem else "no managed Model and Task"
                return problem(f"Best candidate {name}: {message}")
            if refusal := _authored_graph_variation_problem(key, payload):
                return refusal
            lowered = self.lower_evaluate_payload(key, payload, model, candidate.reference, has_differences)
            if isinstance(lowered, ProjectDiagnostic):
                return replace(lowered, detail=f"Best candidate {name}: {lowered.detail}")
            try:
                if "trial_set" in lowered["params"]:
                    raise ValueError("Task trial rule was not resolved")
                RolloutParameters.model_validate(lowered["params"])
            except (KeyError, ValueError) as error:
                return problem(f"Best candidate {name}: {error}")
            if common is not None and not same_typed_value(common, lowered):
                return problem(f"Best candidate {name} resolves a different Evaluate definition or trial order.")
            common = lowered
        assert common is not None
        return common

    def lower_evaluate_trials(
        self, key: RunKey, payload: Payload, reference: object, model_source: _ModelSource | None,
    ) -> Payload | ProjectDiagnostic:
        from feedbax.compiler.trial_execution import evaluation_conditions, pair_trial_source
        from feedbax.studio.canonical_evaluate import CannotRun
        from feedbax.training.evaluation_variant import resolve_evaluation_variant

        if (
            payload.get("evaluation_type") != ROLLOUT_TYPE or not isinstance(reference, ProductRef)
            or model_source is None or model_source.evaluation_graph is None
        ):
            return payload
        source = self.requests.get(reference.request_id)
        if source is None:
            return payload
        train = request_identity.decode_request(source, self.documents.__getitem__).spec
        if not isinstance(train, DocumentTrainingRequestSpec):
            return payload
        raw = payload.get("params")
        if not isinstance(raw, dict):
            return payload
        location = "trial_set"
        try:
            if "trial_set" not in raw or "trials" in raw:
                raise ValueError("Declare a trial-set rule for managed Evaluate.")
            settings = authored_rollout_settings(raw)
            documents = train.trial_documents
            variant = resolve_evaluation_variant(
                model_source.evaluation_graph, documents.task_source, settings.variation,
            )
            if any(target[0] == "task" for target in variant.edit_origins):
                pairing = pair_trial_source(
                    documents.model_name, documents.model, variant.task,
                    self.components or _builtin_model_registry(), self.functions or _builtin_expression_registry(),
                    python_tasks=self.python_tasks, input_overrides=documents.input_overrides,
                    graph_override=variant.graph,
                )
                problems = pairing if isinstance(pairing, tuple) else pairing.problems
                if problems:
                    raise variant.task_refusal(problems[0])
                conditions = pairing.task.condition_ids
            else:
                conditions = evaluation_conditions(variant.task, self.python_tasks)
            trials = resolve_trial_ids(raw["trial_set"], conditions)
            return payload | {"params": {name: value for name, value in raw.items() if name != "trial_set"} | {
                "trials": list(trials),
            }}
        except ValueError as error:
            return ProjectDiagnostic(
                ResolutionCode.INVALID_PAYLOAD,
                error.path if isinstance(error, CannotRun) else f"payload.params.{location}",
                error.reason if isinstance(error, CannotRun) else str(error), key.step,
                row=key.row, variant=key.variant,
            )

    def lower_evaluate_model(
        self, key: RunKey, payload: Payload, model_source: _ModelSource | None, reference: object,
    ) -> Payload | ProjectDiagnostic:
        from feedbax.compiler.model_attachment import lower_evaluate_attachments
        from feedbax.compiler.model_document import ModelCompilationProblem
        from feedbax.studio.schema import validate_graph_connection_schema
        from feedbax.studio.canonical_evaluate import CannotRun
        from feedbax.training.evaluation_variant import resolve_evaluation_variant

        address = None
        try:
            stripped, differences = _model_differences(payload)
            if not differences:
                return stripped
            address = differences[0].address
            if payload.get("evaluation_type") != ROLLOUT_TYPE:
                raise ValueError("Attachment differences require a rollout Evaluate with an authored Train Model")
            if not isinstance(reference, ProductRef):
                raise ValueError("Attachment differences require a planned Train subject")
            train = request_identity.decode_request(
                self.requests[reference.request_id], self.documents.__getitem__,
            ).spec
            if not isinstance(train, DocumentTrainingRequestSpec):
                raise ValueError("Attachment differences require a Train request")
            if model_source is None:
                raise ValueError("Attachment differences need a saved Model document")
            components = self.components or _builtin_model_registry()
            documents = train.trial_documents
            document, model = model_source.document, model_source.model
            before, lowered, derived = lower_evaluate_attachments(
                document, model, None, components, differences, functions=self.functions,
            )
            raw_params = payload["params"]
            params = authored_rollout_settings(raw_params)
            if not derived:
                return stripped
            prior = params.variation
            if isinstance(prior, dict):
                prior = tuple(prior[name] for name in sorted(prior))
            variation = (*prior, *derived)
            effective = resolve_evaluation_variant(before.graph, documents.task_source, variation)
            issues = validate_graph_connection_schema(
                effective.graph, "/graph",
                component_registry=components,
            )
            if issue := next((issue for issue in issues if issue.severity == "error"), None):
                issue_path = (issue.location or {}).get("path", "/graph")
                address = lowered.source_address(issue_path) or address
                raise ValueError(issue.message)
            return stripped | {"params": raw_params | {
                "variation": [edit.model_dump(mode="json", exclude_unset=True) for edit in variation],
            }}
        except ModelCompilationProblem as error:
            address = error.address
            detail = str(error)
        except CannotRun as error:
            return _model_difference_problem(key, address, error.reason, path=error.path)
        except ValueError as error:
            detail = str(error)
        return _model_difference_problem(key, address, detail, path=pointer(address.stored_path()) if address else None)

    def compile_step(
        self, variant: str, name: str, active: tuple[tuple[str, str], ...]
    ) -> tuple[_CompiledRun, ...]:
        project = self.variants[variant]
        authored = project.steps[name]
        kind = authored.kind
        step = PlannedStep(kind, authored.group, MappingProxyType(deepcopy(authored.inputs)))
        self.steps[variant, name] = step
        wanted = self.wanted_sources.get((variant, name), set())
        role = _ROLES.get(kind)
        if (role is None and kind != "report") or (kind == "analyze" and self.operations is None):
            self.diagnostics.append(
                ProjectDiagnostic(
                    ResolutionCode.UNSUPPORTED_PLAN,
                    "kind",
                    f"{kind} lowering is unavailable",
                    name,
                    variant=variant,
                )
            )
            return ()
        template = self.compose(variant, name, bool(wanted))
        if isinstance(template, ProjectDiagnostic):
            self.diagnostics.append(replace(template, variant=variant))
            return ()
        self.steps[variant, name] = replace(
            step, inputs=MappingProxyType(deepcopy(dict(template.inputs))), sweep=template.sweep,
        )
        sweep = template.sweep
        if sweep is not None:
            axes = tuple(axis for axis, value in sweep.axes.items() if value is not None)
            if axes or any(row is not None for row in sweep.rows.values()):
                self.swept_step_axes[variant, name] = axes
        if kind == "report":
            return self.compile_report_step(variant, name, template, active)
        assert role is not None
        extra_roles = sorted(template.inputs.keys() - {role.name})
        if extra_roles:
            self.diagnostics.extend(
                ProjectDiagnostic(
                    ResolutionCode.BAD_REFERENCE,
                    f"inputs.{extra}",
                    "unknown input role",
                    name,
                    variant=variant,
                )
                for extra in extra_roles
            )
            return ()
        selected = self.select_input(variant, name, template, role, active)
        if isinstance(selected, ProjectDiagnostic):
            self.diagnostics.append(replace(selected, variant=variant))
            return ()
        members, omissions, subset = selected
        if subset is not None:
            self.subsets[variant, name, role.name] = subset
        by_key = {item.member.key: item for item, _ in members}
        subject = template.inputs.get("subject") if kind == "evaluate" else None
        has_checkpoint_axis = kind == "evaluate" and sweep is not None and any(
            isinstance(axis, CheckpointRuleAxis) for axis in sweep.axes.values()
        )
        paired_step = _paired_checkpoint_step(authored, template.inputs.get(role.name))
        rows = self.expand_rows(variant, name, template, {role.name: members})
        if isinstance(rows, ProjectDiagnostic):
            self.diagnostics.append(rows)
            return ()

        basis = template.row_basis if self.previous is not None else None
        train_rows = _TrainRows() if kind == "train" and not wanted and len(rows) > 1 else None
        result = []
        tracked_template = template if wanted else None
        for row in rows:
            run_key = RunKey(variant, name, row.id)
            key = row.input_keys.get(role.name)
            chosen = by_key[key] if key is not None else None
            if chosen is None and paired_step is not None:
                parent_axes = self.swept_step_axes.get((variant, paired_step))
                inherited_id = (
                    DEFAULT_SWEEP_ROW if parent_axes is None else row.inherited_id(parent_axes)
                )
                if inherited_id is not None:
                    chosen = by_key.get(inherited_id)
                    producer_key = RunKey(variant, paired_step, inherited_id)
                    if chosen is None and (
                        producer_key in self.excluded_checkpoint_rows
                        or (subset is not None and isinstance(self.rows.get(producer_key), PlannedRun))
                    ):
                        # Carry an intentional subset exclusion through inherited checkpoint pairs.
                        self.excluded_checkpoint_rows.add(run_key)
                        continue
            coordinates = MappingProxyType(deepcopy(dict(row.coordinates)))
            self.rows[run_key] = PlannedRow(run_key, kind, row.label, coordinates)
            if chosen is None and (paired_step is not None or len(members) > 1):
                candidates = ", ".join(by_key) or "none"
                detail = f"role requires one row; candidates are {candidates}"
                if paired_step is not None:
                    detail = (
                        f"no inherited row in {paired_step!r} matches {row.id!r}. "
                        f"Candidates: {candidates}. Select one row or use a {role.name} input axis."
                    )
                self.diagnostics.append(
                    ProjectDiagnostic(
                        ResolutionCode.BAD_REFERENCE,
                        f"inputs.{role.name}",
                        detail,
                        name,
                        row=row.id,
                        variant=variant,
                    )
                )
                continue
            if chosen is None and members:
                chosen = members[0][0]
            if (
                chosen is not None and isinstance(chosen.member.reference, BestRank)
                and CHECKPOINT_RULE_TARGET in row.patches
            ):
                self.diagnostics.append(ProjectDiagnostic(
                    ResolutionCode.BAD_REFERENCE, "inputs.subject.checkpoint",
                    "Best requires the selected Train's last checkpoint without row overrides.",
                    name, row=row.id, variant=variant,
                ))
                continue
            if kind == "evaluate" and chosen is not None and isinstance(chosen.member.reference, ProductRef):
                reference = chosen.member.reference
                try:
                    write = row.patches.get(CHECKPOINT_RULE_TARGET)
                    selected_subject = checkpoint_rule_write(subject, write) if write is not None else subject
                    rule = selected_subject.checkpoint if isinstance(selected_subject, StepReference) else None
                    if rule is None and has_checkpoint_axis:
                        raise ValueError("The selected row has no checkpoint rule")
                    horizon = None
                    if isinstance(rule, UpdateCadence):
                        horizon = self.checkpoint_horizon(reference.request_id)
                    selector = _checkpoint(rule, horizon=horizon)
                except ValueError as error:
                    self.diagnostics.append(ProjectDiagnostic(
                        ResolutionCode.BAD_REFERENCE, "inputs.subject.checkpoint", str(error),
                        name, row=row.id, variant=variant,
                    ))
                    continue
                chosen = replace(chosen, member=chosen.member.model_copy(update={
                    "reference": reference.model_copy(update={"select": selector}),
                }))
            inputs = {role.name: chosen} if chosen is not None else {}
            definition = self.select_definition(template, run_key, row, inputs)
            if kind == "analyze":
                assert chosen is not None
                analyze = self.lower_analyze_row(template, run_key, row, chosen, definition)
                if analyze is not None:
                    result.append(analyze)
                continue
            lowered = self.lower_row(
                template, run_key, row, kind, inputs, basis, definition, train_rows,
            )
            if len(lowered.diagnostics) == 1 and lowered.diagnostics[0].code == ResolutionCode.AUTHORED_GRAPH_VARIATION:
                (problem,) = lowered.diagnostics
                sources_available = variant not in self.variant_sources or self.variant_sources[variant] is not None
                if definition is None and sources_available:
                    if tracked_template is None:
                        tracked_template = self.compose(variant, name, track=True)
                    if isinstance(tracked_template, StepComposition):
                        definition = self.project_definition(tracked_template, run_key, row, inputs)
                origin = definition.sources.get("/payload" + problem.path) if definition is not None else None
                if origin is not None:
                    problem = self.locate_authored_problem(problem, origin)
                    lowered = replace(lowered, diagnostics=(problem,))
            if (run := self.record_run(run_key, lowered, inputs, omissions, template.requires)) is not None:
                result.append(run)
        return tuple(result)

    def locate_authored_problem(self, problem: ProjectDiagnostic, origin: ValueSource) -> ProjectDiagnostic:
        if origin.kind == "document":
            if origin.name in self.project.documents:
                return replace(
                    problem, document=origin.name, document_path=origin.path,
                    authored_file=f"documents/{origin.name}.json", authored_path="/payload" + origin.path,
                )
        elif origin.step is not None and origin.step in self.project.steps:
            return replace(problem, authored_file=f"steps/{origin.step}.json", authored_path=origin.path)
        return problem

    def select_definition(
        self, template: StepComposition, key: RunKey, row: ExpandedRow,
        inputs: Mapping[str, PlannedInput],
    ) -> SelectedRunDefinition | None:
        if key.row not in self.wanted_sources.get((key.variant, key.step), ()):
            return None
        definition = self.project_definition(template, key, row, inputs)
        if definition is not None:
            self.selected_definitions[key] = (
                replace(definition, payload=deepcopy(definition.payload))
                if self.previous is not None else definition
            )
        return definition

    def record_run(
        self, key: RunKey, lowered: _LoweredRow, inputs: Mapping[str, PlannedInput],
        omissions: Mapping[str, str], requires: tuple[str, ...],
    ) -> PlannedRun | None:
        row = self.rows[key]
        self.diagnostics.extend(
            replace(problem, document=lowered.document_ids.get("payload"))
            if row.kind == "report" and problem.document is None else problem
            for problem in lowered.diagnostics
        )
        if lowered.model_source is not None and (definition := self.selected_definitions.get(key)):
            if MODEL_ATTACHMENTS in definition.payload:
                self.selected_definitions[key] = _train_model_definition(definition, lowered.model_source)
        if (encoded := lowered.request) is None:
            self.rows[key] = replace(
                row, document_ids=lowered.document_ids, document_uses=lowered.document_uses,
            )
            return None
        self.requests[encoded.id] = encoded.data
        self.documents.update(encoded.documents)
        if lowered.evaluation_source is not None:
            self.train_evaluation_sources[encoded.id] = _TrainEvaluationSource(
                encoded.data, tuple(encoded.documents.items()), lowered.evaluation_source,
            )
        if lowered.model_source is not None:
            self.model_sources[key] = lowered.model_source
        run = PlannedRun(
            key, row.kind, row.label, row.coordinates, encoded.id,
            MappingProxyType(dict(inputs)), MappingProxyType(dict(omissions)), requires,
            admission_problem=lowered.admission_problem,
            document_ids=lowered.document_ids,
            document_uses=lowered.document_uses,
        )
        self.rows[key] = run
        return run

    def compile_report_step(
        self, variant: str, name: str, template: StepComposition, active: tuple[tuple[str, str], ...],
    ) -> tuple[PlannedRun, ...]:
        selected = {}
        for name_of_role in sorted(template.inputs):
            role = _Role(name_of_role, True, "analyze", PLOTLY_FIGURE_ROLE)
            result = self.select_input(variant, name, template, role, active)
            if isinstance(result, ProjectDiagnostic):
                self.diagnostics.append(replace(result, variant=variant))
                return ()
            members, _, subset = result
            selected[role.name] = members
            if subset is not None:
                self.subsets[variant, name, role.name] = subset
        rows = self.expand_rows(variant, name, template, selected)
        if isinstance(rows, ProjectDiagnostic):
            self.diagnostics.append(rows)
            return ()
        by_role = {
            role: {binding.member.key: binding for binding, _ in members}
            for role, members in selected.items()
        }
        basis = template.row_basis if self.previous is not None else None
        runs = []
        for row in rows:
            key = RunKey(variant, name, row.id)
            coordinates = MappingProxyType(deepcopy(dict(row.coordinates)))
            self.rows[key] = PlannedRow(key, "report", row.label, coordinates)
            inputs = {}
            for role, candidates in by_role.items():
                if (member := row.input_keys.get(role)) is not None:
                    inputs[role] = candidates[member]
                elif len(candidates) == 1:
                    inputs[role] = next(iter(candidates.values()))
                else:
                    self.diagnostics.append(ProjectDiagnostic(
                        ResolutionCode.BAD_REFERENCE, f"inputs.{role}",
                        f"figure role requires one row; candidates are {', '.join(candidates) or 'none'}",
                        name, row=row.id, variant=variant,
                    ))
                    break
            else:
                definition = self.select_definition(template, key, row, inputs)
                lowered = self.lower_row(template, key, row, "report", inputs, basis, definition)
                if (run := self.record_run(key, lowered, inputs, {}, template.requires)) is not None:
                    runs.append(run)
        return tuple(runs)

    def lower_analyze_row(
        self,
        template: StepComposition,
        key: RunKey,
        row: ExpandedRow,
        source: PlannedInput,
        definition: SelectedRunDefinition | None,
    ) -> PlannedAnalyzeRun | None:
        from feedbax.contracts.project_analyze import lower_analyze

        assert self.operations is not None
        resolved = definition if definition is not None else self.resolve_row(template, key, row, None)
        if isinstance(resolved, ProjectDiagnostic):
            self.diagnostics.append(replace(resolved, variant=key.variant))
            return None
        producer = source.member.reference
        assert isinstance(producer, ProductRef)
        evaluation = request_identity.decode_request(
            self.requests[producer.request_id], self.documents.__getitem__,
        )
        elements, encoded, problems, figures = lower_analyze(
            key,
            resolved.payload,
            resolved.document_ids,
            resolved.document_pointers,
            self.variants[key.variant].documents,
            tuple(replace(problem, row=key.row, variant=key.variant) for problem in resolved.document_problems),
            source,
            evaluation.spec,
            self.operations,
        )
        self.diagnostics.extend(problems)
        for request in encoded:
            self.requests[request.id] = request.data
            self.documents.update(request.documents)
        original = self.rows[key]
        run = PlannedAnalyzeRun(
            key, original.kind, original.label, original.coordinates,
            MappingProxyType(elements), template.requires, source, MappingProxyType(figures),
            document_ids=MappingProxyType(dict(resolved.document_ids)),
            document_uses=resolved.document_uses,
        )
        self.rows[key] = run
        return run

    def rank_input(
        self, name: str, template: StepComposition, role: _Role, members: _Members,
        subset: PlannedSubset, reference: StepReference,
    ) -> tuple[_Members, dict[str, str], PlannedSubset] | ProjectDiagnostic:
        path = f"inputs.{role.name}.where"
        terms = subset.terms
        if terms.when is not None:
            return ProjectDiagnostic(
                ResolutionCode.BAD_REFERENCE, f"{path}.when",
                "When requires a many-valued input and cannot supply an input axis.", name,
            )
        best = terms.best
        assert best is not None
        if role.name != "subject":
            reason = "Best currently binds saved Evaluate subjects."
        elif reference.checkpoint not in (None, "last") or (
            template.sweep is not None and any(isinstance(axis, CheckpointRuleAxis) for axis in template.sweep.axes.values())
        ):
            reason = "Best requires the selected Train's last checkpoint."
        else:
            return self.rank_groups(name, template, role, members, subset)
        return ProjectDiagnostic(ResolutionCode.BAD_REFERENCE, f"{path}.best", reason, name)

    def rank_groups(
        self, name: str, template: StepComposition, role: _Role, members: _Members, subset: PlannedSubset,
    ) -> tuple[_Members, dict[str, str], PlannedSubset] | ProjectDiagnostic:
        best = subset.terms.best
        assert best is not None
        path = f"inputs.{role.name}.where.best"

        def refusal(detail: str, field: str = path) -> ProjectDiagnostic:
            return ProjectDiagnostic(ResolutionCode.BAD_REFERENCE, field, detail, name)

        axes = tuple(sorted(best.per))
        if len(set(axes)) != len(axes):
            return refusal("Best group axes must be distinct.", f"{path}.per")
        if not members:
            return refusal("Best requires at least one candidate.")
        groups: dict[tuple[str, ...], list[tuple[PlannedInput, _CompiledRun]]] = {}
        for item, producer in members:
            for axis in axes:
                if axis not in producer.coordinates:
                    return refusal(
                        f"Best group axis {axis!r} is missing on {producer.key.variant}/{producer.key.row}.",
                        f"{path}.per",
                    )
            group = tuple(producer.coordinates[axis].value_id for axis in axes)
            grouped = groups.setdefault(group, [])
            if grouped and any(
                not same_typed_value(producer.coordinates[axis].value, grouped[0][1].coordinates[axis].value)
                for axis in axes
            ):
                return refusal(f"Best group {dict(zip(axes, group, strict=True))!r} has conflicting axis values.",
                               f"{path}.per")
            grouped.append((item, producer))
        if best.k * len(groups) > 1 and not (
            template.sweep is not None and any(
                isinstance(axis, InputAxis) and axis.input == role.name for axis in template.sweep.axes.values()
            )
        ):
            return refusal("Best with more than one rank requires a subject input axis.")

        ranked = []
        planned_groups = []
        for group, grouped in groups.items():
            coordinates = dict(zip(axes, group, strict=True))
            if best.k > len(grouped):
                target = f" in group {coordinates!r}" if coordinates else ""
                return refusal(f"Best requests {best.k} ranks from {len(grouped)} candidates{target}.")
            candidates = []
            keys = []
            for item, producer in grouped:
                assert isinstance(producer, PlannedRun)
                if producer.admission_problem is not None:
                    return refusal(
                        f"Best candidate {producer.key.row!r} cannot produce a checkpoint: "
                        f"{producer.admission_problem.detail}",
                    )
                keys.append(producer.key)
                candidates.append(BestCandidate(
                    key=item.member.key, variant=producer.key.variant, row=producer.key.row,
                    reference=ProductRef(request_id=producer.request_id, role=role.product_role, select=LastMember()),
                ))
            ranks = []
            for position in range(1, best.k + 1):
                rank = BestRank(
                    metric=best.by, order=best.order, count=best.k, rank=position, candidates=tuple(candidates),
                    group=coordinates,
                )
                ranks.append(rank.key)
                ranked.append((PlannedInput(
                    None, InputMember(key=rank.key, reference=rank), candidates=tuple(keys),
                ), grouped[0][1]))
            planned_groups.append(PlannedRankGroup(MappingProxyType(coordinates), tuple(ranks)))
        return tuple(ranked), {}, replace(subset, rank_groups=tuple(planned_groups))

    def subset_origin(
        self, variant: str, step: str, role: str, document: str | None, term: Literal["best", "rows"],
    ) -> ValueSource:
        if document is not None:
            authored = ("documents", document, "payload", term)
            source = ValueSource("document", document, pointer((term,)))
        else:
            steps = self.variants[variant].steps
            while role not in steps[step].inputs and isinstance(base := steps[step].base, StepBase):
                step = base.step
            path = ("inputs", role, "where", term)
            authored = ("steps", step, *path)
            source = ValueSource("step", step, pointer(path), step=step)
        for target in reversed(self.variant_rows[variant].patches):
            parts = tuple(target.split("."))
            shared = min(len(authored), len(parts))
            if authored[:shared] == parts[:shared]:
                return ValueSource("variant", variant, pointer(parts), variant=variant)
        return source

    def select_input(
        self,
        variant: str,
        name: str,
        template: StepComposition,
        role: _Role,
        active: tuple[tuple[str, str], ...],
    ) -> (
        tuple[_Members, dict[str, str], PlannedSubset | None]
        | ProjectDiagnostic
    ):
        reference = template.inputs.get(role.name)
        path = f"inputs.{role.name}"
        if reference is None or isinstance(reference, OmittedReference):
            if role.required:
                return ProjectDiagnostic(
                    ResolutionCode.BAD_REFERENCE, path, "required input is missing", name
                )
            return (), {role.name: reference.not_applicable} if reference else {}, None
        if not isinstance(reference, StepReference):
            return ProjectDiagnostic(
                ResolutionCode.UNSUPPORTED_PLAN,
                path,
                "a same-project step or row reference is required",
                name,
            )
        element_input = role.producer_kind == "analyze"
        if element_input:
            if reference.element is None or reference.product != role.product_role:
                return ProjectDiagnostic(
                    ResolutionCode.BAD_REFERENCE, path,
                    f"declare a render element and its {role.product_role!r} product", name,
                )
        elif reference.element is not None or reference.product is not None:
            return ProjectDiagnostic(
                ResolutionCode.BAD_REFERENCE, path,
                "this input selects a step row without an element or product", name,
            )
        if reference.checkpoint is not None and role.product_role != "checkpoint":
            return ProjectDiagnostic(
                ResolutionCode.BAD_REFERENCE, f"{path}.checkpoint",
                "this product does not take a Train checkpoint selector", name,
            )
        terms = reference.where
        subset_document = None
        if isinstance(terms, NamedSubset):
            subset_document = terms.subset
            document = self.variants[variant].documents.get(subset_document)
            if document is None:
                return ProjectDiagnostic(
                    ResolutionCode.MISSING_DOCUMENT,
                    f"{path}.where",
                    terms.subset,
                    name,
                    document=subset_document,
                )
            try:
                terms = SubsetTerms.model_validate(document.payload)
            except ValidationError as error:
                return ProjectDiagnostic(
                    ResolutionCode.INVALID_DOCUMENT,
                    f"{path}.where",
                    str(error),
                    name,
                    document=subset_document,
                )
        variants = (
            (variant,)
            if reference.variants is None
            else self.variant_order
            if reference.variants == "all"
            else tuple(reference.variants)
        )
        if not variants or len(variants) != len(set(variants)):
            return ProjectDiagnostic(
                ResolutionCode.BAD_REFERENCE,
                f"{path}.variants",
                "variants must name a nonempty set of distinct variants",
                name,
            )
        if terms is not None and terms.rows:
            addressed_variants = tuple(dict.fromkeys(row.variant for row in terms.rows))
            if reference.variants is None:
                variants = addressed_variants
            elif outside := sorted(set(addressed_variants).difference(variants)):
                return ProjectDiagnostic(
                    ResolutionCode.BAD_REFERENCE, f"{path}.where.rows",
                    f"row variants outside reference scope: {outside!r}", name,
                )
        runs: tuple[_CompiledRun, ...] = ()
        failures: list[ProjectDiagnostic] = []
        for source_variant in variants:
            target = source_variant, reference.step
            if target in active:
                chain = " -> ".join(f"{v}/{s}" for v, s in (*active, target))
                return ProjectDiagnostic(ResolutionCode.STEP_CYCLE, path, chain, name)
            project = self.variants.get(source_variant)
            producer = project.steps.get(reference.step) if project is not None else None
            if producer is None or producer.kind != role.producer_kind:
                return ProjectDiagnostic(
                    ResolutionCode.BAD_REFERENCE,
                    path,
                    f"{source_variant}/{reference.step} must be a {role.producer_kind} step",
                    name,
                )
            runs += self.step(*target, active)
            failures.extend(
                problem for problem in self.diagnostics if (problem.variant, problem.step) == target
                and (not element_input or problem.element is None)
            )
        addresses = (
            {(row.variant, row.row) for row in terms.rows}
            if terms is not None and terms.rows is not None else None
        )

        def selected_failure(failure: ProjectDiagnostic) -> bool:
            if failure.row is None:
                return True
            if reference.row is not None:
                return failure.row == reference.row
            return addresses is None or (failure.variant, failure.row) in addresses

        if any(map(selected_failure, failures)):
            return ProjectDiagnostic(
                ResolutionCode.BAD_REFERENCE,
                path,
                f"referenced step {reference.step!r} has unplannable rows",
                name,
            )
        candidate_count = len(runs)

        def member_key(run: _CompiledRun) -> str:
            key = run.key
            return producer_member_key(key.variant, key.row, gathers_variants=reference.variants is not None)

        if reference.row is not None:
            selected = [
                SelectedRow(key=member_key(run), variant=run.key.variant, row=run.key.row)
                for run in runs if run.key.row == reference.row
            ]
            if not selected:
                return ProjectDiagnostic(
                    ResolutionCode.BAD_REFERENCE, f"{path}.row",
                    f"unknown row ID {reference.row!r}", name,
                )
            terms = SubsetTerms(rows=selected)
        try:
            keyed_runs = select_step_rows(
                terms,
                runs,
                address=lambda run: (run.key.variant, run.key.row),
                member_key=member_key,
                coordinates=lambda run: {
                    axis: coordinate.value_id for axis, coordinate in run.coordinates.items()
                },
            )
        except SubsetSelectionError as error:
            return ProjectDiagnostic(
                ResolutionCode.BAD_REFERENCE, f"{path}.where.{error.path}", error.detail, name
            )
        expands_input = template.sweep is not None and any(
            isinstance(axis, InputAxis) and axis.input == role.name
            for axis in template.sweep.axes.values()
        )
        if not keyed_runs and not expands_input:
            return ProjectDiagnostic(
                ResolutionCode.BAD_REFERENCE,
                path,
                f"no plannable rows for {reference.step!r}",
                name,
            )
        rule = None if role.name == "subject" else reference.checkpoint
        if isinstance(rule, (CheckpointBatches, CheckpointCadence, UpdateCadence, tuple)) and role.name != "subject":
            return ProjectDiagnostic(
                ResolutionCode.BAD_REFERENCE, f"{path}.checkpoint",
                "finite checkpoint sets require an Evaluate subject", name,
            )
        members = []
        for key, run in keyed_runs:
            try:
                selector = _checkpoint(rule)
            except ValueError as error:
                return ProjectDiagnostic(
                    ResolutionCode.BAD_REFERENCE, f"{path}.checkpoint", str(error), name,
                )
            if isinstance(run, PlannedAnalyzeRun):
                assert reference.element is not None
                element = run.elements.get(reference.element)
                if element is None or not isinstance(element.outputs.get(role.product_role), PlotlyFigurePort):
                    address = f"{run.key.variant}/{run.key.step}/{run.key.row}/{reference.element}"
                    return ProjectDiagnostic(
                        ResolutionCode.BAD_REFERENCE, f"{path}.element",
                        f"{address} has no planned {role.product_role!r} product", name,
                    )
                request = element.request_id
            else:
                request = run.request_id
            members.append((
                PlannedInput(
                    run.key,
                    InputMember(
                        key=key,
                        reference=ProductRef(
                            request_id=request,
                            role=role.product_role,
                            select=selector,
                        ),
                    ),
                    producer_element=reference.element,
                ),
                run,
            ))
        source = None
        if terms is not None and role.name == "subject":
            if terms.best is not None:
                source = SubsetSource(
                    self.subset_origin(variant, name, role.name, subset_document, "best"), reference.step,
                )
            elif reference.row is None and terms.rows is not None:
                source = SubsetSource(
                    self.subset_origin(variant, name, role.name, subset_document, "rows"), reference.step,
                    tuple(SelectedRow(key=key, variant=run.key.variant, row=run.key.row) for key, run in keyed_runs),
                )
        subset = (
            PlannedSubset(
                deepcopy(terms), candidate_count,
                tuple(run.key for _, run in keyed_runs), subset_document,
                source=source,
            )
            if terms is not None
            else None
        )
        if terms is not None and (terms.best is not None or terms.when is not None):
            return self.rank_input(name, template, role, tuple(members), subset, reference)
        return tuple(members), {}, subset


def compile_project(
    project: Project,
    *,
    readers: BaseReaders = BaseReaders(),
    source_runs: Iterable[RunKey] = (),
    memo: ProjectPlanMemo | None = None,
    operations: OperationRegistry | None = None,
    components: ComponentRegistry | None = None,
    functions: FunctionRegistry | None = None,
    python_tasks: PythonTaskRegistry | None = None,
    capture_sources: bool = False,
) -> ProjectPlan:
    """Plan static requests and value sources. Analyze lowering uses the supplied operations."""
    readers = replace(
        readers,
        external=cache(readers.external) if readers.external is not None else None,
        figure_external=cache(readers.figure_external) if readers.figure_external is not None else None,
    )
    if capture_sources or memo is None or any(
        reader is not None for reader in (readers.run, readers.external, readers.row, readers.figure_external)
    ):
        return _Compiler(
            project, readers, source_runs, operations=operations, components=components,
            functions=functions, python_tasks=python_tasks, capture_sources=capture_sources,
        ).plan()
    authority = _compiler_authority(components, operations, functions, python_tasks)
    ticket, previous = memo._begin(authority)
    compiler = _Compiler(project, readers, source_runs, previous, operations, components, functions, python_tasks)
    plan = compiler.plan()
    if authority == _compiler_authority(components, operations, functions, python_tasks):
        memo._publish(ticket, authority, compiler.work)
    return plan


@dataclass(frozen=True)
class PlanRequest:
    """The authored address and producer attribution of a planned request."""

    key: RunKey | ElementKey
    request_id: RequestId
    element_inputs: Mapping[str, ElementKey]

    @property
    def kind(self) -> RequestKind:
        return cast(RequestKind, self.request_id.split(":", 1)[0])

    @property
    def path(self) -> str:
        return self.key.payload_path(self.kind) if isinstance(self.key, ElementKey) else "payload"


@dataclass(frozen=True)
class PlanRequests:
    """The request objects and authored addresses used in a Plan comparison."""

    entries: Mapping[RunKey | ElementKey, PlanRequest]
    requests: Mapping[RequestId, bytes]
    documents: Mapping[str, bytes]
    subset_sources: Mapping[tuple[str, str, str], SubsetSource] = field(default_factory=dict)
    source_index: PlanSourceIndex | None = None

    @classmethod
    def from_plan(cls, plan: ProjectPlan) -> PlanRequests:
        return cls._from_requests(plan, (*plan.runs, *plan.elements.values()))

    @classmethod
    def _from_requests(
        cls, plan: ProjectPlan, requests: Iterable[PlannedRun | PlannedElement],
    ) -> PlanRequests:
        requests = tuple(requests)
        entries = {
            request.key: PlanRequest(
                request.key,
                request.request_id,
                {
                    role: element
                    for role, binding in request.inputs.items()
                    if (element := binding.element) is not None
                },
            )
            for request in requests
        }
        sources = {
            key: subset.source for key, subset in plan.subsets.items()
            if subset.source is not None
        }
        steps = {(key.variant, key.step) for key in entries if isinstance(key, RunKey)} if sources else set()
        subset_sources = {key: source for key, source in sources.items() if key[:2] in steps}
        source_index = None
        if plan.source_snapshot is not None:
            producers = {
                request.key: sources for request in requests
                if (sources := {
                    role: source for role, binding in request.inputs.items()
                    if (source := _upstream_source(binding)) is not None
                })
            }
            source_index = PlanSourceIndex(plan.source_snapshot, producers)
        return cls(entries, plan.requests, plan.documents, subset_sources, source_index)


def diff_analyze_elements(before: ProjectPlan, after: ProjectPlan) -> Mapping[ElementKey, ElementChange]:
    """Compare element requests and locate the authored edit or changed input."""
    changes = diff_plan_requests(
        PlanRequests._from_requests(before, before.elements.values()),
        PlanRequests._from_requests(after, after.elements.values()),
    )
    return MappingProxyType({
        key: change for key, change in changes.items() if isinstance(key, ElementKey)
    })


def diff_plan_requests(
    before: ProjectPlan | PlanRequests, after: ProjectPlan | PlanRequests,
) -> Mapping[RunKey | ElementKey, ElementChange]:
    """Compare all authored request addresses in two plans."""
    if isinstance(before, ProjectPlan):
        before = PlanRequests.from_plan(before)
    if isinstance(after, ProjectPlan):
        after = PlanRequests.from_plan(after)
    changes = {}
    for key in dict.fromkeys((*before.entries, *after.entries)):
        left, right = before.entries.get(key), after.entries.get(key)
        if left is None:
            changes[key] = ElementChange("added")
        elif right is None:
            changes[key] = ElementChange("removed")
        elif left.request_id == right.request_id:
            changes[key] = ElementChange("unchanged")
        elif left.kind != right.kind:
            changes[key] = ElementChange("changed", right.path)
        else:
            changes[key] = _request_change(before, after, left, right)
    return MappingProxyType(changes)


def _request_change(
    before: PlanRequests, after: PlanRequests, left: PlanRequest, right: PlanRequest,
) -> ElementChange:
    old = request_identity.decode_request(before.requests[left.request_id], before.documents.__getitem__)
    new = request_identity.decode_request(after.requests[right.request_id], after.documents.__getitem__)
    change = first_request_change(old, new)
    if change.path == "inputs":
        role = next(
            role for role in sorted(old.inputs.keys() | new.inputs.keys())
            if old.inputs.get(role) != new.inputs.get(role)
        )
        cause = right.element_inputs.get(role)
        if cause != left.element_inputs.get(role):
            cause = None
        input_path = f"inputs.{role}"
        parts = ("inputs", role)
        if isinstance(right.key, ElementKey):
            return ElementChange(
                "changed", f"{right.path}.{input_path}", cause,
                _review_source(before, left.key, parts), _review_source(after, right.key, parts),
            )
        source_key = right.key.variant, right.key.step, role
        old_source = before.subset_sources.get(source_key)
        new_source = after.subset_sources.get(source_key)
        return ElementChange(
            "changed", input_path, cause,
            before_source=old_source.origin if old_source is not None else _review_source(before, left.key, parts),
            after_source=new_source.origin if new_source is not None else _review_source(after, right.key, parts),
        )
    if not change.parts:
        return ElementChange("changed", f"{right.path}.{change.path}")
    relative = change.parts[1:]
    if relative == ("analysis_type",):
        relative = ("operation",)
    elif right.kind == "render" and relative[:1] == ("document",):
        relative = ("figure", *relative[1:])
    elif right.kind == "report" and relative[:1] == ("document",):
        relative = relative[1:]
    prefix = ("payload",)
    if isinstance(right.key, ElementKey):
        prefix += ("operations" if right.kind == "analyze" else "views", right.key.name)
    parts = (*prefix, *relative)
    return ElementChange(
        "changed", ".".join(parts),
        before_source=_review_source(before, left.key, parts),
        after_source=_review_source(after, right.key, parts),
    )


def _review_source(
    plan: PlanRequests, key: RunKey | ElementKey, parts: tuple[str, ...],
) -> ValueSource | None:
    if plan.source_index is None:
        return None
    if (source := plan.source_index.source(key, parts)) is not None:
        return source
    if plan.entries[key].kind == "evaluate" and parts[:3] == ("payload", "params", "trials"):
        for rule in (("ids", *parts[3:]), ("count",)):
            if (source := plan.source_index.source(key, ("payload", "params", "trial_set", *rule))) is not None:
                return source
        return plan.source_index.source(key, ("inputs", "subject"))
    return None
