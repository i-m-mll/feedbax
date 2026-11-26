"""Compile authored project requests and resolve explicit Store subjects."""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import cache, cached_property, partial
from threading import Lock
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

from pydantic import Field, TypeAdapter, ValidationError
from pydantic_core import to_json

from feedbax.application.composition import application_python_tasks, load_application
from feedbax.checkpoint_coordinates import MAX_CHECKPOINT_COORDINATE
from feedbax.plugins.declarations import UnknownDeclaration, missing_declaration
from feedbax.contracts import request_identity
from feedbax.contracts._request_codec import same_typed_value
from feedbax.contracts.analysis_document import AnalyzeDatasetInput, AnalyzeInput, AnalyzeModelInput, AnalyzeTaskInput
from feedbax.contracts.base import sha256_bytes
from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.dataset import replica_coordinate
from feedbax.contracts.expression_values import Level, Problem, ScopedValue, TextSpan, UnresolvedValue, ValueBase
from feedbax.contracts.figure_document import PLOTLY_FIGURE_ROLE, FigureDocument, PlotlyFigurePort
from feedbax.contracts.intent_diff import first_request_change
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.loaded_model import IMPORTED_MODEL_MEDIA_TYPE, ImportedModelSource
from feedbax.contracts.model_address import AttachmentElement, ComponentElement, FieldElement, ModelAddress
from feedbax.contracts.model_attachment import MODEL_ATTACHMENTS, MODEL_DIFFERENCE_BATCHES, ModelDifference, SetField
from feedbax.contracts.matrix_core import apply_stored_patch_in_place, apply_value_patch_in_place
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.stored_model import STORED_MODEL_DOCUMENT_SCHEMA_ID, StoredModelDocument
from feedbax.contracts.model_input import DocumentInput, InitialModelSource, authored_document_input, encode_document_input
from feedbax.contracts.objective_document import ObjectiveDocument
from feedbax.contracts.model_values import model_owner_problem
from feedbax.contracts.operation import DatasetPort
from feedbax.contracts.studio_authoring import InputProducer, InputRole
from feedbax.contracts.studio_place import StudioTaskStart, path_parts, task_start, variation_targets
from feedbax.contracts.project import (
    DifferenceNotApplied, Differences, Document, DocumentReference, InlineBase, LoadedModelDocument, Payload, Project, RootDocument, Step,
    StepBase, StepKind,
    apply_differences, base_chain, diff, json_pointer_parts, member_name, nearest_shared_base,
)
from feedbax.contracts.run_rule import RuleRefusal, bind_metric_conditions
from feedbax.contracts.project_storage import stored_keyed_collection
from feedbax.contracts.expression_text import RuleBest
from feedbax.contracts.project_inputs import (
    BarrierCheckpoint,
    BatchCheckpoint,
    BestCheckpoint,
    CheckpointRule,
    CheckpointBatches,
    CheckpointCadence,
    DocumentInputReference,
    InitialModelReference,
    InputBinding,
    LoadedModelReference,
    NamedSubset,
    OmittedReference,
    ObjectiveDocumentReference,
    ProjectName,
    Reference,
    RunReference,
    BoundRank,
    SelectedAxisValueRefusal,
    StepReference,
    SubsetTerms,
    TaskDocumentReference,
    producer_member_key,
    select_step_rows,
)
from feedbax.contracts.project_resolution import (
    BaseReaders,
    LoadedModelRead,
    ProjectDiagnostic,
    ResolutionCode,
    ResolvedStep,
    RunKey,
    StepComposition,
    StepTemplate,
    ResolvedImportedModel, TerminalTrainSource,
    _TrialResolution,
    compose_document,
    compose_step,
    composed_loaded_model_projection,
    reference_write_target,
    _select,
)
from feedbax.contracts.result_address import (
    AddressProblem, ElementHolders, analyze_cards, payload_elements, shared_element_keys,
)
from feedbax.contracts.rollout import (
    ROLLOUT_TYPE, CannotRun, RolloutParameters, RolloutSettings, VariationEdits,
    authored_rollout_settings, authored_variation, resolve_trial_ids,
)
from feedbax.contracts.retained_state import RetainedStateField
from feedbax.contracts.rollout_dataset import common_rollout_port, rollout_dataset_port, rollout_update_axis
from feedbax.contracts.project_sweep import (
    CHECKPOINT_RULE_TARGET,
    CheckpointRuleAxis,
    DEFAULT_SWEEP_ROW,
    ExpandedRow,
    GeneratedAxis,
    InputAxis,
    NamedAxis,
    SetAxis,
    Sweep,
    SweepCoordinate,
    SweepInputCoordinate,
    SweepExpansionCode,
    SweepExpansionError,
    SweepInputMember,
    axis_coordinates,
    checkpoint_rule_write,
    checkpoint_sweep_refusal,
    numbered_run_name, sweep_row_name,
    sweep_value_name,
    expand_sweep,
    model_field_target,
    sweep_patch,
)
from feedbax.contracts.project_sources import ValueSource, changed_sources, patch_source, place_source, pointer, source_shape
from feedbax.contracts.report_document import ReportDocument
from feedbax.contracts.set_document import SetProblem, document_subject_model, model_slot, resolve_set_axes, trial_model_path
from feedbax.contracts.request_identity import EncodedRequest, RequestDocumentMemo
from feedbax.contracts.request_refs import (
    BarrierMember,
    BatchMember,
    BatchSetMember,
    BestMember,
    BestCandidate,
    BestRank,
    ConditionalProduct,
    ContentRef,
    InputMember,
    LastMember,
    MemberLabel,
    ProductRef,
    ProductSelector,
    ReferenceValue,
    RequestId,
    RequestKind,
    UpdateCadence,
    UpdateSeriesMember,
    best_rank_key,
    is_synthesized_model_product,
)
from feedbax.contracts.request_specs import (
    DocumentTrainingConfig,
    DocumentTrainingRequestSpec,
    SynthesisRequestSpec,
    TrainContinuation,
    ReportRequestSpec,
    TrialDocuments,
)
from feedbax.contracts.synthesis import SynthesisStepPayload
from feedbax.contracts.task_document import (
    PYTHON_TASK_SCHEMA_ID, PythonTaskSelection, StoredTaskDocument, TaskAdditions, TaskDocument, add_to_task,
)
from feedbax.contracts.training import lower_stored_trial_documents
from feedbax.contracts.train_settings import normalize_train_method, setting_field_title

if TYPE_CHECKING:
    from feedbax.compiler.model_attachment import LoweredEvaluateAttachments
    from feedbax.compiler.set_fields import SetFieldProblem
    from feedbax.analysis.operations import OperationRegistry
    from feedbax.contracts.effective_success import TerminalTrainSubject
    from feedbax.contracts.request_specs import RequestSpec, TrialDocuments
    from feedbax.tasks.registry import PythonTaskRegistry
    from feedbax.training.methods import TrainingMethodRegistry
    from feedbax.component_registry import ComponentRegistry
    from feedbax.compiler.model_document import LoweredModelDocument
    from feedbax.contracts.expression_registry import FunctionRegistry
    from feedbax.plot.templates import FigureTemplateRegistry
    from feedbax.synthesis.methods import SynthesisMethodRegistry


@cache
def _builtin_model_registry() -> ComponentRegistry:
    from feedbax.component_registry import ComponentRegistry

    registry = ComponentRegistry()
    registry.seal()
    return registry


@dataclass(frozen=True)
class ElementKey:
    """One request of an Analyze element: the element itself, or one combination of the labels it maps over."""

    run: RunKey
    name: str
    member: str | None = None

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
    rows: tuple[BoundRank, ...] = ()


@dataclass(frozen=True)
class PlannedSubset:
    terms: SubsetTerms
    candidates: int
    members: tuple[RunKey, ...]
    document: str | None
    rank_groups: tuple[PlannedRankGroup, ...] = ()
    source: SubsetSource | None = None
    metric_conditions: Mapping[RunKey, ConditionalProduct] | None = None
    best: RuleBest | None = None
    """The rule's lowest or highest term, which ranks the members once they have run."""

    @property
    def conditional(self) -> bool:
        return self.metric_conditions is not None

    @property
    def ranks(self) -> tuple[str, ...]:
        return tuple(rank for group in self.rank_groups for rank in group.ranks)


STEP_LABELS = ("source", "model", "element")
"""The member labels whose values are named things of the project; a request identifies them by content.

``source`` is the step whose states the data came from, ``model`` the Train step, and ``element`` the operation
output a listed member came from."""

RESERVED_LABELS = frozenset({*STEP_LABELS, "variant", "member"})


@dataclass(frozen=True)
class LabelName:
    """What a coordinate value names in the project."""

    key: str
    label: str | None = None
    kind: Literal["step", "row", "run", "document", "element"] = "step"
    card: str | None = None
    """The Analyze card holding an element, which tells apart elements of one key in two cards."""

    @property
    def printed(self) -> str:
        return self.label or {
            "step": "Step", "row": "Run", "run": "Recorded run", "document": "Document", "element": "Element",
        }[self.kind]


type LabelNames = Mapping[str, Mapping[str, LabelName]]
"""Each named coordinate's value IDs in one request, and what each names."""


def label_value_names(labels: Mapping[str, MemberLabel], names: LabelNames) -> dict[str, str]:
    """Each label's value by name: the key of the step or element a step-valued value names, else its value id."""
    return {
        label: named.key if (named := names.get(label, {}).get(value.value_id)) is not None else value.value_id
        for label, value in labels.items()
    }


class PendingProduct(ReferenceValue):
    """The product of a row that has no request, which a reading row waits on."""

    variant: str
    step: str
    row: str

    @property
    def run(self) -> RunKey:
        return RunKey(self.variant, self.step, self.row)


class PendingMember(ReferenceValue):
    """A member whose producer has no request: it keeps its key and labels, and no request reads it."""

    key: str
    reference: PendingProduct
    labels: dict[str, MemberLabel] = Field(default_factory=dict)


@dataclass(frozen=True)
class PlannedInput:
    producer: RunKey | None
    member: InputMember | PendingMember
    producer_element: str | None = None
    producer_member: str | None = None
    """The mapped request of ``producer_element`` this member reads, when the element maps over labels."""
    candidates: tuple[RunKey, ...] = ()
    terminal: TerminalTrainSubject | None = None
    coordinates: Mapping[str, SweepCoordinate] = field(default_factory=dict)
    """What the producer row is about, carried by an input axis and paired by a zip."""
    selection: StepReference | None = field(default=None, compare=False)
    """The authored selection that chose this member."""
    named: Mapping[str, LabelName] = field(default_factory=dict)
    """What the member's step-valued labels name. The request identifies each by content (`_identified`)."""
    model_replicas: tuple[str, ...] | None = None
    """Replica IDs of a document-bound Model available before executing the operation."""
    best_path: str = field(default="inputs.subject.where.rule", compare=False)
    """Where a ranked member's Best is authored: its role's, or its list item's."""

    @property
    def sources(self) -> tuple[RunKey, ...]:
        return (self.producer,) if self.producer is not None else self.candidates

    @property
    def request_member(self) -> InputMember:
        """The member a request reads. A row that reads a pending member is blocked and never lowered."""
        assert isinstance(self.member, InputMember), f"lowered a member waiting on {self.waits_on}"
        return self.member

    @property
    def waits_on(self) -> RunKey | None:
        """The producer row that has no request, when this member's product waits on one."""
        return self.member.reference.run if isinstance(self.member, PendingMember) else None

    @property
    def element(self) -> ElementKey | None:
        return (
            ElementKey(self.producer, self.producer_element, self.producer_member)
            if self.producer and self.producer_element else None
        )


def plan_document_input(
    reference: DocumentInputReference, document: RootDocument, objects: MutableMapping[str, bytes],
    *, python_tasks: PythonTaskRegistry | None = None,
) -> PlannedInput:
    return _plan_document_content(authored_document_input(reference, document, python_tasks=python_tasks), objects)


def _plan_document_content(value: DocumentInput, objects: MutableMapping[str, bytes]) -> PlannedInput:
    content, raw = encode_document_input(value)
    objects[content.sha256] = raw
    return PlannedInput(None, InputMember(key=DEFAULT_SWEEP_ROW, reference=content))


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
class TrainFacts:
    """What a Train row's Task and run controls say, whether or not its request lowers."""

    task: TaskDocument | PythonTaskSelection | None
    """None when the Task does not resolve."""
    origin: int | None
    """The completed-update count the run starts from: 0 when fresh; None when the run controls do not resolve."""
    target_total: int | None
    """The completed-update count at the run's end; None when the run controls do not resolve."""
    input_bindings: dict[str, str | None] = field(default_factory=dict)
    """Authored choices before implicit matches are bound in the Train request."""


_CHECKPOINT_INITIALIZATION: TypeAdapter[Literal["initialize_from"] | TrainContinuation | None] = TypeAdapter(
    Literal["initialize_from"] | TrainContinuation | None,
)


def _train_facts(payload: Payload, python_tasks: PythonTaskRegistry) -> TrainFacts:
    """Read the authored Task and run controls, resolving Python task selections."""
    from feedbax.tasks.registry import resolve_python_task_instance

    trial = payload.get("trial_documents")
    task: TaskDocument | PythonTaskSelection | None = None
    bindings = {}
    if isinstance(trial, dict):
        raw_bindings = trial.get("input_bindings", {})
        if isinstance(raw_bindings, dict):
            bindings = {
                port: signal for port, signal in raw_bindings.items()
                if isinstance(port, str) and (signal is None or isinstance(signal, str))
            }
        try:
            if trial.get("task") is not None:
                task = StoredTaskDocument.model_validate(trial["task"]).lower()
            elif trial.get("python_task") is not None:
                selection = PythonTaskSelection.model_validate(trial["python_task"])
                resolved = resolve_python_task_instance(selection, python_tasks)
                task = None if isinstance(resolved, tuple) else request_identity.canonical_python_task_selection(
                    resolved.selection(),
                )
        except ValidationError:
            task = None
    try:
        batches = DocumentTrainingConfig.model_validate(payload.get("training_config")).n_batches
        start = _CHECKPOINT_INITIALIZATION.validate_python(payload.get("checkpoint_initialization"))
    except ValidationError:
        return TrainFacts(task, None, None, bindings)
    origin = start.source_completed_updates if isinstance(start, TrainContinuation) else 0
    return TrainFacts(task, origin, origin + batches, bindings)


@dataclass(frozen=True)
class PlannedRow:
    """An expanded row, including a row whose request validation fails."""

    key: RunKey
    kind: StepKind
    label: str | None
    coordinates: Mapping[str, SweepCoordinate]
    axis_keys: Mapping[str, str] = field(default_factory=dict, kw_only=True)
    """Each axis of the row's sweep and its value id there, or for an input axis the member key, which an address's
    selector matches."""
    input_axes: Mapping[str, SweepInputCoordinate] = field(default_factory=dict, kw_only=True)
    """Each input axis's member on this row; ``axes`` terms select it by its key."""
    kept_axes: tuple[str, ...] = field(default=(), kw_only=True)
    """Optional axes for which this row is the unvalued member."""

    step_name: str | None = field(default=None, kw_only=True)
    number: int = field(default=1, kw_only=True)
    """The display name of the step when this is its only row."""

    @property
    def name(self) -> str:
        return self.step_name or self.label or numbered_run_name(self.number)

    @property
    def selection_coordinates(self) -> dict[str, SweepCoordinate]:
        return axis_coordinates(self.coordinates, self.input_axes, self.kept_axes)

    document_ids: Mapping[str, str] = field(default_factory=dict, kw_only=True)
    document_uses: frozenset[str] = field(default=frozenset(), kw_only=True)
    # Documents an Evaluate row reads through its subject run rather than naming them.
    followed_documents: frozenset[str] = field(default=frozenset(), kw_only=True)
    # The Model document an Evaluate row names, and its subject run's Model document.
    evaluated_model: tuple[str, str] | None = field(default=None, kw_only=True)
    train: TrainFacts | None = field(default=None, kw_only=True)
    resolved: bool = field(default=True, kw_only=True)
    """Whether the row's definition resolves; one whose own patches or documents are refused has none."""
    problems: tuple[RunProblem, ...] = field(default=(), kw_only=True)
    """Why the row cannot run, each at its own path: Problems in its documents, and its admission problems."""


@dataclass(frozen=True)
class BlockedRow(PlannedRow):
    """A row an input of which waits on a producer row that has no request; it is never lowered."""

    inputs: Mapping[str, tuple[PlannedInput, ...]]
    blocked_by: RunKey


@dataclass(frozen=True)
class RunProblem:
    path: str
    """A path in the row's request, or in its definition's ``inputs``, as a JSON pointer or a dotted path."""
    detail: str
    model_address: ModelAddress | None = None
    variable: str | None = None
    """The Task variable a link at ``path`` belongs to."""
    start: StudioTaskStart | None = None
    """The Model value a link at ``path`` is missing for."""
    variation_target: str | None = None
    """The dotted path the variation entry at ``path`` patches, which places the problem."""
    sweep_source: ValueSource | None = None
    """The authored sweep member that supplied the Model field at this Problem."""


@dataclass(frozen=True)
class PlannedRun(PlannedRow):
    request_id: RequestId
    inputs: Mapping[str, PlannedInput]
    omissions: Mapping[str, str]
    requires: tuple[str, ...]
    dataset_port: DatasetPort | None = None
    metric_names: frozenset[str] = frozenset()

    @property
    def admission_problem(self) -> RunProblem | None:
        """The reason the run cannot be admitted: the first of its problems."""
        return self.problems[0] if self.problems else None


def common_train_metrics(rows: Iterable[PlannedRow]) -> frozenset[str]:
    names = [row.metric_names for row in rows if isinstance(row, PlannedRun) and row.kind == "train"]
    return frozenset.intersection(*names) if names else frozenset()


@dataclass(frozen=True)
class PlannedElement:
    key: ElementKey
    kind: Literal["analyze", "render"]
    request_id: RequestId
    inputs: Mapping[str, tuple[PlannedInput, ...]]
    outputs: Mapping[str, DatasetPort | PlotlyFigurePort]
    names: LabelNames = field(default_factory=dict)
    """What the step-valued label values of this request and of the operations it reads name."""
    mapped: Mapping[str, MemberLabel] = field(default_factory=dict)
    """The labels this request is one combination of, when its element maps over them; a reader's member carries
    them."""
    entries: tuple[SweepInputMember, ...] = ()
    """The labelled entries its dataset outputs carry from the stacked inputs it takes whole."""
    label: str | None = None
    """The operation's or figure's display label, which ``element`` prints for an output another card lists."""

    @property
    def stack_key(self) -> str:
        """The key of a reader's member that reads this request: its mapped labels by value id, so that only an edit
        of what the request reads changes it."""
        return ",".join(f"{label}={value.value_id}" for label, value in self.mapped.items()) or "default"

    @property
    def member_labels(self) -> dict[str, str]:
        """Each label this request is one combination of, and the name of its value: a step or element key, or a
        sweep value id."""
        return label_value_names(self.mapped, self.names)

    @property
    def datasets(self) -> Mapping[str, DatasetPort]:
        return MappingProxyType({
            role: port for role, port in self.outputs.items() if isinstance(port, DatasetPort)
        })

    @property
    def path(self) -> str:
        return self.key.payload_path(self.kind)


def planned_input_groups(
    node: PlannedRun | PlannedElement,
) -> Mapping[str, tuple[PlannedInput, ...]]:
    if isinstance(node, PlannedElement):
        return node.inputs
    return {role: (binding,) for role, binding in node.inputs.items()}


@dataclass(frozen=True)
class PlannedAnalyzeRun(PlannedRow):
    elements: Mapping[ElementKey, PlannedElement]
    """Each request of each element, one per element unless the element maps over labels."""
    requires: tuple[str, ...]
    inputs: Mapping[str, tuple[PlannedInput, ...]]
    figures: Mapping[str, FigureDocument]
    dataset_inputs: Mapping[str, DatasetPort]
    """The resolved ports of this row's dataset inputs, including stack coordinates."""
    dataset_outputs: Mapping[str, Mapping[str, DatasetPort]]
    """Each operation's readable dataset products, with the labels of their stacked members."""

    def requests(self, name: str) -> tuple[PlannedElement, ...]:
        """The requests of the element ``name``, in plan order."""
        return tuple(element for key, element in self.elements.items() if key.name == name)


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
    label: str = field(kw_only=True)
    can_sweep_checkpoint: bool = False
    unexpanded_axis_titles: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectedRunDefinition:
    payload: Payload
    sources: Mapping[str, ValueSource]
    document_ids: Mapping[str, str]
    document_uses: frozenset[str]
    document_pointers: Mapping[str, str] = field(default_factory=dict)
    document_problems: tuple[ProjectDiagnostic, ...] = ()
    input_documents: Mapping[str, RootDocument] = field(default_factory=dict)
    pinned_model: ModelDocument | None = None
    task_additions: tuple[TaskAdditions, ...] = ()


@dataclass(frozen=True)
class SourceSweep:
    inputs: dict[str, tuple[SweepInputMember, ...]] = field(default_factory=dict)
    integer_sites: dict[str, bool] = field(default_factory=dict)
    sets: dict[str, NamedAxis] = field(default_factory=dict)


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
        return _Compiler(self.snapshot.project, readers, (), python_tasks=application_python_tasks())

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
                checkpoint_subject=project.steps[step].kind == "evaluate", sets=sweep.sets,
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
            kind = self._resolver.variants[run.variant].steps[run.step].kind
            model = {"train": ("payload", "trial_documents", "model"), "synthesize": ("payload", "model")}.get(kind)
            if model is not None and parts[:len(model)] == model and template.has_model_differences(row):
                return self._model_field_source(template, run, row, parts)
            return template.source_leaf(row, parts[1:])
        except (KeyError, ValueError) as error:
            raise ValueError(f"Invalid saved source for {run.variant}/{run.step}/{run.row}: {error}") from error

    def expanded_step(self, variant: str, step: str) -> tuple[StepTemplate, Mapping[str, ExpandedRow]]:
        """The authored template and expanded rows used to resolve the step's sources."""
        return self._templates(variant, step)

    def _model_field_source(
        self, template: StepTemplate, run: RunKey, row: ExpandedRow, parts: tuple[str, ...],
    ) -> ValueSource | None:
        definition = self._resolver.project_definition(template.composition, run, row, {})
        if definition is None:
            raise ValueError("Cannot resolve the managed Model sources")
        payload, model = self._resolver.lower_producer_model(
            run, definition.payload, definition.document_ids, definition.document_pointers,
        )
        if isinstance(payload, ProjectDiagnostic):
            raise ValueError(payload.detail)
        if model is not None:
            definition = _producer_model_definition(definition, model)
        return definition.sources.get(pointer(parts))


def own_task_variation(subject_task: TaskDocument | PythonTaskSelection, task: Any, variation: Any) -> list[Any] | None:
    """The variation an Evaluate step that names its own Task, of content ``task``, runs: its authored ``variation``
    after a replacement of the subject's Task, or None when ``task`` is the subject's."""
    try:
        if type(subject_task).model_validate(task) == subject_task:
            return None
    except ValidationError:
        pass
    if isinstance(variation, dict):
        variation = [variation[name] for name in sorted(variation)]
    if not isinstance(variation, list):
        raise ValueError("Declare the variation as a list.")
    return [{"op": "replace", "path": "task", "value": task}, *variation]


_AUTHORED_EVALUATION_TASK = TypeAdapter(StoredTaskDocument | PythonTaskSelection)


def resolve_authored_evaluation_task(
    task: StoredTaskDocument | PythonTaskSelection, variation: VariationEdits,
) -> StoredTaskDocument | PythonTaskSelection:
    """Apply Evaluate Task edits in authored member order and address space."""
    for origin, edit in authored_variation(variation):
        root, _, relative = edit.path.partition(".")
        if root != "task":
            continue
        try:
            if not relative and edit.op == "replace":
                task = _AUTHORED_EVALUATION_TASK.validate_python(edit.value)
            elif relative and (isinstance(task, StoredTaskDocument) or relative.startswith("params.")):
                content = task.model_dump(mode="json", exclude_unset=True)
                apply_stored_patch_in_place({"task": content}, edit)
                task = type(task).model_validate(content)
            else:
                raise ValueError("expected an addressed Task edit")
        except (ValueError, KeyError, OverflowError) as error:
            raise CannotRun(origin, str(error)) from error
    return task


def _lower_evaluation_task_edits(
    raw: dict, source: TaskDocument | PythonTaskSelection,
    stored_source: StoredTaskDocument | None,
) -> dict:
    settings = authored_rollout_settings(raw)
    edits = tuple(authored_variation(settings.variation))
    task_edits = tuple((origin, edit) for origin, edit in edits if edit.path.partition(".")[0] == "task")
    if not task_edits:
        return raw
    if stored_source is None and isinstance(source, TaskDocument):
        for origin, edit in task_edits:
            if edit.path == "task" and edit.op == "replace":
                break
            if edit.path.startswith("task.validation_trials."):
                raise CannotRun(origin, "Name a Task document to edit its authored validation trial keys.")
    authored = stored_source if stored_source is not None else (
        StoredTaskDocument.from_lowered(source) if isinstance(source, TaskDocument) else source
    )
    selected = resolve_authored_evaluation_task(authored, settings.variation)
    lowered = selected.lower().without_markers() if isinstance(selected, StoredTaskDocument) else selected
    graph_edits = [edit.model_dump(mode="json", exclude_unset=True) for _, edit in edits
                   if edit.path.partition(".")[0] != "task"]
    return raw | {"variation": [
        {"op": "replace", "path": "task", "value": lowered.model_dump(mode="json")}, *graph_edits,
    ]}


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
    source_snapshot: SourceSnapshot | None = field(default=None, repr=False, compare=False)
    exception_matches: Mapping[tuple[str, str], Mapping[str, tuple[str, ...]]] = field(default_factory=dict)
    variant_projects: Mapping[str, Project] = field(default_factory=dict, repr=False, compare=False)
    """Each variant's project, its row applied; a variant whose row does not apply has none."""

    def detached(self) -> ProjectPlan:
        """Return this plan with its own copies of the selected payloads, which callers may change."""
        return replace(self, selected_definitions=MappingProxyType({
            key: replace(definition, payload=deepcopy(definition.payload))
            for key, definition in self.selected_definitions.items()
        }))

    def row(self, key: RunKey) -> PlannedRow | None:
        return self._rows_by_key.get(key)

    @cached_property
    def _rows_by_key(self) -> Mapping[RunKey, PlannedRow]:
        return MappingProxyType({row.key: row for row in self.rows})

    @property
    def elements(self) -> Mapping[ElementKey, PlannedElement]:
        return MappingProxyType({
            key: element
            for row in self.rows if isinstance(row, PlannedAnalyzeRun)
            for key, element in row.elements.items()
        })

    @property
    def light_requests(self) -> Mapping[RunKey | ElementKey, PlannedRun | PlannedElement]:
        requests: dict[RunKey | ElementKey, PlannedRun | PlannedElement] = {
            run.key: run for run in self.runs if run.kind in {"report", "synthesize"}
        }
        requests.update(self.elements)
        return requests

    @cached_property
    def evaluated_models(self) -> frozenset[tuple[str, str]]:
        """Each Model document an Evaluate row names, paired with its subject run's Model document."""
        return frozenset(row.evaluated_model for row in self.rows if row.evaluated_model is not None)

    def document_users(self, variant: str, document: str) -> frozenset[str]:
        return frozenset(
            row.key.step for row in self.rows
            if row.key.variant == variant and document in row.document_uses - row.followed_documents
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
    gathers: bool = False
    """Whether the role takes a list of selections."""
    operation_outputs: bool = False
    """Whether a selection may instead name an Analyze operation's dataset product, read from the card that holds it."""
    documents: tuple[Literal["model", "task", "loaded_model"], ...] = ()
    """The documents the role may read instead of a step's rows."""
    checkpoint_sets: bool = False
    """Whether a selection's checkpoint may name a finite set of updates, each a row of the reading step."""

    @property
    def producers(self) -> tuple[InputProducer, ...]:
        primary = InputProducer(
            kind=self.producer_kind, product=self.product_role, operation_outputs=self.operation_outputs,
        )
        if self.producer_kind == "train" and self.product_role == "checkpoint":
            return primary, InputProducer(kind="synthesize", product="model", operation_outputs=False)
        return (primary,)

    @property
    def entry(self) -> InputRole:
        return InputRole(
            required=self.required, gathers=self.gathers,
            producers=list(self.producers),
            documents=list(self.documents), checkpoint_sets=self.checkpoint_sets,
        )


_ROLES = {
    "train": _Role("checkpoint", False, "train", "checkpoint", documents=("loaded_model",)),
    "evaluate": _Role(
        "subject", True, "train", "checkpoint", gathers=True, documents=("model", "loaded_model"), checkpoint_sets=True,
    ),
}
_ANALYZE_TASK_ROLE = InputRole(required=True, gathers=False, producers=[], documents=["task"], checkpoint_sets=False)


def _analyze_role(name: str, declaration: AnalyzeInput) -> _Role | None:
    """The role an Analyze document's input declares; a Task input reads only a Task document and has none."""
    match declaration:
        case AnalyzeModelInput():
            return _Role(name, True, "train", "checkpoint", documents=("model", "loaded_model"))
        case AnalyzeDatasetInput():
            return _Role(name, True, "evaluate", "states", gathers=True, operation_outputs=True)
        case AnalyzeTaskInput():
            return None


def step_input_roles() -> dict[str, dict[str, InputRole]]:
    """Each step kind's inputs by name, as the planner resolves them."""
    return {kind: {role.name: role.entry} for kind, role in _ROLES.items()}


def analyze_input_roles() -> dict[str, InputRole]:
    """The input an Analyze document's declaration of each kind gives its step."""
    roles = {}
    for kind in ("model", "dataset", "task"):
        role = _analyze_role(kind, _ANALYZE_INPUT.validate_python({"kind": kind}))
        roles[kind] = role.entry if role is not None else _ANALYZE_TASK_ROLE
    return roles
_ANALYZE_INPUTS = TypeAdapter(dict[ProjectName, AnalyzeInput])
_ANALYZE_INPUT: TypeAdapter[AnalyzeInput] = TypeAdapter(AnalyzeInput)
DOCUMENTS: TypeAdapter[dict[str, Document]] = TypeAdapter(dict[str, Document])

type _CompiledRun = PlannedRun | PlannedAnalyzeRun
type _Members = tuple[tuple[PlannedInput, PlannedRow | None], ...]


def _identified(
    members: _Members, *, labelled: bool,
) -> tuple[_Members, dict[tuple[str, LabelName], str]]:
    """Give each step-valued label value an ID that digests the references of the members sharing it.

    The ID names no step, so Change Key and Rename leave every request alone, while members from one step still
    share one value. Two values whose members are the same requests, such as a step and its duplicate, are told
    apart by their order. A ``labelled`` role's members carry the IDs as labels, ahead of their other labels.
    """
    sharing: dict[tuple[str, LabelName], set[str]] = {}
    for binding, _ in members:
        reference = canonical_json_v2_bytes(binding.member.reference.model_dump(mode="json")).decode()
        for label, name in binding.named.items():
            sharing.setdefault((label, name), set()).add(reference)
    ids: dict[tuple[str, LabelName], str] = {}
    taken: Counter[tuple[str, str]] = Counter()
    for (label, name), references in sharing.items():
        digest = "members-" + sha256_bytes(canonical_json_v2_bytes(sorted(references)))
        taken[label, digest] += 1
        ids[label, name] = digest if taken[label, digest] == 1 else f"{digest}-{taken[label, digest]}"
    if not labelled:
        return members, ids
    return tuple(
        (replace(binding, member=binding.member.model_copy(update={"labels": {
            **{label: MemberLabel(value_id=ids[label, name]) for label, name in binding.named.items()},
            **binding.member.labels,
        }})), run) if binding.named else (binding, run)
        for binding, run in members
    ), ids


def _member_label(binding: PlannedInput, group: Mapping[str, SweepCoordinate]) -> str:
    """What a member no row produces reads as.

    A grouped rank, or a rank of a list item, reads as its step, its group's values and its rank
    (``train, lr=0.1, rank 1``); any other member as its key.
    """
    rank = binding.member.reference
    if isinstance(rank, BestRank) and (rank.group or binding.member.key != rank.key) and "source" in binding.named:
        values = (sweep_value_name(group[axis]) for axis in sorted(rank.group))
        return " · ".join((binding.named["source"].printed, *values, f"Rank {rank.rank}"))
    return next((name.printed for name in binding.named.values()), "Input")


def _labelled(role: _Role, members: _Members) -> bool:
    """Whether a role's members carry their labels: an Evaluate's always, and operation outputs when several
    elements are listed.

    One operation output read alone is the dataset the same wire reads inside one card, so a Split that cuts the
    wire changes no request.
    """
    if role.producer_kind != "evaluate":
        return False
    if any(binding.producer_element is None for binding, _ in members):
        return True
    return len({(binding.producer, binding.producer_element) for binding, _ in members}) > 1


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
        case UnresolvedValue():
            raise ValueError("Expected a checkpoint rule.")


def _update_outside_run(rule: CheckpointRule | None, origin: int, total: int) -> int | None:
    """The first update an At or List rule selects outside a run from ``origin`` to ``total`` completed updates."""
    match rule:
        case BatchCheckpoint(at=batch):
            updates: tuple[int, ...] = (batch,)
        case CheckpointBatches(batches=listed):
            updates = tuple(listed)
        case _:
            return None
    return next((update for update in updates if not origin < update <= total), None)


def _selection_path(authored: object, role: str, selection: StepReference | None) -> str:
    """Where a member's selection is authored: its role, or its item of a listed role."""
    if isinstance(authored, list) and selection is not None and selection in authored:
        return f"inputs.{role}.{authored.index(selection)}"
    return f"inputs.{role}"


def _paired_checkpoint_step(step: Step, reference: InputBinding | None) -> str | None:
    if (
        step.kind == "train"
        and isinstance(step.base, StepBase)
        and not step.base.unresolved
        and step.base.row is None
        and isinstance(reference, StepReference)
        and not reference.unresolved
        and reference.step == step.base.step
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
    document: str | None
    model: ModelDocument
    stamp: str
    evaluation_problem: ProjectDiagnostic | None = None
    evaluation_graph: GraphSpec | None = None
    evaluation_states: tuple[RetainedStateField, ...] = ()
    trained: _TrainedSource | None = None
    unpaired: bool = False
    """The Train's Model, Task and Objective do not pair, so nothing evaluates what it would train."""
    authored_model: ModelDocument | None = None
    trial: TrainFacts | None = None
    stored_task: StoredTaskDocument | None = None

    @cached_property
    def evaluation_stamp(self) -> tuple[str, str | None]:
        trial = self.trial
        return self.stamp, sha256_bytes(to_json((
            trial.task.model_dump(mode="json") if trial is not None and trial.task is not None else None,
            trial.input_bindings if trial is not None else None,
            self.stored_task.model_dump(mode="json", exclude_unset=True) if self.stored_task is not None else None,
        )))


@dataclass(frozen=True)
class _TrainedSource:
    """What a Train run's checkpoint holds, as its plan knows it."""

    graph: GraphSpec
    method: str
    replicates: int


EVALUATION_MODEL = "payload.trial_documents.model"
EVALUATION_TASK = "payload.trial_documents.task"


@dataclass(frozen=True)
class _EvaluationModel:
    """The differences an Evaluate step's Model document applies to its subject run's model."""

    document: str
    differences: Differences


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


def _with_producer_model(
    payload: Payload, model: _ModelSource, document_ids: Mapping[str, str],
    document_pointers: Mapping[str, str],
) -> Payload:
    data = StoredModelDocument.from_lowered(model.model).model_dump(mode="json", exclude_unset=True)
    draft = deepcopy(payload)
    main = trial_model_path(payload)
    apply_value_patch_in_place(draft, main, "replace", data, path=pointer(main))
    for slot, name in document_ids.items():
        if name != model.document or not slot.startswith("payload."):
            continue
        path = tuple(slot.removeprefix("payload.").split("."))
        if path != main:
            selected = _select(data, json_pointer_parts(document_pointers.get(slot)))
            apply_value_patch_in_place(draft, path, "replace", selected, path=slot)
    return draft


_MODEL_COLLECTION = partial(stored_keyed_collection, STORED_MODEL_DOCUMENT_SCHEMA_ID)


def _model_with_stored_differences(model: ModelDocument, differences: Differences) -> ModelDocument:
    stored = StoredModelDocument.from_lowered(model).model_dump(mode="json", exclude_unset=True)
    content = apply_differences(stored, differences, keyed_collection=_MODEL_COLLECTION)
    return StoredModelDocument.model_validate(content).lower()


def _producer_model_definition(definition: SelectedRunDefinition, model: _ModelSource) -> SelectedRunDefinition:
    if model.authored_model is not None:
        model = replace(model, model=model.authored_model)
    payload = _with_producer_model(definition.payload, model, definition.document_ids, definition.document_pointers)
    prefix = trial_model_path(definition.payload)
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


def _refused_difference(
    model: _EvaluationModel, trained: ModelDocument, components: ComponentRegistry, functions,
) -> ModelAddress | None:
    """Address the first of an Evaluate Model document's differences that the trained model refuses on its own."""
    from feedbax.compiler.model_attachment import lower_evaluate_attachments

    for at, difference in model.differences.items():
        try:
            evaluated = _model_with_stored_differences(trained, {at: difference})
            lower_evaluate_attachments(trained, evaluated, None, components, (), functions=functions)
        except ValueError:
            return ModelAddress.at_stored_path(json_pointer_parts(at))
    return None


def _model_difference_problem(
    key: RunKey, document: str | None, address: ModelAddress | None, detail: str, *, path: str | None = None,
) -> ProjectDiagnostic:
    return ProjectDiagnostic(
        ResolutionCode.INVALID_PAYLOAD, path or f"payload.{MODEL_ATTACHMENTS}", detail, key.step,
        row=key.row, variant=key.variant, document=document, model_address=address,
    )


def _sweep_problem(
    error: SweepExpansionError, *, step: str | None = None, variant: str | None = None,
    document: str | None = None,
) -> ProjectDiagnostic:
    if isinstance(error, SetProblem):
        path = pointer(error.location)
        return ProjectDiagnostic(
            ResolutionCode.INVALID_SWEEP, error.path, error.detail, step=step, variant=variant,
            document=error.document, document_path=path,
            authored_file=f"documents/{error.document}.json", authored_path="/payload" + path,
        )
    address = None
    if error.target is not None:
        try:
            target = model_field_target(error.target)
            if target is not None:
                address = target.address
                document = target.document or document
                step = target.step or step
        except ValueError:
            pass  # The sweep diagnostic already reports the malformed target.
    return ProjectDiagnostic(
        ResolutionCode.INVALID_SWEEP, error.path, error.detail, step=step, variant=variant,
        document=document if address is not None else None, model_address=address, span=error.span,
        authored_file=f"steps/{step}.json" if step else None,
        authored_path=pointer(error.path.split(".")),
    )


def _model_parameter(model: ModelDocument, address: ModelAddress):
    from feedbax.contracts.model_document import CompositeComponent

    element = address.element
    if not isinstance(element, FieldElement) or not isinstance(element.parent, ComponentElement):
        return None, ()
    graph = model
    for name in address.composites:
        component = graph.interior.components.get(name)
        if not isinstance(component, CompositeComponent):
            return None, ()
        graph = component
    return graph.interior.components.get(element.parent.id), element.path


def _integer_model_site(
    model: ModelDocument, address: ModelAddress, components: ComponentRegistry,
) -> bool:
    from feedbax.contracts.model_document import CompositeComponent, LeafComponent

    component, path = _model_parameter(model, address)
    if isinstance(component, CompositeComponent):
        return path == ("substeps",)
    if not isinstance(component, LeafComponent) or len(path) != 2 or path[0] != "fields":
        return False
    declaration = components.get(component.type)
    return declaration is not None and any(
        parameter.name == path[1] and parameter.type == "int" for parameter in declaration.params.param_schema
    )


def _model_field_title(model: ModelDocument, address: ModelAddress, components: ComponentRegistry) -> str | None:
    from feedbax.contracts.model_document import CompositeComponent, LeafComponent

    element = address.element
    match element:
        case FieldElement(parent=AttachmentElement(), path=("operands", str(operand), "value")):
            return {"low": "Low", "high": "High", "initial": "Initial"}.get(operand)
        case FieldElement(parent=AttachmentElement(), path=("delay",)):
            return "Delay"
    component, path = _model_parameter(model, address)
    if isinstance(component, CompositeComponent) and path == ("substeps",):
        return "Substeps"
    if isinstance(component, LeafComponent) and len(path) == 2 and path[0] == "fields":
        declaration = components.get(component.type)
        if declaration is not None and path[1] in declaration.params.model.model_fields:
            return declaration.params.field_title(path[1])
    return None


def _integer_train_site(
    path: str, payload: Payload, python_tasks: PythonTaskRegistry,
    training_methods: TrainingMethodRegistry | None,
) -> bool:
    from feedbax.tasks.registry import python_task_parameter_site

    documents = payload.get("trial_documents")
    site = python_task_parameter_site(path, documents, python_tasks)
    if site is not None:
        return site.expected.base is ValueBase.INTEGER and not site.expected.shape
    segments = path.split(".")
    annotation = DocumentTrainingRequestSpec
    if segments[0] == "settings" and training_methods is not None and isinstance(payload.get("method"), str):
        try:
            annotation = training_methods.resolve(payload["method"]).settings_model
        except KeyError:
            return False
        segments = segments[1:]
    return _integer_schema_site(segments, annotation)


def _integer_schema_site(segments: Sequence[str], annotation: type) -> bool:
    for segment in segments:
        fields = getattr(annotation, "model_fields", {})
        if segment not in fields:
            return False
        annotation = fields[segment].annotation
        args = get_args(annotation)
        if len(args) == 2 and type(None) in args:
            annotation = next(arg for arg in args if arg is not type(None))
    return annotation is int


def _integer_subject_task_site(
    path: str, tasks: Iterable[TaskDocument | PythonTaskSelection | None], python_tasks: PythonTaskRegistry,
) -> bool:
    """Whether ``path`` names an integer parameter of the Python Task an Evaluate runs from every subject."""
    from feedbax.tasks.registry import python_task_parameter_site

    sites = {
        site is not None and site.expected.base is ValueBase.INTEGER and not site.expected.shape
        for task in tasks
        for site in (python_task_parameter_site(path, {"python_task": task.model_dump(mode="json")}, python_tasks)
                     if isinstance(task, PythonTaskSelection) else None,)
    }
    return sites == {True}


@dataclass(frozen=True)
class _RowLoweringStamp:
    kind: StepKind
    payload_and_sources: str
    members: bytes


@dataclass(frozen=True)
class _LoweredRow:
    request: EncodedRequest | None
    diagnostics: tuple[ProjectDiagnostic, ...]
    lowering_stamp: _RowLoweringStamp | None = None
    problems: tuple[RunProblem, ...] = ()
    document_ids: Mapping[str, str] = field(default_factory=dict)
    document_uses: frozenset[str] = frozenset()
    model_source: _ModelSource | None = None
    train: TrainFacts | None = None
    encoded_spec: RequestSpec | None = None
    resolved: bool = True
    pinned_model: ModelDocument | None = None
    followed_documents: frozenset[str] = frozenset()
    dataset_port: DatasetPort | None = None
    metric_names: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _EvaluatePayload:
    """A managed Evaluate's lowered payload and declared Task dataset fields.

    ``payload`` is None when the Task does not resolve, so its trial rule does not lower."""

    payload: Payload | None
    problems: tuple[ProjectDiagnostic, ...] = ()
    dataset_port: DatasetPort | None = None


def _map_problem[T](
    result: T | ProjectDiagnostic | _EvaluatePayload, move: Callable[[ProjectDiagnostic], ProjectDiagnostic],
) -> T | ProjectDiagnostic | _EvaluatePayload:
    """Place the problem of a lowered Evaluate payload with ``move``."""
    if isinstance(result, ProjectDiagnostic):
        return move(result)
    if isinstance(result, _EvaluatePayload) and result.problems:
        return replace(result, problems=tuple(map(move, result.problems)))
    return result


@dataclass(frozen=True)
class _TrainPreparation:
    checkpoint_problem: RunProblem | None
    evaluation: GraphSpec | ProjectDiagnostic
    graph: GraphSpec
    lowered: LoweredModelDocument
    evaluation_states: tuple[RetainedStateField, ...]


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
    from feedbax.contracts.expression_registry import FunctionRegistry

    registry = FunctionRegistry()
    register_builtins(registry)
    registry.seal()
    return registry


def _trial_place(problem: Problem) -> tuple[str, tuple[str, ...]]:
    """The trial document a pairing Problem lies in, and its path in that document."""
    match problem.path:
        case ("model" | "task" | "python_task" | "objective" as role, *parts):
            pass
        case ("terms", *_):
            role, parts = "objective", problem.path
        case _:
            role, parts = "task", problem.path
    return str(role), tuple(map(str, parts))


def _trial_problems(
    problems: Iterable[Problem], model: ModelDocument | None, *, initial: bool = False, synthesis: bool = False,
) -> tuple[RunProblem, ...]:
    """Each Problem pairing a row's trial documents with ``model``, at its path in the request, or for an Evaluate of an
    initial Model in its definition's ``inputs``."""
    placed = []
    for problem in problems:
        role, parts = _trial_place(problem)
        if initial:
            slot = ("inputs", "subject" if role == "model" else "task" if role == "python_task" else role)
        elif synthesis:
            slot = ("task" if role == "python_task" else role,)
        else:
            slot = ("trial_documents", role)
        start = (
            task_start(model, problem.model_address)
            if model is not None and problem.model_address is not None and role == "task" and parts[:1] == ("links",)
            else None
        )
        placed.append(RunProblem(
            pointer((*slot, *parts)), problem.reason, problem.model_address, problem.variable, start=start,
        ))
    return tuple(placed)


def _set_condition_placement(problem: Problem, origins: Mapping[str, str] | None) -> str | None:
    role, path = _trial_place(problem)
    if role == "task" and origins and len(path) > 1 and path[0] == "variables" and (axis := origins.get(path[1])):
        return pointer(("set_fields", axis, "condition"))
    return None


def _evaluate_set_problem(key: RunKey, error: SetFieldProblem) -> ProjectDiagnostic:
    path = ("set_fields", error.axis)
    if error.condition:
        path = (*path, "condition")
    elif error.entry is not None:
        path = (*path, "entries", error.entry)
    return ProjectDiagnostic(
        ResolutionCode.INVALID_PAYLOAD, pointer(path), str(error), key.step, row=key.row, variant=key.variant,
    )


def _set_condition_problems(problems: Iterable[RunProblem], origins: Mapping[str, str]) -> tuple[RunProblem, ...]:
    """Report generated Task selection Problems on the authored Set condition."""
    if not origins:
        return tuple(problems)
    placed = []
    reported = set()
    for problem in problems:
        parts = json_pointer_parts(problem.path) if problem.path.startswith("/") else ()
        if len(parts) > 3 and parts[:3] == ("trial_documents", "task", "variables") and (axis := origins.get(parts[3])):
            if axis not in reported:
                placed.append(replace(problem, path=pointer(("set_fields", axis, "condition"))))
                reported.add(axis)
        else:
            placed.append(problem)
    return tuple(placed)


def _checkpoint_fit_problem(
    source: _TrainedSource, preparation: _TrainPreparation, spec: DocumentTrainingRequestSpec,
    components: ComponentRegistry,
) -> RunProblem | None:
    from feedbax.compiler.checkpoint_fit import checkpoint_misfit

    if source.replicates != spec.training_config.replicates:
        return RunProblem("/training_config/replicates", f"The checkpoint has {source.replicates} replicates.")
    continuation = spec.continuation
    if continuation is not None and continuation.state == "preserve" and source.method != spec.method:
        return RunProblem("/method", f"Carrying state needs the checkpoint's method, {source.method!r}.")
    if (misfit := checkpoint_misfit(source.graph, preparation.graph, components)) is None:
        return None
    if not misfit.path:
        return RunProblem("/trial_documents/model", misfit.reason)
    graph_path = (
        "graph", *(token for name in misfit.path[:-1] for token in ("subgraphs", name)), "nodes", misfit.path[-1],
    )
    address = preparation.lowered.source_address(pointer(graph_path))
    path = ("trial_documents", "model", *address.stored_path()) if address is not None else graph_path
    return RunProblem(pointer(path), misfit.reason, address)


class _PayloadProblems(Exception):
    """Problems at dotted payload paths found while lowering a row."""

    def __init__(self, problems: tuple[tuple[str, str], ...], *, missing_key: str | None = None):
        super().__init__(problems)
        self.problems = problems
        self.missing_key = missing_key
        """The declaration key no registered declaration provides, when that is the one problem."""

    @classmethod
    def unknown(cls, path: str, error: UnknownDeclaration) -> _PayloadProblems:
        return cls(((path, str(error.args[0])),), missing_key=error.key)


def _synthesis_request(
    payload: Payload, methods: SynthesisMethodRegistry, python_tasks: PythonTaskRegistry,
) -> SynthesisRequestSpec:
    from feedbax.tasks.registry import resolve_python_task_instance

    authored = SynthesisStepPayload.model_validate(payload)
    if isinstance(authored.model, DocumentReference):
        raise _PayloadProblems((("model", "Choose a Model document."),))
    if isinstance(authored.objective, DocumentReference):
        raise _PayloadProblems((("objective", "Choose an Objective document."),))
    authored_task = authored.task
    if isinstance(authored_task, DocumentReference):
        raise _PayloadProblems((("task", "Choose a Task document."),))
    model, objective = authored.model.lower(), authored.objective.lower()
    if isinstance(authored_task, PythonTaskSelection):
        resolved = resolve_python_task_instance(authored_task, python_tasks)
        if isinstance(resolved, tuple):
            raise _PayloadProblems(tuple(("task", problem.reason) for problem in resolved))
        documents = TrialDocuments(
            model=model, task=None, python_task=resolved.selection(), objective=objective,
            input_bindings=authored.input_bindings,
        )
    else:
        documents = TrialDocuments(
            model=model, task=authored_task.lower(), objective=objective, input_bindings=authored.input_bindings,
        )
    try:
        method = methods.resolve(authored.method)
    except UnknownDeclaration as error:
        raise _PayloadProblems.unknown("method", error) from error
    try:
        settings = method.declaration.settings_model.model_validate(authored.settings).model_dump(mode="json")
    except ValidationError as error:
        raise _PayloadProblems(tuple(
            (".".join(("settings", *map(str, failure["loc"]))), failure["msg"])
            for failure in error.errors(include_url=False, include_input=False)
        )) from error
    return SynthesisRequestSpec(trial_documents=documents, method=method.key, settings=settings, target=authored.target)


def _unregistered_component(payload: Payload, components: ComponentRegistry) -> tuple[str, str] | None:
    """The dotted payload path and type of the first leaf component of the row's Model that no package registers."""
    slot = trial_model_path(payload)
    model = payload.get(slot[0])
    if len(slot) == 2:
        model = model.get(slot[1]) if isinstance(model, dict) else None

    def walk(owner: object, at: tuple[str, ...]) -> tuple[str, str] | None:
        interior = owner.get("interior") if isinstance(owner, dict) else None
        members = interior.get("components") if isinstance(interior, dict) else None
        for name, component in (members.items() if isinstance(members, dict) else ()):
            here = (*at, "interior", "components", name)
            if not isinstance(component, dict):
                continue
            if component.get("kind") == "leaf":
                if isinstance(type_key := component.get("type"), str) and components.get(type_key) is None:
                    return ".".join(here), type_key
            elif (found := walk(component, here)) is not None:
                return found
        return None

    return walk(model, slot)


def _encode_row(
    key: RunKey,
    kind: StepKind,
    payload: Payload,
    inputs: Mapping[str, PlannedInput],
    lowering_stamp: _RowLoweringStamp | None,
    document_memo: RequestDocumentMemo,
    prepare_trial: Callable[[TrialDocuments], _TrainPreparation | tuple[Problem, ...]],
    document_ids: Mapping[str, str],
    python_tasks: PythonTaskRegistry,
    model_source: _ModelSource | None = None,
    prepared_train: _PreparedTrainRow | None = None,
    remember_train: Callable[[_PreparedTrainRow], None] | None = None,
    training_methods: TrainingMethodRegistry | None = None,
    previous: _LoweredRow | None = None,
    document_pointers: Mapping[str, str] = MappingProxyType({}),
    checkpoint_source: _TrainedSource | None = None,
    components: ComponentRegistry | None = None,
    synthesis_methods: SynthesisMethodRegistry | None = None,
) -> _LoweredRow:
    if components is not None and kind in ("train", "synthesize") and (
        unregistered := _unregistered_component(payload, components)
    ) is not None:
        path, type_key = unregistered
        place = f"payload.{path}"
        document, document_path = _payload_document(place, document_ids, document_pointers)
        return _LoweredRow(None, (ProjectDiagnostic(
            ResolutionCode.INVALID_PAYLOAD, place, missing_declaration("component", type_key), key.step,
            row=key.row, variant=key.variant, document=document, document_path=document_path, missing_key=type_key,
        ),), lowering_stamp)
    checkpoint_problem = None
    metric_names = frozenset()
    pairing_problems: tuple[RunProblem, ...] = ()
    preparation: _TrainPreparation | None = None
    if prepared_train is not None:
        payload = payload | {"trial_documents": prepared_train.documents}
        preparation = prepared_train.preparation
        checkpoint_problem = preparation.checkpoint_problem
        model_source = prepared_train.model
    elif kind == "train":
        from feedbax.tasks.registry import resolve_python_task_instance
        try:
            stored_trial = payload.get("trial_documents")
            authored = lower_stored_trial_documents(stored_trial)
            if model_source is not None and authored.task is not None:
                assert isinstance(stored_trial, dict)
                model_source = replace(model_source, stored_task=StoredTaskDocument.model_validate(stored_trial["task"]))
            if authored.python_task is not None:
                resolved = resolve_python_task_instance(authored.python_task, python_tasks)
                if not isinstance(resolved, tuple):
                    authored = authored.model_copy(update={"python_task": resolved.selection()})
            else:
                resolved = None
            payload = payload | {"trial_documents": authored}
            prepared = resolved if isinstance(resolved, tuple) else prepare_trial(authored)
        except ValueError as error:
            return _LoweredRow(None, (ProjectDiagnostic(
                ResolutionCode.INVALID_PAYLOAD, "payload.trial_documents", str(error), key.step,
                row=key.row, variant=key.variant,
            ),), lowering_stamp)
        assert model_source is not None
        if isinstance(prepared, tuple):
            # The Python task does not resolve, or the Model, Task and Objective do not pair: the row is planned and
            # cannot run.
            pairing_problems = _trial_problems(prepared, authored.model)
            model_source = replace(model_source, unpaired=True)
        else:
            preparation = prepared
            checkpoint_problem = preparation.checkpoint_problem
    if preparation is not None and prepared_train is None:
        assert model_source is not None
        if isinstance(preparation.evaluation, ProjectDiagnostic):
            model_source = replace(
                model_source, evaluation_problem=replace(preparation.evaluation, document=model_source.document),
            )
        else:
            model_source = replace(
                model_source, evaluation_graph=preparation.evaluation, evaluation_states=preparation.evaluation_states,
            )
    missing_key = None
    try:
        if kind == "report":
            document = ReportDocument.model_validate(payload)
            expected = set(document.figure_roles)
            invalid = [
                (ResolutionCode.INVALID_PAYLOAD, "payload.blocks", "The report has no blocks."),
            ] if not document.blocks else []
            invalid += [
                (ResolutionCode.BAD_REFERENCE, f"inputs.{role}", "required figure input is missing")
                for role in sorted(expected - inputs.keys())
            ] + [
                (ResolutionCode.BAD_REFERENCE, f"inputs.{role}", "unknown Report figure role")
                for role in sorted(inputs.keys() - expected)
            ]
            if invalid:
                return _LoweredRow(
                    None,
                    tuple(ProjectDiagnostic(
                        code, path, detail, key.step, row=key.row, variant=key.variant,
                    ) for code, path, detail in invalid),
                    lowering_stamp,
                )
            spec = ReportRequestSpec(document=document)
        elif kind == "synthesize":
            assert synthesis_methods is not None
            spec = _synthesis_request(payload, synthesis_methods, python_tasks)
            documents = spec.trial_documents
            prepared = prepare_trial(documents)
            compiled_model = prepared.graph if isinstance(prepared, _TrainPreparation) else None
            evaluation = prepared.evaluation if isinstance(prepared, _TrainPreparation) else None
            authored = model_source or _ModelSource(
                document_ids.get("payload.model"), documents.model, sha256_bytes(to_json(documents.model.model_dump(mode="json"))),
            )
            model_source = replace(
                authored,
                evaluation_problem=evaluation if isinstance(evaluation, ProjectDiagnostic) else None,
                evaluation_graph=compiled_model if isinstance(evaluation, GraphSpec) else None,
                unpaired=isinstance(prepared, tuple),
                stored_task=(StoredTaskDocument.model_validate(payload["task"]) if documents.task is not None else None),
            )
            if isinstance(prepared, tuple):
                pairing_problems = _trial_problems(prepared, documents.model, synthesis=True)
            else:
                if (problem := prepared.checkpoint_problem) is not None:
                    checkpoint_problem = replace(problem, path=problem.path.removeprefix("/trial_documents"))
        else:
            spec = request_identity.REQUEST_KINDS[kind].spec_type.model_validate(payload)
        if isinstance(spec, DocumentTrainingRequestSpec):
            if (spec.checkpoint_initialization is None) != ("checkpoint" not in inputs):
                raise ValueError("/inputs/checkpoint: a predecessor and initialization policy must be declared together")
            if (
                (start := inputs.get("checkpoint")) is not None
                and isinstance(reference := start.member.reference, ContentRef)
                and reference.media_type == IMPORTED_MODEL_MEDIA_TYPE
                and spec.checkpoint_initialization != "initialize_from"
            ):
                raise ValueError("/inputs/checkpoint: a Loaded Model supports only initialize_from.")
            if (start := inputs.get("checkpoint")) is not None and is_synthesized_model_product(start.member.reference):
                if spec.checkpoint_initialization != "initialize_from":
                    raise ValueError("/inputs/checkpoint: a Synthesize model supports only initialize_from.")
                if spec.training_config.replicates != 1:
                    raise ValueError("/training_config/replicates: A Synthesize model contains one replica.")
            if training_methods is None:
                raise ValueError("Train requires the installed method catalog.")
            try:
                training_methods.resolve(spec.method)
            except UnknownDeclaration as error:
                raise _PayloadProblems.unknown("method", error) from error
            try:
                spec, method, _ = normalize_train_method(spec, training_methods)
                metric_names = method.metric_names(spec)
            except ValidationError as error:
                raise _PayloadProblems(tuple(
                    (".".join(("settings", *map(str, failure["loc"]))), failure["msg"])
                    for failure in error.errors(include_url=False, include_input=False)
                )) from error
        if (
            previous is not None and previous.request is not None
            and previous.lowering_stamp is not None and lowering_stamp is not None
            and previous.lowering_stamp.members == lowering_stamp.members
            and same_typed_value(spec, previous.encoded_spec)
        ):
            encoded = previous.request
            encoded_spec = previous.encoded_spec
        else:
            encoded = request_identity.encode_request(
                spec,
                {role: (binding.request_member,) for role, binding in inputs.items()},
                document_memo=document_memo,
                capture_train_task=kind == "train",
            )
            encoded_spec = spec if lowering_stamp is not None else None
    except ValidationError as error:
        problems = tuple(
            (".".join(map(str, failure["loc"])), failure["msg"])
            for failure in error.errors(include_url=False, include_input=False)
        )
    except _PayloadProblems as error:
        problems = error.problems
        missing_key = error.missing_key
    except ValueError as error:
        problems = (("", str(error)),)
    else:
        if remember_train is not None and preparation is not None:
            assert isinstance(spec, DocumentTrainingRequestSpec) and model_source is not None
            remember_train(_PreparedTrainRow(spec.trial_documents, preparation, model_source))
        if isinstance(spec, DocumentTrainingRequestSpec) and preparation is not None:
            assert model_source is not None
            model_source = replace(model_source, trained=_TrainedSource(
                preparation.graph, spec.method, spec.training_config.replicates,
            ))
            if checkpoint_problem is None and checkpoint_source is not None:
                assert components is not None
                checkpoint_problem = _checkpoint_fit_problem(checkpoint_source, preparation, spec, components)
        documents = spec.trial_documents if isinstance(spec, (DocumentTrainingRequestSpec, SynthesisRequestSpec)) else None
        if pairing_problems:
            refusals = pairing_problems
        elif documents is not None and not documents.model.interior.components:
            refusals = (RunProblem("/trial_documents/model/interior/components", "Model has no components."),)
        elif documents is not None and not any(term.enabled for term in documents.objective.terms):
            refusals = (RunProblem("/trial_documents/objective/terms", "Objective has no enabled terms."),)
        else:
            refusals = (checkpoint_problem,) if checkpoint_problem is not None else ()
        return _LoweredRow(
            encoded, (), lowering_stamp, problems=refusals,
            model_source=model_source, encoded_spec=encoded_spec, metric_names=metric_names,
        )
    return _LoweredRow(
        None,
        tuple(
            ProjectDiagnostic(
                ResolutionCode.INVALID_PAYLOAD, place, message, key.step, row=key.row, variant=key.variant,
                document=document, document_path=document_path, missing_key=missing_key,
            )
            for path, message in problems
            for place in (f"payload.{path}" if path else "payload",)
            for document, document_path in (_payload_document(place, document_ids, document_pointers),)
        ),
        lowering_stamp,
    )


def _payload_document(
    path: str, document_ids: Mapping[str, str], document_pointers: Mapping[str, str],
) -> tuple[str | None, str | None]:
    """Name the document that supplies the value at ``path`` and the pointer to it in that document's payload."""
    parts = tuple(path.split("."))
    for length in range(len(parts), 1, -1):
        slot = ".".join(parts[:length])
        if (document := document_ids.get(slot)) is not None:
            return document, pointer((*json_pointer_parts(document_pointers.get(slot)), *parts[length:]))
    return None, None


@dataclass
class _PlanWork:
    compositions: dict[
        tuple[str, str, bool], _WorkEntry[StepComposition | ProjectDiagnostic]
    ] = field(default_factory=dict)
    sweeps: dict[
        tuple[str, str], _WorkEntry[
            tuple[tuple[ExpandedRow, ...] | ProjectDiagnostic, dict[str, tuple[str, ...]], dict[str, str]]
        ]
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


def compiler_authority(
    components: ComponentRegistry | None, operations: OperationRegistry | None,
    functions: FunctionRegistry | None, python_tasks: PythonTaskRegistry | None,
    training_methods: TrainingMethodRegistry | None,
    figure_templates: FigureTemplateRegistry | None,
    synthesis_methods: SynthesisMethodRegistry | None = None,
) -> tuple[object, ...]:
    """Identify the compiler code and registries whose output a memoized plan depends on."""
    return (
        compose_step, StepTemplate.resolve, expand_sweep, source_shape, tuple(_ROLES.items()),
        request_identity.request_codec_authority(), components, operations, functions, python_tasks, training_methods,
        figure_templates, synthesis_methods,
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
        *,
        python_tasks: PythonTaskRegistry,
        training_methods: TrainingMethodRegistry | None = None,
        figure_templates: FigureTemplateRegistry | None = None,
        synthesis_methods: SynthesisMethodRegistry | None = None,
        capture_sources: bool = False,
    ):
        self.project = project
        self.readers = readers
        self.previous = previous
        self.operations = operations
        self.figure_templates = figure_templates
        self.components = components
        self.functions = functions
        self.python_tasks = python_tasks
        self.training_methods = training_methods
        self.synthesis_methods = synthesis_methods
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
        self.waiting: dict[RunKey, RunKey] = {}
        """Each row with a ``blocked_input``, and the producer it names."""
        self.runs: dict[tuple[str, str], tuple[_CompiledRun, ...]] = {}
        self.swept_step_axes: dict[tuple[str, str], tuple[str, ...]] = {}
        self.requests: dict[RequestId, bytes] = {}
        self.documents: dict[str, bytes] = {}
        self.diagnostics: list[ProjectDiagnostic] = []
        self.subsets: dict[tuple[str, str, str], PlannedSubset] = {}
        self.exception_matches: dict[tuple[str, str], Mapping[str, tuple[str, ...]]] = {}
        self.selected_definitions: dict[RunKey, SelectedRunDefinition] = {}
        self.model_sources: dict[RunKey, _ModelSource] = {}
        self.terminal_sources: dict[RequestId, TerminalTrainSource] = {}
        self.element_holders: dict[str, ElementHolders] = {}

    def expand_variants(self, known_sites: Mapping[str, bool] | None = None) -> None:
        def integer_site(path: str) -> bool:
            if (target := model_field_target(path)) is not None:

                address = target.address
                if target.step is not None:
                    step = self.project.steps.get(target.step)
                    if step is None or step.kind not in {"train", "evaluate", "synthesize"}:
                        return False
                    template = compose_step(self.project, target.step, readers=self.readers)
                    if isinstance(template, ProjectDiagnostic):
                        return False
                    resolved = template.resolve()
                    if isinstance(resolved, ProjectDiagnostic):
                        return False
                    if step.kind in {"train", "synthesize"}:
                        _, source = self.lower_producer_model(
                            RunKey(DEFAULT_SWEEP_ROW, target.step, DEFAULT_SWEEP_ROW),
                            resolved.payload, resolved.document_ids, resolved.document_pointers,
                        )
                        model = source.model if source is not None else None
                    elif isinstance(reference := resolved.inputs.get("subject"), InitialModelReference):
                        _, differences = _model_differences(resolved.payload)
                        model = self.initial_model_source(reference, resolved.input_documents["subject"], differences).document
                    else:
                        return False
                    return model is not None and _integer_model_site(
                        model, address, self.components or _builtin_model_registry(),
                    )
                if target.document is None:
                    raise ValueError("A project Model field target must name a step or document")
                entry = compose_document(self.project.documents.get, target.document)
                return isinstance(entry, RootDocument) and entry.schema_id == STORED_MODEL_DOCUMENT_SCHEMA_ID and _integer_model_site(
                    StoredModelDocument.model_validate(entry.payload).lower(), address,
                    self.components or _builtin_model_registry(),
                )
            parts = path.split(".", 4)
            if len(parts) == 5 and parts[0] == "steps" and parts[2:4] == ["base", "inline"]:
                step = self.project.steps.get(parts[1])
                if step is not None and step.kind == "train" and isinstance(step.base, InlineBase):
                    return _integer_train_site(parts[4], step.base.inline, self.python_tasks, self.training_methods)
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
            document = None
            step_name = None
            try:
                from feedbax.compiler.model_attachment import resolve_effective_model
                from feedbax.compiler.model_document import ModelCompilationProblem

                based_writes: dict[str, list[tuple[str, OverridePatch | SetField]]] = {}
                for path, write in row.patches.items():
                    address = None
                    document = None
                    step_name = None
                    patch = sweep_patch(path, write)
                    if isinstance(patch, SetField):
                        address = patch.address
                        target = model_field_target(path)
                        assert target is not None
                        step_name, document = target.step, target.document
                        if document is None:
                            raise ValueError("A project Model field target names a Model document")
                        entry = data["documents"].get(document)
                        if entry is None or entry["schema"] != STORED_MODEL_DOCUMENT_SCHEMA_ID:
                            raise ValueError("The addressed Model document does not exist")
                        if "payload" not in entry:
                            based_writes.setdefault(document, []).append((path, patch))
                            if origins is not None:
                                source = ValueSource("variant", row.id, pointer(("sweep", path)), variant=row.id)
                                origins = place_source(
                                    origins, ("documents", document, "payload", *patch.address.stored_path()),
                                    source_shape(patch.value, source),
                                )
                            continue
                        before = entry["payload"]
                        model = resolve_effective_model(
                            StoredModelDocument.model_validate(before).lower(),
                            self.components or _builtin_model_registry(), (patch,), functions=self.functions,
                        )
                        entry["payload"] = StoredModelDocument.from_lowered(model).model_dump(mode="json", exclude_unset=True)
                        if origins is not None:
                            source = ValueSource("variant", row.id, pointer(("sweep", path)), variant=row.id)
                            authored = origins["documents"][document]
                            authored["payload"] = changed_sources(before, entry["payload"], authored["payload"], source)
                        continue
                    parts = tuple(path.split("."))
                    if len(parts) < 2 or parts[0] not in {"documents", "steps"}:
                        raise ValueError("a variant must patch a named document or step")
                    if parts[0] == "steps" and (
                        reached := reference_write_target(data["steps"], parts[1:])
                    ) is not None:
                        step_name = parts[1]
                        raise ValueError(
                            "A variant cannot write through a step's document reference; "
                            f"write documents.{reached[0]}.payload.{reached[1]}"
                        )
                    if len(parts) >= 4 and parts[0] == "documents" and parts[2] == "payload" and (
                        "base" in data["documents"].get(parts[1], {})
                    ):
                        based_writes.setdefault(parts[1], []).append((path, patch))
                        if origins is not None and patch.op != "remove":
                            source = ValueSource("variant", row.id, pointer(parts), variant=row.id)
                            origins = place_source(origins, parts, source_shape(patch.value, source))
                        continue
                    if parts[0] == "steps" and parts[2:3] == ("deltas",) and len(parts) > 3 and parts[1] in data["steps"]:
                        data["steps"][parts[1]].setdefault("deltas", [])
                    apply_stored_patch_in_place(data, patch)
                    if origins is not None:
                        source = ValueSource("variant", row.id, pointer(parts), variant=row.id)
                        origins = patch_source(origins, patch, source_shape(patch.value, source))
                for document in sorted(based_writes, key=lambda name: len(base_chain(
                    DOCUMENTS.validate_python(data["documents"]), name,
                ))):
                    path = based_writes[document][0][0]
                    documents = DOCUMENTS.validate_python(data["documents"])
                    base = compose_document(documents.get, documents[document].base.document)
                    if isinstance(base, ProjectDiagnostic):
                        raise ValueError(base.detail)
                    keyed = partial(stored_keyed_collection, base.schema_id)
                    content = {"documents": {document: {"payload": apply_differences(
                        base.payload, documents[document].differences, keyed_collection=keyed,
                    )}}}
                    for path, patch in based_writes[document]:
                        if isinstance(patch, SetField):
                            model = resolve_effective_model(
                                StoredModelDocument.model_validate(content["documents"][document]["payload"]).lower(),
                                self.components or _builtin_model_registry(), (patch,), functions=self.functions,
                            )
                            content["documents"][document]["payload"] = StoredModelDocument.from_lowered(model).model_dump(
                                mode="json", exclude_unset=True,
                            )
                        else:
                            apply_stored_patch_in_place(content, patch)
                    data["documents"][document]["differences"] = {
                        at: difference.model_dump(mode="json") for at, difference in diff(
                            base.payload, content["documents"][document]["payload"], keyed_collection=keyed,
                        ).items()
                    }
                self.variants[row.id] = Project.model_validate(data)
                self.variant_sources[row.id] = origins
            except ValueError as error:
                if isinstance(error, ModelCompilationProblem):
                    address = error.address
                self.diagnostics.append(
                    ProjectDiagnostic(
                        ResolutionCode.INVALID_SWEEP, path, str(error), step=step_name, variant=row.id,
                        document=document, model_address=address,
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
        self.diagnostics.extend(self.shared_key_diagnostics())
        counts = Counter((key.variant, key.step) for key in self.rows)
        numbered: Counter[tuple[str, str]] = Counter()
        for key, row in self.rows.items():
            at = key.variant, key.step
            numbered[at] += 1
            self.rows[key] = replace(
                row, number=numbered[at], step_name=self.steps[at].label if counts[at] == 1 else None,
            )
        return ProjectPlan(
            self.variant_order,
            MappingProxyType(self.steps),
            tuple(self.rows.values()),
            tuple(
                cast(PlannedRun, self.rows[run.key])
                for runs in self.runs.values() for run in runs if isinstance(run, PlannedRun)
            ),
            MappingProxyType(self.requests),
            MappingProxyType(self.documents),
            tuple(self.diagnostics),
            MappingProxyType(self.subsets),
            MappingProxyType(self.selected_definitions),
            SourceSnapshot(self.project.model_copy(deep=True), self.variant_sites, self.source_sweeps)
            if self.source_sweeps is not None else None,
            MappingProxyType(self.exception_matches),
            MappingProxyType(dict(self.variants)),
        )

    def holders(self, variant: str) -> ElementHolders:
        if variant not in self.element_holders:
            project = self.variants[variant]

            def elements_of(card: str) -> frozenset[str]:
                composition = self.compose(variant, card, False)
                if isinstance(composition, ProjectDiagnostic):
                    return frozenset()
                try:
                    return payload_elements(composition.bind(project).authored_payload())
                except ValueError:
                    return frozenset()

            self.element_holders[variant] = ElementHolders(analyze_cards(project), elements_of)
        return self.element_holders[variant]

    def shared_key_diagnostics(self) -> list[ProjectDiagnostic]:
        """A diagnostic on each Analysis document holding a key an unrelated document holds, once for every variant
        that shares it."""
        found = {variant: shared_element_keys(project) for variant, project in self.variants.items()}
        everywhere = set.intersection(*(set(keys) for keys in found.values())) if found else set()
        problems = []
        for variant, keys in found.items():
            for key in keys:
                if key in everywhere and variant != self.variant_order[0]:
                    continue
                problems.append(ProjectDiagnostic(
                    ResolutionCode.INVALID_DOCUMENT, f"payload.{key.family}.{key.element}",
                    f"{member_name(self.variants[variant].documents, key.holder)} holds the same key in an unrelated document.",
                    document=key.document, element=key.element,
                    variant=None if key in everywhere else variant,
                ))
        return problems

    def step(
        self, variant: str, name: str, active: tuple[tuple[str, str], ...]
    ) -> tuple[_CompiledRun, ...]:
        key = variant, name
        if key not in self.runs:
            self.runs[key] = self.compile_step(variant, name, (*active, key))
        return self.runs[key]

    def step_rows(self, variant: str, name: str) -> tuple[PlannedRow, ...]:
        """Every expanded row of a compiled step, whether or not it can run."""
        return tuple(row for key, row in self.rows.items() if key.variant == variant and key.step == name)

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

    def run_name(self, row: PlannedRow) -> str:
        rows = self.step_rows(row.key.variant, row.key.step)
        step = self.variants[row.key.variant].steps[row.key.step]
        if len(rows) == 1:
            return replace(row, step_name=step.label).name
        number = next(i for i, candidate in enumerate(rows, 1) if candidate.key == row.key)
        return replace(row, number=number).name

    def expand_rows(
        self,
        variant: str,
        name: str,
        composition: StepComposition,
        members: Mapping[str, _Members], *, sweep: Sweep | None = None,
    ) -> tuple[ExpandedRow, ...] | ProjectDiagnostic:
        sweep = sweep if sweep is not None else composition.sweep
        kind = self.variants[variant].steps[name].kind
        checkpoint_axis = kind == "evaluate" and composition.sweep is not None and any(
            isinstance(axis, CheckpointRuleAxis) for axis in composition.sweep.axes.values()
        )
        def member(role: str, item: PlannedInput, producer: PlannedRow | None) -> SweepInputMember:
            horizon = producer.train.target_total \
                if checkpoint_axis and role == "subject" and producer is not None and producer.train is not None else None
            if producer is not None and item.producer is not None:
                return SweepInputMember(item.member.key, item.coordinates, self.run_name(producer), horizon)
            group = {
                axis: producer.coordinates[axis] for axis in item.member.reference.group
            } if producer is not None and isinstance(item.member.reference, BestRank) else {}
            return SweepInputMember(item.member.key, group, _member_label(item, group), horizon)

        fanout = {
            role: tuple(member(role, item, producer) for item, producer in bindings)
            for role, bindings in members.items()
        }
        subjects = tuple(
            self.model_sources.get(key)
            for item, _ in members.get("subject", ())
            for key in item.sources
        ) if kind == "evaluate" else ()
        subject_tasks = tuple(
            producer.train.task for _, producer in members.get("subject", ())
            if producer is not None and producer.train is not None
        )

        def expand(read_document=None, _previous=None):
            @cache
            def resolved_definition():
                return self.resolve_row(
                    composition, RunKey(variant, name, DEFAULT_SWEEP_ROW),
                    ExpandedRow(DEFAULT_SWEEP_ROW, None, {}, {}, {}), read_document,
                )

            @cache
            def producer_model():
                resolved = resolved_definition()
                if isinstance(resolved, ProjectDiagnostic):
                    return None
                _, model = self.lower_producer_model(
                    RunKey(variant, name, DEFAULT_SWEEP_ROW), resolved.payload, resolved.document_ids, resolved.document_pointers,
                )
                return model

            def model_sites() -> tuple[tuple[str | None, ModelDocument], ...]:
                if kind in {"train", "synthesize"}:
                    source = producer_model()
                    return ((source.document, source.model),) if source is not None else ()
                if isinstance(reference := composition.inputs.get("subject"), InitialModelReference):
                    resolved = resolved_definition()
                    if isinstance(resolved, ResolvedStep):
                        _, differences = _model_differences(resolved.payload)
                        source = self.initial_model_source(reference, resolved.input_documents["subject"], differences)
                        return ((resolved.document_ids.get("inputs.subject"), source.document),)
                return tuple((source.document, source.model) for source in subjects if source is not None)

            site_documents: dict[str, str | None] = {}

            def integer_site(path: str) -> bool:
                target = model_field_target(path)
                if target is None:
                    if kind == "evaluate":
                        return _integer_subject_task_site(path, subject_tasks, self.python_tasks)
                    if kind not in {"train", "synthesize"}:
                        return False
                    resolved = resolved_definition()
                    if kind == "synthesize" and isinstance(resolved, ResolvedStep):
                        segments = path.split(".")
                        method = resolved.payload.get("method")
                        if segments[0] == "settings" and isinstance(method, str) and self.synthesis_methods is not None:
                            try:
                                settings = self.synthesis_methods.resolve(method).declaration.settings_model
                            except KeyError:
                                return False
                            return _integer_schema_site(segments[1:], settings)
                        return False
                    return isinstance(resolved, ResolvedStep) and _integer_train_site(
                        path, resolved.payload, self.python_tasks, self.training_methods,
                    )
                if target.scope is not None:
                    raise ValueError("Use an unscoped Model field target in a step sweep")
                sources = model_sites()
                documents = {document for document, _ in sources}
                if len(documents) == 1:
                    site_documents[path] = documents.pop()
                types = {
                    _integer_model_site(
                        model, target.address, self.components or _builtin_model_registry(),
                    )
                    for _, model in sources
                }
                if len(types) > 1:
                    raise ValueError("The addressed range has different site types in the selected Train subjects")
                return True in types

            sites: dict[str, bool] = {}

            def capture_site(path: str) -> bool:
                sites[path] = answer = integer_site(path)
                return answer

            @cache
            def model_drafts():
                from feedbax.compiler.model_edit import ModelEditDraft
                from feedbax.contracts.model_edit import ModelEditState

                return tuple(ModelEditDraft(ModelEditState(document=model), lambda _: {}) for _, model in model_sites())

            @cache
            def field_title(path: str) -> str | None:
                target = model_field_target(path)
                if target is None:
                    parts = tuple(path.split("."))
                    if kind == "train" and parts == ("seed",):
                        return "Seed"
                    if kind == "train" and parts[0] == "training_config":
                        return setting_field_title(DocumentTrainingConfig.model_json_schema(), parts[1:], None)
                    if kind == "train" and parts[0] == "settings" and self.training_methods is not None:
                        resolved = resolved_definition()
                        method = resolved.payload.get("method") if isinstance(resolved, ResolvedStep) else None
                        if isinstance(method, str):
                            try:
                                schema = self.training_methods.resolve(method).settings_model.model_json_schema()
                            except KeyError:
                                return None
                            return setting_field_title(schema, parts[1:], resolved.payload.get("settings"))
                    if kind == "evaluate" and parts[0] == "params":
                        resolved = resolved_definition()
                        if isinstance(resolved, ResolvedStep) and resolved.payload.get("evaluation_type") == ROLLOUT_TYPE:
                            return setting_field_title(
                                RolloutSettings.model_json_schema(), parts[1:], resolved.payload.get("params"),
                            )
                    return None
                labels = {
                    _model_field_title(model, target.address, self.components or _builtin_model_registry())
                    for _, model in model_sites()
                }
                return labels.pop() if len(labels) == 1 else None

            @cache
            def field_unit(path: str) -> str | None:
                target = model_field_target(path)
                if target is None:
                    return None
                units = {
                    draft.field_unit(target.address, functions=self.functions) for draft in model_drafts()
                }
                return units.pop() if len(units) == 1 else None

            def unexpanded_titles(rows: tuple[ExpandedRow, ...]) -> dict[str, str]:
                if sweep is None:
                    return {}
                pending = {key for key, axis in sweep.axes.items()
                           if isinstance(axis, GeneratedAxis | NamedAxis) and axis.label is None}
                for row in rows:
                    pending.difference_update(row.coordinates)
                    if not pending:
                        return {}
                titles = {}
                for key in pending:
                    axis = sweep.axes[key]
                    if isinstance(axis, GeneratedAxis):
                        paths = {axis.path}
                    else:
                        assert isinstance(axis, NamedAxis)
                        paths = {path for value in axis.values.values() if value is not None
                                 for path, patch in value.patches.items() if patch is not None}
                    if len(paths) == 1 and (title := field_title(paths.pop())) is not None:
                        titles[key] = title
                return titles

            def with_field_presentation(row: ExpandedRow) -> ExpandedRow:
                styles: dict[str, set[tuple[str | None, str | None]]] = {}
                for path, origin in row.patch_origins.items():
                    if origin[0] not in {"sweep_value", "set_entry"}:
                        continue
                    axis = origin[1]
                    coordinate = row.coordinates.get(axis)
                    if coordinate is not None:
                        number = type(coordinate.value) in (int, float)
                        title = field_title(path) if number or origin[0] == "sweep_value" else None
                        style = title, field_unit(path) if number else None
                        if style != (None, None):
                            styles.setdefault(axis, set()).add(style)
                if not styles:
                    return row
                coordinates = dict(row.coordinates)
                for axis, choices in styles.items():
                    if len(choices) == 1:
                        (label, unit), = choices
                        coordinate = coordinates[axis]
                        authored = sweep.axes[axis] if sweep is not None else None
                        name = authored.label if authored is not None else None
                        coordinates[axis] = replace(coordinate, axis_label=name or label or coordinate.axis_label, unit=unit)
                choices = axis_coordinates(coordinates, row.input_axes, row.kept_axes)
                assert sweep is not None
                label = sweep_row_name(
                    choices[axis] for axis in sweep.axes
                    if axis in choices and (axis in row.axis_value_ids or axis in row.kept_axes)
                )
                return replace(row, coordinates=coordinates, label=label)

            try:
                sets = {}
                if composition.sweep is not None and any(
                    isinstance(axis, SetAxis) for axis in composition.sweep.axes.values()
                ):
                    resolved = resolved_definition()
                    if isinstance(resolved, ProjectDiagnostic):
                        return resolved, {}
                    read = read_document or self.variants[variant].documents.get
                    model = model_slot(resolved.payload, resolved.document_ids, resolved.document_pointers)
                    if model is None and kind == "evaluate":
                        model = document_subject_model(resolved, read)
                    if model is None and kind == "evaluate":
                        source_models = {
                            (source.document, to_json(source.model.model_dump(mode="json"))) for source in subjects
                            if source is not None and source.document is not None
                        }
                        if len(source_models) > 1:
                            raise SweepExpansionError(
                                SweepExpansionCode.INVALID_SET, "inputs.subject", "the subject runs read different Models",
                            )
                        if source_models:
                            (document, _), = source_models
                            # The row patches the composed document. Its stored key
                            # presence selects each strict Add or Replace.
                            composed = compose_document(read, document)
                            if not isinstance(composed, ProjectDiagnostic) and isinstance(composed.payload, dict):
                                model = document, composed.payload
                    sets = resolve_set_axes(composition.sweep, read, model)
                for axis_name, axis in (composition.sweep.axes if composition.sweep is not None else {}).items():
                    if axis_name in RESERVED_LABELS and not isinstance(axis, InputAxis):
                        raise SweepExpansionError(
                            SweepExpansionCode.COORDINATE_CONFLICT, f"sweep.axes.{axis_name}",
                            f"{axis_name!r} is the name of a member label",
                        )
                exception_matches: dict[str, tuple[str, ...]] = {}
                rows = expand_sweep(
                    sweep, inputs=fanout,
                    integer_site=capture_site if self.source_sweeps is not None else integer_site,
                    checkpoint_subject=kind == "evaluate",
                    exception_matches=exception_matches, sets=sets,
                )
                rows = tuple(with_field_presentation(row) for row in rows)
                if self.source_sweeps is not None:
                    axes = sweep.axes if sweep is not None else {}
                    roles = {axis.input for axis in axes.values() if isinstance(axis, InputAxis)}
                    if checkpoint_axis:
                        roles.add("subject")
                    source = SourceSweep({
                        role: tuple(
                            replace(member, coordinates=deepcopy(dict(member.coordinates)))
                            for member in members
                        )
                        for role, members in fanout.items() if role in roles
                    }, sites, sets)
                    if source.inputs or source.integer_sites or source.sets:
                        self.source_sweeps[variant, name] = source
                return rows, exception_matches, unexpanded_titles(rows)
            except SweepExpansionError as error:
                document = site_documents.get(error.target) if error.target is not None else None
                return _sweep_problem(error, step=name, variant=variant, document=document), {}, unexpanded_titles(())

        if self.previous is None:
            rows, matches, titles = expand()
        else:
            rows, matches, titles = _work(
                self.previous.sweeps,
                self.work.sweeps,
                (variant, name),
                (
                    composition,
                    sweep,
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
        if titles:
            at = variant, name
            self.steps[at] = replace(self.steps[at], unexpanded_axis_titles=MappingProxyType(titles))
        return rows

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
        resolved = composition.bind(
            project,
            readers=self.readers,
            variant_sources=self.variant_sources.get(key.variant),
            read_document=reader,
        ).resolve(
            row, include_sources=sources, partial_figures=project.steps[key.step].kind == "analyze",
            trial_resolution=trial_resolution,
        )
        if not row.task_additions or isinstance(resolved, ProjectDiagnostic):
            return resolved
        flat = resolved.kind == "synthesize"
        raw = resolved.payload if flat else resolved.payload.get("trial_documents")
        explicit = resolved.input_documents.get("task")
        if not isinstance(raw, dict) or "task" not in raw:
            if resolved.kind == "evaluate" and explicit is None:
                return resolved  # The selected checkpoint supplies the Task later.
            if explicit is None:
                return ProjectDiagnostic(
                    ResolutionCode.INVALID_PAYLOAD, "payload.task" if flat else "payload.trial_documents.task",
                    "A Set entry's Task variables and links need a Task document.", key.step, row=key.row, variant=key.variant,
                )
        path = "inputs.task" if explicit is not None else "payload.task" if flat else "payload.trial_documents.task"
        try:
            content = explicit.payload if explicit is not None else raw["task"]
            task = StoredTaskDocument.model_validate(content)
            selected = StoredTaskDocument.from_lowered(
                add_to_task(task.lower(), row.task_additions), trial_keys=task.validation_trials or None,
            ).model_dump(mode="json", exclude_unset=True)
        except ValueError as error:
            return ProjectDiagnostic(
                ResolutionCode.INVALID_PAYLOAD, path, str(error), key.step, row=key.row, variant=key.variant,
            )
        if explicit is not None:
            return replace(resolved, input_documents=resolved.input_documents | {
                "task": explicit.model_copy(update={"payload": selected}),
            })
        writes = {"task": selected} if flat else {"trial_documents": raw | {"task": selected}}
        return replace(resolved, payload=resolved.payload | writes)

    def project_definition(
        self,
        composition: StepComposition,
        key: RunKey,
        row: ExpandedRow,
        inputs: Mapping[str, tuple[PlannedInput, ...]],
    ) -> SelectedRunDefinition | None:
        def project(reader=None, _previous=None):
            resolved = self.resolve_row(composition, key, row, reader, sources=True)
            if isinstance(resolved, ProjectDiagnostic) or resolved.sources is None:
                return None
            sources = {f"/payload{path}": source for path, source in resolved.sources.items()}
            for at, subset in self.input_subsets(key, composition, inputs):
                if subset.source is not None:
                    sources[pointer((*at, "where"))] = subset.source.origin
            for role, bindings in inputs.items():
                if len(bindings) == 1 and (source := _upstream_source(bindings[0])) is not None:
                    sources[pointer(("inputs", role))] = source
            return SelectedRunDefinition(
                resolved.payload, MappingProxyType(sources), MappingProxyType(resolved.document_ids), resolved.document_uses,
                MappingProxyType(resolved.document_pointers),
                resolved.document_problems,
                MappingProxyType(resolved.input_documents), task_additions=row.task_additions,
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
                    tuple(entry.model_dump(mode="json") for entry in row.task_additions),
                )),
                tuple(sorted(inputs.items())),
                tuple(
                    subset.source.origin for _, subset in self.input_subsets(key, composition, inputs)
                    if subset.source is not None
                ),
            ),
            self.authored_inputs[key.variant],
            project,
        )

    def input_subsets(
        self, key: RunKey, composition: StepComposition, inputs: Mapping[str, object],
    ) -> Iterator[tuple[tuple[str, ...], PlannedSubset]]:
        """The subset of each selection of the roles in ``inputs``, by the selection's authored path."""
        for role in inputs:
            binding = composition.inputs.get(role)
            items = (
                ((("inputs", role, str(index)), f"{role}.{index}") for index in range(len(binding)))
                if isinstance(binding, list) else ((("inputs", role), role),)
            )
            for at, input_id in items:
                if (subset := self.subsets.get((key.variant, key.step, input_id))) is not None:
                    yield at, subset

    def sweep_problem_source(
        self, problem: ProjectDiagnostic | RunProblem, composition: StepComposition, key: RunKey, row: ExpandedRow,
    ) -> ValueSource | None:
        if not row.patch_origins:
            return None
        paths = []
        if problem.model_address is not None:
            paths = [
                path for path in row.patch_origins
                if (target := model_field_target(path)) is not None and target.address == problem.model_address
            ]
        elif key.step and (setting := problem.path.removeprefix("payload.")).startswith("settings."):
            parts = setting.split(".")
            # A union validation path includes its selected class between the field and its parameter.
            paths = [
                path for path in row.patch_origins if path.startswith("settings.")
                and path.split(".")[-1] == parts[-1]
                and all(part in set(parts) for part in path.split("."))
            ]
        if len(paths) != 1:
            return None
        template = composition.bind(self.variants[key.variant], variant_sources=self.variant_sources.get(key.variant))
        return template.sweep_source(row, paths[0])

    def locate_sweep_problem(
        self, problem: ProjectDiagnostic, composition: StepComposition, key: RunKey, row: ExpandedRow,
    ) -> ProjectDiagnostic:
        if problem.authored_path is not None or (origin := self.sweep_problem_source(problem, composition, key, row)) is None:
            return problem
        return replace(self.locate_authored_problem(problem, origin), code=ResolutionCode.INVALID_SWEEP)

    def locate_sweep_run_problems(
        self, problems: tuple[RunProblem, ...], composition: StepComposition, key: RunKey, row: ExpandedRow,
    ) -> tuple[RunProblem, ...]:
        return tuple(
            replace(problem, sweep_source=source)
            if problem.model_address is not None
            and (source := self.sweep_problem_source(problem, composition, key, row)) is not None
            and source.kind == "sweep_value" else problem
            for problem in problems
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
        fields = self.variants[key.variant].steps[key.step].set_fields
        member = to_json(tuple(
            (role, binding.member.model_dump(mode="json")) for role, binding in sorted(inputs.items())
        ))
        subject = inputs.get("subject") if kind == "evaluate" else None
        model_source = self.model_sources.get(subject.producer) if subject is not None and subject.producer else None
        if model_source is None and subject is not None and subject.terminal is not None:
            reference = subject.member.reference
            assert isinstance(reference, ProductRef)
            terminal = self.terminal_sources[reference.request_id]
            documents = terminal.train.trial_documents
            preparation = self.prepare_train_trial(documents)
            if isinstance(preparation, tuple):
                return _LoweredRow(None, tuple(ProjectDiagnostic(
                    ResolutionCode.INVALID_PAYLOAD, "inputs.subject", problem.reason,
                    key.step, row=key.row, variant=key.variant, model_address=problem.model_address,
                ) for problem in preparation))
            evaluation = preparation.evaluation
            model_source = _ModelSource(
                None, documents.model, terminal.request_id,
                evaluation_graph=evaluation if isinstance(evaluation, GraphSpec) else None,
                evaluation_states=preparation.evaluation_states,
                evaluation_problem=evaluation if isinstance(evaluation, ProjectDiagnostic) else None,
            )
        source_models = (
            tuple(self.model_sources.get(producer) for producer in subject.candidates)
            if subject is not None and subject.candidates
            else (model_source,) if model_source is not None else ()
        )
        source_stamp = tuple(source.evaluation_stamp if source is not None else None for source in source_models)
        source_problem = model_source.evaluation_problem if model_source is not None else None
        checkpoint = inputs.get("checkpoint") if kind == "train" else None
        trained = self.model_sources.get(checkpoint.producer) if checkpoint is not None and checkpoint.producer else None
        checkpoint_source = trained.trained if trained is not None else None

        def lower(reader=None, previous=None):
            if definition is None:
                resolved = self.resolve_row(
                    composition, key, row, reader,
                    trial_resolution=train_rows.resolution if train_rows is not None else None,
                )
                if isinstance(resolved, ProjectDiagnostic):
                    return _LoweredRow(None, (replace(resolved, variant=key.variant),), resolved=False)
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
            if fields:
                document_uses = document_uses | frozenset(field.axis.set for field in fields.values())
            authored_model = None
            prepared_train = None
            remember_train = None
            train = _train_facts(payload, self.python_tasks) if kind == "train" else None
            has_differences = MODEL_ATTACHMENTS in payload
            model_lineage = ()
            if kind == "evaluate" and (document := document_ids.get(EVALUATION_MODEL)) is not None:
                authored = self.variants[key.variant].documents
                names = (document, *(source.document for source in source_models if source and source.document))
                model_lineage = tuple(base_chain(authored, name) for name in names)
                if reader is not None:
                    for chain in model_lineage:
                        for name in chain:
                            reader(name)
            lowering_stamp = (
                _RowLoweringStamp(kind, sha256_bytes(to_json((
                    payload, tuple(sorted(document_ids.items())), tuple(sorted(document_pointers.items())),
                    source_stamp, model_lineage, tuple(entry.model_dump(mode="json") for entry in row.task_additions),
                ))), member)
                if basis is not None and not fields else None
            )
            if previous is not None and lowering_stamp is not None and previous.lowering_stamp == lowering_stamp:
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
                            None, (problem,), lowering_stamp, document_ids=document_ids, document_uses=document_uses,
                        )
            if source_problem is not None and payload.get("evaluation_type") == ROLLOUT_TYPE:
                return _LoweredRow(
                    None, (replace(source_problem, step=key.step, row=key.row, variant=key.variant),),
                    lowering_stamp, document_ids=document_ids, document_uses=document_uses,
                )
            condition_origins = {}
            if kind in {"train", "synthesize"}:
                flat = kind == "synthesize"
                trial = payload if flat else payload.get("trial_documents")
                if (
                    train_rows is not None and isinstance(trial, dict)
                    and train_rows.resolution.contains(trial) and not has_differences and not fields
                ):
                    if (entry := train_rows.prepared) is not None and entry[0] is trial:
                        prepared_train = entry[1]
                    else:
                        def remember_train(prepared):
                            train_rows.prepared = trial, prepared
                if prepared_train is None:
                    payload, authored_model = self.lower_producer_model(key, payload, document_ids, document_pointers)
                from feedbax.compiler.set_fields import SetFieldProblem, lower_trial_sets

                if isinstance(payload, dict) and authored_model is not None and (
                    fields
                ):
                    trial = payload if flat else payload["trial_documents"]
                    try:
                        stored_task = StoredTaskDocument.model_validate(trial.get("task"))
                        model, task, condition_origins = lower_trial_sets(
                            authored_model.model, stored_task.lower(), fields,
                            reader or self.variants[key.variant].documents.get,
                            self.components or _builtin_model_registry(),
                            functions=self.functions or _builtin_expression_registry(),
                        )
                    except (SetFieldProblem, ValidationError) as error:
                        if isinstance(error, SetFieldProblem):
                            place = ("set_fields", error.axis)
                            if error.condition:
                                place = (*place, "condition")
                            elif error.entry is not None:
                                place = (*place, "entries", error.entry)
                            problem = RunProblem(pointer(place), str(error))
                        else:
                            problem = RunProblem("/task" if flat else "/trial_documents/task", "A Set field needs a Task document.")
                        return _LoweredRow(
                            None, (), lowering_stamp, problems=(problem,), document_ids=document_ids,
                            document_uses=document_uses, train=train,
                        )
                    authored_model = replace(
                        authored_model, model=model, authored_model=authored_model.model,
                        stamp=sha256_bytes(to_json((authored_model.document, model.model_dump(mode="json")))),
                    )
                    payload = _with_producer_model(payload, authored_model, document_ids, document_pointers)
                    trial = payload if flat else payload["trial_documents"]
                    previous_keys = stored_task.validation_trials
                    trial["task"] = StoredTaskDocument.from_lowered(
                        task, trial_keys=previous_keys if len(previous_keys) == len(task.validation_trials) else None,
                    ).model_dump(mode="json")
                    if not flat:
                        train = _train_facts(payload, self.python_tasks)
            elif kind == "evaluate":
                reference = subject.member.reference if subject is not None else None
                read = reader or self.variants[key.variant].documents.get
                if isinstance(reference, BestRank):
                    assert subject is not None
                    payload = self.lower_best_evaluate(
                        key, payload, subject, reference, has_differences, document_ids, read, row.task_additions,
                    )
                else:
                    payload = self.lower_evaluate_payload(
                        key, payload, model_source, reference, has_differences, document_ids, read, row.task_additions,
                    )
            if isinstance(payload, ProjectDiagnostic):
                payload = self.locate_sweep_problem(payload, composition, key, row)
                return _LoweredRow(
                    None, (payload,), lowering_stamp, document_ids=document_ids, document_uses=document_uses,
                    train=train,
                )
            refusals: tuple[RunProblem, ...] = ()
            dataset_port = None
            if isinstance(payload, _EvaluatePayload):
                dataset_port = payload.dataset_port
                refusals = tuple(
                    RunProblem(problem.path, problem.detail, problem.model_address, problem.variation_target)
                    for problem in payload.problems
                )
                if payload.payload is None:
                    return _LoweredRow(
                        None, (), lowering_stamp,
                        problems=self.locate_sweep_run_problems(refusals, composition, key, row), document_ids=document_ids,
                        document_uses=document_uses, train=train,
                    )
                payload = payload.payload
            lowered = _encode_row(
                key, kind, payload, inputs, lowering_stamp,
                self.work.request_documents, self.prepare_train_trial, document_ids,
                self.python_tasks, model_source=authored_model, document_pointers=document_pointers,
                prepared_train=prepared_train, remember_train=remember_train,
                training_methods=self.training_methods, previous=previous,
                checkpoint_source=checkpoint_source, components=self.components or _builtin_model_registry(),
                synthesis_methods=self.synthesis_methods,
            )
            return replace(
                lowered,
                diagnostics=tuple(self.locate_sweep_problem(problem, composition, key, row)
                                  for problem in lowered.diagnostics),
                document_ids=document_ids,
                document_uses=document_uses,
                model_source=lowered.model_source or authored_model,
                train=train,
                problems=self.locate_sweep_run_problems(
                    (*refusals, *_set_condition_problems(lowered.problems, condition_origins)), composition, key, row,
                ),
                dataset_port=dataset_port,
            )

        if self.previous is None:
            return lower()
        return _work(
            self.previous.rows,
            self.work.rows,
            key,
            (kind, basis, to_json(tuple(row.patches.items())), tuple(sorted(inputs.items())), source_stamp,
             to_json(tuple(entry.model_dump(mode="json") for entry in row.task_additions)),
             to_json({name: value.model_dump(mode="json") for name, value in fields.items()})),
            self.authored_inputs[key.variant],
            lower,
        )

    def prepare_train_trial(self, authored: TrialDocuments) -> _TrainPreparation | tuple[Problem, ...]:
        from feedbax.execution.trial_resolution import resolve_trial_execution

        identity = to_json(
            authored.model_dump(exclude_unset=True, round_trip=True), inf_nan_mode="constants",
        )
        if (prepared := self.trial_preparations.get(identity)) is not None:
            return prepared
        components = self.components or _builtin_model_registry()
        # The plan needs the resolved graph, source addresses and conditions, never a constructed model.
        paired = resolve_trial_execution(
            authored.model, authored.task_source,
            components, self.functions or _builtin_expression_registry(),
            python_tasks=self.python_tasks, objective=authored.objective, input_bindings=authored.input_bindings,
        )
        result: _TrainPreparation | tuple[Problem, ...]
        if isinstance(paired, tuple):
            result = paired
        else:
            graph = paired.graph.graph
            checkpoint_problem = None
            if problem := model_owner_problem(graph, "checkpoint", components):
                address = paired.lowered.source_address(pointer(problem.path))
                path = ("trial_documents", "model", *address.stored_path()) if address is not None else problem.path
                checkpoint_problem = RunProblem(pointer(path), problem.reason, address)
            evaluation: GraphSpec | ProjectDiagnostic = graph
            if problem := model_owner_problem(graph, "evaluate", components):
                address = paired.lowered.source_address(pointer(problem.path))
                evaluation = ProjectDiagnostic(
                    ResolutionCode.INVALID_PAYLOAD, pointer(("inputs", "subject", *problem.path)), problem.reason,
                    model_address=address,
                )
            result = _TrainPreparation(
                checkpoint_problem, evaluation, graph, paired.lowered, paired.retained_state_fields,
            )
        self.trial_preparations[identity] = result
        return result

    def lower_producer_model(
        self, key: RunKey, payload: Payload, document_ids: Mapping[str, str], document_pointers: Mapping[str, str],
    ) -> tuple[Payload | ProjectDiagnostic, _ModelSource | None]:
        from feedbax.compiler.model_attachment import resolve_effective_model
        from feedbax.compiler.model_document import ModelCompilationProblem

        slot = trial_model_path(payload)
        raw = payload if len(slot) == 1 else payload.get("trial_documents")
        model_payload = raw.get("model") if isinstance(raw, dict) else None
        document = document_ids.get("payload." + ".".join(slot))
        address = None
        has_differences = MODEL_ATTACHMENTS in payload
        try:
            payload, differences = _model_differences(payload)
            if not isinstance(model_payload, dict) or "interior" not in model_payload:
                if differences:
                    address = differences[0].address
                    raise ValueError("Model differences require an authored Model")
                return payload, None
            model = StoredModelDocument.model_validate(model_payload).lower()
            if differences:
                model = resolve_effective_model(
                    model, self.components or _builtin_model_registry(), differences,
                    functions=self.functions,
                )
            model_source = _ModelSource(
                document, model, sha256_bytes(to_json((document, model.model_dump(mode="json")))),
            )
            if differences:
                payload = _with_producer_model(payload, model_source, document_ids, document_pointers)
            return payload, model_source
        except ModelCompilationProblem as error:
            address, detail = error.address, str(error)
        except (KeyError, ValueError) as error:
            if not has_differences:
                return payload, None  # The ordinary Train validator reports invalid model documents.
            detail = str(error)
        return _model_difference_problem(key, document, address, detail), None

    def lower_evaluate_payload(
        self, key: RunKey, payload: Payload, model_source: _ModelSource | None,
        reference: object, has_differences: bool,
        document_ids: Mapping[str, str], read: Callable[[str], Document | None],
        task_additions: tuple[TaskAdditions, ...],
    ) -> Payload | ProjectDiagnostic | _EvaluatePayload:
        original = payload.get("params")
        slots = self.evaluation_slots(key, payload, model_source, reference, document_ids, read, task_additions)
        if isinstance(slots, ProjectDiagnostic):
            return slots
        lowered, model, task = slots
        result = self._lower_evaluate_payload(key, lowered, model_source, reference, has_differences, model, read)

        def move(problem: ProjectDiagnostic) -> ProjectDiagnostic:
            parts = path_parts(problem.path)
            if task and parts[:3] == ("params", "variation", "0"):
                return replace(
                    problem, path=f"/trial_documents/task{pointer(parts[3:])}", document=document_ids.get(EVALUATION_TASK),
                )
            if parts[:2] == ("params", "variation"):
                index = int(parts[2]) - task if len(parts) > 2 and parts[2].isdigit() else None
                target = variation_targets(original)(str(index) if index is not None else parts[2]) if len(parts) > 2 else None
                return replace(problem, variation_target=target)
            return problem

        return _map_problem(result, move)

    def evaluation_slots(
        self, key: RunKey, payload: Payload, model_source: _ModelSource | None, reference: object,
        document_ids: Mapping[str, str], read: Callable[[str], Document | None],
        task_additions: tuple[TaskAdditions, ...],
    ) -> tuple[Payload, _EvaluationModel | None, bool] | ProjectDiagnostic:
        """Lower an Evaluate step's own Model and Task documents against its subject run's.

        The Task slot becomes the task replacement ahead of the authored variation, or nothing when it is the
        subject's task. The Model slot becomes the differences of its content from the nearest document it
        shares with the subject run's Model document.
        """
        slots = payload.get("trial_documents", {})
        payload = {name: value for name, value in payload.items() if name != "trial_documents"}

        def problem(path: str, detail: str) -> ProjectDiagnostic:
            return ProjectDiagnostic(
                ResolutionCode.INVALID_PAYLOAD, path, detail, key.step, row=key.row, variant=key.variant,
                document=document_ids.get(path),
            )

        if not isinstance(slots, dict) or slots.keys() - {"model", "task"}:
            return problem("payload.trial_documents", "An Evaluate reads a Model and a Task document.")
        documents_read = self.evaluation_source_documents(reference) if isinstance(reference, ProductRef) else None
        if documents_read is None or model_source is None:
            if not slots and not task_additions:
                return payload, None, False
            path = EVALUATION_MODEL if "model" in slots else EVALUATION_TASK
            return problem(path, "Requires a step that produced a Model.")
        model = None
        if "model" in slots:
            name = document_ids.get(EVALUATION_MODEL)
            subject = model_source.document
            documents = self.variants[key.variant].documents
            if subject is not None:
                for member in base_chain(documents, subject):
                    read(member)
            content = slots["model"]
            if name is None or not isinstance(content, dict):
                return problem(EVALUATION_MODEL, "Name a Model document.")
            shared = nearest_shared_base(documents, subject, name) if subject is not None else None
            if shared is None:
                detail = f"Shares no document with {member_name(documents, subject)}." if subject else "Requires a subject Model document."
                return replace(problem(EVALUATION_MODEL, detail), code=ResolutionCode.BAD_REFERENCE)
            base = compose_document(read, shared)
            if isinstance(base, ProjectDiagnostic):
                return replace(base, step=key.step, row=key.row, variant=key.variant)
            assert isinstance(base.payload, dict)
            if differences := diff(base.payload, content, keyed_collection=_MODEL_COLLECTION):
                model = _EvaluationModel(name, differences)
        params = payload.get("params")
        has_task_slot = "task" in slots or bool(task_additions)
        if payload.get("evaluation_type") != ROLLOUT_TYPE or not isinstance(params, dict):
            if has_task_slot:
                return problem("payload.params", "Declare rollout parameters.")
            return payload, model, False
        authored = model_source.trial.task if model_source.trial is not None else None
        inherited = authored or documents_read.task_source
        stored_task = model_source.stored_task
        try:
            if "task" in slots:
                selected = _AUTHORED_EVALUATION_TASK.validate_python(slots["task"])
            else:
                selected = stored_task
            if task_additions:
                task = selected.lower() if isinstance(selected, StoredTaskDocument) else inherited
                if not isinstance(task, TaskDocument) or isinstance(selected, PythonTaskSelection):
                    return problem(EVALUATION_TASK, "A Set entry's Task variables and links need a Task document.")
                trial_keys = selected.validation_trials if isinstance(selected, StoredTaskDocument) else None
                selected = StoredTaskDocument.from_lowered(add_to_task(task, task_additions), trial_keys=trial_keys or None)
            if has_task_slot and selected is not None:
                stored_content = selected.model_dump(mode="json")
                chosen = selected.lower() if isinstance(selected, StoredTaskDocument) else selected
                if chosen != inherited:
                    variation = params.get("variation", [])
                    values = tuple(variation[name] for name in sorted(variation)) if isinstance(variation, dict) else variation
                    params = params | {"variation": [{"op": "replace", "path": "task", "value": stored_content}, *values]}
                else:
                    has_task_slot = False
                stored_task = selected if isinstance(selected, StoredTaskDocument) else None
            lowered = _lower_evaluation_task_edits(params, inherited, stored_task)
        except CannotRun as error:
            return replace(problem(error.path, error.reason), variation_target=variation_targets(payload.get("params"))(
                path_parts(error.path)[-1],
            ))
        except (TypeError, ValueError) as error:
            return problem(EVALUATION_TASK if has_task_slot else "payload.params.variation", str(error))
        return payload | {"params": lowered}, model, has_task_slot

    def _lower_evaluate_payload(
        self, key: RunKey, payload: Payload, model_source: _ModelSource | None,
        reference: object, has_differences: bool, model: _EvaluationModel | None,
        read: Callable[[str], Document | None],
    ) -> Payload | ProjectDiagnostic | _EvaluatePayload:
        fields = self.variants[key.variant].steps[key.step].set_fields
        if not has_differences and model is None and not fields:
            return self.lower_evaluate_trials(key, payload, reference, model_source)
        original = payload.get("params")
        lowered = self.lower_evaluate_model(key, payload, model_source, reference, model, read)
        if isinstance(lowered, (ProjectDiagnostic, _EvaluatePayload)):
            return lowered
        varied, condition_origins = lowered
        result = self.lower_evaluate_trials(key, varied, reference, model_source, condition_origins=condition_origins)
        # Derived Model patches append to an ordered copy. Report a Task
        # refusal at the original named edit without changing request encoding.
        if isinstance(original, dict):
            named = original.get("variation")
            params = varied.get("params")
            if isinstance(named, dict) and isinstance(params, dict) and isinstance(params.get("variation"), list):
                return _map_problem(result, lambda problem: _authored_variation_problem(problem, tuple(sorted(named))))
        return result

    def initial_model_source(
        self, reference: InitialModelReference, document: RootDocument, differences: tuple[ModelDifference, ...],
    ) -> InitialModelSource:
        from feedbax.compiler.model_attachment import resolve_effective_model

        source = authored_document_input(reference, document)
        assert isinstance(source, InitialModelSource)
        if not differences:
            return source
        model = resolve_effective_model(
            source.document, self.components or _builtin_model_registry(), differences, functions=self.functions,
        )
        return source.model_copy(update={"document": model})

    def lower_initial_evaluate(
        self, template: StepComposition, key: RunKey, row: ExpandedRow,
    ) -> tuple[_LoweredRow, dict[str, PlannedInput]]:
        from feedbax.compiler.model_document import ModelCompilationProblem
        from feedbax.compiler.set_fields import SetFieldProblem
        from feedbax.execution.trial_resolution import evaluation_conditions, model_compilation_problems, pair_trial_source
        from feedbax.contracts.rollout import CannotRun
        from feedbax.tasks.registry import resolve_python_task_instance
        from feedbax.training.evaluation_variant import resolve_evaluation_variant

        resolved = self.resolve_row(template, key, row, None)
        if isinstance(resolved, ProjectDiagnostic):
            return _LoweredRow(None, (replace(resolved, variant=key.variant),), resolved=False), {}
        path = "inputs.subject"
        project = self.variants[key.variant]
        fields = project.steps[key.step].set_fields
        document_ids = resolved.document_ids
        document_uses = resolved.document_uses | frozenset(field.axis.set for field in fields.values())
        followed = frozenset()
        replica_ids = ()
        try:
            if resolved.payload.get("evaluation_type") != ROLLOUT_TYPE:
                raise ValueError("An initial Model subject requires a rollout Evaluate")
            subject_ref = template.inputs["subject"]
            payload, differences = _model_differences(resolved.payload)
            subject_document = resolved.input_documents["subject"]
            if isinstance(subject_ref, LoadedModelReference):
                source = authored_document_input(subject_ref, subject_document)
                assert isinstance(source, ImportedModelSource)
                imported = self.read_imported_model(key.variant, subject_ref, subject_document)
                model, replica_ids = imported.model, imported.replica_ids
                navigation = LoadedModelDocument.model_validate(subject_document.payload).model
                if (
                    EVALUATION_MODEL not in document_ids and not navigation.unresolved and navigation.pointer is None
                    and (nav := project.documents.get(navigation.document)) is not None
                    and nav.schema_id == STORED_MODEL_DOCUMENT_SCHEMA_ID
                ):
                    followed = frozenset({navigation.document})
                    document_ids = {**document_ids, EVALUATION_MODEL: navigation.document}
                    document_uses |= followed
            else:
                assert isinstance(subject_ref, InitialModelReference)
                source = self.initial_model_source(subject_ref, subject_document, differences)
                model = source.document
                differences = ()
                # Set sweep resolution can also expose the same row's patched initial Model in the Model slot.
                if payload.get("trial_documents") == {"model": subject_document.payload}:
                    payload = {name: value for name, value in payload.items() if name != "trial_documents"}
            effective_model = model
            if isinstance(subject_ref, LoadedModelReference) and "trial_documents" in payload:
                path = EVALUATION_MODEL
                slots = payload["trial_documents"]
                payload = {name: value for name, value in payload.items() if name != "trial_documents"}
                if not isinstance(slots, dict) or slots.keys() != {"model"}:
                    raise ValueError("A Loaded Model rollout takes its Task from its Task input.")
                name = resolved.document_ids.get(EVALUATION_MODEL)
                shared = (
                    nearest_shared_base(project.documents, navigation.document, name)
                    if name is not None and not navigation.unresolved and navigation.pointer is None else None
                )
                if shared is None or not isinstance(slots["model"], dict):
                    raise ValueError("The Evaluate Model must share a document with the Loaded Model.")
                base = compose_document(project.documents.get, shared)
                if isinstance(base, ProjectDiagnostic):
                    raise ValueError(base.detail)
                assert isinstance(base.payload, dict)
                changes = diff(base.payload, slots["model"])
                effective_model = _model_with_stored_differences(model, changes)
            components = self.components or _builtin_model_registry()
            functions = self.functions or _builtin_expression_registry()
            tasks = self.python_tasks
            path = "inputs.task"
            task_ref = template.inputs.get("task")
            if not isinstance(task_ref, TaskDocumentReference):
                raise ValueError("Declare the Task document for the initial Model rollout")
            task_document = resolved.input_documents["task"]
            if task_document.schema_id == PYTHON_TASK_SCHEMA_ID:
                selection = PythonTaskSelection.model_validate(task_document.payload)
                if isinstance(unresolved := resolve_python_task_instance(selection, tasks), tuple):
                    # The Task's conditions are unknown, so its trial rule does not lower to a request.
                    return _LoweredRow(
                        None, (), problems=_trial_problems(unresolved, None, initial=True),
                        document_ids=document_ids, document_uses=document_uses,
                    ), {}
            task = authored_document_input(task_ref, resolved.input_documents["task"], python_tasks=tasks)
            assert isinstance(task, (TaskDocument, PythonTaskSelection))
            sources: dict[str, DocumentInput] = {"subject": source, "task": task}
            if (objective_ref := template.inputs.get("objective")) is not None:
                path = "inputs.objective"
                if not isinstance(objective_ref, ObjectiveDocumentReference):
                    raise ValueError("Declare an Objective document input")
                objective = authored_document_input(objective_ref, resolved.input_documents["objective"])
                assert isinstance(objective, ObjectiveDocument)
                sources["objective"] = objective
            path = "payload.params"
            raw = payload["params"]
            if not isinstance(raw, dict) or "trial_set" not in raw or "trials" in raw:
                raise ValueError("Declare a trial-set rule for managed Evaluate")
            if problem := _authored_graph_variation_problem(key, payload):
                return _LoweredRow(None, (problem,)), {}
            stored_task = (
                StoredTaskDocument.model_validate(task_document.payload) if isinstance(task, TaskDocument) else None
            )
            raw = _lower_evaluation_task_edits(raw, task, stored_task)
            payload = payload | {"params": raw}
            condition_origins = {}
            if fields or differences or effective_model != model:
                payload, attached = self.evaluate_model_variation(
                    key, payload, model, effective_model, differences, task, project.documents.get,
                )
                condition_origins = attached.condition_origins
                raw = payload["params"]
            settings = authored_rollout_settings(raw)
            paired = pair_trial_source(
                model, task, components, functions,
                python_tasks=tasks, input_bindings=settings.input_bindings,
            )
            unpaired = paired if isinstance(paired, tuple) else paired.problems
            refusals = _trial_problems(unpaired, model, initial=True)
            conditions = evaluation_conditions(task, tasks)
            if not unpaired:
                assert not isinstance(paired, tuple)
                variant = resolve_evaluation_variant(
                    paired.lowered.graph, task, settings.variation, components=components,
                )
                if variant.task != task or variant.graph != paired.lowered.graph:
                    paired = pair_trial_source(
                        model, variant.task, components, functions,
                        python_tasks=tasks, input_bindings=settings.input_bindings, graph_override=variant.graph,
                    )
                    conditions = evaluation_conditions(variant.task, tasks)
                    if isinstance(paired, tuple) or paired.problems:
                        problems = paired if isinstance(paired, tuple) else paired.problems
                        refusals = tuple(
                            RunProblem(
                                _set_condition_placement(problem, condition_origins) or refusal.path,
                                refusal.reason, problem.model_address,
                            )
                            for problem, refusal in ((item, variant.task_refusal(item)) for item in problems)
                        )
            trials = resolve_trial_ids(raw["trial_set"], conditions)
            payload = payload | {"params": {
                **{name: value for name, value in raw.items() if name != "trial_set"},
                "trials": list(trials),
            }}
            inputs = {role: _plan_document_content(value, self.documents) for role, value in sources.items()}
            narrowed = _encode_row(
                key, "evaluate", payload, inputs, None, self.work.request_documents,
                self.prepare_train_trial, document_ids, self.python_tasks,
            )
            problems = (*refusals, *narrowed.problems) if narrowed.request is not None else narrowed.problems
            return replace(
                narrowed, document_ids=document_ids, document_uses=document_uses,
                problems=self.locate_sweep_run_problems(problems, template, key, row),
                pinned_model=model if isinstance(subject_ref, LoadedModelReference) else None,
                followed_documents=followed,
                dataset_port=(
                    rollout_dataset_port(
                        paired.task.interface.signals, settings.states_schema, None,
                        model_states=paired.model.retained_state_fields,
                        replicate=replica_coordinate(replica_ids) if len(replica_ids) > 1 else None,
                    ) if not isinstance(paired, tuple) and not paired.problems else None
                ),
            ), inputs
        except SetFieldProblem as error:
            problem = _evaluate_set_problem(key, error)
            return _LoweredRow(
                None, (), problems=(RunProblem(problem.path, problem.detail),),
                document_ids=document_ids, document_uses=document_uses,
                pinned_model=model if isinstance(subject_ref, LoadedModelReference) else None,
                followed_documents=followed,
            ), {}
        except ModelCompilationProblem as error:
            problems = _trial_problems(model_compilation_problems(error), None, initial=True)
            return _LoweredRow(
                None, (), problems=self.locate_sweep_run_problems(problems, template, key, row),
                document_ids=document_ids, document_uses=document_uses,
            ), {}
        except (KeyError, OSError, ValueError) as error:
            return _LoweredRow(None, (ProjectDiagnostic(
                ResolutionCode.INVALID_PAYLOAD,
                error.path if isinstance(error, CannotRun) else path,
                error.reason if isinstance(error, CannotRun) else str(error),
                key.step, row=key.row, variant=key.variant, document=resolved.document_ids.get(path),
            ),), document_ids=document_ids, document_uses=document_uses), {}

    def read_imported_model(
        self, variant: str, reference: LoadedModelReference, document: RootDocument,
    ) -> ResolvedImportedModel:
        if self.readers.imported_model is None:
            raise ValueError("A Loaded Model must resolve from the project Store.")
        loaded = LoadedModelDocument.model_validate(document.payload)
        legacy_model = (
            composed_loaded_model_projection(loaded, self.variants[variant].documents.get)
            if loaded.archive is None else None
        )
        return self.readers.imported_model(LoadedModelRead(
            reference.source.document, loaded.model_dump_json().encode(), legacy_model,
        ))

    def lower_best_evaluate(
        self, key: RunKey, payload: Payload, subject: PlannedInput, rank: BestRank, has_differences: bool,
        document_ids: Mapping[str, str], read: Callable[[str], Document | None],
        task_additions: tuple[TaskAdditions, ...],
    ) -> Payload | ProjectDiagnostic | _EvaluatePayload:
        def problem(detail: str) -> ProjectDiagnostic:
            return ProjectDiagnostic(
                ResolutionCode.BAD_REFERENCE, subject.best_path, detail, key.step,
                row=key.row, variant=key.variant,
            )

        if payload.get("evaluation_type") != ROLLOUT_TYPE:
            return problem("Best requires managed rollout Evaluate.")
        common = None
        unpaired = None
        ports = []
        for producer, candidate in zip(subject.candidates, rank.candidates, strict=True):
            model = self.model_sources.get(producer)
            name = f"{producer.variant}/{producer.step}/{producer.row}"
            if model is None or model.evaluation_graph is None:
                message = model.evaluation_problem.detail if model and model.evaluation_problem else "no managed Model and Task"
                return problem(f"Best candidate {name}: {message}")
            if refusal := _authored_graph_variation_problem(key, payload):
                return refusal
            lowered = self.lower_evaluate_payload(
                key, payload, model, candidate.reference, has_differences, document_ids, read, task_additions,
            )
            if isinstance(lowered, ProjectDiagnostic):
                return replace(lowered, detail=f"Best candidate {name}: {lowered.detail}")
            if isinstance(lowered, _EvaluatePayload):
                if lowered.problems:
                    unpaired = unpaired or replace(
                        lowered, problems=tuple(
                            replace(problem, detail=f"Best candidate {name}: {problem.detail}") for problem in lowered.problems
                        ),
                    )
                if lowered.payload is None:
                    assert unpaired is not None
                    return unpaired
                if lowered.dataset_port is not None:
                    ports.append(lowered.dataset_port)
                lowered = lowered.payload
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
        return _EvaluatePayload(
            common, unpaired.problems if unpaired is not None else (),
            common_rollout_port(ports) if len(ports) == len(subject.candidates) else None,
        )

    def lower_evaluate_trials(
        self, key: RunKey, payload: Payload, reference: object, model_source: _ModelSource | None,
        *, condition_origins: Mapping[str, str] | None = None,
    ) -> Payload | ProjectDiagnostic | _EvaluatePayload:
        from feedbax.execution.trial_resolution import pair_trial_source, resolve_python_task
        from feedbax.contracts.rollout import CannotRun
        from feedbax.training.evaluation_variant import resolve_evaluation_task, resolve_evaluation_variant

        if (
            payload.get("evaluation_type") != ROLLOUT_TYPE or not isinstance(reference, ProductRef)
            or model_source is None or model_source.evaluation_graph is None
        ):
            return payload
        source = self.evaluation_source_spec(reference)
        if source is None:
            return payload
        source_documents = source.trial_documents
        raw = payload.get("params")
        if not isinstance(raw, dict):
            return payload
        location = "trial_set"
        try:
            if "trial_set" not in raw or "trials" in raw:
                raise ValueError("Declare a trial-set rule for managed Evaluate.")
            settings = authored_rollout_settings(raw)
            documents = source_documents
            authored = model_source.trial
            if authored is not None and isinstance(authored.task, TaskDocument):
                documents = documents.model_copy(update={"task": authored.task, "input_bindings": authored.input_bindings})
            variant = resolve_evaluation_variant(
                model_source.evaluation_graph, documents.task_source, settings.variation,
                components=self.components or _builtin_model_registry(),
            )
            refusals = ()
            paired_task = None
            state_fields = model_source.evaluation_states
            if settings.input_bindings or variant.task != documents.task_source or variant.graph != model_source.evaluation_graph:
                pairing = pair_trial_source(
                    documents.model, variant.task,
                    self.components or _builtin_model_registry(), self.functions or _builtin_expression_registry(),
                    python_tasks=self.python_tasks, input_bindings=documents.input_bindings | settings.input_bindings,
                    graph_override=variant.graph,
                )
                problems = pairing if isinstance(pairing, tuple) else pairing.problems
                if problems:
                    refusals = tuple(
                        ProjectDiagnostic(
                            ResolutionCode.INVALID_PAYLOAD,
                            _set_condition_placement(problem, condition_origins) or cannot.path, cannot.reason, key.step,
                            row=key.row, variant=key.variant, model_address=problem.model_address,
                        )
                        for problem, cannot in ((item, variant.task_refusal(item)) for item in problems)
                    )
                else:
                    assert not isinstance(pairing, tuple)
                    paired_task = pairing.task
                    state_fields = pairing.model.retained_state_fields
            if paired_task is not None:
                conditions, signals = paired_task.condition_ids, paired_task.interface.signals
            elif isinstance(variant.task, TaskDocument):
                conditions, signals = variant.task.condition_ids, variant.task.signals
            else:
                template = resolve_python_task(variant.task, self.python_tasks)
                if isinstance(template, tuple):
                    if refusals:
                        return _EvaluatePayload(None, refusals)
                    raise ValueError(template[0].reason)
                conditions = template.condition_ids
                signals = tuple(ScopedValue(id=name, name=name, type=type_, level=Level.STEP)
                                for name, type_ in template.signals.items())
            trials = resolve_trial_ids(raw["trial_set"], conditions)
            params = {name: value for name, value in raw.items() if name != "trial_set"} | {"trials": list(trials)}
            if authored is not None and isinstance(authored.task, TaskDocument):
                # Evaluate's Task can change implicit assignments. Send those differences to the worker, whose
                # Train admission fixes the original pairing by ID.
                resolved = documents.resolved_signal_bindings(
                    {signal.id for signal in signals}, settings.input_bindings,
                )
                runtime_task = resolve_evaluation_task(source_documents.task_source, settings.variation)
                runtime_signals = runtime_task.signals if isinstance(runtime_task, TaskDocument) else signals
                runtime = source_documents.resolved_signal_bindings({
                    signal.id for signal in runtime_signals
                })
                overrides = {key: value for key, value in resolved.items() if key not in runtime or runtime[key] != value}
                if overrides:
                    params["input_bindings"] = overrides
                else:
                    params.pop("input_bindings", None)
            lowered = payload | {"params": params}
            replicate = (
                replica_coordinate()
                if isinstance(source, DocumentTrainingRequestSpec) and source.training_config.replicates > 1 else None
            )
            return _EvaluatePayload(
                lowered, refusals,
                rollout_dataset_port(
                    signals, settings.states_schema, rollout_update_axis(reference),
                    model_states=state_fields, replicate=replicate,
                ) if not refusals else None,
            )
        except ValueError as error:
            return ProjectDiagnostic(
                ResolutionCode.INVALID_PAYLOAD,
                error.path if isinstance(error, CannotRun) else f"payload.params.{location}",
                error.reason if isinstance(error, CannotRun) else str(error), key.step,
                row=key.row, variant=key.variant,
            )

    def evaluation_source_documents(self, reference: ProductRef) -> TrialDocuments | None:
        source = self.evaluation_source_spec(reference)
        return source.trial_documents if source is not None else None

    def evaluation_source_spec(self, reference: ProductRef) -> DocumentTrainingRequestSpec | SynthesisRequestSpec | None:
        if (terminal := self.terminal_sources.get(reference.request_id)) is not None:
            return terminal.train
        if (source := self.requests.get(reference.request_id)) is None:
            return None
        producer = request_identity.decode_request(source, self.documents.__getitem__).spec
        return producer if isinstance(producer, (DocumentTrainingRequestSpec, SynthesisRequestSpec)) else None

    def evaluate_model_variation(
        self, key: RunKey, payload: Payload, source: ModelDocument, evaluated: ModelDocument,
        differences: tuple[ModelDifference, ...], task: TaskDocument | PythonTaskSelection,
        read: Callable[[str], Document | None],
    ) -> tuple[Payload, LoweredEvaluateAttachments]:
        """Lower an Evaluate's authored Model and Set fields to qualified rollout edits."""
        from feedbax.compiler.model_attachment import lower_evaluate_attachments
        from feedbax.compiler.set_fields import SetFieldProblem
        from feedbax.training.evaluation_variant import resolve_evaluation_task

        params = payload.get("params")
        settings = authored_rollout_settings(params)
        assert isinstance(params, dict)
        prior = settings.variation
        edits = tuple(prior[name] for name in sorted(prior)) if isinstance(prior, dict) else prior
        chosen_task = resolve_evaluation_task(task, edits)
        fields = self.variants[key.variant].steps[key.step].set_fields
        if fields and not isinstance(chosen_task, TaskDocument):
            axis, field = next(iter(fields.items()))
            raise SetFieldProblem(axis, field.axis.set, None, "A Set field needs a Task document.")
        attached = lower_evaluate_attachments(
            source, evaluated, None, self.components or _builtin_model_registry(), differences,
            functions=self.functions or _builtin_expression_registry(),
            trained_task=task if isinstance(task, TaskDocument) else None,
            evaluated_task=chosen_task if isinstance(chosen_task, TaskDocument) else None,
            sets=(fields, read) if fields else None,
        )
        variation = (*edits, *attached.edits)
        if attached.task is not None and attached.task != chosen_task:
            variation = (*variation, OverridePatch(path="task", op="replace", value=attached.task.model_dump(mode="json")))
        if variation == edits:
            return payload, attached
        return payload | {"params": params | {
            "variation": [edit.model_dump(mode="json", exclude_unset=True) for edit in variation],
        }}, attached

    def lower_evaluate_model(
        self, key: RunKey, payload: Payload, model_source: _ModelSource | None, reference: object,
        model: _EvaluationModel | None, read: Callable[[str], Document | None],
    ) -> tuple[Payload, dict[str, str]] | ProjectDiagnostic | _EvaluatePayload:
        from feedbax.compiler.set_fields import SetFieldProblem
        from feedbax.compiler.model_document import ModelCompilationProblem, model_input_prototypes
        from feedbax.compiler.resolution import resolve_graph_parameters
        from feedbax.application.task_signals import matched_task_inputs, model_task_signals
        from feedbax.compiler.graph_schema import validate_graph_connection_schema
        from feedbax.contracts.rollout import CannotRun
        from feedbax.training.evaluation_variant import resolve_evaluation_variant

        address = None
        document = model_source.document if model_source is not None else None
        differences: tuple[ModelDifference, ...] = ()
        try:
            stripped, differences = _model_differences(payload)
            if differences:
                address = differences[0].address
            else:
                document = model.document if model is not None else document
            if payload.get("evaluation_type") != ROLLOUT_TYPE:
                raise ValueError("Attachment differences require a rollout Evaluate with an authored Train Model")
            if not isinstance(reference, ProductRef):
                raise ValueError("Attachment differences require a planned Train subject")
            documents = self.evaluation_source_documents(reference)
            if documents is None:
                raise ValueError("Attachment differences require a model-producing request")
            if model_source is None:
                raise ValueError("Attachment differences need a saved Model document")
            components = self.components or _builtin_model_registry()
            evaluated = model_source.model
            if model is not None:
                try:
                    evaluated = _model_with_stored_differences(model_source.model, model.differences)
                except DifferenceNotApplied as error:
                    return _model_difference_problem(
                        key, model.document, None, str(error), path=f"{EVALUATION_MODEL}{error.path.replace('/', '.')}",
                    )
                except ValueError as error:
                    return _model_difference_problem(key, model.document, None, str(error), path=EVALUATION_MODEL)
            params = authored_rollout_settings(payload.get("params"))
            source_task = documents.task_source
            try:
                stripped, attached = self.evaluate_model_variation(
                    key, stripped, model_source.model, evaluated, differences, source_task, read,
                )
            except ValueError as error:
                if model is not None and not isinstance(error, (ModelCompilationProblem, SetFieldProblem)):
                    address = _refused_difference(model, model_source.model, components, self.functions)
                raise
            variation = authored_rollout_settings(stripped.get("params")).variation
            if not attached.edits and variation == params.variation:
                return stripped, attached.condition_origins
            effective = resolve_evaluation_variant(
                attached.before.graph, source_task, variation, components=components,
            )
            signals = model_task_signals(effective.task, self.python_tasks)
            bound = matched_task_inputs(attached.model, signals, documents.input_bindings | params.input_bindings)
            prototypes = model_input_prototypes(effective.graph, attached.model, {
                port.name: signal.type for port, signal in bound
            })
            resolution = resolve_graph_parameters(effective.graph, prototypes, components, allow_incomplete=True)
            issues = validate_graph_connection_schema(
                effective.graph, "/graph", component_registry=components, _resolution=resolution,
            )
            if issue := next((issue for issue in issues if issue.severity == "error"), None):
                issue_path = (issue.location or {}).get("path", "/graph")
                address = attached.after.source_address(issue_path) or address
                raise ValueError(issue.message)
            return stripped, attached.condition_origins
        except SetFieldProblem as error:
            return _EvaluatePayload(None, (_evaluate_set_problem(key, error),))
        except ModelCompilationProblem as error:
            return _EvaluatePayload(None, tuple(
                _model_difference_problem(key, document, problem.address, problem.reason,
                                          path=pointer(problem.address.stored_path()))
                for problem in error.problems
            ))
        except CannotRun as error:
            return _model_difference_problem(key, document, address, error.reason, path=error.path)
        except ValueError as error:
            detail = str(error)
        path = pointer(address.stored_path()) if address else EVALUATION_MODEL if not differences else None
        return _model_difference_problem(key, document, address, detail, path=path)

    def compile_step(
        self, variant: str, name: str, active: tuple[tuple[str, str], ...]
    ) -> tuple[_CompiledRun, ...]:
        project = self.variants[variant]
        authored = project.steps[name]
        kind = authored.kind
        step = PlannedStep(kind, authored.group, MappingProxyType(deepcopy(authored.inputs)), label=authored.label)
        self.steps[variant, name] = step
        wanted = self.wanted_sources.get((variant, name), set())
        role = _ROLES.get(kind)
        if (role is None and kind not in ("analyze", "report", "synthesize")) or (kind == "analyze" and self.operations is None):
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
            can_sweep_checkpoint=kind == "evaluate" and checkpoint_sweep_refusal(
                authored.inputs.get("subject"), template.sweep, project_swept=self.project.sweep is not None,
            ) is None,
        )
        sweep = template.sweep
        if sweep is not None:
            axes = tuple(axis for axis, value in sweep.axes.items() if value is not None)
            if axes or any(row is not None for row in sweep.rows.values()):
                self.swept_step_axes[variant, name] = axes
        if kind == "report":
            return self.compile_report_step(variant, name, template, active)
        if kind == "analyze":
            return self.compile_analyze_step(variant, name, template, active)
        if kind == "synthesize":
            if template.inputs:
                self.diagnostics.append(ProjectDiagnostic(
                    ResolutionCode.BAD_REFERENCE, "inputs", "Synthesize reads Model, Task and Objective documents.",
                    name, variant=variant,
                ))
                return ()
            rows = self.expand_rows(variant, name, template, {})
            if isinstance(rows, ProjectDiagnostic):
                self.diagnostics.append(rows)
                return ()
            result = []
            for row in rows:
                key = RunKey(variant, name, row.id)
                self.rows[key] = PlannedRow(
                    key, kind, row.label, MappingProxyType(deepcopy(dict(row.coordinates))),
                    axis_keys=MappingProxyType(dict(row.axis_value_ids)), input_axes=MappingProxyType(dict(row.input_axes)),
                )
                definition = self.select_definition(template, key, row, {})
                lowered = self.lower_row(template, key, row, kind, {}, template.row_basis, definition, None)
                if (run := self.record_run(key, lowered, {}, {}, template.requires)) is not None:
                    result.append(run)
            return tuple(result)
        assert role is not None
        initial = "model" in role.documents and isinstance(
            template.inputs.get(role.name), (InitialModelReference, LoadedModelReference),
        )
        roles = {"subject", "task", "objective"} if initial else {role.name}
        extra_roles = sorted(template.inputs.keys() - roles)
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
        selected = ((), {}, {}) if initial else self.select_input(variant, name, template, role, active)
        if isinstance(selected, ProjectDiagnostic):
            self.diagnostics.append(replace(selected, variant=variant))
            return ()
        members, omissions, subsets = selected
        self.record_subsets(variant, name, subsets)
        by_key = {item.member.key: item for item, _ in members}
        subject = template.inputs.get("subject") if kind == "evaluate" else None
        has_checkpoint_axis = kind == "evaluate" and sweep is not None and any(
            isinstance(axis, CheckpointRuleAxis) for axis in sweep.axes.values()
        )
        paired_step = _paired_checkpoint_step(authored, template.inputs.get(role.name))
        if any(subset.conditional for subset in subsets.values()):
            sweep = sweep or Sweep()
            if not any(isinstance(axis, InputAxis) and axis.input == role.name for axis in sweep.axes.values()):
                axis = role.name if role.name not in sweep.axes else next(
                    f"{role.name}-{index}" for index in range(2, len(sweep.axes) + 3)
                    if f"{role.name}-{index}" not in sweep.axes
                )
                sweep = sweep.model_copy(update={"axes": sweep.axes | {axis: InputAxis(input=role.name)}})
                self.swept_step_axes[variant, name] = tuple(sweep.axes)
        rows = self.expand_rows(variant, name, template, {role.name: members}, sweep=sweep)
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
                        or (subsets and isinstance(self.rows.get(producer_key), PlannedRun))
                    ):
                        # Carry an intentional subset exclusion through inherited checkpoint pairs.
                        self.excluded_checkpoint_rows.add(run_key)
                        continue
            coordinates = MappingProxyType(deepcopy(dict(row.coordinates)))
            self.rows[run_key] = PlannedRow(
                run_key, kind, row.label, coordinates, axis_keys=MappingProxyType(dict(row.axis_value_ids)),
                input_axes=MappingProxyType(dict(row.input_axes)),
                kept_axes=row.kept_axes,
            )
            if initial:
                if has_checkpoint_axis:
                    self.diagnostics.append(ProjectDiagnostic(
                        ResolutionCode.BAD_REFERENCE, "inputs.subject", "Initial models have no checkpoint axis",
                        name, variant=variant,
                    ))
                    break
                lowered, inputs = self.lower_initial_evaluate(template, run_key, row)
                definition = self.select_definition(template, run_key, row, {
                    name: (binding,) for name, binding in inputs.items()
                })
                if definition is not None and lowered.pinned_model is not None:
                    self.selected_definitions[run_key] = replace(definition, pinned_model=lowered.pinned_model)
                if (run := self.record_run(run_key, lowered, inputs, {}, template.requires)) is not None:
                    result.append(run)
                continue
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
            if chosen is not None and self.block_row(run_key, template, row, {role.name: (chosen,)}):
                if kind == "evaluate" and isinstance(
                    problem := self.checkpoint_selector(variant, name, row, chosen, subject, has_checkpoint_axis),
                    ProjectDiagnostic,
                ):
                    self.diagnostics.append(problem)
                continue
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
            if kind == "evaluate" and chosen is not None and isinstance(chosen.member.reference, ProductRef) and (
                chosen.member.reference.role == "checkpoint"
            ):
                reference = chosen.member.reference
                selector = self.checkpoint_selector(variant, name, row, chosen, subject, has_checkpoint_axis)
                if isinstance(selector, ProjectDiagnostic):
                    self.diagnostics.append(selector)
                    continue
                chosen = replace(chosen, member=chosen.member.model_copy(update={
                    "reference": reference.model_copy(update={"select": selector}),
                }))
            inputs = {role.name: chosen} if chosen is not None else {}
            definition_inputs = {role: (binding,) for role, binding in inputs.items()}
            definition = self.select_definition(template, run_key, row, definition_inputs)
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
                        definition = self.project_definition(tracked_template, run_key, row, definition_inputs)
                origin = definition.sources.get("/payload" + problem.path) if definition is not None else None
                if origin is not None:
                    problem = self.locate_authored_problem(problem, origin)
                    lowered = replace(lowered, diagnostics=(problem,))
            if (run := self.record_run(run_key, lowered, inputs, omissions, template.requires)) is not None:
                result.append(run)
        return tuple(result)

    def cannot_run(self, key: RunKey) -> bool:
        """Whether a planned row cannot run: it has no request, an admission problem, or a producer that cannot run."""
        row = self.rows.get(key)
        return (
            key in self.waiting
            or not isinstance(row, (PlannedRun, PlannedAnalyzeRun))
            or bool(row.problems)
        )

    def checkpoint_selector(
        self, variant: str, name: str, row: ExpandedRow, chosen: PlannedInput, subject: InputBinding | None,
        has_checkpoint_axis: bool,
    ) -> ProductSelector | None | ProjectDiagnostic:
        """The selector an Evaluate row's checkpoint rule makes, or why the rule does not parse or asks for an update
        outside its producer's run. Both need only the rule and the producer's extent, so a blocked row has them too."""
        path = f"{_selection_path(subject, 'subject', chosen.selection)}.checkpoint"
        origin = row.patch_origins.get(CHECKPOINT_RULE_TARGET)
        authored_path = None
        if origin is not None:
            match origin:
                case ("sweep_value", axis, value):
                    authored_path = pointer(("sweep", "axes", axis, "values", value, "patches", CHECKPOINT_RULE_TARGET, "value"))
                case ("exception", exception):
                    authored_path = pointer(("sweep", "exceptions", exception, "patches", CHECKPOINT_RULE_TARGET, "value"))
                case ("row", row_id):
                    authored_path = pointer(("sweep", "rows", row_id, "patches", CHECKPOINT_RULE_TARGET, "value"))
        author = row.patch_authors.get(CHECKPOINT_RULE_TARGET, name)

        def problem(detail: str, span: TextSpan | None = None) -> ProjectDiagnostic:
            return ProjectDiagnostic(
                ResolutionCode.BAD_REFERENCE, path, detail, name, row=row.id, variant=variant, span=span,
                authored_file=f"steps/{author}.json",
                authored_path=authored_path or pointer(tuple(path.split("."))),
            )

        try:
            write = row.patches.get(CHECKPOINT_RULE_TARGET)
            chosen_subject = chosen.selection if chosen.selection is not None else subject
            selected_subject = checkpoint_rule_write(chosen_subject, write) if write is not None else chosen_subject
            rule = (
                selected_subject.checkpoint if isinstance(selected_subject, StepReference)
                else "last" if isinstance(selected_subject, RunReference) else None
            )
            if rule is None and has_checkpoint_axis:
                raise ValueError("The selected row has no checkpoint rule")
            if isinstance(rule, UnresolvedValue):
                return problem("Expected a checkpoint rule.", TextSpan(start=0, end=len(rule.text)))
            horizon = None
            if isinstance(rule, UpdateCadence) and chosen.producer is not None:
                facts = self.rows[chosen.producer].train
                horizon = facts.target_total if facts is not None else None
            selector = _checkpoint(rule, horizon=horizon)
        except (OSError, ValueError) as error:
            return problem(str(error))
        facts = self.rows[chosen.producer].train if chosen.producer is not None else None
        if facts is not None and facts.origin is not None and facts.target_total is not None and (
            outside := _update_outside_run(rule, facts.origin, facts.target_total)
        ) is not None:
            assert chosen.producer is not None
            return problem(
                f"Update {outside} is outside {chosen.producer.step!r}, "
                f"which runs updates {facts.origin + 1} to {facts.target_total}.",
            )
        return selector

    def block_row(
        self, key: RunKey, template: StepComposition, row: ExpandedRow,
        inputs: Mapping[str, tuple[PlannedInput, ...]],
    ) -> bool:
        """Give ``key`` one ``blocked_input`` when a producer of its members cannot run, and report whether ``key``
        must stay unlowered because that producer has no request, or has one whose documents do not pair.

        A row left unlowered is a ``BlockedRow`` that keeps its own definition and documents, so it stays authorable.
        A row whose producer has a request is lowered as usual and keeps every problem of its own."""
        waiting = next((
            (role, binding, producer) for role, bindings in inputs.items() for binding in bindings
            if (producer := binding.waits_on) is not None
        ), None) or next((
            (role, binding, producer) for role, bindings in inputs.items() for binding in bindings
            for producer in binding.sources if self.cannot_run(producer)
        ), None)
        if waiting is None:
            return False
        role, binding, producer = waiting
        self.waiting[key] = producer
        self.diagnostics.append(ProjectDiagnostic(
            ResolutionCode.BLOCKED_INPUT, _selection_path(template.inputs.get(role), role, binding.selection),
            f"{producer.variant}/{producer.step}/{producer.row} cannot run",
            key.step, row=key.row, variant=key.variant, blocked_by=producer,
        ))
        source = self.model_sources.get(producer)
        if binding.waits_on is None and not (source is not None and source.unpaired):
            return False
        definition = self.select_definition(template, key, row, inputs)
        resolved = definition if definition is not None else self.resolve_row(template, key, row, None)
        planned = self.rows[key]
        self.rows[key] = BlockedRow(
            planned.key, planned.kind, planned.label, planned.coordinates, MappingProxyType(dict(inputs)), producer,
            axis_keys=planned.axis_keys, input_axes=planned.input_axes, kept_axes=planned.kept_axes,
            resolved=not isinstance(resolved, ProjectDiagnostic),
            **({} if isinstance(resolved, ProjectDiagnostic) else {
                "document_ids": MappingProxyType(dict(resolved.document_ids)),
                "document_uses": resolved.document_uses,
            }),
        )
        return True

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
        inputs: Mapping[str, tuple[PlannedInput, ...]],
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
        followed = lowered.followed_documents
        evaluated = None
        if row.kind == "evaluate" and (subject := inputs.get("subject")) is not None and subject.producer is not None:
            own = lowered.document_ids
            producer = self.rows[subject.producer]
            upstream = producer.document_ids
            if producer.kind == "synthesize":
                upstream = {f"payload.trial_documents.{name}": document for name in ("model", "task")
                            if (document := upstream.get(f"payload.{name}")) is not None}
            if (named := own.get(EVALUATION_MODEL)) is not None and (model := upstream.get(EVALUATION_MODEL)) is not None:
                evaluated = named, model
            slots = {EVALUATION_MODEL: (EVALUATION_MODEL,), EVALUATION_TASK: (EVALUATION_TASK, "payload.trial_documents.python_task")}
            inherited = {
                path: upstream[path] for slot, paths in slots.items() if slot not in own
                for path in paths if path in upstream
            }
            followed = frozenset(inherited.values()) - lowered.document_uses
            lowered = replace(
                lowered, document_ids=MappingProxyType({**own, **inherited}),
                document_uses=lowered.document_uses | followed,
            )
        self.diagnostics.extend(
            replace(problem, document=lowered.document_ids.get("payload"))
            if row.kind == "report" and problem.document is None else problem
            for problem in lowered.diagnostics
        )
        if lowered.model_source is not None and (definition := self.selected_definitions.get(key)):
            if MODEL_ATTACHMENTS in definition.payload:
                self.selected_definitions[key] = _producer_model_definition(definition, lowered.model_source)
        if (encoded := lowered.request) is None:
            self.rows[key] = replace(
                row, document_ids=lowered.document_ids, document_uses=lowered.document_uses,
                followed_documents=followed, evaluated_model=evaluated, train=lowered.train,
                resolved=lowered.resolved, problems=lowered.problems,
            )
            return None
        self.requests[encoded.id] = encoded.data
        self.documents.update(encoded.documents)
        if lowered.model_source is not None:
            self.model_sources[key] = replace(lowered.model_source, trial=lowered.train)
        run = PlannedRun(
            key, row.kind, row.label, row.coordinates, encoded.id,
            MappingProxyType(dict(inputs)), MappingProxyType(dict(omissions)), requires,
            axis_keys=row.axis_keys,
            input_axes=row.input_axes,
            kept_axes=row.kept_axes,
            problems=lowered.problems,
            document_ids=lowered.document_ids,
            document_uses=lowered.document_uses,
            followed_documents=followed,
            evaluated_model=evaluated,
            train=lowered.train,
            dataset_port=lowered.dataset_port,
            metric_names=lowered.metric_names,
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
            members, _, subsets = result
            selected[role.name] = members
            self.record_subsets(variant, name, subsets)
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
            self.rows[key] = PlannedRow(
                key, "report", row.label, coordinates, axis_keys=MappingProxyType(dict(row.axis_value_ids)),
                input_axes=MappingProxyType(dict(row.input_axes)),
                kept_axes=row.kept_axes,
            )
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
                if self.block_row(key, template, row, {role: (binding,) for role, binding in inputs.items()}):
                    continue
                definition = self.select_definition(
                    template, key, row, {role: (binding,) for role, binding in inputs.items()},
                )
                lowered = self.lower_row(template, key, row, "report", inputs, basis, definition)
                if (run := self.record_run(key, lowered, inputs, {}, template.requires)) is not None:
                    runs.append(run)
        return tuple(runs)

    def compile_analyze_step(
        self, variant: str, name: str, template: StepComposition, active: tuple[tuple[str, str], ...],
    ) -> tuple[PlannedAnalyzeRun, ...]:
        def problem(path: str, detail: str, row: str | None = None) -> None:
            self.diagnostics.append(ProjectDiagnostic(
                ResolutionCode.BAD_REFERENCE, path, detail, name, variant=variant, row=row,
            ))

        try:
            authored = template.bind(self.variants[variant]).authored_payload()
            declarations = _ANALYZE_INPUTS.validate_python(authored.get("inputs", {}))
        except ValueError as error:
            problem("payload.inputs", str(error))
            return ()
        for unknown in sorted(template.inputs.keys() - declarations.keys()):
            problem(f"inputs.{unknown}", "The Analyze document does not declare this role.")
        selected = {}
        document_refs = {}
        for name_of_role, declaration in declarations.items():
            reference = template.inputs.get(name_of_role)
            if reference is None or isinstance(reference, OmittedReference):
                continue
            path = f"inputs.{name_of_role}"
            if isinstance(reference, (InitialModelReference, LoadedModelReference, TaskDocumentReference)):
                expected = AnalyzeTaskInput if isinstance(reference, TaskDocumentReference) else AnalyzeModelInput
                if not isinstance(declaration, expected):
                    problem(path, f"A document reference cannot supply a {declaration.kind} role of this kind.")
                    continue
                document_refs[name_of_role] = reference
                continue
            if isinstance(declaration, AnalyzeTaskInput):
                problem(path, "A Task role requires a Task document reference.")
                continue
            if isinstance(declaration, AnalyzeModelInput):
                if not isinstance(reference, StepReference) or (
                    reference.checkpoint not in (None, "last") and not isinstance(reference.checkpoint, BatchCheckpoint)
                ):
                    problem(path, "Select an initial Model document or a Train row's terminal or exact-update checkpoint.")
                    continue
                if isinstance(reference.checkpoint, BatchCheckpoint) and not (
                    0 < reference.checkpoint.at <= MAX_CHECKPOINT_COORDINATE
                ):
                    problem(f"{path}.checkpoint", "Select a positive completed Train update.")
                    continue
            role = _analyze_role(name_of_role, declaration)
            assert role is not None
            resolved = self.select_input(variant, name, template, role, active)
            if isinstance(resolved, ProjectDiagnostic):
                self.diagnostics.append(replace(resolved, variant=variant))
                if resolved.selection_value is not None:
                    return ()
                continue
            members, _, subsets = resolved
            if isinstance(declaration, AnalyzeModelInput) and reference.checkpoint is None:
                members = tuple((replace(binding, member=binding.member.model_copy(update={
                    "reference": binding.member.reference.model_copy(update={"select": LastMember()}),
                })), run) for binding, run in members)
            selected[name_of_role] = members
            self.record_subsets(variant, name, subsets)
        rows = self.expand_rows(variant, name, template, selected)
        if isinstance(rows, ProjectDiagnostic):
            self.diagnostics.append(rows)
            return ()
        runs = []
        for row in rows:
            key = RunKey(variant, name, row.id)
            coordinates = MappingProxyType(deepcopy(dict(row.coordinates)))
            self.rows[key] = PlannedRow(
                key, "analyze", row.label, coordinates, axis_keys=MappingProxyType(dict(row.axis_value_ids)),
                input_axes=MappingProxyType(dict(row.input_axes)),
                kept_axes=row.kept_axes,
            )
            inputs = {}
            for role, members in selected.items():
                chosen = row.input_keys.get(role)
                if chosen is not None:
                    inputs[role] = tuple(binding for binding, _ in members if binding.member.key == chosen)
                elif isinstance(declaration := declarations[role], AnalyzeDatasetInput) and declaration.cardinality == "many":
                    inputs[role] = tuple(binding for binding, _ in members)
                elif len(members) == 1:
                    inputs[role] = (members[0][0],)
                else:
                    candidates = ", ".join(binding.member.key for binding, _ in members)
                    problem(f"inputs.{role}", f"Select one member from {candidates}.", row.id)
            if self.block_row(key, template, row, inputs):
                continue
            definition = self.select_definition(template, key, row, inputs)
            resolved = definition if definition is not None else self.resolve_row(template, key, row, None)
            if isinstance(resolved, ProjectDiagnostic):
                self.diagnostics.append(replace(resolved, variant=variant))
                continue
            for role, reference in document_refs.items():
                if role not in resolved.input_documents:
                    continue
                try:
                    source = authored_document_input(
                        reference, resolved.input_documents[role], python_tasks=self.python_tasks,
                    )
                    ids = None
                    if isinstance(reference, LoadedModelReference):
                        ids = self.read_imported_model(variant, reference, resolved.input_documents[role]).replica_ids
                    elif isinstance(reference, InitialModelReference):
                        ids = ("0",)
                    binding = replace(_plan_document_content(source, self.documents), model_replicas=ids)
                except (OSError, ValueError) as error:
                    problem(f"inputs.{role}", str(error), row.id)
                    continue
                inputs[role] = (binding,)
            runs.append(self.lower_analyze_row(template, key, inputs, resolved))
        return tuple(runs)

    def lower_analyze_row(
        self,
        template: StepComposition,
        key: RunKey,
        inputs: Mapping[str, tuple[PlannedInput, ...]],
        resolved: SelectedRunDefinition | ResolvedStep,
    ) -> PlannedAnalyzeRun:
        from feedbax.planner.project_analyze import DatasetStepSource, lower_analyze

        assert self.operations is not None
        datasets: dict[RunKey, DatasetStepSource] = {}
        for sources in inputs.values():
            for source in sources:
                reference = source.member.reference
                if source.producer is None or source.producer in datasets or not isinstance(reference, ProductRef):
                    continue
                if reference.request_id.partition(":")[0] not in {"evaluate", "synthesize"}:
                    continue
                spec = request_identity.decode_request(self.requests[reference.request_id], self.documents.__getitem__).spec
                row = cast(PlannedRun, self.rows[source.producer])
                if isinstance(spec, SynthesisRequestSpec):
                    assert self.synthesis_methods is not None
                    ports = self.synthesis_methods.resolve(spec.method).declaration.outputs
                else:
                    ports = {"states": row.dataset_port} if row.dataset_port is not None else {}
                datasets[source.producer] = DatasetStepSource(
                    spec, LabelName(row.key.row, row.label, kind="row"), ports,
                    member_name(self.variants[row.key.variant].steps, row.key.step),
                )
        elements, encoded, problems, figures, dataset_inputs, dataset_outputs = lower_analyze(
            key,
            resolved.payload,
            resolved.document_ids,
            resolved.document_pointers,
            self.variants[key.variant].documents,
            tuple(replace(problem, row=key.row, variant=key.variant) for problem in resolved.document_problems),
            inputs,
            datasets,
            {
                producer: self.rows[producer.run].elements[producer]
                for sources in inputs.values() for source in sources
                if (producer := source.element) is not None
            },
            self.operations,
            figure_templates=self.figure_templates,
        )
        self.diagnostics.extend(problems)
        for request in encoded:
            self.requests[request.id] = request.data
            self.documents.update(request.documents)
        original = self.rows[key]
        run = PlannedAnalyzeRun(
            key, original.kind, original.label, original.coordinates,
            MappingProxyType(elements), template.requires, MappingProxyType(dict(inputs)), MappingProxyType(figures),
            MappingProxyType(dataset_inputs),
            MappingProxyType({name: MappingProxyType(ports) for name, ports in dataset_outputs.items()}),
            axis_keys=original.axis_keys,
            input_axes=original.input_axes,
            kept_axes=original.kept_axes,
            document_ids=MappingProxyType(dict(resolved.document_ids)),
            document_uses=resolved.document_uses,
        )
        self.rows[key] = run
        return run

    def rank_input(
        self, name: str, template: StepComposition, role: _Role, members: _Members,
        subset: PlannedSubset, reference: StepReference, at: str,
    ) -> tuple[_Members, dict[str, str], PlannedSubset] | ProjectDiagnostic:
        path = f"{at}.where"
        if role.name != "subject":
            reason = "Best currently binds saved Evaluate subjects."
        elif reference.checkpoint not in (None, "last") or (
            template.sweep is not None and any(isinstance(axis, CheckpointRuleAxis) for axis in template.sweep.axes.values())
        ):
            reason = "Best requires the selected Train's last checkpoint."
        else:
            return self.rank_groups(name, template, role, members, subset, reference, f"{path}.rule")
        return ProjectDiagnostic(ResolutionCode.BAD_REFERENCE, f"{path}.rule", reason, name)

    def rank_groups(
        self, name: str, template: StepComposition, role: _Role, members: _Members, subset: PlannedSubset,
        reference: StepReference, path: str,
    ) -> tuple[_Members, dict[str, str], PlannedSubset] | ProjectDiagnostic:
        best = subset.best
        assert best is not None
        written = subset.terms.rule[best.span.start:best.span.end]
        source = LabelName(reference.step, self.variants[members[0][1].key.variant].steps[reference.step].label) \
            if members else None

        def refusal(detail: str, field: str = path) -> ProjectDiagnostic:
            return ProjectDiagnostic(ResolutionCode.BAD_REFERENCE, field, detail, name, span=best.span)

        axes = tuple(sorted(best.per))
        if len(set(axes)) != len(axes):
            return refusal(f"{written}: per names an axis twice.")
        if not members:
            return refusal("Best requires at least one candidate.")
        groups: dict[tuple[str, ...], list[tuple[PlannedInput, PlannedRow]]] = {}
        for item, producer in members:
            assert producer is not None
            for axis in axes:
                if axis not in producer.coordinates:
                    return refusal(f"{written}: {axis} is missing on {producer.key.variant}/{producer.key.row}.")
            group = tuple(producer.coordinates[axis].value_id for axis in axes)
            grouped = groups.setdefault(group, [])
            if grouped and any(
                not same_typed_value(producer.coordinates[axis].value, grouped[0][1].coordinates[axis].value)
                for axis in axes
            ):
                return refusal(f"{written}: group {dict(zip(axes, group, strict=True))!r} has conflicting axis values.")
            grouped.append((item, producer))
        if best.count * len(groups) > 1 and not (
            template.sweep is not None and any(
                isinstance(axis, InputAxis) and axis.input == role.name for axis in template.sweep.axes.values()
            )
        ):
            return refusal(f"{written}: more than one rank requires a subject input axis.")

        ranked = []
        planned_groups = []
        for group, grouped in groups.items():
            coordinates = dict(zip(axes, group, strict=True))
            if best.count > len(grouped):
                target = f" in group {coordinates!r}" if coordinates else ""
                return refusal(f"{written}: asks for {best.count} of {len(grouped)} runs{target}.")
            candidates = []
            keys = []
            waiting = next((producer.key for _, producer in grouped if self.cannot_run(producer.key)), None)
            for item, producer in grouped:
                keys.append(producer.key)
                if waiting is None:
                    assert isinstance(producer, PlannedRun)
                    candidates.append(BestCandidate(
                        key=item.member.key, variant=producer.key.variant, row=producer.key.row,
                        reference=ProductRef(request_id=producer.request_id, role=role.product_role, select=LastMember()),
                    ))
            ranks = []
            for position in range(1, best.count + 1):
                member: InputMember | PendingMember = (
                    InputMember(key=best_rank_key(position, coordinates), reference=BestRank(
                        metric=best.metric, order=best.order, count=best.count, rank=position, candidates=tuple(candidates),
                        group=coordinates,
                    )) if waiting is None else PendingMember(
                        key=best_rank_key(position, coordinates),
                        reference=PendingProduct(variant=waiting.variant, step=waiting.step, row=waiting.row),
                    )
                )
                ranks.append(member.key)
                ranked.append((PlannedInput(
                    None, member, candidates=tuple(keys), selection=reference,
                    named=MappingProxyType({"source": source}) if source is not None else MappingProxyType({}),
                    best_path=path,
                ), grouped[0][1]))
            planned_groups.append(PlannedRankGroup(MappingProxyType(coordinates), tuple(ranks)))
        return tuple(ranked), {}, replace(subset, rank_groups=tuple(planned_groups))

    def subset_origin(
        self, variant: str, step: str, at: str, document: str | None, term: Literal["rule", "rows"],
    ) -> ValueSource:
        """Where the subset terms of the selection at ``at`` (``inputs.<role>`` or ``inputs.<role>.<i>``) are authored."""
        if document is not None:
            authored = ("documents", document, "payload", term)
            source = ValueSource("document", document, pointer((term,)))
        else:
            steps = self.variants[variant].steps
            _, role, *item = at.split(".")
            while role not in steps[step].inputs and isinstance(base := steps[step].base, StepBase):
                step = base.step
            path = ("inputs", role, *item, "where", term)
            authored = ("steps", step, *path)
            source = ValueSource("step", step, pointer(path), step=step)
        for target in reversed(self.variant_rows[variant].patches):
            parts = tuple(target.split("."))
            shared = min(len(authored), len(parts))
            if authored[:shared] == parts[:shared]:
                return ValueSource("variant", variant, pointer(parts), variant=variant)
        return source

    def read_terminal_train(self, run: str) -> TerminalTrainSource | None:
        if (terminal := self.terminal_sources.get(run)) is not None:
            return terminal
        reader = self.readers.terminal_train
        if reader is None or (terminal := reader(run)) is None:
            return None
        return self.terminal_sources.setdefault(terminal.request_id, terminal)

    def subject_model(self, run: PlannedRow) -> tuple[RunKey | None, LabelName] | None:
        """What an Evaluate row's ``model`` names: its subject's Train step or stored run, or its untrained Model."""
        if run.kind != "evaluate":
            return None
        subject = (
            run.inputs.get("subject") if isinstance(run, PlannedRun)
            else next(iter(run.inputs.get("subject", ())), None) if isinstance(run, BlockedRow) else None
        )
        if subject is not None and subject.producer is not None:
            producer = subject.producer
            return producer, LabelName(producer.step, self.variants[producer.variant].steps[producer.step].label)
        if subject is not None and "source" in subject.named:
            return None, subject.named["source"]
        if subject is not None and isinstance(trained := subject.member.reference, ProductRef):
            return None, LabelName(trained.request_id, kind="run")
        if (
            subject is not None and isinstance(reference := subject.member.reference, ContentRef)
            and reference.media_type == IMPORTED_MODEL_MEDIA_TYPE
            and (document := run.document_ids.get("inputs.subject")) is not None
        ):
            return None, LabelName(document, self.variants[run.key.variant].documents[document].label, kind="document")
        if (document := run.document_ids.get(EVALUATION_MODEL)) is not None:
            return None, LabelName(document, self.variants[run.key.variant].documents[document].label, kind="document")
        return None

    def member_coordinates(self, run: PlannedRow) -> dict[str, SweepCoordinate]:
        """What a gathered row is about: its coordinates, and for an Evaluate row the ``model`` it evaluates."""
        coordinates = dict(run.coordinates)
        if (model := self.subject_model(run)) is None:
            return coordinates
        producer, name = model
        trained = self.rows[producer].coordinates if producer is not None else {}
        for label, coordinate in {"model": SweepCoordinate(name.key, name.label), **trained}.items():
            prior = coordinates.setdefault(label, coordinate)
            if prior.value_id != coordinate.value_id:
                project = self.variants[run.key.variant]
                step_name = member_name(project.steps, run.key.step)
                subject_name = member_name(project.documents, name.key) if name.kind == "document" else name.printed
                raise ValueError(
                    f"{step_name}/{run.key.row} and its subject {subject_name!r} "
                    f"give label {label!r} different values"
                )
        return coordinates

    def member_labels(
        self, run: PlannedRow, coordinates: Mapping[str, SweepCoordinate], *, gathers_variants: bool,
    ) -> tuple[dict[str, MemberLabel], dict[str, LabelName]]:
        """The labels a stacked dataset carries: the content labels, and the step-valued ones by what they name."""
        named = {"source": LabelName(run.key.step, self.variants[run.key.variant].steps[run.key.step].label)}
        if (model := self.subject_model(run)) is not None:
            named["model"] = model[1]
        labels = {"variant": MemberLabel(value_id=run.key.variant)} if gathers_variants else {}
        for label, coordinate in coordinates.items():
            if label not in named:
                labels[label] = MemberLabel(value_id=coordinate.value_id, value=coordinate.value)
        return labels, named

    def select_input(
        self,
        variant: str,
        name: str,
        template: StepComposition,
        role: _Role,
        active: tuple[tuple[str, str], ...],
    ) -> tuple[_Members, dict[str, str], dict[str, PlannedSubset]] | ProjectDiagnostic:
        """The role's members and omissions, and the subset of each selection by its input ID."""
        reference = template.inputs.get(role.name)
        if isinstance(reference, list):
            return self.select_list(variant, name, template, role, active, reference)
        selected = self.select_reference(variant, name, template, role, active, reference, f"inputs.{role.name}")
        if isinstance(selected, ProjectDiagnostic):
            return selected
        members, omissions, subset = selected
        subsets = {role.name: subset} if subset is not None else {}
        return _identified(members, labelled=_labelled(role, members))[0], omissions, subsets

    def record_subsets(self, variant: str, step: str, subsets: Mapping[str, PlannedSubset]) -> None:
        for input_id, subset in subsets.items():
            self.subsets[variant, step, input_id] = subset

    def select_list(
        self,
        variant: str,
        name: str,
        template: StepComposition,
        role: _Role,
        active: tuple[tuple[str, str], ...],
        references: list[StepReference],
    ) -> tuple[_Members, dict[str, str], dict[str, PlannedSubset]] | ProjectDiagnostic:
        """Gather the members of several selections.

        Each selection is named by its source: its step's key, or the key of the operation output it reads. A second
        source of one name, or a second measured selection of one step, takes the next suffix (``train-2``). A ranked
        member is qualified by its selection's name (``train/rank-1``), and a key two selections share is qualified the
        same way, so no edit of one selection renames another's members.
        """
        path = f"inputs.{role.name}"
        if not role.gathers:
            return ProjectDiagnostic(ResolutionCode.BAD_REFERENCE, path, "this input takes one selection", name)
        if not references:
            return ProjectDiagnostic(ResolutionCode.BAD_REFERENCE, path, "list at least one selection", name)
        selections = []
        subsets = {}
        owners: dict[str, list[object]] = {}
        qualifiers = {}
        for index, reference in enumerate(references):
            item = f"{path}.{index}"
            selected = self.select_reference(variant, name, template, role, active, reference, item)
            if isinstance(selected, ProjectDiagnostic):
                return selected
            members, _, subset = selected
            source = reference.element or reference.step
            card = next((binding.producer.step for binding, _ in members if binding.producer is not None), reference.step)
            owner = index if subset is not None and subset.ranks else card
            named = owners.setdefault(source, [])
            if owner not in named:
                named.append(owner)
            position = named.index(owner)
            qualifier = source if position == 0 else f"{source}-{position + 1}"
            qualifiers.update({id(binding): qualifier for binding, _ in members})
            if subset is not None and subset.ranks:
                keys = {key: f"{qualifier}/{key}" for key in subset.ranks}
                members = tuple(
                    (replace(binding, member=binding.member.model_copy(update={"key": keys[binding.member.key]})), run)
                    for binding, run in members
                )
                qualifiers.update({id(binding): qualifier for binding, _ in members})
                subset = replace(subset, rank_groups=tuple(
                    replace(group, ranks=tuple(keys[key] for key in group.ranks)) for group in subset.rank_groups
                ))
            selections.extend(members)
            if subset is not None:
                subsets[item.removeprefix("inputs.")] = subset
        counts = Counter(binding.member.key for binding, _ in selections)
        qualified = []
        for binding, run in selections:
            key = binding.member.key
            if counts[key] > 1 and not isinstance(binding.member.reference, BestRank):
                key = f"{qualifiers[id(binding)]}/{key}"
            qualified.append((replace(binding, member=binding.member.model_copy(update={"key": key})), run))
        gathered, _ = _identified(tuple(qualified), labelled=_labelled(role, tuple(qualified)))
        keys = [binding.member.key for binding, _ in gathered]
        if len(set(keys)) != len(keys):
            repeated = sorted(key for key, count in Counter(keys).items() if count > 1)
            return ProjectDiagnostic(
                ResolutionCode.BAD_REFERENCE, path, f"selections repeat the members {repeated!r}", name,
            )
        return tuple(gathered), {}, subsets

    def select_reference(
        self,
        variant: str,
        name: str,
        template: StepComposition,
        role: _Role,
        active: tuple[tuple[str, str], ...],
        reference: Reference | None,
        path: str,
    ) -> (
        tuple[_Members, dict[str, str], PlannedSubset | None]
        | ProjectDiagnostic
    ):
        if reference is None or isinstance(reference, OmittedReference):
            if role.required:
                return ProjectDiagnostic(
                    ResolutionCode.BAD_REFERENCE, path, "required input is missing", name
                )
            return (), {role.name: reference.not_applicable} if reference else {}, None
        if isinstance(reference, RunReference) and role.name == "subject":
            if template.sweep is not None and any(
                isinstance(axis, CheckpointRuleAxis) for axis in template.sweep.axes.values()
            ):
                return ProjectDiagnostic(
                    ResolutionCode.BAD_REFERENCE, f"{path}.checkpoint",
                    "An existing Train run supplies only its terminal model; checkpoint sweeps need a Train step.",
                    name,
                )
            try:
                terminal = self.read_terminal_train(reference.run)
                if terminal is None:
                    raise ValueError(f"No successful Train result for run {reference.run!r} in this project Store.")
                member = InputMember(
                    key=DEFAULT_SWEEP_ROW,
                    reference=ProductRef(request_id=terminal.request_id, role=role.product_role, select=LastMember()),
                )
            except (OSError, ValueError) as error:
                return ProjectDiagnostic(ResolutionCode.BAD_REFERENCE, path, str(error), name)
            return ((PlannedInput(None, member, terminal=terminal.subject), None),), {}, None
        if isinstance(reference, LoadedModelReference) and role.name == "checkpoint" and role.producer_kind == "train":
            if reference.source.unresolved or reference.source.pointer:
                return ProjectDiagnostic(
                    ResolutionCode.BAD_REFERENCE, path, "Choose a Loaded Model document.", name,
                    missing_subject=reference.source.document if reference.source.unresolved else None,
                )
            document = compose_document(self.variants[variant].documents.get, reference.source.document)
            if isinstance(document, ProjectDiagnostic):
                return replace(document, path=path, step=name)
            try:
                source = authored_document_input(reference, document)
                assert isinstance(source, ImportedModelSource)
                self.read_imported_model(variant, reference, document)
            except (OSError, ValueError) as error:
                return ProjectDiagnostic(ResolutionCode.BAD_REFERENCE, path, str(error), name)
            return ((_plan_document_content(source, self.documents), None),), {}, None
        if not isinstance(reference, StepReference):
            return ProjectDiagnostic(
                ResolutionCode.UNSUPPORTED_PLAN,
                path,
                "a same-project step or row reference is required",
                name,
            )
        if reference.unresolved:
            target = reference.element or reference.step
            return ProjectDiagnostic(
                ResolutionCode.BAD_REFERENCE, path, "The input reference is unresolved.",
                name, missing_subject=target,
            )
        synthesis_dataset = role.operation_outputs and reference.element is None and reference.product is not None
        if role.operation_outputs and reference.product is not None:
            role = replace(role, producer_kind="synthesize" if synthesis_dataset else "analyze", product_role=reference.product)
        element_input = role.producer_kind == "analyze"
        if element_input:
            if reference.element is None or reference.product != role.product_role:
                return ProjectDiagnostic(
                    ResolutionCode.BAD_REFERENCE, path,
                    f"declare a render element and its {role.product_role!r} product", name,
                )
        elif not synthesis_dataset and (reference.element is not None or reference.product is not None):
            return ProjectDiagnostic(
                ResolutionCode.BAD_REFERENCE, path,
                "this input selects a step row without an element or product", name,
            )
        if element_input:
            card = self.holders(variant).card_of(reference.step, reference.element)
            if isinstance(card, AddressProblem):
                return ProjectDiagnostic(
                    ResolutionCode.BAD_REFERENCE, f"{path}.element", card.detail, name,
                    missing_subject=card.missing_subject,
                )
            reference = reference.model_copy(update={"step": card})
        if reference.checkpoint is not None and role.product_role != "checkpoint":
            return ProjectDiagnostic(
                ResolutionCode.BAD_REFERENCE, f"{path}.checkpoint",
                "this product does not take a Train checkpoint selector", name,
            )
        terms = reference.where
        subset_document = None
        if isinstance(terms, NamedSubset):
            subset_document = terms.subset
            if terms.unresolved:
                return ProjectDiagnostic(
                    ResolutionCode.MISSING_DOCUMENT, f"{path}.where", subset_document,
                    name, document=subset_document, missing_subject=subset_document,
                )
            document = compose_document(self.variants[variant].documents.get, subset_document)
            if isinstance(document, ProjectDiagnostic):
                return replace(document, path=f"{path}.where", step=name)
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
        if terms is not None and terms.bound_ranks:
            addressed_variants = tuple(dict.fromkeys(rank.variant for rank in terms.bound_ranks))
            if reference.variants is None:
                variants = addressed_variants
            elif outside := sorted(set(addressed_variants).difference(variants)):
                return ProjectDiagnostic(
                    ResolutionCode.BAD_REFERENCE, f"{path}.where.bound_ranks",
                    f"measured member variants outside reference scope: {outside!r}", name,
                )
        runs: tuple[PlannedRow, ...] = ()
        accepted = {producer.kind: producer.product for producer in role.producers}
        product_roles: set[str] = set()
        for source_variant in variants:
            target = source_variant, reference.step
            if target in active:
                chain = " -> ".join(f"{v}/{member_name(self.variants[v].steps, s)}" for v, s in (*active, target))
                return ProjectDiagnostic(ResolutionCode.STEP_CYCLE, path, chain, name)
            project = self.variants.get(source_variant)
            producer = project.steps.get(reference.step) if project is not None else None
            if producer is None or producer.kind not in accepted:
                label = member_name(project.steps if project is not None else self.project.steps, reference.step)
                return ProjectDiagnostic(
                    ResolutionCode.BAD_REFERENCE,
                    path,
                    f"{source_variant}/{label} must be a {' or '.join(accepted)} step",
                    name, missing_subject=reference.step if producer is None else None,
                )
            product_roles.add(accepted[producer.kind])
            self.step(*target, active)
            runs += self.step_rows(*target)
        if len(product_roles) != 1:
            return ProjectDiagnostic(ResolutionCode.BAD_REFERENCE, path, "The selected steps publish different products.", name)
        product_role, = product_roles
        if product_role == "model" and (
            reference.checkpoint is not None or role.name == "subject" and template.sweep is not None and any(
                isinstance(axis, CheckpointRuleAxis) for axis in template.sweep.axes.values()
            )
        ):
            return ProjectDiagnostic(
                ResolutionCode.BAD_REFERENCE, f"{path}.checkpoint", "Synthesize publishes one model without checkpoints.", name,
            )
        candidate_count = len(runs)

        def member_key(run: PlannedRow) -> str:
            key = run.key
            return producer_member_key(key.variant, key.row, gathers_variants=reference.variants is not None)

        def rule_problem(error: RuleRefusal) -> ProjectDiagnostic:
            problem_path = pointer(tuple(error.path.split(".")))
            return ProjectDiagnostic(
                ResolutionCode.BAD_REFERENCE, f"{path}.where.{error.path}", error.detail, name, span=error.span,
                document=subset_document, variant=variant,
                document_path=problem_path if subset_document is not None else None,
                authored_file=f"documents/{subset_document}.json" if subset_document is not None else f"steps/{name}.json",
                authored_path=f"/payload{problem_path}" if subset_document is not None else pointer(
                    (*path.split("."), "where", *error.path.split(".")),
                ),
                selection_value=error.problem if isinstance(error, SelectedAxisValueRefusal) else None,
            )

        problems = []
        try:
            keyed_runs, best, conditions = select_step_rows(
                terms,
                runs,
                address=lambda run: (run.key.variant, run.key.row),
                member_key=member_key,
                coordinates=lambda run: run.selection_coordinates,
                coordinate_label=sweep_value_name,
                missing=problems.append,
                producer=member_name(self.variants[variants[0]].steps, reference.step),
                final_metrics=common_train_metrics if role.producer_kind == "train" else None,
            )
        except RuleRefusal as error:
            self.diagnostics.extend(map(rule_problem, problems))
            return rule_problem(error)
        if problems and not keyed_runs:
            self.diagnostics.extend(map(rule_problem, problems[1:]))
            return rule_problem(problems[0])
        self.diagnostics.extend(map(rule_problem, problems))
        if (product_role == "model" or synthesis_dataset) and (best is not None or conditions):
            return ProjectDiagnostic(
                ResolutionCode.BAD_REFERENCE, f"{path}.where", "Metric conditions and Best require a Train step.", name,
            )
        expands_input = template.sweep is not None and any(
            isinstance(axis, InputAxis) and axis.input == role.name
            for axis in template.sweep.axes.values()
        )
        if not keyed_runs and not expands_input:
            return ProjectDiagnostic(
                ResolutionCode.BAD_REFERENCE,
                path,
                f"No rows of {member_name(self.variants[variants[0]].steps, reference.step)} are selected.",
                name,
            )
        rule = None if role.name == "subject" else reference.checkpoint
        if isinstance(rule, (CheckpointBatches, CheckpointCadence, UpdateCadence, tuple)) and not role.checkpoint_sets:
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
            requests: tuple[tuple[RequestId, PlannedElement | None], ...] = ()
            if isinstance(run, PlannedAnalyzeRun) and (elements := run.requests(reference.element or "")):
                port = DatasetPort if role.operation_outputs else PlotlyFigurePort
                if not all(isinstance(element.outputs.get(role.product_role), port) for element in elements):
                    address = f"{run.key.variant}/{run.key.step}/{run.key.row}/{reference.element}"
                    return ProjectDiagnostic(
                        ResolutionCode.BAD_REFERENCE, f"{path}.element",
                        f"{address} has no planned {role.product_role!r} product", name,
                    )
                requests = tuple((element.request_id, element) for element in elements)
            elif isinstance(run, PlannedRun) and not element_input:
                requests = ((run.request_id, None),)
            try:
                coordinates = self.member_coordinates(run)
                labels, named = self.member_labels(run, coordinates, gathers_variants=reference.variants is not None)
                if role.operation_outputs and element_input:
                    del named["source"]
                    label = requests[0][1].label if requests and requests[0][1] is not None else None
                    named["element"] = LabelName(reference.element, label, kind="element", card=run.key.step)
                elif role.producer_kind != "evaluate" and not synthesis_dataset:
                    labels = {}
            except ValueError as error:
                return ProjectDiagnostic(ResolutionCode.BAD_REFERENCE, path, str(error), name)
            if not requests:
                members.append((
                    PlannedInput(
                        run.key,
                        PendingMember(
                            key=key,
                            reference=PendingProduct(variant=run.key.variant, step=run.key.step, row=run.key.row),
                            labels=labels,
                        ),
                        producer_element=reference.element,
                        coordinates=MappingProxyType(coordinates),
                        selection=reference,
                        named=MappingProxyType(named),
                    ),
                    run,
                ))
                continue
            for request, element in requests:
                product = ProductRef(request_id=request, role=product_role, select=selector)
                if conditions:
                    assert terms is not None
                    product = ConditionalProduct(
                        request_id=product.request_id, role=product.role, select=product.select,
                        conditions=bind_metric_conditions(
                            terms.rule, conditions, run.selection_coordinates,
                        ),
                    )
                mapped = element.stack_key if element is not None and element.key.member is not None else None
                members.append((
                    PlannedInput(
                        run.key,
                        InputMember(
                            key=key if mapped is None else mapped if len(keyed_runs) == 1 else f"{key}/{mapped}",
                            reference=product,
                            labels=labels | (dict(element.mapped) if element is not None else {}),
                        ),
                        producer_element=reference.element,
                        producer_member=mapped,
                        coordinates=MappingProxyType(coordinates),
                        selection=reference,
                        named=MappingProxyType(named),
                    ),
                    run,
                ))
        source = None
        if terms is not None and role.name in {"subject", "states"}:
            if best is not None and role.name == "subject":
                source = SubsetSource(
                    self.subset_origin(variant, name, path, subset_document, "rule"), reference.step,
                )
            elif terms.bound_ranks is not None:
                source = SubsetSource(
                    self.subset_origin(variant, name, path, subset_document, "bound_ranks"), reference.step,
                    tuple(BoundRank(key=key, variant=run.key.variant, row=run.key.row) for key, run in keyed_runs),
                )
        subset = (
            PlannedSubset(
                deepcopy(terms), candidate_count,
                tuple(run.key for _, run in keyed_runs), subset_document,
                source=source,
                best=best, metric_conditions=MappingProxyType({
                    run.key: product for member, run in members
                    if isinstance((product := member.member.reference), ConditionalProduct)
                }) if conditions else None,
            )
            if terms is not None
            else None
        )
        if subset is not None and best is not None:
            return self.rank_input(name, template, role, tuple(members), subset, reference, path)
        return tuple(members), {}, subset


def project_variants(
    project: Project,
    *,
    components: ComponentRegistry | None = None,
    functions: FunctionRegistry | None = None,
    python_tasks: PythonTaskRegistry | None = None,
    training_methods: TrainingMethodRegistry | None = None,
) -> dict[str, Project]:
    """Resolve authored variants without planning their runs or reading Store artifacts."""
    if project.sweep is None:
        return {DEFAULT_SWEEP_ROW: project}
    if python_tasks is None:
        python_tasks = application_python_tasks()
    if training_methods is None and any(step.kind == "train" for step in project.steps.values()):
        training_methods = load_application().bundle.training_methods
    compiler = _Compiler(
        project, BaseReaders(), (), components=components, functions=functions,
        python_tasks=python_tasks, training_methods=training_methods,
    )
    compiler.expand_variants()
    return compiler.variants


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
    training_methods: TrainingMethodRegistry | None = None,
    figure_templates: FigureTemplateRegistry | None = None,
    synthesis_methods: SynthesisMethodRegistry | None = None,
    capture_sources: bool = False,
) -> ProjectPlan:
    """Plan static requests and value sources with the supplied declaration registries."""
    if python_tasks is None:
        python_tasks = application_python_tasks()
    if training_methods is None and any(step.kind == "train" for step in project.steps.values()):

        training_methods = load_application().bundle.training_methods
    if synthesis_methods is None and any(step.kind == "synthesize" for step in project.steps.values()):
        synthesis_methods = load_application().bundle.synthesis_methods
    readers = replace(
        readers,
        external=cache(readers.external) if readers.external is not None else None,
        figure_external=cache(readers.figure_external) if readers.figure_external is not None else None,
        terminal_train=cache(readers.terminal_train) if readers.terminal_train is not None else None,
        imported_model=cache(readers.imported_model) if readers.imported_model is not None else None,
    )
    if capture_sources or memo is None or any(
        reader is not None for reader in (readers.run, readers.external, readers.row, readers.figure_external)
    ):
        return _Compiler(
            project, readers, source_runs, operations=operations, components=components,
            functions=functions, python_tasks=python_tasks, figure_templates=figure_templates, training_methods=training_methods,
            synthesis_methods=synthesis_methods, capture_sources=capture_sources,
        ).plan()
    authority = compiler_authority(
        components, operations, functions, python_tasks, training_methods, figure_templates, synthesis_methods,
    )
    ticket, previous = memo._begin(authority)
    compiler = _Compiler(
        project, readers, source_runs, previous, operations, components, functions, python_tasks=python_tasks,
        training_methods=training_methods, figure_templates=figure_templates, synthesis_methods=synthesis_methods,
    )
    plan = compiler.plan()
    if authority == compiler_authority(
        components, operations, functions, python_tasks, training_methods, figure_templates, synthesis_methods,
    ):
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
        inputs = {request.key: planned_input_groups(request) for request in requests}
        entries = {
            request.key: PlanRequest(
                request.key,
                request.request_id,
                {
                    role: element
                    for role, bindings in inputs[request.key].items()
                    if len(bindings) == 1 and (element := bindings[0].element) is not None
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
                    role: source for role, bindings in inputs[request.key].items()
                    if (source := (
                        _upstream_source(bindings[0]) if len(bindings) == 1 else
                        _analyze_subset_origin(plan, request.key)
                    )) is not None
                })
            }
            source_index = PlanSourceIndex(plan.source_snapshot, producers)
        return cls(entries, plan.requests, plan.documents, subset_sources, source_index)


def _analyze_subset_origin(plan: ProjectPlan, key: RunKey | ElementKey) -> ValueSource | None:
    row = key.run if isinstance(key, ElementKey) else key
    subset = plan.subsets.get((row.variant, row.step, "states"))
    return subset.source.origin if subset is not None and subset.source is not None else None


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
    path, parts = _authored_request_change(right.key, old, new)
    if parts[:1] == ("inputs",):
        role = parts[1]
        cause = right.element_inputs.get(role)
        if cause != left.element_inputs.get(role):
            cause = None
        row = right.key.run if isinstance(right.key, ElementKey) else right.key
        source_key = row.variant, row.step, role
        old_source = before.subset_sources.get(source_key) if isinstance(right.key, RunKey) else None
        new_source = after.subset_sources.get(source_key) if isinstance(right.key, RunKey) else None
        return ElementChange(
            "changed", path, cause,
            before_source=old_source.origin if old_source is not None else _review_source(before, left.key, parts),
            after_source=new_source.origin if new_source is not None else _review_source(after, right.key, parts),
        )
    return ElementChange(
        "changed", path,
        before_source=_review_source(before, left.key, parts) if parts else None,
        after_source=_review_source(after, right.key, parts) if parts else None,
    )


def authored_request_change(
    key: RunKey | ElementKey, old: request_identity.DecodedRequest, new: request_identity.DecodedRequest,
) -> tuple[str, tuple[str, ...]]:
    """Locate a canonical request change at its authored Plan address, as text and as keys."""
    if old.kind != new.kind:
        path = key.payload_path(new.kind) if isinstance(key, ElementKey) else "payload"
        return path, tuple(path.split("."))
    path, parts = _authored_request_change(key, old, new)
    return path, parts or tuple(path.split("."))


def _authored_request_change(
    key: RunKey | ElementKey, old: request_identity.DecodedRequest, new: request_identity.DecodedRequest,
) -> tuple[str, tuple[str, ...]]:
    change = first_request_change(old, new)
    prefix = ("payload",)
    if isinstance(key, ElementKey):
        prefix += ("operations" if new.kind == "analyze" else "views", key.name)
    base = ".".join(prefix)
    if change.path == "inputs":
        role = next(
            role for role in sorted(old.inputs.keys() | new.inputs.keys())
            if old.inputs.get(role) != new.inputs.get(role)
        )
        input_path = f"inputs.{role}"
        return (f"{base}.{input_path}" if isinstance(key, ElementKey) else input_path), ("inputs", role)
    if not change.parts:
        return f"{base}.{change.path}", ()
    relative = change.parts[1:]
    if relative == ("analysis_type",):
        relative = ("operation",)
    elif new.kind == "render" and relative[:1] == ("document",):
        relative = ("figure", *relative[1:])
    elif new.kind == "report" and relative[:1] == ("document",):
        relative = relative[1:]
    parts = prefix + relative
    return ".".join(parts), parts


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
