"""Compile authored project requests without result access."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import cache
from threading import Lock
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

from pydantic import ValidationError
from pydantic_core import to_json

from feedbax.contracts import request_identity
from feedbax.contracts.base import sha256_bytes
from feedbax.contracts.expression_values import Problem, ValueBase
from feedbax.contracts.figure_document import PLOTLY_FIGURE_ROLE, FigureDocument, PlotlyFigurePort
from feedbax.contracts.intent_diff import first_request_change
from feedbax.contracts.graph import GraphSpec, StudioTaskBindingSpec
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
    StepReference,
    SubsetSelectionError,
    SubsetTerms,
    select_static_rows,
)
from feedbax.contracts.project_resolution import (
    BaseReaders,
    ProjectDiagnostic,
    ResolutionCode,
    ResolvedStep,
    StepComposition,
    StepTemplate,
    compose_step,
    _select,
)
from feedbax.contracts.rollout import ROLLOUT_TYPE, authored_rollout_settings, resolve_trial_ids
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
    TrainingRequestSpec,
)
from feedbax.contracts.studio_training import STUDIO_GRAPH_METHOD, StudioGraphMethodPayload

if TYPE_CHECKING:
    from feedbax.analysis.operations import OperationRegistry
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
class PlannedSubset:
    terms: SubsetTerms
    candidates: int
    members: tuple[RunKey, ...]
    document: str | None


@dataclass(frozen=True)
class PlannedInput:
    producer: RunKey
    member: InputMember
    producer_element: str | None = None

    @property
    def element(self) -> ElementKey | None:
        return ElementKey(self.producer, self.producer_element) if self.producer_element else None


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
    if not isinstance(train, (TrainingRequestSpec, DocumentTrainingRequestSpec)):
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


def _train_model_path(payload: Payload) -> tuple[str, ...]:
    return ("trial_documents", "model") if "trial_documents" in payload else ("graph",)


def _with_train_model(
    payload: Payload, model: _ModelSource, document_ids: Mapping[str, str],
    document_pointers: Mapping[str, str],
) -> Payload:
    data = model.model.model_dump(mode="json")
    draft = deepcopy(payload)
    main = _train_model_path(payload)
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
    prefix = _train_model_path(payload)
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


def _model_difference_problem(key: RunKey, address: ModelAddress | None, detail: str) -> ProjectDiagnostic:
    return ProjectDiagnostic(
        ResolutionCode.INVALID_PAYLOAD, f"payload.{MODEL_ATTACHMENTS}", detail, key.step,
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
    annotation = DocumentTrainingRequestSpec if documents is not None else TrainingRequestSpec
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
    model_document: str | None = None,
    sources: Mapping[str, ValueSource] | None = None,
    components: ComponentRegistry | None = None,
    functions: FunctionRegistry | None = None,
    python_tasks: PythonTaskRegistry | None = None,
    model_source: _ModelSource | None = None,
) -> _LoweredRow:
    checkpoint_problem = None
    document_train = kind == "train" and payload.get("trial_documents") is not None
    if document_train:
        from feedbax.compiler.trial_execution import (
            PreparedTrialExecution, compile_trial_execution, validate_python_task,
        )
        from feedbax.tasks.reaching import builtin_python_tasks
        from feedbax.contracts.request_specs import TrialDocuments

        try:
            authored = TrialDocuments.model_validate(payload["trial_documents"])
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
            paired = compile_trial_execution(
                authored.model_name, authored.model, authored.task_source,
                components or _builtin_model_registry(), functions or _builtin_expression_registry(),
                python_tasks=python_tasks, objective=authored.objective, input_overrides=authored.input_overrides,
            )
        except ValueError as error:
            return _LoweredRow(None, (ProjectDiagnostic(
                ResolutionCode.INVALID_PAYLOAD, "payload.trial_documents", str(error), key.step,
                row=key.row, variant=key.variant,
            ),), encoding_stamp)
        if not isinstance(paired, PreparedTrialExecution):
            return _LoweredRow(None, tuple(ProjectDiagnostic(
                ResolutionCode.INVALID_PAYLOAD,
                ".".join(("payload.trial_documents", *map(str, problem.path))),
                problem.reason, key.step, row=key.row, variant=key.variant,
                document=problem.model_address.document if problem.model_address else None,
                model_address=problem.model_address,
            ) for problem in paired), encoding_stamp)
        graph = paired.executable.resolved.graph
        if problem := model_owner_problem(graph, "checkpoint"):
            address = paired.lowered.source_address(pointer(problem.path))
            path = ("trial_documents", "model", *address.stored_path()) if address is not None else problem.path
            checkpoint_problem = RunProblem(pointer(path), problem.reason, address)
        assert model_source is not None
        if problem := model_owner_problem(graph, "evaluate"):
            address = paired.lowered.source_address(pointer(problem.path))
            model_source = replace(model_source, evaluation_problem=ProjectDiagnostic(
                ResolutionCode.INVALID_PAYLOAD, pointer(("inputs", "subject", *problem.path)), problem.reason,
                document=address.document if address is not None else None, model_address=address,
            ))
        else:
            model_source = replace(model_source, evaluation_graph=graph)
    if kind == "train" and isinstance(graph := payload.get("graph"), dict) and "interior" in graph:
        from feedbax.compiler.model_document import ModelCompilationProblem, lower_model_document
        from feedbax.studio.schema import validate_graph_connection_schema, validate_task_binding_schema

        def model_problem(detail: str, address: ModelAddress | None) -> _LoweredRow:
            path = "payload.graph"
            origin = None
            if address is not None:
                stored = address.stored_path()
                path += "".join(f".{segment}" for segment in stored)
                if sources is not None:
                    for count in range(len(stored), -1, -1):
                        if origin := sources.get(pointer(("payload", "graph", *stored[:count]))):
                            break
            return _LoweredRow(None, (ProjectDiagnostic(
                ResolutionCode.INVALID_PAYLOAD, path, detail, key.step,
                document=model_document, row=key.row, variant=key.variant,
                model_address=address, document_path=origin.path if origin and origin.kind == "document" else None,
                layer=f"{origin.kind}:{origin.name}{origin.path}" if origin and origin.kind != "document" else None,
            ),), encoding_stamp)

        try:
            model = ModelDocument.model_validate(graph)
            binding = StudioTaskBindingSpec.model_validate(payload.get("task_binding_spec") or {})
            components = components or _builtin_model_registry()
            lowered_model = lower_model_document(model_document or "model", model, components, binding)
            derived, binding = lowered_model.graph, lowered_model.task
            issues = validate_graph_connection_schema(
                derived, "/graph", binding, component_registry=components,
            )
            if binding is not None:
                issues.extend(validate_task_binding_schema(
                    binding, derived, "/task_binding_spec", component_registry=components,
                ))
            if issue := next((item for item in issues if item.severity == "error"), None):
                source = (issue.location or {}).get("path", "/graph")
                return model_problem(issue.message, lowered_model.source_address(source))
            payload = payload | {
                "graph": derived.model_dump(mode="json"),
                "task_binding_spec": binding.model_dump(mode="json") if binding else None,
            }
        except ModelCompilationProblem as error:
            return model_problem(str(error), error.address)
        except ValueError as error:
            return _LoweredRow(None, (ProjectDiagnostic(
                ResolutionCode.INVALID_PAYLOAD, "payload.graph", str(error), key.step,
                document=model_document, row=key.row, variant=key.variant,
            ),), encoding_stamp)
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
        elif document_train:
            spec = DocumentTrainingRequestSpec.model_validate(payload)
        else:
            spec = request_identity.REQUEST_KINDS[kind].spec_type.model_validate(payload)
        encoded = request_identity.encode_request(
            spec,
            {role: (binding.member,) for role, binding in inputs.items()},
            document_memo=document_memo,
        )
    except ValidationError as error:
        problems = tuple(
            (".".join(map(str, failure["loc"])), failure["msg"])
            for failure in error.errors(include_url=False, include_input=False)
        )
    except ValueError as error:
        problems = (("", str(error)),)
    else:
        problem = (
            RunProblem("/trial_documents", "Managed Train requires a Model, Task, and Objective document.")
            if kind == "train" and not document_train else
            RunProblem("/trial_documents/model/interior/components", "Model has no components.")
            if isinstance(spec, DocumentTrainingRequestSpec) and not spec.trial_documents.model.interior.components else
            RunProblem("/trial_documents/objective/terms", "Objective has no enabled terms.")
            if isinstance(spec, DocumentTrainingRequestSpec) and not any(
                term.enabled for term in spec.trial_documents.objective.terms
            ) else
            RunProblem("/graph/nodes", "Model has no components.")
            if isinstance(spec, TrainingRequestSpec) and spec.graph is not None and not spec.graph.nodes
            else checkpoint_problem
        )
        return _LoweredRow(
            encoded, (), encoding_stamp, admission_problem=problem,
            model_source=model_source,
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
        tuple[str, str], _WorkEntry[tuple[ExpandedRow, ...] | ProjectDiagnostic]
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
    ):
        self.project = project
        self.readers = readers
        self.previous = previous
        self.operations = operations
        self.components = components
        self.functions = functions
        self.python_tasks = python_tasks
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
        self.selected_definitions: dict[RunKey, SelectedRunDefinition] = {}
        self.model_sources: dict[RunKey, _ModelSource] = {}
        self.checkpoint_horizons: dict[RequestId, int | None] = {}

    def expand_variants(self) -> None:
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

        try:
            variants = expand_sweep(self.project.sweep, integer_site=integer_site)
        except SweepExpansionError as error:
            self.diagnostics.append(_sweep_problem(error))
            return
        self.variant_order = tuple(row.id for row in variants)
        for row in variants:
            if not row.patches:
                self.variants[row.id] = self.project
                continue
            data = self.project.model_dump(mode="json", by_alias=True, exclude_unset=True)
            data.pop("sweep", None)
            data.setdefault("documents", {})
            data.setdefault("steps", {})
            track = any(variant == row.id for variant, _ in self.wanted_sources)
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
        for variant, project in self.variants.items():
            if self.previous is not None:
                self.authored_inputs[variant] = _AuthoredInputs(project)
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
                    item.member.key, producer.coordinates, producer.label,
                    self.checkpoint_horizon(producer.request_id)
                    if checkpoint_axis and role == "subject" and isinstance(producer, PlannedRun) else None,
                )
                for item, producer in bindings
            )
            for role, bindings in members.items()
        }
        subjects = tuple(
            self.model_sources.get(item.producer) for item, _ in members.get("subject", ())
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

            try:
                return expand_sweep(
                    composition.sweep, inputs=fanout, integer_site=integer_site,
                    checkpoint_subject=kind == "evaluate",
                )
            except SweepExpansionError as error:
                return _sweep_problem(error, step=name, variant=variant)

        if self.previous is None:
            return expand()
        return _work(
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
    ) -> ResolvedStep | ProjectDiagnostic:
        project = self.variants[key.variant]
        return composition.bind(
            project,
            readers=self.readers,
            variant_sources=self.variant_sources.get(key.variant),
            read_document=reader,
        ).resolve(row, include_sources=sources, partial_figures=project.steps[key.step].kind == "analyze")

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
                producer = binding.producer
                path = ""
                if binding.producer_element is not None and isinstance(binding.member.reference, ProductRef):
                    path = pointer(("elements", binding.producer_element, "products", binding.member.reference.role))
                sources[f"/inputs/{role}"] = ValueSource(
                    "upstream",
                    producer.step,
                    path=path,
                    step=producer.step,
                    variant=producer.variant,
                    row=producer.row,
                )
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
    ) -> _LoweredRow:
        member = to_json(tuple(
            (role, binding.member.model_dump(mode="json")) for role, binding in sorted(inputs.items())
        ))
        subject = inputs.get("subject") if kind == "evaluate" else None
        model_source = self.model_sources.get(subject.producer) if subject is not None else None
        source_stamp = model_source.stamp if model_source is not None else None
        source_problem = model_source.evaluation_problem if model_source is not None else None

        def lower(reader=None, previous=None):
            if definition is None:
                resolved = self.resolve_row(composition, key, row, reader)
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
            has_differences = MODEL_ATTACHMENTS in payload
            encoding_stamp = (
                (kind, sha256_bytes(to_json((payload, document_ids.get("payload.graph"), source_stamp))), member)
                if basis is not None else None
            )
            if previous is not None and previous.encoding_stamp == encoding_stamp:
                return replace(
                    previous, document_ids=document_ids, document_uses=document_uses,
                )
            if source_problem is not None and payload.get("evaluation_type") == ROLLOUT_TYPE:
                return _LoweredRow(
                    None, (replace(source_problem, step=key.step, row=key.row, variant=key.variant),),
                    encoding_stamp, document_ids=document_ids, document_uses=document_uses,
                )
            if kind == "train":
                payload, authored_model = self.lower_train_model(key, payload, document_ids, document_pointers)
            elif has_differences:
                payload = self.lower_evaluate_model(
                    key, payload, model_source, subject.member.reference if subject is not None else None,
                )
            if kind == "evaluate" and isinstance(payload, dict) and subject is not None:
                payload = self.lower_evaluate_trials(key, payload, subject.member.reference, model_source)
            if isinstance(payload, ProjectDiagnostic):
                return _LoweredRow(None, (payload,), encoding_stamp, document_ids=document_ids, document_uses=document_uses)
            return replace(
                _encode_row(
                    key, kind, payload, inputs, encoding_stamp,
                    self.work.request_documents,
                    document_ids.get("payload.graph"),
                    definition.sources if definition is not None else None,
                    self.components, self.functions, self.python_tasks, model_source=authored_model,
                ),
                document_ids=document_ids,
                document_uses=document_uses,
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

    def lower_train_model(
        self, key: RunKey, payload: Payload, document_ids: Mapping[str, str], document_pointers: Mapping[str, str],
    ) -> tuple[Payload | ProjectDiagnostic, _ModelSource | None]:
        from feedbax.compiler.model_attachment import resolve_effective_model
        from feedbax.compiler.model_document import ModelCompilationProblem

        raw = payload.get("trial_documents")
        managed = isinstance(raw, dict)
        model_payload = raw.get("model") if managed else payload.get("graph")
        document = raw.get("model_name") if managed else document_ids.get("payload.graph", "model")
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

    def lower_evaluate_trials(
        self, key: RunKey, payload: Payload, reference: object, model_source: _ModelSource | None,
    ) -> Payload | ProjectDiagnostic:
        from feedbax.compiler.trial_execution import evaluation_conditions
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
        if not isinstance(train, (TrainingRequestSpec, DocumentTrainingRequestSpec)) or train.trial_documents is None:
            return payload
        raw = payload.get("params")
        if not isinstance(raw, dict):
            return payload
        location = "trial_set"
        try:
            if "trial_set" not in raw or "trials" in raw:
                raise ValueError("Declare a trial-set rule for managed Evaluate.")
            settings = authored_rollout_settings(raw)
            task = resolve_evaluation_variant(
                model_source.evaluation_graph, train.trial_documents.task_source, settings.variation,
            ).task
            conditions = evaluation_conditions(task, self.python_tasks)
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
            if not isinstance(train, (TrainingRequestSpec, DocumentTrainingRequestSpec)):
                raise ValueError("Attachment differences require a Train request")
            if model_source is None:
                raise ValueError("Attachment differences need a saved Model document")
            components = self.components or _builtin_model_registry()
            documents = train.trial_documents
            document, model = model_source.document, model_source.model
            if documents is not None:
                binding = None
                base_task = documents.task_source
            else:
                assert isinstance(train, TrainingRequestSpec)
                binding = train.task_binding_spec or StudioTaskBindingSpec()
                base_task = train.task
            before, lowered, derived = lower_evaluate_attachments(
                document, model, binding, components, differences, functions=self.functions,
            )
            raw_params = payload["params"]
            params = authored_rollout_settings(raw_params)
            if not derived:
                return stripped
            prior = params.variation
            if isinstance(prior, dict):
                prior = tuple(prior[name] for name in sorted(prior))
            variation = (*prior, *derived)
            effective = resolve_evaluation_variant(before.graph, base_task, variation)
            issues = validate_graph_connection_schema(
                effective.graph, "/graph", binding,
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
        except ValueError as error:
            detail = str(error)
        return _model_difference_problem(key, address, detail)

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
        result = []
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
            if kind == "evaluate" and chosen is not None:
                reference = chosen.member.reference
                assert isinstance(reference, ProductRef)
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
            lowered = self.lower_row(template, run_key, row, kind, inputs, basis, definition)
            if (run := self.record_run(run_key, lowered, inputs, omissions, template.requires)) is not None:
                result.append(run)
        return tuple(result)

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
        terms = reference.where
        subset_document = None
        if reference.row is not None:
            terms = SubsetTerms(rows=[reference.row])
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
        if failures and (
            terms is None
            or terms.rows is None
            or any(failure.row is None or failure.row in terms.rows for failure in failures)
        ):
            return ProjectDiagnostic(
                ResolutionCode.BAD_REFERENCE,
                path,
                f"referenced step {reference.step!r} has unplannable rows",
                name,
            )
        candidate_count = len(runs)
        try:
            runs = select_static_rows(
                terms,
                runs,
                key=lambda run: run.key.row,
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
        if not runs and not expands_input:
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
        for run in runs:
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
                        key=run.key.row
                        if reference.variants is None
                        else f"{run.key.variant}/{run.key.row}",
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
        subset = (
            PlannedSubset(
                deepcopy(terms), candidate_count, tuple(run.key for run in runs), subset_document
            )
            if terms is not None
            else None
        )
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
) -> ProjectPlan:
    """Plan static requests and value sources. Analyze lowering uses the supplied operations."""
    readers = replace(
        readers,
        external=cache(readers.external) if readers.external is not None else None,
        figure_external=cache(readers.figure_external) if readers.figure_external is not None else None,
    )
    if memo is None or any(
        reader is not None for reader in (readers.run, readers.external, readers.row, readers.figure_external)
    ):
        return _Compiler(project, readers, source_runs, operations=operations, components=components, functions=functions, python_tasks=python_tasks).plan()
    authority = _compiler_authority(components, operations, functions, python_tasks)
    ticket, previous = memo._begin(authority)
    compiler = _Compiler(project, readers, source_runs, previous, operations, components, functions)
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

    @classmethod
    def from_plan(cls, plan: ProjectPlan) -> PlanRequests:
        return cls._from_requests(plan, (*plan.runs, *plan.elements.values()))

    @classmethod
    def _from_requests(
        cls, plan: ProjectPlan, requests: Iterable[PlannedRun | PlannedElement],
    ) -> PlanRequests:
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
        return cls(entries, plan.requests, plan.documents)


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
        if isinstance(right.key, ElementKey):
            input_path = f"{right.path}.{input_path}"
        return ElementChange("changed", input_path, cause)
    relative = change.path.removeprefix("payload.")
    if relative == "analysis_type":
        relative = "operation"
    elif right.kind == "render" and relative.startswith("document."):
        relative = "figure." + relative.removeprefix("document.")
    elif right.kind == "report":
        relative = relative.removeprefix("document.")
    return ElementChange("changed", f"{right.path}.{relative}")
