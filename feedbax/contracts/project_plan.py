"""Compile authored project requests without result access."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from threading import Lock
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ValidationError
from pydantic_core import to_json

from feedbax.contracts import request_identity
from feedbax.contracts.base import sha256_bytes
from feedbax.contracts.figure_document import PLOTLY_FIGURE_ROLE, PlotlyFigurePort
from feedbax.contracts.intent_diff import first_request_change
from feedbax.contracts.matrix_core import apply_override_patch_in_place
from feedbax.contracts.operation import DatasetPort
from feedbax.contracts.project import Document, Payload, Project, Step, StepBase, StepKind
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
)
from feedbax.contracts.project_sweep import (
    DEFAULT_SWEEP_ROW,
    ExpandedRow,
    InputAxis,
    SweepCoordinate,
    SweepExpansionError,
    SweepInputMember,
    expand_sweep,
    sweep_patches,
)
from feedbax.contracts.project_sources import ValueSource, patch_source, pointer, source_shape
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
)
from feedbax.contracts.request_specs import ReportRequestSpec, TrainingRequestSpec

if TYPE_CHECKING:
    from feedbax.analysis.operations import OperationRegistry


@dataclass(frozen=True)
class RunKey:
    variant: str
    step: str
    row: str


@dataclass(frozen=True)
class ElementKey:
    run: RunKey
    name: str


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
        family = "operations" if self.kind == "analyze" else "views"
        return f"payload.{family}.{self.key.name}"


@dataclass(frozen=True)
class PlannedAnalyzeRun(PlannedRow):
    elements: Mapping[str, PlannedElement]
    requires: tuple[str, ...]
    states: PlannedInput


@dataclass(frozen=True)
class ElementChange:
    """An authored change to a singleton run or element request."""

    state: Literal["unchanged", "changed", "added", "removed"]
    path: str | None = None
    upstream: ElementKey | None = None


@dataclass(frozen=True)
class PlannedStep:
    """The local input declarations and the effective bindings when resolved."""

    kind: StepKind
    group: str | None
    authored_inputs: Mapping[str, InputBinding | None]
    inputs: Mapping[str, InputBinding] | None = None


@dataclass(frozen=True)
class SelectedRunDefinition:
    payload: Payload
    sources: Mapping[str, ValueSource]
    document_ids: Mapping[str, str]
    document_uses: frozenset[str]


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


def _checkpoint(rule: CheckpointRule | None) -> ProductSelector | None:
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
        case BarrierCheckpoint(barrier=name):
            return BarrierMember(name=name)


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
class _LoweredRow:
    request: EncodedRequest | None
    diagnostics: tuple[ProjectDiagnostic, ...]
    encoding_stamp: tuple[StepKind, str, bytes] | None = None
    admission_problem: RunProblem | None = None
    document_ids: Mapping[str, str] = field(default_factory=dict)
    document_uses: frozenset[str] = frozenset()


def _encode_row(
    key: RunKey,
    kind: StepKind,
    payload: Payload,
    inputs: Mapping[str, PlannedInput],
    encoding_stamp: tuple[StepKind, str, bytes] | None,
    document_memo: RequestDocumentMemo | None,
) -> _LoweredRow:
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
            RunProblem("/graph/nodes", "Model has no components.")
            if isinstance(spec, TrainingRequestSpec) and not spec.graph.nodes
            else None
        )
        return _LoweredRow(encoded, (), encoding_stamp, admission_problem=problem)
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


def _compiler_authority() -> tuple[object, ...]:
    return (
        compose_step, StepTemplate.resolve, expand_sweep, source_shape, tuple(_ROLES.items()),
        request_identity.request_codec_authority(),
    )


class _Compiler:
    def __init__(
        self,
        project: Project,
        readers: BaseReaders,
        source_runs: Iterable[RunKey],
        previous: _PlanWork | None = None,
        operations: OperationRegistry | None = None,
    ):
        self.project = project
        self.readers = readers
        self.previous = previous
        self.operations = operations
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

    def expand_variants(self) -> None:
        try:
            variants = expand_sweep(self.project.sweep)
        except SweepExpansionError as error:
            self.diagnostics.append(
                ProjectDiagnostic(ResolutionCode.INVALID_SWEEP, error.path, error.detail)
            )
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
            patches = sweep_patches(row.patches)
            if track and self.previous is not None:
                self.variant_stamps[row.id] = sha256_bytes(to_json(patches))
            origins = source_shape(data, None) if track else None
            path = "sweep"
            try:
                for patch in patches:
                    path = patch.path
                    parts = tuple(path.split("."))
                    if len(parts) < 2 or parts[0] not in {"documents", "steps"}:
                        raise ValueError("a variant must patch a named document or step")
                    apply_override_patch_in_place(data, patch)
                    if origins is not None:
                        source = ValueSource("variant", row.id, pointer(parts), variant=row.id)
                        origins = patch_source(origins, patch, source_shape(patch.value, source))
                self.variants[row.id] = Project.model_validate(data)
                self.variant_sources[row.id] = origins
            except ValueError as error:
                self.diagnostics.append(
                    ProjectDiagnostic(
                        ResolutionCode.INVALID_SWEEP, path, str(error), variant=row.id
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
        fanout = {
            role: tuple(
                SweepInputMember(item.member.key, producer.coordinates, producer.label)
                for item, producer in bindings
            )
            for role, bindings in members.items()
        }

        def expand(_read_document=None, _previous=None):
            try:
                return expand_sweep(composition.sweep, inputs=fanout)
            except SweepExpansionError as error:
                return ProjectDiagnostic(
                    ResolutionCode.INVALID_SWEEP, error.path, error.detail, name, variant=variant
                )

        if self.previous is None:
            return expand()
        return _work(
            self.previous.sweeps,
            self.work.sweeps,
            (variant, name),
            (
                composition,
                to_json(tuple(
                    (role, tuple(
                        (member.key, tuple(member.coordinates.items()), member.label)
                        for member in bindings
                    ))
                    for role, bindings in sorted(fanout.items())
                )),
            ),
            self.authored_inputs[variant],
            expand,
        )

    def resolve_row(
        self,
        composition: StepComposition,
        key: RunKey,
        row: ExpandedRow,
        reader: Callable[[str], Document | None] | None,
        *,
        sources: bool = False,
    ) -> ResolvedStep | ProjectDiagnostic:
        return composition.bind(
            self.variants[key.variant],
            readers=self.readers,
            variant_sources=self.variant_sources.get(key.variant),
            read_document=reader,
        ).resolve(row, include_sources=sources)

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

        def lower(reader=None, previous=None):
            if definition is None:
                resolved = self.resolve_row(composition, key, row, reader)
                if isinstance(resolved, ProjectDiagnostic):
                    return _LoweredRow(None, (replace(resolved, variant=key.variant),))
                payload = resolved.payload
                document_ids = MappingProxyType(resolved.document_ids)
                document_uses = resolved.document_uses
            else:
                if reader is not None:
                    for name, _ in self.work.definitions[key].documents:
                        reader(name)
                payload = definition.payload
                document_ids = definition.document_ids
                document_uses = definition.document_uses
            encoding_stamp = (
                (kind, sha256_bytes(to_json(payload)), member)
                if basis is not None else None
            )
            if previous is not None and previous.encoding_stamp == encoding_stamp:
                return replace(previous, document_ids=document_ids, document_uses=document_uses)
            return replace(
                _encode_row(
                    key, kind, payload, inputs, encoding_stamp,
                    self.work.request_documents if self.previous is not None else None,
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
            (kind, basis, to_json(tuple(row.patches.items())), tuple(sorted(inputs.items()))),
            self.authored_inputs[key.variant],
            lower,
        )

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
            step, inputs=MappingProxyType(deepcopy(dict(template.inputs)))
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
        if (encoded := lowered.request) is None:
            self.rows[key] = replace(
                row, document_ids=lowered.document_ids, document_uses=lowered.document_uses,
            )
            return None
        self.requests[encoded.id] = encoded.data
        self.documents.update(encoded.documents)
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
        elements, encoded, problems = lower_analyze(
            key,
            resolved.payload,
            resolved.document_ids,
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
            MappingProxyType(elements), template.requires, source,
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
        selector = _checkpoint(reference.checkpoint)
        if isinstance(selector, BatchSetMember) and role.name != "subject":
            return ProjectDiagnostic(
                ResolutionCode.BAD_REFERENCE, f"{path}.checkpoint",
                "finite checkpoint sets require an Evaluate subject", name,
            )
        members = []
        for run in runs:
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
) -> ProjectPlan:
    """Plan static requests and value sources. Analyze lowering uses the supplied operations."""
    if memo is None or any(reader is not None for reader in (readers.run, readers.external, readers.row)):
        return _Compiler(project, readers, source_runs, operations=operations).plan()
    authority = _compiler_authority()
    ticket, previous = memo._begin(authority)
    compiler = _Compiler(project, readers, source_runs, previous, operations)
    plan = compiler.plan()
    if authority == _compiler_authority():
        memo._publish(ticket, authority, compiler.work)
    return plan


def diff_analyze_elements(before: ProjectPlan, after: ProjectPlan) -> Mapping[ElementKey, ElementChange]:
    """Compare element requests and locate the authored edit or changed input."""
    return _diff_requests(before, after, before.elements, after.elements)


def diff_plan_requests(
    before: ProjectPlan, after: ProjectPlan,
) -> Mapping[RunKey | ElementKey, ElementChange]:
    """Compare all authored request addresses in two plans."""
    def addresses(plan: ProjectPlan) -> dict[RunKey | ElementKey, PlannedRun | PlannedElement]:
        return {run.key: run for run in plan.runs} | dict(plan.elements)

    return _diff_requests(before, after, addresses(before), addresses(after))


def _diff_requests[K: RunKey | ElementKey](
    before: ProjectPlan, after: ProjectPlan,
    old: Mapping[K, PlannedRun | PlannedElement], new: Mapping[K, PlannedRun | PlannedElement],
) -> Mapping[K, ElementChange]:
    changes = {}
    for key in dict.fromkeys((*old, *new)):
        left, right = old.get(key), new.get(key)
        if left is None:
            changes[key] = ElementChange("added")
        elif right is None:
            changes[key] = ElementChange("removed")
        elif left.request_id == right.request_id:
            changes[key] = ElementChange("unchanged")
        elif left.kind != right.kind:
            path = right.path if isinstance(right, PlannedElement) else "payload"
            changes[key] = ElementChange("changed", path)
        else:
            changes[key] = _request_change(before, after, left, right)
    return MappingProxyType(changes)


def _request_change(
    before: ProjectPlan, after: ProjectPlan,
    left: PlannedRun | PlannedElement, right: PlannedRun | PlannedElement,
) -> ElementChange:
    old = request_identity.decode_request(before.requests[left.request_id], before.documents.__getitem__)
    new = request_identity.decode_request(after.requests[right.request_id], after.documents.__getitem__)
    change = first_request_change(old, new)
    path = right.path if isinstance(right, PlannedElement) else "payload"
    if change.path == "inputs":
        role = next(
            role for role in sorted(old.inputs.keys() | new.inputs.keys())
            if old.inputs.get(role) != new.inputs.get(role)
        )
        earlier, later = left.inputs.get(role), right.inputs.get(role)
        cause = (
            later.element
            if later is not None and earlier is not None and later.element == earlier.element
            else None
        )
        input_path = f"{path}.inputs.{role}" if isinstance(right, PlannedElement) else f"inputs.{role}"
        return ElementChange("changed", input_path, cause)
    relative = change.path.removeprefix("payload.")
    if relative == "analysis_type":
        relative = "operation"
    elif right.kind == "render" and relative.startswith("document."):
        relative = "figure." + relative.removeprefix("document.")
    elif right.kind == "report":
        relative = relative.removeprefix("document.")
    return ElementChange("changed", f"{path}.{relative}")
