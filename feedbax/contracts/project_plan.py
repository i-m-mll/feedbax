"""Compile authored training and evaluation requests without result access."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

from pydantic import ValidationError

from feedbax.contracts.matrix_core import apply_override_patch_in_place
from feedbax.contracts.project import Payload, Project, StepKind
from feedbax.contracts.project_inputs import (
    BarrierCheckpoint,
    BatchCheckpoint,
    BestCheckpoint,
    CheckpointRule,
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
    StepTemplate,
    compose_step,
)
from feedbax.contracts.project_sweep import (
    InputAxis,
    SweepCoordinate,
    SweepExpansionError,
    SweepInputMember,
    expand_sweep,
    sweep_patches,
)
from feedbax.contracts.project_sources import ValueSource, patch_source, pointer, source_shape
from feedbax.contracts.request_identity import REQUEST_KINDS, encode_request
from feedbax.contracts.request_refs import (
    BarrierMember,
    BatchMember,
    BestMember,
    InputMember,
    LastMember,
    ProductRef,
    ProductSelector,
    RequestId,
)


@dataclass(frozen=True)
class RunKey:
    variant: str
    step: str
    row: str


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


@dataclass(frozen=True)
class PlannedRow:
    """An expanded row, including a row whose request validation fails."""

    key: RunKey
    kind: StepKind
    label: str | None
    coordinates: Mapping[str, SweepCoordinate]


@dataclass(frozen=True)
class PlannedRun(PlannedRow):
    request_id: RequestId
    inputs: Mapping[str, PlannedInput]
    omissions: Mapping[str, str]
    requires: tuple[str, ...]


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
}


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
        case BarrierCheckpoint(barrier=name):
            return BarrierMember(name=name)


class _Compiler:
    def __init__(self, project: Project, readers: BaseReaders, source_runs: Iterable[RunKey]):
        self.project = project
        self.readers = readers
        self.wanted_sources: dict[tuple[str, str], set[str]] = {}
        for key in source_runs:
            self.wanted_sources.setdefault((key.variant, key.step), set()).add(key.row)
        self.variants: dict[str, Project] = {}
        self.variant_order: tuple[str, ...] = ()
        self.variant_sources: dict[str, Any] = {}
        self.steps: dict[tuple[str, str], PlannedStep] = {}
        self.rows: dict[RunKey, PlannedRow] = {}
        self.runs: dict[tuple[str, str], tuple[PlannedRun, ...]] = {}
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
            origins = source_shape(data, None) if track else None
            path = "sweep"
            try:
                for patch in sweep_patches(row.patches):
                    path = patch.path
                    parts = tuple(path.split("."))
                    if len(parts) < 2 or parts[0] not in {"documents", "steps"}:
                        raise ValueError("a variant must patch a named document or step")
                    apply_override_patch_in_place(data, patch)
                    if origins is not None:
                        source = ValueSource("variant", row.id, pointer(parts), variant=row.id)
                        patch_source(origins, patch, source_shape(patch.value, source))
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
            for name in sorted(project.steps):
                self.step(variant, name, ())
        return ProjectPlan(
            self.variant_order,
            MappingProxyType(self.steps),
            tuple(self.rows.values()),
            tuple(run for runs in self.runs.values() for run in runs),
            MappingProxyType(self.requests),
            MappingProxyType(self.documents),
            tuple(self.diagnostics),
            MappingProxyType(self.subsets),
            MappingProxyType(self.selected_definitions),
        )

    def step(
        self, variant: str, name: str, active: tuple[tuple[str, str], ...]
    ) -> tuple[PlannedRun, ...]:
        key = variant, name
        if key not in self.runs:
            self.runs[key] = self.compile_step(variant, name, (*active, key))
        return self.runs[key]

    def compile_step(
        self, variant: str, name: str, active: tuple[tuple[str, str], ...]
    ) -> tuple[PlannedRun, ...]:
        project = self.variants[variant]
        authored = project.steps[name]
        kind = authored.kind
        step = PlannedStep(kind, authored.group, MappingProxyType(deepcopy(authored.inputs)))
        self.steps[variant, name] = step
        wanted = self.wanted_sources.get((variant, name), set())
        role = _ROLES.get(kind)
        if role is None:
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
        template = compose_step(
            project,
            name,
            readers=self.readers,
            track_sources=bool(wanted),
            variant_sources=self.variant_sources.get(variant),
        )
        if isinstance(template, ProjectDiagnostic):
            self.diagnostics.append(replace(template, variant=variant))
            return ()
        self.steps[variant, name] = replace(
            step, inputs=MappingProxyType(deepcopy(dict(template.inputs)))
        )
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
        try:
            rows = expand_sweep(
                template.sweep,
                inputs={
                    role.name: tuple(
                        SweepInputMember(
                            key=item.member.key,
                            coordinates=producer.coordinates,
                            label=producer.label,
                        )
                        for item, producer in members
                    )
                },
            )
        except SweepExpansionError as error:
            self.diagnostics.append(
                ProjectDiagnostic(
                    ResolutionCode.INVALID_SWEEP, error.path, error.detail, name, variant=variant
                )
            )
            return ()

        result = []
        for row in rows:
            run_key = RunKey(variant, name, row.id)
            coordinates = MappingProxyType(row.coordinates)
            self.rows[run_key] = PlannedRow(run_key, kind, row.label, coordinates)
            key = row.input_keys.get(role.name)
            chosen = by_key[key] if key is not None else None
            if chosen is None and len(members) > 1:
                self.diagnostics.append(
                    ProjectDiagnostic(
                        ResolutionCode.BAD_REFERENCE,
                        f"inputs.{role.name}",
                        f"role requires one row; candidates are {', '.join(by_key)}",
                        name,
                        row=row.id,
                        variant=variant,
                    )
                )
                continue
            if chosen is None and members:
                chosen = members[0][0]
            resolved = template.resolve(row, include_sources=row.id in wanted)
            if isinstance(resolved, ProjectDiagnostic):
                self.diagnostics.append(replace(resolved, variant=variant))
                continue
            if resolved.sources is not None:
                sources = {f"/payload{path}": source for path, source in resolved.sources.items()}
                if chosen is not None:
                    producer = chosen.producer
                    sources[f"/inputs/{role.name}"] = ValueSource(
                        "upstream",
                        producer.step,
                        step=producer.step,
                        variant=producer.variant,
                        row=producer.row,
                    )
                self.selected_definitions[run_key] = SelectedRunDefinition(
                    resolved.payload, MappingProxyType(sources)
                )
            try:
                spec = REQUEST_KINDS[kind].spec_type.model_validate(resolved.payload)
                encoded = encode_request(spec, {role.name: (chosen.member,)} if chosen else {})
            except ValidationError as error:
                for failure in error.errors(include_url=False, include_input=False):
                    path = ".".join(map(str, failure["loc"]))
                    self.diagnostics.append(
                        ProjectDiagnostic(
                            ResolutionCode.INVALID_PAYLOAD,
                            f"payload.{path}" if path else "payload",
                            failure["msg"],
                            name,
                            row=row.id,
                            variant=variant,
                        )
                    )
                continue
            except ValueError as error:
                self.diagnostics.append(
                    ProjectDiagnostic(
                        ResolutionCode.INVALID_PAYLOAD,
                        "payload",
                        str(error),
                        name,
                        row=row.id,
                        variant=variant,
                    )
                )
                continue
            self.requests[encoded.id] = encoded.data
            self.documents.update(encoded.documents)
            run = PlannedRun(
                run_key,
                kind,
                row.label,
                coordinates,
                encoded.id,
                MappingProxyType({role.name: chosen} if chosen else {}),
                MappingProxyType(omissions),
                template.requires,
            )
            self.rows[run_key] = run
            result.append(run)
        return tuple(result)

    def select_input(
        self,
        variant: str,
        name: str,
        template: StepTemplate,
        role: _Role,
        active: tuple[tuple[str, str], ...],
    ) -> (
        tuple[tuple[tuple[PlannedInput, PlannedRun], ...], dict[str, str], PlannedSubset | None]
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
        runs: tuple[PlannedRun, ...] = ()
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
        members = tuple(
            (
                PlannedInput(
                    run.key,
                    InputMember(
                        key=run.key.row
                        if reference.variants is None
                        else f"{run.key.variant}/{run.key.row}",
                        reference=ProductRef(
                            request_id=run.request_id,
                            role=role.product_role,
                            select=_checkpoint(reference.checkpoint),
                        ),
                    ),
                ),
                run,
            )
            for run in runs
        )
        subset = (
            PlannedSubset(terms, candidate_count, tuple(run.key for run in runs), subset_document)
            if terms is not None
            else None
        )
        return members, {}, subset


def compile_project(
    project: Project,
    *,
    readers: BaseReaders = BaseReaders(),
    source_runs: Iterable[RunKey] = (),
) -> ProjectPlan:
    """Plan static requests and project value sources for the named run keys."""
    return _Compiler(project, readers, source_runs).plan()
