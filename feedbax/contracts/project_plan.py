"""Compile authored training and evaluation requests without result access."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from pydantic import ValidationError

from feedbax.contracts.project import Project, StepKind
from feedbax.contracts.project_inputs import (
    BarrierCheckpoint,
    BatchCheckpoint,
    BestCheckpoint,
    CheckpointRule,
    OmittedReference,
    StepReference,
)
from feedbax.contracts.project_resolution import (
    BaseReaders,
    ProjectDiagnostic,
    ResolutionCode,
    StepTemplate,
    compose_step,
)
from feedbax.contracts.project_sweep import (
    SamplerResolver,
    SweepCoordinate,
    SweepExpansionError,
    SweepInputMember,
    expand_sweep,
)
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
    step: str
    row: str


@dataclass(frozen=True)
class PlannedInput:
    producer: RunKey
    member: InputMember


@dataclass(frozen=True)
class PlannedRun:
    key: RunKey
    kind: StepKind
    request_id: RequestId
    label: str | None
    coordinates: Mapping[str, SweepCoordinate]
    inputs: Mapping[str, PlannedInput]
    omissions: Mapping[str, str]
    requires: tuple[str, ...]


@dataclass(frozen=True)
class ProjectPlan:
    """Named runs and deduplicated objects encoded by the shared request codec."""

    runs: tuple[PlannedRun, ...]
    requests: Mapping[RequestId, bytes]
    documents: Mapping[str, bytes]
    diagnostics: tuple[ProjectDiagnostic, ...]


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
    def __init__(self, project: Project, readers: BaseReaders, sampler: SamplerResolver | None):
        self.project = project
        self.readers = readers
        self.sampler = sampler
        self.runs: dict[str, tuple[PlannedRun, ...]] = {}
        self.requests: dict[RequestId, bytes] = {}
        self.documents: dict[str, bytes] = {}
        self.diagnostics: list[ProjectDiagnostic] = []

    def plan(self) -> ProjectPlan:
        if self.project.sweep is not None:
            self.diagnostics.append(
                ProjectDiagnostic(
                    ResolutionCode.UNSUPPORTED_PLAN,
                    "sweep",
                    "project variants require variant planning",
                )
            )
        else:
            for name in sorted(self.project.steps):
                self.step(name, ())
        return ProjectPlan(
            tuple(run for runs in self.runs.values() for run in runs),
            MappingProxyType(self.requests),
            MappingProxyType(self.documents),
            tuple(self.diagnostics),
        )

    def step(self, name: str, active: tuple[str, ...]) -> tuple[PlannedRun, ...]:
        if name not in self.runs:
            self.runs[name] = self.compile_step(name, (*active, name))
        return self.runs[name]

    def compile_step(self, name: str, active: tuple[str, ...]) -> tuple[PlannedRun, ...]:
        kind = self.project.steps[name].kind
        role = _ROLES.get(kind)
        if role is None:
            self.diagnostics.append(
                ProjectDiagnostic(
                    ResolutionCode.UNSUPPORTED_PLAN, "kind", f"{kind} lowering is unavailable", name
                )
            )
            return ()
        template = compose_step(self.project, name, readers=self.readers)
        if isinstance(template, ProjectDiagnostic):
            self.diagnostics.append(template)
            return ()
        extra_roles = sorted(template.inputs.keys() - {role.name})
        if extra_roles:
            self.diagnostics.extend(
                ProjectDiagnostic(
                    ResolutionCode.BAD_REFERENCE, f"inputs.{extra}", "unknown input role", name
                )
                for extra in extra_roles
            )
            return ()
        selected = self.select_input(name, template, role, active)
        if isinstance(selected, ProjectDiagnostic):
            self.diagnostics.append(selected)
            return ()
        members, omissions = selected
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
                sampler=self.sampler,
            )
        except SweepExpansionError as error:
            self.diagnostics.append(
                ProjectDiagnostic(ResolutionCode.INVALID_SWEEP, error.path, error.detail, name)
            )
            return ()

        result = []
        for row in rows:
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
                    )
                )
                continue
            if chosen is None and members:
                chosen = members[0][0]
            resolved = template.resolve(row)
            if isinstance(resolved, ProjectDiagnostic):
                self.diagnostics.append(resolved)
                continue
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
                        )
                    )
                continue
            except ValueError as error:
                self.diagnostics.append(
                    ProjectDiagnostic(
                        ResolutionCode.INVALID_PAYLOAD, "payload", str(error), name, row=row.id
                    )
                )
                continue
            self.requests[encoded.id] = encoded.data
            self.documents.update(encoded.documents)
            result.append(
                PlannedRun(
                    RunKey(name, row.id),
                    kind,
                    encoded.id,
                    row.label,
                    MappingProxyType(row.coordinates),
                    MappingProxyType({role.name: chosen} if chosen else {}),
                    MappingProxyType(omissions),
                    template.requires,
                )
            )
        return tuple(result)

    def select_input(
        self, name: str, template: StepTemplate, role: _Role, active: tuple[str, ...]
    ) -> tuple[tuple[tuple[PlannedInput, PlannedRun], ...], dict[str, str]] | ProjectDiagnostic:
        reference = template.inputs.get(role.name)
        path = f"inputs.{role.name}"
        if reference is None or isinstance(reference, OmittedReference):
            if role.required:
                return ProjectDiagnostic(
                    ResolutionCode.BAD_REFERENCE, path, "required input is missing", name
                )
            return (), {role.name: reference.not_applicable} if reference else {}
        if (
            not isinstance(reference, StepReference)
            or reference.where is not None
            or reference.variants is not None
        ):
            return ProjectDiagnostic(
                ResolutionCode.UNSUPPORTED_PLAN,
                path,
                "a same-project step or row reference is required",
                name,
            )
        if reference.step in active:
            return ProjectDiagnostic(
                ResolutionCode.STEP_CYCLE, path, " -> ".join((*active, reference.step)), name
            )
        producer = self.project.steps.get(reference.step)
        if producer is None or producer.kind != role.producer_kind:
            return ProjectDiagnostic(
                ResolutionCode.BAD_REFERENCE,
                path,
                f"{reference.step!r} must be a {role.producer_kind} step",
                name,
            )
        runs = self.step(reference.step, active)
        if reference.row is not None:
            runs = tuple(run for run in runs if run.key.row == reference.row)
        elif any(problem.step == reference.step for problem in self.diagnostics):
            runs = ()
        if not runs:
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
                        key=run.key.row,
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
        return members, {}


def compile_project(
    project: Project,
    *,
    readers: BaseReaders = BaseReaders(),
    sampler: SamplerResolver | None = None,
) -> ProjectPlan:
    """Plan train/evaluate rows and resolve their references to requests."""
    return _Compiler(project, readers, sampler).plan()
