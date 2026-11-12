"""Static project resolution with explicit readers for external authorities."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, cast

from pydantic import JsonValue, ValidationError

from feedbax.contracts.base import sha256_bytes
from feedbax.contracts.matrix_core import apply_override_patch_in_place
from feedbax.contracts.project import (
    DocumentReference,
    ExternalBase,
    InlineBase,
    Payload,
    Project,
    RunBase,
    StepBase,
    StepKind,
)
from feedbax.contracts.project_inputs import InputBinding
from feedbax.contracts.project_sweep import (
    ExpandedRow,
    PatchValue,
    Sweep,
    SweepInheritanceError,
    inherit_sweep,
    merge_named,
    sweep_patches,
)
from feedbax.contracts.run_matrix import CompositionDeltaError, apply_composition_deltas
from feedbax.contracts.strict_json import strict_json_loads


class ResolutionCode(StrEnum):
    MISSING_STEP = "missing_step"
    STEP_CYCLE = "step_cycle"
    MISSING_DOCUMENT = "missing_document"
    DOCUMENT_CYCLE = "document_cycle"
    BAD_POINTER = "bad_pointer"
    BAD_REFERENCE = "bad_reference"
    MISSING_AUTHORITY = "missing_authority"
    BAD_EXTERNAL_CONTENT = "bad_external_content"
    INVALID_PAYLOAD = "invalid_payload"
    INVALID_DELTA = "invalid_delta"
    INVALID_SWEEP = "invalid_sweep"
    UNKNOWN_ROW = "unknown_row"
    UNBOUND_REQUIRED = "unbound_required"
    UNKNOWN_SCHEMA = "unknown_schema"
    INVALID_DOCUMENT = "invalid_document"
    UNSUPPORTED_PLAN = "unsupported_plan"


@dataclass(frozen=True)
class ProjectDiagnostic:
    code: ResolutionCode
    path: str
    detail: str
    step: str | None = None
    layer: str | None = None
    document: str | None = None
    row: str | None = None


@dataclass(frozen=True)
class ResolvedStep:
    name: str
    kind: StepKind
    payload: Payload
    sweep: Sweep | None
    inputs: dict[str, InputBinding]


@dataclass(frozen=True)
class SelectedRow:
    payload: Payload
    inputs: dict[str, InputBinding]


@dataclass(frozen=True)
class BaseReaders:
    """Read exact historical payloads and external bytes from caller-owned sources.

    Explicit authored rows resolve within this module. A sweep implementation can
    supply `row` to resolve a generated row from the parent step and its exact id.
    It returns the selected payload and inputs, including any fan-out bindings.
    Missing values return None. OSError from a reader becomes a diagnostic.
    External bytes are checked against the authored pin before parsing. Readers
    must be read-only.
    """

    run: Callable[[str], Payload | None] | None = None
    external: Callable[[str], bytes | None] | None = None
    row: Callable[[ResolvedStep, str], SelectedRow | None] | None = None


class _Problem(Exception):
    def __init__(
        self,
        code: ResolutionCode,
        path: str,
        detail: str,
        *,
        step: str | None = None,
        layer: str | None = None,
        document: str | None = None,
    ):
        self.diagnostic = ProjectDiagnostic(code, path, detail, step, layer, document)
        super().__init__(detail)


def _json_pointer(pointer: str | None) -> tuple[str, ...]:
    if not pointer:
        return ()
    return tuple(token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/"))


def _select(value: Any, parts: tuple[str, ...]) -> Any:
    for part in parts:
        if isinstance(value, dict):
            value = value[part]
        elif isinstance(value, list) and (
            part == "0" or part.isascii() and part.isdigit() and not part.startswith("0")
        ):
            try:
                value = value[int(part)]
            except (IndexError, ValueError):
                raise KeyError(part) from None
        else:
            raise KeyError(part)
    return value


def _pack(value: Any, path: str, step: str | None) -> Any:
    """Treat document references as leaves for the shared patch engine."""
    if isinstance(value, dict):
        if "document" in value and value.keys() <= {"document", "pointer"}:
            try:
                return DocumentReference.model_validate(value)
            except ValidationError as error:
                raise _Problem(ResolutionCode.BAD_REFERENCE, path, str(error), step=step) from error
        return {key: _pack(item, f"{path}.{key}", step) for key, item in value.items()}
    if isinstance(value, list):
        return [_pack(item, f"{path}.{index}", step) for index, item in enumerate(value)]
    return value


class _Documents:
    def __init__(self, project: Project, step: str | None = None):
        self.project = project
        self.step = step

    def select(self, reference: DocumentReference, path: str) -> Any:
        document = self.project.documents.get(reference.document)
        if document is None:
            raise _Problem(
                ResolutionCode.MISSING_DOCUMENT,
                path,
                reference.document,
                step=self.step,
                document=reference.document,
            )
        try:
            return _select(document.payload, _json_pointer(reference.pointer))
        except KeyError as error:
            raise _Problem(
                ResolutionCode.BAD_POINTER,
                path,
                reference.pointer or "",
                step=self.step,
                document=reference.document,
            ) from error

    def _follow(
        self,
        reference: DocumentReference,
        path: str,
        active: tuple[tuple[str, str], ...],
    ) -> tuple[Any, tuple[tuple[str, str], ...]]:
        key = reference.document, reference.pointer or ""
        if key in active:
            chain = " -> ".join(f"{name}#{pointer}" for name, pointer in (*active, key))
            raise _Problem(
                ResolutionCode.DOCUMENT_CYCLE,
                path,
                chain,
                step=self.step,
                document=reference.document,
            )
        return _pack(self.select(reference, path), path, self.step), (*active, key)

    def base(self, reference: DocumentReference) -> Any:
        """Follow root aliases before patching; nested references remain atomic."""
        value: Any = reference
        active: tuple[tuple[str, str], ...] = ()
        while isinstance(value, DocumentReference):
            value, active = self._follow(value, "base", active)
        return value

    def inline(
        self,
        value: Any,
        path: str,
        active: tuple[tuple[str, str], ...] = (),
    ) -> JsonValue:
        if isinstance(value, DocumentReference):
            selected, active = self._follow(value, path, active)
            return self.inline(selected, path, active)
        if isinstance(value, dict):
            return {key: self.inline(item, f"{path}.{key}", active) for key, item in value.items()}
        if isinstance(value, list):
            return [self.inline(item, f"{path}.{i}", active) for i, item in enumerate(value)]
        return cast(JsonValue, value)


@dataclass
class _StepState:
    payload: dict[str, Any] | DocumentReference
    sweep: Sweep | None = None
    inputs: dict[str, InputBinding] = field(default_factory=dict)
    requires: tuple[str, ...] = ()
    written: set[str] = field(default_factory=set)


class _Resolver:
    def __init__(self, project: Project, readers: BaseReaders):
        self.project = project
        self.readers = readers

    def state(self, name: str, active: tuple[str, ...] = ()) -> _StepState:
        if name in active:
            raise _Problem(
                ResolutionCode.STEP_CYCLE,
                "base.step",
                " -> ".join((*active, name)),
                step=active[-1],
            )
        step = self.project.steps.get(name)
        if step is None:
            raise _Problem(
                ResolutionCode.MISSING_STEP, "base.step", name, step=active[-1] if active else name
            )
        base = step.base
        if isinstance(base, StepBase):
            parent = self.state(base.step, (*active, name))
            if base.row is not None:
                parent = self.select_row(name, base, parent)
        else:
            payload = self.read_base(name, base)
            if not isinstance(payload, dict):
                raise _Problem(
                    ResolutionCode.INVALID_PAYLOAD,
                    "base",
                    "step base must select an object",
                    step=name,
                )
            parent = _StepState(
                payload
                if isinstance(base, (DocumentReference, RunBase))
                else _pack(payload, "payload", name)
            )
        payload = parent.payload
        written = parent.written
        for layer in step.deltas:
            if not isinstance(payload, dict):
                raise _Problem(
                    ResolutionCode.INVALID_DELTA,
                    layer.patches[0].path if layer.patches else "base",
                    "a root document reference is atomic",
                    step=name,
                    layer=layer.layer_id,
                )
            patches = [
                patch.model_copy(update={"value": _pack(patch.value, patch.path, name)})
                if patch.op != "remove"
                else patch
                for patch in layer.patches
            ]
            try:
                payload, _, written = apply_composition_deltas(
                    payload,
                    [layer.model_copy(update={"patches": patches})],
                    ancestor_written_paths=written,
                )
            except CompositionDeltaError as error:
                raise _Problem(
                    ResolutionCode.INVALID_DELTA,
                    error.path,
                    str(error),
                    step=name,
                    layer=error.layer,
                ) from error
            except ValueError as error:
                raise _Problem(
                    ResolutionCode.INVALID_DELTA,
                    "deltas",
                    str(error),
                    step=name,
                    layer=layer.layer_id,
                ) from error
        try:
            sweep = inherit_sweep(parent.sweep, step.sweep)
        except SweepInheritanceError as error:
            raise _Problem(
                ResolutionCode.INVALID_SWEEP, f"sweep.rows.{error.row}.from", str(error), step=name
            ) from error
        return _StepState(
            payload,
            sweep,
            merge_named(parent.inputs, step.inputs),
            tuple(dict.fromkeys((*parent.requires, *step.requires))),
            written,
        )

    def read_base(
        self, step: str, base: InlineBase | DocumentReference | RunBase | ExternalBase
    ) -> JsonValue:
        if isinstance(base, InlineBase):
            return base.inline
        if isinstance(base, DocumentReference):
            return _Documents(self.project, step).base(base)
        if isinstance(base, RunBase):
            try:
                payload = self.readers.run(base.run) if self.readers.run is not None else None
            except OSError as error:
                raise _Problem(
                    ResolutionCode.MISSING_AUTHORITY, "base.run", str(error), step=step
                ) from error
            if payload is None:
                raise _Problem(ResolutionCode.MISSING_AUTHORITY, "base.run", base.run, step=step)
            return payload
        try:
            data = self.readers.external(base.path) if self.readers.external is not None else None
        except OSError as error:
            raise _Problem(
                ResolutionCode.MISSING_AUTHORITY, "base.path", str(error), step=step
            ) from error
        if data is None:
            raise _Problem(ResolutionCode.MISSING_AUTHORITY, "base.path", base.path, step=step)
        observed = sha256_bytes(data)
        if observed != base.sha256:
            raise _Problem(
                ResolutionCode.BAD_EXTERNAL_CONTENT,
                "base.sha256",
                f"expected {base.sha256}, got {observed}",
                step=step,
            )
        try:
            payload = strict_json_loads(data, ref=base.path)
        except (ValueError, UnicodeError) as error:
            raise _Problem(
                ResolutionCode.BAD_EXTERNAL_CONTENT, "base.path", str(error), step=step
            ) from error
        try:
            return _select(payload, _json_pointer(base.pointer))
        except KeyError as error:
            raise _Problem(
                ResolutionCode.BAD_POINTER, "base.pointer", base.pointer or "", step=step
            ) from error

    def select_row(self, child: str, base: StepBase, parent: _StepState) -> _StepState:
        assert base.row is not None
        row = parent.sweep.rows.get(base.row) if parent.sweep is not None else None
        if row is not None:
            state = self.apply_row(base.step, parent, base.row, row.patches)
        elif self.readers.row is not None:
            parent_step = self.finish(base.step, parent, check_requires=False)
            try:
                generated = self.readers.row(parent_step, base.row)
            except OSError as error:
                raise _Problem(
                    ResolutionCode.MISSING_AUTHORITY, "base.row", str(error), step=child
                ) from error
            if generated is None:
                raise _Problem(ResolutionCode.UNKNOWN_ROW, "base.row", base.row, step=child)
            state = _StepState(
                generated.payload,
                inputs=generated.inputs,
                requires=parent.requires,
            )
        else:
            raise _Problem(ResolutionCode.UNKNOWN_ROW, "base.row", base.row, step=child)
        selected = self.finish(base.step, state)
        return _StepState(selected.payload, inputs=selected.inputs)

    def apply_row(
        self,
        name: str,
        state: _StepState,
        row: str,
        patches: Mapping[str, PatchValue | None],
    ) -> _StepState:
        payload = deepcopy(state.payload)
        for patch in sweep_patches(patches):
            try:
                if not isinstance(payload, dict):
                    raise ValueError("a root document reference is atomic")
                if patch.op != "remove":
                    patch = patch.model_copy(update={"value": _pack(patch.value, patch.path, name)})
                apply_override_patch_in_place(payload, patch)
            except ValueError as error:
                raise _Problem(
                    ResolutionCode.INVALID_SWEEP, patch.path, str(error), step=name, layer=row
                ) from error
        return _StepState(payload, inputs=state.inputs, requires=state.requires)

    def finish(self, name: str, state: _StepState, *, check_requires: bool = True) -> ResolvedStep:
        payload = _Documents(self.project, name).inline(state.payload, "payload")
        if not isinstance(payload, dict):
            raise _Problem(
                ResolutionCode.INVALID_PAYLOAD,
                "payload",
                "step payload must be an object",
                step=name,
            )
        if check_requires:
            for path in state.requires:
                try:
                    if _select(payload, tuple(path.split("."))) is not None:
                        continue
                except KeyError:
                    pass
                raise _Problem(
                    ResolutionCode.UNBOUND_REQUIRED,
                    path,
                    "required payload path is unbound",
                    step=name,
                )
        return ResolvedStep(
            name,
            self.project.steps[name].kind,
            payload,
            deepcopy(state.sweep),
            deepcopy(state.inputs),
        )


@dataclass(frozen=True)
class StepTemplate:
    """Composed step whose requirements are checked on each final row."""

    _resolver: _Resolver
    name: str
    _state: _StepState

    @property
    def sweep(self) -> Sweep | None:
        return self._state.sweep

    @property
    def inputs(self) -> Mapping[str, InputBinding]:
        return self._state.inputs

    @property
    def requires(self) -> tuple[str, ...]:
        return self._state.requires

    def resolve(self, row: ExpandedRow | None = None) -> ResolvedStep | ProjectDiagnostic:
        try:
            state = self._state
            if row is not None:
                state = self._resolver.apply_row(self.name, state, row.id, row.patches)
            return self._resolver.finish(self.name, state)
        except _Problem as error:
            if row is not None:
                return replace(error.diagnostic, row=row.id)
            return error.diagnostic


def compose_step(
    project: Project, step: str, *, readers: BaseReaders = BaseReaders()
) -> StepTemplate | ProjectDiagnostic:
    """Compose inherited layers and prepare a step for sweep expansion."""
    resolver = _Resolver(project, readers)
    try:
        return StepTemplate(resolver, step, resolver.state(step))
    except _Problem as error:
        return error.diagnostic


def resolve_step(
    project: Project, step: str, *, readers: BaseReaders = BaseReaders()
) -> ResolvedStep | ProjectDiagnostic:
    """Resolve one named step through base composition and static sweep refinement."""
    template = compose_step(project, step, readers=readers)
    return template if isinstance(template, ProjectDiagnostic) else template.resolve()


def validate_project_documents(
    project: Project,
    schemas: Mapping[str, Callable[[JsonValue], object]],
) -> tuple[ProjectDiagnostic, ...]:
    """Validate inlined document payloads using the caller's explicit schema authorities."""
    documents = _Documents(project)
    failures = []
    for name, document in project.documents.items():
        validator = schemas.get(document.schema_id)
        if validator is None:
            failures.append(
                ProjectDiagnostic(
                    ResolutionCode.UNKNOWN_SCHEMA, "schema", document.schema_id, document=name
                )
            )
            continue
        try:
            payload = documents.inline(DocumentReference(document=name), "payload")
            validator(payload)
        except _Problem as error:
            failures.append(replace(error.diagnostic, document=name))
        except ValueError as error:
            if isinstance(error, ValidationError):
                failures.extend(
                    ProjectDiagnostic(
                        ResolutionCode.INVALID_DOCUMENT,
                        ".".join(("payload", *map(str, item["loc"]))),
                        item["msg"],
                        document=name,
                    )
                    for item in error.errors()
                )
            else:
                failures.append(
                    ProjectDiagnostic(
                        ResolutionCode.INVALID_DOCUMENT,
                        "payload",
                        str(error),
                        document=name,
                    )
                )
    return tuple(failures)
