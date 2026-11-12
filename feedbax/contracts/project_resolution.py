"""Static project resolution with explicit readers for external authorities."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import cached_property
from typing import Any, cast

from pydantic import JsonValue, ValidationError
from pydantic_core import to_json

from feedbax.contracts.base import sha256_bytes
from feedbax.contracts.matrix_core import apply_override_patch_in_place
from feedbax.contracts.project import (
    Document,
    DocumentReference,
    ExternalBase,
    InlineBase,
    Payload,
    Project,
    RunBase,
    StepBase,
    StepKind,
    json_pointer_parts,
)
from feedbax.contracts.project_inputs import InputBinding
from feedbax.contracts.project_sources import (
    SourceList,
    SourceObject,
    ValueSource,
    first_source,
    patch_source,
    pointer,
    source_at,
    source_leaves,
    source_shape,
)
from feedbax.contracts.project_sweep import (
    ExpandedRow,
    ListedAxis,
    RangeAxis,
    SampledAxis,
    Sweep,
    SweepExpansionError,
    SweepInheritanceError,
    authored_row,
    inherit_sweep,
    merge_named,
    sweep_patches,
    sweep_value_identity_bytes,
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
    variant: str | None = None


@dataclass(frozen=True)
class ResolvedStep:
    name: str
    kind: StepKind
    payload: Payload
    sweep: Sweep | None
    inputs: dict[str, InputBinding]
    sources: dict[str, ValueSource] | None = None


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
    def __init__(
        self,
        project: Project,
        step: str | None = None,
        variant_sources: Any = None,
        read_document: Callable[[str], Document | None] | None = None,
    ):
        self.step = step
        self.variant_sources = variant_sources
        self.read_document = project.documents.get if read_document is None else read_document

    def select(self, reference: DocumentReference, path: str) -> Any:
        document = self.read_document(reference.document)
        if document is None:
            raise _Problem(
                ResolutionCode.MISSING_DOCUMENT,
                path,
                reference.document,
                step=self.step,
                document=reference.document,
            )
        try:
            return _select(document.payload, json_pointer_parts(reference.pointer))
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

    def base(self, reference: DocumentReference) -> tuple[Any, DocumentReference]:
        """Follow root aliases before patching; nested references remain atomic."""
        value: Any = reference
        active: tuple[tuple[str, str], ...] = ()
        while isinstance(value, DocumentReference):
            reference = value
            value, active = self._follow(value, "base", active)
        return value, reference

    def source(self, reference: DocumentReference, value: Any) -> Any:
        parts = json_pointer_parts(reference.pointer)
        return source_shape(
            value,
            ValueSource("document", reference.document, pointer(parts)),
            source_at(self.variant_sources, ("documents", reference.document, "payload", *parts)),
        )

    def inline_sources(
        self, value: Any, shape: Any, path: str, active: tuple[tuple[str, str], ...] = ()
    ) -> Any:
        if isinstance(value, DocumentReference):
            selected, active = self._follow(value, path, active)
            return self.inline_sources(selected, self.source(value, selected), path, active)
        if isinstance(value, dict):
            return SourceObject(
                first_source(shape) if not isinstance(shape, SourceObject) else shape.source,
                {
                    key: self.inline_sources(
                        item, source_at(shape, (key,)), f"{path}.{key}", active
                    )
                    for key, item in value.items()
                },
            )
        if isinstance(value, list):
            return SourceList(
                first_source(shape) if not isinstance(shape, SourceList) else shape.source,
                (
                    self.inline_sources(item, source_at(shape, (str(i),)), f"{path}.{i}", active)
                    for i, item in enumerate(value)
                ),
            )
        return shape

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
    origins: Any = None


class _Resolver:
    def __init__(
        self,
        project: Project,
        readers: BaseReaders,
        track_sources: bool = False,
        variant_sources: Any = None,
        read_document: Callable[[str], Document | None] | None = None,
    ):
        self.project = project
        self.readers = readers
        self.track_sources = track_sources
        self.variant_sources = variant_sources
        self.read_document = read_document

    def documents(self, step: str) -> _Documents:
        return _Documents(self.project, step, self.variant_sources, self.read_document)

    def base_origins(
        self, name: str, base: InlineBase | RunBase | ExternalBase, payload: Any
    ) -> Any:
        if isinstance(base, RunBase):
            return source_shape(payload, ValueSource("upstream", base.run))
        if isinstance(base, ExternalBase):
            return source_shape(payload, ValueSource("document", base.path, base.pointer or ""))
        path = ("steps", name, "base", "inline")
        return source_shape(
            payload,
            ValueSource("step", name, pointer(path[2:]), step=name),
            source_at(self.variant_sources, path),
        )

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
            if isinstance(base, DocumentReference):
                documents = self.documents(name)
                payload, reference = documents.base(base)
                origins = documents.source(reference, payload) if self.track_sources else None
            else:
                payload = self.read_base(name, base)
                origins = self.base_origins(name, base, payload) if self.track_sources else None
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
                else _pack(payload, "payload", name),
                origins=origins,
            )
        payload = parent.payload
        written = parent.written
        origins = parent.origins
        for i, layer in enumerate(step.deltas):
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
            if origins is not None:
                for j, patch in enumerate(patches):
                    path = ("deltas", str(i), "patches", str(j), "value")
                    source = ValueSource("step", name, pointer(path), step=name)
                    override = source_at(self.variant_sources, ("steps", name, *path))
                    patch_source(origins, patch, source_shape(patch.value, source, override))
        try:
            sweep = inherit_sweep(
                parent.sweep, step.sweep, author=name if self.track_sources else None
            )
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
            origins,
        )

    def read_base(self, step: str, base: InlineBase | RunBase | ExternalBase) -> JsonValue:
        if isinstance(base, InlineBase):
            return base.inline
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
            return _select(payload, json_pointer_parts(base.pointer))
        except KeyError as error:
            raise _Problem(
                ResolutionCode.BAD_POINTER, "base.pointer", base.pointer or "", step=step
            ) from error

    def select_row(self, child: str, base: StepBase, parent: _StepState) -> _StepState:
        assert base.row is not None
        if parent.sweep is not None and parent.sweep.rows.get(base.row) is not None:
            try:
                row = authored_row(parent.sweep, base.row)
            except SweepExpansionError as error:
                raise _Problem(
                    ResolutionCode.INVALID_SWEEP, error.path, error.detail, step=base.step
                ) from error
            if row is None:
                raise _Problem(ResolutionCode.UNKNOWN_ROW, "base.row", base.row, step=child)
            state = self.apply_row(base.step, parent, row)
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
        documents = self.documents(base.step)
        origins = (
            documents.inline_sources(state.payload, state.origins, "payload")
            if state.origins is not None
            else source_shape(
                selected.payload, ValueSource("upstream", base.step, step=base.step, row=base.row)
            )
            if self.track_sources
            else None
        )
        return _StepState(selected.payload, inputs=selected.inputs, origins=origins)

    def row_source(self, author: str, row: ExpandedRow, path: str) -> tuple[ValueSource, Any]:
        kind, *ids = row.patch_origins[path]
        match kind:
            case "sweep_value":
                axis, value = ids
                sweep = self.project.steps[author].sweep
                authored_axis = sweep.axes[axis] if sweep is not None else None
                axis_path = ("sweep", "axes", axis)
                match authored_axis:
                    case ListedAxis(values=values):
                        identity = sweep_value_identity_bytes(row.coordinates[axis].value)
                        index = next(
                            i
                            for i, item in enumerate(values)
                            if sweep_value_identity_bytes(item) == identity
                        )
                        target = (*axis_path, "values", str(index))
                    case SampledAxis():
                        target = (*axis_path, "draws", value)
                    case RangeAxis():
                        target = (*axis_path, "range")
                    case _:
                        target = (*axis_path, "values", value, "patches", path, "value")
                source = ValueSource(
                    "sweep_value", value, pointer(target), step=author, axis=axis, value=value
                )
                if isinstance(authored_axis, (ListedAxis, RangeAxis, SampledAxis)):
                    target_override = source_at(self.variant_sources, ("steps", author, *target))
                    path_override = source_at(
                        self.variant_sources, ("steps", author, *axis_path, "path")
                    )
                    return source, first_source(target_override) or path_override
            case "exception":
                (exception,) = ids
                target = ("sweep", "exceptions", exception, "patches", path, "value")
                source = ValueSource("exception", exception, pointer(target), step=author)
            case _:
                (row_id,) = ids
                target = ("sweep", "rows", row_id, "patches", path, "value")
                source = ValueSource("step", author, pointer(target), step=author)
        return source, source_at(self.variant_sources, ("steps", author, *target))

    def apply_row(
        self,
        name: str,
        state: _StepState,
        row: ExpandedRow,
    ) -> _StepState:
        payload = deepcopy(state.payload)
        origins = deepcopy(state.origins)
        for patch in sweep_patches(row.patches):
            try:
                if not isinstance(payload, dict):
                    raise ValueError("a root document reference is atomic")
                if patch.op != "remove":
                    patch = patch.model_copy(update={"value": _pack(patch.value, patch.path, name)})
                apply_override_patch_in_place(payload, patch)
                if origins is not None:
                    author = row.patch_authors.get(patch.path, name)
                    source, override = self.row_source(author, row, patch.path)
                    patch_source(origins, patch, source_shape(patch.value, source, override))
            except ValueError as error:
                raise _Problem(
                    ResolutionCode.INVALID_SWEEP, patch.path, str(error), step=name, layer=row.id
                ) from error
        return _StepState(payload, inputs=state.inputs, requires=state.requires, origins=origins)

    def finish(self, name: str, state: _StepState, *, check_requires: bool = True) -> ResolvedStep:
        documents = self.documents(name)
        payload = documents.inline(state.payload, "payload")
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
            (
                dict(
                    source_leaves(documents.inline_sources(state.payload, state.origins, "payload"))
                )
                if state.origins is not None
                else None
            ),
        )


@dataclass(frozen=True, eq=False)
class StepComposition:
    """Composed authored state, independent of the project used to resolve rows."""

    name: str
    _state: _StepState

    @cached_property
    def row_basis(self) -> str:
        """Hash the ordered payload and required paths before row-local patches."""
        return sha256_bytes(to_json((self._state.payload, self._state.requires)))

    @property
    def sweep(self) -> Sweep | None:
        return self._state.sweep

    @property
    def inputs(self) -> Mapping[str, InputBinding]:
        return self._state.inputs

    @property
    def requires(self) -> tuple[str, ...]:
        return self._state.requires

    def bind(
        self,
        project: Project,
        *,
        readers: BaseReaders = BaseReaders(),
        variant_sources: Any = None,
        read_document: Callable[[str], Document | None] | None = None,
    ) -> StepTemplate:
        return StepTemplate(
            self, _Resolver(
                project, readers, self._state.origins is not None, variant_sources, read_document
            )
        )


@dataclass(frozen=True)
class StepTemplate:
    """Composed step whose requirements are checked on each final row."""

    composition: StepComposition
    _resolver: _Resolver

    @property
    def name(self) -> str:
        return self.composition.name

    @property
    def sweep(self) -> Sweep | None:
        return self.composition.sweep

    @property
    def inputs(self) -> Mapping[str, InputBinding]:
        return self.composition.inputs

    @property
    def requires(self) -> tuple[str, ...]:
        return self.composition.requires

    def resolve(
        self, row: ExpandedRow | None = None, *, include_sources: bool = False
    ) -> ResolvedStep | ProjectDiagnostic:
        try:
            state = self.composition._state
            if not include_sources and state.origins is not None:
                state = replace(state, origins=None)
            if row is not None:
                state = self._resolver.apply_row(self.composition.name, state, row)
            return self._resolver.finish(self.composition.name, state)
        except _Problem as error:
            if row is not None:
                return replace(error.diagnostic, row=row.id)
            return error.diagnostic


def compose_step(
    project: Project,
    step: str,
    *,
    readers: BaseReaders = BaseReaders(),
    track_sources: bool = False,
    variant_sources: Any = None,
    read_document: Callable[[str], Document | None] | None = None,
) -> StepTemplate | ProjectDiagnostic:
    """Compose inherited layers and prepare a step for sweep expansion."""
    resolver = _Resolver(project, readers, track_sources, variant_sources, read_document)
    try:
        return StepTemplate(StepComposition(step, resolver.state(step)), resolver)
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
