"""Static project resolution with explicit readers for external authorities."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import cached_property
from typing import Any, cast

from pydantic import JsonValue, ValidationError
from pydantic_core import to_json, to_jsonable_python

from feedbax.contracts.base import sha256_bytes
from feedbax.contracts.figure_document import FIGURE_DOCUMENT_SCHEMA, is_figure_document_pin
from feedbax.contracts.matrix_core import apply_override_patch_copy
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.model_attachment import MODEL_ATTACHMENTS, SetField
from feedbax.contracts.project import (
    Document,
    DocumentOverrides,
    DocumentReference,
    ExternalBase,
    InlineBase,
    Payload,
    Project,
    RunBase,
    SetProjectValue,
    StepBase,
    StepKind,
    apply_document_edit,
    apply_document_edits,
    document_edit_ops,
    document_edit_value,
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
    model_field_target,
    sweep_patch,
    sweep_value_identity_bytes,
)
from feedbax.contracts.run_matrix import CompositionDeltaError, apply_composition_deltas
from feedbax.contracts.model_address import ModelAddress
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
    element: str | None = None
    document_path: str | None = None
    model_address: ModelAddress | None = None


@dataclass(frozen=True)
class ResolvedStep:
    name: str
    kind: StepKind
    payload: Payload
    sweep: Sweep | None
    inputs: dict[str, InputBinding]
    sources: dict[str, ValueSource] | None = None
    document_ids: dict[str, str] = field(default_factory=dict)
    document_uses: frozenset[str] = frozenset()
    document_pointers: dict[str, str] = field(default_factory=dict)
    document_problems: tuple[ProjectDiagnostic, ...] = ()


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
    External bytes are checked against the authored pin before parsing.
    Step bases and named Analyze figures use separate read-only authorities.
    """

    run: Callable[[str], Payload | None] | None = None
    external: Callable[[str], bytes | None] | None = None
    row: Callable[[ResolvedStep, str], SelectedRow | None] | None = None
    figure_external: Callable[[str], bytes | None] | None = None


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
        document_path: str | None = None,
        model_address: ModelAddress | None = None,
    ):
        self.diagnostic = ProjectDiagnostic(
            code, path, detail, step, layer, document, document_path=document_path, model_address=model_address,
        )
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


def _read_external(
    base: ExternalBase, external: Callable[[str], bytes | None] | None, path: str,
    *, step: str | None, document: str | None = None,
) -> JsonValue:
    def problem(code: ResolutionCode, field: str, detail: str) -> _Problem:
        return _Problem(
            code, f"{path}.{field}", detail, step=step, document=document,
            document_path=f"payload.{field}" if document is not None else None,
        )

    try:
        data = external(base.path) if external is not None else None
    except OSError as error:
        raise problem(ResolutionCode.MISSING_AUTHORITY, "path", str(error)) from error
    if data is None:
        raise problem(ResolutionCode.MISSING_AUTHORITY, "path", base.path)
    observed = sha256_bytes(data)
    if observed != base.sha256:
        raise problem(
            ResolutionCode.BAD_EXTERNAL_CONTENT, "sha256", f"expected {base.sha256}, got {observed}",
        )
    try:
        payload = strict_json_loads(data, ref=base.path)
    except (ValueError, UnicodeError) as error:
        raise problem(ResolutionCode.BAD_EXTERNAL_CONTENT, "path", str(error)) from error
    try:
        return _select(payload, json_pointer_parts(base.pointer))
    except KeyError as error:
        raise problem(ResolutionCode.BAD_POINTER, "pointer", base.pointer or "") from error


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


def _figure_slot(path: str) -> str | None:
    parts = path.split(".")
    if len(parts) == 4 and parts[:2] == ["payload", "views"] and parts[3] == "figure":
        return parts[2]
    return None


class _Documents:
    def __init__(
        self,
        project: Project,
        step: str | None = None,
        variant_sources: Any = None,
        read_document: Callable[[str], Document | None] | None = None,
        overrides: Mapping[str, DocumentOverrides] | None = None,
        *,
        external: Callable[[str], bytes | None] | None = None,
        partial_figures: bool = False,
        row_patches: Mapping[str, tuple[_DocumentRowPatch, ...]] | None = None,
    ):
        self.step = step
        self.variant_sources = variant_sources
        self.read_document = project.documents.get if read_document is None else read_document
        self.overrides = overrides or {}
        self.row_patches = row_patches or {}
        self.used: set[str] = set()
        self.edited: dict[str, JsonValue] = {}
        self.edited_sources: dict[str, Any] = {}
        self.external = external
        self.pinned: dict[str, tuple[ExternalBase, JsonValue]] = {}
        self.partial_figures = partial_figures
        self.figure_problems: dict[str, ProjectDiagnostic] = {}

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
        self.used.add(reference.document)
        payload = document.payload
        edits = self.overrides.get(reference.document)
        patches = self.row_patches.get(reference.document, ())
        if edits or patches:
            if reference.document not in self.edited:
                draft = deepcopy(payload)
                try:
                    if not isinstance(draft, dict):
                        raise ValueError("Document differences require an object payload")
                    if edits:
                        apply_document_edits(draft, edits)
                except ValueError as error:
                    raise _Problem(
                        ResolutionCode.INVALID_DELTA, path, str(error), step=self.step, document=reference.document
                    ) from error
                for item in patches:
                    try:
                        draft = apply_override_patch_copy(draft, item.patch)
                    except ValueError as error:
                        raise _Problem(
                            ResolutionCode.INVALID_SWEEP, item.patch.path, str(error),
                            step=self.step, document=reference.document,
                        ) from error
                self.edited[reference.document] = draft
            payload = self.edited[reference.document]
        if (
            document.schema_id == FIGURE_DOCUMENT_SCHEMA
            and is_figure_document_pin(payload)
            and (self.step is None or _figure_slot(path) is not None)
        ):
            name = reference.document
            if name not in self.pinned:
                try:
                    pin = ExternalBase.model_validate(payload)
                except ValidationError as error:
                    issue = error.errors(include_url=False, include_input=False)[0]
                    field_path = ".".join(map(str, issue["loc"]))
                    raise _Problem(
                        ResolutionCode.BAD_REFERENCE, f"{path}.{field_path}", issue["msg"],
                        step=self.step, document=name, document_path=f"payload.{field_path}",
                    ) from error
                self.pinned[name] = (
                    pin, _read_external(pin, self.external, path, step=self.step, document=name),
                )
            payload = self.pinned[name][1]
        try:
            return _select(payload, json_pointer_parts(reference.pointer))
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
        if reference.document in self.pinned:
            pin, _ = self.pinned[reference.document]
            return source_shape(
                value, ValueSource("document", pin.path, pointer((*json_pointer_parts(pin.pointer), *parts))),
            )
        name = reference.document
        edits = self.overrides.get(name)
        patches = self.row_patches.get(name, ())
        variant = source_at(self.variant_sources, ("documents", name, "payload"))
        if not edits and not patches:
            return source_shape(
                value, ValueSource("document", name, pointer(parts)), source_at(variant, parts),
            )
        if name not in self.edited_sources:
            document = self.read_document(name)
            assert document is not None and self.step is not None
            shape = source_shape(document.payload, ValueSource("document", name), variant)
            for index, edit in enumerate(document_edit_ops(edits) if edits else ()):
                location = (edit.path,) if isinstance(edits, dict) else (str(index),)
                if not isinstance(edits, dict) and isinstance(edit, SetProjectValue):
                    location += ("value",)
                source = ValueSource(
                    "step", self.step, pointer(("document_overrides", name, *location)), step=self.step,
                )
                apply_document_edit(shape, edit, source_shape(document_edit_value(edit), source))
            for item in patches:
                assert item.source is not None
                shape = patch_source(shape, item.patch, source_shape(item.patch.value, item.source, item.override))
            self.edited_sources[name] = shape
        return source_at(self.edited_sources[name], parts)

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

    def source_leaves(
        self,
        value: Any,
        shape: Any,
        path: str,
        output: str = "",
        active: tuple[tuple[str, str], ...] = (),
    ) -> Iterator[tuple[str, ValueSource]]:
        if path in self.figure_problems:
            return
        if isinstance(value, DocumentReference):
            selected, active = self._follow(value, path, active)
            yield from self.source_leaves(
                selected, self.source(value, selected), path, output, active
            )
        elif isinstance(value, dict) and value:
            for key, item in value.items():
                yield from self.source_leaves(
                    item, source_at(shape, (key,)), f"{path}.{key}",
                    output + pointer((key,)), active,
                )
        elif isinstance(value, list) and value:
            for i, item in enumerate(value):
                index = str(i)
                yield from self.source_leaves(
                    item, source_at(shape, (index,)), f"{path}.{index}", f"{output}/{index}", active
                )
        else:
            container = (isinstance(value, dict) and isinstance(shape, SourceObject)
                         or isinstance(value, list) and isinstance(shape, SourceList))
            source = shape.source if container else first_source(shape)
            if source is not None:
                yield output, source

    def inline(
        self,
        value: Any,
        path: str,
        active: tuple[tuple[str, str], ...] = (),
        *,
        document_ids: dict[str, str] | None = None,
        document_pointers: dict[str, str] | None = None,
    ) -> JsonValue:
        if isinstance(value, DocumentReference):
            try:
                selected, active = self._follow(value, path, active)
            except _Problem as error:
                element = _figure_slot(path)
                if not self.partial_figures or element is None:
                    raise
                problem = error.diagnostic
                self.figure_problems[path] = replace(problem, element=element)
                if document_ids is not None and problem.document is not None:
                    document_ids[path] = problem.document
                return None
            if document_ids is not None:
                document_ids[path] = value.document
            if document_pointers is not None:
                if value.pointer:
                    document_pointers[path] = value.pointer
                else:
                    document_pointers.pop(path, None)
            return self.inline(
                selected, path, active, document_ids=document_ids, document_pointers=document_pointers,
            )
        if isinstance(value, dict):
            return {
                key: self.inline(
                    item, f"{path}.{key}", active,
                    document_ids=document_ids, document_pointers=document_pointers,
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                self.inline(
                    item, f"{path}.{i}", active, document_ids=document_ids, document_pointers=document_pointers,
                )
                for i, item in enumerate(value)
            ]
        return cast(JsonValue, value)


@dataclass(frozen=True)
class _DocumentRowPatch:
    patch: OverridePatch
    source: ValueSource | None
    override: Any = None


@dataclass
class _StepState:
    payload: dict[str, Any] | DocumentReference
    sweep: Sweep | None = None
    inputs: dict[str, InputBinding] = field(default_factory=dict)
    requires: tuple[str, ...] = ()
    written: set[str] = field(default_factory=set)
    origins: Any = None
    document_uses: frozenset[str] = frozenset()
    root_document: str | None = None
    row_document_patches: Mapping[str, tuple[_DocumentRowPatch, ...]] = field(default_factory=dict)


class _Resolver:
    def __init__(
        self,
        project: Project,
        readers: BaseReaders,
        track_sources: bool = False,
        variant_sources: Any = None,
        read_document: Callable[[str], Document | None] | None = None,
        inline_origin: Callable[[Payload, ValueSource], Any] = source_shape,
    ):
        self.project = project
        self.readers = readers
        self.track_sources = track_sources
        self.variant_sources = variant_sources
        self.read_document = read_document
        self.inline_origin = inline_origin

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
        source = ValueSource("step", name, pointer(path[2:]), step=name)
        override = source_at(self.variant_sources, path)
        return (
            self.inline_origin(payload, source)
            if override is None else source_shape(payload, source, override)
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
            root_document = None
            document_uses = frozenset()
            if isinstance(base, DocumentReference):
                documents = self.documents(name)
                payload, reference = documents.base(base)
                origins = documents.source(reference, payload) if self.track_sources else None
                root_document = reference.document
                document_uses = frozenset(documents.used)
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
                document_uses=document_uses,
                root_document=root_document,
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
                    origins = patch_source(origins, patch, source_shape(patch.value, source, override))
        if step.model_attachments:
            batches = {name: [item.model_dump(mode="json", exclude_unset=True) for item in records]
                       for name, records in step.model_attachments.items()}
            inherited = payload.get(MODEL_ATTACHMENTS, {})
            if not isinstance(inherited, dict):
                raise _Problem(
                    ResolutionCode.INVALID_PAYLOAD, MODEL_ATTACHMENTS,
                    "Expected named batches of Model attachment differences", step=name,
                )
            payload = payload | {MODEL_ATTACHMENTS: inherited | batches}
            if origins is not None:
                if MODEL_ATTACHMENTS not in origins:
                    origins = patch_source(origins, OverridePatch(
                        path=MODEL_ATTACHMENTS, op="add", value={},
                    ), source_shape({}, None))
                for batch, records in batches.items():
                    source = ValueSource("step", name, pointer((MODEL_ATTACHMENTS, batch)), step=name)
                    patch = OverridePatch(
                        path=f"{MODEL_ATTACHMENTS}.{batch}",
                        op="replace" if batch in inherited else "add", value=records,
                    )
                    origins = patch_source(origins, patch, source_shape(
                        records, source, source_at(self.variant_sources, ("steps", name, MODEL_ATTACHMENTS, batch)),
                    ))
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
            parent.document_uses,
            parent.root_document,
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
        return _read_external(base, self.readers.external, "base", step=step)

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
                document_uses=parent_step.document_uses,
                root_document=parent.root_document,
            )
        else:
            raise _Problem(ResolutionCode.UNKNOWN_ROW, "base.row", base.row, step=child)
        documents = self.row_documents(base.step, state)
        selected = self.finish(base.step, state, documents=documents)
        origins = (
            documents.inline_sources(state.payload, state.origins, "payload")
            if state.origins is not None
            else source_shape(
                selected.payload, ValueSource("upstream", base.step, step=base.step, row=base.row)
            )
            if self.track_sources
            else None
        )
        return _StepState(
            selected.payload, inputs=selected.inputs, origins=origins, document_uses=selected.document_uses,
            root_document=state.root_document,
        )

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
        payload = state.payload
        origins = state.origins
        documents: dict[str, list[_DocumentRowPatch]] = {}
        path = "sweep"
        address = None
        fields = []
        document_uses = state.document_uses
        try:
            for path, write in row.patches.items():
                address = None
                patch = sweep_patch(path, write)
                if isinstance(patch, SetField):
                    address = patch.address
                    if model_field_target(path).step is not None:
                        raise ValueError("A step-scoped Model target belongs in the project sweep")
                    fields.append((path, patch))
                    continue
                parts = patch.path.split(".")
                if len(parts) >= 4 and parts[0] == "documents" and parts[2] == "payload":
                    source, override = (None, None) if origins is None else self.row_source(
                        row.patch_authors.get(patch.path, name), row, patch.path,
                    )
                    documents.setdefault(parts[1], []).append(_DocumentRowPatch(
                        patch.model_copy(update={"path": ".".join(parts[3:])}), source, override,
                    ))
                    continue
                if not isinstance(payload, dict):
                    raise ValueError("a root document reference is atomic")
                trial_path = "trial_documents.python_task"
                if patch.path.startswith(trial_path + ".params."):
                    trials = payload.get("trial_documents")
                    selection = trials.get("python_task") if isinstance(trials, dict) else None
                    if isinstance(selection, DocumentReference):
                        task_documents = _Documents(
                            self.project, name, self.variant_sources, self.read_document,
                            self.project.steps[name].document_overrides,
                        )
                        resolved = task_documents.inline(selection, "payload." + trial_path)
                        if not isinstance(resolved, dict):
                            raise ValueError("The Python task selection must resolve to an object")
                        document_uses = document_uses.union(task_documents.used)
                        payload = payload | {"trial_documents": trials | {"python_task": resolved}}
                        if origins is not None:
                            override = OverridePatch(path=trial_path, op="replace", value={})
                            origins = patch_source(
                                origins, override,
                                task_documents.source(selection, resolved),
                            )
                if patch.path.startswith(MODEL_ATTACHMENTS + ".") and MODEL_ATTACHMENTS not in payload:
                    payload = payload | {MODEL_ATTACHMENTS: {}}
                    if origins is not None:
                        origins = patch_source(origins, OverridePatch(
                            path=MODEL_ATTACHMENTS, op="add", value={},
                        ), source_shape({}, None))
                if patch.op != "remove":
                    patch = patch.model_copy(update={"value": _pack(patch.value, patch.path, name)})
                payload = apply_override_patch_copy(payload, patch)
                if origins is not None:
                    author = row.patch_authors.get(patch.path, name)
                    source, override = self.row_source(author, row, patch.path)
                    origins = patch_source(origins, patch, source_shape(patch.value, source, override))
            if fields:
                if not isinstance(payload, dict):
                    raise ValueError("a root document reference is atomic")
                batches = payload.get(MODEL_ATTACHMENTS, {})
                if not isinstance(batches, dict):
                    raise ValueError("Model differences must be named batches")
                batch = "row-fields"
                while batch in batches:
                    batch += "-"
                records = [patch.model_dump(mode="json") for _, patch in fields]
                if origins is not None:
                    if MODEL_ATTACHMENTS not in payload:
                        origins = patch_source(origins, OverridePatch(
                            path=MODEL_ATTACHMENTS, op="add", value={},
                        ), source_shape({}, None))
                    sources = []
                    for path, patch in fields:
                        author = row.patch_authors.get(path, name)
                        sources.append(source_shape(patch.model_dump(mode="json"), *self.row_source(author, row, path)))
                    origins = patch_source(origins, OverridePatch(
                        path=f"{MODEL_ATTACHMENTS}.{batch}", op="add", value=records,
                    ), SourceList(None, iter(sources)))
                payload = payload | {MODEL_ATTACHMENTS: batches | {batch: records}}
        except ValueError as error:
            raise _Problem(
                ResolutionCode.INVALID_SWEEP, path, str(error), step=name, layer=row.id,
                document=address.document if address is not None else None, model_address=address,
            ) from error
        return _StepState(
            payload, inputs=state.inputs, requires=state.requires, origins=origins, document_uses=document_uses,
            root_document=state.root_document, row_document_patches={
                document: tuple(patches) for document, patches in documents.items()
            },
        )

    def row_documents(self, name: str, state: _StepState, *, partial_figures: bool = False) -> _Documents:
        return _Documents(
            self.project, name, self.variant_sources, self.read_document, self.project.steps[name].document_overrides,
            external=self.readers.figure_external if self.project.steps[name].kind == "analyze" else None,
            partial_figures=partial_figures,
            row_patches=state.row_document_patches,
        )

    def finish(
        self, name: str, state: _StepState, *, check_requires: bool = True, partial_figures: bool = False,
        documents: _Documents | None = None,
    ) -> ResolvedStep:
        if documents is None:
            documents = self.row_documents(name, state, partial_figures=partial_figures)
        document_ids = {"payload": state.root_document} if state.root_document is not None else {}
        document_pointers: dict[str, str] = {}
        payload = documents.inline(
            state.payload, "payload", document_ids=document_ids, document_pointers=document_pointers,
        )
        if unused := state.row_document_patches.keys() - documents.used:
            document = min(unused)
            raise _Problem(
                ResolutionCode.INVALID_SWEEP, f"documents.{document}",
                "The row does not read this named document", step=name, document=document,
            )
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
                    documents.source_leaves(state.payload, state.origins, "payload")
                )
                if state.origins is not None
                else None
            ),
            document_ids,
            state.document_uses.union(documents.used),
            document_pointers,
            tuple(documents.figure_problems.values()),
        )


@dataclass(frozen=True, eq=False)
class StepComposition:
    """Composed authored state, independent of the project used to resolve rows."""

    name: str
    _state: _StepState
    _document_overrides: Mapping[str, DocumentOverrides]

    @cached_property
    def row_basis(self) -> str:
        """Hash the ordered payload and required paths before row-local patches."""
        return sha256_bytes(to_json((
            self._state.payload, self._state.requires, sorted(self._state.document_uses), self._document_overrides,
        )))

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

    def authored_payload(self) -> Payload:
        """Read composed values before dereferencing their named documents."""
        payload = self.composition._state.payload
        if not isinstance(payload, dict):
            raise ValueError("A root document reference is not an editable payload.")
        return to_jsonable_python(payload, by_alias=True)

    def resolve(
        self, row: ExpandedRow | None = None, *, include_sources: bool = False, partial_figures: bool = False,
    ) -> ResolvedStep | ProjectDiagnostic:
        try:
            state = self.composition._state
            if not include_sources and state.origins is not None:
                state = replace(state, origins=None)
            if row is not None:
                state = self._resolver.apply_row(self.composition.name, state, row)
            return self._resolver.finish(self.composition.name, state, partial_figures=partial_figures)
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
    inline_origin: Callable[[Payload, ValueSource], Any] = source_shape,
) -> StepTemplate | ProjectDiagnostic:
    """Compose inherited layers and prepare a step for sweep expansion."""
    resolver = _Resolver(project, readers, track_sources, variant_sources, read_document, inline_origin)
    try:
        state = resolver.state(step)
        return StepTemplate(StepComposition(step, state, project.steps[step].document_overrides), resolver)
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
    *,
    readers: BaseReaders = BaseReaders(),
) -> tuple[ProjectDiagnostic, ...]:
    """Validate inlined document payloads using the caller's explicit schema authorities."""
    documents = _Documents(project, external=readers.figure_external)
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
