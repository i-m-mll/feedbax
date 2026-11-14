"""Read and edit authored Evaluate attachment differences in the project history."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.model_attachment import (
    MODEL_ATTACHMENTS,
    MODEL_DIFFERENCE_BATCHES,
    AddModelAttachment,
    ChangeModelAttachment,
    RemoveModelAttachment,
    SetField,
)
from feedbax.contracts.model_edit import (
    EvaluateAttachmentAxis,
    EvaluateAttachmentEdit,
    EvaluateAttachmentRows,
    EvaluateAttachmentState,
    EvaluateAttachmentVariant,
    ModelEditProblem,
)
from feedbax.compiler.model_attachment import resolve_effective_model
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.project import Project
from feedbax.contracts.project_plan import RunKey, compile_project
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.project_sweep import model_field_target
from feedbax.contracts.studio_project import AuthoredFileAddress, SetProjectValue, StudioProjectFilePatch
from feedbax.web.services.model_projection import selected_model
from feedbax.web.services.evaluate_sources import evaluate_subject, evaluate_value_source


class EvaluateAttachmentRefused(ValueError):
    def __init__(self, problem: ModelEditProblem):
        super().__init__(problem.detail)
        self.problem = problem


def _key(body):
    return RunKey(body.run.variant, body.run.step, body.run.row)


@dataclass
class _Scope:
    address: AuthoredFileAddress
    field: str
    target: dict
    original: object
    batch: str = "step"
    patch: str | None = None
    exception: tuple[dict, str] | None = None

    def records(self):
        if self.patch is None:
            return self.target.get(self.batch, [])
        write = self.target.get(self.patch)
        if write is None:
            return []
        return write["value"]

    def store(self, records):
        if self.patch is None and records:
            self.target[self.batch] = records
        elif self.patch is None:
            self.target.pop(self.batch, None)
        elif records:
            self.target[self.patch] = {"op": "add", "value": records}
        else:
            self.target.pop(self.patch, None)
            if self.exception is not None and not self.target:
                exceptions, name = self.exception
                exceptions.pop(name, None)

    def edit(self, raw, revision, address):
        return EvaluateAttachmentEdit(
            revision=revision,
            changed=address,
            file=StudioProjectFilePatch(
                address=self.address,
                patches=[SetProjectValue(
                    op="replace" if self.field in self.original.model_fields_set else "add",
                    path=f"/{self.field}",
                    value=raw[self.field],
                )],
            ),
        )


def _batch_id(kind, *identity):
    return kind + "-" + sha256(canonical_json_v2_bytes(list(identity))).hexdigest()


def _scope(project, data, body):
    key, scope = _key(body), body.scope
    authored = project.steps.get(key.step)
    if authored is None or authored.kind != "evaluate":
        raise ValueError("The selected step is unavailable or is not an Evaluate step")
    step = data["steps"][key.step]
    address = AuthoredFileAddress(kind="step", id=key.step)
    if isinstance(scope, EvaluateAttachmentVariant):
        if key.variant != scope.variant:
            raise ValueError("The selected run is outside the requested variant")
        original = project.sweep.rows.get(scope.variant) if project.sweep is not None else None
        if original is None:
            raise ValueError("The selected variant has no authored row")
        row = data["sweep"]["rows"][scope.variant]
        patches = row.setdefault("patches", {})
        batch = _batch_id("variant", scope.variant)
        path = _batch_path(patches, f"steps.{key.step}.{MODEL_ATTACHMENTS}", batch)
        return _Scope(
            AuthoredFileAddress(kind="variant", id=scope.variant), "patches", patches, original,
            patch=path,
        ), row
    if isinstance(scope, EvaluateAttachmentAxis):
        axis = step["sweep"]["axes"][scope.axis]
        if not isinstance(axis.get("values"), dict):
            raise ValueError("The selected group requires a named sweep value")
        value = axis["values"].get(scope.value)
        if value is None:
            raise ValueError("The named sweep value is unavailable in this step")
        patches = value.setdefault("patches", {})
        batch = _batch_id("axis", scope.axis, scope.value)
        return _Scope(address, "sweep", patches, authored, patch=_batch_path(patches, MODEL_ATTACHMENTS, batch)), step
    if isinstance(scope, EvaluateAttachmentRows):
        if key.row not in scope.rows:
            raise ValueError("The selected run is outside the requested rows")
        if step.get("sweep") is None:
            step["sweep"] = {}
        exceptions = step["sweep"].setdefault("exceptions", {})
        rows = sorted(set(scope.rows))
        batch = _batch_id("rows", rows)
        matches = [
            name for name, exception in exceptions.items()
            if exception is not None and exception.get("where") == {"rows": rows}
            and any(
                path.startswith(MODEL_ATTACHMENTS + ".") or _addressed_field(path)
                for path in exception.get("patches", {})
            )
        ]
        name = matches[0] if len(matches) == 1 else batch
        exception = exceptions.setdefault(name, {"where": {"rows": rows}, "patches": {}})
        if exception is None:
            raise ValueError("The selected row exception is disabled")
        patches = exception.setdefault("patches", {})
        return _Scope(
            address, "sweep", patches, authored, patch=_batch_path(patches, MODEL_ATTACHMENTS, batch),
            exception=(exceptions, name),
        ), step
    batch = _batch_id("step", key.step)
    batches = step.setdefault(MODEL_ATTACHMENTS, {})
    if batch not in batches and len(batches) == 1:
        batch = next(iter(batches))
    return _Scope(
        address, MODEL_ATTACHMENTS, batches, authored, batch=batch,
    ), step


def _addressed_field(path: str) -> bool:
    try:
        return model_field_target(path) is not None
    except ValueError:
        return False


def _batch_path(patches, parent, default):
    path = f"{parent}.{default}"
    siblings = [key for key in patches if key.rpartition(".")[0] == parent]
    return siblings[0] if path not in patches and len(siblings) == 1 else path


def _local_records(scope):
    value = scope.records()
    return MODEL_DIFFERENCE_BATCHES.validate_python({"local": value})["local"]


def _apply(records, difference):
    address = difference.address
    if isinstance(difference, AddModelAttachment):
        return [*records, difference]
    for index, record in enumerate(records):
        if record.address != address:
            continue
        if isinstance(difference, SetField):
            return [*records[:index], difference, *records[index + 1:]]
        if isinstance(record, RemoveModelAttachment):
            raise ValueError("The addressed attachment is already removed in this scope")
        if isinstance(record, AddModelAttachment):
            if isinstance(difference, RemoveModelAttachment):
                return records[:index] + records[index + 1:]
            if record.attachment.kind != difference.attachment.kind:
                raise ValueError("Changing an attachment must preserve its kind")
            changes = {"attachment": difference.attachment}
            if "after" in difference.model_fields_set:
                changes["after"] = difference.after
            difference = record.model_copy(update=changes)
        elif (
            isinstance(difference, ChangeModelAttachment) and "after" not in difference.model_fields_set
            and "after" in record.model_fields_set
        ):
            difference = difference.model_copy(update={"after": record.after})
        return [*records[:index], difference, *records[index + 1:]]
    return [*records, difference]


def read_evaluate_attachments(folders, project_id, body, components=None):
    snapshot = folders.read(project_id, revision=body.revision)
    key = _key(body)
    selected = folders.read_plan(project_id, revision=body.revision, source_runs=(key,)).plan
    definition = selected.selected_definitions.get(key)
    data = snapshot.project.model_dump(mode="json", by_alias=True, exclude_unset=True)
    scope, _ = _scope(snapshot.project, data, body)
    _selected_rows(selected, body)
    effective = MODEL_DIFFERENCE_BATCHES.validate_python(
        definition.payload.get(MODEL_ATTACHMENTS, {}) if definition else {},
    )
    differences = tuple(record for batch in effective.values() for record in batch)
    resolved = None
    source = None
    field_sources = {}
    if upstream := evaluate_subject(definition):
        producer, provenance = upstream
        source = evaluate_value_source(provenance)
        train_plan = folders.read_plan(project_id, revision=body.revision, source_runs=(producer,)).plan
        train = train_plan.selected_definitions.get(producer)
        if train is not None:
            document = train.document_ids.get("payload.trial_documents.model") or train.document_ids.get("payload.graph")
            if document and (model := selected_model(train, document)) is not None:
                resolved = resolve_effective_model(
                    document, ModelDocument.model_validate(model), components or folders.components,
                    differences, functions=folders.functions,
                )
                for batch, records in effective.items():
                    for index, record in enumerate(records):
                        site = ("payload", MODEL_ATTACHMENTS, batch, str(index))
                        origin = definition.sources.get(pointer((*site, "value"))) or definition.sources.get(
                            pointer((*site, "op")),
                        )
                        if origin is not None:
                            field_sources[record.address.to_text()] = evaluate_value_source(origin, evaluate_step=key.step)
    return EvaluateAttachmentState(
        revision=body.revision,
        records=tuple(record for record in _local_records(scope) if not isinstance(record, SetField)),
        effective=tuple(
            record for batch in effective.values() for record in batch if not isinstance(record, SetField)
        ),
        resolved_model=resolved,
        source=source,
        field_sources=field_sources,
    )


def _selected_rows(plan, body):
    key, scope = _key(body), body.scope
    rows = {row.key: row for row in plan.rows if row.key.step == key.step}
    if isinstance(scope, EvaluateAttachmentAxis):
        selected = {
            item for item, row in rows.items()
            if (coordinate := row.coordinates.get(scope.axis)) is not None and coordinate.value_id == scope.value
        }
    elif isinstance(scope, EvaluateAttachmentRows):
        if any(item.variant != key.variant for item in rows):
            raise ValueError("A row exception must have a single project variant")
        selected = {item for item in rows if item.row in scope.rows}
        if {item.row for item in selected} != set(scope.rows):
            raise ValueError("The row scope contains an unknown Evaluate row")
    elif isinstance(scope, EvaluateAttachmentVariant):
        selected = {item for item in rows if item.variant == scope.variant}
    else:
        selected = set(rows)
    if key not in selected:
        raise ValueError("The selected run is outside the requested edit scope")
    return selected


def prepare_evaluate_attachment(folders, project_id, body, components):
    snapshot = folders.read(project_id, revision=body.revision)
    settled = folders.read_plan(project_id, revision=body.revision)
    plan = settled.plan
    address = body.difference.address
    try:
        selected = _selected_rows(plan, body)
        data = snapshot.project.model_dump(mode="json", by_alias=True, exclude_unset=True)
        scope, raw = _scope(snapshot.project, data, body)
        difference = body.difference
        if isinstance(difference, SetField) and any(key.variant != body.run.variant for key in selected):
            raise ValueError("The Model field scope includes another variant")
        if isinstance(difference, SetField) and scope.patch is not None:
            path = difference.address.to_text()
            if isinstance(body.scope, EvaluateAttachmentVariant):
                path = f"steps.{body.run.step}.{path}"
            scope.target[path] = {"value": difference.value}
        else:
            records = _apply(_local_records(scope), difference)
            scope.store([record.model_dump(mode="json", exclude_unset=True) for record in records])
        candidate = compile_project(
            Project.model_validate(data), readers=settled.readers,
            operations=folders.operations, components=components, functions=folders.functions,
            python_tasks=folders.python_tasks,
        )
        previous = {run.key: run.request_id for run in plan.runs}
        current = {run.key: run.request_id for run in candidate.runs}
        if not selected <= {row.key for row in candidate.rows}:
            raise ValueError("The edit must retain every selected Evaluate row")
        if not isinstance(difference, SetField):
            for problem in candidate.diagnostics:
                key = RunKey(problem.variant, problem.step, problem.row)
                if key in selected and (key not in current or problem not in plan.diagnostics):
                    address = problem.model_address or address
                    raise ValueError(problem.detail)
            if not selected <= current.keys():
                raise ValueError("Attachment differences must resolve for every selected Evaluate row")
        outside = (previous.keys() | current.keys()) - selected
        if any(current.get(key) != previous.get(key) for key in outside):
            raise ValueError("Attachment differences must preserve every request outside the selected scope")
        return scope.edit(raw, body.revision, address)
    except (KeyError, ValueError) as error:
        raise EvaluateAttachmentRefused(ModelEditProblem(address=address, detail=str(error))) from error
