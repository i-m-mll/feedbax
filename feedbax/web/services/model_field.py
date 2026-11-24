"""Prepare Model field writes against one saved project revision."""

from collections.abc import Iterator
from dataclasses import dataclass, replace

from pydantic import JsonValue

from feedbax.component_registry import ComponentRegistry
from feedbax.compiler.model_document import model_tau_seconds
from feedbax.contracts.expression_values import TIME, Level, LiteralValue, Problem, ProblemCode, Site, ValueBase, ValueType
from feedbax.contracts.model_address import AttachmentElement, ComponentElement, FieldElement, ModelAddress, PortElement, WireElement
from feedbax.contracts.model_document import (
    MODEL_DOCUMENT_SCHEMA_ID, CompositeComponent, DelayAttachment, InterventionAttachment, LeafComponent, ModelDocument,
)
from feedbax.contracts.model_edit import (
    EvaluateModelRequest, ListedRuns, StepRuns, ModelEditCommand, SetField,
)
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.project import BasedDocument, InlineBase, SetProjectValue
from feedbax.contracts.project_resolution import ProjectDiagnostic, compose_document
from feedbax.planner.project_plan import RunKey
from feedbax.contracts.jobs import StudioPlanReviewSource
from feedbax.contracts.project_sources import ValueSource, pointer
from feedbax.contracts.studio_field import (
    FieldResetEdit, FieldTextEdit, ModelFieldRequest, ModelFieldResponse, ModelFieldView,
)
from feedbax.contracts.studio_project import AuthoredFileAddress, StudioProjectFilePatch
from feedbax.contracts.studio_scope import StoreKind
from feedbax.contracts.expression_registry import FunctionRegistry
from feedbax.contracts.expression_text import print_field
from feedbax.studio.authoring_sweeps import SweepTarget, FieldSweeps
from feedbax.studio.numeric_fields import NumericField, prepare_numeric_field, project_numeric_field
from feedbax.studio.selection import StoredEdit, merged_files
from feedbax.web.services.authored_scope import choose_prepared_edit, selected_edit_runs
from feedbax.web.services.axis_values import axis_values
from feedbax.web.services.evaluate_sources import read_evaluate_model
from feedbax.web.services.model_projection import selected_model
from feedbax.web.services.project_folder import ProjectFolderInvalid, ProjectFolderService
from feedbax.web.services.trial_scope import (
    evaluation_document_write, scoped_file, selected_value, selection_scope, source_value_file, variant_file,
)


@dataclass(frozen=True)
class _Field:
    address: ModelAddress
    numeric: NumericField
    literal: bool = False


def _address(owner: ModelAddress, *path: str) -> ModelAddress:
    return ModelAddress(
        composites=owner.composites,
        element=FieldElement(kind="field", parent=owner.element, path=path),
    )


def _duration(
    owner: ModelAddress, label: str, path: tuple[str, ...], value: int | float, unit: str,
    *, stored: bool = False, literal: bool = False,
) -> _Field:
    address = _address(owner, *path)
    return _Field(address, NumericField(
        address.stored_path(), label, value,
        Site(expected=ValueType(base=ValueBase.REAL, dimensions=TIME), finest=Level.RUN, structural=True),
        stored, _positive, unit, duration=True,
    ), literal=literal)


def _positive(value: int | float) -> int | float:
    if value <= 0:
        raise ValueError("Enter a positive duration.")
    return value


def _attachment_fields(branch: ModelDocument | CompositeComponent, owner: ModelAddress) -> Iterator[_Field]:
    element = owner.element
    assert isinstance(element, AttachmentElement)
    parent = element.parent
    if isinstance(parent, WireElement):
        wire = branch.interior.wires.get(parent.id)
        chain = wire.attachments if wire else ()
    elif isinstance(parent, PortElement):
        end = parent.end
        if end.owner == "component":
            component = branch.interior.components.get(end.component)
            ports = component.attachments if component else None
        else:
            ports = branch.interior.attachments
        sources = end.owner == "inputs" or end.owner == "component" and end.direction == "output"
        chain = (ports.sources if sources else ports.sinks).get(end.port, ()) if ports else ()
    else:
        return
    attachment = next((item for item in chain if item.id == element.id), None)
    if isinstance(attachment, DelayAttachment):
        yield _duration(owner, "Delay", ("delay",), attachment.delay.value, attachment.delay.unit, stored=True)
        values = {"initial": attachment.initial}
    elif isinstance(attachment, InterventionAttachment):
        names = {"clamp": ("low", "high"), "scale": ("factor",)}.get(attachment.operation.lower(), ())
        values = {name: attachment.fields.get(name) for name in names}
    else:
        return
    for name, value in values.items():
        if type(value) not in (int, float):
            continue
        address = _address(owner, *(("initial",) if name == "initial" else ("fields", name)))

        def validate(number: int | float, name=name) -> int | float:
            low = number if name == "low" else values.get("low")
            high = number if name == "high" else values.get("high")
            if type(low) in (int, float) and type(high) in (int, float) and low > high:
                raise ValueError("Clamp low must be at most high.")
            return number

        yield _Field(address, NumericField(
            address.stored_path(), name.capitalize(), value,
            Site(expected=ValueType(base=ValueBase.REAL), finest=Level.RUN, structural=True),
            True, validate,
        ))


def _fields(model: ModelDocument, owner: ModelAddress, components: ComponentRegistry) -> Iterator[_Field]:
    if owner.element.kind == "graph" and not owner.composites:
        address = _address(owner, "step_seconds")
        yield _Field(address, NumericField(
            address.stored_path(), "Step", model.step_seconds,
            Site(expected=ValueType(base=ValueBase.REAL, dimensions=TIME), finest=Level.RUN, structural=True),
            "step_seconds" in model.model_fields_set, _positive, "s",
        ))
        return
    branch: ModelDocument | CompositeComponent = model
    for key in owner.composites:
        child = branch.interior.components.get(key)
        if child is None or child.kind != "composite":
            return
        branch = child
    if isinstance(owner.element, AttachmentElement):
        yield from _attachment_fields(branch, owner)
        return
    if not isinstance(owner.element, ComponentElement):
        return
    component = branch.interior.components.get(owner.element.id)
    if isinstance(component, CompositeComponent):
        if component.interior.physical_domain in {"planar_multibody", "spatial_multibody"}:
            address = _address(owner, "substeps")

            def substeps(value: int | float) -> int:
                if type(value) is not int or value < 1:
                    raise ValueError("Substeps requires a positive integer.")
                return value

            yield _Field(address, NumericField(
                address.stored_path(), "Substeps", component.substeps,
                Site(expected=ValueType(base=ValueBase.INTEGER), finest=Level.RUN, structural=True),
                "substeps" in component.model_fields_set, substeps,
            ))
        return
    if not isinstance(component, LeafComponent) or not (definition := components.get(component.type)):
        return
    defaults = definition.params.default_params
    for schema in definition.param_schema:
        name = schema.name
        if name in {"dt", "control_dt"} or name in component.port_counts or "." in name:
            continue
        base = {"int": ValueBase.INTEGER, "float": ValueBase.REAL}.get(schema.type)
        value = component.fields.get(name, defaults.get(name))
        if component.type == "feedbax.VanillaRNN" and name == "tau":
            if type(value) in (int, float):
                yield _duration(owner, "Tau", ("fields", "tau"), value, "st", stored=name in component.fields)
            elif isinstance(value, dict) and value.get("kind") == "literal" and value.get("unit") in ("st", "s"):
                if type(magnitude := value.get("value")) in (int, float):
                    yield _duration(owner, "Tau", ("fields", "tau"), magnitude, value["unit"], stored=True, literal=True)
            continue
        if base is None or type(value) not in (int, float):
            continue

        def validate(value: int | float, name=name) -> int | float:
            values = {**component.fields, name: value}
            if component.type == "feedbax.VanillaRNN" and "tau" in values:
                values["tau"] = model_tau_seconds(values["tau"], model.step_seconds)
            definition.params.require(values)
            return value

        address = _address(owner, "fields", name)
        yield _Field(address, NumericField(
            address.stored_path(), name.replace("_", " ").capitalize(), value,
            Site(expected=ValueType(base=base), finest=Level.RUN, structural=True),
            name in component.fields and "fields" in component.model_fields_set, validate,
        ))


def _addressed(value: JsonValue) -> list[SetProjectValue]:
    raise AssertionError("An addressed Model field target writes at its patch path.")


def _kept_writes(
    project, step_id: str, writes: list[OverridePatch],
) -> tuple[list[StudioProjectFilePatch], SetField | None] | Problem:
    """What gives ``step_id`` a removed sweep's kept value: its Model field as a document write, the rest in its
    definition."""
    fields = [(ModelAddress.from_text(write.path), write) for write in writes if write.path.startswith("{")]
    if any(write.op == "remove" for _, write in fields):
        raise ProjectFolderInvalid("A sweep value removes a Model field.")
    if len(fields) > 1:
        return Problem(code=ProblemCode.STRUCTURE, reason="This sweep value sets more than one Model field.")
    rest = [write for write in writes if not write.path.startswith("{")]
    files = scoped_file(project, step_id, StepRuns(kind="step"), rest).files if rest else []
    return files, next((SetField(op="set_field", address=address, value=write.value) for address, write in fields), None)


def _reset_text(
    project, restored: str | None, owner: ModelAddress, address: ModelAddress, components: ComponentRegistry,
) -> str | Problem:
    """The text of the field at ``address`` in ``restored``, the document a Reset gives the field its value from."""
    document = compose_document(project.documents.get, restored) if restored is not None else None
    if document is None or isinstance(document, ProjectDiagnostic):
        return Problem(code=ProblemCode.STRUCTURE, reason="This value is not changed from another document.")
    field = next((
        item for item in _fields(ModelDocument.model_validate(document.payload), owner, components)
        if item.address == address
    ), None)
    if field is None:
        return Problem(code=ProblemCode.STRUCTURE, reason=f"{restored} has no value here.")
    return print_field(field.numeric.literal(field.numeric.value), field.numeric.site)


def _held_values(
    plan, keys: tuple[RunKey, ...], document: str, owner: ModelAddress, components: ComponentRegistry,
) -> dict[str, list[int | float]]:
    """The distinct values each field of ``owner`` holds in the Model the runs ``keys`` read as ``document``."""
    held: dict[str, list[int | float]] = {}
    for key in keys:
        definition = plan.selected_definitions.get(key)
        payload = selected_model(definition, document) if definition is not None else None
        if payload is None:
            continue
        for field in _fields(ModelDocument.model_validate(payload), owner, components):
            values = held.setdefault(field.address.to_text(), [])
            if field.numeric.value not in values:
                values.append(field.numeric.value)
    return held


def _field_store(
    folders, project_id, project, plan, body, response: ModelFieldResponse,
    *, kind: StoreKind = "whole_step", label="Whole Step", axis: FieldSweeps | None = None, alternatives=(),
) -> ModelFieldResponse:
    if not response.files and not alternatives:
        return response
    selected = selected_edit_runs(plan, body.run.step, body)
    if not body.rows and axis is not None and len(axis.shown) == 1 and (
        value := getattr(body.edit, "value_id", None)
    ) is not None:
        axis_name, _ = axis.shown[0]
        ids = axis_values(item for item in plan.rows if item.key in selected)
        selected = frozenset(item for item in selected if ids.get((item, axis_name)) == value)
        kind, label = "sweep_value", "Sweep Value"
    choice = choose_prepared_edit(
        folders, project_id, body.revision, project, selected,
        body.model_dump(mode="json", exclude={"store_choice"}), body.store_choice,
        (kind, label, body.address.to_text(), StoredEdit(response.files, tuple(response.created))), alternatives,
    )
    if isinstance(choice, list):
        return ModelFieldResponse(fields=response.fields, store_choices=choice)
    return response.model_copy(update={"files": choice.files, "created": list(choice.created)})


def model_fields(
    folders: ProjectFolderService, project_id: str, body: ModelFieldRequest,
    components: ComponentRegistry, functions: FunctionRegistry,
) -> ModelFieldResponse:
    snapshot = folders.read(project_id, revision=body.revision)
    project = snapshot.project
    document = compose_document(project.documents.get, body.document)
    if isinstance(document, ProjectDiagnostic):
        raise ProjectFolderInvalid(f"The Model document cannot be read: {document.detail}")
    if document.schema_id != MODEL_DOCUMENT_SCHEMA_ID:
        raise ProjectFolderInvalid("The Model document does not exist.")
    model = ModelDocument.model_validate(document.payload)
    step = selected = plan = effective_model = None
    shown = project.documents.get(body.document)
    changes, restored = (shown.differences, shown.base.document) if isinstance(shown, BasedDocument) else ({}, None)
    definitions = ()
    runs = None
    held: dict[str, list[int | float]] = {}
    evaluate = None
    variant = None
    if body.evaluate_scope is not None and body.run is not None:
        evaluate = EvaluateModelRequest(revision=body.revision, run=body.run, scope=body.evaluate_scope)
        key = RunKey(body.run.variant, body.run.step, body.run.row)
        plan = folders.read_plan(project_id, revision=body.revision, source_runs=(key,)).plan
        reading = read_evaluate_model(
            folders, project_id, evaluate, components, selected=plan, model_document=body.document,
        )
        effective_model = reading.state.resolved_model
        changes, restored = reading.differences, reading.shared
        candidate = project.steps[key.step]
        if (isinstance(evaluate.scope, StepRuns) and plan.variants[:1] == (key.variant,)
                and isinstance(candidate.base, InlineBase) and not candidate.deltas):
            selected = next(row for row in plan.rows if row.key == key)
            step = candidate
    elif body.run is not None:
        key = RunKey(body.run.variant, body.run.step, body.run.row)
        scoped = tuple(RunKey(key.variant, key.step, row) for row in body.rows)
        plan = folders.read_plan(project_id, revision=body.revision, source_runs=(key, *scoped)).plan
        definitions = tuple(plan.selected_definitions[item] for item in (scoped or (key,))
                            if item in plan.selected_definitions)
        definition = plan.selected_definitions.get(key)
        if definition is not None and (model_payload := selected_model(definition, body.document)) is not None:
            effective_model = ModelDocument.model_validate(model_payload)
            selected = next((row for row in plan.rows if row.key == key), None)
            if plan.steps[(key.variant, key.step)].kind == "train":
                if project.sweep is None:
                    step = project.steps[key.step]
                    picked = scoped or tuple(
                        row.key for row in plan.rows if (row.key.variant, row.key.step) == (key.variant, key.step)
                    )
                    plan = folders.read_plan(project_id, revision=body.revision, source_runs=(key, *picked)).plan
                    runs = tuple(row.coordinates for row in plan.rows if row.key in picked)
                    held = _held_values(plan, picked, body.document, body.owner, components)
                elif key.variant != plan.variants[0] and project.sweep.rows.get(key.variant) is not None:
                    variant = key.variant
    if evaluate is not None and effective_model is None:
        return ModelFieldResponse()
    effective_fields = {
        field.address.to_text(): field for field in _fields(effective_model, body.owner, components)
    } if effective_model is not None else {}
    authored_fields = {field.address.to_text(): field for field in _fields(model, body.owner, components)}
    visible_fields = effective_fields if evaluate or variant is not None else effective_fields | authored_fields
    fields = []
    targets = {}
    for field_key, field in visible_fields.items():
        numeric = field.numeric
        resolved = effective_fields.get(field_key)
        axis = None
        if step is not None:
            target = SweepTarget(numeric.value, _addressed, field_key)
            axis = FieldSweeps(
                step, target, numeric.label, numeric.key[-1:], integer=numeric.site.expected.base == ValueBase.INTEGER,
                runs=runs,
            )
            if evaluate is not None:
                # The addressed sweep can set fields inherited from registry defaults.
                numeric = replace(numeric, stored=True)
        relative = pointer(field.address.stored_path())
        changed = any(relative == path or relative.startswith(path + "/") for path in changes)
        sources = dict.fromkeys(source for definition in definitions
                                if (source := definition.sources.get(f"/payload/trial_documents/model{relative}")) is not None)
        fields.append(project_numeric_field(
            ModelFieldView, numeric, axis, functions, selected, resolved.numeric.value if resolved else None,
            held=held.get(field_key, ()),
            period=(effective_model or model).step_seconds, scoped=bool(body.rows or body.axes),
            address=field.address,
            sources=[StudioPlanReviewSource.model_validate(source, from_attributes=True) for source in sources],
            changed=changed,
        ))
        targets[field_key] = replace(field, numeric=numeric), axis
    if body.edit is None:
        return ModelFieldResponse(fields=fields)
    if body.address is None or body.address.to_text() not in targets or body.run is None or (
        evaluate is None and variant is None and (step is None or plan is None)
    ):
        raise ProjectFolderInvalid("Choose a supported field of the selected Model.")
    field, axis = targets[body.address.to_text()]
    owner_name = body.document

    step_address = AuthoredFileAddress(kind="step", id=body.run.step)

    def step_file(patches: list[SetProjectValue]) -> StudioProjectFilePatch:
        return StudioProjectFilePatch(address=step_address, patches=patches)

    scope = None
    if (body.rows or body.axes) and evaluate is None:
        key = RunKey(body.run.variant, body.run.step, body.run.row)
        scope = selection_scope(project, plan, key, tuple(body.rows), body.axes)
    edit = body.edit
    if isinstance(edit, FieldResetEdit):
        reset = _reset_text(project, restored, body.owner, field.address, components)
        if isinstance(reset, Problem):
            return ModelFieldResponse(problems=[reset])
        edit = FieldTextEdit(kind="text", text=reset)
    prepared = prepare_numeric_field(
        field.numeric, axis, edit, functions, period=(effective_model or model).step_seconds,
        scoped=scope is not None,
    )
    result = prepared.result
    if isinstance(result, tuple):
        return ModelFieldResponse(problems=list(result))
    if isinstance(result, list):
        kept = _kept_writes(project, body.run.step, prepared.keep) if prepared.keep and step is not None else ([], None)
        if isinstance(kept, Problem):
            return ModelFieldResponse(problems=[kept])
        files, field_value = kept
        response = ModelFieldResponse(
            files=merged_files([*files, step_file(result)]), removed_values=prepared.removed,
            command=ModelEditCommand(operation=field_value) if field_value is not None else None,
        )
        return response if response.command is not None else _field_store(
            folders, project_id, project, plan, body, response, axis=axis,
        )
    if scope is not None and selected_value(project, body.run.step, scope) is None:
        if axis is None or axis.target.patch_path is None:
            return ModelFieldResponse(problems=[Problem(
                code=ProblemCode.STRUCTURE, reason="This field has no writable sweep destination.",
            )])
        write = OverridePatch(op="replace", path=axis.target.patch_path, value=result)
        relative = pointer(field.address.stored_path())
        selected = selected_edit_runs(plan, body.run.step, body)
        alternatives = []
        writable = []
        sources = {}
        for run in selected:
            definition = plan.selected_definitions.get(run)
            source = definition.sources.get(f"/payload/trial_documents/model{relative}") if definition is not None else None
            if isinstance(scope, ListedRuns) and isinstance(source, ValueSource) and source.kind in {
                "sweep_value", "exception", "variant",
            }:
                sources[source] = definition
            else:
                writable.append(run.row)
        for source, definition in sources.items():
            held = selected_model(definition, owner_name)
            for part in field.address.stored_path():
                held = held.get(part) if isinstance(held, dict) else None
            file = source_value_file(project, source, held, result)
            if file is not None:
                kind, label = {
                    "sweep_value": ("sweep_value", "Sweep Value"), "exception": ("row_exception", "Row Exception"),
                    "variant": ("variant", "Variant"),
                }[source.kind]
                alternatives.append((kind, label, f"{source.name}:{source.path}", StoredEdit([file])))
        destination = ListedRuns(kind="rows", rows=tuple(sorted(writable))) if sources and writable else scope
        files = scoped_file(project, body.run.step, destination, [write]).files if writable else []
        return _field_store(
            folders, project_id, project, plan, body, ModelFieldResponse(files=files), kind="other_runs",
            label="Other Runs", alternatives=alternatives,
        )
    value = result
    if isinstance(result, LiteralValue):
        if field.address.element.parent.kind == "attachment":
            if result.unit == "st" and result.value != int(result.value):
                return ModelFieldResponse(problems=[Problem(code=ProblemCode.STRUCTURE, reason="Delay in steps must be a whole number.")])
            value = result.model_dump(mode="json", exclude={"kind"})
        elif result.unit == "st" and field.numeric.unit == "st" and not field.literal and field.numeric.stored:
            value = result.value
        else:
            value = result.model_dump(mode="json")
    difference = SetField(op="set_field", address=field.address, value=value)
    if evaluate is not None:
        return ModelFieldResponse(command=ModelEditCommand(operation=difference))
    if variant is not None:
        field_key = difference.address.to_text()
        path = f"documents.{owner_name}.{field_key}"
        if field_key not in authored_fields or isinstance(difference.address.element.parent, AttachmentElement):
            raise ProjectFolderInvalid("This Model field is not part of the named document.")
        write = OverridePatch(op="replace", path=path, value=value)
        chosen_steps = {body.run.step, *body.chosen_steps}
        alternatives = []
        if chosen_steps < plan.document_users(variant, owner_name):
            narrowed = evaluation_document_write(project, plan, owner_name, [
                SetProjectValue(op="replace", path=pointer(difference.address.stored_path()), value=value),
            ], chosen_steps, "model", variant)
            alternatives.append(("selected_readers", "Selected Readers", variant, narrowed))
        return _field_store(
            folders, project_id, project, plan, body,
            ModelFieldResponse(files=[variant_file(project, variant, [write])]), kind="variant", label="Variant",
            alternatives=alternatives,
        )
    return ModelFieldResponse(command=ModelEditCommand(operation=difference))
