"""Prepare Model field writes against one saved project revision."""

from collections.abc import Iterator
from dataclasses import dataclass, replace

from pydantic import JsonValue

from feedbax.component_registry import ComponentRegistry
from feedbax.compiler.model_attachment import resolve_effective_model
from feedbax.compiler.model_document import model_tau_seconds
from feedbax.compiler.model_ports import model_unstored_fields
from feedbax.contracts.expression_values import (
    TIME, CallValue, Expression, Level, LiteralValue, NameValue, Problem, ProblemCode, ScopedValue, Site,
    SweepList, SweepRange, SweepSample, UnresolvedValue, ValueBase, ValueType,
)
from feedbax.contracts.model_address import AttachmentElement, ComponentElement, FieldElement, ModelAddress, PortElement, WireElement
from feedbax.contracts.model_document import (
    CompositeComponent, DelayAttachment, DelayInitialTask, DelayInitialValue, InterventionAttachment,
    LeafComponent, ModelDocument,
)
from feedbax.contracts.stored_model import STORED_MODEL_DOCUMENT_SCHEMA_ID, StoredModelDocument, authored_model_payload
from feedbax.contracts.model_edit import (
    EvaluateModelRequest, ListedRuns, StepRuns, ModelEditCommand, SetField,
)
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.project import BasedDocument, InlineBase, SetProjectValue
from feedbax.contracts.project_resolution import ProjectDiagnostic, compose_document
from feedbax.planner.project_plan import RunKey
from feedbax.contracts.jobs import StudioPlanReviewSource
from feedbax.contracts.project_sources import ValueSource, pointer
from feedbax.contracts.project_sweep import NamedAxis, SetValue
from feedbax.contracts.studio_field import (
    FieldResetEdit, FieldTextEdit, ModelFieldRequest, ModelFieldResponse, ModelFieldView,
)
from feedbax.contracts.studio_project import AuthoredFileAddress, StudioProjectFilePatch
from feedbax.contracts.studio_scope import StoreKind
from feedbax.contracts.expression_registry import FunctionRegistry
from feedbax.contracts.expression_site import project_site, read_field
from feedbax.contracts.expression_text import ParsedField, print_field
from feedbax.contracts.task_document import StoredTaskDocument
from feedbax.contracts.window import store_window_offset, window_offset_expression, window_offset_sites
from feedbax.studio.authoring_sweeps import SweepTarget, FieldSweeps
from feedbax.studio.field_sites import FieldSiteIndex
from feedbax.studio.numeric_fields import NumericField, prepare_numeric_field, project_numeric_field
from feedbax.studio.selection import StoredEdit, merged_files
from feedbax.studio.value_structure import value_structure
from feedbax.web.services.authored_scope import choose_prepared_edit, selected_edit_runs
from feedbax.web.services.axis_values import axis_values
from feedbax.web.services.evaluate_sources import read_evaluate_model
from feedbax.web.services.model_projection import selected_model
from feedbax.web.services.project_folder import ProjectFolderInvalid, ProjectFolderService
from feedbax.web.services.trial_scope import (
    evaluation_document_write, model_content_edits, scoped_file, selected_value, selection_scope, source_value_file, variant_file,
)


@dataclass(frozen=True)
class _Field:
    address: ModelAddress
    numeric: NumericField
    literal: bool = False
    window_offset: bool = False
    expression: Expression | None = None


def _address(owner: ModelAddress, *path: str | int) -> ModelAddress:
    return ModelAddress(
        composites=owner.composites,
        element=FieldElement(kind="field", parent=owner.element, path=path),
    )


def _duration(
    owner: ModelAddress, label: str, path: tuple[str, ...], value: int | float | UnresolvedValue, unit: str,
    *, stored: bool = False, literal: bool = False, sampled: bool = False,
) -> _Field:
    address = _address(owner, *path)
    return _Field(address, NumericField(
        address.stored_path(), label, value,
        Site(expected=ValueType(base=ValueBase.REAL, dimensions=TIME), finest=Level.RUN, structural=True),
        stored, _positive, unit, time_use="length" if sampled else "constant",
    ), literal=literal)


def _positive(value: int | float) -> int | float:
    if value <= 0:
        raise ValueError("Enter a positive duration.")
    return value


def _numeric_value(value: JsonValue) -> int | float | UnresolvedValue | None:
    if type(value) in (int, float):
        return value
    if isinstance(value, dict) and value.get("kind") == "unresolved":
        return UnresolvedValue.model_validate(value)
    return None


def _window_fields(
    attachment: InterventionAttachment, owner: ModelAddress, names: tuple[ScopedValue, ...],
) -> Iterator[_Field]:
    window = attachment.window
    if window is None:
        return
    for (_, index, bound, _), site in window_offset_sites(window, names).items():
        span = window.spans[int(index)]
        anchor = getattr(span, bound)
        expression = window_offset_expression(anchor.offset)
        field = _address(owner, "window", "spans", int(index), bound, "offset")
        is_number = isinstance(expression, LiteralValue) and type(expression.value) in (int, float)
        value = expression.value if is_number else expression if isinstance(expression, UnresolvedValue) else 0
        yield _Field(field, NumericField(
            field.stored_path(), f"{bound.capitalize()} offset", value,
            site, True, lambda number: number, (expression.unit or "s") if is_number else "st", time_use="offset",
        ), window_offset=True, expression=None if is_number or isinstance(expression, UnresolvedValue) else expression)


_OPERAND_TITLES = {"factor": "Factor", "value": "Value", "low": "Low", "high": "High", "initial": "Initial"}
"""The titles of the operands the intervention operations and delays declare."""


def _attachment_fields(
    branch: ModelDocument | CompositeComponent, owner: ModelAddress, names: tuple[ScopedValue, ...],
) -> Iterator[_Field]:
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
        delay = attachment.delay
        if isinstance(delay, UnresolvedValue):
            yield _duration(owner, "Delay", ("delay",), delay, "st", stored=True, sampled=True)
        else:
            yield _duration(owner, "Delay", ("delay",), delay.value, delay.unit, stored=True, sampled=True)
        if isinstance(attachment.initial_source, DelayInitialValue) and not isinstance(attachment.initial_source.value, tuple):
            address = _address(owner, "initial_source", "value")
            yield _Field(address, NumericField(
                address.stored_path(), "Initial", attachment.initial_source.value,
                Site(expected=ValueType(base=ValueBase.REAL), finest=Level.RUN, structural=True),
                True, lambda value: value,
            ))
        return
    if isinstance(attachment, InterventionAttachment):
        yield from _window_fields(attachment, owner, names)
    else:
        return
    values = {name: source.value for name, source in attachment.operands.items() if isinstance(source, DelayInitialValue)}
    for name, source in attachment.operands.items():
        expression = None
        if isinstance(source, DelayInitialTask):
            address = _address(owner, "operands", name)
            expression = NameValue(reference=source.variable)
            value_type = next((item.type for item in names if item.id == source.variable), ValueType(base=ValueBase.REAL))
            value = 0
        elif isinstance(source, DelayInitialValue):
            address = _address(owner, "operands", name, "value")
            value = source.value
            value_type = ValueType(base=ValueBase.REAL, shape=(len(value),) if isinstance(value, tuple) else ())
            if isinstance(value, tuple):
                expression = CallValue(function="vector", args=tuple(LiteralValue(value=item) for item in value))
                value = 0
        else:
            continue

        def validate(number: int | float, name=name) -> int | float:
            low = number if name == "low" else values.get("low")
            high = number if name == "high" else values.get("high")
            if type(low) in (int, float) and type(high) in (int, float) and low > high:
                raise ValueError("Clamp low must be at most high.")
            return number

        yield _Field(address, NumericField(
            address.stored_path(), _OPERAND_TITLES.get(name, "Operand"), value,
            Site(expected=value_type, finest=Level.TRIAL, names=names),
            True, validate,
        ), expression=expression)

def _fields(
    model: ModelDocument, owner: ModelAddress, components: ComponentRegistry, names: tuple[ScopedValue, ...] = (),
) -> Iterator[_Field]:
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
        yield from _attachment_fields(branch, owner, names)
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
    unstored = model_unstored_fields(component.type)
    for schema in definition.param_schema:
        name = schema.name
        if name in unstored or name in component.port_counts or "." in name:
            continue
        base = {"int": ValueBase.INTEGER, "float": ValueBase.REAL}.get(schema.type)
        value = component.fields.get(name, defaults.get(name))
        number = _numeric_value(value)
        if component.type == "feedbax.VanillaRNN" and name == "tau":
            if number is not None:
                yield _duration(owner, "Tau", ("fields", "tau"), number, "st", stored=name in component.fields)
            elif isinstance(value, dict) and value.get("kind") == "literal" and value.get("unit") in ("st", "s"):
                if type(magnitude := value.get("value")) in (int, float):
                    yield _duration(owner, "Tau", ("fields", "tau"), magnitude, value["unit"], stored=True, literal=True)
            continue
        if base is None or number is None:
            continue

        def validate(value: int | float, name=name) -> int | float:
            values = {key: item for key, item in component.fields.items()
                      if not isinstance(_numeric_value(item), UnresolvedValue)}
            values[name] = value
            if component.type == "feedbax.VanillaRNN" and "tau" in values and isinstance(model.step_seconds, float):
                values["tau"] = model_tau_seconds(values["tau"], model.step_seconds)
            definition.params.require(values)
            return value

        address = _address(owner, "fields", name)
        yield _Field(address, NumericField(
            address.stored_path(), definition.params.field_title(name), number,
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


def _reference_model(project, restored: str | None) -> ModelDocument | None:
    document = compose_document(project.documents.get, restored) if restored is not None else None
    if document is None or isinstance(document, ProjectDiagnostic):
        return None
    return StoredModelDocument.model_validate(document.payload).lower()


def _reset_text(
    project, restored: str | None, owner: ModelAddress, address: ModelAddress, components: ComponentRegistry,
) -> str | Problem:
    """The text of the field at ``address`` in ``restored``, the document a Reset gives the field its value from."""
    model = _reference_model(project, restored)
    if model is None:
        return Problem(code=ProblemCode.STRUCTURE, reason="This value is not changed from another document.")
    field = next((
        item for item in _fields(model, owner, components)
        if item.address == address
    ), None)
    if field is None:
        return Problem(code=ProblemCode.STRUCTURE, reason=f"{restored} has no value here.")
    numeric = field.numeric
    expression = field.expression or (numeric.value if isinstance(numeric.value, UnresolvedValue) else numeric.literal(numeric.value))
    return print_field(expression, numeric.site)


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
        for field in _fields(StoredModelDocument.model_validate(payload).lower(), owner, components):
            values = held.setdefault(field.address.to_text(), [])
            if field.expression is None and not isinstance(field.numeric.value, UnresolvedValue) and field.numeric.value not in values:
                values.append(field.numeric.value)
    return held


def _field_store(
    folders, project_id, project, plan, body, response: ModelFieldResponse,
    *, kind: StoreKind = "whole_step", label="Whole Step", axis: FieldSweeps | None = None,
    member_id: str | None = None, alternatives=(),
) -> ModelFieldResponse:
    if not response.files and not alternatives:
        return response
    selected = selected_edit_runs(plan, body.run.step, body)
    member_id = member_id or getattr(body.edit, "value_id", None)
    if not body.rows and member_id is not None and axis is not None and len(axis.covering) == 1:
        axis_name, _ = axis.covering[0]
        selected = frozenset(
            row.key for row in plan.rows
            if row.key in selected and (coordinate := row.coordinates.get(axis_name)) is not None
            and coordinate.value_id == member_id
        )
        kind, label = "sweep_value", "Sweep Value"
    choice = choose_prepared_edit(
        folders, project_id, body.revision, project, selected,
        body.model_dump(mode="json", exclude={"store_choice"}), body.store_choice,
        (kind, label, body.address.to_text(), StoredEdit(response.files, tuple(response.created))), alternatives,
        retain_equivalent=body.edit is not None and body.edit.kind == "to_list",
    )
    if isinstance(choice, list):
        return ModelFieldResponse(
            fields=response.fields, sites=response.sites, function_lists=response.function_lists, store_choices=choice,
        )
    return response.model_copy(update={"files": choice.files, "created": list(choice.created)})


def _address_problems(field: _Field, problems: tuple[Problem, ...]) -> list[Problem]:
    return [problem.model_copy(update={"model_address": field.address}) for problem in problems]


def _held_problems(field: _Field, functions: FunctionRegistry, period: float | None) -> list[Problem]:
    if not isinstance(held := field.numeric.value, UnresolvedValue):
        return []
    edit = FieldTextEdit(kind="text", text=held.text)
    result = prepare_numeric_field(field.numeric, None, edit, functions, period=period).result
    if not isinstance(result, tuple):
        reason = ("Delay in steps must be a whole number."
                  if isinstance(result, LiteralValue) and result.unit == "st" and field.address.element.parent.kind == "attachment"
                  else "This value has not been resolved.")
        result = (Problem(code=ProblemCode.STRUCTURE, reason=reason),)
    return _address_problems(field, result)


def _hold_sweep_member(axis, member: str, held: JsonValue, step_file, problems: list[Problem]) -> ModelFieldResponse:
    if axis is None or len(axis.covering) != 1:
        raise ProjectFolderInvalid("Choose an authored Model sweep value.")
    _, sweep = axis.covering[0]
    if not isinstance(sweep, NamedAxis) or (entry := sweep.values.get(member)) is None:
        raise ProjectFolderInvalid("Choose an authored Model sweep value.")
    path = axis.target.patch_path
    assert path is not None
    updated = entry.model_copy(update={"value": held, "patches": entry.patches | {path: SetValue(value=held)}})
    after = sweep.model_copy(update={"values": sweep.values | {member: updated}})
    return ModelFieldResponse(files=[step_file(axis.write_axis(after))], problems=problems)


def model_fields(
    folders: ProjectFolderService, project_id: str, body: ModelFieldRequest,
    components: ComponentRegistry, functions: FunctionRegistry,
) -> ModelFieldResponse:
    snapshot = folders.read(project_id, revision=body.revision)
    project = snapshot.project
    document = compose_document(project.documents.get, body.document)
    if isinstance(document, ProjectDiagnostic):
        raise ProjectFolderInvalid(f"The Model document cannot be read: {document.detail}")
    if document.schema_id != STORED_MODEL_DOCUMENT_SCHEMA_ID:
        raise ProjectFolderInvalid("The Model document does not exist.")
    model = StoredModelDocument.model_validate(document.payload).lower()
    step = selected = plan = effective_model = None
    shown = project.documents.get(body.document)
    changes, restored = (shown.differences, shown.base.document) if isinstance(shown, BasedDocument) else ({}, None)
    definitions = ()
    visible_axis_values = {}
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
        effective_model = reading.model
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
        definitions = tuple((item, plan.selected_definitions[item]) for item in (scoped or (key,))
                            if item in plan.selected_definitions)
        if definitions:
            visible_axis_values = axis_values(
                row for row in plan.rows if (row.key.variant, row.key.step) == (key.variant, key.step)
            )
        definition = plan.selected_definitions.get(key)
        if definition is not None and (model_payload := selected_model(definition, body.document)) is not None:
            effective_model = StoredModelDocument.model_validate(model_payload).lower()
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
    names: tuple[ScopedValue, ...] = ()
    if plan is not None and body.run is not None:
        definition = plan.selected_definitions.get(RunKey(body.run.variant, body.run.step, body.run.row))
        slots = definition.payload.get("trial_documents") if definition is not None else None
        if isinstance(slots, dict) and isinstance(raw := slots.get("task"), dict):
            try:
                task = StoredTaskDocument.model_validate(raw)
                names = tuple(ScopedValue(id=key, name=variable.label, type=variable.type, level=Level.TRIAL)
                              for key, variable in task.variables.items())
            except ValueError:
                pass
    effective_fields = {
        field.address.to_text(): field for field in _fields(effective_model, body.owner, components, names)
    } if effective_model is not None else {}
    authored_fields = {field.address.to_text(): field for field in _fields(model, body.owner, components, names)}
    visible_fields = effective_fields if evaluate or variant is not None else effective_fields | authored_fields
    reference = _reference_model(project, restored) if changes else None
    reference_fields = {
        field.address.to_text(): field for field in _fields(reference, body.owner, components, names)
    } if reference is not None else {}
    period = (effective_model or model).step_seconds
    if isinstance(period, UnresolvedValue):
        period = None
    fields = []
    sites = FieldSiteIndex()
    targets = {}
    for field_key, field in visible_fields.items():
        numeric = field.numeric
        resolved = effective_fields.get(field_key)
        axis = None
        if step is not None:
            target = SweepTarget(
                numeric.value.model_dump(mode="json") if isinstance(numeric.value, UnresolvedValue) else numeric.value,
                _addressed, field_key,
            )
            axis = FieldSweeps(
                step, target, numeric.label, numeric.key[-1:], integer=numeric.site.expected.base == ValueBase.INTEGER,
                runs=runs,
            )
            # An addressed sweep can set fields inherited from the registry.
            numeric = replace(numeric, stored=True)
        relative = pointer(field.address.stored_path())
        changed = relative in changes
        if not changed and any(relative.startswith(path + "/") for path in changes):
            inherited = reference_fields.get(field_key)
            changed = inherited is None or (
                inherited.numeric.value, inherited.numeric.unit, inherited.expression
            ) != (numeric.value, numeric.unit, field.expression)
        # The window map groups equal coordinates. Keep the stored origin key and path.
        sources = dict.fromkeys(
            replace(source, value=value_id) if (
                source.kind == "sweep_value" and source.step == run.step and source.axis is not None
                and (value_id := visible_axis_values.get((run, source.axis))) is not None
            ) else source
            for run, definition in definitions
            if (source := definition.sources.get(f"/payload/trial_documents/model{relative}")) is not None
        )
        view = project_numeric_field(
            ModelFieldView, numeric, axis, functions, selected, resolved.numeric.value if resolved else None,
            sites=sites, held=held.get(field_key, ()),
            period=period, effective_unit=resolved.numeric.unit if resolved else None,
            scoped=bool(body.rows or body.axes),
            address=field.address, problems=_held_problems(field, functions, period)
            or (_held_problems(resolved, functions, period) if resolved is not None else []),
            sources=[StudioPlanReviewSource.from_source(source) for source in sources],
            changed=changed,
        )
        if field.window_offset or field.address.element.path[0] == "operands" and isinstance(field.address.element.parent, AttachmentElement):
            view = view.model_copy(update={"site": sites.reference(project_site(numeric.site, functions))})
        if field.expression is not None:
            view = view.model_copy(update={
                "text": print_field(field.expression, numeric.site),
                "structure": value_structure(field.expression, numeric.site, functions),
                "conversion": None, "can_sweep": False,
            })
        fields.append(view)
        targets[field_key] = replace(field, numeric=numeric), axis
    if body.edit is None:
        return ModelFieldResponse(fields=fields, **sites.tables(fields))
    if body.address is None or body.address.to_text() not in targets:
        raise ProjectFolderInvalid("Choose a supported field of the selected Model.")
    if body.run is None and (body.rows or body.axes or body.evaluate_scope is not None) or (
        body.run is not None and evaluate is None and variant is None and (step is None or plan is None)
    ):
        raise ProjectFolderInvalid("Choose a supported field of the selected Model.")
    field, axis = targets[body.address.to_text()]
    owner_name = body.document

    def step_file(patches: list[SetProjectValue]) -> StudioProjectFilePatch:
        if body.run is None:
            raise ProjectFolderInvalid("Choose a step that reads this Model to edit its sweep.")
        return StudioProjectFilePatch(address=AuthoredFileAddress(kind="step", id=body.run.step), patches=patches)

    scope = None
    if (body.rows or body.axes) and evaluate is None:
        assert body.run is not None
        key = RunKey(body.run.variant, body.run.step, body.run.row)
        scope = selection_scope(project, plan, key, tuple(body.rows), body.axes)
    edit = body.edit
    if isinstance(edit, FieldResetEdit):
        reset = _reset_text(project, restored, body.owner, field.address, components)
        if isinstance(reset, Problem):
            return ModelFieldResponse(problems=[reset])
        edit = FieldTextEdit(kind="text", text=reset)
    expression = None
    write_address = field.address
    operand = field.address.element.path[0] == "operands" and isinstance(field.address.element.parent, AttachmentElement)
    if (field.window_offset or operand) and isinstance(edit, FieldTextEdit) and edit.value_id is None and not (axis and axis.covering):
        parsed = read_field(edit.text, field.numeric.site, functions)
        if isinstance(parsed, ParsedField):
            node = parsed.value
            if field.window_offset and not isinstance(node, LiteralValue | SweepList | SweepRange | SweepSample):
                expression = store_window_offset(node).model_dump(mode="json")
            elif operand and isinstance(node, NameValue):
                write_address = _address(body.owner, *field.address.element.path[:2])
                expression = DelayInitialTask(kind="task", variable=node.reference).model_dump(mode="json")
            elif operand and isinstance(node, CallValue) and node.function == "vector" and all(
                isinstance(item, LiteralValue) and type(item.value) in (int, float) for item in node.args
            ):
                expression = [item.value for item in node.args]
    prepared = prepare_numeric_field(
        field.numeric, axis, edit, functions, period=period, scoped=scope is not None,
    ) if expression is None else None
    result = prepared.result if prepared is not None else expression
    problems = []
    if isinstance(result, tuple):
        if not isinstance(edit, FieldTextEdit):
            raise ProjectFolderInvalid(result[0].reason)
        problems = _address_problems(field, result)
        result = UnresolvedValue(text=edit.text).model_dump(mode="json")
        member_id = edit.value_id
        if member_id is None and scope is None and selected is not None and axis is not None and len(axis.covering) == 1:
            axis_name, sweep = axis.covering[0]
            coordinate = selected.coordinates.get(axis_name)
            if isinstance(sweep, NamedAxis) and coordinate is not None and sweep.values.get(coordinate.value_id) is not None:
                member_id = coordinate.value_id
        if member_id is not None:
            return _field_store(
                folders, project_id, project, plan, body,
                _hold_sweep_member(axis, member_id, result, step_file, problems), axis=axis, member_id=member_id,
            )
    if prepared is not None and isinstance(result, list):
        assert body.run is not None
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
    value = result
    if isinstance(result, LiteralValue):
        if field.window_offset:
            value = store_window_offset(result).model_dump(mode="json")
        elif field.address.element.parent.kind == "attachment":
            value = result.model_dump(mode="json", exclude={"kind"})
        elif result.unit == "st" and field.numeric.unit == "st" and not field.literal and field.numeric.stored:
            value = result.value
        else:
            value = result.model_dump(mode="json")
    if operand and len(write_address.element.path) == 2 and write_address == field.address:
        if not isinstance(value, dict) or value.get("kind") != "task":
            value = {"kind": "value", "value": value}
    if scope is not None and selected_value(project, body.run.step, scope) is None:
        if axis is None or axis.target.patch_path is None:
            return ModelFieldResponse(problems=[Problem(
                code=ProblemCode.STRUCTURE, reason="This field has no writable sweep destination.",
            )])
        path = axis.target.patch_path if write_address == field.address else write_address.to_text()
        write = OverridePatch(op="replace", path=path, value=value)
        relative = pointer(write_address.stored_path())
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
            for part in write_address.stored_path():
                held = held.get(part) if isinstance(held, dict) else None
            file = source_value_file(project, source, held, value)
            if file is not None:
                kind, label = {
                    "sweep_value": ("sweep_value", "Sweep Value"), "exception": ("row_exception", "Row Exception"),
                    "variant": ("variant", "Variant"),
                }[source.kind]
                alternatives.append((kind, label, f"{source.name}:{source.path}", StoredEdit([file])))
        destination = ListedRuns(kind="rows", rows=tuple(sorted(writable))) if sources and writable else scope
        files = scoped_file(project, body.run.step, destination, [write]).files if writable else []
        return _field_store(
            folders, project_id, project, plan, body, ModelFieldResponse(files=files, problems=problems), kind="other_runs",
            label="Other Runs", alternatives=alternatives,
        )
    difference = SetField(op="set_field", address=write_address, value=value)
    if evaluate is not None:
        return ModelFieldResponse(command=ModelEditCommand(operation=difference), problems=problems)
    if variant is not None:
        assert body.run is not None
        field_key = difference.address.to_text()
        path = f"documents.{owner_name}.{field_key}"
        if field_key not in authored_fields or isinstance(difference.address.element.parent, AttachmentElement):
            raise ProjectFolderInvalid("This Model field is not part of the named document.")
        write = OverridePatch(op="replace", path=path, value=value)
        chosen_steps = {body.run.step, *body.chosen_steps}
        alternatives = []
        if chosen_steps < plan.document_users(variant, owner_name):
            try:
                changed = resolve_effective_model(model, components, (difference,), functions=functions)
            except ValueError as error:
                raise ProjectFolderInvalid(str(error)) from error
            edits = model_content_edits(document.payload, authored_model_payload(changed))
            narrowed = evaluation_document_write(project, plan, owner_name, edits, chosen_steps, "model", variant)
            alternatives.append(("selected_readers", "Selected Readers", variant, narrowed))
        return _field_store(
            folders, project_id, project, plan, body,
            ModelFieldResponse(files=[variant_file(project, variant, [write])], problems=problems), kind="variant", label="Variant",
            alternatives=alternatives,
        )
    return ModelFieldResponse(command=ModelEditCommand(operation=difference), problems=problems)
