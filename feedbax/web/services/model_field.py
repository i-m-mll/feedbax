"""Prepare Model field writes against one saved project revision."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from math import isfinite

from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.expression_units import UNITS
from feedbax.contracts.expression_values import (
    TIME, FieldValue, Level, LiteralValue, Problem, ProblemCode, Site, SweepList, SweepRange, SweepSample,
    ValueBase, ValueType,
)
from feedbax.contracts.model_address import ComponentElement, FieldElement, ModelAddress
from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID, CompositeComponent, LeafComponent, ModelDocument
from feedbax.contracts.model_edit import ModelEditCommand, SetField
from feedbax.contracts.project import ProjectValueEdit, SetProjectValue, document_edit_ops
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.project_sweep import (
    DEFAULT_SWEEP_LIMIT, ListedAxis, NamedAxis, RangeAxis, RangeValues, SampledAxis, SampleValues,
    SetValue, SweepException, axis_to_list,
    sweep_value_identity_bytes,
)
from feedbax.contracts.studio_authoring import (
    StudioAuthoringSweepChange, StudioSetSweepValue, StudioSweepEdit, StudioSweepValue, StudioToSweep,
)
from feedbax.contracts.studio_field import ModelFieldRequest, ModelFieldResponse, StudioFieldView
from feedbax.contracts.studio_project import AuthoredFileAddress, StudioProjectFilePatch
from feedbax.expression_registry import FunctionRegistry
from feedbax.expression_site import FieldKind, field_kind, project_site, read_field
from feedbax.expression_text import ParsedField, print_field
from feedbax.studio.authoring_sweeps import SweepTarget, TrainFieldSweeps
from feedbax.web.services.project_folder import ProjectFolderInvalid, ProjectFolderService


@dataclass(frozen=True)
class _Field:
    address: ModelAddress
    label: str
    value: int | float
    site: Site
    stored: bool
    validate: Callable[[int | float], None]
    unit: str | None = None

    @property
    def path(self) -> tuple[str, ...]:
        return self.address.stored_path()

    def literal(self, value: int | float) -> LiteralValue:
        return LiteralValue(value=value, unit=self.unit)

    def document_edit(self, value: int | float) -> SetProjectValue:
        return SetProjectValue(
            op="replace" if self.stored else "add", path=pointer(self.path), value=value,
        )

    def number(self, node: object, period: float) -> int | float:
        if not isinstance(node, LiteralValue) or type(node.value) not in (int, float):
            raise ValueError(f"{self.label} requires a numeric Value in this editor.")
        value = node.value
        if node.unit == "st":
            value *= period
        elif node.unit is not None and (scale := UNITS[node.unit].scale) != 1:
            value *= scale
        if not isfinite(value) or type(value) is int and abs(value) > 2**53 - 1:
            raise ValueError(f"{self.label} requires a finite representable number.")
        self.validate(value)
        return value


def _address(owner: ModelAddress, *path: str) -> ModelAddress:
    return ModelAddress(
        document=owner.document, composites=owner.composites,
        element=FieldElement(kind="field", parent=owner.element, path=path),
    )


def _positive(value: int | float) -> None:
    if value <= 0:
        raise ValueError("Step requires a positive duration.")


def _fields(model: ModelDocument, owner: ModelAddress, components: ComponentRegistry) -> Iterator[_Field]:
    if owner.element.kind == "graph" and not owner.composites:
        yield _Field(
            _address(owner, "step_seconds"), "Step", model.step_seconds,
            Site(expected=ValueType(base=ValueBase.REAL, dimensions=TIME), finest=Level.RUN, structural=True),
            "step_seconds" in model.model_fields_set, _positive, "s",
        )
        return
    if not isinstance(owner.element, ComponentElement):
        return
    branch: ModelDocument | CompositeComponent = model
    for key in owner.composites:
        child = branch.interior.components.get(key)
        if child is None or child.kind != "composite":
            return
        branch = child
    component = branch.interior.components.get(owner.element.id)
    if not isinstance(component, LeafComponent) or not (definition := components.get(component.type)):
        return
    defaults = definition.params.default_params
    for schema in definition.param_schema:
        name = schema.name
        if name in {"dt", "control_dt"} or name in component.port_counts or "." in name:
            continue
        base = {"int": ValueBase.INTEGER, "float": ValueBase.REAL}.get(schema.type)
        value = component.fields.get(name, defaults.get(name))
        if base is None or type(value) not in (int, float):
            continue

        def validate(value: int | float, name=name) -> None:
            definition.params.require({**component.fields, name: value})

        yield _Field(
            _address(owner, "fields", name), name.replace("_", " ").capitalize(), value,
            Site(expected=ValueType(base=base), finest=Level.RUN, structural=True),
            name in component.fields and "fields" in component.model_fields_set, validate,
        )


def _read(value: object, path: tuple[str, ...]) -> object:
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _list_axis(sweep: TrainFieldSweeps, values: list[int | float]) -> NamedAxis:
    previous = sweep.covering[0][1] if len(sweep.covering) == 1 else None
    if isinstance(previous, ListedAxis | RangeAxis | SampledAxis):
        previous = axis_to_list(previous)
    old = previous.values if isinstance(previous, NamedAxis) else {}
    by_value = {
        sweep_value_identity_bytes(entry.value): key for key, entry in old.items() if entry is not None
    }
    identities = [sweep_value_identity_bytes(value) for value in values]
    if len(set(identities)) != len(identities):
        raise ValueError("Sweep values must be distinct.")
    retained = {by_value[key] for key in identities if key in by_value}
    reserved = set(old)
    entries = {}
    old_keys = list(old)
    for index, (value, identity) in enumerate(zip(values, identities, strict=True)):
        key = by_value.get(identity)
        if key is None and index < len(old_keys) and old_keys[index] not in retained:
            key = old_keys[index]
        if key is None:
            key = sweep._fresh_id(f"value-{index + 1}", reserved)
            reserved.add(key)
        entries[key] = sweep.target.entry(value, old[key].label if key in old else None)
    return NamedAxis(values=entries)


def _sample_axis(value: SweepSample, field: _Field, functions: FunctionRegistry, period: float, seed: int) -> SampledAxis:
    from jax import random, vmap
    from jax.numpy import arange

    from feedbax.expression_compile import Inputs, compile_expression

    if value.count > DEFAULT_SWEEP_LIMIT:
        raise ValueError(f"A sweep can hold at most {DEFAULT_SWEEP_LIMIT} values.")
    draw = value.draw.model_copy(update={"per": "run"})
    compiled = compile_expression(
        draw, field.site.model_copy(update={"structural": False}), functions, field_address=field.path,
        period_seconds=period, root_draw_key=True,
    )
    if isinstance(compiled, tuple):
        raise ValueError(compiled[0].reason)

    def sample(key):
        inputs = Inputs(compiled.parameters, {}, key, key)
        result, _ = compiled.evaluate_step(inputs, compiled.with_trial(inputs, compiled.evaluate_run(inputs)))
        return result

    key = random.key(seed)
    keys = vmap(lambda index: random.fold_in(key, index))(arange(value.count))
    values, defined = vmap(sample)(keys)
    draws = {}
    for index, (member, valid) in enumerate(zip(values.tolist(), defined.tolist(), strict=True)):
        if not valid or not isfinite(member):
            raise ValueError("Sample contains an undefined or nonfinite value.")
        field.validate(member)
        draws[f"draw-{index + 1}"] = member
    return SampledAxis(
        path=".".join(("documents", field.address.document, "payload", *field.path)),
        sampler=SampleValues(distribution=draw, count=value.count, seed=seed), draws=draws,
    )


def model_fields(
    folders: ProjectFolderService, project_id: str, body: ModelFieldRequest,
    components: ComponentRegistry, functions: FunctionRegistry,
) -> ModelFieldResponse:
    snapshot = folders.read(project_id, revision=body.revision)
    project = snapshot.project
    document = project.documents.get(body.owner.document)
    if document is None or document.schema_id != MODEL_DOCUMENT_SCHEMA_ID:
        raise ProjectFolderInvalid("The Model document does not exist.")
    model = ModelDocument.model_validate(document.payload)
    step = selected = definition = plan = None
    if body.run is not None:
        key = RunKey(body.run.variant, body.run.step, body.run.row)
        plan = folders.read_plan(project_id, revision=body.revision, source_runs=(key,)).plan
        definition = plan.selected_definitions.get(key)
        if (definition is not None and definition.document_ids.get("payload.graph") == body.owner.document
                and not definition.document_pointers.get("payload.graph")):
            selected = next((row for row in plan.rows if row.key == key), None)
            if project.sweep is None and plan.steps[(key.variant, key.step)].kind == "train":
                step = project.steps[key.step]
        else:
            definition = None
    fields = []
    targets = {}
    for field in _fields(model, body.owner, components):
        site = project_site(field.site, functions)
        target = SweepTarget(field.value, lambda _value: [], ".".join(("documents", body.owner.document, "payload", *field.path)))
        axis = TrainFieldSweeps(step, target, field.label) if step is not None else None
        view = axis.view() if axis else None
        coordinate = selected.coordinates.get(view.id) if selected and view else None
        effective = _read(definition.payload["graph"], field.path) if definition is not None else None
        source = axis.covering[0][1] if axis and len(axis.covering) == 1 else None
        shown: FieldValue = field.literal(field.value)
        if isinstance(source, RangeAxis):
            rule = source.range
            shown = SweepRange(first=field.literal(rule.start), last=field.literal(rule.stop), count=rule.count, spacing=rule.scale)
        elif isinstance(source, SampledAxis) and not isinstance(source.sampler.distribution, str):
            shown = SweepSample(draw=source.sampler.distribution, count=source.sampler.count)
        members = []
        if isinstance(source, RangeAxis | SampledAxis) and view is not None and view.kind != "list" and view.can_list:
            try:
                members = [
                    StudioSweepValue(id=key, value=entry.value, label=entry.label)
                    for key, entry in axis_to_list(source).values.items()
                    if entry is not None and type(entry.value) in (float, int)
                ]
            except ValueError:
                view = view.model_copy(update={"can_list": False})
        fields.append(StudioFieldView(
            address=field.address, label=field.label, text=print_field(shown, field.site),
            effective=print_field(field.literal(effective), field.site)
            if type(effective) in (int, float) and effective != field.value else None,
            site=site, sweep=view, selected_value=coordinate.value_id if coordinate else None,
            can_sweep=bool(axis and axis.can_sweep and field.stored),
            members=members,
        ))
        targets[field.address.to_text()] = field, axis
    if body.edit is None:
        return ModelFieldResponse(fields=fields)
    if body.address is None or body.address.to_text() not in targets or body.run is None or step is None or plan is None:
        raise ProjectFolderInvalid("Choose a supported Model field on a Train step.")
    field, axis = targets[body.address.to_text()]
    assert axis is not None
    owner_name = body.owner.document

    step_address = AuthoredFileAddress(kind="step", id=body.run.step)

    def step_file(patches: list[ProjectValueEdit]) -> StudioProjectFilePatch:
        return StudioProjectFilePatch(address=step_address, patches=patches)

    try:
        edit = body.edit
        if body.rows and edit.kind != "text":
            raise ValueError("Select the Train step to change the field kind.")
        files = []
        if edit.kind == "to_sweep":
            if not field.stored:
                raise ValueError("Save the field value before sweeping it.")
            change = StudioAuthoringSweepChange(field=field.label, edit=StudioToSweep(kind="toSweep"))
            files = [step_file(axis.edit(change, lambda value: value))]
        elif edit.kind == "to_list":
            if len(axis.covering) != 1:
                raise ValueError("Choose a saved sweep.")
            name, _ = axis.covering[0]
            change = StudioAuthoringSweepChange(field=field.label, edit=StudioSweepEdit(kind="toList", sweep_id=name))
            files = [step_file(axis.edit(change, lambda value: value))]
        elif edit.kind == "to_value":
            name, previous = axis.covering[0] if len(axis.covering) == 1 else (None, None)
            if isinstance(previous, ListedAxis | RangeAxis | SampledAxis):
                previous = axis_to_list(previous)
            entry = previous.values.get(edit.value_id) if isinstance(previous, NamedAxis) and edit.value_id else None
            if name is None or not axis.simple or entry is None or not axis._editable_list(previous):
                raise ValueError("Choose one value of this sweep to keep.")
            override = step.document_overrides.get(owner_name)
            op = field.document_edit(entry.value)
            edits = document_edit_ops(override) if override else []
            edits.append(op)
            files = [step_file([
                SetProjectValue(
                    op="replace" if "document_overrides" in step.model_fields_set else "add",
                    path="/document_overrides",
                    value={
                        **{doc: [item.model_dump(mode="json") for item in document_edit_ops(ops)]
                           for doc, ops in step.document_overrides.items()},
                        owner_name: [item.model_dump(mode="json") for item in edits],
                    },
                ),
                SetProjectValue(op="replace", path=pointer(("sweep", "axes", name)), value=None),
            ])]
        else:
            parsed = read_field(edit.text, field.site, functions)
            if not isinstance(parsed, ParsedField):
                kinds = ", ".join(kind.value.title() for kind in project_site(field.site, functions).kinds)
                return ModelFieldResponse(problems=[
                    problem.model_copy(update={"reason": f"{field.label} takes {kinds}. {problem.reason}"})
                    for problem in parsed
                ])
            value = parsed.value

            def number(item):
                return field.number(item, model.step_seconds)

            if edit.value_id:
                if field_kind(value, functions) != FieldKind.VALUE or len(axis.covering) != 1:
                    raise ValueError("A sweep member takes one Value.")
                name, _ = axis.covering[0]
                change = StudioAuthoringSweepChange(
                    field=field.label, edit=StudioSetSweepValue(
                        kind="setSweepValue", sweep_id=name, value_id=edit.value_id, value=number(value),
                    ),
                )
                files = [step_file(axis.edit(change, lambda value: value))]
            elif isinstance(value, SweepList | SweepRange | SweepSample):
                if body.rows:
                    raise ValueError("Select the Train step to change the sweep.")
                if not field.stored:
                    raise ValueError("Save the field value before sweeping it.")
                if isinstance(value, SweepList):
                    saved_axis = _list_axis(axis, [number(item) for item in value.members])
                elif isinstance(value, SweepRange):
                    if field.site.expected.base != ValueBase.REAL:
                        raise ValueError("Use a list of integers at this field.")
                    assert axis.target.patch_path is not None
                    saved_axis = RangeAxis(path=axis.target.patch_path, range=RangeValues(
                        start=number(value.first), stop=number(value.last), count=value.count, scale=value.spacing,
                    ))
                    for member in axis_to_list(saved_axis).values.values():
                        assert member is not None and type(member.value) in (float, int)
                        field.validate(member.value)
                else:
                    previous = axis.covering[0][1] if len(axis.covering) == 1 else None
                    seed = edit.seed if edit.seed is not None else previous.sampler.seed if isinstance(previous, SampledAxis) else 0
                    saved_axis = _sample_axis(value, field, functions, model.step_seconds, seed)
                files = [step_file(axis.write_axis(saved_axis))]
            elif axis.covering:
                raise ValueError("Choose Value from the kind menu to keep one sweep member.")
            else:
                numeric = number(value)
                if body.rows:
                    available = {row.key.row for row in plan.rows if
                                 (row.key.variant, row.key.step) == (body.run.variant, body.run.step)}
                    if not field.stored or not set(body.rows) <= available or len(set(body.rows)) != len(body.rows):
                        raise ValueError("The selected rows no longer address this stored field.")
                    if not step.sweep:
                        raise ValueError("Select the Train step to edit its single value.")
                    assert axis.target.patch_path is not None
                    name = axis._fresh_id("field", step.sweep.exceptions)
                    exception = SweepException.model_validate({
                        "where": {"rows": body.rows}, "patches": {axis.target.patch_path: SetValue(value=numeric)},
                    })
                    payload = exception.model_dump(mode="json", exclude_defaults=True)
                    patch = SetProjectValue(
                        op="add", path=pointer(("sweep", "exceptions", name)), value=payload,
                    ) if "exceptions" in step.sweep.model_fields_set else SetProjectValue(
                        op="add", path="/sweep/exceptions", value={name: payload},
                    )
                    return ModelFieldResponse(files=[step_file([patch])])
                if set(body.lit_steps) != plan.document_users(body.run.variant, owner_name):
                    raise ValueError("Select all steps that use this Model to change its document value.")
                return ModelFieldResponse(command=ModelEditCommand(operation=SetField(
                    op="set_field", address=field.address, value=numeric,
                )))
    except ValueError as error:
        return ModelFieldResponse(problems=[Problem(code=ProblemCode.STRUCTURE, reason=str(error))])
    return ModelFieldResponse(files=files)
