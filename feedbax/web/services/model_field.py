"""Prepare Model field writes against one saved project revision."""

from collections.abc import Iterator
from dataclasses import dataclass, replace
from functools import partial

from pydantic import JsonValue

from feedbax.component_registry import ComponentRegistry
from feedbax.compiler.model_document import model_tau_seconds
from feedbax.contracts.expression_values import TIME, Level, LiteralValue, Problem, ProblemCode, Site, ValueBase, ValueType
from feedbax.contracts.model_address import AttachmentElement, ComponentElement, FieldElement, ModelAddress, PortElement, WireElement
from feedbax.contracts.model_document import (
    MODEL_DOCUMENT_SCHEMA_ID, CompositeComponent, DelayAttachment, InterventionAttachment, LeafComponent, ModelDocument,
)
from feedbax.contracts.model_edit import (
    EvaluateAttachmentReadRequest, EvaluateAttachmentRequest, EvaluateAttachmentRows, EvaluateAttachmentStep,
    ModelEditCommand, SetField,
)
from feedbax.contracts.model_attachment import MODEL_ATTACHMENTS
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.project import InlineBase, SetProjectValue, Step
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.jobs import StudioPlanReviewSource
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.project_sweep import _paths_overlap, authored_row
from feedbax.contracts.studio_field import ModelFieldRequest, ModelFieldResponse, ModelFieldView
from feedbax.contracts.studio_project import AuthoredFileAddress, StudioProjectFilePatch
from feedbax.expression_registry import FunctionRegistry
from feedbax.studio.authoring_sweeps import SweepTarget, FieldSweeps
from feedbax.studio.numeric_fields import NumericField, prepare_numeric_field, project_numeric_field
from feedbax.studio.selection import merged_files
from feedbax.web.services.evaluate_attachment import (
    EvaluateAttachmentRefused, prepare_evaluate_attachment, read_evaluate_attachments,
)
from feedbax.web.services.model_projection import selected_model
from feedbax.web.services.project_folder import ProjectFolderInvalid, ProjectFolderService
from feedbax.web.services.trial_scope import scoped_file, selection_scope, variant_file


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


def _step_field_write(step: Step, address: ModelAddress, value: JsonValue) -> list[SetProjectValue]:
    return _step_field_writes(step, [(address, value)])


def _step_field_writes(step: Step, values: list[tuple[ModelAddress, JsonValue]]) -> list[SetProjectValue]:
    name = "field-values"
    batches = step.model_attachments
    addresses = [address for address, _ in values]
    records = [record for record in batches.get(name, ()) if record.address not in addresses]
    records.extend(SetField(op="set_field", address=address, value=value) for address, value in values)
    return [SetProjectValue(
        op="replace" if MODEL_ATTACHMENTS in step.model_fields_set else "add",
        path=pointer((MODEL_ATTACHMENTS,)),
        value={
            key: [record.model_dump(mode="json", exclude_unset=True) for record in batch]
            for key, batch in (batches | {name: records}).items()
        },
    )]


def _kept_files(project, step_id: str, step: Step, writes: list[OverridePatch]) -> list[StudioProjectFilePatch]:
    """The files that give ``step`` a sweep's kept value: Model fields among its field values, the rest in its definition."""
    fields = [(ModelAddress.from_text(write.path), write.value) for write in writes if write.path.startswith("{")]
    if any(write.op == "remove" for write in writes if write.path.startswith("{")):
        raise ProjectFolderInvalid("A sweep value removes a Model field.")
    rest = [write for write in writes if not write.path.startswith("{")]
    return [
        *([StudioProjectFilePatch(
            address=AuthoredFileAddress(kind="step", id=step_id), patches=_step_field_writes(step, fields),
        )] if fields else []),
        *([scoped_file(project, step_id, EvaluateAttachmentStep(kind="step"), rest)] if rest else []),
    ]


def model_fields(
    folders: ProjectFolderService, project_id: str, body: ModelFieldRequest,
    components: ComponentRegistry, functions: FunctionRegistry,
) -> ModelFieldResponse:
    snapshot = folders.read(project_id, revision=body.revision)
    project = snapshot.project
    document = project.documents.get(body.document)
    if document is None or document.schema_id != MODEL_DOCUMENT_SCHEMA_ID:
        raise ProjectFolderInvalid("The Model document does not exist.")
    model = ModelDocument.model_validate(document.payload)
    step = selected = plan = effective_model = None
    definitions = ()
    evaluate = None
    variant = None
    if body.evaluate_scope is not None and body.run is not None:
        evaluate = EvaluateAttachmentReadRequest(revision=body.revision, run=body.run, scope=body.evaluate_scope)
        key = RunKey(body.run.variant, body.run.step, body.run.row)
        plan = folders.read_plan(project_id, revision=body.revision, source_runs=(key,)).plan
        effective_model = read_evaluate_attachments(
            folders, project_id, evaluate, components, selected=plan, model_document=body.document,
        ).resolved_model
        candidate = project.steps[key.step]
        if (isinstance(evaluate.scope, EvaluateAttachmentStep) and plan.variants[:1] == (key.variant,)
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
            target = SweepTarget(numeric.value, partial(_step_field_write, step, field.address), field_key)
            axis = FieldSweeps(step, target, numeric.label, numeric.key[-1:], integer=numeric.site.expected.base == ValueBase.INTEGER)
            if evaluate is not None:
                # The addressed sweep can set fields inherited from registry defaults.
                numeric = replace(numeric, stored=True)
        relative = pointer(field.address.stored_path())
        sources = dict.fromkeys(source for definition in definitions
                                if (source := definition.sources.get(f"/payload/trial_documents/model{relative}")) is not None)
        fields.append(project_numeric_field(
            ModelFieldView, numeric, axis, functions, selected, resolved.numeric.value if resolved else None,
            period=(effective_model or model).step_seconds, address=field.address,
            sources=[StudioPlanReviewSource.model_validate(source, from_attributes=True) for source in sources],
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
        if isinstance(scope, EvaluateAttachmentRows) and axis is not None and any(
            source.kind == "sweep_value"
            for field_view in fields if field_view.address == body.address
            for source in field_view.sources
        ):
            raise ProjectFolderInvalid("A selected row already writes this field through a sweep value.")
    prepared = prepare_numeric_field(
        field.numeric, axis, body.edit, functions, period=(effective_model or model).step_seconds,
        scoped=scope is not None,
    )
    result = prepared.result
    if isinstance(result, tuple):
        return ModelFieldResponse(problems=list(result))
    if isinstance(result, list):
        kept = _kept_files(project, body.run.step, step, prepared.keep) if prepared.keep and step is not None else []
        return ModelFieldResponse(files=merged_files([*kept, step_file(result)]), removed_values=prepared.removed)
    if scope is not None:
        if axis is None or axis.target.patch_path is None:
            return ModelFieldResponse(problems=[Problem(
                code=ProblemCode.STRUCTURE, reason="This field has no writable sweep destination.",
            )])
        write = OverridePatch(op="replace", path=axis.target.patch_path, value=result)
        return ModelFieldResponse(files=[scoped_file(project, body.run.step, scope, [write])])
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
        request = EvaluateAttachmentRequest(**evaluate.model_dump(), difference=difference)
        try:
            edit = prepare_evaluate_attachment(folders, project_id, request, components)
        except EvaluateAttachmentRefused as error:
            return ModelFieldResponse(problems=[Problem(code=ProblemCode.STRUCTURE, reason=error.problem.detail)])
        return ModelFieldResponse(files=[edit.file])
    if variant is not None:
        if body.rows or plan is None or set(body.lit_steps) != plan.document_users(variant, owner_name):
            raise ProjectFolderInvalid("Select the steps that use this Model in the authored variant.")
        sweep = project.sweep
        assert sweep is not None
        if any(row is not None and row.source == variant for row in sweep.rows.values()):
            raise ProjectFolderInvalid("This variant supplies inherited patches to another variant.")
        field_key = difference.address.to_text()
        path = f"documents.{owner_name}.{field_key}"
        expanded = authored_row(sweep, variant)
        if expanded is None or any(
            _paths_overlap(other, path) and (other != path or origin[0] != "row")
            for other, origin in expanded.patch_origins.items()
        ):
            raise ProjectFolderInvalid("The Model field has no local variant write destination.")
        if field_key not in authored_fields or isinstance(difference.address.element.parent, AttachmentElement):
            raise ProjectFolderInvalid("This Model field is not part of the named document.")
        keys = [row.key for row in plan.rows if row.key.variant == variant and owner_name in row.document_uses]
        if key not in keys:
            raise ProjectFolderInvalid("The selected run no longer uses this Model.")
        definitions = folders.read_plan(project_id, revision=body.revision, source_runs=keys).plan.selected_definitions
        relative = pointer(difference.address.stored_path())
        for selected_key in keys:
            definition = definitions.get(selected_key)
            if definition is None or selected_model(definition, owner_name) is None:
                raise ProjectFolderInvalid("A selected run no longer uses this Model directly.")
            slot = "trial_documents/model" if "trial_documents" in definition.payload else "graph"
            source = definition.sources.get(f"/payload/{slot}{relative}")
            if source is None:
                raise ProjectFolderInvalid("The Model field has no authored source in every selected run.")
            if not (
                source.kind == "document" and source.name == owner_name and source.path == relative
                or source.kind == "variant" and source.variant == variant and source.path == pointer(("sweep", path))
            ):
                raise ProjectFolderInvalid("The Model field is controlled by a different authored source.")
        write = OverridePatch(op="replace", path=path, value=value)
        return ModelFieldResponse(files=[variant_file(project, variant, [write])])
    if body.address.to_text() not in authored_fields:
        return ModelFieldResponse(files=[step_file(_step_field_write(step, field.address, value))])
    if set(body.lit_steps) != plan.document_users(body.run.variant, owner_name):
        return ModelFieldResponse(problems=[Problem(
            code=ProblemCode.STRUCTURE, reason="Select all steps that use this Model to change its document value.",
        )])
    return ModelFieldResponse(command=ModelEditCommand(operation=difference))
