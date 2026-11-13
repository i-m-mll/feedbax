"""Prepare Model field writes against one saved project revision."""

from collections.abc import Iterator
from dataclasses import dataclass

from feedbax.component_registry import ComponentRegistry
from feedbax.compiler.model_document import model_tau_seconds
from feedbax.contracts.expression_values import TIME, Level, LiteralValue, Problem, ProblemCode, Site, ValueBase, ValueType
from feedbax.contracts.model_address import AttachmentElement, ComponentElement, FieldElement, ModelAddress, PortElement, WireElement
from feedbax.contracts.model_document import (
    MODEL_DOCUMENT_SCHEMA_ID, CompositeComponent, DelayAttachment, InterventionAttachment, LeafComponent, ModelDocument,
)
from feedbax.contracts.model_edit import ModelEditCommand, SetField
from feedbax.contracts.project import ProjectValueEdit, SetProjectValue, document_edit_ops
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.studio_field import ModelFieldRequest, ModelFieldResponse, ModelFieldView
from feedbax.contracts.studio_project import AuthoredFileAddress, StudioProjectFilePatch
from feedbax.expression_registry import FunctionRegistry
from feedbax.studio.authoring_sweeps import SweepTarget, TrainFieldSweeps
from feedbax.studio.numeric_fields import NumericField, prepare_numeric_field, project_numeric_field
from feedbax.web.services.project_folder import ProjectFolderInvalid, ProjectFolderService


@dataclass(frozen=True)
class _Field:
    address: ModelAddress
    numeric: NumericField
    value_only: bool = False

    @property
    def path(self) -> tuple[str, ...]:
        return self.address.stored_path()

    def document_edit(self, value: int | float) -> SetProjectValue:
        return SetProjectValue(
            op="replace" if self.numeric.stored else "add", path=pointer(self.path), value=value,
        )


def _address(owner: ModelAddress, *path: str) -> ModelAddress:
    return ModelAddress(
        document=owner.document, composites=owner.composites,
        element=FieldElement(kind="field", parent=owner.element, path=path),
    )


def _duration(owner: ModelAddress, label: str, path: tuple[str, ...], value: int | float, unit: str) -> _Field:
    address = _address(owner, *path)
    return _Field(address, NumericField(
        address.stored_path(), label, value,
        Site(expected=ValueType(base=ValueBase.REAL, dimensions=TIME), finest=Level.RUN, structural=True),
        False, _positive, unit, duration=True,
    ), value_only=True)


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
        chain = ports.sources.get(end.port, ()) if ports else ()
    else:
        return
    attachment = next((item for item in chain if item.id == element.id), None)
    if isinstance(attachment, DelayAttachment):
        yield _duration(owner, "Delay", ("delay",), attachment.delay.value, attachment.delay.unit)
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
            False, validate,
        ), value_only=True)


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
                yield _duration(owner, "Tau", ("fields", "tau"), value, "st")
            elif isinstance(value, dict) and value.get("kind") == "literal" and value.get("unit") in ("st", "s"):
                if type(magnitude := value.get("value")) in (int, float):
                    yield _duration(owner, "Tau", ("fields", "tau"), magnitude, value["unit"])
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


def _read(value: object, path: tuple[str, ...]) -> object:
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


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
        numeric = field.numeric
        target = SweepTarget(numeric.value, lambda _value: [], ".".join(("documents", body.owner.document, "payload", *field.path)))
        axis = TrainFieldSweeps(step, target, numeric.label) if step is not None and not field.value_only else None
        effective = _read(definition.payload["graph"], field.path) if definition is not None and not field.value_only else None
        fields.append(project_numeric_field(
            ModelFieldView, numeric, axis, functions, selected, effective, address=field.address,
        ))
        targets[field.address.to_text()] = field, axis
    if body.edit is None:
        return ModelFieldResponse(fields=fields)
    if body.address is None or body.address.to_text() not in targets or body.run is None or step is None or plan is None:
        raise ProjectFolderInvalid("Choose a supported Model field on a Train step.")
    field, axis = targets[body.address.to_text()]
    owner_name = body.owner.document

    step_address = AuthoredFileAddress(kind="step", id=body.run.step)

    def step_file(patches: list[ProjectValueEdit]) -> StudioProjectFilePatch:
        return StudioProjectFilePatch(address=step_address, patches=patches)

    def keep_value(value: int | float) -> list[SetProjectValue]:
        override = step.document_overrides.get(owner_name)
        edits = document_edit_ops(override) if override else []
        return [SetProjectValue(
            op="replace" if "document_overrides" in step.model_fields_set else "add",
            path="/document_overrides",
            value={
                **{doc: [item.model_dump(mode="json") for item in document_edit_ops(ops)]
                   for doc, ops in step.document_overrides.items()},
                owner_name: [item.model_dump(mode="json") for item in (*edits, field.document_edit(value))],
            },
        )]

    if body.rows:
        available = {row.key.row for row in plan.rows if
                     (row.key.variant, row.key.step) == (body.run.variant, body.run.step)}
        if not set(body.rows) <= available or len(set(body.rows)) != len(body.rows):
            raise ProjectFolderInvalid("The selected rows no longer address this stored field.")
    result = prepare_numeric_field(
        field.numeric, axis, body.edit, functions, period=model.step_seconds, rows=tuple(body.rows),
        keep_value=keep_value,
    )
    if isinstance(result, tuple):
        return ModelFieldResponse(problems=list(result))
    if isinstance(result, list):
        return ModelFieldResponse(files=[step_file(result)])
    if set(body.lit_steps) != plan.document_users(body.run.variant, owner_name):
        return ModelFieldResponse(problems=[Problem(
            code=ProblemCode.STRUCTURE, reason="Select all steps that use this Model to change its document value.",
        )])
    value = result
    if isinstance(result, LiteralValue):
        if field.address.element.parent.kind == "attachment":
            if result.unit == "st" and result.value != int(result.value):
                return ModelFieldResponse(problems=[Problem(code=ProblemCode.STRUCTURE, reason="Delay in steps must be a whole number.")])
            value = result.model_dump(mode="json", exclude={"kind"})
        else:
            value = result.model_dump(mode="json")
    return ModelFieldResponse(command=ModelEditCommand(operation=SetField(op="set_field", address=field.address, value=value)))
