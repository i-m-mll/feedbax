"""One projection and edit recipe for numeric Studio fields."""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

from feedbax.contracts.expression_units import UNITS
from feedbax.contracts.expression_values import (
    FieldValue, LiteralValue, Problem, ProblemCode, Site, SweepList, SweepRange, SweepSample, ValueBase,
)
from feedbax.contracts.project import SetProjectValue
from feedbax.contracts.project_plan import PlannedRow
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.project_sweep import (
    DEFAULT_SWEEP_LIMIT, ListedAxis, NamedAxis, RangeAxis, RangeValues, SampledAxis, SampleValues,
    SweepException, SweepExceptionTerms, sweep_value_identity_bytes,
)
from feedbax.contracts.studio_authoring import (
    StudioAuthoringSweepChange, StudioNameSweepValue, StudioSetSweepValue, StudioSweepEdit, StudioToSweep,
)
from feedbax.contracts.studio_field import FieldKindEdit, FieldNameEdit, FieldTextEdit, StudioFieldView, StudioSweepValue
from feedbax.expression_registry import FunctionRegistry
from feedbax.expression_site import FieldKind, field_kind, project_site, read_field
from feedbax.expression_text import ParsedField, print_field
from feedbax.studio.authoring_sweeps import TrainFieldSweeps


@dataclass(frozen=True)
class NumericField:
    key: tuple[str, ...]
    label: str
    value: int | float
    site: Site
    stored: bool
    validate: Callable[[int | float], int | float]
    unit: str | None = None
    duration: bool = False

    def literal(self, value: int | float) -> LiteralValue:
        return LiteralValue(value=value, unit=self.unit)

    def number(self, node: object, period: float | None) -> int | float:
        if not isinstance(node, LiteralValue) or type(node.value) not in (int, float):
            raise ValueError(f"{self.label} requires a numeric Value in this editor.")
        if self.duration:
            duration = self.duration_value(node)
            if duration.unit != self.unit:
                raise ValueError(f"{self.label} sweep values must use {self.unit}. Set the field's unit first.")
            return duration.value
        value = node.value
        if node.unit == "st":
            if period is None:
                raise ValueError("This field has no model period.")
            value *= period
        elif node.unit is not None and (scale := UNITS[node.unit].scale) != 1:
            value *= scale
        if not isfinite(value) or type(value) is int and abs(value) > 2**53 - 1:
            raise ValueError(f"{self.label} requires a finite representable number.")
        return self.validate(value)

    def duration_value(self, node: object) -> LiteralValue:
        if not isinstance(node, LiteralValue) or type(node.value) not in (int, float):
            raise ValueError(f"{self.label} requires a duration.")
        unit = node.unit or self.unit
        assert unit is not None
        magnitude = node.value
        if unit != "st":
            magnitude *= UNITS[unit].scale
            unit = "s"
        return LiteralValue(value=self.validate(magnitude), unit=unit)


def _list_axis(sweep: TrainFieldSweeps, values: list[int | float]) -> NamedAxis:
    previous = sweep.covering[0][1] if len(sweep.covering) == 1 else None
    if isinstance(previous, ListedAxis | RangeAxis | SampledAxis):
        previous = sweep.to_list(previous)
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


def _sample_axis(value: SweepSample, field: NumericField, functions: FunctionRegistry, period: float | None, seed: int, path: str) -> SampledAxis:
    from jax import random, vmap
    from jax.numpy import arange

    from feedbax.expression_compile import Inputs, compile_expression

    if value.count > DEFAULT_SWEEP_LIMIT:
        raise ValueError(f"A sweep can hold at most {DEFAULT_SWEEP_LIMIT} values.")
    draw = value.draw.model_copy(update={"per": "run"})
    if field.duration:
        # A kept sample's numeric patches must be in the field's authored unit.
        # Compile step-valued draws with a unit step to avoid a nominal-period conversion.
        period = 1.0 if field.unit == "st" else None

        def with_saved_unit(node):
            if isinstance(node, LiteralValue):
                literal = field.duration_value(node)
                if literal.unit != field.unit:
                    raise ValueError(f"{field.label} sample bounds must use {field.unit}.")
                return literal
            raise ValueError(f"{field.label} sample bounds must be literal durations.")

        draw = draw.model_copy(update={"args": tuple(with_saved_unit(arg) for arg in draw.args)})
    compiled = compile_expression(
        draw, field.site.model_copy(update={"structural": False}), functions, field_address=field.key,
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
        draws[f"draw-{index + 1}"] = field.validate(member)
    return SampledAxis(
        path=path,
        sampler=SampleValues(distribution=draw, count=value.count, seed=seed), draws=draws,
    )


def project_numeric_field[View: StudioFieldView](
    view_type: type[View], field: NumericField, axis: TrainFieldSweeps | None,
    functions: FunctionRegistry, selected: PlannedRow | None, effective: object, **address,
) -> View:
    site = project_site(field.site, functions)
    view = axis.view() if axis else None
    coordinate = selected.coordinates.get(view.id) if selected and view else None
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
                for key, entry in axis.to_list(source).values.items()
                if entry is not None and type(entry.value) in (float, int)
            ]
        except ValueError:
            view = view.model_copy(update={"can_list": False})
    return view_type(
        **address, label=field.label, text=print_field(shown, field.site), unit=field.unit,
        effective=print_field(field.literal(effective), field.site)
        if type(effective) in (int, float) and effective != field.value else None,
        site=site, sweep=view, selected_value=coordinate.value_id if coordinate else None,
        can_sweep=bool(axis and axis.can_sweep and field.stored),
        can_change_axis=bool(axis and axis.simple and view and (
            view.editable if view.kind == "list" else view.can_list
        )),
        members=members,
    )


def prepare_numeric_field(
    field: NumericField, axis: TrainFieldSweeps | None, edit: FieldTextEdit | FieldKindEdit | FieldNameEdit,
    functions: FunctionRegistry, *, period: float | None = None, rows: tuple[str, ...] = (),
    keep_value: Callable[[int | float], list[SetProjectValue]] | None = None,
    selection: SweepExceptionTerms | None = None,
) -> int | float | LiteralValue | list[SetProjectValue] | tuple[Problem, ...]:
    """Return a constant for the owner to write, sweep patches, or field problems."""
    try:
        return _prepare(field, axis, edit, functions, period=period,
                        selection=selection or (SweepExceptionTerms(rows=list(rows)) if rows else None),
                        keep_value=keep_value)
    except ValueError as error:
        return (Problem(code=ProblemCode.STRUCTURE, reason=str(error)),)


def _prepare(
    field: NumericField, axis: TrainFieldSweeps | None, edit: FieldTextEdit | FieldKindEdit | FieldNameEdit,
    functions: FunctionRegistry, *, period: float | None, selection: SweepExceptionTerms | None,
    keep_value: Callable[[int | float], list[SetProjectValue]] | None,
) -> int | float | LiteralValue | list[SetProjectValue] | tuple[Problem, ...]:
    if selection is not None and edit.kind not in ("text", "name_value"):
        raise ValueError("Select the Train step to change the field kind.")
    covered = axis.covering if axis else []
    previous = covered[0][1] if len(covered) == 1 else None

    def sweep() -> TrainFieldSweeps:
        if axis is None or not field.stored:
            raise ValueError("This field has no writable sweep destination.")
        return axis

    if edit.kind == "to_sweep":
        command = StudioAuthoringSweepChange(field=field.label, edit=StudioToSweep(kind="toSweep"))
        return sweep().edit(command, field.validate)
    if edit.kind == "name_value":
        if previous is None:
            raise ValueError("Choose a saved sweep value.")
        command = StudioAuthoringSweepChange(field=field.label, edit=StudioNameSweepValue(
            kind="nameSweepValue", sweep_id=covered[0][0], value_id=edit.value_id,
            label=edit.label.strip() or None if edit.label else None,
        ))
        return sweep().edit(command, field.validate)
    if edit.kind == "to_list":
        if previous is None:
            raise ValueError("Choose a saved sweep.")
        command = StudioAuthoringSweepChange(
            field=field.label, edit=StudioSweepEdit(kind="toList", sweep_id=covered[0][0]),
        )
        return sweep().edit(command, field.validate)
    if edit.kind == "to_value":
        owner = sweep()
        named = owner.to_list(previous) if isinstance(previous, ListedAxis | RangeAxis | SampledAxis) else previous
        entry = named.values.get(edit.value_id) if isinstance(named, NamedAxis) and edit.value_id else None
        if entry is None or not owner._editable_list(named) or keep_value is None:
            raise ValueError("Choose one value of this sweep to keep.")
        return [
            *keep_value(entry.value),
            SetProjectValue(op="replace", path=pointer(("sweep", "axes", covered[0][0])), value=None),
        ]
    assert isinstance(edit, FieldTextEdit)
    parsed = read_field(edit.text, field.site, functions)
    if not isinstance(parsed, ParsedField):
        kinds = ", ".join(kind.value.title() for kind in project_site(field.site, functions).kinds)
        return tuple(problem.model_copy(update={"reason": f"{field.label} takes {kinds}. {problem.reason}"}) for problem in parsed)
    value = parsed.value
    def number(item: object) -> int | float:
        return field.number(item, period)

    if edit.value_id:
        if field_kind(value, functions) != FieldKind.VALUE or previous is None:
            raise ValueError("A sweep member takes one Value.")
        command = StudioAuthoringSweepChange(
            field=field.label, edit=StudioSetSweepValue(
                kind="setSweepValue", sweep_id=covered[0][0], value_id=edit.value_id, value=number(value),
            ),
        )
        return sweep().edit(command, field.validate)
    if isinstance(value, SweepList | SweepRange | SweepSample):
        if selection is not None:
            raise ValueError("Select the Train step to change the sweep.")
        owner = sweep()
        if isinstance(value, SweepList):
            return owner.write_axis(_list_axis(owner, [number(item) for item in value.members]))
        path = owner.target.generated_path
        if path is None:
            raise ValueError("This field writes more than one setting. Enter a sweep list.")
        if isinstance(value, SweepRange):
            generated = RangeAxis(path=path, range=RangeValues(
                start=number(value.first), stop=number(value.last), count=value.count, scale=value.spacing,
            ))
            for member in owner.to_list(generated).values.values():
                assert member is not None and type(member.value) in (float, int)
                field.validate(member.value)
            return owner.write_axis(generated)
        if field.duration and value.draw.family not in {"uniform", "loguniform", "normal", "choice", "truncexp"}:
            raise ValueError("Choose a draw with scalar duration parameters.")
        seed = edit.seed if edit.seed is not None else previous.sampler.seed if isinstance(previous, SampledAxis) else 0
        return owner.write_axis(_sample_axis(value, field, functions, period, seed, path))
    if covered:
        raise ValueError("Choose Value from the kind menu to keep one sweep member.")
    if field.duration and selection is None:
        return field.duration_value(value)
    numeric = number(value)
    if selection is None:
        return numeric
    owner = sweep()
    authored = owner.step.sweep
    if authored is None:
        raise ValueError("Select the Train step to edit its single value.")
    if len(selection.axes) == 1:
        (axis_id, values), = selection.axes.items()
        existing = authored.axes.get(axis_id)
        if len(values) == 1 and isinstance(existing, NamedAxis) and (entry := existing.values.get(values[0])) is not None:
            updated = entry.model_copy(update={"patches": entry.patches | owner.target.entry(numeric).patches})
            return [SetProjectValue(
                op="replace", path=pointer(("sweep", "axes", axis_id, "values", values[0])),
                value=updated.model_dump(mode="json", exclude_unset=True),
            )]
    name = owner._fresh_id("field", authored.exceptions)
    exception = SweepException.model_validate({
        "where": selection.model_dump(mode="json", exclude_unset=True),
        "patches": owner.target.entry(numeric).patches,
    })
    payload = exception.model_dump(mode="json", exclude_defaults=True)
    if "exceptions" in authored.model_fields_set:
        return [SetProjectValue(op="add", path=pointer(("sweep", "exceptions", name)), value=payload)]
    return [SetProjectValue(op="add", path="/sweep/exceptions", value={name: payload})]
