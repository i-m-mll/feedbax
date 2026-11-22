"""One projection and edit recipe for numeric Studio fields."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field as dataclass_field
from math import isfinite

from feedbax.contracts.expression_units import UNITS
from feedbax.contracts.expression_values import (
    FieldValue, LiteralValue, Problem, ProblemCode, Site, SweepList, SweepRange, SweepSample,
)
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.project import SetProjectValue
from feedbax.contracts.project_plan import PlannedRow
from feedbax.contracts.project_sweep import (
    DEFAULT_SWEEP_LIMIT, ListedAxis, NamedAxis, RangeAxis, RangeValues, SampledAxis, SampleValues,
    sweep_value_identity_bytes,
)
from feedbax.contracts.studio_authoring import (
    StudioAuthoringSweepChange, StudioNameSweepValue, StudioSetSweepValue, StudioSweepEdit, StudioToSweep,
)
from feedbax.contracts.studio_field import FieldKindEdit, FieldNameEdit, FieldTextEdit, StudioFieldView, StudioSweepValue
from feedbax.contracts.expression_registry import FunctionRegistry
from feedbax.expression_site import FieldKind, field_kind, project_site, read_field
from feedbax.contracts.expression_text import ParsedField, print_field
from feedbax.studio.authoring_sweeps import FieldSweeps, drop_axis, named_axis
from feedbax.studio.value_structure import value_structure

_STORED_KINDS = (FieldKind.VALUE, FieldKind.SWEEP)
"""The kinds a numeric field stores: a number, or a sweep of numbers."""


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
                raise ValueError(f"{self.label} sweep values are in {self.unit}.")
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


def _list_axis(sweep: FieldSweeps, values: list[int | float]) -> NamedAxis:
    previous = sweep.covering[0][1] if len(sweep.covering) == 1 else None
    named = isinstance(previous, NamedAxis)
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
        # A name belongs to its value: it stays with the value, never with a reused id, and a rule's values have none.
        label = old[key].label if key is not None and named else None
        if key is None and index < len(old_keys) and old_keys[index] not in retained:
            key = old_keys[index]
        if key is None:
            key = sweep._fresh_id(f"value-{index + 1}", reserved)
            reserved.add(key)
        entry = sweep.target.entry(value, label)
        if key in retained:
            # Edits made for this value's runs stay with the value.
            entry = entry.model_copy(update={"patches": old[key].patches | entry.patches})
        entries[key] = entry
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
    view_type: type[View], field: NumericField, axis: FieldSweeps | None,
    functions: FunctionRegistry, selected: PlannedRow | None, effective: object, *, held: Sequence[object] = (),
    period: float | None = None, scoped: bool = False, **address,
) -> View:
    """The field as the selected runs have it; ``scoped`` says they are not the whole step, as for the edit.

    ``held`` are the distinct values the selected runs hold; several, outside a sweep of the field, read as Mixed with
    those values as its members.
    """
    own = not scoped or bool(axis and axis.for_selection)
    site = project_site(field.site, functions)
    site = site.model_copy(update={"kinds": tuple(kind for kind in site.kinds if kind in _STORED_KINDS)})
    view = axis.view() if axis else None
    coordinate = selected.coordinates.get(view.id) if selected and view else None
    source = axis.shown[0][1] if axis and len(axis.covering) == 1 and len(axis.shown) == 1 else None
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
    elif view is None and len(held) > 1:
        members = [
            StudioSweepValue(id=f"value-{index + 1}", value=value, label=None)
            for index, value in enumerate(held) if type(value) in (float, int)
        ]
    return view_type(
        **address, label=field.label, text=print_field(shown, field.site), unit=field.unit,
        structure=value_structure(shown, field.site, functions),
        conversion=_conversion(field, period) if source is None else None,
        effective=print_field(field.literal(effective), field.site)
        if type(effective) in (int, float) and effective != field.value else None,
        mixed=len(held) > 1,
        site=site, sweep=view, selected_value=coordinate.value_id if coordinate else None,
        can_sweep=bool(own and axis and axis.can_sweep and field.stored),
        can_change_axis=bool(own and axis and axis.simple and view and (
            view.editable if view.kind == "list" else view.can_list
        )),
        members=members,
    )


def _conversion(field: NumericField, period: float | None) -> str | None:
    if not field.duration or period is None or type(field.value) not in (int, float):
        return None
    if field.unit == "s":
        return f"{round(field.value / period)} st"
    return f"{float(f'{field.value * period:.8g}'):g} s"


@dataclass(frozen=True)
class _Rewrite:
    """Patches that replace the covering sweep: by `after`, or by the field's single value `kept`. `keep` are the writes
    the selected runs take: the kept value's definition, or the removal of their own value under a new sweep."""

    patches: list[SetProjectValue]
    after: NamedAxis | None = None
    kept: int | float | None = None
    keep: list[OverridePatch] = dataclass_field(default_factory=list)


@dataclass(frozen=True)
class PreparedField:
    """A constant for the owner to write, sweep patches, or field problems.

    `removed` names, per sweep, the values whose runs leave the project. A run whose value keeps its id stays,
    and so does a run of the value a sweep is reduced to, since its request is unchanged: `keep` are the writes, at
    paths in the step's definition, that the owner stores on the step to give it that value's definition.
    """

    result: int | float | LiteralValue | list[SetProjectValue] | tuple[Problem, ...]
    removed: dict[str, list[str]] = dataclass_field(default_factory=dict)
    keep: list[OverridePatch] = dataclass_field(default_factory=list)


def _removed(sweep: FieldSweeps | None, rewrite: _Rewrite) -> dict[str, list[str]]:
    if sweep is None or len(sweep.covering) != 1:
        return {}
    axis_id, previous = sweep.covering[0]
    before = named_axis(sweep, previous)
    if before is None:
        return {}
    if rewrite.after is not None:
        gone = [value_id for value_id in before.values if value_id not in rewrite.after.values]
    else:
        kept = None if rewrite.kept is None else sweep_value_identity_bytes(rewrite.kept)
        gone = [value_id for value_id, entry in before.values.items()
                if entry is None or sweep_value_identity_bytes(entry.value) != kept]
    return {axis_id: gone} if gone else {}


def prepare_numeric_field(
    field: NumericField, axis: FieldSweeps | None, edit: FieldTextEdit | FieldKindEdit | FieldNameEdit,
    functions: FunctionRegistry, *, period: float | None = None, scoped: bool = False,
) -> PreparedField:
    """The edit's sweep patches, or the value the caller stores for the selected runs.

    ``scoped`` says the selected runs are not the whole step: then the field keeps its kind and its sweep, and a value
    is returned as a number for the caller to store by the selection. Runs that are a set of sweep values take a sweep
    of the field made for them only, edited as a step's sweep is.
    """
    try:
        result = _prepare(field, axis, edit, functions, period=period, scoped=scoped)
    except ValueError as error:
        return PreparedField((Problem(code=ProblemCode.STRUCTURE, reason=str(error)),))
    if isinstance(result, _Rewrite):
        return PreparedField(result.patches, _removed(axis, result), result.keep)
    return PreparedField(result)


def _prepare(
    field: NumericField, axis: FieldSweeps | None, edit: FieldTextEdit | FieldKindEdit | FieldNameEdit,
    functions: FunctionRegistry, *, period: float | None, scoped: bool,
) -> int | float | LiteralValue | list[SetProjectValue] | _Rewrite | tuple[Problem, ...]:
    if axis is not None and axis.for_selection:
        scoped = False
    if scoped and edit.kind not in ("text", "name_value"):
        raise ValueError(f"{field.label} has one kind for every run.")
    covered = axis.covering if axis else []
    previous = covered[0][1] if len(covered) == 1 else None

    def sweep() -> FieldSweeps:
        if axis is None or not field.stored:
            raise ValueError("This field has no writable sweep destination.")
        return axis

    if edit.kind == "to_sweep":
        command = StudioAuthoringSweepChange(field=field.label, edit=StudioToSweep(kind="toSweep"))
        owner = sweep()
        return _Rewrite(owner.edit(command, field.validate), keep=owner.release())
    if edit.kind == "name_value":
        if previous is None:
            raise ValueError(f"{field.label} is not swept.")
        command = StudioAuthoringSweepChange(field=field.label, edit=StudioNameSweepValue(
            kind="nameSweepValue", sweep_id=covered[0][0], value_id=edit.value_id,
            label=edit.label.strip() or None if edit.label else None,
        ))
        return sweep().edit(command, field.validate)
    if edit.kind == "to_list":
        if previous is None:
            raise ValueError(f"{field.label} is not swept.")
        command = StudioAuthoringSweepChange(
            field=field.label, edit=StudioSweepEdit(kind="toList", sweep_id=covered[0][0]),
        )
        return sweep().edit(command, field.validate)
    if edit.kind == "to_value":
        owner = sweep()
        named = owner.to_list(previous) if isinstance(previous, ListedAxis | RangeAxis | SampledAxis) else previous
        entry = named.values.get(edit.value_id) if isinstance(named, NamedAxis) and edit.value_id else None
        if entry is None or not owner._editable_list(named):
            raise ValueError(f"{field.label} has no such sweep value.")
        return _Rewrite([drop_axis(covered[0][0])], kept=entry.value, keep=owner.keep(entry.value))
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
        if scoped:
            raise ValueError(f"{field.label} has one sweep for every run.")
        owner = sweep()
        if isinstance(value, SweepList):
            listed = _list_axis(owner, [number(item) for item in value.members])
            return _Rewrite(owner.write_axis(listed), after=listed, keep=owner.release())
        path = owner.target.generated_path
        if path is None:
            raise ValueError(f"{field.label} writes more than one setting and takes a list.")
        if isinstance(value, SweepRange):
            generated = RangeAxis(path=path, range=RangeValues(
                start=number(value.first), stop=number(value.last), count=value.count, scale=value.spacing,
            ))
            for member in owner.to_list(generated).values.values():
                assert member is not None and type(member.value) in (float, int)
                field.validate(member.value)
            return _Rewrite(owner.write_axis(generated), after=named_axis(owner, generated), keep=owner.release())
        if field.duration and value.draw.family not in {"uniform", "loguniform", "normal", "choice", "truncexp"}:
            raise ValueError(f"{field.label} takes draws with scalar durations.")
        seed = edit.seed if edit.seed is not None else previous.sampler.seed if isinstance(previous, SampledAxis) else 0
        sampled = _sample_axis(value, field, functions, period, seed, path)
        return _Rewrite(owner.write_axis(sampled), after=named_axis(owner, sampled), keep=owner.release())
    if covered and not scoped:
        if len(covered) != 1:
            raise ValueError(f"{field.label} is swept by {len(covered)} sweeps.")
        assert axis is not None
        kept = number(value)
        return _Rewrite([drop_axis(covered[0][0])], kept=kept, keep=axis.keep(kept))
    if field.duration and not scoped:
        return field.duration_value(value)
    return number(value)
