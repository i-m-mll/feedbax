"""Store an edit of a step's definition by the scope of the runs it is made for."""

from dataclasses import dataclass
from functools import partial
from hashlib import sha256

from pydantic import TypeAdapter

from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.matrix_core import apply_override_patches
from feedbax.contracts.model_edit import (
    EvaluateAttachmentAxis, EvaluateAttachmentRows, EvaluateAttachmentStep, EvaluateAttachmentVariant,
)
from feedbax.contracts.project import InlineBase, SetProjectValue
from feedbax.contracts.project_plan import RunKey
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.project_sweep import NamedAxis, SetValue, RemoveValue, _compose_writes
from feedbax.contracts.run_matrix import MatrixCompositionDelta
from feedbax.contracts.studio_project import AuthoredFileAddress, StudioProjectFilePatch
from feedbax.studio.selection import TrainSelection
from feedbax.web.services.project_folder import ProjectFolderInvalid


_SWEEP_WRITE = TypeAdapter(SetValue | RemoveValue)


@dataclass(frozen=True)
class SweepValuesScope:
    """Runs selected as values of the step's sweeps, other than one named value of one sweep."""

    axes: dict[str, tuple[str, ...]]


type RunScope = (
    EvaluateAttachmentStep | EvaluateAttachmentAxis | SweepValuesScope | EvaluateAttachmentRows | EvaluateAttachmentVariant
)


def selection_scope(project, plan, key, rows=(), axes=None) -> RunScope:
    """Where an edit made for the selected runs is stored.

    No rows is the step, or the variant outside the primary one. Rows that are exactly the runs of a set of values per
    sweep are those values: one named value's patches, or an exception over the values. Other rows are an exception over
    those rows.
    """
    axes = axes or {}
    if not rows:
        if axes:
            raise ProjectFolderInvalid("Sweep values were given without their runs.")
        if key.variant != plan.variants[0]:
            return EvaluateAttachmentVariant(kind="variant", variant=key.variant)
        return EvaluateAttachmentStep(kind="step")
    if key.variant != plan.variants[0]:
        raise ProjectFolderInvalid("A selection of runs is stored in the primary variant only.")
    available = {row.key.row: row for row in plan.rows if (row.key.variant, row.key.step) == (key.variant, key.step)}
    if not set(rows) <= available.keys() or len(set(rows)) != len(rows):
        raise ProjectFolderInvalid("The selected rows no longer belong to this step.")
    covered = {
        row_id for row_id, row in available.items()
        if all(axis in row.coordinates and row.coordinates[axis].value_id in values for axis, values in axes.items())
    }
    if not axes or set(rows) != covered:
        return EvaluateAttachmentRows(kind="rows", rows=tuple(rows))
    if len(axes) == 1:
        (axis, values), = axes.items()
        sweep = project.steps[key.step].sweep
        named = sweep.axes.get(axis) if sweep else None
        if len(values) == 1 and isinstance(named, NamedAxis) and named.values.get(values[0]) is not None:
            return EvaluateAttachmentAxis(kind="axis_value", axis=axis, value=values[0])
    return SweepValuesScope({axis: tuple(values) for axis, values in axes.items()})


def file_value(address, raw, parts, value):
    parent = raw
    for part in parts[:-1]:
        parent = parent[part]
    leaf = parts[-1]
    return StudioProjectFilePatch(address=address, patches=[
        SetProjectValue(op="replace" if leaf in parent else "add", path=pointer(parts), value=value),
    ])


def selected_runs(plan, key, scope, *, lit_steps=()):
    rows = [row for row in plan.rows if row.key.step == key.step]
    if isinstance(scope, EvaluateAttachmentVariant):
        selected = {row.key for row in rows if row.key.variant == scope.variant}
    elif isinstance(scope, EvaluateAttachmentAxis):
        selected = {
            row.key for row in rows
            if (coordinate := row.coordinates.get(scope.axis)) is not None and coordinate.value_id == scope.value
        }
    elif isinstance(scope, EvaluateAttachmentRows):
        selected = {row.key for row in rows if row.key.row in scope.rows}
        if {item.row for item in selected} != set(scope.rows):
            raise ProjectFolderInvalid("The selection contains an unknown run row.")
    else:
        selected = {
            row.key for row in plan.rows
            if row.key.step in (lit_steps or (key.step,))
        }
    if key not in selected or any(item.variant != key.variant for item in selected):
        raise ProjectFolderInvalid("The edit must address runs in the selected variant only.")
    return selected


def _store_patches(previous, writes):
    authored = {path: _SWEEP_WRITE.validate_python(value) for path, value in previous.items() if value is not None}
    changes = {}
    for patch in writes:
        changes[patch.path] = (
            RemoveValue(op="remove") if patch.op == "remove" else SetValue(op=patch.op, value=patch.value)
        )
    composed = _compose_writes(authored, changes)
    return {path: value for path, value in previous.items() if value is None and path not in changes} | {
        path: write.model_dump(mode="json", exclude_unset=True) for path, write in composed.items()
    }


def variant_file(project, variant, writes):
    row = project.sweep.rows.get(variant) if project.sweep else None
    if row is None:
        raise ProjectFolderInvalid("The variant needs an authored row.")
    raw = row.model_dump(mode="json", by_alias=True, exclude_unset=True)
    return file_value(AuthoredFileAddress(kind="variant", id=variant), raw, ("patches",),
                      _store_patches(raw.get("patches", {}), writes))


def sweep_value_file(address, raw, axis, value, writes):
    parts = ("sweep", "axes", axis, "values", value)
    authored = raw["sweep"]["axes"][axis]["values"][value]
    return file_value(address, raw, (*parts, "patches"), _store_patches(authored.get("patches", {}), writes))


def row_exception_file(address, raw, rows, writes, *, name_prefix):
    rows = sorted(set(rows))
    return _exception_file(address, raw, {"rows": rows}, rows, writes, name_prefix=name_prefix)


def values_exception_file(address, raw, axes, writes):
    where = {"axes": {axis: sorted(set(values)) for axis, values in sorted(axes.items())}}
    return _exception_file(address, raw, where, where, writes, name_prefix="values-")


def _exception_file(address, raw, where, identity, writes, *, name_prefix):
    sweep = raw.get("sweep") or {}
    name = name_prefix + sha256(canonical_json_v2_bytes(identity)).hexdigest()
    exceptions = sweep.get("exceptions", {})
    prior = exceptions.get(name, {"where": where})
    if prior is None or prior.get("where") != where or prior.get("exclude"):
        raise ProjectFolderInvalid("The selected exception has an incompatible definition.")
    updated = prior | {"patches": _store_patches(prior.get("patches", {}), writes)}
    return file_value(address, raw, ("sweep",), sweep | {"exceptions": exceptions | {name: updated}})


def scoped_file(project, step_id, scope: RunScope, writes) -> StudioProjectFilePatch:
    """The file that stores ``writes``, paths in the step's definition, for the runs ``scope`` selects."""
    authored = project.steps[step_id]
    raw = authored.model_dump(mode="json", by_alias=True, exclude_unset=True)
    address = AuthoredFileAddress(kind="step", id=step_id)
    if isinstance(scope, EvaluateAttachmentVariant):
        writes = [
            OverridePatch(path=f"steps.{step_id}.base.inline.{patch.path}", op=patch.op, **(
                {"value": patch.value} if patch.op != "remove" else {}
            ))
            for patch in writes
        ]
        return variant_file(project, scope.variant, writes)
    if isinstance(scope, EvaluateAttachmentAxis):
        axis = authored.sweep.axes.get(scope.axis) if authored.sweep else None
        if not isinstance(axis, NamedAxis) or axis.values.get(scope.value) is None:
            raise ProjectFolderInvalid("The group needs a named sweep value authored on this step.")
        return sweep_value_file(address, raw, scope.axis, scope.value, writes)
    if isinstance(scope, SweepValuesScope):
        return values_exception_file(address, raw, scope.axes, writes)
    if isinstance(scope, EvaluateAttachmentRows):
        return row_exception_file(address, raw, scope.rows, writes, name_prefix="trial-rows-")
    if isinstance(authored.base, InlineBase) and not authored.deltas:
        updated = apply_override_patches(authored.base.inline, writes)
        return file_value(address, raw, ("base", "inline"), updated)
    # Compose a local difference without modifying the inherited definition.
    layer = MatrixCompositionDelta(
        layer_id="trial-authoring", patches=writes, acknowledges_ancestor_paths=[patch.path for patch in writes],
    )
    return file_value(address, raw, ("deltas",), [*raw.get("deltas", []), layer.model_dump(mode="json", exclude_unset=True)])


def train_selection(folders, project_id, revision, project, step_id, run, rows=(), axes=None) -> TrainSelection:
    """The runs a Train edit names, read from the plan of ``revision``, and the write that stores an edit for them.

    Without rows the selection is every run of the step in the named run's variant.
    """
    if run is None:
        if rows or axes:
            raise ProjectFolderInvalid("Rows were given without the run they belong to.")
        return TrainSelection((), None, None, False, partial(scoped_file, project, step_id, EvaluateAttachmentStep(kind="step")))
    key = RunKey(run.variant, run.step, run.row)
    if key.step != step_id:
        raise ProjectFolderInvalid(f"Run {key.row!r} belongs to step {key.step!r}, not {step_id!r}.")
    plan = folders.read_plan(project_id, revision=revision, source_runs=(key,)).plan
    scope = selection_scope(project, plan, key, tuple(rows), axes)
    shown = tuple(RunKey(key.variant, step_id, row) for row in rows) or tuple(
        row.key for row in plan.rows if (row.key.variant, row.key.step) == (key.variant, step_id)
    )
    if set(shown) != {key}:
        plan = folders.read_plan(project_id, revision=revision, source_runs=tuple(dict.fromkeys((key, *shown)))).plan
    definitions = plan.selected_definitions
    payloads = tuple(definitions[item].payload for item in shown if item in definitions)
    scoped = not isinstance(scope, EvaluateAttachmentStep)
    named = definitions.get(key)
    return TrainSelection(
        payloads, next((row for row in plan.rows if row.key == key), None),
        named.payload if isinstance(scope, SweepValuesScope | EvaluateAttachmentAxis | EvaluateAttachmentRows)
        and named is not None else None, scoped,
        partial(scoped_file, project, step_id, scope),
    )
