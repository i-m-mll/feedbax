"""Write selected-run trial edits using authored Project scopes."""

from hashlib import sha256

from pydantic import TypeAdapter

from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.matrix_core import apply_override_patches
from feedbax.contracts.model_edit import EvaluateAttachmentAxis, EvaluateAttachmentRows, EvaluateAttachmentVariant
from feedbax.contracts.project import InlineBase, SetProjectValue
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.project_sweep import NamedAxis, SetValue, RemoveValue, _compose_writes
from feedbax.contracts.run_matrix import MatrixCompositionDelta
from feedbax.contracts.studio_project import AuthoredFileAddress, StudioProjectFilePatch
from feedbax.web.services.project_folder import ProjectFolderInvalid


_SWEEP_WRITE = TypeAdapter(SetValue | RemoveValue)


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


def scoped_file(project, key, scope, writes):
    authored = project.steps[key.step]
    raw = authored.model_dump(mode="json", by_alias=True, exclude_unset=True)
    address = AuthoredFileAddress(kind="step", id=key.step)
    if isinstance(scope, EvaluateAttachmentVariant):
        writes = [
            OverridePatch(path=f"steps.{key.step}.base.inline.{patch.path}", op=patch.op, **(
                {"value": patch.value} if patch.op != "remove" else {}
            ))
            for patch in writes
        ]
        return variant_file(project, key.variant, writes)
    if isinstance(scope, EvaluateAttachmentAxis):
        axis = authored.sweep.axes.get(scope.axis) if authored.sweep else None
        if not isinstance(axis, NamedAxis) or axis.values.get(scope.value) is None:
            raise ProjectFolderInvalid("The group needs a named sweep value authored on this step.")
        parts = ("sweep", "axes", scope.axis, "values", scope.value)
        value = raw["sweep"]["axes"][scope.axis]["values"][scope.value]
        return file_value(address, raw, (*parts, "patches"), _store_patches(value.get("patches", {}), writes))
    if isinstance(scope, EvaluateAttachmentRows):
        sweep = raw.get("sweep") or {}
        rows = sorted(set(scope.rows))
        name = "trial-rows-" + sha256(canonical_json_v2_bytes(rows)).hexdigest()
        exceptions = sweep.get("exceptions", {})
        prior = exceptions.get(name, {"where": {"rows": rows}})
        if prior is None or prior.get("where") != {"rows": rows} or prior.get("exclude"):
            raise ProjectFolderInvalid("The selected row exception has an incompatible definition.")
        updated = prior | {"patches": _store_patches(prior.get("patches", {}), writes)}
        return file_value(address, raw, ("sweep",), sweep | {"exceptions": exceptions | {name: updated}})
    if isinstance(authored.base, InlineBase) and not authored.deltas:
        updated = apply_override_patches(authored.base.inline, writes)
        return file_value(address, raw, ("base", "inline"), updated)
    # Compose a local difference without modifying the inherited definition.
    layer = MatrixCompositionDelta(
        layer_id="trial-authoring", patches=writes, acknowledges_ancestor_paths=[patch.path for patch in writes],
    )
    return file_value(address, raw, ("deltas",), [*raw.get("deltas", []), layer.model_dump(mode="json", exclude_unset=True)])
