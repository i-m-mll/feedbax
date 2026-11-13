"""Prepare checkpoint-rule edits through the saved Plan and file editor."""

from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.model_edit import EvaluateAttachmentAxis
from feedbax.contracts.project import RemoveProjectValue
from feedbax.contracts.project_inputs import StepReference
from feedbax.contracts.project_plan import RunKey, compile_project
from feedbax.contracts.project_sweep import CHECKPOINT_RULE_TARGET, CheckpointRuleAxis, NamedAxis
from feedbax.contracts.request_refs import UpdateCadence
from feedbax.contracts.studio_checkpoint_rule import (
    StudioCheckpointRuleEdit, StudioCheckpointRuleRequest, StudioCheckpointRuleToSweep,
)
from feedbax.contracts.studio_project import AuthoredFileAddress
from feedbax.web.services.project_folder import ProjectFolderInvalid, ProjectFolderService, preview_authored_files
from feedbax.web.services.trial_scope import file_value, row_exception_file, sweep_value_file


def prepare_checkpoint_rule_edit(
    folders: ProjectFolderService, project_id: str, body: StudioCheckpointRuleRequest,
) -> StudioCheckpointRuleEdit:
    snapshot = folders.read(project_id, revision=body.revision)
    project = snapshot.project
    settled = folders.read_plan(project_id, revision=body.revision)
    plan = settled.plan
    key = RunKey(body.run.variant, body.run.step, body.run.row)
    step = project.steps.get(key.step)
    planned = plan.steps.get((key.variant, key.step))
    if step is None or step.kind != "evaluate" or planned is None:
        raise ProjectFolderInvalid("Choose an Evaluate step.")
    if project.sweep is not None:
        raise ProjectFolderInvalid("Checkpoint sweep edits belong to the step definition.")
    rows = {row.key: row for row in plan.rows if row.key.step == key.step and row.key.variant == key.variant}
    if key not in rows:
        raise ProjectFolderInvalid("The selected Evaluate run changed.")
    address = AuthoredFileAddress(kind="step", id=key.step)
    raw = step.model_dump(mode="json", by_alias=True, exclude_unset=True)
    sweep = planned.sweep
    edit = body.edit
    if isinstance(edit, StudioCheckpointRuleToSweep):
        local = step.inputs.get("subject")
        if not isinstance(local, StepReference) or not isinstance(local.checkpoint, (tuple, UpdateCadence)):
            raise ProjectFolderInvalid("Choose a locally authored checkpoint list or cadence.")
        if sweep is not None and (sweep.combination not in (None, "cross") or any(sweep.rows.values())):
            raise ProjectFolderInvalid("To Sweep requires a cross of generated rows.")
        if sweep is not None and "update" in sweep.axes:
            raise ProjectFolderInvalid("This step already has an update axis.")
        authored_sweep = raw.get("sweep") or {}
        axis = CheckpointRuleAxis(checkpoint_rule=local.checkpoint)
        axes = authored_sweep.get("axes", {}) | {"update": axis.model_dump(mode="json", by_alias=True)}
        file = file_value(address, raw, ("sweep",), authored_sweep | {"axes": axes})
        file.patches.append(RemoveProjectValue(op="remove", path="/inputs/subject/checkpoint"))
        selected = set(rows)
    else:
        scope = edit.scope
        rule = edit.model_dump(mode="json", by_alias=True, exclude_none=True)["rule"]
        writes = [OverridePatch(path=CHECKPOINT_RULE_TARGET, op="replace", value=rule)]
        if isinstance(scope, EvaluateAttachmentAxis):
            selected = {
                row_key for row_key, row in rows.items()
                if (coordinate := row.coordinates.get(scope.axis)) is not None and coordinate.value_id == scope.value
            }
            axis = step.sweep.axes.get(scope.axis) if step.sweep else None
            if isinstance(axis, NamedAxis) and axis.values.get(scope.value) is not None:
                file = sweep_value_file(address, raw, scope.axis, scope.value, writes)
            else:
                raise ProjectFolderInvalid("Choose a named sweep value authored on this step.")
        else:
            selected = {row_key for row_key in rows if row_key.row in scope.rows}
            if len(selected) != len(scope.rows):
                raise ProjectFolderInvalid("The selected Evaluate rows changed.")
            file = row_exception_file(address, raw, scope.rows, writes, name_prefix="checkpoint-rows-")
        if key not in selected:
            raise ProjectFolderInvalid("The selected run is outside the requested scope.")
    try:
        candidate = compile_project(
            preview_authored_files(project, [file]), readers=settled.readers,
            operations=folders.operations, components=folders.components,
            functions=folders.functions, python_tasks=folders.python_tasks,
        )
    except ValueError as error:
        raise ProjectFolderInvalid(str(error)) from error
    problems = set(plan.diagnostics)
    for problem in candidate.diagnostics:
        if problem not in problems and problem.step == key.step and problem.variant == key.variant:
            raise ProjectFolderInvalid(problem.detail)
    before = {run.key: run.request_id for run in plan.runs}
    after = {run.key: run.request_id for run in candidate.runs}
    converting = isinstance(edit, StudioCheckpointRuleToSweep)
    if not converting and (before.keys() & selected) - after.keys():
        raise ProjectFolderInvalid("The checkpoint edit removes a selected Evaluate run.")
    protected = {
        run.key for run in (*plan.runs, *candidate.runs)
        if run.kind == "train" or (
            run.key.step == key.step and (
                run.key.variant != key.variant or not converting
            )
        )
    } - selected
    if any(before.get(row) != after.get(row) for row in protected):
        raise ProjectFolderInvalid("The checkpoint edit changes runs outside the selected scope.")
    return StudioCheckpointRuleEdit(revision=body.revision, files=[file])
