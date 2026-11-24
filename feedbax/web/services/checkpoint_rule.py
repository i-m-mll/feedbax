"""Prepare checkpoint-rule edits through the saved Plan and file editor."""

from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.jobs import StudioProjectWindowMapProblem
from feedbax.contracts.model_edit import AxisValueRuns
from feedbax.contracts.project import RemoveProjectValue
from feedbax.contracts.project_io import project_declaration_name
from feedbax.contracts.project_inputs import StepReference
from feedbax.planner.project_plan import RunKey, compile_project
from feedbax.contracts.project_sweep import CHECKPOINT_RULE_TARGET, CheckpointRuleAxis, NamedAxis, checkpoint_sweep_refusal
from feedbax.contracts.request_refs import UpdateCadence
from feedbax.contracts.studio_checkpoint_rule import (
    StudioCheckpointRuleEdit, StudioCheckpointRuleRequest, StudioCheckpointRuleToSweep,
)
from feedbax.contracts.studio_project import AuthoredFileAddress
from feedbax.web.services.axis_values import axis_values
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
    rows = {row.key: row for row in plan.rows if row.key.step == key.step and row.key.variant == key.variant}
    if key not in rows:
        raise ProjectFolderInvalid("The selected Evaluate run changed.")
    address = AuthoredFileAddress(kind="step", id=key.step)
    raw = step.model_dump(mode="json", by_alias=True, exclude_unset=True)
    sweep = planned.sweep
    edit = body.edit
    converting = isinstance(edit, StudioCheckpointRuleToSweep)
    if converting:
        local = step.inputs.get("subject")
        if refusal := checkpoint_sweep_refusal(local, sweep, project_swept=project.sweep is not None):
            raise ProjectFolderInvalid(refusal)
        assert isinstance(local, StepReference) and isinstance(local.checkpoint, (tuple, UpdateCadence))
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
        if isinstance(scope, AxisValueRuns):
            ids = axis_values(rows.values())
            selected = {row_key for row_key in rows if ids.get((row_key, scope.axis)) == scope.value}
            axis = step.sweep.axes.get(scope.axis) if step.sweep else None
            if isinstance(axis, NamedAxis) and axis.values.get(scope.value) is not None:
                file = sweep_value_file(address, raw, scope.axis, scope.value, writes)
            else:
                raise ProjectFolderInvalid("Choose a named sweep value authored on this step.")
        else:
            selected = {row_key for row_key in rows if row_key.row in scope.rows}
            if len(selected) != len(scope.rows):
                raise ProjectFolderInvalid("The selected Evaluate rows changed.")
            file = row_exception_file(address, raw, scope.rows, writes)
        if key not in selected:
            raise ProjectFolderInvalid("The selected run is outside the requested scope.")
    try:
        candidate = compile_project(
            preview_authored_files(project, [file]), readers=settled.readers,
            operations=folders.operations, components=folders.components,
            functions=folders.functions, python_tasks=folders.python_tasks,
            figure_templates=folders.figure_templates, training_methods=folders.training_methods,
        )
    except ValueError as error:
        raise ProjectFolderInvalid(str(error)) from error
    problems = []
    declaration = project_declaration_name(settled.configuration.project)
    for problem in candidate.diagnostics:
        if (not converting and problem.row is not None and problem.step is not None and problem.variant is not None
            and RunKey(problem.variant, problem.step, problem.row) in selected
            and problem.path.startswith("inputs.subject.") and problem.path.endswith(".checkpoint")):
            problems.append(StudioProjectWindowMapProblem.from_diagnostic(problem, declaration=declaration))
        elif problem not in plan.diagnostics and problem.step == key.step and problem.variant == key.variant:
            raise ProjectFolderInvalid(problem.detail)
    before = {run.key: run.request_id for run in plan.runs}
    after = {run.key: run.request_id for run in candidate.runs}
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
    return StudioCheckpointRuleEdit(revision=body.revision, files=[file], problems=problems)
