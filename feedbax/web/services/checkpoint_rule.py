"""Prepare checkpoint-rule edits through the saved Plan and file editor."""

from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue

from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.jobs import StudioProjectWindowMapProblem
from feedbax.contracts.model_edit import AxisValueRuns, ListedRuns
from feedbax.contracts.project import Project, RemoveProjectValue, StepBase
from feedbax.contracts.project_io import project_declaration_name
from feedbax.contracts.project_inputs import InputBinding, StepReference, stored_checkpoint_rule
from feedbax.planner.project_plan import BlockedRow, PlannedRun, PlanSourceIndex, ProjectPlan, RunKey
from feedbax.contracts.project_sweep import (
    CHECKPOINT_RULE_TARGET, CheckpointRuleAxis, NamedAxis, authored_row, checkpoint_sweep_refusal,
)
from feedbax.contracts.request_refs import UpdateCadence
from feedbax.contracts.studio_checkpoint_rule import (
    StudioCheckpointRuleEdit, StudioCheckpointRuleRequest, StudioCheckpointRuleSet, StudioCheckpointRuleToSweep,
)
from feedbax.contracts.studio_project import AuthoredFileAddress, StudioProjectFilePatch
from feedbax.contracts.studio_scope import StudioStoreChoice
from feedbax.studio.selection import StoredEdit
from feedbax.web.services.authored_scope import choose_authored_store, choose_prepared_edit, preview_plan
from feedbax.web.services.axis_values import axis_values
from feedbax.web.services.project_folder import ProjectFolderInvalid, ProjectFolderService, preview_authored_files
from feedbax.web.services.store_choice import EditStore
from feedbax.web.services.trial_scope import file_value, reached_steps, row_exception_file, variant_file


@dataclass(frozen=True)
class _CheckpointSource:
    kind: Literal["step", "variant", "sweep_value", "exception", "row"]
    owner: str
    parts: tuple[str, ...] = ()

    def file(self, project: Project, rule: JsonValue) -> StudioProjectFilePatch:
        if self.kind == "variant":
            return variant_file(project, self.owner, [OverridePatch(
                path=".".join(self.parts), op="replace", value=rule,
            )])
        step = project.steps[self.owner]
        address = AuthoredFileAddress(kind="step", id=self.owner)
        raw = step.model_dump(mode="json", by_alias=True, exclude_unset=True)
        if self.kind == "step":
            return file_value(address, raw, self.parts, rule)
        target = raw
        for part in self.parts:
            target = target[part]
        patches = target.get("patches", {})
        return file_value(address, raw, (*self.parts, "patches"), patches | {
            CHECKPOINT_RULE_TARGET: {"op": "replace", "value": rule},
        })


def _ancestors(project: Project, step: str):
    seen = set()
    while step not in seen and step in project.steps:
        seen.add(step)
        yield step, project.steps[step]
        base = project.steps[step].base
        if not isinstance(base, StepBase):
            break
        step = base.step


def _subject_index(plan: ProjectPlan, key: RunKey, subjects: list[StepReference]) -> int | None:
    row = plan.row(key)
    subject = (
        row.inputs.get("subject") if isinstance(row, PlannedRun)
        else next(iter(row.inputs.get("subject", ())), None) if isinstance(row, BlockedRow)
        else None
    )
    if subject is not None and subject.selection in subjects:
        return subjects.index(subject.selection)
    for problem in plan.diagnostics:
        if (problem.variant, problem.step, problem.row) != (key.variant, key.step, key.row):
            continue
        match problem.path.split("."):
            case ["inputs", "subject", index, "checkpoint"] if index.isdigit() and int(index) < len(subjects):
                return int(index)
    return 0 if len(subjects) == 1 else None


def _step_source(
    project: Project, variant_project: Project, plan: ProjectPlan, key: RunKey, subject: InputBinding | None,
) -> _CheckpointSource | None:
    parts = ("inputs", "subject")
    if isinstance(subject, list):
        index = _subject_index(plan, key, subject)
        if index is None:
            return None
        parts = (*parts, str(index))
    elif not isinstance(subject, StepReference):
        return None
    parts = (*parts, "checkpoint")
    for name, step in _ancestors(variant_project, key.step):
        if "subject" not in step.inputs:
            continue
        path = ("steps", name, *parts)
        authored = authored_row(project.sweep, key.variant)
        if authored is not None:
            for length in range(len(path), 2, -1):
                origin = authored.patch_origins.get(".".join(path[:length]))
                if origin is not None and origin[0] == "row":
                    return _CheckpointSource("variant", origin[1], path)
        local = project.steps[name].inputs.get("subject")
        if (isinstance(local, StepReference) and len(parts) == 3
                or isinstance(local, list) and len(parts) == 4 and int(parts[2]) < len(local)):
            return _CheckpointSource("step", name, parts)
        break
    return None


def _checkpoint_sources(project: Project, plan: ProjectPlan) -> dict[RunKey, _CheckpointSource | None]:
    assert plan.source_snapshot is not None
    index = PlanSourceIndex(plan.source_snapshot)
    result = {}
    for row in plan.rows:
        if row.kind != "evaluate":
            continue
        key = row.key
        template, rows = index.expanded_step(key.variant, key.step)
        expansion = rows[key.row]
        origin = expansion.patch_origins.get(CHECKPOINT_RULE_TARGET)
        owner = expansion.patch_authors.get(CHECKPOINT_RULE_TARGET, key.step)
        source = None
        if origin is None:
            source = _step_source(project, plan.variant_projects[key.variant], plan, key, template.inputs.get("subject"))
        elif origin[0] == "sweep_value":
            _, axis, value = origin
            sweep = project.steps[owner].sweep
            if sweep is not None and isinstance(sweep.axes.get(axis), NamedAxis):
                source = _CheckpointSource("sweep_value", owner, ("sweep", "axes", axis, "values", value))
        elif origin[0] in ("exception", "row"):
            source = _CheckpointSource(
                origin[0], owner, ("sweep", "exceptions" if origin[0] == "exception" else "rows", origin[1]),
            )
        result[key] = source
    return result


def _named_source(project: Project, step: str, scope: AxisValueRuns) -> _CheckpointSource | None:
    for name, candidate in _ancestors(project, step):
        axis = candidate.sweep.axes.get(scope.axis) if candidate.sweep else None
        if isinstance(axis, NamedAxis) and axis.values.get(scope.value) is not None:
            return _CheckpointSource("sweep_value", name, ("sweep", "axes", scope.axis, "values", scope.value))
    return None


def _choose_checkpoint_store(
    project: Project, plan: ProjectPlan, key: RunKey, body: StudioCheckpointRuleRequest,
    edit: StudioCheckpointRuleSet,
) -> EditStore[StudioProjectFilePatch | None] | list[StudioStoreChoice]:
    rows = {row.key: row for row in plan.rows if (row.key.variant, row.key.step) == (key.variant, key.step)}
    ids = axis_values(plan.rows)
    scope = edit.scope
    if isinstance(scope, AxisValueRuns):
        selected = {item for item in rows if ids.get((item, scope.axis)) == scope.value}
    elif isinstance(scope, ListedRuns):
        selected = {item for item in rows if item.row in scope.rows}
        if len(selected) != len(scope.rows):
            raise ProjectFolderInvalid("The selected Evaluate rows changed.")
    else:
        selected = set(rows)
    if key not in selected:
        raise ProjectFolderInvalid("The selected run is outside the requested scope.")

    sources = _checkpoint_sources(project, plan)
    rule = stored_checkpoint_rule(edit.rule)
    preferred: EditStore[StudioProjectFilePatch | None] = EditStore(
        "other_runs", "Other Runs", key.step, frozenset(), None,
    )
    options: list[EditStore[StudioProjectFilePatch | None]] = []
    base_reads = {
        item for item in selected
        if (source := sources[item]) is not None and source.kind in ("step", "variant")
    }
    for source in dict.fromkeys(sources[item] for item in sorted(selected, key=lambda item: item.row)):
        if source is None:
            continue
        kind, label = {
            "step": ("whole_step", "Whole Step"), "variant": ("variant", "Variant"),
            "sweep_value": ("sweep_value", "Sweep Value"), "exception": ("row_exception", "Row Exception"),
            "row": ("row_exception", "Row Exception"),
        }[source.kind]
        reached = frozenset(item for item, owner in sources.items() if owner == source)
        option = EditStore(kind, label, f"{source.kind}:{source.owner}:{source.parts}", reached, source.file(project, rule))
        if source == sources[key] and not isinstance(scope, AxisValueRuns | ListedRuns):
            preferred = option
        elif source.kind not in ("step", "variant") or isinstance(scope, ListedRuns) and base_reads != selected:
            options.append(option)

    if isinstance(scope, AxisValueRuns) and (source := _named_source(project, key.step, scope)) is not None:
        reached = frozenset(item for item, prior in sources.items()
            if ids.get((item, scope.axis)) == scope.value
            and _named_source(project, item.step, scope) == source
            and prior is not None and (prior == source or prior.kind in ("step", "variant"))
        )
        preferred = EditStore(
            "sweep_value", "Sweep Value", f"{source.owner}:{source.parts}", reached, source.file(project, rule),
        )
        options = [option for option in options if option.value != preferred.value]
    elif base_reads and (isinstance(scope, ListedRuns) or preferred.runs != selected):
        keys = [item.row for item in base_reads]
        step = project.steps[key.step]
        raw = step.model_dump(mode="json", by_alias=True, exclude_unset=True)
        file = row_exception_file(AuthoredFileAddress(kind="step", id=key.step), raw, keys, [
            OverridePatch(path=CHECKPOINT_RULE_TARGET, op="replace", value=rule),
        ])
        reached = frozenset(item for item, owner in sources.items()
            if item.step in reached_steps(project, {key.step}) and item.row in keys
            and owner is not None and owner.kind in ("step", "variant")
        )
        selected_rows = EditStore("other_runs", "Selected Runs", f"{key.step}:{sorted(keys)}", reached, file)
        if isinstance(scope, ListedRuns) and base_reads == selected:
            preferred = selected_rows
        else:
            options.append(selected_rows)
    if not preferred.runs and len(options) == 1 and options[0].runs == selected:
        preferred, options = options[0], []
    return choose_authored_store(
        body.revision, body.model_dump(mode="json", exclude={"store_choice"}), selected, preferred, options,
        body.store_choice,
    )


def prepare_checkpoint_rule_edit(
    folders: ProjectFolderService, project_id: str, body: StudioCheckpointRuleRequest,
) -> StudioCheckpointRuleEdit:
    snapshot = folders.read(project_id, revision=body.revision)
    project = snapshot.project
    settled = folders.read_plan(project_id, revision=body.revision)
    converting = isinstance(body.edit, StudioCheckpointRuleToSweep)
    plan = settled.plan if converting else preview_plan(folders, settled, project, capture_sources=True)
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
        store = choose_prepared_edit(
            folders, project_id, body.revision, project, frozenset(selected),
            body.model_dump(mode="json", exclude={"store_choice"}), body.store_choice,
            ("whole_step", "Whole Step", key.step, StoredEdit([file])),
        )
        if isinstance(store, list):
            return StudioCheckpointRuleEdit(revision=body.revision, files=[], store_choices=store)
    else:
        decision = _choose_checkpoint_store(project, plan, key, body, edit)
        if isinstance(decision, list):
            return StudioCheckpointRuleEdit(revision=body.revision, files=[], store_choices=decision)
        assert decision.value is not None
        file, selected = decision.value, decision.runs
    try:
        candidate = preview_plan(folders, settled, preview_authored_files(project, [file]))
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
    return StudioCheckpointRuleEdit(revision=body.revision, files=[file], problems=problems)
