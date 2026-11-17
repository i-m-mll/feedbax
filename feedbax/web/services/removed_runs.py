"""Report the planned runs an authored edit takes out of the saved Plan."""

from feedbax.contracts.jobs import StudioSavedTrainKey
from feedbax.contracts.project_plan import compile_project
from feedbax.contracts.studio_removed_runs import StudioRemovedRuns, StudioRemovedRunsRequest
from feedbax.web.services.project_folder import ProjectFolderInvalid, ProjectFolderService, preview_authored_files


def removed_runs(folders: ProjectFolderService, project_id: str, body: StudioRemovedRunsRequest) -> StudioRemovedRuns:
    project = folders.read(project_id, revision=body.revision).project
    settled = folders.read_plan(project_id, revision=body.revision)
    try:
        candidate = compile_project(
            preview_authored_files(project, body.files), readers=settled.readers,
            operations=folders.operations, components=folders.components,
            functions=folders.functions, python_tasks=folders.python_tasks,
            figure_templates=folders.figure_templates,
        )
    except ValueError as error:
        raise ProjectFolderInvalid(str(error)) from error
    after = {run.key: run.request_id for run in candidate.runs}
    return StudioRemovedRuns(revision=body.revision, runs=[
        StudioSavedTrainKey(variant=run.key.variant, step=run.key.step, row=run.key.row)
        for run in settled.plan.runs if after.get(run.key) != run.request_id
    ])
