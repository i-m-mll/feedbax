"""Open a folder with planned figures and inspect its saved Plan identities."""

from copy import deepcopy
import json
from pathlib import Path
import sys

from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.desktop.release import AppPaths
from feedbax.web.services.project_folder import ProjectFolderService
from tests.test_project_analyze_plan import _operations, _project


def seed(home: str, project_id: str):
    paths = AppPaths.resolve(instance_root=Path(home))
    root = paths.web / "projects" / project_id
    project = _project()
    figure = project.documents["velocity-figure"].payload
    figure["caption"] = "Reach-aligned velocity"
    comparison = deepcopy(figure)
    comparison["caption"] = "Velocity comparison"
    views = project.documents["velocity-analysis"].payload["views"]
    views["comparison"] = {**views["profile"], "figure": comparison}
    for name, data in dump_text_project(
        project, ProjectConfiguration(project=project_id, store="results"),
    ).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return project_id


def requests(home: str, project_id: str, revision: str):
    paths = AppPaths.resolve(instance_root=Path(home))
    folders = ProjectFolderService(paths.web / "projects", operations=_operations())
    plan = folders.read_plan(project_id, revision=revision).plan
    return {
        "upstream": sorted([
            *(run.request_id for run in plan.runs if run.kind != "report"),
            *(element.request_id for element in plan.elements.values()),
        ]),
        "report": next(run.request_id for run in plan.runs if run.kind == "report"),
    }


if __name__ == "__main__":
    action, *args = sys.argv[1:]
    print(json.dumps({"seed": seed, "requests": requests}[action](*args)))
