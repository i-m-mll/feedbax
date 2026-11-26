"""Open a folder with planned figures and inspect its saved Plan identities."""

from copy import deepcopy
import json
from pathlib import Path
import sys

from feedbax.contracts.analysis_document import ANALYSIS_SCHEMA
from feedbax.contracts.project import RootDocument, Step
from feedbax.contracts.project_sweep import Sweep
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.desktop.release import AppPaths
from feedbax.web.services.project_folder import ProjectFolderService
from tests.test_project_analyze_plan import _operations, _project

_FIXTURES = Path(__file__).parent


def seed(home: str, project_id: str):
    paths = AppPaths.resolve(instance_root=Path(home))
    root = paths.web / "projects" / project_id
    project = _project()
    project.sweep = Sweep.model_validate({"rows": {"base": {}, "compare": {"from": "base"}}})
    figure = project.documents["velocity-figure"].payload
    figure["caption"] = "Reach-aligned velocity"
    comparison = deepcopy(figure)
    comparison["caption"] = "Velocity comparison"
    views = project.documents["velocity-analysis"].payload["views"]
    views["profile"]["label"] = "Velocity profile"
    views["comparison"] = {**views["profile"], "label": "Velocity comparison", "figure": comparison}
    for name, data in dump_text_project(
        project, ProjectConfiguration(project=project_id, store="results"),
    ).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return ProjectFolderService(root.parent).open(project_id).snapshot.project_id


def seed_labels(home: str, project_id: str):
    """The Analyze project with a Report of its figure, and an unrelated Analysis document holding the key `held`."""
    paths = AppPaths.resolve(instance_root=Path(home))
    root = paths.web / "projects" / project_id
    project = _project()
    analysis = project.documents["velocity-analysis"].payload
    project.documents["other-analysis"] = RootDocument.model_validate({
        "schema": ANALYSIS_SCHEMA,
        "label": "Other analysis",
        "payload": {**analysis, "operations": {"held": {**analysis["operations"]["align"], "label": "Held"}}, "views": {}},
    })
    project.documents["velocity-report"] = RootDocument.model_validate({
        "schema": "feedbax.report_document.v1",
        "label": "Velocity report",
        "payload": json.loads((_FIXTURES / "velocity-summary.report.json").read_bytes()),
    })
    project.steps["report"] = Step.model_validate(json.loads((_FIXTURES / "velocity-report-step.json").read_bytes()))
    for name, data in dump_text_project(
        project, ProjectConfiguration(project=project_id, store="results"),
    ).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return ProjectFolderService(root.parent).open(project_id).snapshot.project_id


def requests(home: str, project_id: str, revision: str):
    paths = AppPaths.resolve(instance_root=Path(home))
    folders = ProjectFolderService(paths.web / "projects", operations=_operations())
    plan = folders.read_plan(project_id, revision=revision).plan
    return {
        "upstream": sorted([
            *(run.request_id for run in plan.runs if run.kind != "report"),
            *(element.request_id for element in plan.elements.values()),
        ]),
        "report": {run.key.variant: run.request_id for run in plan.runs if run.kind == "report"},
    }


if __name__ == "__main__":
    action, *args = sys.argv[1:]
    print(json.dumps({"seed": seed, "seed_labels": seed_labels, "requests": requests}[action](*args)))
