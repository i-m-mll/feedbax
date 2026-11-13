"""Use the canonical nested model and Evaluate fixture in the mounted window."""

import json
from pathlib import Path
import sys

from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.desktop.release import AppPaths
from tests.test_model_document_plan import _model_project


if __name__ == "__main__":
    home, project_id, scope = sys.argv[1:]
    raw = _model_project()
    if scope == "variant":
        raw["sweep"] = {"rows": {"control": {}, "treated": {}}}
    elif scope == "sweep":
        raw["steps"]["baseline"]["sweep"] = {"axes": {"seed": {"values": {
            "control": {"patches": {"params.seed": {"op": "replace", "value": 2}}},
            "comparison": {"patches": {"params.seed": {"op": "replace", "value": 3}}},
        }}}}
    project = Project.model_validate(raw)
    paths = AppPaths.resolve(instance_root=Path(home))
    root = paths.web / "projects" / project_id
    for name, data in dump_text_project(project, ProjectConfiguration(project=project_id, store="results")).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print(json.dumps(project_id))
