"""Use the canonical nested model and Evaluate fixture in the mounted window."""

import json
from pathlib import Path
import sys

from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.desktop.release import AppPaths
from tests.test_document_train_pair import authored_documents, document_project
from tests.test_model_document_plan import _add_attachment, _attachment_owner, _end, _model_project, _planar_arm, _wire


def add_physical_interior(raw, scope):
    raw["steps"].pop("baseline")
    interior = raw["documents"]["reach-model"]["payload"]["interior"]
    if scope == "equations":
        interior["components"]["equations"] = {
            "kind": "composite",
            "name": "Equations",
            "ports": {"physical": [{"id": "free", "name": "Free end", "type": "translational"}]},
            "interior": {
                "physical_domain": "translational",
                "components": {"mass": {"kind": "leaf", "name": "Mass", "type": "feedbax.Mass", "fields": {"mass": 1.0}}},
            },
        }
        return

    arm = _planar_arm()
    interior["components"]["arm"] = arm
    if scope == "mechanics-port":
        arm["ports"]["physical"] = [{"id": "mount", "name": "Mount", "type": "frame"}]
        return

    interior["components"]["linear"] = {
        "kind": "leaf", "name": "Readout", "type": "feedbax.Linear",
        "fields": {"input_size": 2, "output_size": 6, "initialization": "zeros"},
    }
    interior["wires"].update(
        sample=_wire(_end("arm", "pos"), _end("linear", "input")),
        drive=_wire(_end("linear", "output"), _end("arm", "drive")),
    )


if __name__ == "__main__":
    home, project_id, scope = sys.argv[1:]
    if scope == "managed-fields":
        raw = document_project(authored_documents())
        raw["steps"]["train"]["model_attachments"] = {
            "wire": [_add_attachment(_attachment_owner("read"), "training-scale", None, 1.0)],
        }
    else:
        raw = _model_project()
    if scope in {"equations", "mechanics-port", "within-step"}:
        add_physical_interior(raw, scope)
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
