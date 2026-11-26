"""Use the canonical nested model and Evaluate fixture in the mounted window."""

import json
from pathlib import Path
import sys

from pydantic import TypeAdapter

from feedbax.contracts.model_document import ModelComponent
from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.contracts.stored_model import stored_model_component
from feedbax.desktop.release import AppPaths
from tests.support.model_documents import changed_model
from tests.test_document_train_pair import authored_documents, document_project
from tests.test_model_document_plan import _add_attachment, _attachment_owner, _end, _model_project, _planar_arm


def add_physical_interior(raw, scope):
    raw["steps"].pop("baseline")
    interior = raw["documents"]["reach-model"]["payload"]["interior"]
    if scope == "equations":
        interior["components"]["equations"] = {
            "kind": "composite",
            "label": "Equations",
            "ports": {"physical": {"free": {"label": "Free end", "type": "translational"}}},
            "interior": {
                "physical_domain": "translational",
                "components": {"mass": {"kind": "leaf", "label": "Mass", "type": "feedbax.Mass", "fields": {"mass": 1.0}}},
            },
        }
        return

    arm = stored_model_component(TypeAdapter(ModelComponent).validate_python(_planar_arm())).model_dump(
        mode="json", exclude_defaults=True,
    )
    interior["components"]["arm"] = arm
    if scope == "mechanics-port":
        arm["ports"]["physical"] = {"mount": {"label": "Mount", "type": "frame"}}
        return

    interior["components"]["linear"] = {
        "kind": "leaf", "label": "Readout", "type": "feedbax.Linear",
        "fields": {"output_size": 6, "initialization": "zeros"},
    }
    interior["wires"].update(
        sample={"source": _end("arm", "pos"), "target": _end("linear", "input")},
        drive={"source": _end("linear", "output"), "target": _end("arm", "drive")},
    )


if __name__ == "__main__":
    home, folder_name, scope = sys.argv[1:]
    if scope == "managed-fields":
        raw = document_project(authored_documents())
        model = raw["documents"][raw["steps"]["train"]["base"]["inline"]["trial_documents"]["model"]["document"]]
        model["payload"] = changed_model(
            model["payload"], [_add_attachment(_attachment_owner("read"), "training-scale", None, 1.0)],
        )
    else:
        raw = _model_project()
    if scope in {"equations", "mechanics-port", "within-step"}:
        add_physical_interior(raw, scope)
    if scope == "variant":
        raw["sweep"] = {"rows": {"control": {}, "treated": {}}}
    elif scope == "sweep":
        raw["steps"]["baseline"]["sweep"] = {"axes": {"seed": {"values": {
            "control": {"label": "Control", "patches": {"params.seed": {"op": "replace", "value": 2}}},
            "comparison": {"label": "Comparison", "patches": {"params.seed": {"op": "replace", "value": 3}}},
        }}}}
    project = Project.model_validate(raw)
    paths = AppPaths.resolve(instance_root=Path(home))
    root = paths.web / "projects" / folder_name
    for name, data in dump_text_project(project, ProjectConfiguration(project=folder_name, store="results")).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print(json.dumps(folder_name))
