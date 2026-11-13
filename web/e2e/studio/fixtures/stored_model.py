"""Use the canonical nested model and Evaluate fixture in the mounted window."""

import json

from tests.test_model_document_plan import _model_project


if __name__ == "__main__":
    project = _model_project()
    print(json.dumps([
        {"kind": kind, "id": name, kind: value}
        for kind in ("document", "step")
        for name, value in project[f"{kind}s"].items()
    ]))
