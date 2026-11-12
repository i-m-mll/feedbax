"""Prepare and reopen the browser proof's real Project folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project, load_text_project


def seed(root: Path) -> None:
    corpus_path = Path(__file__).resolve().parents[4] / "tests/fixtures/request_identity_corpus.json"
    corpus = json.loads(corpus_path.read_bytes())
    request = next(row["spec"] for row in corpus["requests"] if row["kind"] == "train")
    model = request.pop("graph")
    project = Project.model_validate({
        "documents": {"model": {"schema": model["schema_id"], "payload": model}},
        "steps": {"fit": {"kind": "train", "base": {"inline": {"graph": {"document": "model"}, **request}}}},
    })
    root.mkdir(parents=True)
    for relative, data in dump_text_project(project, ProjectConfiguration(project=root.name, store="results")).items():
        path = root / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)


def inspect_authored_train(root: Path) -> dict:
    from feedbax.contracts.graph import GraphSpec
    from feedbax.contracts.project_plan import compile_project
    from feedbax.contracts.project_resolution import ProjectDiagnostic, resolve_step, validate_project_documents
    from feedbax.contracts.training import TaskSpec

    text = load_text_project(root)
    project = text.project
    assert not validate_project_documents(project, {
        "feedbax.spec.graph": GraphSpec.model_validate,
        "feedbax.spec.task": TaskSpec.model_validate,
    })
    step = resolve_step(project, "train")
    assert not isinstance(step, ProjectDiagnostic), step
    assert step.payload == {field: project.documents[name].payload for field, name in (
        ("graph", "model"), ("task", "task"),
    )}
    plan = compile_project(project)
    return {
        "revision": text.revision,
        "runs": len(plan.runs),
        "diagnostics": sorted([problem.code, problem.path] for problem in plan.diagnostics),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["seed", "read", "inspect-authoring"])
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    if args.action == "seed":
        seed(args.root)
    elif args.action == "inspect-authoring":
        print(json.dumps(inspect_authored_train(args.root)))
    else:
        reopened = load_text_project(args.root)
        model = reopened.project.documents["model"].payload
        print(json.dumps({"revision": reopened.revision, "gain": model["nodes"]["gain"]["params"]["gain"]}))
