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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["seed", "read"])
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    if args.action == "seed":
        seed(args.root)
    else:
        reopened = load_text_project(args.root)
        model = reopened.project.documents["model"].payload
        print(json.dumps({"revision": reopened.revision, "gain": model["nodes"]["gain"]["params"]["gain"]}))
