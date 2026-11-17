"""Prepare and reopen the browser proof's real Project folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project, load_text_project
from tests.support.managed_train_fixture import managed_train_project


def seed(root: Path) -> None:
    project = managed_train_project()
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
        model = reopened.project.documents["point-mass"].payload
        print(json.dumps({"revision": reopened.revision, "mass": model["interior"]["components"]["plant"]["fields"]["mass"]}))
