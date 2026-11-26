"""Prepare and reopen the browser proof's real Project folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project, load_text_project
from tests.support.browser_seed_actions import FolderSeed, NonSeed
from tests.support.managed_train_fixture import managed_train_project


def seed(root: Path) -> None:
    project = managed_train_project()
    root.mkdir(parents=True)
    for relative, data in dump_text_project(project, ProjectConfiguration(project=root.name, store="results")).items():
        path = root / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)


def read(root: Path) -> None:
    reopened = load_text_project(root)
    train = reopened.project.steps["train"].base.inline
    print(json.dumps({"revision": reopened.revision, "batch": train["training_config"]["batch_size"]}))


ACTIONS = {
    "seed": FolderSeed(seed),
    "read": NonSeed(read, "reads the existing project after the browser edits it"),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=ACTIONS)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    ACTIONS[args.action].command(args.root)
