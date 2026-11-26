"""Seed a bounded saved pipeline and read its status through the CLI."""

import json
import os
import sys
from functools import partial
from pathlib import Path

from feedbax.desktop.release import AppPaths
from feedbax.web.services.project_folder import ProjectFolderService
from tests.support.browser_seed_actions import HomeSeed, NonSeed


def seed(project: str, root: Path, *, variants: bool = False) -> None:
    from feedbax.contracts.project import Project
    from feedbax.application.project_writer import dump_text_project
    from feedbax.contracts.project_io import ProjectConfiguration
    from web.e2e.studio.fixtures.canonical_evaluate import _evaluation_project

    raw = _evaluation_project()
    raw["steps"] = {name: raw["steps"][name] for name in ("train", "conditions")}
    if variants:
        raw["sweep"] = {"rows": {"base": {}, "compare": {"from": "base"}}}
    for name, data in dump_text_project(
        Project.model_validate(raw), ProjectConfiguration(project=project, store="results"),
    ).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    document = ProjectFolderService(root.parent).open(project)
    print(json.dumps(document.snapshot.project_id))


def seed_home(instance: str, project: str, *, variants: bool = False) -> int:
    paths = AppPaths.resolve(instance_root=Path(instance))
    seed(project, paths.web / "projects" / project, variants=variants)
    return 0


def status(instance: str, project: str) -> int:
    from feedbax.cli import main as cli

    paths = AppPaths.resolve(instance_root=Path(instance))
    projects = paths.web / "projects"
    os.environ["FEEDBAX_HOST_ROOT"] = str(paths.host)
    _, location = ProjectFolderService(projects).open_native_id(project)
    return cli(["status", str(location.path), "--json"])


ACTIONS = {
    "seed": HomeSeed(seed_home),
    "seed-variants": HomeSeed(partial(seed_home, variants=True)),
    "status": NonSeed(status, "reads current Host-backed status through the CLI"),
}

if __name__ == "__main__":
    action, *arguments = sys.argv[1:]
    raise SystemExit(ACTIONS[action].command(*arguments))
