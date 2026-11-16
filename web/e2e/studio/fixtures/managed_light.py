"""Seed a bounded saved pipeline and read its status through the CLI."""

import json
import os
import sys
from pathlib import Path

from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.desktop.release import AppPaths


def seed(project: str, root: Path, *, variants: bool = False) -> None:
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
    print(json.dumps(project))


def main(action: str, instance: str, project: str) -> int:
    paths = AppPaths.resolve(instance_root=Path(instance))
    root = paths.web / "projects" / project
    if action in {"seed", "seed-variants"}:
        seed(project, root, variants=action == "seed-variants")
        return 0
    if action == "status":
        from feedbax.cli import main as cli

        os.environ["FEEDBAX_HOST_ROOT"] = str(paths.host)
        return cli(["status", str(root), "--json"])
    raise ValueError(f"Unknown action {action!r}")


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
