"""A uv project whose package supplies an operation, and the Studio project beside it that uses the operation."""

from pathlib import Path
import shutil
import subprocess

from feedbax.contracts.project import Project
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.examples import load_shipped_projects
from tests.support.hang import HANG_SECONDS

REPO = Path(__file__).resolve().parents[2]
_FIXTURE = REPO / "tests" / "fixtures" / "downstream" / "environment_project"
OPERATION = "environment_fixture.speed"
PACKAGE_SOURCE = Path("src") / "environment_fixture"


def rollout_project(*operations: str) -> Project:
    """The shipped point mass, one rollout of it, and an Analyze step for each operation reading the rollout."""
    raw = load_shipped_projects()["point-mass-reaching"].model_dump(mode="json", by_alias=True, exclude_unset=True)
    task = raw["documents"]["simple-reaches-task"]["payload"]
    task["validation_trials"] = task["validation_trials"][:1]
    raw["steps"] = {
        "rollout": {
            "kind": "evaluate",
            "base": {"inline": {"evaluation_type": "feedbax.rollout", "params": {
                "seed": 17, "trial_policy": "paired", "variation": [], "trial_set": {"kind": "every_condition"},
            }}},
            "inputs": {
                "subject": {"kind": "initial_model", "source": {"document": "point-mass"}},
                "task": {"kind": "task", "source": {"document": "simple-reaches-task"}},
            },
        },
    } | {
        f"analysis-{index}": {
            "kind": "analyze",
            "base": {"inline": {
                "inputs": {"states": {"kind": "dataset"}},
                "operations": {"speed": {
                    "operation": operation, "params": {"velocity": "/v1/model/state/plant/effector~1velocity"},
                    "inputs": {"states": {"input": "states"}},
                }},
            }},
            "inputs": {"states": {"step": "rollout"}},
        }
        for index, operation in enumerate(operations)
    }
    return Project.model_validate(raw)


def write_project(folder: Path, project: Project) -> Path:
    for name, data in dump_text_project(project, ProjectConfiguration(project=folder.name, store="results")).items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return folder


def environment_project(root: Path, *, feedbax: Path = REPO) -> Path:
    """A locked uv project at ``root`` depending on ``feedbax`` by path, holding the Studio project ``project/``."""
    shutil.copytree(_FIXTURE / "src", root / "src")
    (root / "pyproject.toml").write_text((_FIXTURE / "pyproject.toml.in").read_text().replace("@FEEDBAX@", str(feedbax)))
    subprocess.run(
        ["uv", "lock", "--offline", "--project", str(root)],
        check=True, timeout=HANG_SECONDS, capture_output=True, text=True,
    )
    return write_project(root / "project", rollout_project(OPERATION))
