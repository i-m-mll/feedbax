"""A uv project whose package supplies an operation, and the Studio project beside it that uses the operation."""

from itertools import islice
from pathlib import Path
import shutil
import subprocess

from feedbax.contracts.project import Project
from feedbax.application.project_writer import dump_text_project
from feedbax.contracts.project_io import ProjectConfiguration
from feedbax.examples import load_shipped_projects
from tests.support.hang import HANG_SECONDS

REPO = Path(__file__).resolve().parents[2]
_FIXTURE = REPO / "tests" / "fixtures" / "downstream" / "environment_project"
OPERATION = "environment_fixture.speed"
COMPONENT = "environment_fixture.Readout"
PACKAGE_SOURCE = Path("src") / "environment_fixture"


def _keep_first_validation_trial(task: dict) -> None:
    task["validation_trials"] = dict(islice(task["validation_trials"].items(), 1))


def rollout_project(*operations: str) -> Project:
    """The shipped point mass, one rollout of it, and an Analyze step for each operation reading the rollout."""
    raw = load_shipped_projects()["point-mass-reaching"].model_dump(mode="json", by_alias=True, exclude_unset=True)
    _keep_first_validation_trial(raw["documents"]["simple-reaches-task"]["payload"])
    raw["steps"] = {
        "rollout": {
            "kind": "evaluate",
            "label": "Rollout",
            "base": {"inline": {"evaluation_type": "feedbax.rollout", "params": {
                "seed": 17, "trial_policy": "paired", "variation": [], "trial_set": {"kind": "every_condition"},
            }}},
            "inputs": {
                "subject": {"kind": "initial_model", "source": {"document": "point-mass"}},
                "task": {"kind": "task", "source": {"document": "simple-reaches-task"}},
            },
        },
    } | {f"analysis-{index}": analysis_step(operation, "rollout", f"Analysis {index + 1}") for index, operation in enumerate(operations)}
    return Project.model_validate(raw)


def analysis_step(operation: str, source: str, label: str = "Analysis") -> dict:
    return {
        "kind": "analyze",
        "label": label,
        "base": {"inline": {
            "inputs": {"states": {"kind": "dataset"}},
            "operations": {"speed": {
                "label": "Speed",
                "operation": operation, "params": {"velocity": "/v1/model/state/plant/effector~1velocity"},
                "inputs": {"states": {"input": "states"}},
            }},
        }},
        "inputs": {"states": {"step": source}},
    }


def trained_project() -> Project:
    """A model whose readout is the package's component, trained for two updates, evaluated, and analyzed by the
    package's operation."""
    from tests.support.managed_train_fixture import managed_evaluation_step, managed_train_project

    raw = managed_train_project().model_dump(mode="json", by_alias=True, exclude_unset=True)
    sources = raw["steps"]["train"]["base"]["inline"]["trial_documents"]
    model = raw["documents"][sources["model"]["document"]]["payload"]
    _keep_first_validation_trial(raw["documents"][sources["task"]["document"]]["payload"])
    components = model["interior"]["components"]
    components["readout"]["type"] = COMPONENT
    components["readout"]["fields"]["input_size"] = components["cell"]["fields"]["hidden_size"]
    raw["steps"] = {
        "train": raw["steps"]["train"],
        "evaluate": managed_evaluation_step("train"),
        "analysis": analysis_step(OPERATION, "evaluate"),
    }
    return Project.model_validate(raw)


def write_project(folder: Path, project: Project) -> Path:
    for name, data in dump_text_project(project, ProjectConfiguration(project=folder.name, store="results")).items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return folder


def environment_project(root: Path, project: Project | None = None, *, feedbax: Path = REPO) -> Path:
    """A locked uv project at ``root`` depending on ``feedbax`` by path, holding the Studio project ``project/``."""
    shutil.copytree(_FIXTURE / "src", root / "src")
    (root / "pyproject.toml").write_text((_FIXTURE / "pyproject.toml.in").read_text().replace("@FEEDBAX@", str(feedbax)))
    subprocess.run(
        ["uv", "lock", "--offline", "--project", str(root)],
        check=True, timeout=HANG_SECONDS, capture_output=True, text=True,
    )
    return write_project(root / "project", project or rollout_project(OPERATION))
