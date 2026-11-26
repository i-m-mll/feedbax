"""Read the authored projects that ship with Feedbax."""

from dataclasses import dataclass
from importlib.resources import as_file, files

from feedbax.contracts.project import Project
from feedbax.contracts.project_io import PROJECT_DECLARATION_SUFFIX, load_text_project


@dataclass(frozen=True)
class ShippedExample:
    label: str
    project: Project


def load_shipped_examples() -> dict[str, ShippedExample]:
    """Return the bundled examples by their folder key."""
    examples = {}
    for folder in sorted(files(__name__).iterdir(), key=lambda entry: entry.name):
        if folder.is_dir() and any(child.name.endswith(PROJECT_DECLARATION_SUFFIX) for child in folder.iterdir()):
            with as_file(folder) as path:
                text = load_text_project(path)
            label = text.configuration.settings.get("example_label")
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"The shipped example {folder.name!r} needs an example label.")
            examples[folder.name] = ShippedExample(label, text.project)
    return examples


def load_shipped_projects() -> dict[str, Project]:
    return {key: example.project for key, example in load_shipped_examples().items()}
