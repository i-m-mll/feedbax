"""Read the authored projects that ship with Feedbax."""

from importlib.resources import as_file, files

from feedbax.contracts.project import Project
from feedbax.contracts.project_io import PROJECT_DECLARATION_SUFFIX, load_text_project


def load_shipped_projects() -> dict[str, Project]:
    """Return the bundled projects by their folder name."""
    projects = {}
    for folder in sorted(files(__name__).iterdir(), key=lambda entry: entry.name):
        if folder.is_dir() and any(child.name.endswith(PROJECT_DECLARATION_SUFFIX) for child in folder.iterdir()):
            with as_file(folder) as path:
                projects[folder.name] = load_text_project(path).project
    return projects
