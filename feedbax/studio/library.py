"""Libraries of documents: folders in a project's layout that installed packages ship."""

from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable

from feedbax.contracts.project import DOCUMENT, Document, RootDocument
from feedbax.contracts.project_resolution import ProjectDiagnostic, compose_document
from feedbax.contracts.strict_json import strict_json_loads
from feedbax.plugins.bootstrap import installed_packages
from feedbax.plugins.declarations import DistributionSource

_EXAMPLES = "examples"
"""The folder of a package that holds its libraries, one folder each."""


@dataclass(frozen=True)
class LibrarySource:
    """A folder of documents in a project's layout, ``name`` within the distribution that ships it.

    An entry is read whole: a document based on another in the folder is given its composed payload, so taking it
    into a project carries nothing that points back into the library.
    """

    distribution: str
    name: str
    folder: Traversable

    def documents(self) -> dict[str, RootDocument | ProjectDiagnostic]:
        stored = self._stored()
        return {name: compose_document(stored.get, name) for name in sorted(stored)}

    def _stored(self) -> dict[str, Document]:
        folder = self.folder / "documents"
        return {
            entry.name.removesuffix(".json"): DOCUMENT.validate_python(
                strict_json_loads(entry.read_bytes(), ref=f"{self.distribution}/{self.name}/{entry.name}"),
            )
            for entry in folder.iterdir() if entry.is_file() and entry.name.endswith(".json")
        }


def library_sources(packages: tuple[DistributionSource, ...]) -> tuple[LibrarySource, ...]:
    """The libraries of ``packages``: each folder under a package's ``examples`` that holds ``documents``."""
    sources = []
    for package in packages:
        root = files(package.package.partition(".")[0]) / _EXAMPLES
        if not root.is_dir():
            continue
        for folder in sorted(root.iterdir(), key=lambda entry: entry.name):
            if folder.is_dir() and (folder / "documents").is_dir():
                sources.append(LibrarySource(package.distribution, folder.name, folder))
    return tuple(sources)


FIRST_TRAIN_TASK = ("point-mass-reaching", "simple-reaches-task")
"""The library and name of the task document a project's first Train reads: feedbax's simple reaches."""


def first_train_task() -> RootDocument:
    library, name = FIRST_TRAIN_TASK
    (source,) = (
        source for source in installed_library_sources() if source.distribution == "feedbax" and source.name == library
    )
    document = source.documents()[name]
    if isinstance(document, ProjectDiagnostic):
        raise ValueError(document.detail)
    return document


def installed_library_sources() -> tuple[LibrarySource, ...]:
    return library_sources(installed_packages())
