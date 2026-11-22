"""Class-owned component histories across current integer revisions."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from pathlib import Path
import inspect

from feedbax.contracts.migrations import ComponentMigration

Transform = Callable[[dict[str, Any]], dict[str, Any]]


def require_revision(value: Any) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"Component revision must be a strictly positive integer: {value!r}")
    return value


@dataclass(frozen=True)
class Change:
    destination: int
    migrate: Transform | None = None
    renamed_from: str | None = None
    identity: bool = False

    def __post_init__(self):
        require_revision(self.destination)
        if self.destination < 2 or not (self.migrate or self.renamed_from or self.identity):
            raise ValueError(
                "A Change requires a destination >= 2 and a transform, rename, or explicit identity"
            )
        if self.identity and self.migrate is not None:
            raise ValueError("A Change cannot declare both identity and a transform")
        if self.renamed_from is not None and (not self.renamed_from or "." in self.renamed_from):
            raise ValueError("renamed_from must be an old local class name")


@dataclass(frozen=True, init=False)
class History:
    changes: tuple[Change, ...]
    source: Path | None

    def __init__(self, *changes: Change):
        if tuple(change.destination for change in changes) != tuple(range(2, len(changes) + 2)):
            raise ValueError(
                "Change destinations must be consecutive explicit integers starting at 2"
            )
        frame = inspect.currentframe().f_back
        filename = frame.f_code.co_filename
        object.__setattr__(
            self, "source", None if filename.startswith("<") else Path(filename).resolve()
        )
        object.__setattr__(self, "changes", changes)

    @property
    def revision(self):
        return len(self.changes) + 1

    def endpoints(self, namespace: str, current_name: str):
        names = {self.revision: current_name}
        name = current_name
        renamed_names = {name}
        for change in reversed(self.changes):
            if change.renamed_from is not None:
                if change.renamed_from in renamed_names:
                    raise ValueError("Component history contains a cyclic or redundant rename")
                renamed_names.add(change.renamed_from)
                name = change.renamed_from
            names[change.destination - 1] = name
        return {revision: (f"{namespace}.{name}", revision) for revision, name in names.items()}

    def compile(self, namespace: str, current_name: str):
        endpoints = self.endpoints(namespace, current_name)
        edges = []
        for change in self.changes:
            source_type, source_revision = endpoints[change.destination - 1]
            target_type, target_revision = endpoints[change.destination]
            edges.append(
                ComponentMigration(
                    source_type=source_type,
                    target_type=target_type,
                    owner=namespace,
                    source_param_schema_version=source_revision,
                    target_param_schema_version=target_revision,
                    migration_id=f"{namespace}.{current_name}.revision.{change.destination}",
                    migrate_params=change.migrate,
                )
            )
        return tuple(edges)
