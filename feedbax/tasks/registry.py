"""Registry for objective-free Python tasks available to managed requests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel

from feedbax.contracts.expression_values import Problem, Site, ValueType
from feedbax.registry_errors import RegistryCollisionError

if TYPE_CHECKING:
    from feedbax.compiler.model_document import ResolvedModelInterface
    from feedbax.expression_registry import FunctionRegistry
    from feedbax.tasks.document import TrialSource


@dataclass(frozen=True)
class PythonTaskTemplate:
    signals: Mapping[str, ValueType]
    prepare: Callable[[ResolvedModelInterface, FunctionRegistry], TrialSource | tuple[Problem, ...]]
    condition_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PythonTaskDefinition:
    key: str
    label: str
    parameters: type[BaseModel]
    template: Callable[..., PythonTaskTemplate]
    parameter_sites: Mapping[str, Site]


class PythonTaskRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, PythonTaskDefinition] = {}
        self._sealed = False

    def register(self, definition: PythonTaskDefinition) -> None:
        if self._sealed:
            raise RuntimeError("Python task registry is sealed")
        if definition.key in self._entries:
            raise RegistryCollisionError(f"Duplicate Python task {definition.key!r}")
        self._entries[definition.key] = definition

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def resolve(self, key: str) -> PythonTaskDefinition | None:
        return self._entries.get(key)

    def seal(self) -> None:
        self._sealed = True


def python_task_parameter_site(path: str, trial_documents, registry: PythonTaskRegistry) -> Site | None:
    prefix = "trial_documents.python_task.params."
    if not path.startswith(prefix) or not isinstance(trial_documents, dict):
        return None
    selection = trial_documents.get("python_task")
    if not isinstance(selection, dict) or not isinstance(selection.get("type"), str):
        return None
    definition = registry.resolve(selection["type"])
    return definition.parameter_sites.get(path.removeprefix(prefix)) if definition else None
