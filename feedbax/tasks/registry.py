"""Registry for objective-free Python tasks available to managed requests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from feedbax.contracts.expression_values import Problem, ProblemCode, Site, ValueType
from feedbax.contracts.task_document import PythonTaskSelection
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
class PythonTaskDefinition[Settings: BaseModel]:
    key: str
    label: str
    parameters: type[Settings]
    template: Callable[[Settings], PythonTaskTemplate]
    parameter_sites: Mapping[str, Site]


@dataclass(frozen=True)
class PythonTaskInstance[Parameters: BaseModel]:
    definition: PythonTaskDefinition[Parameters]
    parameters: Parameters

    def selection(self) -> PythonTaskSelection:
        return PythonTaskSelection(type=self.definition.key, params=self.parameters.model_dump(mode="json"))

    def template(self) -> PythonTaskTemplate:
        return self.definition.template(self.parameters)


class PythonTaskRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, PythonTaskDefinition[Any]] = {}
        self._sealed = False

    def register(self, definition: PythonTaskDefinition[Any]) -> None:
        if self._sealed:
            raise RuntimeError("Python task registry is sealed")
        if definition.key in self._entries:
            raise RegistryCollisionError(f"Duplicate Python task {definition.key!r}")
        self._entries[definition.key] = definition

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def resolve(self, key: str) -> PythonTaskDefinition[Any] | None:
        return self._entries.get(key)

    def seal(self) -> None:
        self._sealed = True


def resolve_python_task_instance(
    selection: PythonTaskSelection, registry: PythonTaskRegistry,
) -> PythonTaskInstance[BaseModel] | tuple[Problem, ...]:
    definition = registry.resolve(selection.type)
    if definition is None:
        return (Problem(code=ProblemCode.UNKNOWN_NAME, path=("python_task", "type"),
                        reason=f"Python task {selection.type!r} is not registered."),)
    try:
        parameters = definition.parameters.model_validate(selection.params)
    except ValidationError as error:
        return tuple(Problem(code=ProblemCode.TYPE_MISMATCH,
                             path=("python_task", "params", *item["loc"]), reason=item["msg"])
                     for item in error.errors(include_url=False))
    return PythonTaskInstance(definition, parameters)


def python_task_parameter_site(path: str, trial_documents, registry: PythonTaskRegistry) -> Site | None:
    prefix = "trial_documents.python_task.params."
    if not path.startswith(prefix) or not isinstance(trial_documents, dict):
        return None
    selection = trial_documents.get("python_task")
    if not isinstance(selection, dict) or not isinstance(selection.get("type"), str):
        return None
    definition = registry.resolve(selection["type"])
    return definition.parameter_sites.get(path.removeprefix(prefix)) if definition else None
