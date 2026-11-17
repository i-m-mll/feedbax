"""Registry for objective-free Python tasks available to managed requests."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel

from feedbax.contracts.expression_values import Problem, Site, ValueType
from feedbax.plugins.declarations import (
    CallableDeclaration, DeclarationRegistry, DistributionSource, ResolvedDeclaration,
)

if TYPE_CHECKING:
    from feedbax.compiler.model_document import ResolvedModelInterface
    from feedbax.expression_registry import FunctionRegistry
    from feedbax.tasks.document import TrialSource


@dataclass(frozen=True)
class PythonTaskTemplate:
    signals: Mapping[str, ValueType]
    prepare: Callable[[ResolvedModelInterface, FunctionRegistry], TrialSource | tuple[Problem, ...]]
    condition_ids: tuple[str, ...] = ()


class PythonTask[Settings: BaseModel](ABC):
    """Defines the trial interface and preparation for a task authored in Python."""

    key: ClassVar[str | None] = None
    label: ClassVar[str | None] = None
    settings_model: type[Settings]
    renamed_from: ClassVar[tuple[str, ...]] = ()
    parameter_sites: ClassVar[Mapping[str, Site]] = {}

    @staticmethod
    @abstractmethod
    def template(settings: Settings) -> PythonTaskTemplate: ...


@dataclass(frozen=True)
class PythonTaskDefinition[Settings: BaseModel]:
    key: str
    label: str
    parameters: type[Settings]
    template: Callable[[Settings], PythonTaskTemplate]
    parameter_sites: Mapping[str, Site]
    source: DistributionSource
    renamed_from: tuple[str, ...]


class PythonTaskRegistry:
    def __init__(self) -> None:
        self._declarations = DeclarationRegistry[PythonTaskDefinition[Any]]("Python task")

    def register_declaration[Settings: BaseModel](
        self, task: type[PythonTask[Settings]], *, key: str, source: DistributionSource,
    ) -> None:
        self._declarations.require_mutable()
        metadata = vars(task)
        declaration = CallableDeclaration(
            key=key, label=metadata.get("label"), settings_model=task.settings_model,
            renamed_from=metadata.get("renamed_from", ()),
        )
        entry = PythonTaskDefinition(
            key=key,
            label=declaration.label or task.__name__,
            parameters=task.settings_model,
            template=task.template,
            parameter_sites=dict(task.parameter_sites),
            source=source,
            renamed_from=declaration.renamed_from,
        )
        self._declarations.add(key, source, task, entry, renamed_from=entry.renamed_from)

    def keys(self) -> tuple[str, ...]:
        return self._declarations.keys()

    def declaration_record(self, key: str) -> ResolvedDeclaration[PythonTaskDefinition[Any]]:
        return self._declarations.resolve(key)

    def resolve(self, key: str) -> PythonTaskDefinition[Any] | None:
        try:
            return self.declaration_record(key).value
        except KeyError:
            return None

    def seal(self) -> None:
        self._declarations.seal()


def python_task_parameter_site(path: str, trial_documents, registry: PythonTaskRegistry) -> Site | None:
    prefix = "trial_documents.python_task.params."
    if not path.startswith(prefix) or not isinstance(trial_documents, dict):
        return None
    selection = trial_documents.get("python_task")
    if not isinstance(selection, dict) or not isinstance(selection.get("type"), str):
        return None
    definition = registry.resolve(selection["type"])
    return definition.parameter_sites.get(path.removeprefix(prefix)) if definition else None


def application_python_tasks() -> PythonTaskRegistry:
    from feedbax.plugins.composition import load_application

    return load_application().bundle.python_tasks
