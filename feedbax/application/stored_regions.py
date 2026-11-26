"""Adapt installed application registries to authored parameter regions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel

from feedbax.application.composition import load_application
from feedbax.contracts.analysis_document import OperationElement
from feedbax.contracts.stored_model import StoredLeafComponent, model_leaf_expression_values
from feedbax.contracts.stored_regions import StoredRegion, StoredTrainInline, declared_stored_region
from feedbax.contracts.synthesis import SynthesisStepPayload
from feedbax.contracts.task_document import PythonTaskSelection
from feedbax.contracts.training_document import TrainingDocument


if TYPE_CHECKING:
    from feedbax.analysis.operations import OperationRegistry
    from feedbax.component_registry import ComponentRegistry
    from feedbax.synthesis.methods import SynthesisMethodRegistry
    from feedbax.tasks.registry import PythonTaskRegistry
    from feedbax.training.methods import TrainingMethodRegistry


@dataclass(frozen=True)
class StoredRegionRegistries:
    """The installed schemas supplied by the same registries that make the plan."""

    training_methods: TrainingMethodRegistry | None = None
    synthesis_methods: SynthesisMethodRegistry | None = None
    operations: OperationRegistry | None = None
    python_tasks: PythonTaskRegistry | None = None
    components: ComponentRegistry | None = None

    def _registry(self, family: str):
        supplied = getattr(self, family)
        if supplied is not None:
            return supplied
        return getattr(load_application().bundle, family)

    def region(self, owner: BaseModel, field: str) -> StoredRegion | None:
        """Find the declaration owning an open value field, if there is one."""
        try:
            if isinstance(owner, (StoredTrainInline, TrainingDocument)) and field == "settings":
                method = getattr(owner, "method", None)
                return StoredRegion(self._registry("training_methods").resolve(method).settings_model
                                    if isinstance(method, str) else None)
            if isinstance(owner, SynthesisStepPayload) and field == "settings":
                method = getattr(owner, "method", None)
                return StoredRegion(self._registry("synthesis_methods").resolve(method).declaration.settings_model
                                    if isinstance(method, str) else None)
            if isinstance(owner, OperationElement) and field == "params":
                operation = getattr(owner, "operation", None)
                return StoredRegion(self._registry("operations").resolve(operation).value.parameter_model
                                    if isinstance(operation, str) else None)
            if isinstance(owner, PythonTaskSelection) and field == "params":
                task = self._registry("python_tasks").resolve(owner.type)
                return StoredRegion(task.parameters if task is not None else None)
            if isinstance(owner, StoredLeafComponent) and field == "fields":
                components = self._registry("components")
                declaration = components.get(owner.type)
                if declaration is not None and owner.revision in (None, declaration.revision):
                    from feedbax.compiler.model_document import model_tau_seconds, split_model_field_context

                    lowered, authored = model_leaf_expression_values(owner, components)
                    _, context = split_model_field_context(declaration, owner.fields | lowered)
                    if declaration.name == "feedbax.VanillaRNN" and "tau" in context:
                        model_tau_seconds(context["tau"], 1.0)
                    return StoredRegion(
                        declaration.params.model, component=True, contextual=frozenset(context),
                        lowered_expressions=lowered, authored_expressions=authored,
                    )
                return StoredRegion(None)
        except KeyError:
            return StoredRegion(None)
        return declared_stored_region(owner, field)

    def feedbax_models(self) -> Iterable[type[BaseModel]]:
        """Enumerate the declared defaults that belong in Feedbax's manifest."""
        for family in ("training_methods", "synthesis_methods", "operations", "python_tasks", "components"):
            registry = self._registry(family)
            for key in registry.names() if family == "components" else registry.keys():
                record = registry.resolve(key) if family == "operations" else registry.declaration_record(key)
                if record.source.distribution != "feedbax":
                    continue
                value = record.value
                match family:
                    case "training_methods":
                        yield value.settings_model
                    case "synthesis_methods":
                        yield value.declaration.settings_model
                    case "operations":
                        yield value.parameter_model
                    case "python_tasks":
                        yield value.parameters
                    case "components":
                        yield value.params.model
