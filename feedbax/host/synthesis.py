"""Run a declared synthesis function on a resolved trial and publish its model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from feedbax.analysis.operation_inputs import verify_dataset
from feedbax.contracts.expression_values import Problem, ProblemCode
from feedbax.contracts.method_checkpoint import MethodCheckpointModel
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project import json_pointer_parts
from feedbax.contracts.request_specs import SynthesisRequestSpec
from feedbax.execution.trial_resolution import resolve_trial_execution
from feedbax.host.model_checkpoint import ModelCheckpointError, ModelParameterLayout, SemanticModelWriter
from feedbax.runtime._graph_key_schedule import model_initialization_key
from feedbax.synthesis import ParameterSlot, SynthesisProblem

if TYPE_CHECKING:
    from collections.abc import Mapping

    from feedbax.application.bundle import ApplicationRegistryBundle
    from feedbax.contracts.artifact_custody import ArtifactBlobProvider
    from feedbax.dataset import Dataset
    from feedbax.plugins.declarations import DistributionSource


@dataclass(frozen=True)
class SynthesisDelivery:
    model: MethodCheckpointModel
    datasets: Mapping[str, Dataset]
    provider: DistributionSource


def _slot_problem(address: str, reason: str) -> Problem:
    try:
        resolved = ModelAddress.from_text(address)
    except ValueError:
        return Problem(code=ProblemCode.STRUCTURE, path=("parameters", address), reason=reason)
    return Problem(
        code=ProblemCode.STRUCTURE, path=("model", *resolved.stored_path()), reason=reason, model_address=resolved,
    )


def synthesize_model(
    spec: SynthesisRequestSpec, *, registry: ApplicationRegistryBundle,
    artifacts: ArtifactBlobProvider, transaction: Path,
) -> SynthesisDelivery | tuple[Problem, ...]:
    """Fit a method result to every parameter before releasing a portable tree."""
    method = registry.synthesis_methods.resolve(spec.method)
    if method.key != spec.method:
        raise ValueError(f"Use the canonical synthesis method key {method.key!r} in a managed request.")
    settings = method.declaration.settings_model.model_validate(spec.settings)
    documents = spec.trial_documents
    trial = resolve_trial_execution(
        documents.model, documents.task_source, registry.components, registry.expression_functions,
        python_tasks=registry.python_tasks, objective=documents.objective, input_bindings=documents.input_bindings,
    )
    if isinstance(trial, tuple):
        return trial
    prepared = trial.realize(initialization_key=model_initialization_key(0, "0"), artifact_provider=artifacts)
    if isinstance(prepared, tuple):
        return prepared
    assert prepared.objective is not None
    template = prepared.executable.graph
    try:
        layout = ModelParameterLayout.from_model(prepared.lowered, template)
        target_path = spec.target.stored_path()
        if uncovered := tuple(
            _slot_problem(address, "No model source provides this parameter outside the Synthesize target.")
            for address in layout.addresses
            if ModelAddress.from_text(address).stored_path()[:len(target_path)] != target_path
        ):
            return uncovered
        slots = MappingProxyType({
            address: ParameterSlot(tuple(layout.arrays[slot].shape), layout.arrays[slot].dtype.name)
            for address, slot in layout.addresses.items()
        })
        result = method.method(SynthesisProblem(
            graph=template, task=prepared.execution.task, objective=prepared.objective, slots=slots, target=spec.target,
        ), settings)
        if isinstance(result, tuple):
            return result
        if misfit := layout.fitting(result.parameters):
            return tuple(_slot_problem(problem.address, problem.detail) for problem in misfit)
        outputs = method.declaration.outputs
        if result.datasets.keys() != outputs.keys():
            raise ValueError(f"Synthesis method must return exactly its declared Dataset roles {sorted(outputs)!r}.")
        for name, dataset in result.datasets.items():
            verify_dataset(dataset, outputs[name], settings)
        model = layout.apply(result.parameters)
        written = SemanticModelWriter(layout.graph, artifacts).publish(
            transaction, (model,), templates=(template,), replica_ids=("0",),
        )
        return SynthesisDelivery(written, result.datasets, method.source)
    except ModelCheckpointError as error:
        return (Problem(code=ProblemCode.STRUCTURE, path=json_pointer_parts(error.path), reason=str(error)),)
