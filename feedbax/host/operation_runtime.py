"""Run an operation on the exact inputs bound by the managed Store."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from feedbax.contracts.loaded_model import IMPORTED_MODEL_MEDIA_TYPE, ImportedModelSource, decode_imported_model_document

from feedbax.contracts.model_input import (
    CheckpointModelSource,
    INITIAL_MODEL_MEDIA_TYPE,
    InitialModelSource,
    read_document_input,
)
from feedbax.contracts.operation import DatasetPort, ModelPort, TaskPort
from feedbax.contracts.task_document import PythonTaskSelection
from feedbax.dataset import Dataset
from feedbax.analysis.operation_inputs import (
    require_task_source,
    uses_replicate_axis,
    verify_dataset,
)
from feedbax.dataset_replicates import replicate_ids, select_replicate, stack_replicates
from feedbax.tasks.registry import resolve_python_task_instance

if TYPE_CHECKING:
    from pydantic import BaseModel

    from feedbax.analysis.operations import RegisteredOperation
    from feedbax.contracts.effective_success import Bindings
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
    from feedbax.application.bundle import ApplicationRegistryBundle
    from feedbax.plugins.declarations import ResolvedDeclaration


def run_operation(
    operation: ResolvedDeclaration[RegisteredOperation],
    parameters: BaseModel,
    bindings: Bindings,
    datasets: Mapping[str, Dataset],
    *,
    registry: ApplicationRegistryBundle,
    artifacts: ImmutableArtifactBlobProvider,
) -> Dataset:
    definition = operation.value.definition
    if numerical := definition.numerical:
        import jax

        bits = 64 if jax.config.jax_enable_x64 else 32
        backend = jax.default_backend()
        if (backend, bits) != (numerical.backend, numerical.floating_point_bits):
            raise ValueError(
                f"Operation {definition.key!r} requires {numerical.backend} float{numerical.floating_point_bits}, "
                f"worker has {backend} float{bits}"
            )

    arguments: dict = dict(datasets)
    task_source = None
    model_role = None
    model_member = None
    for role, port in definition.inputs.items():
        if isinstance(port, TaskPort):
            (member,) = bindings[role]
            installed = registry.python_tasks.resolve(port.task_key) if port.task_key is not None else None
            task_source = require_task_source(
                role, read_document_input(member.content, artifacts), installed.key if installed else port.task_key,
            )
            if isinstance(task_source, PythonTaskSelection):
                task = resolve_python_task_instance(task_source, registry.python_tasks)
                if isinstance(task, tuple):
                    raise ValueError(task[0].reason)
                if task.definition.parameters is not operation.value.task_parameters:
                    raise ValueError(f"Operation {definition.key!r} declares the wrong settings for {port.task_key!r}")
                arguments[role] = task
            else:
                arguments[role] = task_source
        elif isinstance(port, ModelPort):
            model_role = role
            (model_member,) = bindings[role]

    model = None
    if model_member is not None:
        from feedbax.persistence.model_input import model_product_source
        from feedbax.host.model_input import prepare_model_task, reconstruct_model

        producer_documents = None
        if model_member.content.media_type == INITIAL_MODEL_MEDIA_TYPE:
            source = read_document_input(model_member.content, artifacts)
            if not isinstance(source, InitialModelSource):
                raise ValueError(f"Operation input {model_role!r} requires an initial Model document")
        elif model_member.content.media_type == IMPORTED_MODEL_MEDIA_TYPE:
            imported = read_document_input(model_member.content, artifacts)
            assert isinstance(imported, ImportedModelSource)
            pinned = decode_imported_model_document(artifacts.get_object(imported.model))
            source = CheckpointModelSource(
                tree=imported.weights, encoding=imported.encoding, initial=InitialModelSource(document=pinned),
            )
        else:
            source, producer_documents = model_product_source(model_member, artifacts=artifacts)
        paired_task = task_source or (producer_documents.task_source if producer_documents is not None else None)
        if paired_task is None:
            model = reconstruct_model(source, registry=registry, artifacts=artifacts)
        else:
            input_bindings = producer_documents.bindings_for_same_task(paired_task) if producer_documents is not None else None
            paired = prepare_model_task(
                source, paired_task, registry=registry, artifacts=artifacts, input_bindings=input_bindings,
            )
            if isinstance(paired, tuple):
                raise ValueError(paired[0].reason)
            model = paired.model
            if (role := operation.value.prepared_task_role) is not None:
                arguments[role] = paired.paired.execution
        if isinstance(model, tuple):
            raise ValueError(model[0].reason)

    ids = model.replica_ids if model is not None else None
    mapped = {}
    for role, dataset in datasets.items():
        port = definition.inputs[role]
        assert isinstance(port, DatasetPort)
        if uses_replicate_axis(port) or (coordinates := replicate_ids(dataset)) is None:
            verify_dataset(dataset, port, parameters)
            continue
        if ids is None:
            ids = coordinates
        elif set(ids) != set(coordinates):
            raise ValueError(f"Dataset input {role!r} has different replicate IDs from the operation inputs")
        mapped[role] = {key: position for position, key in enumerate(coordinates)}

    (output,) = definition.outputs.values()

    def execute(inputs) -> Dataset:
        result = operation.defining_object(params=parameters, **inputs)
        if not isinstance(result, Dataset):
            raise TypeError(f"Operation {definition.key!r} returned a value that is not a Dataset")
        verify_dataset(result, output)
        return result

    if ids is None:
        return execute(arguments)
    results = []
    for position, key in enumerate(ids):
        inputs = dict(arguments)
        for role, positions in mapped.items():
            dataset = select_replicate(datasets[role], positions[key])
            port = definition.inputs[role]
            assert isinstance(port, DatasetPort)
            verify_dataset(dataset, port, parameters)
            inputs[role] = dataset
        if model is not None:
            inputs[model_role] = model.graphs[position]
        results.append(execute(inputs))
    return stack_replicates(ids, results)
