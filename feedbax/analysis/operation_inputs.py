"""Check an operation's inputs against its declared ports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from feedbax.contracts.model_input import DocumentInput, TaskInputSource
from feedbax.contracts.dataset import FieldDescriptor
from feedbax.contracts.operation import DatasetPort
from feedbax.contracts.task_document import PythonTaskSelection, TaskDocument
from feedbax.dataset import Dataset

if TYPE_CHECKING:
    from pydantic import BaseModel


def _leading(name: str, actual: FieldDescriptor | None, required: FieldDescriptor) -> tuple[str, ...]:
    count = len(actual.dims) - len(required.dims) if actual is not None else -1
    if (
        actual is None
        or count < 0
        or actual.dims[count:] != required.dims
        or (required.units is not None and actual.units != required.units)
        or (required.frame is not None and actual.frame != required.frame)
    ):
        raise ValueError(
            f"Dataset field {name!r} requires {required.dims}, after any stacking dimensions, "
            "with the declared units and frame."
        )
    return actual.dims[:count]


def verify_dataset(
    dataset: Dataset, port: DatasetPort, parameters: BaseModel | None = None,
) -> tuple[str, ...]:
    """Check a dataset against a port and return the stacking dimensions in front of its declared fields.

    Every declared variable has the same stacking dimensions; a declared coordinate has some of them.
    """
    descriptor = dataset.descriptor
    if descriptor.schema_id not in port.schema_ids:
        raise ValueError(f"Operation dataset requires one of {port.schema_ids}, got {descriptor.schema_id!r}")
    port = port.for_dimensions(descriptor.dimensions)
    variables = dict(port.variables)
    if parameters is not None:
        variables.update({getattr(parameters, parameter): field for parameter, field in port.selected_variables.items()})
    leading = {_leading(name, descriptor.variables.get(name), field) for name, field in variables.items()}
    if len(leading) > 1:
        raise ValueError(f"Dataset variables have different stacking dimensions: {sorted(leading)}")
    stacking = leading.pop() if leading else ()
    for name, field in port.coordinates.items():
        dims = _leading(name, descriptor.coords.get(name), field)
        if [dim for dim in stacking if dim in dims] != list(dims):
            raise ValueError(f"Dataset coordinate {name!r} is stacked on {dims}, outside {stacking}.")
    return stacking


def require_task_source(role: str, task: DocumentInput, key: str | None) -> TaskInputSource:
    if isinstance(task, TaskDocument) and key is None:
        return task
    if isinstance(task, PythonTaskSelection) and task.type == key:
        return task
    raise ValueError(f"Operation input {role!r} requires {key or 'a Task document'}")
