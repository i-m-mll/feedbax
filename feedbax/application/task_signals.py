"""Resolve the task signals available for authored Model input bindings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from feedbax.application.composition import application_python_tasks
from feedbax.contracts.expression_values import Level, ScopedValue
from feedbax.contracts.model_document import ModelDocument, SignalInput
from feedbax.contracts.task_document import PythonTaskSelection, TaskDocument
from feedbax.contracts.trial_bindings import selected_signal_id
from feedbax.execution.trial_resolution import resolve_python_task

if TYPE_CHECKING:
    from feedbax.tasks.registry import PythonTaskRegistry


def model_task_signals(
    source: TaskDocument | PythonTaskSelection | None, python_tasks: PythonTaskRegistry | None,
) -> tuple[ScopedValue, ...]:
    if isinstance(source, PythonTaskSelection):
        template = resolve_python_task(source, python_tasks or application_python_tasks())
        if not isinstance(template, tuple):
            return tuple(ScopedValue(id=name, name=name, type=type_, level=Level.STEP)
                         for name, type_ in template.signals.items())
    if isinstance(source, TaskDocument):
        return tuple(ScopedValue(id=item.id, name=item.name, type=item.type, level=Level.STEP) for item in source.signals)
    return ()


def matched_task_inputs(
    model: ModelDocument, signals: tuple[ScopedValue, ...], overrides: Mapping[str, str | None],
) -> tuple[tuple[SignalInput, ScopedValue], ...]:
    values = {item.id: item for item in signals}
    names = {item.name: item.id for item in signals}
    return tuple(
        (port, values[signal]) for port in model.ports.inputs
        if (signal := selected_signal_id(port.id, port.name, names, overrides)) is not None and signal in values
    )
