"""Authored label parts for the validation trials of a planned row."""

from collections.abc import Callable, Iterable, Mapping

from feedbax.compiler.set_fields import listed_set_validation_choices
from feedbax.contracts.expression_values import LiteralValue
from feedbax.contracts.member_family import expand_members
from feedbax.contracts.project import Document
from feedbax.contracts.project_sweep import DEFAULT_SWEEP_ROW
from feedbax.contracts.set_document import read_set
from feedbax.contracts.task_document import StoredTaskDocument, TaskDocument


type TrialNameParts = Mapping[str, tuple[str, ...]]


def trial_name_parts(
    authored: StoredTaskDocument | None, task: TaskDocument, read: Callable[[str], Document | None],
) -> TrialNameParts:
    """Pair generated ids with their authored labels using the lowered Task's fixed Set choices."""
    base_names = authored.validation_trial_labels() if authored is not None else {}
    selectors = listed_set_validation_choices(task)
    labels = {}
    expanded_sets = {}
    for variable in task.variables:
        if variable.set_choices is None or variable.id not in selectors:
            continue
        document = variable.set_choices.set
        if document not in expanded_sets:
            expanded_sets[document] = {
                member.key: member.value.label for member in expand_members(read_set(read, document).entries)
            }
        labels[variable.id] = tuple(expanded_sets[document][entry] for entry in selectors[variable.id])
    names = {}
    for number, trial in enumerate(task.validation_trials, 1):
        if trial.id is None:
            continue
        base = trial.id
        if labels and "=" in base:
            # Authored trial keys never contain '='. The first selector suffix follows the base key.
            base = base.split("=", 1)[0].rpartition("--")[0]
        base_label = base_names.get(base, "Trial 1" if base == DEFAULT_SWEEP_ROW else f"Trial {number}")
        parts = [base_label]
        for axis, entry_labels in sorted(labels.items()):
            value = trial.variables.get(axis)
            if isinstance(value, LiteralValue) and type(value.value) is int and 1 <= value.value <= len(entry_labels):
                parts.append(entry_labels[value.value - 1])
        names[trial.id] = tuple(parts)
    return names


def numbered_trial_parts(ids: Iterable[str]) -> TrialNameParts:
    """Python Tasks declare trial IDs and order, without a separate label rule."""
    return {id: (f"Trial {number}",) for number, id in enumerate(ids, 1)}
