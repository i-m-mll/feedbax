"""Preserve root-input signal selections when Studio renames either end."""

from collections import defaultdict
from dataclasses import dataclass

from pydantic import ValidationError

from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.model_document import MODEL_DOCUMENT_SCHEMA_ID
from feedbax.contracts.model_edit import ListedRuns, StepRuns, VariantRuns
from feedbax.contracts.task_document import PYTHON_TASK_SCHEMA_ID, TASK_DOCUMENT_SCHEMA_ID, PythonTaskSelection, TaskDocument
from feedbax.execution.trial_resolution import resolve_python_task
from feedbax.contracts.rollout import authored_rollout_settings
from feedbax.contracts.project_inputs import RunReference
from feedbax.contracts.trial_bindings import selected_signal_id
from feedbax.planner.project_plan import compile_project, own_task_variation
from feedbax.training.evaluation_variant import resolve_evaluation_task
from feedbax.web.services.evaluate_sources import evaluate_subject
from feedbax.studio.selection import StoredEdit
from feedbax.web.services.trial_scope import (
    authored_payload, base_descendants, preview_stored, scoped_file, variant_file, variant_view,
)


def input_names(model):
    return {port["id"]: port["name"] for port in model.get("ports", {}).get("inputs", ())}


def _signal_names(task, python_tasks):
    if isinstance(task, TaskDocument):
        return {signal.id: signal.name for signal in task.signals}
    if isinstance(task, PythonTaskSelection):
        template = resolve_python_task(task, python_tasks)
        if not isinstance(template, tuple):
            return {name: name for name in template.signals}
    return {}


def _task(documents):
    if raw := documents.get("task"):
        return TaskDocument.model_validate(raw)
    if raw := documents.get("python_task"):
        return PythonTaskSelection.model_validate(raw)
    return None


def _renamed(before, after):
    return {id_ for id_ in before.keys() & after.keys() if before[id_] != after[id_]}


@dataclass(frozen=True)
class _Pairing:
    location: str
    settings: dict
    inputs: dict[str, str]
    signals: dict[str, str]
    bindings: dict[str, str | None]

    def selected(self) -> dict[str, str | None]:
        by_name = {name: id_ for id_, name in self.signals.items()}
        return {
            id_: selected_signal_id(id_, name, by_name, self.bindings)
            for id_, name in self.inputs.items()
        }


def _checkpoint_documents(plan, key, definition, read_terminal):
    if (upstream := evaluate_subject(definition)) and (train := plan.selected_definitions.get(upstream[0])):
        return train.payload.get("trial_documents")
    reference = plan.steps[key.variant, key.step].inputs.get("subject")
    if isinstance(reference, RunReference) and read_terminal is not None:
        try:
            terminal = read_terminal(reference.run)
        except (OSError, ValueError):
            return None
        if terminal is not None:
            return terminal.train.trial_documents.model_dump(mode="json")
    return None


def _pairing(plan, key, python_tasks, read_terminal):
    definition = plan.selected_definitions.get(key)
    if definition is None:
        return None
    try:
        return _read_pairing(plan, key, definition, python_tasks, read_terminal)
    except ValidationError:
        return None


def _read_pairing(plan, key, definition, python_tasks, read_terminal):
    payload = definition.payload
    documents = payload.get("trial_documents") or {}
    if plan.steps[key.variant, key.step].kind == "train":
        model, task = documents.get("model"), _task(documents)
        return _Pairing(
            "trial_documents", documents, input_names(model), _signal_names(task, python_tasks),
            documents.get("input_bindings", {}),
        ) if isinstance(model, dict) else None

    subject = definition.input_documents.get("subject")
    task_document = definition.input_documents.get("task")
    settings = payload.get("params") or {}
    bindings = settings.get("input_bindings", {})
    if subject is not None and subject.schema_id == MODEL_DOCUMENT_SCHEMA_ID:
        model = subject.payload
        kinds = {TASK_DOCUMENT_SCHEMA_ID: TaskDocument, PYTHON_TASK_SCHEMA_ID: PythonTaskSelection}
        task = kinds[task_document.schema_id].model_validate(task_document.payload) if (
            task_document and task_document.schema_id in kinds
        ) else None
    elif trained := _checkpoint_documents(plan, key, definition, read_terminal):
        model, task = trained.get("model"), _task(trained)
        bindings = trained.get("input_bindings", {}) | bindings
    else:
        return None
    if task is not None:
        try:
            variation = settings.get("variation", [])
            if own := documents.get("task"):
                variation = own_task_variation(task, own, variation) or variation
            params = authored_rollout_settings(settings | {"variation": variation})
            task = resolve_evaluation_task(task, params.variation)
        except ValueError:
            return None
    return _Pairing(
        "params", settings, input_names(model), _signal_names(task, python_tasks), bindings,
    ) if isinstance(model, dict) else None


def _preserved(before, after):
    renamed_inputs = _renamed(before.inputs, after.inputs)
    renamed_signals = _renamed(before.signals, after.signals)
    old_selected = before.selected()
    wanted = after.selected()
    for id_ in before.inputs.keys() & after.inputs.keys():
        old_signal, new_signal = old_selected[id_], wanted[id_]
        if old_signal != new_signal and (old_signal is None or old_signal in after.signals) and (
            id_ in renamed_inputs or old_signal in renamed_signals or new_signal in renamed_signals
            or (id_ in before.bindings, before.bindings.get(id_)) != (id_ in after.bindings, after.bindings.get(id_))
        ):
            wanted[id_] = old_signal
    return wanted


def preserve_renamed_bindings(folders, project_id, revision, project, stored):
    """Append binding changes to the document's transaction, using every affected run's old pairing."""
    settled = folders.read_plan(project_id, revision=revision)
    kinds = {
        kind: {row.key for row in settled.plan.rows if row.kind == kind}
        for kind in ("train", "evaluate")
    }
    keys = set.union(*kinds.values())
    if not keys:
        return stored
    settled = folders.read_plan(project_id, revision=revision, source_runs=keys)
    before = settled.plan
    from feedbax.application.composition import application_python_tasks

    python_tasks = folders.python_tasks or application_python_tasks()
    read_terminal = settled.readers.terminal_train

    def compile_after():
        return compile_project(
            preview_stored(project, stored), readers=settled.readers, source_runs=keys,
            operations=folders.operations, components=folders.components, functions=folders.functions,
            python_tasks=folders.python_tasks, training_methods=folders.training_methods,
            figure_templates=folders.figure_templates,
        )

    candidate = compile_after()
    primary = before.variants[0]
    for run_keys in kinds.values():
        desired = {}
        for key in sorted(run_keys, key=lambda key: (key.variant, key.step, key.row)):
            old_pair = _pairing(before, key, python_tasks, read_terminal)
            new_pair = _pairing(candidate, key, python_tasks, read_terminal)
            if old_pair is not None and new_pair is not None:
                desired[key] = _preserved(old_pair, new_pair)
        for default_group in (True, False):
            extras = _store_repairs(project, stored, candidate, desired, primary, default_group, python_tasks, read_terminal)
            if extras != stored:
                stored = extras
                candidate = compile_after()
    return stored


def _store_repairs(project, stored, plan, desired, primary, default_group, python_tasks, read_terminal):
    groups = defaultdict(dict)
    for key, wanted in desired.items():
        if (key.variant == primary) != default_group:
            continue
        if (pairing := _pairing(plan, key, python_tasks, read_terminal)) is None:
            continue
        actual = pairing.selected()
        keep = {id_: wanted[id_] for id_ in wanted.keys() & actual.keys() if wanted[id_] != actual[id_]}
        groups[key.variant, key.step][key.row] = pairing.location, pairing.settings, keep

    authored = preview_stored(project, stored)
    ordered = sorted(groups, key=lambda group: (group[0], -len(base_descendants(authored, {group[1]}))))
    for variant, step in ordered:
        rows = groups[variant, step]
        changed = {row: (path, settings, keep) for row, (path, settings, keep) in rows.items() if keep}
        if not changed:
            continue
        variant_id = None if variant == primary else variant
        view = variant_view(plan, authored, variant_id)
        path, _, first = next(iter(changed.values()))
        if len(changed) == len(rows) and all(keep == first for _, _, keep in changed.values()):
            scope = StepRuns(kind="step") if variant_id is None else VariantRuns(kind="variant", variant=variant)
            settings = authored_payload(view, step)[path]
            extra = scoped_file(authored, step, scope, _writes(path, settings, first), view=view)
            stored += extra
            authored = preview_stored(authored, extra)
            continue
        for row, (path, settings, keep) in changed.items():
            extra = scoped_file(view, step, ListedRuns(kind="rows", rows=(row,)), _writes(path, settings, keep))
            next_view = preview_stored(view, extra)
            if variant_id is not None:
                extra = _variant_row_repair(authored, view, next_view, variant, step)
            view = next_view
            stored += extra
            authored = preview_stored(authored, extra)
    return stored


def _variant_row_repair(project, before, after, variant, step):
    raw = before.steps[step].model_dump(mode="json", exclude_unset=True)
    edited = after.steps[step].model_dump(mode="json", exclude_unset=True)
    path = f"steps.{step}.sweep"
    if "sweep" not in raw:
        writes = [OverridePatch(path=path, op="add", value=edited["sweep"])]
    elif "exceptions" not in raw["sweep"]:
        writes = [OverridePatch(path=f"{path}.exceptions", op="add", value=edited["sweep"]["exceptions"])]
    else:
        initial = raw["sweep"]["exceptions"]
        writes = [
            OverridePatch(path=f"{path}.exceptions.{name}", op="replace" if name in initial else "add", value=value)
            for name, value in edited["sweep"]["exceptions"].items() if initial.get(name) != value
        ]
    return StoredEdit([variant_file(project, variant, writes)], reached=frozenset({step}))


def _writes(path, settings, bindings):
    if "input_bindings" not in settings:
        return [OverridePatch(path=f"{path}.input_bindings", op="add", value=bindings)]
    return [
        OverridePatch(
            path=f"{path}.input_bindings.{id_}",
            op="replace" if id_ in settings["input_bindings"] else "add", value=signal,
        )
        for id_, signal in sorted(bindings.items())
    ]
