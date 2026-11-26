"""Project and prepare edits for one saved trial pairing."""


from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import cache, partial

from pydantic import TypeAdapter
from pydantic_core import to_json

from feedbax.application.stored_regions import compose_document
from feedbax.contracts.base import sha256_bytes
from feedbax.execution.trial_resolution import TrialPair, compile_unpaired_task, pair_trial_source
from feedbax.contracts.expression_storage import (
    StoredExpression, StoredTime, lower_expression, stored_expression_value, stored_reference, stored_set_reference,
)
from feedbax.contracts.expression_values import (
    TIME, Level, LiteralValue, Problem, ProblemCode, ScopedValue, SequenceValue, Site, SweepList, SweepSample, TextSpan,
    UnresolvedValue, SetVariableOrigin, ValueBase, ValueType,
)
from feedbax.tasks.document import TrialSource, zero
from feedbax.tasks.registry import resolve_python_task_instance
from feedbax.tasks.structure import restructure
from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.member_family import FamilyProblem, expand_family
from feedbax.contracts.studio_scope import StoreKind
from feedbax.compiler.model_reset import ModelResetField
from feedbax.compiler.model_signals import model_measure_scope
from feedbax.contracts.stored_members import scoped_set_member_id
from feedbax.compiler.set_fields import resolve_trial_set_condition, trial_set_condition_site
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_edit import AxisValueRuns, ListedRuns, VariantRuns
from feedbax.contracts.model_attachment import MODEL_ATTACHMENTS, MODEL_DIFFERENCE_BATCHES
from feedbax.compiler.model_attachment import lower_model_difference_values, resolve_effective_model
from feedbax.contracts.project import (
    LoadedModelDocument, Payload, Project, RemoveProjectValue, RootDocument, SetProjectValue,
    apply_differences, apply_document_edits, diff,
    OrderDocumentMembers, OrderProjectValue, base_chain, json_pointer_parts, member_name,
)
from feedbax.contracts.loaded_model import LOADED_MODEL_SCHEMA_ID
from feedbax.contracts.project_resolution import ProjectDiagnostic, ResolutionCode, compose_step, resolve_step
from feedbax.contracts.project_sweep import SetAxis
from feedbax.contracts.set_document import SetEntry, read_set
from feedbax.planner.project_plan import EVALUATION_TASK, PlannedRow, PlannedRun, RunKey, compile_project, resolve_authored_evaluation_task
from feedbax.project_members import generated_name
from feedbax.project_model_selection import stored_trial_task_slot, trial_document_slots
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_specs import DocumentTrainingConfig, DocumentTrainingRequestSpec, EvaluationRequestSpec, SynthesisRequestSpec
from feedbax.contracts.request_refs import BestRank
from feedbax.contracts.rollout import ROLLOUT_TYPE, RolloutParameters, authored_rollout_settings, authored_variation
from feedbax.contracts.project_sources import ValueSource, pointer
from feedbax.contracts.project_storage import stored_keyed_collection, stored_project_value
from feedbax.contracts.stored_members import project_member
from feedbax.contracts.studio_project import AuthoredFileAddress, StudioProjectFilePatch
from feedbax.contracts.studio_sets import SetEntryTask
from feedbax.contracts.studio_place import authored_task_link_path, task_start_path
from feedbax.studio.authoring import training_document_choice
from feedbax.studio.field_sites import FieldSiteIndex
from feedbax.training.evaluation_variant import resolve_evaluation_variant
from feedbax.contracts.studio_trials import (
    StudioObjectiveTermValues, StudioTaskChoice, StudioTaskType, StudioTrialAuthoring, StudioTrialAuthoringRequest, StudioTrialBindingEdit,
    StudioTrialDocumentEdit, StudioTrialDocumentReset, StudioTrialEpoch, StudioTrialEpochInstance,
    StudioTrialExpressionEdit, StudioTrialField,
    StudioTrialInput, StudioTrialObjectiveSource, StudioTrialObjectiveTermAdd,
    StudioTrialOutput, StudioTrialRelate, StudioTrialReset, StudioTrialSelectDocument, StudioTrialStepScope,
    StudioTrialStructureEdit, StudioTrialUnresolvedLink,
)
from feedbax.contracts.task_document import (
    PYTHON_TASK_SCHEMA_ID, STORED_TASK_DOCUMENT_SCHEMA_ID, PythonTaskSelection,
    StoredLength, StoredTaskDocument, StoredTaskDraftError, StoredUntil, StoredVariable, TaskDocument, Variable,
)
from feedbax.contracts.task_structure import RenameMember as RenameTaskMember
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.stored_model import STORED_MODEL_DOCUMENT_SCHEMA_ID, StoredModelDocument
from feedbax.contracts.time_units import TimeValue, length_steps
from feedbax.contracts.objective_document import (
    STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID, ObjectiveDocument, StoredObjectiveDocument, StoredObjectiveTerm,
)
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.training import TrialDocuments, lower_stored_trial_documents
from feedbax.contracts.trial_bindings import selected_signal_id
from feedbax.contracts.window import StoredWindowPosition, window_offset_sites
from feedbax.contracts.expression_check import check, references
from feedbax.contracts.expression_site import FieldKind, project_site, read_field
from feedbax.contracts.expression_text import parse, print_field
from feedbax.studio.value_structure import value_structure
from feedbax.objectives.document import ObjectiveExecutor, ObjectiveUse, compile_objective, epoch_outcome_references
from feedbax.application.bundle import ApplicationRegistryBundle
from feedbax.web.services.authored_scope import edited_authored_rows
from feedbax.web.services.evaluate_sources import evaluate_subject, evaluate_value_source
from feedbax.web.services.project_folder import ProjectFolderInvalid, ProjectFolderService
from feedbax.web.services.set_authoring import prepare_set_entry_task, trial_set_fields
from feedbax.web.services.trial_expression_menus import TrialExpressionField
from feedbax.web.services.store_choice import EditStore, all_readers_store, choose_store
from feedbax.studio.selection import StoredEdit
from feedbax.studio.numeric_fields import project_numeric_field, time_conversion
from feedbax.web.services.trial_fields import (
    FamilyEditor, FamilyField, TrialFieldSweep, at, family_indices, family_problem, index_pointer,
    family_rule_edits, is_family_member_path, is_family_rule, is_family_rule_edit, is_window_offset, numeric_nodes,
    objective_editor, routes_to_sweep,
    site_expression, trial_field_sweep, trial_field_sweep_edit, trial_time_use, validation_editor, with_family_indices,
)
from feedbax.web.services.trial_scope import (
    authored_payload, base_value_edits, changed_from_base, content_edits, document_write, evaluation_document_write, file_value, lit_steps, preview_stored, scoped_file,
    selected_runs, source_value_file, variant_view,
)


@dataclass(frozen=True)
class _InitialTrialDocuments:
    model: ModelDocument
    task_source: TaskDocument | PythonTaskSelection | StoredTaskDocument
    objective: ObjectiveDocument | None
    input_bindings: dict[str, str | None]

    @property
    def task(self) -> TaskDocument | None:
        return self.task_source if isinstance(self.task_source, TaskDocument) else None


@dataclass(frozen=True)
class _DraftTrialDocuments:
    model: ModelDocument
    task_source: StoredTaskDocument
    objective: ObjectiveDocument | None
    input_bindings: dict[str, str | None]
    task = None


@dataclass(frozen=True)
class _SetVariableView:
    origin: SetVariableOrigin
    source: SetEntry
    generated: str
    scoped: bool


@dataclass(frozen=True)
class _Projection:
    pair: TrialPair | None
    variables: tuple[ScopedValue, ...]
    signals: tuple[ScopedValue, ...]
    epochs: tuple[StudioTrialEpoch, ...]
    fields: dict[tuple[str | int, ...], Site]
    problems: tuple[Problem, ...]
    task_origins: dict[tuple[str, ...], str] = field(default_factory=dict)
    resets: tuple[ModelResetField, ...] = ()
    partial_task: TrialSource | None = None
    epoch_instances: tuple[StudioTrialEpochInstance, ...] = ()
    output_paths: dict[str, tuple[str, ...]] = field(default_factory=dict)
    set_condition_variables: tuple[Variable, ...] | None = None
    stored_task: StoredTaskDocument | PythonTaskSelection | None = None
    family_editor: FamilyEditor = field(default_factory=FamilyEditor)
    set_variables: dict[str, _SetVariableView] = field(default_factory=dict)
    generated_set_members: set[str] = field(default_factory=set)


def _project(
    documents, bundle, *, show_objective=True, graph_override=None,
    objective_use: ObjectiveUse = ObjectiveUse.TRAIN,
) -> _Projection:
    from feedbax.compiler.model_document import complete_model_fields

    task = documents.task_source
    if isinstance(task, StoredTaskDocument):
        return _project_task_drafts(documents, task, show_objective)
    fields = {}
    if isinstance(task, PythonTaskSelection):
        resolved = resolve_python_task_instance(task, bundle.python_tasks)
        if isinstance(resolved, tuple):
            return _Projection(None, (), (), (), {}, resolved)
        definition = resolved.definition
        template = resolved.template()
        signals = tuple(ScopedValue(id=name, name=name, type=kind, level=Level.STEP)
                        for name, kind in template.signals.items())
        epochs = ()
        variables = ()
        fields.update({("python_task", "params", name): site
                       for name, site in definition.parameter_sites.items()})
    else:
        signals = tuple(ScopedValue(id=signal.id, name=signal.name, type=signal.type, level=Level.STEP)
                        for signal in task.signals)
        variables = ()
        epochs = tuple(StudioTrialEpoch(id=epoch.id, name=epoch.name, steps=None) for epoch in task.epochs)
    resets = []

    task_sites = {}
    paired = pair_trial_source(
        complete_model_fields(documents.model, bundle.components), task, bundle.components, bundle.expression_functions,
        python_tasks=bundle.python_tasks, input_bindings=documents.input_bindings,
        task_sites=task_sites, graph_override=graph_override, reset_fields=resets, keyed_root_ports=True,
    )
    partial = None
    if isinstance(paired, tuple):
        problems = paired
        paired = None
        if all(problem.path[:1] == ("model",) for problem in problems):
            preview = compile_unpaired_task(
                documents.model, task, bundle.expression_functions, python_tasks=bundle.python_tasks, sites=task_sites,
            )
            if isinstance(preview, tuple):
                problems += preview
            else:
                partial = preview
    else:
        problems = paired.problems
    fields.update({("task", *path): site for path, site in task_sites.items()})
    if paired is None and partial is None:
        if show_objective and documents.objective is not None:
            fields.update(_unpaired_objective_sites(documents, signals))
        return _Projection(None, variables, signals, epochs, fields, problems, resets=tuple(resets))
    source = paired.task if paired is not None else partial.task
    variables, signals = source.interface.variables, source.interface.signals
    names = {epoch.id: epoch.name for epoch in epochs}
    epoch_instances = tuple(
        StudioTrialEpochInstance(id=epoch, name=names.get(epoch, epoch), steps=steps, occurrence=occurrence)
        for occurrence, (epoch, steps) in enumerate(zip(source.epochs, source.step_counts, strict=True))
    )
    epochs = tuple(
        StudioTrialEpoch(id=instance.id, name=instance.name, steps=instance.steps)
        for instance in {item.id: item for item in epoch_instances}.values()
    )
    output_paths = {}
    if documents.objective is not None:
        if paired is not None:
            measures = model_measure_scope(documents.model, paired.model.signals)
            outputs, output_paths = measures.names, measures.paths
        else:
            outputs = partial.outputs
        objective_sites = {} if show_objective else None
        objective = compile_objective(
            documents.objective, (*outputs, *variables, *signals), bundle.expression_functions,
            task=source, sites=objective_sites, use=objective_use,
        )
        if objective_sites:
            fields.update({("objective", *path): site for path, site in objective_sites.items()})
        if not isinstance(objective, ObjectiveExecutor):
            problems += tuple(problem.model_copy(update={"path": ("objective", *problem.path)})
                              for problem in objective)
    return _Projection(
        paired, variables, signals, epochs, fields, problems, resets=tuple(resets),
        partial_task=source if paired is None else None, epoch_instances=epoch_instances, output_paths=output_paths,
    )


def _project_task_drafts(documents, task: StoredTaskDocument, show_objective: bool) -> _Projection:
    """Keep authored Task fields visible while their draft has no executable form."""
    variables = tuple(ScopedValue(id=key, name=value.label, type=value.type, level=Level.TRIAL)
                      for key, value in task.variables.items())
    signals = tuple(ScopedValue(id=key, name=value.label, type=value.type, level=Level.STEP)
                    for key, value in task.signals.items())
    epochs = tuple(StudioTrialEpoch(id=key, name=epoch.label, steps=None) for key, epoch in task.epochs.items())
    time = ValueType(base=ValueBase.REAL, dimensions=TIME)
    length = Site(expected=time, finest=Level.TRIAL, names=variables)
    names = (*_declared_output_names(documents.model), *variables, *signals, *(
        ScopedValue(id=key, name=key, type=time, level=Level.STEP) for key in ("time", "t")
    ))
    condition = Site(expected=ValueType(base=ValueBase.BOOLEAN), finest=Level.STEP, names=names)
    fields = _unpaired_objective_sites(documents, signals, variables=variables) if show_objective and documents.objective else {}
    problems = []
    for key, epoch in task.epochs.items():
        if isinstance(epoch.end, StoredUntil) and isinstance(epoch.end.limit, UnresolvedValue):
            path = ("task", "epochs", key, "end")
            fields.update({(*path, "limit"): length, (*path, "length"): length, (*path, "condition"): condition})
            problems.append(Problem(code=ProblemCode.MISSING_LIMIT, path=(*path, "limit"),
                                    reason="An Until limit needs a literal time in seconds or steps."))
    try:
        task.lower(first=True)
    except StoredTaskDraftError as error:
        if error.path[:1] == ("validation_trials",) and error.path[2:3] not in (("over",), ("combination",), ("label",)):
            problems.append(Problem(code=ProblemCode.STRUCTURE, path=error.path, reason=str(error)))
    return _Projection(None, variables, signals, epochs, fields, tuple(problems), stored_task=task)


def _declared_output_names(model: ModelDocument) -> tuple[ScopedValue, ...]:
    return tuple(
        ScopedValue(
            id=_output_address(port.id).to_text(), name=port.name, level=Level.STEP,
            type=ValueType(base=ValueBase.REAL, shape=tuple(
                size for size in (port.type.shape if port.type and port.type.shape else ())
                if type(size) is int and size > 0
            )),
        )
        for port in model.ports.outputs
    )


def _unpaired_objective_sites(documents, signals, *, variables=None) -> dict[tuple[str | int, ...], Site]:
    """Each term's Measure and Weight over the names the documents declare, for a Model and Task that do not pair.

    A commit through one of these sites is parsed, not checked: the declared names have no resolved types."""
    if variables is None:
        variables = tuple(
            ScopedValue(id=variable.id, name=variable.name, type=variable.type, level=Level.STEP)
            for variable in (documents.task.variables if documents.task is not None else ())
        )
    names = (
        *_declared_output_names(documents.model),
        *((ScopedValue(id="time", name="time", type=ValueType(base=ValueBase.REAL), level=Level.STEP),)
          if isinstance(documents.task_source, TaskDocument | StoredTaskDocument) else ()),
        *variables,
        *signals,
    )
    weight_names = (*names, *(
        ScopedValue(id=name, name=name, type=ValueType(base=ValueBase.REAL), level=Level.STEP)
        for name in ("t", "T", "u")
    ))
    sites: dict[tuple[str | int, ...], Site] = {}
    for term in documents.objective.terms:
        path = ("objective", "terms", term.id)
        sites[(*path, "measure")] = Site(expected=term.measure_type, names=names, finest=Level.STEP)
        sites[(*path, "weight")] = Site(expected=ValueType(base=ValueBase.REAL), names=weight_names, finest=Level.STEP)
        sites.update({(*path, "window", *at): site for at, site in window_offset_sites(term.window, names).items()})
    return sites


def _output_address(port: str) -> ModelAddress:
    return ModelAddress.model_validate({
        "composites": [],
        "element": {"kind": "port", "end": {"owner": "outputs", "port": port}},
    })


def _input_address(port: str) -> ModelAddress:
    return ModelAddress.model_validate({
        "composites": [],
        "element": {"kind": "port", "end": {"owner": "inputs", "port": port}},
    })


def _input_rows(documents, projection) -> list[StudioTrialInput]:
    paired = projection.pair
    resolved = {port.id: port.type for port in paired.model.inputs} if paired else {}
    signals = {signal.id for signal in projection.signals}
    return [
        StudioTrialInput(
            id=port.id, address=address, name=port.name, type=resolved.get(address.to_text()),
            required=port.default is None,
            signal_id=selected_signal_id(port.id, signals, documents.input_bindings),
            explicit=port.id in documents.input_bindings,
        )
        for port in documents.model.ports.inputs
        for address in (_input_address(port.id),)
    ]


_TASK_DOCUMENT_SCHEMAS = frozenset({STORED_TASK_DOCUMENT_SCHEMA_ID, PYTHON_TASK_SCHEMA_ID})


def _authoring_task(task: StoredTaskDocument, *, first: bool = True) -> TaskDocument | StoredTaskDocument:
    try:
        return task.lower(first=first)
    except StoredTaskDraftError:
        return task


def _task_payload(entry: RootDocument) -> TaskDocument | PythonTaskSelection | StoredTaskDocument:
    if entry.schema_id == STORED_TASK_DOCUMENT_SCHEMA_ID:
        return _authoring_task(StoredTaskDocument.model_validate(entry.payload))
    assert entry.schema_id == PYTHON_TASK_SCHEMA_ID
    return PythonTaskSelection.model_validate(entry.payload)


def _with_task(documents, task, *, input_bindings: dict[str, str | None] | None = None):
    bindings = documents.input_bindings if input_bindings is None else input_bindings
    if isinstance(documents, _InitialTrialDocuments):
        return replace(documents, task_source=task, input_bindings=bindings)
    if isinstance(task, StoredTaskDocument):
        return _DraftTrialDocuments(documents.model, task, documents.objective, bindings)
    if isinstance(documents, _DraftTrialDocuments):
        return TrialDocuments(
            model=documents.model, task=task if isinstance(task, TaskDocument) else None,
            python_task=task if isinstance(task, PythonTaskSelection) else None,
            objective=documents.objective, input_bindings=bindings,
        )
    if isinstance(task, PythonTaskSelection):
        return documents.model_copy(update={"task": None, "python_task": task, "input_bindings": bindings})
    return documents.model_copy(update={"task": task, "python_task": None, "input_bindings": bindings})


def _authored_trial_problem(problem: Problem) -> Problem:
    path = authored_task_link_path(problem.path)
    if path[:2] == ("task", "validation_trials"):
        path = path[1:]
    return problem if path is problem.path else problem.model_copy(update={"path": path})


def _task_choices(
    folders: ProjectFolderService, project: Project, documents: TrialDocuments | _InitialTrialDocuments,
    bundle: ApplicationRegistryBundle, graph: GraphSpec | None, objective_use: ObjectiveUse,
) -> tuple[list[StudioTaskChoice], list[Problem]]:
    """Project the available Tasks and share equal problems across their pairings."""
    choices = []
    problems: list[Problem] = []
    indices: dict[str, int] = {}
    pairing_context = sha256_bytes(to_json((
        documents.model, documents.objective, graph, sorted(documents.input_bindings.items()), objective_use.value,
    )))

    def pair(entry: RootDocument) -> tuple[Problem, ...]:
        return _project(
            _with_task(documents, _task_payload(entry)), bundle,
            show_objective=False, graph_override=graph, objective_use=objective_use,
        ).problems

    for name, document in sorted(project.documents.items()):
        if document.schema_id not in _TASK_DOCUMENT_SCHEMAS:
            continue
        entry = compose_document(project.documents.get, name)
        if isinstance(entry, ProjectDiagnostic):
            continue
        key = (
            bundle.components, bundle.expression_functions, bundle.python_tasks, pairing_context,
            entry.schema_id, sha256_bytes(to_json(entry.payload)),
        )
        found = folders.task_choice_problems(key, partial(pair, entry))
        references = []
        for lowered in found:
            problem = _authored_trial_problem(lowered)
            index = indices.setdefault(problem.model_dump_json(), len(problems))
            if index == len(problems):
                problems.append(problem)
            references.append(index)
        choices.append(StudioTaskChoice(document=name, problem_indices=list(dict.fromkeys(references))))
    return choices, problems


def _field_document(documents, role):
    """The document of ``role`` whose fields the view edits as expressions, if the step reads one."""
    if role == "objective":
        return documents.objective
    return documents.task_source if role == "task" and isinstance(documents.task_source, TaskDocument) else None


def _one_expression_problem(path):
    return (Problem(code=ProblemCode.STRUCTURE, reason="This field takes one expression.", path=path),)


def _one_expression(text, site, functions, path):
    parsed = parse(text, site, functions, keep_unknown_names=True)
    value = UnresolvedValue(text=text) if isinstance(parsed, tuple) else parsed.value
    return _one_expression_problem(path) if isinstance(value, SweepList | SequenceValue | SweepSample) else value


_STORED_EXPRESSION = TypeAdapter(StoredExpression)
_STORED_TIME = TypeAdapter(StoredTime)


def _set_spellings(names):
    return {
        name.id: stored_set_reference(origin)
        for name in names
        if (origin := name.set_entry) is not None
        and name.id == scoped_set_member_id(origin.axis, origin.entry, origin.variable)
    }


def _expression_edit(change, documents, stored_task, stored_objective, projection, bundle):
    role, *rest = json_pointer_parts(change.path)
    site = projection.fields.get((role, *rest))
    document = stored_objective if role == "objective" else stored_task if role == "task" else None
    if site is None or document is None or index_pointer(document.model_dump(mode="json"), tuple(rest[:-1])) is None:
        raise ProjectFolderInvalid("The selected field is unavailable.")
    one = _one_expression_problem((role, *rest))
    if change.edit.kind != "text" or change.edit.value_id is not None:
        return one
    text = change.edit.text
    raw = stored_project_value(document, keep_explicit=True)
    if is_family_rule((role, *rest)) and isinstance(at(raw, tuple(rest)), str):
        return StudioTrialDocumentEdit(kind=role, edits=[
            SetProjectValue(op="replace", path=index_pointer(raw, tuple(rest)), value=text),
        ])
    end = _end_of(document, role, rest)
    if end is not None and text.strip() == "rest":
        return StudioTrialDocumentEdit(kind=role, edits=_end_edits(raw, end, "rest", {}))
    window_offset = is_window_offset(role, tuple(rest))
    if window_offset and not text.strip() and isinstance(at(document, tuple(rest[:-1])), StoredWindowPosition):
        stored = index_pointer(raw, tuple(rest))
        return StudioTrialDocumentEdit(kind=role, edits=[RemoveProjectValue(op="remove", path=stored)]) if stored else ()
    condition, separator, limit = text.rpartition("≤") if end is not None else ("", "", "")
    if separator:
        condition_path = (role, *rest[:-1], "condition")
        condition_site = projection.fields[condition_path]
        value = _one_expression(condition.strip(), condition_site, bundle.expression_functions, condition_path)
        if isinstance(value, tuple):
            return value
        limit = _until_limit(limit.strip(), documents, (role, *rest[:-1], "limit"))
        if isinstance(limit, tuple):
            return limit
        return StudioTrialDocumentEdit(kind=role, edits=_end_edits(
            raw, end, "until", {
                "condition": stored_expression_value(value, scopes=_set_spellings(condition_site.names)),
                "limit": limit,
            },
        ))
    if _epoch_end_field((role, *rest), "limit"):
        limit = _until_limit(text, documents, (role, *rest))
        if isinstance(limit, tuple):
            return limit
        return StudioTrialDocumentEdit(kind=role, edits=[SetProjectValue(
            op="replace", path=index_pointer(raw, tuple(rest)), value=limit,
        )])
    value = _one_expression(text, site, bundle.expression_functions, (role, *rest))
    if isinstance(value, tuple):
        return value
    stored_value = stored_expression_value(value, scopes=_set_spellings(site.names))
    if end is not None and not isinstance(at(document, tuple(rest[:-1])), StoredLength):
        return StudioTrialDocumentEdit(kind=role, edits=_end_edits(
            raw, end, "length", {"length": stored_value},
        ))
    stored = index_pointer(raw, tuple(rest))
    path = stored or index_pointer(raw, tuple(rest[:-1])) + pointer(tuple(rest[-1:]))
    return StudioTrialDocumentEdit(kind=role, edits=[SetProjectValue(
        op="replace" if stored else "add", path=path, value=stored_value,
    )])


def _until_limit(text, documents, path):
    limit = _STORED_TIME.validate_python(text)
    period = documents.model.step_seconds
    if isinstance(limit, TimeValue) and (isinstance(period, float) or limit.unit == "st"):
        try:
            length_steps(limit, period if isinstance(period, float) else 1)
        except ValueError as error:
            return (Problem(code=ProblemCode.STRUCTURE, reason=str(error), path=path),)
    return _STORED_TIME.dump_python(limit, mode="json")


def _epoch_end_field(path: tuple[str | int, ...], field: str) -> bool:
    return path[:2] == ("task", "epochs") and path[3:] == ("end", field)


def _end_of(document, role, rest) -> str | None:
    """The pointer of the End whose length ``rest`` addresses in the task document, which ``rest`` makes Rest and a length
    makes a length again; None for any other field."""
    if role != "task" or len(rest) != 4 or rest[0] != "epochs" or tuple(rest[2:]) != ("end", "length"):
        return None
    return index_pointer(document.model_dump(mode="json"), tuple(rest[:-1]))


def _end_edits(raw, end: str, kind: str, fields: dict) -> list:
    """The edits that make the End at ``end`` of ``kind`` with ``fields``, each under the End so that it stays the
    field's edit: its kind, then each field it gains or loses."""
    held = raw
    for part in json_pointer_parts(end):
        held = held[int(part)] if isinstance(held, list) else held[part]
    edits = [SetProjectValue(op="replace" if "kind" in held else "add", path=f"{end}/kind", value=kind)]
    edits += [SetProjectValue(op="replace" if key in held else "add", path=f"{end}/{key}", value=value)
              for key, value in fields.items()]
    edits += [RemoveProjectValue(op="remove", path=f"{end}/{key}") for key in held if key != "kind" and key not in fields]
    if kind == "until":
        epoch = json_pointer_parts(end)[1]
        edits.extend(
            RemoveProjectValue(op="remove", path=pointer(("validation_trials", key, "epoch_steps", epoch)))
            for key, trial in raw.get("validation_trials", {}).items()
            if epoch in trial.get("epoch_steps", ())
        )
    return edits


def _field_label(path) -> str:
    return str(path[-1]).replace("_", " ").capitalize()


def _swept_document(plan, key, definition, role) -> str | None:
    """The document whose fields at ``role`` the run reads whole: a Train's own, and an Evaluate's own Task or else its
    subject's."""
    if plan.steps[key.variant, key.step].kind == "train":
        return _whole_document(definition, f"payload.trial_documents.{role}")
    if plan.steps[key.variant, key.step].kind == "synthesize":
        return _whole_document(definition, f"payload.{role}")
    if role != "task":
        return None
    if (own := _whole_document(definition, _evaluation_task_path(definition))) is not None:
        return own
    upstream = evaluate_subject(definition)
    producer = next((row for row in plan.rows if upstream and row.key == upstream[0]), None)
    if producer is None:
        return None
    slot = "payload.task" if producer.kind == "synthesize" else "payload.trial_documents.task"
    return producer.document_ids.get(slot)


@dataclass(frozen=True)
class _SweepSource:
    document: str
    payload: Payload
    parsed: StoredTaskDocument | StoredObjectiveDocument


def _sweep_source(project, plan, key, definition, role) -> _SweepSource | None:
    """Resolve and validate the authored document once for the view's numeric fields."""
    validate = {"objective": StoredObjectiveDocument.model_validate, "task": StoredTaskDocument.model_validate}.get(role)
    if validate is None or (document := _swept_document(plan, key, definition, role)) is None:
        return None
    entry = compose_document(project.documents.get, document)
    if isinstance(entry, ProjectDiagnostic) or not isinstance(entry.payload, dict):
        return None
    return _SweepSource(document, entry.payload, validate(entry.payload))


def _field_sweep(project, key, path, site, label, sweep_source) -> TrialFieldSweep | None:
    """The number at ``path`` of a document the step reads whole, when the step can sweep it."""
    role, *rest = path
    if (
        is_family_member_path(tuple(path)) and (role == "task" or rest[2:3] == ["over"])
        or is_window_offset(role, tuple(rest[:7]))
        or (source := sweep_source(role)) is None
    ):
        return None
    return trial_field_sweep(project.steps[key.step], source.payload, source.parsed, role, tuple(map(str, rest)), site, label)


def _task_field(task):
    return "python_task" if isinstance(task, PythonTaskSelection) else "task"


def _edit_value(value, change):
    raw = deepcopy(value)
    schema = STORED_TASK_DOCUMENT_SCHEMA_ID if change.kind == "task" else STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID
    apply_document_edits(raw, change.edits, keyed_collection=partial(stored_keyed_collection, schema))
    return raw


def _changed_value(documents, definition, kind):
    field = _task_field(documents.task_source) if kind == "task" else "objective"
    if isinstance(documents, _InitialTrialDocuments):
        return field, definition.input_documents[kind].payload
    slots, prefix = trial_document_slots(definition.payload)
    field = kind if not prefix else field
    return field, slots[field]


def _override(path, op, value=None):
    return OverridePatch(path=path, op=op, **({"value": value} if op != "remove" else {}))


def _edit_patches(prefix, change):
    patches = []
    for edit in change.edits:
        if isinstance(edit, OrderProjectValue):
            raise ProjectFolderInvalid("Ordering a collection needs a whole-step edit of a named document.")
        parts = json_pointer_parts(edit.path)
        if any(not part or "." in part for part in parts):
            raise ProjectFolderInvalid("This scoped edit needs path-safe field names.")
        path = ".".join((prefix, *parts))
        patches.append(_override(path, edit.op, edit.value if isinstance(edit, SetProjectValue) else None))
    return patches


def _value_writes(prefix, raw, change):
    edits = family_rule_edits(raw, change.kind, change.edits)
    change = change.model_copy(update={"edits": edits})
    if change.kind == "task" and any(json_pointer_parts(edit.path)[:1] == ("links",) for edit in change.edits):
        edited = _edit_value(raw, change)
        edits = content_edits(raw, edited, schema_id=STORED_TASK_DOCUMENT_SCHEMA_ID, dotted=True)
        change = _present_ops(raw, change.model_copy(update={"edits": edits}))
    return _edit_patches(prefix, change)


def _authored_task_entry(raw):
    if not isinstance(raw, (dict, list)):
        return None
    entries = raw.items() if isinstance(raw, dict) else enumerate(raw)
    names = [name for name, edit in entries if isinstance(edit, dict) and edit.get("path") == "task"]
    return names[-1] if names else None


def _restore_variation_refs(definition, index, edit):
    """Retain the authored references of a non-Task edit copied into a new array value."""
    prefix = f"payload.params.variation.{index}."
    references = {
        name.removeprefix(prefix): {"document": document}
        | ({"pointer": path} if (path := definition.document_pointers.get(name)) else {})
        for name, document in definition.document_ids.items() if name.startswith(prefix)
    }
    if not references:
        return edit
    preserved = deepcopy(edit)
    for path, ref in references.items():
        parts = path.split(".")
        if any(".".join(parts[:end]) in references for end in range(1, len(parts))):
            continue
        apply_document_edits(preserved, [SetProjectValue(op="replace", path=pointer(parts), value=ref)])
    return preserved


def _task_selection_writes(definition, value):
    params = definition.payload["params"]
    replacement = {"op": "replace", "path": "task", "value": value}
    if "variation" not in params:
        return [_override("params.variation", "add", {"task": replacement})]
    raw = params["variation"]
    if isinstance(raw, list):
        retained = [
            _restore_variation_refs(definition, index, edit) for index, edit in enumerate(raw)
            if edit["path"].partition(".")[0] != "task"
        ]
        return [_override("params.variation", "replace", [*retained, replacement])]
    entry = _authored_task_entry(raw)
    writes = [
        _override(f"params.variation.{name}", "remove")
        for name, edit in raw.items() if edit["path"].startswith("task.")
    ]
    target = entry or "task"
    if entry is None:
        while target in raw:
            target += "-selection"
    writes.append(_override(f"params.variation.{target}", "replace" if entry == target else "add", replacement))
    return writes


def _producer_trial_writes(change, project, definition, documents, projection):
    raw, prefix = trial_document_slots(definition.payload)
    if isinstance(change, StudioTrialSelectDocument) and change.role != "task":
        if change.role not in _DOCUMENT_SCHEMAS:
            raise ProjectFolderInvalid("Choose the step's Model, Task or Objective document.")
        _chosen_document(project, change.document, change.role)
        return [_override(f"{prefix}{change.role}", "replace" if change.role in raw else "add",
                          {"document": change.document})]
    if isinstance(change, StudioTrialSelectDocument):
        task = _chosen_task(project, change.document)
        if not prefix:
            return [_override("task", "replace" if "task" in raw else "add", {"document": change.document})]
        field = _task_field(task)
        if field == "python_task":
            return [
                _override("trial_documents.task", "replace" if "task" in raw else "add", None),
                _override("trial_documents.python_task", "replace" if "python_task" in raw else "add",
                          {"document": change.document}),
            ]
        writes = [_override("trial_documents.task", "replace" if "task" in raw else "add",
                            {"document": change.document})]
        if "python_task" in raw:
            writes.append(_override("trial_documents.python_task", "remove"))
        return writes
    if isinstance(change, StudioTrialBindingEdit):
        row = next((row for row in _input_rows(documents, projection) if row.id == change.input_id), None)
        if row is None:
            raise ProjectFolderInvalid("The ID is not a declared Model input for this row.")
        if "input_bindings" in raw:
            return [_override(
                f"{prefix}input_bindings.{row.id}", "replace" if row.explicit else "add", change.signal_id,
            )]
        return [_override(f"{prefix}input_bindings", "add", {row.id: change.signal_id})]
    field, value = _changed_value(documents, definition, change.kind)
    return _value_writes(f"{prefix}{field}", value, change)


_DOCUMENT_SCHEMAS = {"model": STORED_MODEL_DOCUMENT_SCHEMA_ID, "objective": STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID}


def _chosen_document(project, document, role):
    entry = project.documents.get(document)
    if entry is None or entry.schema_id != _DOCUMENT_SCHEMAS[role]:
        raise ProjectFolderInvalid(
            f"{member_name(project.documents, document)} is not a{'n' if role == 'objective' else ''} {role.title()} document."
        )


def _chosen_task(project, document):
    entry = compose_document(project.documents.get, document)
    if isinstance(entry, ProjectDiagnostic) or entry.schema_id not in _TASK_DOCUMENT_SCHEMAS:
        raise ProjectFolderInvalid(f"{member_name(project.documents, document)} is not a Task document.")
    return _task_payload(entry)


def _evaluation_task_path(definition):
    if "subject" in definition.input_documents:
        return "inputs.task"
    entry = _authored_task_entry(definition.payload.get("params", {}).get("variation"))
    return f"payload.params.variation.{entry}.value" if entry is not None else EVALUATION_TASK


def _initial_trial_documents(definition, bundle, *, first: bool, objective=None) -> _InitialTrialDocuments:
    subject = definition.input_documents["subject"]
    if subject.schema_id == LOADED_MODEL_SCHEMA_ID:
        model = definition.pinned_model
        if model is None:
            raise ProjectFolderInvalid("The selected Loaded Model is unavailable.")
    else:
        model = StoredModelDocument.model_validate(subject.payload).lower(bundle.components)
    task = definition.input_documents["task"]
    if task.schema_id == PYTHON_TASK_SCHEMA_ID:
        source = PythonTaskSelection.model_validate(task.payload)
    else:
        source = _authoring_task(StoredTaskDocument.model_validate(task.payload), first=first)
    if objective is None and (entry := definition.input_documents.get("objective")) is not None:
        objective = StoredObjectiveDocument.model_validate(entry.payload).lower(first=first)
    differences = MODEL_DIFFERENCE_BATCHES.validate_python(definition.payload.get(MODEL_ATTACHMENTS, {}))
    if differences and subject.schema_id != LOADED_MODEL_SCHEMA_ID:
        writes = tuple(item for batch in differences.values() for item in batch)
        model = resolve_effective_model(
            model, bundle.components, lower_model_difference_values(model, bundle.components, writes),
            functions=bundle.expression_functions,
        )
    params = authored_rollout_settings(definition.payload.get("params"))
    return _InitialTrialDocuments(
        model, source, objective,
        params.input_bindings,
    )


def _evaluate_documents(plan, key, definition, bundle, *, first: bool, objective):
    upstream = evaluate_subject(definition)
    authored = None
    if upstream:
        subject = next((row for row in plan.runs if row.key == upstream[0]), None)
        raw = plan.requests.get(subject.request_id) if subject else None
        if raw is None:
            raise ProjectFolderInvalid("The selected Evaluate has no planned model-producing subject.")
        request = decode_request(raw, plan.documents.__getitem__).spec
        if not isinstance(request, (DocumentTrainingRequestSpec, SynthesisRequestSpec)):
            raise ProjectFolderInvalid("The selected Evaluate has no managed model-producing subject.")
        documents = request.trial_documents
        if objective is not None:
            documents = documents.model_copy(update={"objective": objective})
        if subject.train is not None and isinstance(subject.train.task, TaskDocument):
            authored = subject.train
            documents = documents.model_copy(update={"task": authored.task, "input_bindings": authored.input_bindings})
    elif "subject" in definition.input_documents:
        documents = _initial_trial_documents(definition, bundle, first=first, objective=objective)
    else:
        raise ProjectFolderInvalid("The selected Evaluate has no authored trial pairing.")
    stored_source = None
    if first and upstream:
        stored_source = _stored_task_view(plan, definition, documents.task_source, include_variation=False)
        if isinstance(stored_source, StoredTaskDocument):
            preview_task = _authoring_task(stored_source)
            task = documents.task_source
            if isinstance(task, TaskDocument) and isinstance(preview_task, TaskDocument):
                preview_task = task.model_copy(update={"validation_trials": preview_task.validation_trials})
            documents = _with_task(documents, preview_task)
    objective_use = ObjectiveUse.INHERITED if upstream else ObjectiveUse.TRAIN
    projection = _project(documents, bundle, objective_use=objective_use)
    if projection.pair is None:
        return documents, projection, None
    try:
        if definition.payload.get("evaluation_type") != ROLLOUT_TYPE:
            raise ValueError("Task authoring requires a saved rollout Evaluate.")
        params = authored_rollout_settings(definition.payload.get("params"))
        own = (definition.payload.get("trial_documents") or {}).get("task")
        if own is not None:
            stored_source = stored_trial_task_slot(own)
        elif stored_source is None:
            stored_source = _stored_task_view(plan, definition, documents.task_source, include_variation=False)
        stored_task = resolve_authored_evaluation_task(stored_source, params.variation)
        authored_task = _authoring_task(stored_task, first=first) if isinstance(stored_task, StoredTaskDocument) else stored_task
        if isinstance(authored_task, StoredTaskDocument):
            draft = _with_task(documents, authored_task, input_bindings=documents.input_bindings | params.input_bindings)
            return draft, replace(_project(draft, bundle), task_origins=_task_edit_origins(
                tuple(authored_variation(params.variation)),
            )), None
        task = authored_task
        authored_bindings = params.input_bindings
        edits = tuple(authored_variation(params.variation))
        source_graph = projection.pair.lowered.graph
        graph_edits = tuple(patch for _, patch in edits if patch.path.split(".", 1)[0] != "task")
        graph = resolve_evaluation_variant(source_graph, task, graph_edits, components=bundle.components).graph if (
            graph_edits
        ) else source_graph
        run = next((row for row in plan.runs if row.key == key), None)
        if run is not None:
            effective = decode_request(plan.requests[run.request_id], plan.documents.__getitem__)
            if isinstance(effective.spec, EvaluationRequestSpec):
                settings = RolloutParameters.model_validate(effective.spec.params)
                subject_ref = effective.inputs["subject"][0].reference
                if upstream and isinstance(subject_ref, BestRank):
                    candidate = next(candidate for candidate in subject_ref.candidates
                                     if (candidate.variant, candidate.row) == (upstream[0].variant, upstream[0].row))
                    settings = settings.for_best_candidate(candidate.key)
                if settings.variation and settings.variation != params.variation:
                    lowered = resolve_evaluation_variant(
                        source_graph, documents.task_source, settings.variation, components=bundle.components,
                    )
                    graph = lowered.graph
                    if any(patch.path.split(".", 1)[0] == "task"
                           for _, patch in authored_variation(settings.variation)):
                        task = _with_authored_task_details(lowered.task, authored_task)
                        if first and isinstance(task, TaskDocument) and isinstance(authored_task, TaskDocument):
                            task = task.model_copy(update={"validation_trials": authored_task.validation_trials})
                params = settings
    except ValueError as error:
        return documents, _Projection(
            projection.pair, projection.variables, projection.signals, projection.epochs,
            projection.fields, (*projection.problems, Problem(
                code=ProblemCode.STRUCTURE, path=("variation",), reason=str(error),
            )),
        ), None
    documents = _with_task(
        documents, task, input_bindings=documents.input_bindings | authored_bindings | params.input_bindings,
    )
    paired = _project(documents, bundle, graph_override=graph, objective_use=objective_use)
    project = plan.variant_projects[key.variant]
    if (
        isinstance(stored_task, StoredTaskDocument) and isinstance(task, TaskDocument)
        and (definition.task_additions or project.steps[key.step].set_fields)
    ):
        set_ids = {id for id, _ in _set_variables(project, key, plan.row(key))}
        additions = {
            variable.id: project_member(variable, StoredVariable, label=variable.name)
            for variable in task.variables if variable.id in set_ids and variable.id not in stored_task.variables
        }
        if additions:
            stored_task = stored_task.model_copy(update={"variables": stored_task.variables | additions})
    return documents, replace(
        paired, task_origins=_task_edit_origins(edits), stored_task=stored_task,
        set_condition_variables=authored_task.variables if isinstance(authored_task, TaskDocument) else (),
    ), graph


def _with_authored_task_details(task, authored):
    """Keep authored names and Scene markers on the Task used by the runtime variant."""
    if not isinstance(task, TaskDocument) or not isinstance(authored, TaskDocument):
        return task
    source = {variable.id: variable for variable in authored.variables}
    signals = {signal.id: signal.name for signal in authored.signals}
    epochs = {epoch.id: epoch.name for epoch in authored.epochs}
    return task.model_copy(update={
        "variables": tuple(
            variable.model_copy(update={"name": item.name, "marker": item.marker})
            if (item := source.get(variable.id)) is not None else variable
            for variable in task.variables
        ),
        "signals": tuple(signal.model_copy(update={"name": signals.get(signal.id, signal.name)}) for signal in task.signals),
        "epochs": tuple(epoch.model_copy(update={"name": epochs.get(epoch.id, epoch.name)}) for epoch in task.epochs),
    })


def _task_edit_origins(edits):
    origins = {}
    for path, patch in edits:
        parts = tuple(patch.path.split("."))
        if parts[0] == "task":
            if len(parts) == 1:
                origins.clear()
            origins[parts] = path
    return origins


def _definition(plan, key):
    """The run's definition; a run whose own definition is refused is refused with its problem."""
    definition = plan.selected_definitions.get(key)
    if definition is None:
        problem = next((
            item for item in plan.diagnostics
            if item.step == key.step and item.row == key.row and item.variant in (None, key.variant)
        ), None)
        if problem is not None:
            raise ProjectFolderInvalid(problem.detail)
        raise ProjectFolderInvalid("The selected run no longer belongs to this step.")
    return definition


def _effective_task_payload(plan, definition, *, include_variation=True):
    """The stored Task source, when the selected pairing still has an authored Task."""
    variation = definition.payload.get("params", {}).get("variation")
    if include_variation and (entry := _authored_task_entry(variation)) is not None:
        return variation[entry].get("value")
    if task := definition.input_documents.get("task"):
        return task.payload
    slots, _ = trial_document_slots(definition.payload)
    if task := slots.get("task"):
        return task
    if not (upstream := evaluate_subject(definition)):
        return None
    if parent := plan.selected_definitions.get(upstream[0]):
        return _effective_task_payload(plan, parent)
    row = plan.row(upstream[0])
    if row is None:
        return None
    slot = "payload.task" if row.kind == "synthesize" else "payload.trial_documents.task"
    if not (document := row.document_ids.get(slot)):
        return None
    project = plan.variant_projects[upstream[0].variant]
    task = compose_document(project.documents.get, document)
    return None if isinstance(task, ProjectDiagnostic) else task.payload


def _stored_task_view(plan, definition, task, *, include_variation=True, scopes=None):
    if isinstance(task, PythonTaskSelection | StoredTaskDocument):
        return task
    payload = _effective_task_payload(plan, definition, include_variation=include_variation)
    authored = StoredTaskDocument.model_validate(payload) if isinstance(payload, dict) else None
    if authored is not None and _authoring_task(authored) == task:
        return authored
    return StoredTaskDocument.from_lowered(
        task, validation_trials=authored.validation_trials if authored is not None else None, scopes=scopes,
    )


def _stored_objective_view(plan, key, definition, objective=None) -> StoredObjectiveDocument | None:
    step = plan.steps[key.variant, key.step]
    if step.kind in ("train", "synthesize"):
        slots, _ = trial_document_slots(definition.payload)
        value = slots.get("objective")
        return StoredObjectiveDocument.model_validate(value) if value is not None else None
    if entry := definition.input_documents.get("objective"):
        return StoredObjectiveDocument.model_validate(entry.payload)
    if subject := evaluate_subject(definition):
        parent = subject[0]
        if upstream := plan.selected_definitions.get(parent):
            return _stored_objective_view(plan, parent, upstream, objective)
        row = plan.row(parent)
        if row is not None:
            slot = "payload.objective" if row.kind == "synthesize" else "payload.trial_documents.objective"
            project = plan.variant_projects[parent.variant]
            if (name := row.document_ids.get(slot)) and not isinstance(
                entry := compose_document(project.documents.get, name), ProjectDiagnostic,
            ):
                return StoredObjectiveDocument.model_validate(entry.payload)
    return StoredObjectiveDocument.from_lowered(objective) if objective is not None else None


def _planned_family_problems(plan, key, families, preview):
    """Use addressed Problems from the full Plan without expanding for authoring."""
    if not families.families:
        return ()
    row = plan.row(key)
    candidates = [
        (json_pointer_parts(problem.path), problem.detail, None)
        for problem in (row.problems if row is not None else ()) if problem.path.startswith("/")
    ]
    candidates.extend(
        (tuple(problem.path.split(".")), problem.detail, problem.span)
        for problem in plan.diagnostics
        if (problem.variant, problem.step, problem.row) == (key.variant, key.step, key.row)
        and problem.path.startswith(("payload.", "inputs."))
    )
    held = {(problem.path, problem.reason) for problem in preview}
    found = []
    for path, reason, span in candidates:
        if path[:1] == ("payload",):
            path = path[1:]
        if path[:1] in (("trial_documents",), ("inputs",)):
            path = path[1:]
        if not is_family_member_path(path) or pointer(path[:3]) not in families.families:
            continue
        if (path, reason) not in held:
            found.append(Problem(code=ProblemCode.STRUCTURE, path=path, reason=reason, span=span))
            held.add((path, reason))
        if path[3:4] in (("over",), ("combination",), ("label",)):
            family = pointer(path[:3])
            families.families[family] = families.families[family].model_copy(update={"member_count": None})
    return tuple(found)


def _pairing(plan, key, bundle, *, definition=None, first: bool = True):
    if definition is None:
        definition = _definition(plan, key)
    families = FamilyEditor()
    objective = (
        objective_editor(stored, families, bundle.expression_functions)
        if first and (stored := _stored_objective_view(plan, key, definition)) is not None else None
    )
    documents, projected, graph = _pairing_documents(
        plan, key, bundle, definition, first=first, objective=objective,
    )
    if first:
        stored_task = projected.stored_task or _stored_task_view(
            plan, definition, documents.task_source, scopes=_set_spellings(projected.variables),
        )
        if isinstance(stored_task, StoredTaskDocument) and _effective_task_payload(plan, definition) is not None:
            validation_editor(stored_task, families, projected.variables, bundle.expression_functions)
        fields = {
            path: with_family_indices(site, families.names.get(path[:3], ()))
            if path[:2] == ("objective", "terms") else site
            for path, site in projected.fields.items()
            if path[:2] != ("task", "validation_trials")
        }
        problems = (*map(families.problem, projected.problems), *families.problems)
        projected = replace(
            projected, fields=fields | {path: authored.site for path, authored in families.fields.items()},
            problems=(*problems, *_planned_family_problems(plan, key, families, problems)),
            family_editor=families, stored_task=stored_task,
        )
    origins, trial_sets = _set_origins(plan, key, definition, projected)
    hidden = _generated_set_members(plan, key, definition, projected, origins) if first and trial_sets else set()

    def member(value: ScopedValue) -> ScopedValue:
        return value.model_copy(update={"set_entry": item.origin}) if (item := origins.get(value.id)) else value

    if origins or hidden:
        @cache
        def indices(axis, name, key):
            item = next(
                item for item in origins.values()
                if (item.origin.axis, item.origin.set, item.origin.entry) == (axis, name, key)
            )
            path = ("set_fields", axis, "entries", key)
            try:
                preview = next(expand_family(key, item.source, first=True))
                return family_indices(preview, bundle.expression_functions, path)
            except FamilyProblem as error:
                return (), (family_problem(error, path),)

        family_problems = []
        fields = {}
        for path, site in projected.fields.items():
            if len(path) > 2 and path[0] == "task" and path[1] in {"variables", "signals"} and path[2] in hidden:
                continue
            scoped = site.model_copy(update={"names": tuple(member(name) for name in site.names if name.id not in hidden)})
            if (path[:2] == ("task", "variables") and len(path) == 4 and path[3] in ("value", "count")
                    and (item := origins.get(path[2])) is not None and item.source.over is not None):
                names, problems = indices(item.origin.axis, item.origin.set, item.origin.entry)
                family_problems.extend(problem for problem in problems if problem not in family_problems)
                scoped = with_family_indices(scoped, names)
            fields[path] = scoped
        condition_variables = projected.set_condition_variables
        projected = replace(
            projected, variables=tuple(member(value) for value in projected.variables if value.id not in hidden),
            signals=tuple(value for value in projected.signals if value.id not in hidden),
            fields=fields, set_variables=origins, problems=(*projected.problems, *family_problems),
            generated_set_members=hidden, set_condition_variables=(
                tuple(value for value in condition_variables if value.id not in hidden)
                if condition_variables is not None else None
            ),
        )
    return documents, projected, graph


def _pairing_documents(plan, key, bundle, definition, *, first: bool, objective):
    step = plan.steps[key.variant, key.step]
    if step.kind == "train":
        stored = definition.payload.get("trial_documents")
        try:
            documents = lower_stored_trial_documents(stored, components=bundle.components, first=first, objective=objective)
        except StoredTaskDraftError:
            draft = _DraftTrialDocuments(
                model=StoredModelDocument.model_validate(stored["model"]).lower(bundle.components),
                task_source=StoredTaskDocument.model_validate(stored["task"]),
                objective=objective if objective is not None else StoredObjectiveDocument.model_validate(stored["objective"]).lower(first=first),
                input_bindings=stored.get("input_bindings", {}),
            )
            return draft, _project(draft, bundle), None
        run = next((item for item in plan.runs if item.key == key), None)
        task = run.train.task if run is not None and run.train is not None else None
        if isinstance(task, TaskDocument):
            if first and documents.task is not None:
                task = task.model_copy(update={"validation_trials": documents.task.validation_trials})
            if task != documents.task_source and (encoded := plan.requests.get(run.request_id)) is not None:
                spec = decode_request(encoded, plan.documents.__getitem__).spec
                if isinstance(spec, DocumentTrainingRequestSpec):
                    documents = spec.trial_documents.model_copy(update={
                        "task": _with_authored_task_details(task, documents.task),
                        "objective": documents.objective,
                    })
        return documents, _project(documents, bundle), None
    if step.kind == "synthesize":
        slots = definition.payload
        task = stored_trial_task_slot(slots.get("task"))
        if isinstance(task, StoredTaskDocument):
            task = _authoring_task(task, first=first)
        model = StoredModelDocument.model_validate(slots.get("model")).lower(bundle.components)
        if objective is None:
            objective = StoredObjectiveDocument.model_validate(slots.get("objective")).lower(first=first)
        bindings = slots.get("input_bindings", {})
        documents = _DraftTrialDocuments(model, task, objective, bindings) if isinstance(
            task, StoredTaskDocument,
        ) else TrialDocuments(
            model=model,
            task=task if isinstance(task, TaskDocument) else None,
            python_task=task if isinstance(task, PythonTaskSelection) else None,
            objective=objective, input_bindings=bindings,
        )
        return documents, _project(documents, bundle), None
    if step.kind == "evaluate":
        return _evaluate_documents(plan, key, definition, bundle, first=first, objective=objective)
    raise ProjectFolderInvalid("The selected run has no authored trial pairing.")


def read_run_trial_sample(
    folders: ProjectFolderService, project: str, revision: str, key: RunKey, bundle: ApplicationRegistryBundle,
) -> tuple[TrialDocuments | _InitialTrialDocuments, GraphSpec | None]:
    """Read the selected run's model and trial inputs, including its effective Set and Evaluate changes."""
    plan = folders.read_plan(project, revision=revision, source_runs=(key,)).plan
    definition = _definition(plan, key)
    run = plan.row(key)
    if isinstance(run, PlannedRun) and run.kind == "train":
        spec = decode_request(plan.requests[run.request_id], plan.documents.__getitem__).spec
        if isinstance(spec, DocumentTrainingRequestSpec):
            documents = spec.trial_documents
            if run.train is not None and run.train.task is not None:
                documents = _with_task(documents, run.train.task)
            return documents, None
    try:
        documents, _, graph = _pairing(plan, key, bundle, definition=definition, first=False)
        if isinstance(documents.task_source, StoredTaskDocument):
            documents.task_source.lower()
    except (ValueError, TypeError, KeyError) as error:
        raise ProjectFolderInvalid(str(error)) from error
    return documents, graph


def _set_variables(project: Project, key: RunKey, row: PlannedRow | None) -> Iterator[tuple[str, _SetVariableView]]:
    """Find each visible variable's authored Set entry, retaining its runtime key."""
    if row is None:
        return
    template = compose_step(project, key.step)
    sweep = template.sweep if not isinstance(template, ProjectDiagnostic) else None
    arms = [(axis, read.set, row.axis_keys[axis]) for axis, read in (sweep.axes.items() if sweep is not None else ())
            if isinstance(read, SetAxis) and axis in row.axis_keys]
    arms.extend((axis, field.axis.set, None) for axis, field in project.steps[key.step].set_fields.items())
    for axis, name, selected in arms:
        try:
            entries = read_set(project.documents.get, name).entries
        except ValueError:
            continue
        keys = entries if selected is None else (row.set_entries.get(axis, selected),)
        for entry in keys:
            source = entries.get(entry)
            if source is None:
                continue
            try:
                generated = next(expand_family(entry, source, first=True)).key if selected is None else selected
            except FamilyProblem:
                continue  # The Plan reports the invalid family at its authored rule.
            for variable in source.task_variables:
                id = scoped_set_member_id(axis, generated, variable) if selected is None else variable
                origin = SetVariableOrigin(step=key.step, axis=axis, set=name, entry=entry, variable=variable)
                yield id, _SetVariableView(origin, source, generated, scoped=selected is None)


def _set_origins(plan, key, definition, projection) -> tuple[dict[str, _SetVariableView], bool]:
    """Read authored Set origins for this run and any inherited placements."""
    sources = [key]
    if (
        ("task",) not in projection.task_origins
        and not definition.payload.get("trial_documents", {}).get("task")
        and (subject := evaluate_subject(definition)) is not None
    ):
        sources.insert(0, subject[0])
    origins = {}
    trial_sets = False
    for source in sources:
        variant = plan.variant_projects[source.variant]
        trial_sets |= bool(variant.steps[source.step].set_fields)
        origins.update(_set_variables(variant, source, plan.row(source)))
    return origins, trial_sets


def _generated_set_members(plan, key, definition, projection, origins) -> set[str]:
    """Keep authored Task members and each Set entry's existing editor representative."""
    task = projection.stored_task
    payload = _effective_task_payload(plan, definition, include_variation=False)
    if not isinstance(task, StoredTaskDocument) or not isinstance(payload, dict):
        return set()
    authored = stored_trial_task_slot(payload)
    if plan.steps[key.variant, key.step].kind == "evaluate" and projection.task_origins:
        settings = authored_rollout_settings(definition.payload.get("params"))
        authored = resolve_authored_evaluation_task(authored, settings.variation)
    if not isinstance(authored, StoredTaskDocument):
        return set()
    visible = authored.variables.keys() | authored.signals.keys() | origins.keys()
    members = task.variables.keys() | task.signals.keys() | {
        value.id for value in (*projection.variables, *projection.signals)
    }
    return members - visible


def _set_variable_edit(project, plan, key, path, change, site):
    if path[:2] != ("task", "variables") or len(path) != 4 or path[-1] not in ("value", "count"):
        return None
    choices = [item for id, item in _set_variables(project, key, plan.row(key)) if id == path[2]]
    if len(choices) > 1:
        raise ProjectFolderInvalid("The variable belongs to more than one selected Set entry.")
    if not choices:
        return None
    [item] = choices
    origin, source = item.origin, item.source
    (edit,) = change.edits
    assert isinstance(edit, SetProjectValue)
    value = edit.value
    if item.scoped:
        refs = {scoped_set_member_id(origin.axis, item.generated, key): key for key in source.task_variables}
        expression = lower_expression(_STORED_EXPRESSION.validate_python(value))
        value = stored_expression_value(expression, scopes=_set_spellings(site.names) | refs)
    previous = stored_project_value(source.task_variables[origin.variable], keep_explicit=True)
    variables = source.task_variables | {origin.variable: StoredVariable.model_validate(previous | {path[-1]: value})}
    return item, SetEntryTask(
        kind="set_entry_task", step=key.step, axis=origin.axis, entry=origin.entry, task_variables=variables,
    )


def _set_condition_site(definition, projection):
    variables = projection.set_condition_variables
    if variables is None:
        task = definition.payload.get("trial_documents", {}).get("task")
        if task is not None:
            authored = StoredTaskDocument.model_validate(task)
            variables = tuple(variable.lower(key) for key, variable in authored.variables.items())
        else:
            variables = ()
    resolved = {variable.id: variable for variable in projection.variables}
    return trial_set_condition_site(tuple(
        resolved.get(variable.id) or ScopedValue(id=variable.id, name=variable.name, type=variable.type, level=Level.TRIAL)
        for variable in variables
    ))


def _set_condition_edit(expression, path, body, folders, project, settled, key, bundle, definition, projection):
    step = project.steps[key.step]
    if step.kind not in {"train", "evaluate"} or len(path) != 3 or path[-1] != "condition" or path[1] not in step.set_fields:
        raise ProjectFolderInvalid("The selected Set condition is unavailable.")
    edit = expression.edit
    if edit.kind != "text" or edit.value_id is not None:
        return StoredEdit([]), _one_expression_problem(path), None
    field = step.set_fields[path[1]]
    address = AuthoredFileAddress(kind="step", id=key.step)
    if edit.text.strip():
        site = _set_condition_site(definition, projection)
        condition = _one_expression(edit.text, site, bundle.expression_functions, path)
        if isinstance(condition, tuple):
            return StoredEdit([]), condition, None
        saved = stored_expression_value(condition, scopes=_set_spellings(site.names))
        if field.condition is not None and saved == _STORED_EXPRESSION.dump_python(field.condition, mode="json"):
            return StoredEdit([]), (), None
        raw = step.model_dump(mode="json", by_alias=True, exclude_unset=True)
        file = file_value(address, raw, path, saved)
    elif field.condition is not None:
        file = StudioProjectFilePatch(address=address, patches=[RemoveProjectValue(op="remove", path=pointer(path))])
    else:
        return StoredEdit([]), (), None
    selected = selected_runs(settled.plan, key, body.scope or StudioTrialStepScope(kind="step"))
    return _validate_files(
        body, project, settled, key, selected, folders, bundle,
        [("whole_step", "Whole Step", key.step, StoredEdit([file]))], edited_rows=edited_authored_rows,
    )


def _prepare_edits(
    body, folders, project_id, project, settled, key, bundle, definition, documents, projected,
    stored_task, stored_objective, *, change, sweep_source,
):
    scope = body.scope
    plan = settled.plan
    step = plan.steps[key.variant, key.step]
    expression = change if isinstance(change, StudioTrialExpressionEdit) else None
    if expression is not None:
        path = json_pointer_parts(expression.path)
        if path[:1] == ("set_fields",):
            return _set_condition_edit(expression, path, body, folders, project, settled, key, bundle, definition, projected)
        holder = next((path[:end] for end in range(len(path), 0, -1) if path[:end] in projected.fields), path)
        site = projected.fields.get(holder)
        swept = _field_sweep(project, key, path, site, _field_label(holder), sweep_source) if (
            site is not None and projected.pair is not None
        ) else None
        if holder != path and swept is None:
            raise ProjectFolderInvalid("The selected field is unavailable.")
        if swept is not None and (holder != path or routes_to_sweep(expression.edit, swept, site, bundle.expression_functions)):
            readers = lambda name: plan.document_users(key.variant, name)  # noqa: E731
            source = sweep_source(swept.role)
            assert source is not None
            shown = source.document
            variant = None if key.variant == plan.variants[0] else key.variant
            made = trial_field_sweep_edit(
                expression.edit, swept, key.step, source.payload, bundle.expression_functions,
                (lambda edits: document_write(project, shown, edits, {key.step}, readers)) if step.kind in {"train", "synthesize"}
                else lambda edits: evaluation_document_write(project, plan, shown, edits, {key.step}, "task", variant),
            )
            if isinstance(made, tuple):
                return StoredEdit([]), made, None
            selected = selected_runs(plan, key, scope or StudioTrialStepScope(kind="step"))
            return _validate_files(
                body, project, settled, key, selected, folders, bundle,
                [("whole_step", "Whole Step", key.step, made)],
            )
        change = _expression_edit(expression, documents, stored_task, stored_objective, projected, bundle)
        if not isinstance(change, StudioTrialDocumentEdit):
            return StoredEdit([]), change, None
        entry = _set_variable_edit(project, plan, key, path, change, site)
        if entry is not None:
            item, command = entry
            name = item.origin.set
            if scope is None and item.source.over is not None and not item.scoped:
                selected = {
                    row.key for row in plan.rows if (row.key.variant, row.key.step) == (key.variant, key.step)
                    and row.set_entries.get(command.axis) == command.entry
                }
            else:
                selection = scope or (
                    StudioTrialStepScope(kind="step") if item.scoped
                    else AxisValueRuns(kind="axis_value", axis=command.axis, value=item.generated)
                )
                selected = selected_runs(plan, key, selection)
            stored = prepare_set_entry_task(project, name, command)
            narrowed = prepare_set_entry_task(project, name, command, {run.step for run in selected})
            candidates = [("set_entry", "Set Entry", name, stored)]
            if narrowed != stored:
                candidates.append(("selected_readers", "Selected Readers", name, narrowed))
            return _validate_files(body, project, settled, key, selected, folders, bundle, candidates)
    renames = ()
    structural = isinstance(change, StudioTrialStructureEdit)
    if structural:
        renames = tuple(item for item in change.changes if isinstance(item, RenameTaskMember))
        change = _structure_edit(change, documents.task_source, stored_task, _set_spellings(projected.variables))
        if not isinstance(change, StudioTrialDocumentEdit):
            return StoredEdit([]), change, None
    document = None
    document_path = None
    edited_role = (change.document if isinstance(change, StudioTrialDocumentReset)
                   else change.kind if isinstance(change, StudioTrialDocumentEdit) else None)
    if edited_role == "task" and isinstance(change, StudioTrialDocumentEdit) and any(
        json_pointer_parts(edit.path)[:1] == ("validation_trials",) for edit in change.edits
    ) and _effective_task_payload(plan, definition) is None:
        raise ProjectFolderInvalid("This Task's validation rows have no authored keys to edit.")
    if isinstance(documents, _InitialTrialDocuments) and edited_role is not None:
        document_path = f"inputs.{edited_role}"
    elif step.kind in {"train", "synthesize"} and edited_role is not None:
        field, _ = _changed_value(documents, definition, edited_role)
        _, prefix = trial_document_slots(definition.payload)
        document_path = f"payload.{prefix}{field}"
    if document_path is not None:
        document = _whole_document(definition, document_path)
    if isinstance(change, StudioTrialDocumentReset):
        restored = document if document_path is not None else (
            _whole_document(definition, _evaluation_task_path(definition)) if change.document == "task" else None
        )
        if restored is None:
            raise ProjectFolderInvalid("Reset needs a value the step reads from a document.")
        change = StudioTrialDocumentEdit(kind=change.document, edits=base_value_edits(project, restored, change.paths))

    variant = None if key.variant == plan.variants[0] else key.variant

    def readers(name):
        return plan.document_users(key.variant, name)

    scope = scope or StudioTrialStepScope(kind="step")
    whole = isinstance(scope, StudioTrialStepScope)
    chosen = set(scope.chosen_steps) if whole and scope.chosen_steps else {key.step}
    view = variant_view(plan, project, variant if whole else None)
    users = set(lit_steps(view.documents, readers, document)) if document else set()
    if document and whole and (key.step not in chosen or not chosen <= users):
        raise ProjectFolderInvalid("The chosen steps no longer match this document.")
    selected = selected_runs(plan, key, scope, chosen_steps=chosen)
    evaluate_field = step.kind == "evaluate" and isinstance(change, StudioTrialDocumentEdit) and change.kind == "task"

    def store_for(target, members, *, step_id=key.step, selected_view=view):
        if step.kind == "evaluate":
            if isinstance(change, StudioTrialSelectDocument):
                role = change.role
                if role == "task":
                    _chosen_task(project, change.document)
                elif role == "model" and whole:
                    _chosen_document(project, change.document, role)
                else:
                    raise ProjectFolderInvalid("An Evaluate reads its Objective and Training from its subject.")
                reference = {"document": change.document}
                if isinstance(target, StudioTrialStepScope | VariantRuns):
                    slots = (authored_payload(selected_view, step_id) or {}).get("trial_documents")
                    writes = [
                        _override(f"trial_documents.{role}", "replace" if role in slots else "add", reference)
                        if isinstance(slots, dict) else _override("trial_documents", "add", {role: reference}),
                    ]
                else:
                    writes = _task_selection_writes(definition, reference)
            elif evaluate_field and isinstance(target, StudioTrialStepScope | VariantRuns):
                return _evaluate_task_write(project, plan, key, members, change, variant)
            elif evaluate_field:
                return _evaluate_field_files(project, key, target, change, stored_task)
            else:
                raise ProjectFolderInvalid("Evaluate authoring edits the task variation.")
        elif step.kind == "train" and isinstance(change, StudioTrialSelectDocument) and change.role == "training":
            writes = training_document_choice(project, step_id, bundle.training_methods, change.document)
        else:
            writes = _producer_trial_writes(change, project, definition, documents, projected)
        return scoped_file(
            project, step_id, target, writes, chosen=members, users=readers, view=selected_view,
        ) if writes else StoredEdit([])

    stored_scope = VariantRuns(kind="variant", variant=variant) if whole and variant else scope
    if document is not None and whole:
        shown = stored_objective if change.kind == "objective" else stored_task
        raw = stored_project_value(shown, keep_explicit=True)

        def adapt(target):
            edits = family_rule_edits(target, change.kind, change.edits)
            if change.kind == "objective":
                return _objective_edits(target, change.model_copy(update={"edits": edits}))
            if structural:
                return _task_structure_edits(raw, target, change)
            return edits

        def write_for(members):
            return document_write(project, document, change.edits, members, readers, variant, view=view, adapt=adapt)

        candidates = [("all_readers", "All Steps", document, write_for(users))]
        if chosen != users:
            candidates.append(("selected_readers", "Only This Step", document, write_for(chosen)))
    else:
        stored = store_for(stored_scope, chosen)
        kind, label = (
            ("sweep_value", "Sweep Value") if isinstance(scope, AxisValueRuns)
            else ("row_exception", "Selected Runs") if isinstance(scope, ListedRuns)
            else ("variant", "Variant") if variant
            else ("whole_step", "Whole Step")
        )
        candidates = [(kind, label, repr(stored_scope), stored)]

    def controlling(plan, missed):
        if step.kind not in {"train", "synthesize"}:
            return
        writes = _producer_trial_writes(change, project, definition, documents, projected)
        if not writes or any(write.op != "replace" for write in writes):
            return
        for run in sorted(missed, key=lambda item: (item.variant, item.step, item.row)):
            row = plan.selected_definitions.get(run)
            if row is None:
                continue
            origins = []
            for write in writes:
                indexed = index_pointer(row.payload, tuple(write.path.split(".")))
                origin = row.sources.get(f"/payload{indexed}") if indexed else None
                if origin is None:
                    break
                current = at(row.payload, json_pointer_parts(indexed))
                candidate = source_value_file(project, origin, current, write.value)
                if candidate is None:
                    break
                origins.append((origin, candidate))
            if len(origins) != len(writes):
                continue
            source = origins[0][0]
            if any(_source_store(origin) != _source_store(source) for origin, _ in origins[1:]):
                continue
            identity = repr(_source_store(source))
            label: dict[str, tuple[StoreKind, str]] = {
                "sweep_value": ("sweep_value", "Sweep Value"),
                "exception": ("row_exception", "Row Exception"),
                "variant": ("variant", "Variant"),
            }
            kind, name = label[source.kind]
            yield kind, name, identity, StoredEdit([]).then(file for _, file in origins)

    return _validate_files(
        body, project, settled, key, selected, folders, bundle, candidates,
        controlling=controlling if isinstance(change, StudioTrialDocumentEdit | StudioTrialBindingEdit) else None,
        other_runs=lambda rows: store_for(ListedRuns(kind="rows", rows=tuple(sorted(row.row for row in rows))), chosen),
        renames=renames,
    )


def _source_store(origin: ValueSource):
    return origin.kind, origin.step, origin.name, origin.variant, origin.axis, origin.value


def _structure_edit(change, task, stored_task, scopes) -> StudioTrialDocumentEdit | tuple[Problem, ...]:
    """Apply the structural commands atomically and store their keyed document edits."""
    if not isinstance(task, TaskDocument):
        raise ProjectFolderInvalid("A Python task has no structure to change.")
    changed = task
    for each in change.changes:
        changed = restructure(changed, each)
        if isinstance(changed, Problem):
            return (changed,)
    removed_variables = stored_task.variables.keys() - {member.id for member in changed.variables}
    removed_epochs = stored_task.epochs.keys() - {member.id for member in changed.epochs}
    trials = {}
    for key, trial in stored_task.validation_trials.items():
        changes = {}
        if removed_variables & trial.variables.keys():
            changes["variables"] = {name: value for name, value in trial.variables.items() if name not in removed_variables}
        if removed_epochs & trial.epoch_steps.keys():
            changes["epoch_steps"] = {name: value for name, value in trial.epoch_steps.items() if name not in removed_epochs}
        trials[key] = trial.model_copy(update=changes) if changes else trial
    after = StoredTaskDocument.from_lowered(changed, validation_trials=trials, scopes=scopes)
    raw = stored_project_value(stored_task, keep_explicit=True)
    edits = content_edits(
        raw, stored_project_value(after, keep_explicit=True), schema_id=STORED_TASK_DOCUMENT_SCHEMA_ID,
    )
    if not edits:
        raise ProjectFolderInvalid("The change leaves the task as it is.")
    return _present_ops(raw, StudioTrialDocumentEdit(kind="task", edits=edits))


def _task_structure_edits(shown, target, change):
    """Carry the selected structural changes to a destination that has its own member values."""
    keyed = partial(stored_keyed_collection, STORED_TASK_DOCUMENT_SCHEMA_ID)
    differences = diff(shown, _edit_value(shown, change), keyed_collection=keyed)
    applicable = {
        path: difference for path, difference in differences.items()
        if isinstance(at(target, json_pointer_parts(path) if isinstance(difference, OrderDocumentMembers)
                         else json_pointer_parts(path)[:-1]), dict)
    }
    return content_edits(
        target, apply_differences(target, applicable, keyed_collection=keyed),
        schema_id=STORED_TASK_DOCUMENT_SCHEMA_ID,
    )


def _objective_edits(target, change):
    """Place edits on the same keyed terms, preserving each destination's own members."""
    terms = target.get("terms", {})
    edits = []
    for edit in change.edits:
        parts = json_pointer_parts(edit.path)
        if isinstance(edit, OrderProjectValue):
            if at(target, parts) is not None:
                edits.append(edit)
        elif parts == ("terms",) and edit.op == "add" and isinstance(edit.value, dict) and "terms" in target:
            edits.extend(SetProjectValue(op="add", path=pointer(("terms", key)), value=value)
                         for key, value in edit.value.items() if key not in terms)
        elif isinstance(at(target, parts[:-1]), dict):
            edits.append(edit)
    return edits


def _whole_document(definition, path: str) -> str | None:
    """The document the step reads whole at ``path``, its payload."""
    return None if definition.document_pointers.get(path) else definition.document_ids.get(path)


def _evaluate_task_write(project, plan, key, chosen, change, variant: str | None) -> StoredEdit:
    """Store an Evaluate's Task edit in a document its chosen steps read."""
    row = plan.row(key)
    document = row.document_ids.get(EVALUATION_TASK) or row.document_ids.get("payload.trial_documents.python_task")
    if document is None:
        raise ProjectFolderInvalid("The selected Evaluate reads no Task document.")
    return evaluation_document_write(
        project, plan, document, change.edits, chosen, "task", variant,
        adapt=lambda target: family_rule_edits(target, "task", change.edits),
    )


def _present_ops(raw, change):
    """``change`` with each set written as a replace where ``raw`` holds its target and as an add where it does not."""
    def holds(parts):
        value = raw
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
                value = value[int(part)]
            else:
                return False
        return True

    return change.model_copy(update={"edits": [
        edit.model_copy(update={"op": "replace" if holds(json_pointer_parts(edit.path)) else "add"})
        if isinstance(edit, SetProjectValue) and not is_family_rule_edit(change.kind, json_pointer_parts(edit.path))
        else edit for edit in change.edits
    ]})


def _evaluate_field_files(project, key, scope, change, task):
    """The step's writes of the selected runs' Task fields, at the Task's stored paths."""
    raw = stored_project_value(task, keep_explicit=True)
    return scoped_file(project, key.step, scope, _value_writes(
        f"trial_documents.{_task_field(task)}", raw, _present_ops(raw, change),
    ))


def _rename_scope_problem(
    task: TaskDocument | None, fields: dict[tuple[str | int, ...], Site], renames: tuple[RenameTaskMember, ...],
) -> Problem | None:
    """Refuse a rename that makes a Task reference ambiguous in an expression site."""
    if task is None:
        return None
    members = {
        "variable": ("variables", task.variables),
        "epoch": ("epochs", task.epochs),
        "signal": ("signals", task.signals),
    }
    for change in renames:
        section, group = members[change.member]
        if not any(item.id == change.id and item.name == change.name for item in group):
            continue
        references = epoch_outcome_references(change.id).keys() if change.member == "epoch" else {change.id}
        for site in fields.values():
            other_names = {item.name for item in site.names if item.id not in references}
            name = next((item.name for item in site.names if item.id in references and item.name in other_names), None)
            if name is not None:
                return Problem(
                    code=ProblemCode.DUPLICATE_NAME, reason=f"The name {name!r} is already in scope.",
                    path=(section, change.id, "name"),
                )
    return None


def _validate_files(
    body, project, settled, key, selected, folders, bundle, candidates,
    *, controlling=None, other_runs=None, edited_rows=None, renames: tuple[RenameTaskMember, ...] = (),
):
    """Calculate each store's reach on the authored definitions before preparing its files."""
    possible = {
        row.key for row in settled.plan.rows if row.kind in {"train", "evaluate", "synthesize"}
    }
    if not selected <= possible:
        raise ProjectFolderInvalid("The selected runs no longer belong to this edit.")

    def compile_sources(project, source_runs):
        return compile_project(
            project, readers=settled.readers, source_runs=source_runs,
            operations=folders.operations, components=bundle.components,
            functions=bundle.expression_functions, python_tasks=bundle.python_tasks,
            figure_templates=bundle.figure_templates, training_methods=bundle.training_methods,
            synthesis_methods=bundle.synthesis_methods,
        )

    before = settled.plan
    if not possible <= before.selected_definitions.keys():
        before = folders.read_plan(settled.project_id, revision=body.revision, source_runs=possible).plan

    def preview(kind, label, source, stored):
        if not stored.files and not stored.created:
            return EditStore(kind, label, source, frozenset(), (stored, project, before))
        authored = preview_stored(project, stored)
        after = compile_sources(authored, possible)
        runs = _changed_runs(before, after, stored.reached) if edited_rows is None else edited_rows(before, after)
        value = stored, authored, after
        if kind == "all_readers":
            return all_readers_store(source, runs, value)
        if kind == "selected_readers" and len({run.step for run in runs}) > 1:
            label = "Selected Steps"
        return EditStore(kind, label, source, runs, value)

    preferred, *alternatives = [preview(*candidate) for candidate in candidates]
    missed = selected - preferred.runs
    if controlling is not None and missed:
        sources = {candidate[2]: candidate for candidate in controlling(before, missed)}
        alternatives.extend(preview(*candidate) for candidate in sources.values())
    remaining = preferred.runs & selected
    if (
        other_runs is not None and remaining and preferred.runs != selected
        and not any(store.runs == remaining for store in alternatives)
        and all(run.step == key.step and run.variant == before.variants[0] for run in remaining)
    ):
        if isinstance(narrowed := other_runs(remaining), StoredEdit):
            option = preview("other_runs", "Other Runs" if missed else "Selected Runs", key.step, narrowed)
            if option.runs and option.runs <= remaining and option.runs != preferred.runs:
                alternatives.append(option)
    alternatives = [
        store for store in alternatives if store.runs != preferred.runs or store.value[0] != preferred.value[0]
    ]
    # A whole-step selection covers the rows that the edit creates in those same steps.
    if body.scope is None or isinstance(body.scope, StudioTrialStepScope):
        steps = {(run.variant, run.step) for run in selected}
        selected = selected | {
            run for store in (preferred, *alternatives) for run in store.runs - possible
            if (run.variant, run.step) in steps
        }
    decision = choose_store(
        body.revision, body.model_dump(mode="json", exclude={"store_choice"}), selected,
        preferred, alternatives, choice=body.store_choice,
    )
    if decision == "stale":
        raise ProjectFolderInvalid("The store choice is stale.")
    if decision == "unavailable":
        return StoredEdit([]), (), None
    if isinstance(decision, list):
        return StoredEdit([]), decision, None
    stored, authored, candidate = decision.value
    if renames:
        for reached in decision.runs & candidate.selected_definitions.keys():
            documents, projection, _ = _pairing(candidate, reached, bundle)
            if (problem := _rename_scope_problem(documents.task, projection.fields, renames)) is not None:
                return StoredEdit([]), (problem,), None
    return stored, (), (authored, candidate)


def _trial_member_orders(definition):
    """The member orders that are part of resolved Task and Objective content."""
    def orders(value, schema, path=()):
        if not isinstance(value, dict):
            return
        if stored_keyed_collection(schema, path):
            yield path, tuple(value)
        for key, child in value.items():
            yield from orders(child, schema, (*path, key))

    slots, _ = trial_document_slots(definition.payload)
    return tuple(
        (role, tuple(orders(payload, schema)))
        for role, schema in (
            ("task", STORED_TASK_DOCUMENT_SCHEMA_ID), ("objective", STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID),
        )
        if (payload := definition.input_documents[role].payload if role in definition.input_documents else slots.get(role))
        is not None
    )


def _changed_runs(before, after, reached) -> frozenset[RunKey]:
    """Run identities whose own resolved definition or sweep coordinates change."""
    authored = edited_authored_rows(before, after)

    def values(plan):
        return {
            row.key: (
                (definition.payload, definition.input_documents, definition.task_additions, _trial_member_orders(definition))
                if (definition := plan.selected_definitions.get(row.key)) is not None else None,
                row.document_ids, row.coordinates, row.train,
            )
            for row in plan.rows if row.kind in {"train", "evaluate", "synthesize"}
            and (row.key in authored or row.key.step in reached)
        }

    old, new = values(before), values(after)
    return frozenset(key for key in old.keys() | new.keys() if old.get(key) != new.get(key))


def _held_problems(path, problems) -> list[Problem]:
    """The problems about the field at ``path``: a task's are addressed within the task, an Objective's with its role."""
    own = (path, path[1:]) if path[0] == "task" else (path,)
    return [problem for problem in problems if any(problem.path[:len(at)] == at for at in own)]


def _field_problems(path, site: Site, text: str, functions, problems, *, family=False) -> list[Problem]:
    """The field's problems, each at the field's path; one about a term of its text carries the term's place there."""
    held = _held_problems(path, problems)
    if not held or not text:
        return [problem.model_copy(update={"path": path}) for problem in held]
    read = read_field(text, site, functions)
    if isinstance(read, tuple):
        if family:
            span = next((problem.span for problem in read if problem.span is not None), None)
            termed = [problem.model_copy(update={"span": problem.span or span}) for problem in held]
        else:
            termed = read if any(problem.span for problem in read) else held
        return [problem.model_copy(update={"path": path}) for problem in termed]
    within = len(path) - 1 if path[0] == "task" else len(path)
    return [
        problem.model_copy(update={"path": path, "span": problem.span or read.spans.get(problem.path[within:])})
        for problem in held
    ]


def _trial_fields(
    path, site: Site, expression, functions, sweep, row, problems, name_paths, period,
    menu_fields: dict[str, TrialExpressionField], sites: FieldSiteIndex,
    *, objective_use: ObjectiveUse = ObjectiveUse.TRAIN, authored: FamilyField | None = None, family=False,
) -> list[StudioTrialField]:
    """The field at ``path`` with its text, its structure where its value names only what is in scope, its problems,
    and each number in it the step can sweep: ``sweep`` gives the sweep of the number at a path, if the step can sweep it.

    A name no longer in scope is printed as its reference. A field that is a number carries its sweep; each number given
    to a draw in it is a field of its own.
    """
    missing = references(expression) - {item.id for item in site.names} if expression is not None else set()
    shown = expression is not None and not missing
    field_path = pointer(tuple(map(str, path)))
    projection = project_site(site, functions, expression, name_paths=name_paths, defer_parts=True)
    sequence_rule = is_family_rule(tuple(path)) and isinstance(expression, SequenceValue | UnresolvedValue)
    if not _epoch_end_field(path, "limit") and not sequence_rule:
        menu_fields[field_path] = TrialExpressionField(site, expression)
    fields = []
    for inner, label in numeric_nodes(expression, functions) if shown else ():
        swept = sweep((*path, *inner), site, label or _field_label(path))
        if swept is None:
            continue
        current = at(expression, inner)
        view = project_numeric_field(
            StudioTrialField, swept.field, swept.axis, functions, row, current.value,
            sites=sites, period=period, effective_unit=current.unit, defer_parts=True,
            path=pointer(tuple(map(str, (*path, *inner)))),
        )
        if inner:
            menu_fields[view.path] = TrialExpressionField(swept.field.site, None)
            fields.append(view)
        elif view.sweep is not None or view.can_sweep:
            return [view.model_copy(update={
                "site": sites.reference(projection), "problems": _field_problems(path, site, view.text, functions, problems, family=family),
            })]
    if _epoch_end_field(path, "limit"):
        projection = projection.model_copy(update={"kinds": (FieldKind.VALUE,), "functions": (), "parts": ()})
    elif sequence_rule:
        projection = projection.model_copy(update={"kinds": (FieldKind.SWEEP,), "functions": (), "parts": ()})
    else:
        projection = projection.model_copy(update={"kinds": tuple(kind for kind in projection.kinds if kind != FieldKind.SWEEP)})
    printed = site.model_copy(update={"names": (*site.names, *(
        ScopedValue(id=reference, name=stored_reference(reference), type=ValueType(base=ValueBase.REAL), level=Level.RUN)
        for reference in sorted(missing)
    ))})
    text = print_field(expression, printed, fragment=authored is not None) if expression is not None else ""
    marks = _field_problems(path, site, text, functions, problems, family=family)
    reports_missing = path[0] != "objective" or objective_use != ObjectiveUse.INHERITED
    if not marks and (missing and reports_missing or isinstance(expression, UnresolvedValue)):
        found = read_field(text, site, functions)
        if isinstance(found, tuple):
            marks = [problem.model_copy(update={"path": path}) for problem in found]
    return [StudioTrialField(
        path=field_path, label=authored.label if authored is not None else _field_label(path),
        site=sites.reference(projection), text=text,
        structure=value_structure(expression, site, functions) if shown else None,
        conversion=time_conversion(expression, period, trial_time_use(path[0], path[1:])),
        problems=marks,
    ), *fields]


def _condition_field(placement, site, functions, row, menu_fields, sites: FieldSiteIndex):
    path = "set_fields", placement.axis, "condition"
    condition = placement.condition
    problems = ()
    if condition is not None:
        parsed = resolve_trial_set_condition(condition, placement.sampling, site, functions)
        checked = parsed if isinstance(parsed, tuple) else check(parsed.value, site, functions, spans=parsed.spans)
        if isinstance(checked, tuple):
            problems = tuple(item.model_copy(update={"path": (*path, *item.path)}) for item in checked)
        elif row is not None:
            problems = tuple(
                Problem(code=ProblemCode.STRUCTURE, reason=item.detail, path=path)
                for item in row.problems if item.path == pointer(path)
            )
    (view,) = _trial_fields(path, site, condition, functions, lambda *_: None, row, problems, {}, None, menu_fields, sites)
    if view.text and view.problems:
        span = TextSpan(start=0, end=len(view.text))
        view = view.model_copy(update={"problems": [
            item if item.span else item.model_copy(update={"span": span})
            for item in view.problems
        ]})
    return view


def _batch_size(kind: str, payload) -> int | None:
    """How many trials the step runs at once: a Train's batch size, an Evaluate's sampled Trials; nothing for a value
    that is not one, which its own field reports."""
    if kind == "train":
        config = payload.get("training_config")
        count = (config if isinstance(config, dict) else {}).get(
            "batch_size", DocumentTrainingConfig.model_fields["batch_size"].default,
        )
    else:
        params = payload.get("params")
        rule = params.get("trial_set") if isinstance(params, dict) else None
        count = rule.get("count") if isinstance(rule, dict) and rule.get("kind") == "sample" else None
    return count if isinstance(count, int) and count >= 1 else None


def _objective_term_add(project, definition, step, objective, term):
    """Allocate a term key once for the shown Objective and every possible based destination."""
    if objective is None:
        raise ProjectFolderInvalid("This step has no Objective document.")
    used = set(objective.terms)
    slot = ("inputs.objective" if step == "evaluate" else
            "payload.objective" if step == "synthesize" else "payload.trial_documents.objective")
    if document := definition.document_ids.get(slot):
        for name in project.documents:
            if document in base_chain(project.documents, name):
                entry = compose_document(project.documents.get, name)
                if not isinstance(entry, ProjectDiagnostic):
                    used.update(entry.payload.get("terms", {}))
    key = generated_name("term", used.__contains__)
    value = stored_project_value(term, keep_explicit=True)
    op = (SetProjectValue(op="add", path=pointer(("terms", key)), value=value)
          if "terms" in objective.model_fields_set
          else SetProjectValue(op="add", path="/terms", value={key: value}))
    return key, StudioTrialDocumentEdit(kind="objective", edits=[op])


def trial_authoring(
    folders: ProjectFolderService, project_id: str, body: StudioTrialAuthoringRequest,
    bundle: ApplicationRegistryBundle,
) -> StudioTrialAuthoring:
    key = RunKey(body.run.variant, body.run.step, body.run.row)
    snapshot = folders.read(project_id, revision=body.revision)
    settled = folders.read_plan(project_id, revision=snapshot.revision, source_runs=(key,))
    plan = settled.plan
    step = plan.steps.get((key.variant, key.step))
    if step is None or step.kind not in ("train", "evaluate", "synthesize"):
        raise ProjectFolderInvalid("The selected run has no authored trial pairing.")
    try:
        project = snapshot.project
        definition = plan.selected_definitions.get(key)
        change = body.change
        if (
            definition is None and step.kind == "train"
            and isinstance(change, StudioTrialSelectDocument) and change.role == "training"
            and (body.scope is None or isinstance(body.scope, StudioTrialStepScope))
            and any(
                item.code == ResolutionCode.INVALID_SWEEP and item.step == key.step and item.row == key.row
                and item.variant == key.variant for item in plan.diagnostics
            )
        ):
            # The Training document can be replaced from the authored step even when a sweep row no longer resolves.
            base = resolve_step(plan.variant_projects[key.variant], key.step, readers=settled.readers)
            if not isinstance(base, ProjectDiagnostic):
                definition = base
        if definition is None:
            definition = _definition(plan, key)
        sweep_source = cache(partial(_sweep_source, project, plan, key, definition))
        documents, projected, graph = _pairing(plan, key, bundle, definition=definition)
        stored_task = projected.stored_task or _stored_task_view(
            plan, definition, documents.task_source, scopes=_set_spellings(projected.variables),
        )
        stored_objective = _stored_objective_view(plan, key, definition, documents.objective)
        stored, refused = StoredEdit([]), ()
        created_term = None
        store_choices = None
        if body.store_choice is not None and body.change is None:
            raise ProjectFolderInvalid("A store choice requires the edit it was offered for.")
        if isinstance(change, StudioTrialRelate):
            from feedbax.web.services.trial_relate import related_term

            if projected.pair is None or documents.objective is None:
                raise ProjectFolderInvalid("Relate needs a paired Model, Task, and Objective.")
            term = StoredObjectiveTerm.from_value(
                related_term(change, documents.model, projected.pair, bundle.components),
                label=None, scopes=_set_spellings(projected.variables),
            )
            created_term, change = _objective_term_add(project, definition, step.kind, stored_objective, term)
        elif isinstance(change, StudioTrialObjectiveTermAdd):
            term = StoredObjectiveTerm.from_value(
                change.term, label=change.term.label, scopes=_set_spellings(projected.variables),
            )
            created_term, change = _objective_term_add(project, definition, step.kind, stored_objective, term)
        if change is not None:
            stored, refused, preview = _prepare_edits(
                body, folders, project_id, snapshot.project, settled, key, bundle, definition, documents, projected,
                stored_task, stored_objective, change=change, sweep_source=sweep_source,
            )
            if isinstance(refused, list):
                store_choices, refused = refused, ()
            if preview is not None:
                project, plan = preview
                if changed := plan.selected_definitions.get(key):
                    definition = changed
                    documents, projected, graph = _pairing(plan, key, bundle)
                    stored_task = projected.stored_task or _stored_task_view(
                        plan, definition, documents.task_source, scopes=_set_spellings(projected.variables),
                    )
                    stored_objective = _stored_objective_view(plan, key, definition, documents.objective)
                else:
                    problems = tuple(
                        Problem(code=ProblemCode.STRUCTURE, reason=item.detail)
                        for item in plan.diagnostics
                        if item.step == key.step and item.row in (None, key.row)
                        and item.variant in (None, key.variant)
                    )
                    projected = replace(projected, problems=problems)
                sweep_source = cache(partial(_sweep_source, project, plan, key, definition))
    except (ValueError, TypeError, KeyError) as error:
        raise ProjectFolderInvalid(str(error)) from error
    paired = projected.pair
    compiled = paired.task if paired is not None else projected.partial_task
    task = documents.task_source
    own_prefix = "payload." if step.kind == "synthesize" else "payload.trial_documents."
    task_path = (
        _evaluation_task_path(definition) if step.kind == "evaluate"
        else "payload.task" if step.kind == "synthesize" else f"{own_prefix}{_task_field(task)}"
    )
    model_document = definition.document_ids.get(f"{own_prefix}model")
    objective_path = f"{own_prefix}objective"
    objective_document = definition.document_ids.get(objective_path)
    objective_source = StudioTrialObjectiveSource(kind="own", step=key.step) if documents.objective is not None else None
    if isinstance(documents, _InitialTrialDocuments):
        model_document = definition.document_ids.get("inputs.subject")
        subject = definition.input_documents["subject"]
        if subject.schema_id == LOADED_MODEL_SCHEMA_ID:
            model_document = model_document or LoadedModelDocument.model_validate(subject.payload).model.document
        objective_path = "inputs.objective"
        objective_document = definition.document_ids.get(objective_path)
    task_source = None
    task_field_sources = {}
    reset_model = documents.model
    if step.kind == "evaluate" and (upstream := evaluate_subject(definition)):
        producer = next((row for row in plan.rows if row.key == upstream[0]), None)
        if producer is not None:
            upstream_prefix = "payload." if producer.kind == "synthesize" else "payload.trial_documents."
            model_document = producer.document_ids.get(f"{upstream_prefix}model")
            objective_document = producer.document_ids.get(f"{upstream_prefix}objective")
        if model_document is not None:
            authored_model = compose_document(project.documents.get, model_document)
            if not isinstance(authored_model, ProjectDiagnostic):
                reset_model = StoredModelDocument.model_validate(authored_model.payload).lower(bundle.components)
        objective_source = StudioTrialObjectiveSource(
            kind="synthesize" if producer is not None and producer.kind == "synthesize" else "train",
            step=upstream[0].step,
        )
        task_source = evaluate_value_source(upstream[1])
        for target, origin_path in projected.task_origins.items():
            if target[0] != "task":
                continue
            origin = definition.sources.get(f"/payload{origin_path}/value") or definition.sources.get(
                f"/payload{origin_path}/op",
            )
            if origin is not None:
                source = evaluate_value_source(origin, evaluate_step=key.step)
                if target == ("task",):
                    task_source = source
                else:
                    task_field_sources[target[-1]] = source
    row = next((item for item in plan.rows if item.key == key), None)
    output_paths = {_output_address(port.id).to_text(): (port.name,)
                    for port in documents.model.ports.outputs} | projected.output_paths
    authored_signals = (
        {signal.id: signal for signal in task.signals} if isinstance(task, TaskDocument)
        else task.signals if isinstance(task, StoredTaskDocument) else {}
    )

    def expression(path):
        if (authored := projected.family_editor.fields.get(path)) is not None:
            return authored.expression
        document = _field_document(documents, path[0])
        if (path[:2] == ("task", "variables") and len(path) == 4 and path[3] in ("value", "count")
                and (entry := projected.set_variables.get(path[2])) is not None and entry.source.over is not None):
            value = getattr(entry.source.task_variables[entry.origin.variable], path[3])
            return lower_expression(value) if value is not None else None
        if path[0] == "objective" and stored_objective is not None:
            found = site_expression(stored_objective, tuple(map(str, path[1:])))
            return lower_expression(found) if found is not None else None
        if document is None and path[0] == "task" and isinstance(task, StoredTaskDocument):
            found = site_expression(task, tuple(map(str, path[1:])))
            return lower_expression(found) if found is not None else None
        if document is None:
            return None
        raw_path = tuple(map(str, path[1:]))
        if _epoch_end_field(path, "limit"):
            stored_limit = at(document, raw_path)
            if isinstance(stored_limit, TimeValue):
                return LiteralValue(value=stored_limit.value, unit=stored_limit.unit)
        found = site_expression(document, raw_path)
        if found is None and path[0] == "task" and path[1] == "signals" and path[3:] == ("absent",):
            signal = authored_signals[path[2]]
            return zero(signal.type.base, signal.type.shape)
        return found

    def sweep(path, site, label):
        if paired is None or _epoch_end_field(path, "limit"):
            return None
        return _field_sweep(project, key, path, site, label, sweep_source)

    clock = documents.model.step_seconds
    period = clock if isinstance(clock, float) else None
    fields = []
    sites = FieldSiteIndex()
    menu_fields: dict[str, TrialExpressionField] = {}
    objective_use = (
        ObjectiveUse.INHERITED if objective_source is not None and objective_source.kind in {"train", "synthesize"}
        else ObjectiveUse.TRAIN
    )
    for path, site in projected.fields.items():
        if path[:2] == ("task", "signals") and path[2] not in authored_signals:
            continue  # Internal gate and operand signals are not editable Task fields.
        fields.extend(_trial_fields(
            path, site, expression(path), bundle.expression_functions, sweep, row, projected.problems,
            output_paths, period, menu_fields, sites, objective_use=objective_use,
            authored=projected.family_editor.fields.get(path),
            family=pointer(path[:3]) in projected.family_editor.families,
        ))
    set_fields = trial_set_fields(project, key.step)
    if set_fields:
        site = _set_condition_site(definition, projected)
        fields.extend(
            _condition_field(placement, site, bundle.expression_functions, row, menu_fields, sites)
            for placement in set_fields
        )
    field_problems = [
        problem for field in fields for problem in field.problems
        if not _held_problems(problem.path, projected.problems)
    ]
    tasks, task_problems = _task_choices(folders, project, documents, bundle, graph, objective_use)
    resets = [
        StudioTrialReset(
            address=field.address, address_text=field.address.to_authored_text(),
            path=task_start_path(reset_model, field.address), type=field.type, finest=field.finest,
            required=field.required,
        )
        for field in projected.resets
    ]
    reset_keys = {reset.address_text for reset in resets}
    if (generated := projected.generated_set_members) and isinstance(stored_task, StoredTaskDocument):
        stored_task = stored_task.model_copy(update={
            "variables": {id: value for id, value in stored_task.variables.items() if id not in generated},
            "signals": {id: value for id, value in stored_task.signals.items() if id not in generated},
            "links": {address: id for address, id in stored_task.links.items() if id not in generated},
        })
    unresolved_links = [
        StudioTrialUnresolvedLink(key=key, path=task_start_path(reset_model, ModelAddress.from_authored_text(key)))
        for key in stored_task.links if key not in reset_keys
    ] if isinstance(stored_task, StoredTaskDocument) else []
    return StudioTrialAuthoring(
        revision=snapshot.revision, run=body.run,
        expression_context=folders.trial_expression_menus.remember(
            project_id, snapshot.revision, key, menu_fields,
        ),
        model=model_document,
        task_document=definition.document_ids.get(task_path) if task_path else None,
        objective_document=objective_document,
        task=stored_task, objective=stored_objective, objective_source=objective_source,
        objective_term_values={
            key: StudioObjectiveTermValues(enabled=term.enabled, reduction=term.reduction)
            for key, term in (stored_objective.terms.items() if stored_objective is not None else ())
        },
        families=projected.family_editor.families,
        task_changed=changed_from_base(project, _whole_document(definition, task_path) if task_path else None),
        objective_changed=changed_from_base(project, _whole_document(
            definition, objective_path,
        )) if objective_source is not None and objective_source.kind == "own" else [],
        task_source=task_source, task_field_sources=task_field_sources,
        set_fields=set_fields,
        task_types=[StudioTaskType(type=key, label=entry.label, parameter_schema=entry.parameters.model_json_schema())
                    for key in bundle.python_tasks.keys() if (entry := bundle.python_tasks.resolve(key))],
        variables=list(projected.variables), signals=list(projected.signals), epochs=list(projected.epochs),
        epoch_instances=list(projected.epoch_instances),
        period_seconds=period,
        n_steps=compiled.n_steps if compiled else None,
        batch_size=_batch_size(step.kind, definition.payload),
        reads_model=compiled.reads_model if compiled else False,
        tasks=tasks, task_problems=task_problems,
        input_bindings=documents.input_bindings, inputs=_input_rows(documents, projected),
        outputs=[StudioTrialOutput(
            address=ModelAddress.from_text(value.id),
            value=value.model_copy(update={"name": ".".join(output_paths.get(value.id, (value.name,)))}),
        ) for value in paired.model.outputs] if paired else [],
        resets=resets, unresolved_links=unresolved_links,
        fields=fields, **sites.tables(fields), problems=[
            *(_authored_trial_problem(problem) for problem in (*projected.problems, *field_problems)), *refused,
        ],
        created_term=created_term if stored.files or stored.created else None,
        files=stored.files, created=list(stored.created),
        store_choices=store_choices,
    )
