"""Join the acknowledged compiler projection to observed Jobs attempts."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import cache
from hashlib import sha256
from itertools import chain
from math import isfinite, prod
import json
from mimetypes import guess_type
from types import MappingProxyType
from typing import NotRequired, TypedDict
from urllib.parse import quote

import numpy as np
from pydantic import JsonValue
from pydantic_core import to_json

from feedbax.analysis.report_document import PLOTLY_SCRIPT_NAME, REPORT_HTML_NAME, report_content_security_policy
from feedbax.execution.trial_resolution import evaluation_conditions
from feedbax.contracts._request_codec import same_typed_value
from feedbax.contracts.artifact_custody import ContentObjectRef, ContentTreeRef
from feedbax.contracts.dataset import DATASET_MEDIA_TYPE, DatasetDescriptor
from feedbax.contracts.effective_success import (
    RequestObject, Success, read_effective_request, read_success_request, success_content_ref,
)
from feedbax.contracts.expression_values import Expression
from feedbax.contracts.figure_document import PLOTLY_FIGURE_ROLE, PlotlyFigurePort
from feedbax.contracts.model_address import FieldElement
from feedbax.contracts.objective_document import ObjectiveDocument
from feedbax.contracts.operation import DatasetPort
from feedbax.contracts.intent_diff import first_request_change
from feedbax.contracts.jobs import (
    StudioActivityJob,
    StudioBoundBestSubject,
    JobStatus,
    StudioBestRankGroup,
    StudioPlanRequest,
    StudioProjectWindowMap,
    StudioProjectWindowMapAxis,
    StudioProjectWindowMapCoordinate,
    StudioProjectWindowMapInputAxis,
    StudioProjectWindowMapEdge,
    StudioProjectWindowMapElement,
    StudioProjectWindowMapRequest,
    StudioProjectWindowMapException,
    StudioProjectWindowMapFigure,
    StudioProjectWindowMapInput,
    StudioDefinitionChange,
    StudioProjectWindowMapProblem,
    StudioProjectWindowMapRow,
    StudioProjectWindowMapStep,
    StudioProjectWindowMapSubset,
    StudioEvaluationCheckpointSeries,
    StudioEvaluationObjectiveSeries,
    StudioSavedEvaluationRead,
    StudioSavedEvaluationReplayRequest,
    StudioSavedEvaluationResult,
    StudioSavedEvaluationRow,
    StudioSavedLightHistory,
    StudioSavedLightResult,
    StudioSavedTrainAttempt,
    StudioSavedTrainChange,
    StudioSavedTrainKey,
    StudioSavedTrainResult,
    StudioSavedTrainRow,
    StudioSavedTrainRuns,
    StudioTrialName,
)
from feedbax.contracts.project_inputs import (
    BatchCheckpoint, BoundRank, InputBinding, SelectedAxisValue, StepReference, SubsetTerms,
    producer_member_key, subset_to_rows,
)
from feedbax.contracts.studio_analyze import StudioOperationDatasetPage
from feedbax.contracts.studio_place import (
    StudioStepPlace, definition_place, document_path, path_parts, variation_targets,
)
from feedbax.contracts.studio_project import AuthoredFileAddress
from feedbax.contracts.studio_subset import (
    StudioRunRuleCheck, StudioRunRuleChosen, StudioRunRuleRefused, StudioRunRuleRequest, StudioSubsetToRowsEdit,
    StudioSubsetToRowsRequest,
)
from feedbax.contracts.project_io import project_declaration_name
from feedbax.planner.project_plan import (
    BlockedRow, ElementKey, PlannedAnalyzeRun, PlannedElement, PlannedInput, PlannedRow, PlannedRun, ProjectPlan, RunKey,
    RunProblem,
    authored_request_change, common_train_metrics, planned_input_groups,
)
from feedbax.planner.project_plan_io import run_key
from feedbax.planner.trial_names import TrialNameParts, numbered_trial_parts
from feedbax.contracts.project_resolution import ProjectDiagnostic
from feedbax.contracts.project_sources import pointer
from feedbax.contracts.result_address import parse_labels
from feedbax.contracts.project_sweep import (
    CheckpointRuleAxis, SetAxis, authored_sweep_axis_name, axis_coordinates, model_field_target, sweep_value_name,
)
from feedbax.contracts.set_document import SET_DOCUMENT_SCHEMA_ID
from feedbax.contracts.run_rule import (
    METRIC_CONDITION_FALSE, RuleRefusal, bind_metric_conditions, choose, metric_condition_holds,
)
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_refs import (
    BatchMember, BatchSetMember, BestRank, ConditionalProduct, LastMember, ProductRef, RequestId, UpdateSeriesMember,
    is_synthesized_model_product,
)
from feedbax.contracts.request_specs import (
    DocumentTrainingRequestSpec, EvaluationRequestSpec, RenderRequestSpec, SynthesisRequestSpec,
)
from feedbax.contracts.trial_clock import StudioSavedFigureMeanRequest
from feedbax.contracts.trial_record import StudioTrialReplayResponse
from feedbax.dataset import Dataset
from feedbax.persistence.completion import completion_file
from feedbax.persistence.dataset import open_stored_dataset
from feedbax.host.plan_history import PlanRowRef, PlanSuccess, plan_run_key, settled_results
from feedbax.host.job_completion import training_binding_path
from feedbax.host.light_jobs import dataset_member
from feedbax.jobs import (
    EVALUATION_SUBJECT_REQUIRED,
    JobConflict,
    JobNotFound,
    JobService,
    PlanJobAttempt,
    PlanJobAttempts,
    PlanJobReader,
    active_plan_attempt,
    plan_attempt_key,
    saved_request_status,
)
from feedbax.orchestration.session_jobs import ApprovedRowStatus
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import SuccessView
from feedbax.persistence.store_damage import inspect_without_recording
from feedbax.persistence.train_metrics import read_final_train_metrics
from feedbax.persistence.canonical_evaluate import EvaluationResult, decode_rollout_request
from feedbax.host.live_evaluate import live_objective_series
from feedbax.contracts.rollout import CannotRun
from feedbax.persistence.canonical_train import BoundStudioTrainingJobInput
from feedbax.project_model_selection import trial_document_slots
from feedbax.tasks.registry import PythonTaskRegistry
from feedbax.web.services.axis_values import gather_axes
from feedbax.web.services.project_folder import ProjectFolderInvalid, ProjectFolderService, SettledProjectPlan


@dataclass(frozen=True)
class TrainRowHistory:
    status: JobStatus
    reason: str | None
    result: StudioSavedTrainResult | None
    previous: StudioSavedTrainResult | None
    change: StudioSavedTrainChange | None


@dataclass(frozen=True)
class PlanStatusRow:
    key: RunKey | ElementKey
    label: str
    status: JobStatus
    reason: str | None
    requests: tuple[PlanStatusRow, ...] = ()
    """The requests of an element that maps over labels, whose summary this row is."""

    def flattened(self) -> Iterator[PlanStatusRow]:
        yield self
        yield from self.requests


@dataclass(frozen=True)
class LightHistory:
    status: JobStatus
    reason: str | None
    result: StudioSavedLightResult | None
    previous: StudioSavedLightResult | None
    changed: StudioDefinitionChange | None = None


@dataclass(frozen=True)
class SavedLightProduct:
    content: bytes
    media_type: str
    content_security_policy: str | None = None


@dataclass(frozen=True)
class ProjectPlanRead:
    settled: SettledProjectPlan
    jobs: PlanJobAttempts
    attempts_by_run: Mapping[RunKey, tuple[PlanJobAttempt, ...]]
    history: Mapping[RunKey, TrainRowHistory]
    evaluations: Mapping[RunKey, StudioSavedEvaluationRow]
    evaluation_problems: tuple[StudioProjectWindowMapProblem, ...]
    light: Mapping[RunKey | ElementKey, LightHistory]
    openings: Mapping[RunKey, ApprovedRowStatus] = field(default_factory=dict)

    def status_rows(self) -> Iterator[PlanStatusRow]:
        """Project saved attempt states and the compiler's refused rows and elements."""
        plan = self.settled.plan
        problems: dict[RunKey, list[ProjectDiagnostic]] = defaultdict(list)
        for problem in plan.diagnostics:
            if problem.variant is not None and problem.step is not None and problem.row is not None:
                problems[RunKey(problem.variant, problem.step, problem.row)].append(problem)

        for row in plan.rows:
            label = row.label or row.key.row
            observed = self.history.get(row.key) or self.evaluations.get(row.key) or self.light.get(row.key)
            if observed is not None:
                yield PlanStatusRow(row.key, label, observed.status, observed.reason)
                continue

            row_problems = []
            element_problems: dict[str, list[ProjectDiagnostic]] = defaultdict(list)
            for problem in problems[row.key]:
                if problem.element is None:
                    row_problems.append(problem)
                else:
                    element_problems[problem.element].append(problem)
            if isinstance(row, PlannedAnalyzeRun):
                states = []
                for name in dict.fromkeys((*(key.name for key in row.elements), *element_problems)):
                    requests = [
                        PlanStatusRow(element.key, label, observed.status, observed.reason)
                        if (observed := self.light.get(element.key)) is not None
                        else _planned_status(element.key, label, element_problems[name])
                        for element in row.requests(name)
                    ]
                    key = ElementKey(row.key, name)
                    if not requests:
                        states.append(_planned_status(key, label, element_problems[name]))
                    elif len(requests) == 1 and requests[0].key == key:
                        states.append(requests[0])
                    else:
                        states.append(replace(
                            _analysis_status(key, label, element_problems[name], requests), requests=tuple(requests),
                        ))
                yield _analysis_status(row.key, label, problems[row.key], states)
                yield from states
            elif isinstance(row, PlannedRun):
                yield _planned_status(row.key, label, [*row_problems, *row.problems])
            else:
                reason = None if row_problems or row.problems else "No request could be planned."
                yield PlanStatusRow(row.key, label, JobStatus.CANNOT_RUN, reason)

    @classmethod
    def without_history(cls, settled: SettledProjectPlan) -> ProjectPlanRead:
        jobs = PlanJobAttempts(0, {})
        return cls(settled, jobs, {}, _train_history(settled, jobs), {}, (), {})

    def response(
        self, *, include_window_map: bool, variants: tuple[str, ...] | None = None,
        evaluation_tasks: PythonTaskRegistry | None = None,
    ) -> StudioSavedTrainRuns:
        result = StudioSavedTrainRuns(
            revision=self.settled.revision,
            rows=[
                StudioSavedTrainRow(
                    key=StudioSavedTrainKey.model_validate(run.key, from_attributes=True),
                    name=run.name,
                    label=run.label,
                    request_id=run.request_id,
                    attempts=[
                        StudioSavedTrainAttempt.model_validate(attempt, from_attributes=True)
                        for attempt in self.attempts_by_run.get(run.key, ())
                    ],
                    status=self.history[run.key].status,
                    reason=self.history[run.key].reason,
                    result=self.history[run.key].result,
                    previous=self.history[run.key].previous,
                    change=self.history[run.key].change,
                    previous_attempts=[
                        StudioSavedTrainAttempt.model_validate(attempt, from_attributes=True)
                        for attempt in self.previous_train_attempts(run)
                    ],
                )
                for run in self.settled.plan.runs
                if run.kind == "train" and (variants is None or run.key.variant in variants)
            ],
        )
        if include_window_map:
            result.window_map = self.window_map(variants=variants, evaluation_tasks=evaluation_tasks)
        return result

    def previous_train_attempts(self, run: PlannedRun) -> tuple[PlanJobAttempt, ...]:
        previous = self.history[run.key].previous
        if previous is None or previous.request_id == run.request_id:
            return ()
        attempts = self.jobs.by_request[previous.request_id]
        by_id = {attempt.job_id: attempt for attempt in attempts}
        lineage = []
        job_id: str | None = previous.job_id
        while job_id is not None:
            attempt = by_id.pop(job_id, None)
            if attempt is None:
                break
            lineage.append(attempt)
            job_id = attempt.continues
        return tuple(reversed(lineage))

    def window_map(
        self, *, variants: tuple[str, ...] | None = None,
        evaluation_tasks: PythonTaskRegistry | None = None,
    ) -> StudioProjectWindowMap:
        """Project authored steps and compiler facts from the acknowledged Plan."""
        plan = self.settled.plan
        document_ids: dict[tuple[str, str], dict[str, str]] = {}
        document_uses: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in plan.rows:
            if not row.resolved:
                continue
            key = row.key.variant, row.key.step
            document_uses[key].update(row.document_uses - row.followed_documents)
            shared = document_ids.get(key)
            document_ids[key] = (
                dict(row.document_ids)
                if shared is None
                else {path: name for path, name in shared.items() if row.document_ids.get(path) == name}
            )
        statuses = {item.key: item for row in self.status_rows() for item in row.flattened()}
        elements = plan.elements.values()
        conditions: dict[str, list[StudioTrialName] | None] = {}

        def task_conditions(row: PlannedRow) -> list[StudioTrialName] | None:
            if evaluation_tasks is None:
                return None
            if row.trial_names is not None:
                return _named_trials(tuple(row.trial_names), row.trial_names)
            task = row.train.task if row.train is not None else None
            if isinstance(row, PlannedRun) and row.kind == "synthesize":
                spec = decode_request(plan.requests[row.request_id], plan.documents.__getitem__).spec
                if isinstance(spec, SynthesisRequestSpec):
                    task = spec.trial_documents.task_source
            if task is None:
                return None
            identity = to_json(task.model_dump(mode="json")).decode()
            if identity not in conditions:
                try:
                    ids = evaluation_conditions(task, evaluation_tasks)
                    conditions[identity] = _named_trials(ids, numbered_trial_parts(ids))
                except ValueError:
                    # An unresolved Task has no authorable trial domain.
                    conditions[identity] = None
            return conditions[identity]

        counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        for key, status in statuses.items():
            if isinstance(key, RunKey) and status.status is not JobStatus.NOT_CHOSEN:
                counts[key.variant, key.step][status.status.value] += 1

        edge_keys = [
            (producer, run.key) for run in plan.runs for binding in run.inputs.values()
            for producer in binding.sources
        ]
        edge_keys.extend(
            (producer, row.key)
            for row in plan.rows if isinstance(row, (PlannedAnalyzeRun, BlockedRow))
            for members in row.inputs.values()
            for binding in members
            for producer in binding.sources
        )

        check_metric = _final_metric_check(self.settled, self.history)

        def inputs(variant: str, step_id: str) -> Iterator[StudioProjectWindowMapInput]:
            step = plan.steps[variant, step_id]
            for role, binding in (step.inputs if step.inputs is not None else step.authored_inputs).items():
                for input_id, reference in _step_references(role, binding):
                    subset = plan.subsets.get((variant, step_id, input_id))
                    members = subset.members if subset is not None else ()
                    if subset is not None and (conditions := subset.metric_conditions) is not None:
                        members = _settled_metric_members(members, lambda key: (
                            check_metric(key, condition.request_id, condition.conditions)
                            if (condition := conditions.get(key)) is not None else None
                        ))
                    axis = next((
                        (name, item) for name, item in step.sweep.axes.items()
                        if isinstance(item, CheckpointRuleAxis)
                    ), None) if step.kind == "evaluate" and role == "subject" and step.sweep else None
                    yield StudioProjectWindowMapInput(
                        id=input_id,
                        source_step_id=reference.step,
                        checkpoint=(
                            axis[1].checkpoint_rule if axis is not None
                            else reference.checkpoint
                        ),
                        checkpoint_axis=axis[0] if axis is not None else None,
                        can_sweep_checkpoint=input_id == "subject" and step.can_sweep_checkpoint,
                        metric_names=sorted(common_train_metrics(
                            row for source_variant in (
                                plan.variant_projects if reference.variants == "all"
                                else reference.variants or (variant,)
                            ) for row in step_rows.get((source_variant, reference.step), ())
                        )),
                        subset=StudioProjectWindowMapSubset(
                            terms=subset.terms, count=len(subset.ranks or members), total=subset.candidates,
                            ranks=list(subset.ranks),
                            rank_groups=[
                                StudioBestRankGroup(coordinates=dict(group.coordinates), ranks=list(group.ranks))
                                for group in subset.rank_groups if group.coordinates
                            ],
                            document=subset.document,
                            members=[
                                StudioSavedTrainKey.model_validate(member, from_attributes=True) for member in members
                            ] if evaluation_tasks is not None and step.kind == "evaluate" and role == "subject" else None,
                        ) if subset is not None else StudioProjectWindowMapSubset(
                            terms=reference.where, count=0,
                            total=len(step_rows.get((variant, reference.step), ())), document=None,
                        ) if isinstance(reference.where, SubsetTerms) else None,
                    )

        def exceptions(variant: str, step_id: str) -> Iterator[StudioProjectWindowMapException]:
            at = variant, step_id
            step = plan.steps[at]
            sweep = step.sweep
            if sweep is None and (project := plan.variant_projects.get(variant)) is not None:
                sweep = project.steps[step_id].sweep
            if sweep is not None:
                matches = plan.exception_matches.get(at, {})
                for name, rule in sweep.exceptions.items():
                    if rule is None:
                        continue
                    field_labels = {}
                    for path, patch in rule.patches.items():
                        if patch is None:
                            continue
                        if label := step.exception_field_labels.get(path):
                            field_labels[path] = label
                        elif (target := model_field_target(path)) is not None and isinstance(
                            target.address.element, FieldElement,
                        ):
                            field = str(target.address.element.path[-1]).replace("_", " ")
                            field_labels[path] = field[:1].upper() + field[1:]
                    yield StudioProjectWindowMapException(
                        id=name, count=len(matches.get(name, ())), rule=rule, field_labels=field_labels,
                    )

        def includes(variant: str | None) -> bool:
            return variants is None or variant is None or variant in variants

        rule_metrics: dict[tuple[str, str], set[str]] = defaultdict(set)
        step_rows: dict[tuple[str, str], list[PlannedRow]] = defaultdict(list)
        for row in plan.rows:
            at = row.key.variant, row.key.step
            step_rows[at].append(row)
            for attempt in self.attempts_by_run.get(row.key, ()):
                rule_metrics[at].update(attempt.metric_names)
        gathered = {at: gather_axes(rows) for at, rows in step_rows.items()}
        axis_values = {
            at: value for _, values in gathered.values() for at, value in values.items()
        }

        def axes(variant: str, step_id: str) -> list[StudioProjectWindowMapAxis]:
            at = variant, step_id
            projected = {axis.id: axis for axis in gathered.get(at, ([], {}))[0]}
            step = plan.steps[at]
            sweep = step.sweep
            project = plan.variant_projects.get(variant)
            if sweep is None and project is not None:
                sweep = project.steps[step_id].sweep
            if sweep is not None:
                for number, (key, axis) in enumerate(sweep.axes.items(), 1):
                    if axis is None:
                        continue
                    if key not in projected:
                        title = step.unexpanded_axis_titles.get(key)
                        if isinstance(axis, SetAxis) and project is not None:
                            document = project.documents.get(axis.set)
                            if document is not None and document.schema_id == SET_DOCUMENT_SCHEMA_ID:
                                title = document.label
                        projected[key] = StudioProjectWindowMapAxis(
                            id=key, name=axis.label or title or authored_sweep_axis_name(axis, number), values=[],
                        )
                    elif axis.label is not None:
                        projected[key] = projected[key].model_copy(update={"name": axis.label})
            return list(projected.values())

        def figure_reference(key: RunKey, element: str) -> StepReference | None:
            candidates = [
                row.selection_coordinates
                for row in step_rows[key.variant, key.step]
            ]
            row = plan.row(key)
            assert row is not None
            axes = {}
            for axis, value in row.selection_coordinates.items():
                others = [
                    values for values in candidates if axis in values and values[axis].value_id == value.value_id
                ]
                if len(others) != len(candidates):
                    axes[axis] = [SelectedAxisValue(id=value.value_id, label=sweep_value_name(value))]
                    candidates = others
            if len(candidates) != 1:
                return None
            return StepReference(
                step=key.step, **({"where": SubsetTerms(clauses=[axes])} if axes else {}),
                element=element, product=PLOTLY_FIGURE_ROLE,
            )

        problems = [
            *(StudioProjectWindowMapProblem.from_diagnostic(
                problem, declaration=project_declaration_name(self.settled.configuration.project),
            ) for problem in plan.diagnostics if includes(problem.variant)),
            *row_problems(plan, variants),
            *(problem for problem in self.evaluation_problems if includes(problem.variant)),
        ]
        return StudioProjectWindowMap(
            variants=[variant for variant in plan.variants if includes(variant)],
            steps=[
                StudioProjectWindowMapStep(
                    variant=variant, step_id=step_id, label=step.label,
                    kind=step.kind, group=step.group,
                    inputs=list(inputs(variant, step_id)),
                    document_ids=document_ids.get((variant, step_id), {}),
                    document_uses=sorted(document_uses[variant, step_id]),
                    exceptions=list(exceptions(variant, step_id)),
                    status_counts=dict(counts[variant, step_id]),
                    axes=axes(variant, step_id),
                    rule_metrics=sorted(rule_metrics[variant, step_id]),
                )
                for (variant, step_id), step in plan.steps.items() if includes(variant)
            ],
            rows=[
                StudioProjectWindowMapRow(
                    key=StudioSavedTrainKey.model_validate(row.key, from_attributes=True),
                    name=row.name,
                    label=row.label,
                    status=statuses[row.key].status,
                    reason=statuses[row.key].reason,
                    opening_step=opening.opening_step if (opening := self.openings.get(row.key)) is not None else None,
                    collection_progress=next((
                        attempt.collection_progress for attempt in reversed(self.attempts_by_run.get(row.key, ()))
                        if attempt.status == JobStatus.COLLECTING
                    ), None),
                    light=StudioSavedLightHistory(result=light.result, previous=light.previous, changed=light.changed)
                    if (light := self.light.get(row.key)) is not None else None,
                    evaluation_conditions=task_conditions(row),
                    origin=row.train.origin if row.train is not None else None,
                    target_total=row.train.target_total if row.train is not None else None,
                    resolved=row.resolved,
                    member_key=producer_member_key(row.key.variant, row.key.row, gathers_variants=False)
                    if row.kind in {"train", "evaluate", "synthesize"} else None,
                    coordinates={
                        axis: StudioProjectWindowMapCoordinate(
                            value_id=coordinate.value_id, label=sweep_value_name(coordinate), value=coordinate.value,
                            axis_value=axis_values[row.key, axis],
                        )
                        for axis, coordinate in axis_coordinates(row.coordinates, {}, row.kept_axes).items()
                    },
                    input_axes={
                        axis: StudioProjectWindowMapInputAxis(
                            value_id=member.value_id, label=member.label, carries=list(member.carries),
                            axis_value=axis_values[row.key, axis],
                        )
                        for axis, member in row.input_axes.items()
                    },
                )
                for row in plan.rows if includes(row.key.variant)
            ],
            edges=[
                StudioProjectWindowMapEdge(
                    from_key=StudioSavedTrainKey.model_validate(producer, from_attributes=True),
                    to_key=StudioSavedTrainKey.model_validate(consumer, from_attributes=True),
                )
                for producer, consumer in edge_keys if includes(consumer.variant)
            ],
            problems=problems,
            requests=[
                StudioPlanRequest(**plan_request_address(request.key), request_id=request.request_id)
                for request in chain(plan.runs, elements)
                if includes(run_key(request.key).variant)
            ],
            evaluations=[row for row in self.evaluations.values() if includes(row.key.variant)],
            analysis_elements=[
                StudioProjectWindowMapElement(
                    key=StudioSavedTrainKey.model_validate(run, from_attributes=True),
                    name=name, kind=requests[0].kind,
                    status=statuses[ElementKey(run, name)].status, reason=statuses[ElementKey(run, name)].reason,
                    figure=figure_reference(run, name)
                    if isinstance(requests[0].outputs.get(PLOTLY_FIGURE_ROLE), PlotlyFigurePort) else None,
                    requests=[
                        StudioProjectWindowMapRequest(
                            labels=dict(parse_labels(element.key.member or "")),
                            request_id=element.request_id,
                            status=statuses[element.key].status, reason=statuses[element.key].reason,
                            **({"result": light.result, "previous": light.previous, "changed": light.changed}
                               if (light := self.light.get(element.key)) is not None else {}),
                        )
                        for element in requests
                    ],
                )
                for (run, name), requests in _by_element(elements).items()
                if includes(run.variant)
            ],
            analysis_figures=[
                StudioProjectWindowMapFigure(
                    key=StudioSavedTrainKey.model_validate(row.key, from_attributes=True),
                    name=name, data=list(figure.data), caption=figure.caption,
                )
                for row in plan.rows if includes(row.key.variant) and isinstance(row, PlannedAnalyzeRun)
                for name, figure in row.figures.items()
            ],
            regroup_refused=sorted(self.settled.regroup_refused),
        )

def row_problems(plan: ProjectPlan, variants: tuple[str, ...] | None = None) -> Iterator[StudioProjectWindowMapProblem]:
    """Each Problem that keeps a row of ``plan`` from running, once, at its own path."""
    for row in plan.rows:
        if variants is not None and row.key.variant not in variants:
            continue
        for problem in row.problems:
            parts = path_parts(problem.path)
            place = definition_place(
                parts, row.document_ids.get, member=problem.model_address, variable=problem.variable, start=problem.start,
                variation=lambda _name, target=problem.variation_target: target,
            )
            source = problem.sweep_source
            yield StudioProjectWindowMapProblem(
                code="cannot_run", detail=problem.detail,
                path=".".join(path_parts(source.path)[:5]) if source is not None else problem.path,
                variant=row.key.variant, step=row.key.step, row=row.key.row,
                place=place, document_path=document_path(parts, place),
                address=f"steps/{source.step if source is not None else row.key.step}.json",
                authored_path=source.path if source is not None else None,
            )


def _by_element(elements: Iterable[PlannedElement]) -> dict[tuple[RunKey, str], list[PlannedElement]]:
    grouped: dict[tuple[RunKey, str], list[PlannedElement]] = {}
    for element in elements:
        grouped.setdefault((element.key.run, element.key.name), []).append(element)
    return grouped


class PlanRequestAddressFields(TypedDict):
    key: StudioSavedTrainKey
    element: NotRequired[str]
    member: NotRequired[str]


def plan_request_address(key: RunKey | ElementKey) -> PlanRequestAddressFields:
    address: PlanRequestAddressFields = {
        "key": StudioSavedTrainKey.model_validate(run_key(key), from_attributes=True),
    }
    if isinstance(key, ElementKey):
        address["element"] = key.name
        if key.member is not None:
            address["member"] = key.member
    return address


def _step_references(
    role: str, binding: InputBinding | None,
) -> Iterator[tuple[str, StepReference]]:
    if isinstance(binding, StepReference):
        yield role, binding
    elif isinstance(binding, list):
        for index, reference in enumerate(binding):
            yield f"{role}.{index}", reference


def _planned_status(
    key: RunKey | ElementKey, label: str, problems: Sequence[ProjectDiagnostic | RunProblem],
) -> PlanStatusRow:
    status = JobStatus.CANNOT_RUN if problems else JobStatus.NOT_RUN
    return PlanStatusRow(key, label, status, None)


def _analysis_status(
    key: RunKey | ElementKey, label: str, problems: list[ProjectDiagnostic], elements: list[PlanStatusRow],
) -> PlanStatusRow:
    """The summary status of a card's elements, or of a mapped element's requests, as a swept step's runs have."""
    if problems or not elements:
        return _planned_status(key, label, problems)
    for status in (
        JobStatus.CORRUPT, JobStatus.CANNOT_RUN, JobStatus.FAILED, JobStatus.BLOCKED,
        JobStatus.STOPPING, JobStatus.RUNNING, JobStatus.STARTING, JobStatus.QUEUED,
        JobStatus.INTERRUPTED, JobStatus.STOPPED, JobStatus.STALE, JobStatus.WAITING, JobStatus.NOT_RUN,
    ):
        if (child := next((row for row in elements if row.status == status), None)) is not None:
            return PlanStatusRow(key, label, status, child.reason)
    return PlanStatusRow(key, label, JobStatus.SUCCEEDED, None)


def _figure_task_labels(
    view: ProjectPlanRead, store: ImmutableArtifactBlobProvider, receipt: Success, spec: RenderRequestSpec, content: bytes,
) -> bytes:
    from feedbax.web.services.figure_labels import project_figure_task_labels

    evaluations: dict[tuple[str, str], list[DatasetPort]] = defaultdict(list)
    for run in view.settled.plan.runs:
        row = view.evaluations.get(run.key)
        result = row.result if row is not None else None
        if (
            result is not None and result.request_id is not None
            and isinstance(run, PlannedRun) and run.dataset_port is not None
        ):
            evaluations[result.job_id, result.request_id].append(run.dataset_port)
    return project_figure_task_labels(content, spec.document, receipt.inputs, store, evaluations)


class ProjectPlanService:
    def __init__(
        self, folders: ProjectFolderService, jobs: JobService,
        remote_attempts: Callable[[SettledProjectPlan], Sequence[StudioActivityJob]] | None = None,
        remote_openings: Callable[[SettledProjectPlan], Mapping[RunKey, ApprovedRowStatus]] | None = None,
    ) -> None:
        self.folders = folders
        self.jobs = jobs
        self._remote_attempts = remote_attempts
        self._remote_openings = remote_openings

    def read(
        self, project_id: str, *, revision: str, source_runs: Iterable[RunKey] = (),
        stopped_checkpoint_jobs: Sequence[str] = (), maintain_validity: bool = False,
    ) -> ProjectPlanRead:
        """Read one saved revision and its request-scoped attempt evidence."""
        settled = self.folders.read_plan(project_id, revision=revision, source_runs=source_runs)
        remote = self._remote_attempts(settled) if self._remote_attempts is not None else ()
        openings = self._remote_openings(settled) if self._remote_openings is not None else None
        view = read_settled_project_plan(
            settled, self.jobs, stopped_checkpoint_jobs=stopped_checkpoint_jobs, remote_attempts=remote, openings=openings,
        )
        if maintain_validity and self.jobs.maintain_plan_validity(self.folders, settled, view.jobs):
            view = read_settled_project_plan(
                settled, self.jobs, stopped_checkpoint_jobs=stopped_checkpoint_jobs,
                remote_attempts=remote, openings=openings,
            )
        return view

    def response(self, view: ProjectPlanRead, *, include_window_map: bool) -> StudioSavedTrainRuns:
        evaluation_tasks = self.jobs.registries.python_tasks
        return view.response(include_window_map=include_window_map, evaluation_tasks=evaluation_tasks)

    def _saved_light_receipt(
        self, project: str, revision: str, job_id: str, role: str,
    ) -> tuple[ProjectPlanRead, ImmutableArtifactBlobProvider, Success]:
        view = self.read(project, revision=revision)
        visible = next((result for history in view.light.values() for result in (history.result, history.previous)
                        if result is not None and result.job_id == job_id and role in result.products), None)
        if visible is None:
            raise JobNotFound(job_id)
        store = ImmutableArtifactBlobProvider(view.settled.store_path)
        success = SuccessView(store)
        ref = success.attempt_ref(job_id)
        if ref is None or ref.sha256 != visible.receipt.manifest_sha256:
            raise JobConflict("The saved result receipt could not be authenticated.")
        try:
            return view, store, success.read_receipt(ref)
        except (OSError, TypeError, ValueError, KeyError) as error:
            raise JobConflict("The saved result receipt could not be authenticated.") from error

    def light_product(
        self, project: str, revision: str, job_id: str, role: str, member: str,
    ) -> SavedLightProduct:
        """Read a product of a current or previous result visible on the saved Plan."""
        view, store, receipt = self._saved_light_receipt(project, revision, job_id, role)
        try:
            product, = receipt.products[role]
            content = success_content_ref(product.content)
            if isinstance(content, ContentTreeRef):
                files = {entry.relative_path: entry.object for entry in store.get_tree(content).files}
                selected = files.get(member)
                if selected is None:
                    raise JobNotFound(job_id)
                media_type = guess_type(member)[0] or "application/octet-stream"
                policy = None
                if member == REPORT_HTML_NAME:
                    script = files.get(PLOTLY_SCRIPT_NAME)
                    if script is not None:
                        store.verify_object(script)
                    policy = report_content_security_policy(script.sha256 if script is not None else None)
                return SavedLightProduct(store.get_object(selected), media_type, policy)
            if role != PLOTLY_FIGURE_ROLE or member != "figure.json":
                raise JobNotFound(job_id)
            request = read_effective_request(receipt, store.get_object)
            if not isinstance(request.spec, RenderRequestSpec):
                raise ValueError("Expected a saved figure request.")
            labelled = _figure_task_labels(view, store, receipt, request.spec, store.get_object(content))
            return SavedLightProduct(labelled, "application/json")
        except (OSError, TypeError, ValueError, KeyError) as error:
            raise JobConflict("The saved product could not be authenticated.") from error


    @contextmanager
    def _operation_dataset(self, project: str, revision: str, job_id: str, role: str) -> Iterator[Dataset]:
        _, store, receipt = self._saved_light_receipt(project, revision, job_id, role)
        if not receipt.request.id.startswith(("analyze:", "synthesize:")):
            raise JobNotFound(job_id)
        try:
            product, = receipt.products[role]
            with open_stored_dataset(store, dataset_member(product)) as dataset:
                yield dataset
        except (OSError, TypeError, ValueError, KeyError) as error:
            raise JobConflict("The saved operation product could not be authenticated.") from error

    def operation_dataset(self, project: str, revision: str, job_id: str, role: str) -> DatasetDescriptor:
        """Read the labelled descriptor of a visible Analyze product."""
        with self._operation_dataset(project, revision, job_id, role) as dataset:
            return dataset.descriptor

    def operation_dataset_page(
        self, project: str, revision: str, job_id: str, role: str, *, field: str, offset: int, limit: int,
    ) -> StudioOperationDatasetPage:
        with self._operation_dataset(project, revision, job_id, role) as dataset:
            descriptor = dataset.descriptor
            if field not in descriptor.variables and field not in descriptor.coords:
                raise JobNotFound(field)
            shape = descriptor.shape(field)
            total = prod(shape)
            stop = min(total, offset + limit)
            inline = descriptor.field(field).inline_values
            if offset >= stop:
                values = []
            elif inline is not None:
                values = list(inline[offset:stop])
            else:
                values = _operation_values(dataset, field, shape, offset, stop)
            return StudioOperationDatasetPage(field=field, offset=offset, total=total, values=values)

    def figure_mean(self, project: str, body: StudioSavedFigureMeanRequest) -> SavedLightProduct:
        from feedbax.host.light_worker import render_saved_figure

        view, store, receipt = self._saved_light_receipt(project, body.revision, body.job_id, PLOTLY_FIGURE_ROLE)
        try:
            request = read_effective_request(receipt, store.get_object)
            content = render_saved_figure(store, request, receipt.inputs, mean=body.mean)
            assert isinstance(request.spec, RenderRequestSpec)
            return SavedLightProduct(_figure_task_labels(view, store, receipt, request.spec, content), "application/json")
        except CannotRun as error:
            raise ProjectFolderInvalid(str(error)) from error
        except (OSError, TypeError, ValueError, KeyError) as error:
            raise JobConflict(f"The saved figure could not be drawn on this clock. {error}") from error

    def check_run_rule(self, project_id: str, body: StudioRunRuleRequest) -> StudioRunRuleCheck:
        """What ``body.rule`` chooses among the producer's planned rows, as the plan would choose them."""
        settled = self.folders.read_plan(project_id, revision=body.revision)
        plan = settled.plan
        step = plan.steps.get((body.variant, body.step))
        if step is None:
            raise ProjectFolderInvalid(f"{body.step} is not a step of {body.variant}.")
        rows = [row for row in plan.rows if (row.key.variant, row.key.step) == (body.variant, body.step)]
        coordinates = {id(row): row.selection_coordinates for row in rows}
        metrics = sorted(common_train_metrics(rows)) if body.ranked else []
        try:
            chosen = choose(
                body.rule, rows, coordinates=lambda row: coordinates[id(row)], producer=step.label,
                ranked=body.ranked, final_metrics=common_train_metrics if step.kind == "train" else None,
            )
        except RuleRefusal as refusal:
            return StudioRunRuleCheck(
                total=len(rows), metric_names=metrics, result=StudioRunRuleRefused(reason=refusal.detail, span=refusal.span),
            )
        best = chosen.best
        groups = {tuple(coordinates[id(row)][axis].value_id for axis in best.per) for row in chosen.rows} if best else ()
        members = chosen.rows
        if chosen.conditions:
            attempts = self.jobs.read_plan_attempts(
                project=settled.project_id, store=settled.store_path, project_root=settled.project_root,
                requests={row.request_id for row in members if isinstance(row, PlannedRun)},
            )
            check_metric = _final_metric_check(settled, _train_history(settled, attempts))
            members = _settled_metric_members(members, lambda row: (
                check_metric(row.key, row.request_id, bind_metric_conditions(
                    body.rule, chosen.conditions, coordinates[id(row)],
                )) if isinstance(row, PlannedRun) else None
            ))
        return StudioRunRuleCheck(total=len(rows), metric_names=metrics, result=StudioRunRuleChosen(
            runs=[row.key.row for row in members],
            count=best.count * len(groups) if best else len(members),
            conditional=bool(chosen.conditions),
        ))

    def subset_to_rows(self, project_id: str, body: StudioSubsetToRowsRequest) -> StudioSubsetToRowsEdit:
        """Replace one measured selection of an Evaluate subject with the rows its chosen attempts bound.

        The other selections of the input, and every row they give, are left as they are.
        """
        from feedbax.web.services.trial_scope import file_value
        from feedbax.web.services.project_folder import preview_authored_files
        from feedbax.planner.project_plan import compile_project

        view = self.read(project_id, revision=body.revision)
        before = view.settled.plan
        subset = before.subsets.get((body.variant, body.step, body.input))
        project = self.folders.read(project_id, revision=body.revision).project
        step = project.steps.get(body.step)
        role, *item = body.input.split(".")
        binding = step.inputs.get(role) if step and role == "subject" else None
        index = int(item[0]) if item else None
        selection = (
            binding[index] if isinstance(binding, list) and index is not None and index < len(binding)
            else binding if index is None else None
        )
        if project.sweep is not None or not isinstance(selection, StepReference) or selection.where is None:
            raise ProjectFolderInvalid("Choose a locally authored Best input in the base project.")
        if subset is None or not subset.ranks or len(body.attempts) != len(subset.ranks):
            raise ProjectFolderInvalid("Choose one bound attempt for every Best rank.")
        runs = {
            run.key: run for run in before.runs if (run.key.variant, run.key.step) == (body.variant, body.step)
        }
        attempts = set(body.attempts)
        bound = {
            attempt.job_id: _planned_bound(attempt.bound_subject, runs[key].inputs.get("subject"))
            for key, history in view.attempts_by_run.items()
            if key in runs
            for attempt in history if attempt.job_id in attempts and attempt.bound_subject is not None
        }
        by_rank = {member.key: member for member in bound.values() if member is not None}
        if (
            bound.keys() != attempts or by_rank.keys() != set(subset.ranks)
            or any(
                len({by_rank[key].selection_sha256 for key in group.ranks}) != 1
                for group in subset.rank_groups
            )
        ):
            raise ProjectFolderInvalid("The chosen attempts do not share one complete Best selection.")
        frozen = subset_to_rows(subset.terms, (
            BoundRank(key=key, variant=by_rank[key].variant, row=by_rank[key].row) for key in subset.ranks
        ))
        address = AuthoredFileAddress(kind="step", id=body.step)
        authored = step.model_dump(mode="json", exclude_unset=True)
        at = ("inputs", "subject", *((str(index),) if index is not None else ()))
        file = file_value(address, authored, (*at, "where"), frozen.model_dump(mode="json", exclude_unset=True))
        if selection.checkpoint is None:
            file.patches.extend(file_value(address, authored, (*at, "checkpoint"), "last").patches)
        after = compile_project(
            preview_authored_files(project, [file]), readers=view.settled.readers,
            operations=self.folders.operations, components=self.folders.components,
            functions=self.folders.functions, python_tasks=self.folders.python_tasks,
            figure_templates=self.folders.figure_templates,
        )
        selected = {
            run.key: run for run in after.runs
            if (run.key.variant, run.key.step) == (body.variant, body.step)
        }
        error = "The selected producer rows cannot preserve this Evaluate definition."
        if selected.keys() != runs.keys():
            raise ProjectFolderInvalid(error)
        for key, run in selected.items():
            binding = run.inputs["subject"]
            previous = runs[key]
            if previous.inputs["subject"].member.key not in by_rank:
                if run.request_id != previous.request_id:
                    raise ProjectFolderInvalid(error)
                continue
            member = by_rank.get(binding.member.key)
            if (
                member is None or binding.producer is None
                or (binding.producer.variant, binding.producer.row) != (member.variant, member.row)
                or not isinstance(binding.member.reference, ProductRef)
                or binding.member.reference.request_id != member.producer_request_id
                or any(
                    axis not in run.coordinates or run.coordinates[axis].value_id != coordinate.value_id
                    or not same_typed_value(run.coordinates[axis].value, coordinate.value)
                    for axis, coordinate in previous.coordinates.items()
                )
                or not same_typed_value(
                    json.loads(before.requests[previous.request_id])["spec"],
                    json.loads(after.requests[run.request_id])["spec"],
                )
            ):
                raise ProjectFolderInvalid(error)
        return StudioSubsetToRowsEdit(revision=body.revision, files=[file])

    def _evaluation_result(
        self, project_id: str, body: StudioSavedEvaluationRead,
    ) -> tuple[SettledProjectPlan, EvaluationResult]:
        settled = self.folders.read_plan(project_id, revision=body.revision)
        key = RunKey(body.key.variant, body.key.step, body.key.row)
        run = settled.plan.row(key)
        if not isinstance(run, PlannedRun) or run.kind != "evaluate":
            raise JobNotFound(body.job_id)
        return settled, self.jobs.read_saved_evaluation_result(plan=settled, run=run, job_id=body.job_id)

    def evaluation_replay(
        self, project_id: str, body: StudioSavedEvaluationReplayRequest,
    ) -> StudioTrialReplayResponse:
        _, result = self._evaluation_result(project_id, body)
        try:
            return result.replay(
                body.trial, checkpoint=body.checkpoint, replicate=body.replicate, aggregate=body.aggregate, mean=body.mean,
                components=self.folders.components,
            )
        except CannotRun as error:
            raise ProjectFolderInvalid(str(error)) from error
        except (OSError, ValueError, KeyError) as error:
            raise JobConflict("The retained data for the selected Evaluate trial could not be read and authenticated.") from error

    @inspect_without_recording()
    def evaluation_objective(
        self, project_id: str, body: StudioSavedEvaluationRead,
    ) -> StudioEvaluationObjectiveSeries:
        settled = self.folders.read_plan(project_id, revision=body.revision)
        plan = settled.plan
        key = RunKey(body.key.variant, body.key.step, body.key.row)
        run = next((run for run in plan.runs if run.key == key and run.kind == "evaluate"), None)
        if run is None:
            raise JobNotFound(body.job_id)
        try:
            result = self.jobs.read_evaluation_objective_attempt(
                project=settled.project_id, store=settled.store_path, job_id=body.job_id, request_id=run.request_id,
            )
        except JobNotFound:
            return self._retained_evaluation_objective(project_id, body)
        try:
            _, params, subject = decode_rollout_request(run.request_id, plan.requests[run.request_id])
            if not isinstance(subject, ProductRef) or not isinstance(subject.select, UpdateSeriesMember):
                raise JobNotFound(body.job_id)
            train = decode_request(plan.requests[subject.request_id], plan.documents.__getitem__).spec
            if not isinstance(train, DocumentTrainingRequestSpec):
                raise ValueError("Evaluate subject has no authored Objective.")
            requested = list(subject.select.updates)
            objective = train.trial_documents.objective
            if isinstance(result, EvaluationResult):
                series = result.objective_series(requested, objective=objective)
            else:
                series = live_objective_series(
                    ImmutableArtifactBlobProvider(settled.store_path),
                    project_id=settled.project_id, job_id=body.job_id, request_id=run.request_id,
                    producer_request=subject.request_id, params=params, requested=requested,
                    objective=objective, observation=result,
                )
            return self._name_objective_families(settled, key, subject.request_id, objective, series)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise JobConflict("The selected Evaluate objective could not be read and authenticated.") from error

    def _retained_evaluation_objective(
        self, project_id: str, body: StudioSavedEvaluationRead,
    ) -> StudioEvaluationObjectiveSeries:
        settled, result = self._evaluation_result(project_id, body)
        try:
            store = ImmutableArtifactBlobProvider(settled.store_path)
            receipts = SuccessView(store)
            ref = receipts.attempt_ref(body.job_id)
            if ref is None:
                raise JobNotFound(body.job_id)
            saved = receipts.read_receipt(ref)
            _, _, subject = decode_rollout_request(saved.request.id, store.get_object(saved.request.object))
            if not isinstance(subject, ProductRef) or not isinstance(subject.select, UpdateSeriesMember):
                raise JobNotFound(body.job_id)
            (bound,) = saved.inputs["subject"]
            producer = bound.evaluation_subject
            if producer is None or producer.producer.id != subject.request_id:
                raise ValueError("Saved Evaluate names a different Train request.")
            train = receipts.read_request(producer.producer).spec
            if not isinstance(train, DocumentTrainingRequestSpec):
                raise ValueError("Evaluate subject has no authored Objective.")
            objective = train.trial_documents.objective
            series = result.objective_series(list(subject.select.updates), objective=objective)
            key = RunKey(body.key.variant, body.key.step, body.key.row)
            return self._name_objective_families(settled, key, subject.request_id, objective, series)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise JobConflict("The retained Evaluate objective could not be read and authenticated.") from error

    def _name_objective_families(
        self, settled: SettledProjectPlan, evaluate: RunKey, producer_request: RequestId,
        objective: ObjectiveDocument, series: StudioEvaluationObjectiveSeries,
    ) -> StudioEvaluationObjectiveSeries:
        """Read family labels from the selected definition without including them in the request."""
        families = {term.family for term in objective.terms if term.enabled and term.family is not None}
        if not families or not series.points:
            return series
        run = settled.plan.row(evaluate)
        binding = run.inputs.get("subject") if isinstance(run, PlannedRun) else None
        producer = settled.plan.row(binding.producer) if binding and binding.producer is not None else None
        if not isinstance(producer, PlannedRun) or producer.request_id != producer_request:
            return series
        definition = settled.plan.selected_definitions.get(producer.key)
        if definition is None:
            selected = self.folders.read_plan(settled.project_id, revision=settled.revision, source_runs=(producer.key,))
            definition = selected.plan.selected_definitions.get(producer.key)
        if definition is None:
            return series
        slots, _ = trial_document_slots(definition.payload)
        authored = slots.get("objective")
        terms = authored.get("terms") if isinstance(authored, dict) else None
        if not isinstance(terms, dict):
            return series
        names = {
            pointer((key,)): label for key in families
            if isinstance(member := terms.get(key), dict) and isinstance(label := member.get("label"), str)
        }
        return series.model_copy(update={"points": [
            point.model_copy(update={"terms": [
                term.model_copy(update={"name": names[term.path]}) if term.path in names else term
                for term in point.terms
            ]})
            for point in series.points
        ]})


def _final_metric_check(
    settled: SettledProjectPlan, history: Mapping[RunKey, TrainRowHistory],
) -> Callable[[RunKey, RequestId, tuple[Expression, ...]], bool | None]:
    @cache
    def read_metrics(job_id: str, request: RequestId) -> Callable[[str], float]:
        view = SuccessView(ImmutableArtifactBlobProvider(settled.store_path))
        evidence = view.attempt_ref(job_id)
        if evidence is None:
            raise ValueError("The Train result has no accepted Store receipt.")
        return read_final_train_metrics(view.store, view.read_receipt(evidence), request)

    def check(key: RunKey, request: RequestId, conditions: tuple[Expression, ...]) -> bool | None:
        producer = history.get(key)
        if producer is None or producer.status is not JobStatus.SUCCEEDED or producer.result is None:
            return None
        return metric_condition_holds(conditions, read_metrics(producer.result.job_id, request))

    return check


def _settled_metric_members[T](members: tuple[T, ...], check: Callable[[T], bool | None]) -> tuple[T, ...]:
    """Keep the candidate set until every Train has a result. Invalid metrics cannot choose a row."""
    passing = []
    for member in members:
        try:
            decision = check(member)
        except (OSError, ValueError, KeyError):
            continue
        if decision is None:
            return members
        if decision:
            passing.append(member)
    return tuple(passing)


def read_settled_project_plan(
    settled: SettledProjectPlan, reader: PlanJobReader, *, stopped_checkpoint_jobs: Sequence[str] = (),
    remote_attempts: Sequence[StudioActivityJob] = (),
    openings: Mapping[RunKey, ApprovedRowStatus] | None = None,
) -> ProjectPlanRead:
    """Join one saved Plan to the verified Host evidence used by Studio."""
    light = settled.plan.light_requests
    runs = settled.plan.runs
    requests = {run.request_id for run in runs} | {node.request_id for node in light.values()}
    jobs = reader.read_plan_attempts(
        project=settled.project_id, store=settled.store_path, requests=requests,
        stopped_checkpoint_jobs=stopped_checkpoint_jobs,
        addresses={plan_attempt_key(run.key) for run in runs} | {plan_attempt_key(key) for key in light},
        project_root=settled.project_root,
    )
    if remote_attempts:
        by_request = dict(jobs.by_request)
        for observed in remote_attempts:
            if observed.request_id not in requests:
                continue
            local = by_request.get(observed.request_id, ())
            if any(attempt.job_id == observed.job_id for attempt in local):
                continue
            by_request[observed.request_id] = (*local, PlanJobAttempt(
                observed.job_id, observed.status, observed.phase, observed.progress, observed.reason, None,
                continues=observed.continues, collection_progress=observed.collection_progress,
            ))
        jobs = replace(jobs, by_request=MappingProxyType(by_request))
    history = dict(_train_history(settled, jobs))
    conditional = {
        run.key: binding for run in runs for binding in run.inputs.values()
        if isinstance(binding.member.reference, ConditionalProduct)
    }
    decisions: dict[RunKey, tuple[JobStatus, str]] = {}
    if conditional:
        check_metric = _final_metric_check(settled, history)
        for key, binding in conditional.items():
            try:
                reference = binding.member.reference
                assert isinstance(reference, ConditionalProduct) and binding.producer is not None
                if check_metric(binding.producer, reference.request_id, reference.conditions) is False:
                    decisions[key] = JobStatus.NOT_CHOSEN, METRIC_CONDITION_FALSE
            except (OSError, ValueError, KeyError) as error:
                decisions[key] = JobStatus.CANNOT_RUN, str(error)
        for key in history.keys() & decisions.keys():
            status, reason = decisions[key]
            history[key] = replace(history[key], status=status, reason=reason)
    for key, opening in (openings or {}).items():
        if key in history:
            history[key] = replace(history[key], status=opening.status, reason=opening.reason)
    evaluations = {}
    evaluation_problems = []
    for run in runs:
        if run.kind == "evaluate":
            row, problem = _evaluation_row(settled, run, jobs, reader, history)
            if (decision := decisions.get(run.key)) is not None:
                status, reason = decision
                row = row.model_copy(update={"status": status, "reason": reason, "can_run": False})
                if status is JobStatus.NOT_CHOSEN:
                    problem = None
            if (opening := (openings or {}).get(run.key)) is not None:
                row = row.model_copy(update={"status": opening.status, "reason": opening.reason})
            evaluations[run.key] = row
            if problem is not None:
                evaluation_problems.append(problem)
    return ProjectPlanRead(
        settled,
        jobs,
        MappingProxyType({
            run.key: attempts
            for run in runs
            if (attempts := jobs.by_request.get(run.request_id)) is not None
        }),
        MappingProxyType(history),
        MappingProxyType(evaluations),
        tuple(evaluation_problems),
        _light_histories(settled, jobs, history, evaluations),
        openings or {},
    )


def _dependency_state(
    plan: ProjectPlan, inputs: Mapping[RunKey | ElementKey, JobStatus],
) -> tuple[JobStatus, str] | None:
    waiting = {
        JobStatus.NOT_RUN, JobStatus.WAITING, JobStatus.QUEUED, JobStatus.STARTING,
        JobStatus.RUNNING, JobStatus.STOPPING, JobStatus.AWAITING_AUTHORIZATION, JobStatus.COLLECTING,
    }
    for states, status, message in (
        ({JobStatus.STALE}, JobStatus.STALE, "{} is out of date."),
        (waiting, JobStatus.WAITING, "Waiting for {}."),
    ):
        if (source := next((key for key, state in inputs.items() if state in states), None)) is not None:
            key = source.run if isinstance(source, ElementKey) else source
            label = plan.steps[key.variant, key.step].label
            return status, message.format(label)
    return None


def _dependency_status(
    dependency: tuple[JobStatus, str] | None, status: JobStatus, reason: str | None,
) -> tuple[JobStatus, str | None]:
    """Explain open or stale rows while preserving a current receipt's validity."""
    if dependency is not None and (
        status in {JobStatus.NOT_RUN, JobStatus.QUEUED}
        or dependency[0] is JobStatus.STALE and status is JobStatus.STALE
    ):
        return dependency
    return status, reason


def _light_histories(
    settled: SettledProjectPlan, jobs: PlanJobAttempts,
    trains: Mapping[RunKey, TrainRowHistory], evaluations: Mapping[RunKey, StudioSavedEvaluationRow],
) -> Mapping[RunKey | ElementKey, LightHistory]:
    nodes = settled.plan.light_requests
    histories: dict[RunKey | ElementKey, LightHistory] = {}

    def visit(key: RunKey | ElementKey) -> LightHistory:
        if key in histories:
            return histories[key]
        node = nodes[key]
        row = _light_history(settled, jobs, node)
        if row.status in {JobStatus.NOT_RUN, JobStatus.STALE, JobStatus.QUEUED}:
            inputs: dict[RunKey | ElementKey, JobStatus] = {}
            for bindings in planned_input_groups(node).values():
                for binding in bindings:
                    if binding.terminal is not None:
                        continue
                    for source in (binding.element,) if binding.element is not None else binding.sources:
                        if source in nodes:
                            inputs[source] = visit(source).status
                        elif isinstance(source, RunKey) and (upstream := trains.get(source) or evaluations.get(source)):
                            inputs[source] = upstream.status
            status, reason = _dependency_status(_dependency_state(settled.plan, inputs), row.status, row.reason)
            row = replace(row, status=status, reason=reason)
        histories[key] = row
        return row

    for key in nodes:
        visit(key)
    return MappingProxyType(histories)


def _light_history(
    settled: SettledProjectPlan, jobs: PlanJobAttempts, node: PlannedRun | PlannedElement,
) -> LightHistory:
    key = node.key
    plan = settled.plan
    attempts = jobs.by_request.get(node.request_id, ())
    successes = tuple(attempt for attempt in attempts if attempt.light_success is not None)
    current = successes[-1] if successes else None
    addressed = jobs.by_address.get(plan_attempt_key(key), ())
    prior = next((attempt for attempt in reversed(addressed) if (
        attempt.light_success is not None and attempt != current
    )), None)
    if prior is None and len(successes) > 1:
        prior = successes[-2]
    changed = None
    if prior is not None and not attempts:
        assert prior.light_success is not None
        store = ImmutableArtifactBlobProvider(settled.store_path)
        request = prior.light_success.request
        old = SuccessView(store).read_request(request)
        new = decode_request(plan.requests[node.request_id], plan.documents.__getitem__)
        path, parts = authored_request_change(key, old, new)
        changed = StudioDefinitionChange(path=path, place=definition_place(
            parts, node.document_ids.get if isinstance(node, PlannedRun) else lambda _slot: None,
            element=key.name if isinstance(key, ElementKey) else None,
            element_name=node.label if isinstance(node, PlannedElement) else None,
            variation=variation_targets(new.spec.params if isinstance(new.spec, EvaluationRequestSpec) else None),
        ))
    status, reason = saved_request_status(
        attempts, success=current, changed=changed is not None,
        refused=isinstance(node, PlannedRun) and node.admission_problem is not None,
    )
    return LightHistory(status, reason, _light_result(settled, current), _light_result(settled, prior), changed)


def _operation_values(
    dataset: Dataset, field: str, shape: tuple[int, ...], offset: int, stop: int,
) -> list[JsonValue]:
    """Read trailing-axis slices instead of expanding an entire tensor for a page."""
    if not shape:
        return [_operation_value(np.asarray(dataset.data(field)).item())]
    values = []
    while offset < stop:
        *prefix, start = map(int, np.unravel_index(offset, shape))
        count = min(shape[-1] - start, stop - offset)
        selection = (*prefix, slice(start, start + count))
        values.extend(_operation_value(value) for value in np.asarray(dataset.data(field, selection)).tolist())
        offset += count
    return values


def _operation_value(value: bool | int | float | complex) -> JsonValue:
    if isinstance(value, complex):
        return {"real": _operation_value(value.real), "imag": _operation_value(value.imag)}
    if (type(value) is int and abs(value) >= 2**53) or (type(value) is float and not isfinite(value)):
        return str(value)
    return value


def _light_result(settled: SettledProjectPlan, attempt: PlanJobAttempt | None) -> StudioSavedLightResult | None:
    if attempt is None or attempt.light_success is None or attempt.successful_receipt is None:
        return None
    success = attempt.light_success
    project = f"/api/projects/{quote(settled.project_id, safe='')}"
    attempt_path = f"{settled.revision}/{attempt.job_id}"
    if success.request.id.startswith(("analyze:", "synthesize:")):
        products = {
            role: f"{project}/operation-products/{attempt_path}/{quote(role, safe='')}"
            for role, members in success.products.items()
            if len(members) == 1 and members[0].content.media_type == DATASET_MEDIA_TYPE
        }
    else:
        products = {
            role: f"{project}/light-results/{attempt_path}/{role}/{entry}"
            for role, entry in ((PLOTLY_FIGURE_ROLE, "figure.json"), ("report", REPORT_HTML_NAME))
            if role in success.products
        }
    return StudioSavedLightResult(
        job_id=attempt.job_id, request_id=success.request.id, revision=attempt.revision, receipt=attempt.successful_receipt,
        products=products, validity=attempt.validity, provenance=attempt.provenance,
    )


def _evaluation_params(settled: SettledProjectPlan, run: PlannedRun) -> Mapping[str, object] | None:
    spec = decode_request(settled.plan.requests[run.request_id], settled.plan.documents.__getitem__).spec
    return spec.params if isinstance(spec, EvaluationRequestSpec) else None


def _named_trials(ids: Sequence[str], names: TrialNameParts) -> list[StudioTrialName]:
    known = [names[id] for id in ids if id in names]
    varying = {
        index for index, parts in enumerate(zip(*known)) if len(set(parts)) > 1
    }
    result = []
    for id in ids:
        parts = names.get(id)
        label = None
        if parts is not None:
            label = " · ".join(part for index, part in enumerate(parts) if index in varying) or parts[0]
        result.append(StudioTrialName(id=id, name=label))
    return result


def _evaluation_row(
    settled: SettledProjectPlan, run: PlannedRun, jobs: PlanJobAttempts, reader: PlanJobReader,
    history: Mapping[RunKey, TrainRowHistory],
) -> tuple[StudioSavedEvaluationRow, StudioProjectWindowMapProblem | None]:
    key = StudioSavedTrainKey.model_validate(run.key, from_attributes=True)
    series = None
    rule = None
    planned = run.inputs.get("subject")
    reference = planned.member.reference if planned is not None else None
    if isinstance(reference, ProductRef):
        match reference.select:
            case UpdateSeriesMember(rule=rule, updates=updates):
                series = StudioEvaluationCheckpointSeries(
                    subject=StudioSavedTrainKey.model_validate(planned.producer, from_attributes=True),
                    requested=list(updates),
                )
            case BatchMember(batch=batch):
                rule = BatchCheckpoint(at=batch)
            case LastMember():
                rule = "last"
    sources = planned.sources if planned is not None and planned.terminal is None else ()
    inputs: dict[RunKey | ElementKey, JobStatus] = {source: upstream.status for source in sources if (upstream := history.get(source)) is not None}
    if is_synthesized_model_product(reference):
        for source in sources:
            if (producer := settled.plan.light_requests.get(source)) is not None:
                inputs[source] = _light_history(settled, jobs, producer).status
    dependency = _dependency_state(settled.plan, inputs)
    trials = []
    problem = None
    can_run = False
    try:
        params, subject = reader.evaluation_parameters(settled.plan, run)
        trials = list(params.trials)
        match subject:
            case ProductRef(request_id=request) if planned is None or planned.terminal is None:
                producers = (request,)
            case BestRank(candidates=candidates):
                producers = tuple(candidate.reference.request_id for candidate in candidates)
            case _:
                producers = ()
        if is_synthesized_model_product(subject):
            can_run = any(
                attempt.light_success is not None or attempt.status in {
                    JobStatus.QUEUED, JobStatus.STARTING, JobStatus.RUNNING,
                }
                for attempt in jobs.by_request.get(subject.request_id, ())
            )
            if not can_run and dependency is None:
                raise CannotRun("/inputs/subject", "Synthesize must publish its model before Evaluate can run.")
        else:
            can_run = set(producers) <= jobs.checkpoint_producers
        if not can_run and dependency is None:
            raise CannotRun("/inputs/subject", EVALUATION_SUBJECT_REQUIRED)
        if can_run and isinstance(subject, ProductRef) and isinstance(subject.select, BatchSetMember) and not set(subject.select.batches) <= set(
            jobs.checkpoint_catalogs.get(subject.request_id, ())
        ):
            raise CannotRun("/inputs/subject/checkpoint", "admitted Train catalogue omits requested batches")
        if (
            isinstance(subject, ProductRef) and isinstance(subject.select, BatchMember) and planned is not None
            and planned.producer is not None
            and (held := jobs.finished_checkpoints.get(subject.request_id)) is not None
            and subject.select.batch not in held
        ):
            raise CannotRun(
                _subject_checkpoint_path(settled, run, planned),
                f"Update {subject.select.batch} was not saved by {planned.producer.step!r}, "
                f"which saved updates {', '.join(map(str, held)) or 'none'}.",
            )
    except CannotRun as error:
        can_run = False
        problem = StudioProjectWindowMapProblem(
            code="cannot_run", path=error.path, detail=error.reason,
            variant=key.variant, step=key.step, row=key.row,
            place=definition_place(
                path_parts(error.path), run.document_ids.get,
                variation=variation_targets(_evaluation_params(settled, run)),
            ),
        )
    attempts = jobs.by_request.get(run.request_id, ())
    success = next((attempt for attempt in reversed(attempts) if (
        attempt.successful_receipt is not None
        and attempt.output_size_bytes is not None
    )), None)
    previous = next((attempt for attempt in reversed(jobs.by_address.get(plan_attempt_key(run.key), ())) if (
        attempt.request_id != run.request_id and attempt.successful_receipt is not None
        and attempt.output_size_bytes is not None
    )), None)
    names = run.trial_names or numbered_trial_parts(trials)
    result = _saved_evaluation_result(success, planned, names)
    prior = _saved_evaluation_result(previous, None, names)
    retained = bool(attempts and attempts[-1].collected_evaluate) or (
        success is not None and success.collected_evaluate
    )
    status, reason = saved_request_status(
        attempts, success=success, refused=problem is not None and not retained, changed=prior is not None,
    )
    if problem is None:
        status, reason = _dependency_status(dependency, status, reason)
    displayed_attempt = active_plan_attempt(attempts)
    if displayed_attempt is None and attempts:
        displayed_attempt = success if status is JobStatus.SUCCEEDED else attempts[-1]
    return StudioSavedEvaluationRow(
        key=key, request_id=run.request_id,
        status=status, reason=reason,
        can_run=can_run, trials=_named_trials(trials, names), result=result, previous=prior,
        checkpoint_series=series, checkpoint_rule=rule,
        checkpoint_attempt=displayed_attempt.checkpoint_attempt if series is not None and displayed_attempt else None,
        subject_rank_key=planned.member.key if isinstance(reference, BestRank) and reference.group else None,
        bound_subject=_planned_bound(attempts[-1].bound_subject, planned) if attempts else None,
    ), problem



def _saved_evaluation_result(
    attempt: PlanJobAttempt | None, planned: PlannedInput | None, names: TrialNameParts,
) -> StudioSavedEvaluationResult | None:
    if attempt is None:
        return None
    assert attempt.successful_receipt is not None and attempt.output_size_bytes is not None
    return StudioSavedEvaluationResult(
        job_id=attempt.job_id, request_id=attempt.request_id, revision=attempt.revision,
        receipt=attempt.successful_receipt, trials=_named_trials(attempt.trials, names),
        checkpoint_source=attempt.checkpoint_source, size_bytes=attempt.output_size_bytes,
        checkpoint_sources=list(attempt.checkpoint_sources) or None,
        checkpoint_gaps=list(attempt.checkpoint_gaps) if attempt.checkpoint_gaps is not None else None,
        placement=attempt.evaluation_placement if attempt.checkpoint_gaps is not None else None,
        bound_subject=_planned_bound(attempt.bound_subject, planned), validity=attempt.validity,
        provenance=attempt.provenance,
        loaded_model=attempt.loaded_model,
        origin=attempt.model_origin,
        replicates=attempt.replicates,
    )

def _subject_checkpoint_path(settled: SettledProjectPlan, run: PlannedRun, planned: PlannedInput) -> str:
    """Where the checkpoint of the selection that chose ``planned`` is authored: its list item's, or the role's."""
    step = settled.plan.steps[run.key.variant, run.key.step]
    authored = (step.inputs if step.inputs is not None else step.authored_inputs).get("subject")
    if isinstance(authored, list) and planned.selection in authored:
        return f"/inputs/subject/{authored.index(planned.selection)}/checkpoint"
    return "/inputs/subject/checkpoint"


def _planned_bound(bound: StudioBoundBestSubject | None, planned: PlannedInput | None) -> StudioBoundBestSubject | None:
    """An attempt's bound rank, keyed by the member it fills: a request keys a rank by its position, the plan by its
    selection too."""
    if bound is None or planned is None:
        return bound
    return bound.model_copy(update={"key": planned.member.key})


def _train_history(
    settled: SettledProjectPlan, jobs: PlanJobAttempts
) -> Mapping[RunKey, TrainRowHistory]:
    ref_rows = _history_rows(settled, jobs)
    successful = {
        request: tuple(
            attempt for attempt in attempts
            if attempt.successful_receipt is not None
            and (attempt.train_request is not None or attempt.binding_sha256 is not None)
        )
        for request, attempts in jobs.by_request.items()
    }
    by_attempt = {
        (request, attempt.job_id): attempt
        for request, attempts in successful.items()
        for attempt in attempts
    }
    changes: dict[tuple[str, str], StudioSavedTrainChange] = {}
    history = {}
    for run in settled.plan.runs:
        if run.kind != "train":
            continue
        attempts = jobs.by_request.get(run.request_id, ())
        row = ref_rows.get(plan_run_key(run.key.variant, run.key.step, run.key.row))
        results = successful.get(run.request_id, ())
        last, prior = None, None
        if row is not None:
            admitted_results = successful.get(row["request_id"], ())
            admitted = _success(row["request_id"], admitted_results[-1]) if admitted_results else None
            last, prior = settled_results(row, admitted)
        current = _success(run.request_id, results[-1]) if results else None
        if last is not None and last["request_id"] != run.request_id:
            previous = last
        elif len(results) > 1:
            previous = _success(run.request_id, results[-2])
        else:
            previous = prior
        if previous is None:
            addressed = jobs.by_address.get(plan_attempt_key(run.key), ())
            former = next((
                attempt for attempt in reversed(addressed)
                if attempt.request_id is not None and attempt.request_id != run.request_id
                and (attempt.request_id, attempt.job_id) in by_attempt
            ), None)
            if former is not None and former.request_id is not None:
                previous = _success(former.request_id, former)
        older = _result(previous, by_attempt)
        change = None
        if older is not None and older.request_id != run.request_id:
            key = (older.job_id, run.request_id)
            if key not in changes:
                changes[key] = _definition_change(
                    settled, run, older, by_attempt[older.request_id, older.job_id]
                )
            change = changes[key]
        status, reason = saved_request_status(
            attempts, success=results[-1] if results else None,
            refused=run.admission_problem is not None, changed=change is not None,
        )
        history[run.key] = TrainRowHistory(status, reason, _result(current, by_attempt), older, change)
    return MappingProxyType(history)


def _history_rows(settled: SettledProjectPlan, jobs: PlanJobAttempts) -> Mapping[str, PlanRowRef]:
    ref = jobs.plan_ref
    if ref is None:
        return {}
    try:
        store = ImmutableArtifactBlobProvider(settled.store_path)
        plan = json.loads(store.get_object(ContentObjectRef.model_validate(ref["plan"])))
        rows = ref["rows"]
        if plan["requests"] != {key: row["request_id"] for key, row in rows.items()}:
            raise ValueError("Authored Train Plan request map differs from its ref")
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ProjectFolderInvalid(f"Saved Train Plan history is unavailable: {error}") from error
    return rows


def _success(request_id: str, attempt: PlanJobAttempt) -> PlanSuccess:
    return {"request_id": request_id, "job_id": attempt.job_id}


def _result(
    pointer: PlanSuccess | None, attempts: Mapping[tuple[str, str], PlanJobAttempt]
) -> StudioSavedTrainResult | None:
    if pointer is None:
        return None
    attempt = attempts.get((pointer["request_id"], pointer["job_id"]))
    if attempt is None:
        return None
    assert attempt.successful_receipt is not None
    return StudioSavedTrainResult(
        request_id=pointer["request_id"], job_id=attempt.job_id,
        receipt=attempt.successful_receipt, progress=attempt.progress, revision=attempt.revision,
        validity=attempt.validity, provenance=attempt.provenance,
    )


def _definition_change(
    settled: SettledProjectPlan,
    run: PlannedRun,
    previous: StudioSavedTrainResult,
    attempt: PlanJobAttempt,
) -> StudioSavedTrainChange:
    root = settled.store_path
    try:
        request = attempt.train_request
        if request is None:
            relative = training_binding_path(root, attempt.job_id).relative_to(root).as_posix()
            with completion_file(root, relative) as (stream, _):
                raw = stream.read()
            if sha256(raw).hexdigest() != attempt.binding_sha256:
                raise ValueError("Previous Train binding changed")
            launch = BoundStudioTrainingJobInput.model_validate_json(raw)
            if (launch.project_id, launch.request_id, launch.manifest_root) != (
                settled.project_id, previous.request_id, root,
            ):
                raise ValueError("Previous Train binding differs from its saved request")
            binding = launch.canonical_request
            request = RequestObject(id=launch.request_id, object=binding.request, documents=binding.documents)
        store = ImmutableArtifactBlobProvider(root)
        old = read_success_request(request, store.get_object)
        if old.id != previous.request_id:
            raise ValueError("Previous Train object has a different request ID")
    except (OSError, ValueError, KeyError):
        return StudioSavedTrainChange(
            path="request_id", place=StudioStepPlace(part="step"), before=previous.request_id, after=run.request_id,
        )
    new = decode_request(settled.plan.requests[run.request_id], settled.plan.documents.__getitem__)
    change = first_request_change(old, new)
    return StudioSavedTrainChange(
        path=change.path, place=definition_place(change.parts or path_parts(change.path), run.document_ids.get),
        before=change.left, after=change.right,
    )
