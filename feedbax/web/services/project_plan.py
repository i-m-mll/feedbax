"""Join the acknowledged compiler projection to observed Jobs attempts."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType

from feedbax.contracts.artifact_custody import ContentObjectRef
from feedbax.contracts.figure_document import PLOTLY_FIGURE_ROLE, PlotlyFigurePort
from feedbax.contracts.intent_diff import first_request_change
from feedbax.contracts.jobs import (
    JobStatus,
    StudioProjectWindowMap,
    StudioProjectWindowMapCoordinate,
    StudioProjectWindowMapEdge,
    StudioProjectWindowMapElement,
    StudioProjectWindowMapFigure,
    StudioProjectWindowMapInput,
    StudioProjectWindowMapProblem,
    StudioProjectWindowMapRow,
    StudioProjectWindowMapStep,
    StudioProjectWindowMapSubset,
    StudioSavedEvaluationReplayRequest,
    StudioSavedEvaluationResult,
    StudioSavedEvaluationRow,
    StudioSavedTrainChange,
    StudioSavedTrainKey,
    StudioSavedTrainResult,
)
from feedbax.contracts.project_inputs import InputBinding, StepReference
from feedbax.contracts.request_refs import BatchSetMember
from feedbax.contracts.project_plan import ElementKey, PlannedAnalyzeRun, PlannedRun, RunKey
from feedbax.contracts.project_resolution import ProjectDiagnostic
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.workspace_replay import WorkspaceReplayTrial
from feedbax.host.completion import completion_file
from feedbax.host.plan_history import PlanRowRef, PlanSuccess, plan_run_key, settled_results
from feedbax.job_completion import training_binding_path
from feedbax.jobs import (
    EVALUATION_SUBJECT_REQUIRED,
    JobConflict,
    JobNotFound,
    JobService,
    PlanJobAttempt,
    PlanJobAttempts,
    PlanJobReader,
    saved_request_status,
)
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.studio.canonical_evaluate import CannotRun
from feedbax.studio.canonical_train import BoundStudioTrainingJobInput
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


@dataclass(frozen=True)
class ProjectPlanRead:
    settled: SettledProjectPlan
    jobs: PlanJobAttempts
    attempts_by_run: Mapping[RunKey, tuple[PlanJobAttempt, ...]]
    history: Mapping[RunKey, TrainRowHistory]
    evaluations: Mapping[RunKey, StudioSavedEvaluationRow]
    evaluation_problems: tuple[StudioProjectWindowMapProblem, ...]

    def status_rows(self) -> Iterator[PlanStatusRow]:
        """Project saved attempt states and the compiler's refused rows and elements."""
        plan = self.settled.plan
        problems: dict[RunKey, list[ProjectDiagnostic]] = defaultdict(list)
        for problem in plan.diagnostics:
            if problem.variant is not None and problem.step is not None and problem.row is not None:
                problems[RunKey(problem.variant, problem.step, problem.row)].append(problem)

        for row in plan.rows:
            label = row.label or row.key.row
            observed = self.history.get(row.key) or self.evaluations.get(row.key)
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
                yield _planned_status(row.key, label, problems[row.key])
                for name in dict.fromkeys((*row.elements, *element_problems)):
                    yield _planned_status(ElementKey(row.key, name), label, element_problems[name])
            elif isinstance(row, PlannedRun):
                yield _planned_status(row.key, label, row_problems)
            else:
                reason = _problem_reasons(row_problems) or "No request could be planned."
                yield PlanStatusRow(row.key, label, JobStatus.CANNOT_RUN, reason)

    def window_map(self) -> StudioProjectWindowMap:
        """Project authored steps and compiler facts from the acknowledged Plan."""
        plan = self.settled.plan
        document_ids: dict[tuple[str, str], dict[str, str]] = {}
        document_uses: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in plan.rows:
            key = row.key.variant, row.key.step
            document_uses[key].update(row.document_uses)
            shared = document_ids.get(key)
            document_ids[key] = (
                dict(row.document_ids)
                if shared is None
                else {path: name for path, name in shared.items() if row.document_ids.get(path) == name}
            )
        statuses = {row.key: row for row in self.status_rows()}
        counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        for key, status in statuses.items():
            if isinstance(key, RunKey):
                counts[key.variant, key.step][status.status.value] += 1

        edge_keys = [
            (binding.producer, run.key) for run in plan.runs for binding in run.inputs.values()
        ]
        edge_keys.extend(
            (row.states.producer, row.key) for row in plan.rows if isinstance(row, PlannedAnalyzeRun)
        )

        def inputs(variant: str, step_id: str) -> Iterator[StudioProjectWindowMapInput]:
            step = plan.steps[variant, step_id]
            for role, binding in (step.inputs if step.inputs is not None else step.authored_inputs).items():
                subset = plan.subsets.get((variant, step_id, role))
                for input_id, reference in _step_references(role, binding):
                    yield StudioProjectWindowMapInput(
                        id=input_id,
                        source_step_id=reference.step,
                        subset=StudioProjectWindowMapSubset(
                            terms=subset.terms, count=len(subset.members), total=subset.candidates,
                            document=subset.document,
                        ) if subset is not None else None,
                    )

        return StudioProjectWindowMap(
            variants=list(plan.variants),
            steps=[
                StudioProjectWindowMapStep(
                    variant=variant, step_id=step_id, kind=step.kind, group=step.group,
                    inputs=list(inputs(variant, step_id)),
                    document_ids=document_ids.get((variant, step_id), {}),
                    document_uses=sorted(document_uses[variant, step_id]),
                    status_counts=dict(counts[variant, step_id]),
                )
                for (variant, step_id), step in plan.steps.items()
            ],
            rows=[
                StudioProjectWindowMapRow(
                    key=StudioSavedTrainKey.model_validate(row.key, from_attributes=True),
                    label=row.label or row.key.row,
                    status=statuses[row.key].status,
                    reason=statuses[row.key].reason,
                    coordinates={
                        axis: StudioProjectWindowMapCoordinate.model_validate(
                            coordinate, from_attributes=True
                        )
                        for axis, coordinate in row.coordinates.items()
                    },
                )
                for row in plan.rows
            ],
            edges=[
                StudioProjectWindowMapEdge(
                    from_key=StudioSavedTrainKey.model_validate(producer, from_attributes=True),
                    to_key=StudioSavedTrainKey.model_validate(consumer, from_attributes=True),
                )
                for producer, consumer in edge_keys
            ],
            problems=[
                *(StudioProjectWindowMapProblem.model_validate(problem, from_attributes=True)
                  for problem in plan.diagnostics),
                *(StudioProjectWindowMapProblem(
                    code="cannot_run", path=problem.path, detail=problem.detail,
                    variant=run.key.variant, step=run.key.step, row=run.key.row,
                    document=run.document_ids.get("payload.graph"),
                ) for run in plan.runs if (problem := run.admission_problem) is not None),
                *self.evaluation_problems,
            ],
            evaluations=list(self.evaluations.values()),
            analysis_elements=[
                StudioProjectWindowMapElement(
                    key=StudioSavedTrainKey.model_validate(element.key.run, from_attributes=True),
                    name=element.key.name, kind=element.kind, request_id=element.request_id, status="not_run",
                    figure=StepReference(
                        step=element.key.run.step, row=element.key.run.row,
                        element=element.key.name, product=PLOTLY_FIGURE_ROLE,
                    ) if isinstance(element.outputs.get(PLOTLY_FIGURE_ROLE), PlotlyFigurePort) else None,
                )
                for element in plan.elements.values() if statuses[element.key].status is JobStatus.NOT_RUN
            ],
            analysis_figures=[
                StudioProjectWindowMapFigure(
                    key=StudioSavedTrainKey.model_validate(row.key, from_attributes=True),
                    name=name, data=list(figure.data), caption=figure.caption,
                )
                for row in plan.rows if isinstance(row, PlannedAnalyzeRun)
                for name, figure in row.figures.items()
            ],
        )


def _step_references(
    role: str, binding: InputBinding | None,
) -> Iterator[tuple[str, StepReference]]:
    if isinstance(binding, StepReference):
        yield role, binding
    elif isinstance(binding, list):
        for index, reference in enumerate(binding):
            if isinstance(reference, StepReference):
                yield f"{role}.{index}", reference
    elif isinstance(binding, dict):
        for key, reference in binding.items():
            if isinstance(reference, StepReference):
                yield f"{role}.{key}", reference


def _problem_reasons(problems: Iterable[ProjectDiagnostic]) -> str | None:
    return " | ".join(f"{problem.path}: {problem.detail}" for problem in problems) or None


def _planned_status(
    key: RunKey | ElementKey, label: str, problems: list[ProjectDiagnostic],
) -> PlanStatusRow:
    status = JobStatus.CANNOT_RUN if problems else JobStatus.NOT_RUN
    return PlanStatusRow(key, label, status, _problem_reasons(problems))


class ProjectPlanService:
    def __init__(self, folders: ProjectFolderService, jobs: JobService) -> None:
        self.folders = folders
        self.jobs = jobs

    def read(
        self, project_id: str, *, revision: str, source_runs: Iterable[RunKey] = ()
    ) -> ProjectPlanRead:
        """Read one saved revision and its request-scoped attempt evidence."""
        settled = self.folders.read_plan(project_id, revision=revision, source_runs=source_runs)
        return read_settled_project_plan(settled, self.jobs)

    def evaluation_replay(
        self, project_id: str, body: StudioSavedEvaluationReplayRequest,
    ) -> WorkspaceReplayTrial:
        view = self.read(project_id, revision=body.revision)
        key = RunKey(body.key.variant, body.key.step, body.key.row)
        row = view.evaluations.get(key)
        receipt = next((
            attempt.successful_receipt for attempt in view.attempts_by_run.get(key, ())
            if attempt.job_id == body.job_id
        ), None)
        if row is None or receipt is None:
            raise JobNotFound(body.job_id)
        if body.trial not in row.trials:
            raise ProjectFolderInvalid("The selected trial is absent from the saved Evaluate request.")
        result = self.jobs.read_evaluation_result(
            project=project_id, store=view.settled.store_path, job_id=body.job_id,
        )
        if result.receipt != receipt:
            raise JobConflict("The selected Evaluate receipt changed.")
        try:
            return result.replay_trial(body.trial, anchor_id="effector", checkpoint=body.checkpoint)
        except CannotRun as error:
            raise ProjectFolderInvalid(str(error)) from error
        except (OSError, ValueError, KeyError) as error:
            raise JobConflict("The retained data for the selected Evaluate trial could not be read and authenticated.") from error


def read_settled_project_plan(settled: SettledProjectPlan, reader: PlanJobReader) -> ProjectPlanRead:
    """Join one saved Plan to the verified Host evidence used by Studio."""
    requests = {run.request_id for run in settled.plan.runs if run.kind in {"train", "evaluate"}}
    jobs = reader.read_plan_attempts(
        project=settled.project_id, store=settled.store_path, requests=requests
    )
    evaluations = {}
    evaluation_problems = []
    for run in settled.plan.runs:
        if run.kind == "evaluate":
            row, problem = _evaluation_row(settled, run, jobs, reader)
            evaluations[run.key] = row
            if problem is not None:
                evaluation_problems.append(problem)
    return ProjectPlanRead(
        settled,
        jobs,
        MappingProxyType({
            run.key: attempts
            for run in settled.plan.runs
            if (attempts := jobs.by_request.get(run.request_id)) is not None
        }),
        _train_history(settled, jobs),
        MappingProxyType(evaluations),
        tuple(evaluation_problems),
    )


def _evaluation_row(
    settled: SettledProjectPlan, run: PlannedRun, jobs: PlanJobAttempts, reader: PlanJobReader,
) -> tuple[StudioSavedEvaluationRow, StudioProjectWindowMapProblem | None]:
    key = StudioSavedTrainKey.model_validate(run.key, from_attributes=True)
    trials = []
    problem = None
    try:
        params, subject = reader.evaluation_parameters(settled.plan, run)
        trials = list(params.trials)
        if subject.request_id not in jobs.checkpoint_producers:
            raise CannotRun("/inputs/subject", EVALUATION_SUBJECT_REQUIRED)
        if isinstance(subject.select, BatchSetMember) and not set(subject.select.batches) <= set(
            jobs.checkpoint_catalogs.get(subject.request_id, ())
        ):
            raise CannotRun("/inputs/subject/checkpoint", "admitted Train catalogue omits requested batches")
    except CannotRun as error:
        problem = StudioProjectWindowMapProblem(
            code="cannot_run", path=error.path, detail=str(error),
            variant=key.variant, step=key.step, row=key.row, document=None,
        )
    attempts = jobs.by_request.get(run.request_id, ())
    success = next((attempt for attempt in reversed(attempts) if (
        attempt.successful_receipt is not None
        and (attempt.checkpoint_source is not None or attempt.checkpoint_sources)
        and attempt.output_size_bytes is not None
    )), None)
    result = None
    if success is not None:
        assert success.successful_receipt is not None and success.output_size_bytes is not None
        result = StudioSavedEvaluationResult(
            job_id=success.job_id, receipt=success.successful_receipt,
            checkpoint_source=success.checkpoint_source, size_bytes=success.output_size_bytes,
            checkpoint_sources=list(success.checkpoint_sources) or None,
        )
    status, reason = saved_request_status(attempts, success=success, problem=problem.detail if problem else None)
    return StudioSavedEvaluationRow(
        key=key, request_id=run.request_id,
        status=status, reason=reason,
        can_run=problem is None, trials=trials, result=result,
    ), problem


def _train_history(
    settled: SettledProjectPlan, jobs: PlanJobAttempts
) -> Mapping[RunKey, TrainRowHistory]:
    ref_rows = _history_rows(settled, jobs)
    successful = {
        request: tuple(
            attempt for attempt in attempts
            if attempt.successful_receipt is not None and attempt.binding_sha256 is not None
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
            problem=run.admission_problem.detail if run.admission_problem else None,
            changed_path=change.path if change else None,
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
        receipt=attempt.successful_receipt, progress=attempt.progress,
    )


def _definition_change(
    settled: SettledProjectPlan,
    run: PlannedRun,
    previous: StudioSavedTrainResult,
    attempt: PlanJobAttempt,
) -> StudioSavedTrainChange:
    root = settled.store_path
    relative = training_binding_path(root, attempt.job_id).relative_to(root).as_posix()
    try:
        with completion_file(root, relative) as (stream, _):
            raw = stream.read()
        if sha256(raw).hexdigest() != attempt.binding_sha256:
            raise ValueError("Previous Train binding changed")
        launch = BoundStudioTrainingJobInput.model_validate_json(raw)
        binding = launch.canonical_request
        if binding is None or (launch.project_id, launch.request_id, launch.manifest_root) != (
            settled.project_id, previous.request_id, root,
        ):
            raise ValueError("Previous Train binding differs from its saved request")
        store = ImmutableArtifactBlobProvider(root)
        docs = {ref.sha256: ref for ref in binding.documents}
        old = decode_request(store.get_object(binding.request), lambda digest: store.get_object(docs[digest]))
        if old.id != previous.request_id:
            raise ValueError("Previous Train object has a different request ID")
    except (OSError, ValueError, KeyError):
        return StudioSavedTrainChange(path="request_id", before=previous.request_id, after=run.request_id)
    new = decode_request(settled.plan.requests[run.request_id], settled.plan.documents.__getitem__)
    change = first_request_change(old, new)
    return StudioSavedTrainChange(path=change.path, before=change.left, after=change.right)
