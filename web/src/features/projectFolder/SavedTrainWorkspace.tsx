import { useEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactNode } from 'react';
import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { fetchSavedTrainRuns, runSavedTrain } from '@/api/jobs';
import { apiErrorMessage } from '@/api/request';
import { StudioTrainResults } from '@/components/activity/StudioTrainResults';
import { studioCountOf, studioName, studioStep } from '@/components/activity/studioVocabulary';
import { HeaderActivity } from '@/components/layout/HeaderActivity';
import { StudioErrorBoundary } from '@/components/layout/StudioErrorBoundary';
import { bindSweepFold, StudioPipelineMap, StudioPipelineMapDrawer, type PipelineMapStepSource } from '@/components/pipeline/StudioPipelineMap';
import { canvasViewportCommands, type CanvasViewport } from '@/components/canvas/StudioCanvas';
import { StudioRunControls, StudioRunDetails, StudioRuns } from '@/components/pipeline/StudioRuns';
import { Dialog } from '@/components/ui/Dialog';
import { humanizeLabel } from '@/components/values/ValueSpecField';
import { foldedMapSweep, orderedMapSweeps, type PipelineCardPresentation } from '@/components/pipeline/pipelineMapLayout';
import type { PlanView } from '@/features/pipeline/planView';
import { selectRun, selectStep, selectSweepValue, selectedRuns, type PipelineSelection, type PipelineSelectionState } from '@/features/pipeline/selection';
import { ProjectWindow } from '@/features/projectWindow/ProjectWindow';
import { StudioProblems } from '@/features/projectWindow/StudioProblems';
import { bindStudioCommand, commandAttributes, jobAllowsCommand, type StudioCommand } from '@/features/projectWindow/studioCommands';
import {
  projectWindowSlotView, showProjectWindowView, useProjectWindowState, type ProjectWindowSlot, type ProjectWindowView,
} from '@/features/projectWindow/projectWindowState';
import type {
  PresentationDocument, PresentationPosition, StudioActivityJob, StudioProjectWindowMapProblem, StudioSavedTrainRuns,
} from '@/generated/studioContracts';
import { useStudioConnection } from '@/hooks/useStudioConnection';
import { useStudioJobActions, useStudioJobs } from '@/hooks/useStudioJobs';
import { directTrainStep } from './authoredEdits';
import { useManagedProjectDetails, useSavedEvaluationScene, type NamedDocumentScope } from './managedProjectDetails';
import { useModelSelectionStore } from './model/modelSelection';
import { createCheckpointTrain } from './createCheckpointTrain';
import { createAnalyze, createReport, isReportFigureSource, type ReportFigureSource } from './createManagedStep';
import { canCreateFirstTrain, createFirstTrain } from './createFirstTrain';
import { openProjectEditor } from './openProjectEditor';
import type { ProjectEditController } from './projectEditController';
import { projectWindowPlan, planVariants, savedRunId, type SavedTrainRow as SavedRow } from './projectWindowPlan';
import { problemReason, problemTarget, type ProblemReveal } from './projectWindowProblems';
import { settledWindowSelection, useProjectWindowSelection } from './projectWindowSelection';
import { StudioTrainAuthoringForm } from './StudioTrainAuthoringForm';
import { useManagedAnalyze } from './useManagedAnalyze';
import { useManagedReport } from './useManagedReport';

type EditorSession = Awaited<ReturnType<typeof openProjectEditor>>;
type WindowPanel = { kind: 'activity' } | {
  kind: 'problems';
  scope: { step: string; variant: string | null } | null;
  opener: HTMLElement | null;
};
export interface ProjectWindowCommandScope {
  readonly editor: ProjectEditController;
  readonly selection: PipelineSelectionState;
  readonly variantId: string | null;
  readonly plan: PlanView;
  readonly projection: StudioSavedTrainRuns | undefined;
  onSelectionChange(selection: PipelineSelectionState): void;
  selectCreatedStep(stepId: string, view?: 'model' | 'runs' | 'analysis' | 'document'): void;
}

export type ProjectWindowCommands = {
  onNewStep?: (source: PipelineMapStepSource, scope: ProjectWindowCommandScope) => void | Promise<void>;
  onNewSweep?: (stepId: string, scope: ProjectWindowCommandScope) => void;
};

async function checkpointNewStep(source: PipelineMapStepSource, scope: ProjectWindowCommandScope) {
  if (!scope.projection) return;
  const id = await createCheckpointTrain(scope.editor, source, scope.plan, scope.projection);
  scope.selectCreatedStep(id);
}

const viewsByKind = {
  train: ['runs', 'model', 'task', 'objective', 'training', 'scene', 'results'],
  evaluate: ['runs', 'model', 'task', 'scene', 'results'],
  analyze: ['runs', 'analysis', 'figures'],
  report: ['runs', 'document'],
} as const satisfies Record<string, readonly ProjectWindowView[]>;
const emptyViews: readonly ProjectWindowView[] = [];
const control = 'rounded px-2 py-1 hover:bg-shell-hover focus-visible:outline-2';
const { Icon: TrainIcon, tint: trainTint, name: trainName } = studioStep('train');
const noView = <p className="p-3 text-sm text-shell-muted" role="status">{studioName('unavailable')}</p>;

function TrainReceipts({ row }: { row: SavedRow }) {
  const results = [
    ...(row.result ? [{ label: 'Current result', result: row.result }] : []),
    ...(row.previous ? [{ label: 'Previous result', result: row.previous }] : []),
  ];
  if (!results.length) return null;
  return (
    <details className="mt-3 rounded border border-shell-divider p-3 text-sm">
      <summary className="cursor-pointer font-medium">
        {results.length === 2 ? 'Compare receipts' : results[0].label}
      </summary>
      {row.change && (
        <p className="mt-2 break-all">
          <code>{row.change.path}</code>{' '}
          <code>{JSON.stringify(row.change.before)}</code> → <code>{JSON.stringify(row.change.after)}</code>
        </p>
      )}
      <div className="mt-2 grid gap-3 sm:grid-cols-2">
        {results.map(({ label, result }) => (
          <section key={label} aria-label={label} className="min-w-0">
            <h3 className="font-medium">{label}</h3>
            <dl className="space-y-1">
              <div><dt>Request</dt><dd className="break-all"><code>{result.request_id}</code></dd></div>
              <div><dt>Receipt</dt><dd className="break-all"><code>{result.receipt.manifest_id}</code></dd></div>
              <div><dt>SHA-256</dt><dd className="break-all"><code>{result.receipt.manifest_sha256}</code></dd></div>
              {result.progress && <div><dt>Completed updates</dt><dd>{result.progress.completed_updates}</dd></div>}
            </dl>
          </section>
        ))}
      </div>
    </details>
  );
}

export default function StudioProjectWorkspace({
  projectId, onNewStep = checkpointNewStep, onNewSweep,
}: { projectId: string } & ProjectWindowCommands) {
  const [load, setLoad] = useState<{ session: EditorSession } | { error: string }>();
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    const lifetime = new AbortController();
    setLoad(undefined);
    void openProjectEditor(projectId, lifetime.signal).then(
      (session) => { if (!lifetime.signal.aborted) setLoad({ session }); },
      (cause: unknown) => { if (!lifetime.signal.aborted) setLoad({ error: apiErrorMessage(cause) }); },
    );
    return () => lifetime.abort();
  }, [projectId, retry]);

  if (!load) return <main aria-busy="true" />;
  if ('error' in load) return (
    <main className="p-4"><p role="alert">{load.error}</p>
      <button type="button" className={control} onClick={() => setRetry((attempt) => attempt + 1)}>{studioName('retry')}</button>
    </main>
  );
  return <ManagedProjectWindow key={projectId} session={load.session} onNewStep={onNewStep} onNewSweep={onNewSweep} />;
}

function ManagedProjectWindow({ session, onNewStep, onNewSweep }: { session: EditorSession } & ProjectWindowCommands) {
  const { editor } = session;
  const snapshot = useSyncExternalStore(editor.subscribe, editor.getSnapshot);
  const { saved, configuration, pending, outside } = snapshot;
  const jobs = useStudioJobs();
  const connection = useStudioConnection();
  const actions = useStudioJobActions();
  const [windowState, setWindowState] = useProjectWindowState(saved.project_id);
  const modelWindow = useModelSelectionStore();
  const mapViewport = useRef<CanvasViewport>(null);
  const selection = useProjectWindowSelection(Object.keys(saved.project.steps ?? {})[0] ?? null);
  const [commandError, setCommandError] = useState<string>();
  const [busy, setBusy] = useState(false);
  const [panel, setPanel] = useState<WindowPanel | null>(null);
  const [focusRequest, requestSlotFocus] = useState(0);
  const [pivotSlots, setPivotSlots] = useState<Partial<Record<ProjectWindowSlot, string>>>({});
  const [problemReveal, setProblemReveal] = useState<{
    revision: string;
    variant: string | null;
    runId: string | null;
    reveal: ProblemReveal;
  }>();
  const [excludedUsers, setExcludedUsers] = useState<Readonly<Record<string, readonly string[]>>>({});
  const [focusCreatedStep, setFocusCreatedStep] = useState<string>();
  const [focusFirstModel, setFocusFirstModel] = useState<string>();
  const [firstStepRevision, setFirstStepRevision] = useState<string>();
  const [reportChoices, setReportChoices] = useState<{ revision: string; figures: ReportFigureSource[] }>();
  const inFlight = useRef(false);
  const [sweepStep, setSweepStep] = useState<string>();
  const submission = useRef<{ intent: string; token: string }>();
  const projection = useQuery({
    queryKey: ['saved-train-runs', saved.project_id, saved.revision, jobs.data?.cursor],
    queryFn: ({ signal }) => fetchSavedTrainRuns(saved.project_id, saved.revision, signal),
    placeholderData: keepPreviousData,
    gcTime: 0,
    retry: false,
  });
  const data = projection.data?.revision === saved.revision ? projection.data : undefined;
  const variants = useMemo(() => planVariants(projection.data), [projection.data]);
  const variant = selection.variantId ?? variants[0] ?? null;
  const missingVariant = selection.variantId !== null && !variants.includes(selection.variantId);
  const primaryVariant = variant === (variants[0] ?? null);
  const plan = useMemo(() => projectWindowPlan(saved.project, projection.data, variant),
    [saved.project, projection.data, variant]);
  const step = plan.steps.find(({ id }) => id === selection.stepId);
  const state = data ? settledWindowSelection(plan, selection.pipeline) : selection.pipeline;
  const latestState = useRef(state);
  latestState.current = state;
  const singleStep = state.selection?.kind === 'steps' && state.selection.stepIds.length === 1 && state.selection.stepIds[0] === step?.id;
  const wholeStep = singleStep && primaryVariant;
  const canAuthorSweep = (id: string) => primaryVariant && directTrainStep(saved.project, id) !== null;
  const authored = step && saved.project.steps[step.id];
  const checkpoint = authored && authored.inputs?.checkpoint;
  const linked = wholeStep && !saved.project.sweep && authored?.kind === 'train' &&
    'step' in authored.base && authored.base.row == null && checkpoint &&
    'step' in checkpoint && checkpoint.step === authored.base.step && checkpoint.checkpoint === 'last';
  const authoring = step && (canAuthorSweep(step.id) || linked) && state.selection ? {
    editor, stepId: step.id, step, selection: state.selection,
    onChooseSweepValue: (sweepId: string, valueId: string, extend: boolean) => {
      selection.select(selectSweepValue(latestState.current, step.id, sweepId, valueId, extend));
    },
  } : null;
  const availableViews = step ? viewsByKind[step.kind] : emptyViews;
  const primaryView = projectWindowSlotView(windowState, 'primary', availableViews);
  const secondaryView = projectWindowSlotView(windowState, 'secondary', availableViews);
  const focusSlot = secondaryView && windowState.focusedSlot === 'secondary' ? 'secondary' : 'primary';
  const focusView = focusSlot === 'secondary' ? secondaryView : primaryView;
  const revealed = problemReveal?.revision === saved.revision && problemReveal.variant === variant &&
    problemReveal.reveal.problem.step === step?.id && problemReveal.runId === (state.focus?.runId ?? null)
    ? problemReveal.reveal : undefined;
  function documentScope(field: 'graph' | 'task'): NamedDocumentScope | undefined {
    const document = step?.documentIds?.[`payload.${field}`];
    if (!document || !step || !singleStep || !data || step.kind !== 'train') return undefined;
    const localVariant = Boolean(saved.project.sweep && !primaryVariant && field === 'graph');
    if (!localVariant && (!wholeStep || saved.project.sweep)) return undefined;
    const users = new Set(plan.steps.filter(({ documentUses }) => documentUses?.includes(document)).map(({ id }) => id));
    if (!users.has(step.id)) return undefined;
    const key = JSON.stringify([step.id, document, variant]);
    if (localVariant) return { document, key, users, lit: users, localVariant };
    const excluded = excludedUsers[key] ?? [];
    return { document, key, users, lit: new Set([...users].filter((id) => id === step.id || !excluded.includes(id))) };
  }
  const documentScopes = { graph: documentScope('graph'), task: documentScope('task') };
  const activeDocumentScope = windowState.mode === 'map' ? undefined
    : focusView === 'model' ? documentScopes.graph : focusView === 'task' ? documentScopes.task : undefined;
  function chooseMapStep(id: string, extend: boolean) {
    if (extend && activeDocumentScope && activeDocumentScope.users.size > 1) {
      if (activeDocumentScope.localVariant || id === step?.id || !activeDocumentScope.users.has(id)) return;
      const { key } = activeDocumentScope;
      setExcludedUsers((current) => {
        const ids = current[key] ?? [];
        return { ...current, [key]: ids.includes(id) ? ids.filter((excluded) => excluded !== id) : [...ids, id] };
      });
      return;
    }
    if (!extend) setExcludedUsers({});
    setProblemReveal(undefined);
    selection.select(selectStep(state, id, extend));
  }
  const row = data?.rows.find(({ key }) => key.step === step?.id && savedRunId(key) === state.focus?.runId);
  const attempt = row?.attempts.at(-1);
  const ready = Boolean(data && !pending && !outside);
  const canAuthorContinuation = ready && !saved.project.sweep;
  useEffect(() => {
    if (!ready || !focusFirstModel || step?.id !== focusFirstModel) return;
    requestSlotFocus((count) => count + 1);
    setFocusFirstModel(undefined);
  }, [ready, focusFirstModel, step?.id]);
  const names = useMemo(() => new Map([[saved.project_id, configuration.project]]), [saved.project_id, configuration.project]);

  const runTarget: PipelineSelection = state.selection ?? { kind: 'steps', stepIds: plan.steps.map(({ id }) => id) };
  const runIds = new Set(plan.steps.flatMap((item) => selectedRuns(item, runTarget).map(({ id }) => id)));
  const runRows = data?.rows.filter(({ key }) => runIds.has(savedRunId(key))) ?? [];
  const evaluationRows = data?.window_map?.evaluations?.filter(({ key }) => runIds.has(savedRunId(key))) ?? [];
  const sceneRows = evaluationRows.filter(({ key }) => key.step === step?.id);
  const selectedEvaluation = sceneRows.find(({ key }) => savedRunId(key) === state.focus?.runId)
    ?? (sceneRows.length === 1 ? sceneRows[0] : undefined);
  const runnableTrainRows = runRows.filter(({ status }) => status !== 'cannot_run');
  const runnableRows = state.selection
    ? [...runnableTrainRows, ...evaluationRows.filter(({ can_run }) => can_run)] : runnableTrainRows;
  const evaluationReplay = useSavedEvaluationScene(saved.project_id, saved.revision, selectedEvaluation, windowState.mode !== 'map');
  const definitionRows = data?.window_map?.rows.filter(({ key }) => key.step === step?.id && key.variant === variant) ?? [];
  const selectedDefinitionRows = definitionRows.filter(({ key }) => runIds.has(savedRunId(key)));
  const namedRun = selectedDefinitionRows.find(({ key }) => savedRunId(key) === state.focus?.runId)?.key ??
    selectedDefinitionRows[0]?.key ?? definitionRows[0]?.key;
  const selectedSweepValues = new Map<string, Set<string>>();
  if (state.selection && state.selection.kind !== 'steps') {
    for (const row of selectedDefinitionRows) {
      for (const [axis, { value_id }] of Object.entries(row.coordinates)) {
        const values = selectedSweepValues.get(axis) ?? new Set<string>();
        values.add(value_id);
        selectedSweepValues.set(axis, values);
      }
    }
  }
  const evaluated = step?.kind === 'evaluate' && namedRun && state.selection;
  const modelSource = step?.kind === 'evaluate' ? plan.steps.find(({ id }) =>
    id === step.inputs.find(({ id }) => id === 'subject')?.sourceStepId)?.documentIds?.['payload.graph'] : undefined;
  const modelProblems = data?.window_map?.problems.filter((problem) => problem.model_address &&
    (problem.step === step?.id || step?.kind === 'evaluate' && problem.document === modelSource) &&
    (problem.variant === null || problem.variant === variant) &&
    (!problem.row || !namedRun || problem.step !== step?.id || problem.row === namedRun.row));
  const axis = evaluated && state.selection?.kind === 'values' && state.selection.terms.length === 1
    ? state.selection.terms[0] : undefined;
  const evaluateScope = evaluated && (state.selection?.kind === 'steps'
    ? singleStep ? (primaryVariant ? { kind: 'step' } as const : { kind: 'variant', variant: namedRun.variant } as const) : undefined
    : axis?.valueIds.length === 1 ? { kind: 'axis_value', axis: axis.sweepId, value: axis.valueIds[0] } as const
      : selectedDefinitionRows.length ? { kind: 'rows', rows: selectedDefinitionRows.map(({ key }) => key.row) } as const
        : undefined);
  function saveElementPosition(document: string, node: string, position: PresentationPosition) {
    const { outside, saved } = editor.getSnapshot();
    const current = (outside ?? saved).presentation ?? {};
    const prior = current.documents?.[document];
    return session.savePresentation({
      ...current, documents: { ...current.documents, [document]: {
        ...prior, positions: { ...prior?.positions, [node]: position },
      } },
    });
  }
  const detailInputs = {
    editor, step, stepId: step?.id ?? null, focus: state.focus, selection: state.selection,
    variantId: variant, evaluation: selectedEvaluation, evaluationReplay, onChooseSweepValue: authoring?.onChooseSweepValue,
    namedRun, documentScopes, modelWindow, evaluateScope: evaluateScope || undefined, modelSource, modelProblems, selectedSweepValues,
    fieldRows: state.selection && state.selection.kind !== 'steps' ? selectedDefinitionRows.map(({ key }) => key.row) : undefined,
  };
  function setModelPath(slot: ProjectWindowSlot, document: string, path: string[]) {
    setWindowState((current) => ({
      ...current, modelPaths: { ...current.modelPaths, [slot]: { document, path } },
    }));
  }
  const primary = useManagedProjectDetails({ ...detailInputs, activeView: windowState.mode === 'map' ? 'hidden' : primaryView ?? 'hidden',
    problemReveal: focusSlot === 'primary' ? revealed : undefined, modelPath: windowState.modelPaths?.primary,
    setModelPath: (document, path) => setModelPath('primary', document, path) });
  const secondary = useManagedProjectDetails({ ...detailInputs, activeView: windowState.mode === 'map' ? 'hidden' : secondaryView ?? 'hidden',
    problemReveal: focusSlot === 'secondary' ? revealed : undefined, modelPath: windowState.modelPaths?.secondary,
    setModelPath: (document, path) => setModelPath('secondary', document, path) });
  const focusedModel = windowState.mode !== 'map' && focusView === 'model'
    ? (focusSlot === 'primary' ? primary : secondary).model : undefined;
  const analyze = useManagedAnalyze({
    editor, key: step?.kind === 'analyze' ? namedRun : undefined, projection: data, wholeStep,
    focusedView: windowState.mode === 'map' ? null : focusView, reveal: revealed, savePosition: saveElementPosition,
  });
  const report = useManagedReport({
    editor, snapshot, plan, projection: data, variant, stepId: step?.kind === 'report' ? step.id : undefined,
    wholeStep, focused: windowState.mode !== 'map' && focusView === 'document',
  });
  const jobsById = new Map(jobs.data?.jobs.map((job) => [job.job_id, job]));
  const selectedJobs = state.selection ? runRows.flatMap((item) => {
    const latest = item.attempts.at(-1);
    const job = latest && jobsById.get(latest.job_id);
    return job ? [job] : [];
  }) : [];
  const exactRow = row ?? (runRows.length === 1 && state.selection ? runRows[0] : undefined);
  const exactJob = exactRow?.attempts.at(-1);
  const resumable = exactJob && jobsById.get(exactJob.job_id);

  async function perform(work: () => Promise<void>) {
    if (inFlight.current) return;
    inFlight.current = true;
    setBusy(true);
    setCommandError(undefined);
    try {
      await work();
    } catch (cause) {
      setCommandError(apiErrorMessage(cause));
    } finally {
      inFlight.current = false;
      setBusy(false);
    }
  }

  function run() {
    if (!ready || !data || !runnableRows.length) return;
    const keys = runnableRows.map(({ key }) => key);
    void perform(async () => {
      if (await editor.flush() !== data.revision) throw new Error('The project changed before the run was submitted.');
      const project = editor.getSnapshot().saved.project_id;
      const intent = JSON.stringify([data.revision, keys]);
      const token = submission.current?.intent === intent ? submission.current.token : crypto.randomUUID();
      submission.current = { intent, token };
      const result = await runSavedTrain(project, { revision: data.revision, run_keys: keys, token });
      submission.current = undefined;
      const refusal = result.items.find(({ disposition }) => disposition === 'rejected');
      if (refusal?.reason) setCommandError(refusal.reason);
      actions.refresh();
    });
  }

  function stop(jobIds: string[], mode: 'checkpoint' | 'force') {
    void perform(async () => {
      const result = await actions.stop(jobIds, mode);
      const failure = result.find(({ error }) => error)?.error;
      if (failure) setCommandError(failure.message);
    });
  }

  const problemPanel = panel?.kind === 'problems' ? panel : null;
  const problemScope = problemPanel?.scope;
  const problems = data?.window_map?.problems ?? [];
  const problemSteps = new Set(problems.flatMap((problem) =>
    problem.step && (problem.variant === null || problem.variant === variant) ? [problem.step] : []));
  const shownProblems = problemScope ? problems.filter((problem) =>
    problem.step === problemScope.step &&
    (problem.variant === null || problem.variant === problemScope.variant)) : problems;
  function openProjectProblems(opener: HTMLElement | null = null) {
    setPanel({ kind: 'problems', scope: null, opener });
  }
  function addressedStep(problem: StudioProjectWindowMapProblem) {
    return data?.window_map?.steps.find(({ step_id, variant: scope }) =>
      step_id === problem.step && scope === (problem.variant ?? variant));
  }
  function revealProblem(problem: StudioProjectWindowMapProblem) {
    const destination = addressedStep(problem);
    if (!destination) return;
    const { step_id: stepId, variant: scope, kind } = destination;
    const row = data?.window_map?.rows.find(({ key }) =>
      key.step === stepId && key.row === problem.row && key.variant === scope);
    const target = problemTarget(problem);
    const view = kind === 'analyze' ? 'analysis' : kind === 'report' ? 'document'
      : viewsByKind[kind].some((item) => item === target.view) ? target.view : 'runs';
    const runId = row ? savedRunId(row.key) : null;
    selection.select(runId
      ? selectRun(state, { stepId, runId })
      : selectStep(state, stepId), scope);
    setExcludedUsers({});
    setProblemReveal({ revision: saved.revision, variant: scope, runId, reveal: { problem, target: { ...target, view } } });
    setWindowState((current) => showProjectWindowView(current, view, focusSlot));
    requestSlotFocus((count) => count + 1);
    setPanel(null);
  }
  function showRun(match: SavedRow) {
    const target = { stepId: match.key.step, runId: savedRunId(match.key) };
    selection.select(selectRun(state, target), match.key.variant);
    setWindowState((current) => showProjectWindowView(current, 'results'));
    requestSlotFocus((count) => count + 1);
  }
  async function openActivityRun(job: StudioActivityJob) {
    const match = data?.rows.find((item) => item.attempts.some(({ job_id }) => job_id === job.job_id));
    if (!match) return false;
    showRun(match);
    return true;
  }

  const present = saved.presentation;
  const presentation = useMemo(() => Object.fromEntries(Object.entries(present?.steps ?? {}).map(([id, value]) => [
    id, { sweepOrder: value.sweep_order, foldedSweepId: value.folded_sweep_id },
  ])), [present]);
  async function changePresentation(stepId: string, next: PipelineCardPresentation) {
    const current = editor.getSnapshot().saved.presentation ?? {};
    const updated: PresentationDocument = {
      ...current, steps: { ...current.steps, [stepId]: {
        ...current.steps?.[stepId], sweep_order: [...(next.sweepOrder ?? [])], folded_sweep_id: next.foldedSweepId,
      } },
    };
    try {
      await session.savePresentation(updated);
    } catch (error) {
      setCommandError(apiErrorMessage(error));
    }
  }

  const projectFailure = snapshot.error ?? projection.error ?? connection.error;
  const failure = commandError ?? (projectFailure ? apiErrorMessage(projectFailure) : connection.isError ? studioName('unavailable') : null);
  function retry() {
    setCommandError(undefined);
    if (snapshot.error && snapshot.pending) void editor.flush().catch((error) => setCommandError(apiErrorMessage(error)));
    connection.refetch();
    void projection.refetch();
  }
  async function moveHistory(action: 'undo' | 'redo') {
    try {
      const changed = await editor[action]();
      if (changed?.length) {
        const address = changed[0];
        modelWindow.setState({ reveal: address, selection: changed });
        if (focusView !== 'model') setWindowState((current) => showProjectWindowView(current, 'model', focusSlot));
      }
      await editor.flush();
    } catch (error) {
      setCommandError(apiErrorMessage(error));
    }
  }

  const commandScope: ProjectWindowCommandScope = {
    editor, selection: state, variantId: variant, onSelectionChange: selection.select,
    plan, projection: data,
    selectCreatedStep: (id, view = 'runs') => {
      selection.select(selectStep(state, id));
      if (view === 'model') setFocusFirstModel(id);
      if (view === 'runs') setFocusCreatedStep(id);
      if (view === 'analysis' || view === 'document') requestSlotFocus((count) => count + 1);
      setWindowState((current) => showProjectWindowView(
        view === 'runs' ? { ...current, detailsDrawerOpen: true } : current, view,
      ));
    },
  };

  function chooseFirstTrain() {
    if (!firstStepRevision) return;
    setFirstStepRevision(undefined);
    void perform(async () => {
      const id = await createFirstTrain(commandScope.editor, firstStepRevision);
      commandScope.selectCreatedStep(id, 'model');
    });
  }

  function chooseReportFigure(revision: string, figure: ReportFigureSource) {
    setReportChoices(undefined);
    void perform(async () => {
      const id = await createReport(editor, revision, figure, humanizeLabel(figure.name));
      commandScope.selectCreatedStep(id, 'document');
    });
  }

  function newStepCommand(source?: PipelineMapStepSource, subject?: string) {
    if (!source) {
      if ((step?.kind === 'evaluate' || step?.kind === 'analyze') &&
          (state.selection?.kind !== 'steps' || state.selection.stepIds.length === 1)) {
        return newStepCommand({
          stepId: step.id,
          selection: state.selection && 'stepId' in state.selection && state.selection.stepId === step.id
            ? state.selection : null,
        });
      }
      if (!canCreateFirstTrain(saved.project)) return undefined;
      return bindStudioCommand('newStep', () => setFirstStepRevision(saved.revision), { enabled: ready && !busy });
    }
    const sourceStep = plan.steps.find(({ id }) => id === source.stepId);
    if (sourceStep?.kind === 'evaluate' && !saved.project.sweep && variant === 'default') {
      const rows = source.selection ? selectedRuns(sourceStep, source.selection) : sourceStep.runs;
      const evaluation = rows.length === 1 && data?.window_map?.evaluations?.find((row) =>
        row.result && savedRunId(row.key) === rows[0].id);
      if (!evaluation) return undefined;
      return bindStudioCommand('newStep', () => { void perform(async () => {
        const id = await createAnalyze(editor, saved.revision, evaluation.key);
        commandScope.selectCreatedStep(id, 'analysis');
      }); }, { enabled: ready && !busy, name: 'analyze', subject });
    }
    if (sourceStep?.kind === 'analyze' && !saved.project.sweep && variant === 'default') {
      const rows = new Set((source.selection ? selectedRuns(sourceStep, source.selection) : sourceStep.runs).map(({ id }) => id));
      const figures = (data?.window_map?.analysis_elements ?? []).filter(isReportFigureSource)
        .filter(({ key }) => rows.has(savedRunId(key)));
      if (!figures.length) return undefined;
      const selected = step?.id === sourceStep.id && namedRun && figures.find(({ key, name }) =>
        savedRunId(key) === savedRunId(namedRun) && name === analyze.selectedFigure);
      return bindStudioCommand('newStep', () => {
        const choice = selected || (figures.length === 1 ? figures[0] : undefined);
        if (choice) chooseReportFigure(saved.revision, choice);
        else setReportChoices({ revision: saved.revision, figures });
      }, { enabled: ready && !busy, name: 'report', subject });
    }
    if (!onNewStep || sourceStep?.kind !== 'train') return undefined;
    const runs = source.selection ? selectedRuns(sourceStep, source.selection) : sourceStep.runs;
    const runnable = runs.some(({ status }) => status !== 'cannot_run');
    return bindStudioCommand('newStep', () => { void perform(async () => { await onNewStep(source, commandScope); }); },
      { enabled: canAuthorContinuation && !busy && runnable, subject });
  }

  function newSweepCommand(stepId: string, subject?: string) {
    if (!onNewSweep && !canAuthorSweep(stepId)) return undefined;
    return bindStudioCommand('newSweep', () => {
      if (onNewSweep) onNewSweep(stepId, commandScope);
      else setSweepStep(stepId);
    }, { enabled: ready && !busy, subject });
  }

  const mapProps = {
    plan, state, problemSteps,
    onSelectionChange: (value: PipelineSelectionState) => {
      setProblemReveal(undefined);
      selection.select(value);
    },
    onProblems: (step: string, opener: HTMLButtonElement) => setPanel({ kind: 'problems', scope: { step, variant }, opener }),
    newStepCommand, newSweepCommand, onChooseStep: chooseMapStep,
    documentScope: activeDocumentScope && activeDocumentScope.users.size > 1 ? activeDocumentScope : undefined,
  };
  const activity = bindStudioCommand('activity', () => setPanel((current) => current?.kind === 'activity' ? null : { kind: 'activity' }));
  const problemCommand = bindStudioCommand('problems', () => openProjectProblems());
  const retryCommand = bindStudioCommand('retry', retry);
  const runCommand = bindStudioCommand('run', run, { enabled: ready && !busy && runnableRows.length > 0 });
  const runCommands: StudioCommand[] = [runCommand];
  if (step?.kind === 'train') {
    for (const [id, mode] of [['stop', 'checkpoint'], ['forceStop', 'force']] as const) {
      const eligible = selectedJobs.filter((job) => jobAllowsCommand(job, id)).map(({ job_id }) => job_id);
      if (id === 'forceStop' && !eligible.length) continue;
      runCommands.push(bindStudioCommand(id, () => stop(eligible, mode), { enabled: !busy && eligible.length > 0 }));
    }
    if (resumable && jobAllowsCommand(resumable, 'resume')) {
      runCommands.push(bindStudioCommand('resume', () => {
        void perform(async () => { await actions.resume(resumable.job_id); });
      }, { enabled: !busy }));
    }
    runCommands.push(bindStudioCommand('openRun', () => { if (exactRow) showRun(exactRow); }, { enabled: !!exactJob }));
    const continuation = newStepCommand({
      stepId: step.id, selection: state.selection?.kind === 'values' || state.selection?.kind === 'rows' ? state.selection : null,
    });
    if (continuation) runCommands.push(continuation);
  }
  function pivotCommand(slot: ProjectWindowSlot) {
    if (!step || step.sweeps.length < 2) return undefined;
    return bindStudioCommand('pivot', () => setPivotSlots((current) => ({
      ...current, [slot]: current[slot] === step.id ? undefined : step.id,
    })));
  }
  const selectedAxis = state.selection?.kind === 'values' && state.selection.terms.length === 1
    ? step?.sweeps.find(({ id }) => state.selection?.kind === 'values' && id === state.selection.terms[0].sweepId) : undefined;
  const cardPresentation = step ? presentation[step.id] ?? {} : {};
  const folded = step && selectedAxis ? foldedMapSweep(step, orderedMapSweeps(step, cardPresentation), cardPresentation) : undefined;
  const commands = [
    ...runCommands,
    ...(focusedModel?.commands ?? []),
    ...analyze.commands,
    ...report.commands,
    newStepCommand(),
    step && selectedAxis && bindSweepFold(step.id, selectedAxis, folded?.id === selectedAxis.id, cardPresentation, changePresentation),
    step && newSweepCommand(step.id),
    windowState.mode !== 'map' && focusView === 'runs' && pivotCommand(focusSlot),
    bindStudioCommand('undo', () => { void moveHistory('undo'); }, { enabled: !!snapshot.undoName && !busy && !snapshot.saving }),
    bindStudioCommand('redo', () => { void moveHistory('redo'); }, { enabled: !!snapshot.redoName && !busy && !snapshot.saving }),
    bindStudioCommand('clearFocus', () => selection.select({ ...state, focus: null }), { enabled: !!state.focus }),
    activity, problemCommand, failure && retryCommand,
  ].filter((command): command is StudioCommand => !!command);

  return (
    <ProjectWindow
      state={windowState} onStateChange={setWindowState} projectName={configuration.project}
      saveStatus={pending || snapshot.error ? studioName('edited') : undefined}
      availableViews={availableViews}
      commands={commands} mapCommands={canvasViewportCommands(() => mapViewport.current, !!data && plan.steps.length > 0)}
      focusRequest={focusRequest}
      onBack={selection.back} onForward={selection.forward}
      map={<StudioErrorBoundary region={studioName('pipelineMap')}><StudioPipelineMap {...mapProps}
        activeStepId={selection.stepId} viewportRef={mapViewport} fitReady={!!data}
        presentation={presentation} onPresentationChange={changePresentation} /></StudioErrorBoundary>}
      mapDrawer={<StudioPipelineMapDrawer {...mapProps} />}
      compactMap={<StudioPipelineMap {...mapProps} mode="line" />}
      variant={(variants.length > 1 || missingVariant) && <select aria-label={studioName('variant')}
        value={variant ?? ''} disabled={!data}
        onChange={(event) => selection.setVariant(event.target.value)}>
        {missingVariant && variant && <option value={variant} disabled>{variant}</option>}
        {variants.map((id) => <option key={id} value={id}>{id}</option>)}
      </select>}
      overlay={<>
        {focusedModel?.overlay}
        {analyze.overlay}
        {report.overlay}
        {reportChoices?.revision === saved.revision && <Dialog ariaLabel={studioName('report')} manageFocus
          onClose={() => setReportChoices(undefined)}
          panelClassName="grid w-[min(24rem,90vw)] gap-1 rounded-shell-control border border-shell-divider bg-shell-chrome p-2 text-shell-foreground shadow-xl">
          {reportChoices.figures.map((figure) => <button type="button" className={control}
            key={JSON.stringify([figure.key, figure.name])}
            onClick={() => chooseReportFigure(reportChoices.revision, figure)}>
            {figure.name}{figure.key.row !== 'default' && ' · ' + figure.key.row}
          </button>)}
        </Dialog>}
        {firstStepRevision && <Dialog ariaLabel={studioName('newStep')} onClose={() => setFirstStepRevision(undefined)} manageFocus
          panelClassName="w-[min(20rem,90vw)] rounded-shell-control border border-shell-divider bg-shell-chrome p-2 text-shell-foreground shadow-xl">
          <button type="button" className={`${control} flex w-full items-center gap-2 text-sm`} data-dialog-autofocus onClick={chooseFirstTrain}>
            <TrainIcon size={16} aria-hidden="true" style={{ color: trainTint }} />
            {trainName}
          </button>
        </Dialog>}
        {sweepStep && <Dialog ariaLabel={studioName('newSweep')} onClose={() => setSweepStep(undefined)} manageFocus
          panelClassName="grid max-h-[75dvh] w-[min(24rem,90vw)] gap-2 overflow-auto rounded-shell-control border border-shell-divider bg-shell-chrome p-3 text-shell-foreground shadow-xl">
          <button type="button" className={`${control} justify-self-end`} data-dialog-autofocus onClick={() => setSweepStep(undefined)}>{studioName('close')}</button>
          <StudioTrainAuthoringForm editor={editor} stepId={sweepStep} picker={() => {
            const current = latestState.current;
            if (current.selection?.kind !== 'values' || current.selection.stepId !== sweepStep) {
              selection.select(selectStep(current, sweepStep));
            }
            setSweepStep(undefined);
          }} />
        </Dialog>}
      </>}
      clock={evaluationReplay.clock}
      failure={failure && <div role="alert" className="flex min-w-0 items-center gap-1">
        <span className="min-w-0 break-words">{failure}</span>
        <button type="button" className={control} {...commandAttributes(retryCommand)} onClick={retryCommand.invoke}>{studioName('retry')}</button>
      </div>}
      problems={<StudioProblems command={problemCommand} count={problems.length} open={!!problemPanel}
        opener={problemPanel?.opener ?? null} onClose={() => setPanel(null)}
        onOpen={(opener) => {
          if (problemPanel && !problemPanel.scope) {
            setPanel(null);
            opener.focus();
          } else {
            openProjectProblems(opener);
          }
        }}>
        <ProblemList problems={shownProblems} canReveal={(problem) => !!addressedStep(problem)}
          onChoose={revealProblem} showVariant={variants.length > 1 && !problemPanel?.scope} />
      </StudioProblems>}
      activity={<HeaderActivity placement="above" open={panel?.kind === 'activity'} command={activity}
        onToggle={activity.invoke}
        onClose={() => setPanel(null)} onOpenRun={openActivityRun} projectNames={names} />}
      renderView={(view, slot) => {
        const details = slot === 'primary' ? primary : secondary;
        const withProblem = (body: ReactNode) => revealed?.target.view === view && slot === focusSlot
          ? <ProblemDestination reveal={revealed}>{body}</ProblemDestination> : body;
        if (view === 'analysis' || view === 'figures') {
          const detail = analyze[view];
          return { ...detail, body: withProblem(<StudioErrorBoundary region={studioName(view)}>{detail.body}</StudioErrorBoundary>) };
        }
        if (view === 'model' || view === 'task' || view === 'scene') {
          const header = view !== 'scene' && authoring && wholeStep
            ? <StudioTrainAuthoringForm {...authoring} view={view} documentPicker />
            : details[view].header;
          const scope = slot === focusSlot && view === focusView ? activeDocumentScope : undefined;
          return { ...details[view], header: <>
            {header}
            {details[view].path}
            {scope && scope.users.size > 1 && <span className="whitespace-nowrap text-sm text-shell-muted" role="status">
              {' · '}{studioCountOf(scope.lit.size, scope.users.size)}
            </span>}
          </>, body: withProblem(<StudioErrorBoundary region={studioName(view)}>
            {view === 'task' && authoring && wholeStep && <StudioTrainAuthoringForm {...authoring} taskBinding />}
            {details[view].body}
          </StudioErrorBoundary>) };
        }
        if (!step) return { body: null };
        if (view === 'document') return { body: withProblem(
          <StudioErrorBoundary region={studioName('document')}>{report.body ?? noView}</StudioErrorBoundary>,
        ) };
        if (view === 'runs') {
          const runSelection = { step, state, onSelectionChange: mapProps.onSelectionChange };
          const pivot = pivotCommand(slot);
          const pivotActive = pivotSlots[slot] === step.id;
          const run = state.focus?.stepId === step.id ? step.runs.find(({ id }) => id === state.focus?.runId) : undefined;
          return {
            header: <StudioRunControls commands={step.kind === 'train' || step.kind === 'evaluate' ? runCommands : undefined}
              pivotCommand={pivot} pivotActive={pivotActive} reason={run?.reason} />,
            drawer: (authoring || step.inputs.length > 0) ? <div className="space-y-3 text-sm">
              {authoring && <StudioTrainAuthoringForm {...authoring}
                  focusFirstField={ready && focusCreatedStep === step.id && focusSlot === slot}
                  onDetachedEditError={setCommandError}
                  onFirstFieldFocused={() => setFocusCreatedStep(undefined)} />}
              {step.inputs.length > 0 && <section aria-label={studioName('inputs')} className="space-y-1">
                <h3 className="font-semibold">{studioName('inputs')}</h3>
                <dl>{step.inputs.map(({ id, label, sourceStepId, subset }) => <div key={id} className="border-t border-shell-divider py-1">
                  <dt>{label}</dt>
                  <dd className="text-shell-muted">{plan.steps.find((item) => item.id === sourceStepId)?.label}</dd>
                  {subset && <dd className="text-shell-muted">{[...subset.terms, studioCountOf(subset.count, subset.total)].join(' · ')}</dd>}
                </div>)}</dl>
              </section>}
            </div> : undefined,
            inspector: selectedRuns(step, state.selection).length > 0
              ? <StudioRunDetails plan={plan} {...runSelection} /> : undefined,
            body: withProblem(<StudioErrorBoundary region={studioName('runs')}>
              <StudioRuns plan={plan} {...runSelection}
                region={step.kind === 'train' || step.runs.length > 0
                  ? { kind: 'result', missing: [] } : { kind: 'empty' }}
                pivotActive={Boolean(pivot) && pivotActive} />
            </StudioErrorBoundary>),
          };
        }
        if ((view === 'training' || view === 'objective') && authoring) {
          return { body: withProblem(<StudioTrainAuthoringForm {...authoring} view={view} />) };
        }
        if (view === 'results' && row && ready) return { body: <div className="p-3">
          {attempt && <StudioTrainResults project={saved.project_id} rows={data!.rows} row={row} attempt={attempt} />}
          <TrainReceipts key={savedRunId(row.key)} row={row} />
        </div> };
        return { body: withProblem(noView) };
      }}
    />
  );
}

function ProblemDestination({ reveal, children }: { reveal: ProblemReveal; children: ReactNode }) {
  const reason = useRef<HTMLParagraphElement>(null);
  useEffect(() => {
    reason.current?.focus();
  }, [reveal]);
  return <div className="flex h-full min-h-0 flex-col">
    <p ref={reason} tabIndex={-1} className="border-b border-shell-divider px-3 py-2 text-sm">
      <span className="font-medium">{reveal.target.label}</span>{' · '}{problemReason(reveal.problem)}
    </p>
    <div className="min-h-0 flex-1 overflow-auto">{children}</div>
  </div>;
}

function ProblemList({ problems, onChoose, canReveal, showVariant }: {
  problems: readonly StudioProjectWindowMapProblem[];
  onChoose(problem: StudioProjectWindowMapProblem): void;
  canReveal(problem: StudioProjectWindowMapProblem): boolean;
  showVariant: boolean;
}) {
  const groups = new Map<string | null, StudioProjectWindowMapProblem[]>();
  for (const problem of problems) {
    const name = problem.document ?? problem.step ?? null;
    if (!groups.has(name)) groups.set(name, []);
    groups.get(name)!.push(problem);
  }
  return <div className="space-y-2">{[...groups].map(([name, entries]) => <section key={name ?? ''}>
    {name && <h2 className="font-medium">{name}</h2>}
    <ul>{entries.map((problem, index) => {
      const { label } = problemTarget(problem);
      const context = [problem.step !== name ? problem.step : null, showVariant ? problem.variant : null].filter(Boolean);
      const entry = <><span className="font-medium">{label}</span>{' · '}{problemReason(problem)}
        {context.length > 0 && <span className="block text-xs text-shell-muted">{context.join(' · ')}</span>}
      </>;
      return <li key={index} className="break-words">
        {canReveal(problem)
          ? <button className={`${control} w-full text-left`} type="button" onClick={() => onChoose(problem)}>{entry}</button>
          : <div className="px-2 py-1">{entry}</div>}
      </li>;
    })}</ul>
  </section>)}</div>;
}
