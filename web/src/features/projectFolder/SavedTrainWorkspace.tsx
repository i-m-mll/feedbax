import { useEffect, useId, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { TriangleAlert } from 'lucide-react';
import { fetchSavedTrainRuns, runSavedTrain } from '@/api/jobs';
import { apiErrorMessage } from '@/api/request';
import { StudioTrainResults } from '@/components/activity/StudioTrainResults';
import { studioCountOf, studioName, studioStep } from '@/components/activity/studioVocabulary';
import { HeaderActivity } from '@/components/layout/HeaderActivity';
import { StudioErrorBoundary } from '@/components/layout/StudioErrorBoundary';
import { bindSweepFold, StudioPipelineMap, StudioPipelineMapDrawer, type PipelineMapStepSource } from '@/components/pipeline/StudioPipelineMap';
import { StudioRuns } from '@/components/pipeline/StudioRuns';
import { Dialog } from '@/components/ui/Dialog';
import { foldedMapSweep, orderedMapSweeps, type PipelineCardPresentation } from '@/components/pipeline/pipelineMapLayout';
import type { PlanView } from '@/features/pipeline/planView';
import { selectRun, selectStep, selectSweepValue, selectedRuns, type PipelineSelection, type PipelineSelectionState } from '@/features/pipeline/selection';
import { ProjectWindow } from '@/features/projectWindow/ProjectWindow';
import { bindStudioCommand, commandAttributes, jobAllowsCommand, type StudioCommand } from '@/features/projectWindow/studioCommands';
import {
  projectWindowSlotView, showProjectWindowView, useProjectWindowState, type ProjectWindowSlot, type ProjectWindowView,
} from '@/features/projectWindow/projectWindowState';
import type { PresentationDocument, StudioActivityJob, StudioSavedTrainRuns } from '@/generated/studioContracts';
import { useStudioConnection } from '@/hooks/useStudioConnection';
import { useStudioJobActions, useStudioJobs } from '@/hooks/useStudioJobs';
import { directTrainStep } from './authoredEdits';
import { useManagedProjectDetails, useSavedEvaluationScene, type NamedDocumentScope } from './managedProjectDetails';
import { createCheckpointTrain } from './createCheckpointTrain';
import { canCreateFirstTrain, createFirstTrain } from './createFirstTrain';
import { openProjectEditor } from './openProjectEditor';
import type { ProjectEditController } from './projectEditController';
import { projectWindowPlan, planVariants, savedRunId, type SavedTrainRow as SavedRow } from './projectWindowPlan';
import { settledWindowSelection, useProjectWindowSelection } from './projectWindowSelection';
import { StudioTrainAuthoringForm } from './StudioTrainAuthoringForm';

type EditorSession = Awaited<ReturnType<typeof openProjectEditor>>;
type MapProblem = NonNullable<StudioSavedTrainRuns['window_map']>['problems'][number];
export interface ProjectWindowCommandScope {
  readonly editor: ProjectEditController;
  readonly selection: PipelineSelectionState;
  readonly variantId: string | null;
  readonly plan: PlanView;
  readonly projection: StudioSavedTrainRuns | undefined;
  onSelectionChange(selection: PipelineSelectionState): void;
  selectCreatedStep(stepId: string, view?: 'model' | 'runs'): void;
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
    <details className="mt-3 rounded border border-slate-200 p-3 text-sm">
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
  const selection = useProjectWindowSelection(Object.keys(saved.project.steps ?? {})[0] ?? null);
  const [commandError, setCommandError] = useState<string>();
  const [busy, setBusy] = useState(false);
  const [panel, setPanel] = useState<'activity' | 'problems' | null>(null);
  const [focusRequest, requestSlotFocus] = useState(0);
  const [pivotSlots, setPivotSlots] = useState<Partial<Record<ProjectWindowSlot, string>>>({});
  const [problemStep, setProblemStep] = useState<string>();
  const [excludedUsers, setExcludedUsers] = useState<Readonly<Record<string, readonly string[]>>>({});
  const problemCountId = useId();
  const [focusCreatedStep, setFocusCreatedStep] = useState<string>();
  const [focusFirstModel, setFocusFirstModel] = useState<string>();
  const [firstStepRevision, setFirstStepRevision] = useState<string>();
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
  const variants = useMemo(() => planVariants(data), [data]);
  const variant = selection.variantId && variants.includes(selection.variantId) ? selection.variantId : variants[0] ?? null;
  const plan = useMemo(() => projectWindowPlan(saved.project, data, variant), [saved.project, data, variant]);
  const step = plan.steps.find(({ id }) => id === selection.stepId);
  const state = data ? settledWindowSelection(plan, selection.pipeline) : selection.pipeline;
  const latestState = useRef(state);
  latestState.current = state;
  const wholeStep = state.selection?.kind === 'steps' && state.selection.stepIds.length === 1 && selection.variantId === null;
  const canAuthorSweep = (id: string) => selection.variantId === null && directTrainStep(saved.project, id) !== null;
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
  function documentScope(field: 'graph' | 'task'): NamedDocumentScope | undefined {
    const document = step?.documentIds?.[`payload.${field}`];
    if (!document || !step || !wholeStep || saved.project.sweep || step.kind !== 'train') return undefined;
    const users = new Set(plan.steps.filter(({ documentUses }) => documentUses?.includes(document)).map(({ id }) => id));
    if (!users.has(step.id)) return undefined;
    const key = JSON.stringify([step.id, document, variant]);
    const excluded = excludedUsers[key] ?? [];
    return { document, key, users, lit: new Set([...users].filter((id) => id === step.id || !excluded.includes(id))) };
  }
  const documentScopes = { graph: documentScope('graph'), task: documentScope('task') };
  const activeDocumentScope = windowState.mode === 'map' ? undefined
    : focusView === 'model' ? documentScopes.graph : focusView === 'task' ? documentScopes.task : undefined;
  function chooseMapStep(id: string, extend: boolean) {
    if (extend && activeDocumentScope && activeDocumentScope.users.size > 1) {
      if (id === step?.id || !activeDocumentScope.users.has(id)) return;
      const { key } = activeDocumentScope;
      setExcludedUsers((current) => {
        const ids = current[key] ?? [];
        return { ...current, [key]: ids.includes(id) ? ids.filter((excluded) => excluded !== id) : [...ids, id] };
      });
      return;
    }
    if (!extend) setExcludedUsers({});
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
  const namedRun = definitionRows.find(({ key }) => savedRunId(key) === state.focus?.runId)?.key ?? definitionRows[0]?.key;
  const detailInputs = {
    editor, step, stepId: step?.id ?? null, focus: state.focus, selection: state.selection,
    variantId: selection.variantId, evaluation: selectedEvaluation, evaluationReplay, onChooseSweepValue: authoring?.onChooseSweepValue,
    namedRun, documentScopes,
  };
  const primary = useManagedProjectDetails({ ...detailInputs, activeView: windowState.mode === 'map' ? 'hidden' : primaryView ?? 'hidden' });
  const secondary = useManagedProjectDetails({ ...detailInputs, activeView: windowState.mode === 'map' ? 'hidden' : secondaryView ?? 'hidden' });
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

  function openProblems(stepId?: string) {
    setProblemStep(stepId);
    setPanel('problems');
  }
  const problems = (data?.window_map?.problems ?? []).filter((problem) => !problem.variant || problem.variant === variant);
  function revealProblem(problem: MapProblem) {
    if (!problem.step || !plan.steps.some(({ id }) => id === problem.step)) return;
    const target = data?.rows.find(({ key }) => key.step === problem.step && key.row === problem.row && key.variant === variant);
    selection.select(target
      ? selectRun(state, { stepId: target.key.step, runId: savedRunId(target.key) })
      : selectStep(state, problem.step));
    setWindowState((current) => showProjectWindowView(current, 'runs'));
    requestSlotFocus((count) => count + 1);
    setPanel(null);
  }
  function showRun(match: SavedRow) {
    const target = { stepId: match.key.step, runId: savedRunId(match.key) };
    selection.select(selectRun(state, target), match.key.variant === variants[0] ? null : match.key.variant);
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
  function moveHistory(action: 'undo' | 'redo') {
    editor[action]();
    void editor.flush().catch((error) => setCommandError(apiErrorMessage(error)));
  }

  const commandScope: ProjectWindowCommandScope = {
    editor, selection: state, variantId: variant, onSelectionChange: selection.select,
    plan, projection: data,
    selectCreatedStep: (id, view = 'runs') => {
      selection.select(selectStep(state, id));
      if (view === 'model') setFocusFirstModel(id);
      else setFocusCreatedStep(id);
      setWindowState((current) => showProjectWindowView(current, view));
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

  function newStepCommand(source?: PipelineMapStepSource, subject?: string) {
    if (!source) {
      if (!canCreateFirstTrain(saved.project)) return undefined;
      return bindStudioCommand('newStep', () => setFirstStepRevision(saved.revision), { enabled: ready && !busy });
    }
    const sourceStep = plan.steps.find(({ id }) => id === source.stepId);
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
    plan, state, onSelectionChange: selection.select, onProblems: openProblems,
    newStepCommand, newSweepCommand, onChooseStep: chooseMapStep,
    documentScope: activeDocumentScope && activeDocumentScope.users.size > 1 ? activeDocumentScope : undefined,
  };
  const activity = bindStudioCommand('activity', () => setPanel((current) => current === 'activity' ? null : 'activity'));
  const problemCommand = bindStudioCommand('problems', () => openProblems());
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
    newStepCommand(),
    step && selectedAxis && bindSweepFold(step.id, selectedAxis, folded?.id === selectedAxis.id, cardPresentation, changePresentation),
    step && newSweepCommand(step.id),
    windowState.mode !== 'map' && focusView === 'runs' && pivotCommand(focusSlot),
    bindStudioCommand('undo', () => moveHistory('undo'), { enabled: !!snapshot.undoName && !busy }),
    bindStudioCommand('redo', () => moveHistory('redo'), { enabled: !!snapshot.redoName && !busy }),
    bindStudioCommand('clearFocus', () => selection.select({ ...state, focus: null }), { enabled: !!state.focus }),
    activity, problemCommand, failure && retryCommand,
  ].filter((command): command is StudioCommand => !!command);

  return (
    <ProjectWindow
      state={windowState} onStateChange={setWindowState} projectName={configuration.project}
      saveStatus={pending || snapshot.error ? studioName('edited') : undefined}
      availableViews={availableViews}
      commands={commands} focusRequest={focusRequest}
      onBack={selection.back} onForward={selection.forward}
      map={<StudioErrorBoundary region={studioName('pipelineMap')}><StudioPipelineMap {...mapProps}
        presentation={presentation} onPresentationChange={changePresentation} /></StudioErrorBoundary>}
      mapDrawer={<StudioPipelineMapDrawer {...mapProps} />}
      compactMap={<StudioPipelineMap {...mapProps} mode="line" />}
      variant={variants.length > 1 && <select aria-label={studioName('variant')} value={variant ?? ''} onChange={(event) => selection.setVariant(event.target.value === variants[0] ? null : event.target.value)}>
        {variants.map((id) => <option key={id} value={id}>{id}</option>)}
      </select>}
      overlay={<>
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
      problems={<>
        <button type="button" className={control} aria-expanded={panel === 'problems'} aria-describedby={problemCountId} {...commandAttributes(problemCommand)} onClick={problemCommand.invoke}>
          <TriangleAlert aria-hidden="true" size={14} className="mr-1 inline" />{studioName('problems')} <span id={problemCountId}>{problems.length}</span>
        </button>
        {panel === 'problems' && <Dialog ariaLabel={studioName('problems')} onClose={() => setPanel(null)} manageFocus
          panelClassName="grid max-h-[75dvh] w-[min(34rem,90vw)] gap-2 overflow-auto rounded-shell-control border border-shell-divider bg-shell-chrome p-3 text-shell-foreground shadow-xl">
          <button type="button" className={`${control} justify-self-end`} data-dialog-autofocus onClick={() => setPanel(null)}>{studioName('close')}</button>
          <ProblemList problems={problems.filter((problem) => !problemStep || problem.step === problemStep)} onChoose={revealProblem} />
        </Dialog>}
      </>}
      activity={<HeaderActivity placement="above" open={panel === 'activity'} command={activity}
        onToggle={activity.invoke}
        onClose={() => setPanel(null)} onOpenRun={openActivityRun} projectNames={names} />}
      renderView={(view, slot) => {
        const details = slot === 'primary' ? primary : secondary;
        if (view === 'model' || view === 'task' || view === 'scene') {
          const header = view !== 'scene' && authoring && wholeStep
            ? <StudioTrainAuthoringForm {...authoring} view={view} documentPicker />
            : details[view].header;
          const scope = slot === focusSlot && view === focusView ? activeDocumentScope : undefined;
          return { ...details[view], header: <>
            {header}
            {scope && scope.users.size > 1 && <span className="whitespace-nowrap text-sm text-shell-muted" role="status">
              {' · '}{studioCountOf(scope.lit.size, scope.users.size)}
            </span>}
          </>, body: <StudioErrorBoundary region={studioName(view)}>{details[view].body}</StudioErrorBoundary> };
        }
        if (!step) return { body: null };
        if (view === 'runs') return { body: <StudioErrorBoundary region={studioName('runs')}>
          <StudioRuns plan={plan} step={step} state={state} onSelectionChange={selection.select}
            region={step.kind === 'train' || step.kind === 'evaluate' && step.runs.length
              ? { kind: 'result', missing: [] } : { kind: 'empty' }}
            definition={authoring && <StudioTrainAuthoringForm {...authoring}
              focusFirstField={ready && focusCreatedStep === step.id && windowState.focusedSlot === slot}
              onFirstFieldFocused={() => setFocusCreatedStep(undefined)} />}
            commands={step.kind === 'train' || step.kind === 'evaluate' ? runCommands : undefined}
            pivotCommand={pivotCommand(slot)} pivotActive={pivotSlots[slot] === step.id} />
        </StudioErrorBoundary> };
        if ((view === 'training' || view === 'objective') && authoring) {
          return { body: <StudioTrainAuthoringForm {...authoring} view={view} /> };
        }
        if (view === 'results' && row && ready) return { body: <div className="p-3">
          {attempt && <StudioTrainResults project={saved.project_id} rows={data!.rows} row={row} attempt={attempt} />}
          <TrainReceipts key={savedRunId(row.key)} row={row} />
        </div> };
        return { body: noView };
      }}
    />
  );
}

function ProblemList({ problems, onChoose }: { problems: readonly MapProblem[]; onChoose(problem: MapProblem): void }) {
  const groups = new Map<string | null, MapProblem[]>();
  for (const problem of problems) {
    const name = problem.document ?? problem.step ?? null;
    if (!groups.has(name)) groups.set(name, []);
    groups.get(name)!.push(problem);
  }
  return <div className="space-y-2">{[...groups].map(([name, entries]) => <section key={name ?? ''}>
    {name && <h2 className="font-medium">{name}</h2>}
    <ul>{entries.map((problem, index) => <li key={index}>
      {problem.step
        ? <button className={control} type="button" onClick={() => onChoose(problem)}>{problem.path}: {problem.detail}</button>
        : <span>{problem.path}: {problem.detail}</span>}
    </li>)}</ul>
  </section>)}</div>;
}
