import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { Command, TriangleAlert } from 'lucide-react';
import { appTip } from '@window-shell/browser/tooltip';
import { fetchSavedTrainRuns, runSavedTrain } from '@/api/jobs';
import { apiErrorMessage } from '@/api/request';
import { StudioTrainResults } from '@/components/activity/StudioTrainResults';
import { studioName } from '@/components/activity/studioVocabulary';
import { HeaderActivity } from '@/components/layout/HeaderActivity';
import { StudioErrorBoundary } from '@/components/layout/StudioErrorBoundary';
import { StudioPipelineMap, StudioPipelineMapDrawer, type PipelineMapStepSource } from '@/components/pipeline/StudioPipelineMap';
import { StudioRuns } from '@/components/pipeline/StudioRuns';
import { Dialog } from '@/components/ui/Dialog';
import type { PipelineCardPresentation } from '@/components/pipeline/pipelineMapLayout';
import { selectRun, selectStep, selectedRuns, type PipelineSelection, type PipelineSelectionState } from '@/features/pipeline/selection';
import { ProjectWindow } from '@/features/projectWindow/ProjectWindow';
import {
  projectWindowSlotView, projectWindowViews, showProjectWindowView, useProjectWindowState, type ProjectWindowView,
} from '@/features/projectWindow/projectWindowState';
import type { PresentationDocument, StudioActivityJob, StudioSavedTrainRuns } from '@/generated/studioContracts';
import { useStudioConnection } from '@/hooks/useStudioConnection';
import { useStudioJobActions, useStudioJobs } from '@/hooks/useStudioJobs';
import { useManagedProjectDetails } from './managedProjectDetails';
import { openProjectEditor } from './openProjectEditor';
import type { ProjectEditController } from './projectEditController';
import { projectWindowPlan, planVariants, savedRunId, type SavedTrainRow as SavedRow } from './projectWindowPlan';
import { useProjectWindowSelection } from './projectWindowSelection';
import { StudioTrainAuthoringForm } from './StudioTrainAuthoringForm';

type EditorSession = Awaited<ReturnType<typeof openProjectEditor>>;
type MapProblem = NonNullable<StudioSavedTrainRuns['window_map']>['problems'][number];
export interface ProjectWindowCommandScope {
  readonly editor: ProjectEditController;
  readonly selection: PipelineSelectionState;
  readonly variantId: string | null;
  onSelectionChange(selection: PipelineSelectionState): void;
}

export type ProjectWindowCommands = {
  onNewStep?: (source: PipelineMapStepSource | undefined, scope: ProjectWindowCommandScope) => void;
  onNewSweep?: (stepId: string, scope: ProjectWindowCommandScope) => void;
};

const viewsByKind = {
  train: ['runs', 'model', 'task', 'objective', 'training', 'scene', 'results'],
  evaluate: ['runs', 'model', 'task', 'scene', 'results'],
  analyze: ['runs', 'analysis', 'figures'],
  report: ['runs', 'document'],
} as const satisfies Record<string, readonly ProjectWindowView[]>;
const emptyViews: readonly ProjectWindowView[] = [];
const viewShortcuts = Object.fromEntries(projectWindowViews.slice(0, 9).map((view, index) => [view, `⌘${index + 1}`]));
const control = 'rounded px-2 py-1 hover:bg-shell-hover focus-visible:outline-2';
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

export default function StudioProjectWorkspace({ projectId, onNewStep, onNewSweep }: { projectId: string } & ProjectWindowCommands) {
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
  const [panel, setPanel] = useState<'activity' | 'problems' | 'commands' | null>(null);
  const [problemStep, setProblemStep] = useState<string>();
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
  const state = selection.pipeline;
  const availableViews = step ? viewsByKind[step.kind] : emptyViews;
  const primaryView = projectWindowSlotView(windowState, 'primary', availableViews);
  const secondaryView = projectWindowSlotView(windowState, 'secondary', availableViews);
  const detailInputs = { editor, stepId: step?.id ?? null, focus: state.focus, selection: state.selection, variantId: selection.variantId };
  const primary = useManagedProjectDetails({ ...detailInputs, activeView: windowState.mode === 'map' ? 'hidden' : primaryView ?? 'hidden' });
  const secondary = useManagedProjectDetails({ ...detailInputs, activeView: windowState.mode === 'map' ? 'hidden' : secondaryView ?? 'hidden' });
  const selected = step ? selectedRuns(step, state.selection) : [];
  const resultId = state.focus?.runId ?? (selected.length === 1 ? selected[0].id : undefined);
  const row = data?.rows.find(({ key }) => key.step === step?.id && savedRunId(key) === resultId);
  const attempt = row?.attempts.at(-1);
  const ready = Boolean(data && !pending && !outside);
  const wholeStep = state.selection?.kind === 'steps' && state.selection.stepIds.length === 1 && selection.variantId === null;
  const names = useMemo(() => new Map([[saved.project_id, configuration.project]]), [saved.project_id, configuration.project]);

  async function run(target: PipelineSelection) {
    if (busy || !ready || !data) return;
    const ids = new Set(plan.steps.flatMap((item) => selectedRuns(item, target).map(({ id }) => id)));
    const keys = data.rows.filter(({ key }) => ids.has(savedRunId(key))).map(({ key }) => key);
    if (!keys.length) return;
    setBusy(true);
    setCommandError(undefined);
    try {
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
    } catch (cause) {
      setCommandError(apiErrorMessage(cause));
    } finally {
      setBusy(false);
    }
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
    setPanel(null);
  }
  async function openActivityRun(job: StudioActivityJob) {
    const match = data?.rows.find((item) => item.attempts.some(({ job_id }) => job_id === job.job_id));
    if (!match) return false;
    const target = { stepId: match.key.step, runId: savedRunId(match.key) };
    selection.select(selectRun(state, target), match.key.variant === variants[0] ? null : match.key.variant);
    setWindowState((current) => showProjectWindowView(current, 'results'));
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

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (!event.metaKey && !event.ctrlKey) return;
      const key = event.key.toLowerCase();
      const editable = event.target instanceof HTMLElement && event.target.closest('input, textarea, select, [contenteditable="true"]');
      if (key === 'k') {
        event.preventDefault();
        setPanel((current) => current === 'commands' ? null : 'commands');
      } else if (!editable) {
        const view = /^[1-9]$/.test(key) ? projectWindowViews[Number(key) - 1] : undefined;
        if (view && availableViews.includes(view)) {
          event.preventDefault();
          setWindowState((current) => showProjectWindowView(current, view));
        } else if (key === '[' && selection.back) {
          event.preventDefault();
          selection.back();
        } else if (key === ']' && selection.forward) {
          event.preventDefault();
          selection.forward();
        } else if (key === 'z') {
          event.preventDefault();
          moveHistory(event.shiftKey ? 'redo' : 'undo');
        } else if (key === 'r' && ready) {
          event.preventDefault();
          void run(state.selection ?? { kind: 'steps', stepIds: plan.steps.map(({ id }) => id) });
        }
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  const commandScope: ProjectWindowCommandScope = {
    editor, selection: state, variantId: variant, onSelectionChange: selection.select,
  };
  const mapProps = {
    plan, state, onSelectionChange: selection.select, onProblems: openProblems,
    onNewStep: onNewStep && ((source?: PipelineMapStepSource) => onNewStep(source, commandScope)),
    onNewSweep: onNewSweep && ((stepId: string) => onNewSweep(stepId, commandScope)),
  };
  const focusView = windowState.focusedSlot === 'secondary' ? secondaryView : primaryView;
  const focusDetails = windowState.focusedSlot === 'secondary' ? secondary : primary;

  return (
    <ProjectWindow
      state={windowState} onStateChange={setWindowState} projectName={configuration.project}
      saveStatus={pending || snapshot.error ? studioName('edited') : undefined}
      availableViews={availableViews}
      viewShortcuts={viewShortcuts}
      onBack={selection.back} onForward={selection.forward}
      map={<StudioErrorBoundary region={studioName('pipelineMap')}><StudioPipelineMap {...mapProps}
        presentation={presentation} onPresentationChange={changePresentation} /></StudioErrorBoundary>}
      mapDrawer={<StudioPipelineMapDrawer {...mapProps} />}
      compactMap={<StudioPipelineMap {...mapProps} mode="line" />}
      variant={variants.length > 1 && <select aria-label={studioName('variant')} value={variant ?? ''} onChange={(event) => selection.setVariant(event.target.value === variants[0] ? null : event.target.value)}>
        {variants.map((id) => <option key={id} value={id}>{id}</option>)}
      </select>}
      commandPalette={<>
        <button className={control} type="button" aria-label={studioName('commandPalette')} {...appTip(`${studioName('commandPalette')} · ⌘K`)}
          onClick={() => setPanel('commands')}><Command aria-hidden="true" size={18} /></button>
        {panel === 'commands' && <Dialog ariaLabel={studioName('commandPalette')} onClose={() => setPanel(null)}
          panelClassName="grid w-64 gap-1 rounded-shell-control border border-shell-divider bg-shell-chrome p-3 text-shell-foreground shadow-xl">
          <button type="button" className={control} autoFocus disabled={!snapshot.undoName} onClick={() => { moveHistory('undo'); setPanel(null); }}>{studioName('undo')}</button>
          <button type="button" className={control} disabled={!snapshot.redoName} onClick={() => { moveHistory('redo'); setPanel(null); }}>{studioName('redo')}</button>
        </Dialog>}
      </>}
      clock={focusView === 'scene' ? focusDetails.scene.clock : undefined}
      failure={failure && <div role="alert" className="flex min-w-0 items-center gap-1">
        <span className="truncate" {...appTip(failure)}>{failure}</span>
        <button type="button" className={control} onClick={retry}>{studioName('retry')}</button>
      </div>}
      problems={<>
        <button type="button" className={control} aria-expanded={panel === 'problems'} onClick={() => openProblems()}>
          <TriangleAlert aria-hidden="true" size={14} className="mr-1 inline" />{studioName('problems')} {problems.length}
        </button>
        {panel === 'problems' && <Dialog ariaLabel={studioName('problems')} onClose={() => setPanel(null)}
          panelClassName="grid max-h-[75dvh] w-[min(34rem,90vw)] gap-2 overflow-auto rounded-shell-control border border-shell-divider bg-shell-chrome p-3 text-shell-foreground shadow-xl">
          <button type="button" className={`${control} justify-self-end`} autoFocus onClick={() => setPanel(null)}>{studioName('close')}</button>
          <ProblemList problems={problems.filter((problem) => !problemStep || problem.step === problemStep)} onChoose={revealProblem} />
        </Dialog>}
      </>}
      activity={<HeaderActivity placement="above" open={panel === 'activity'}
        onToggle={() => setPanel((current) => current === 'activity' ? null : 'activity')}
        onClose={() => setPanel(null)} onOpenRun={openActivityRun} projectNames={names} />}
      renderView={(view, slot) => {
        const details = slot === 'primary' ? primary : secondary;
        if (view === 'model' || view === 'task' || view === 'scene') {
          const header = view !== 'scene' && step?.kind === 'train' && wholeStep
            ? <StudioTrainAuthoringForm editor={editor} stepId={step.id} view={view} />
            : details[view].header;
          return { ...details[view], header, body: <StudioErrorBoundary region={studioName(view)}>{details[view].body}</StudioErrorBoundary> };
        }
        if (!step) return { body: null };
        if (view === 'runs') return { body: <StudioErrorBoundary region={studioName('runs')}>
          <StudioRuns plan={plan} step={step} state={state} onSelectionChange={selection.select}
            region={step.kind === 'train' ? { kind: 'result', missing: [] } : { kind: 'empty' }}
            definition={step.kind === 'train' && wholeStep && <StudioTrainAuthoringForm editor={editor} stepId={step.id} />}
            commands={step.kind === 'train' ? [{ name: 'run', enabled: ready && !busy, perform: run }] : undefined} />
        </StudioErrorBoundary> };
        if ((view === 'training' || view === 'objective') && step.kind === 'train' && wholeStep) {
          return { body: <StudioTrainAuthoringForm editor={editor} stepId={step.id} view={view} /> };
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
