import { useEffect, useId, useMemo, useRef, useState, useSyncExternalStore, type ReactNode } from 'react';
import { keepPreviousData, useQuery, useQueryClient } from '@tanstack/react-query';
import { approveSavedRun, fetchExecutionTargets, fetchSavedTrainRuns, quoteSavedRun, runSavedTrain } from '@/api/jobs';
import { ApiRequestError, apiErrorMessage } from '@/api/request';
import { prepareCheckpointRuleEdit, prepareSubsetToRows } from '@/api/studioAuthoring';
import { StudioTrainResults, type TrainLossResult } from '@/components/activity/StudioTrainResults';
import { StudioRegion, studioRegionIssue, type StudioRegionState } from '@/components/activity/StudioRegion';
import { studioAccessibleName, studioCountOf, studioName, studioReason } from '@/components/activity/studioVocabulary';
import { HeaderActivity } from '@/components/layout/HeaderActivity';
import { bindSweepFold, StudioPipelineMap, StudioPipelineMapDrawer, type PipelineMapStepSource } from '@/components/pipeline/StudioPipelineMap';
import { canvasViewportCommands, type CanvasViewport } from '@/components/canvas/StudioCanvas';
import { useStepInsertMenu, type StepInsertion } from '@/components/pipeline/StepInsertMenu';
import { StudioRunControls, StudioRunDetails, StudioRuns } from '@/components/pipeline/StudioRuns';
import { StudioMachineApproval, StudioRunMachine } from '@/components/pipeline/StudioRunMachine';
import { StudioSweepOverview } from '@/components/pipeline/StudioSweepOverview';
import { X } from 'lucide-react';
import { Button, IconButton } from '@/components/ui/Button';
import { Dialog } from '@/components/ui/Dialog';
import { Select } from '@/components/ui/Select';
import { TextField } from '@/components/ui/TextField';
import { humanizeLabel } from '@/components/values/humanizeLabel';
import { foldedMapSweep, orderedMapSweeps, type PipelineCardPresentation } from '@/components/pipeline/pipelineMapLayout';
import type { PlanView } from '@/features/pipeline/planView';
import { selectRun, selectStep, selectSweepValue, selectedRuns, type PipelineSelection, type PipelineSelectionState } from '@/features/pipeline/selection';
import { ProjectWindow } from '@/features/projectWindow/ProjectWindow';
import { StudioProblems } from '@/features/projectWindow/StudioProblems';
import { bindStudioCommand, commandAttributes, jobAllowsCommand, type StudioCommand } from '@/features/projectWindow/studioCommands';
import {
  projectWindowSlotView, showProjectWindowView, useProjectWindowState, type ProjectWindowSlot, type ProjectWindowView,
} from '@/features/projectWindow/projectWindowState';
import {
  ModelAddressSchema, StepReferenceSchema, type PresentationDocument, type PresentationPosition, type ProjectValueEdit,
  type StudioActivityJob, type StudioJobSubmission, type StudioMachineQuote, type StudioProjectDocument, type StudioProjectWindowMapProblem,
  type StudioSavedTrainKey, type StudioSavedTrainRuns,
  type StudioCheckpointRuleRequest,
} from '@/generated/studioContracts';
import { useStudioConnection } from '@/hooks/useStudioConnection';
import { useStudioJobActions, useStudioJobs } from '@/hooks/useStudioJobs';
import { authoredPointer, directTrainStep, modelDocumentId } from './authoredEdits';
import { useManagedProjectDetails, useSavedEvaluationScene, type NamedDocumentScope } from './managedProjectDetails';
import { useModelSelectionStore } from './model/modelSelection';
import { createCheckpointTrain } from './createCheckpointTrain';
import { analyzeInput, createAnalyze, createReport, isReportFigureSource, type ReportFigureSource } from './createManagedStep';
import { canCreateFirstTrain, createFirstTrain } from './createFirstTrain';
import { checkpointLabel, createEvaluation, editableCheckpointRule, exactCheckpoint, managedTrialDomain, ordinaryEvaluation, type EditableCheckpointRule, type StoppedEvaluationSubject } from './evaluateAuthoring';
import { bindProjectEditor } from './openProjectEditor';
import type { ProjectEditController } from './projectEditController';
import { bestRankGroups, projectWindowPlan, planVariants, savedRunId, stepSourceRows, type SavedTrainRow as SavedRow } from './projectWindowPlan';
import { problemReason, problemTarget, removeGraphVariation, type ProblemReveal } from './projectWindowProblems';
import { settledWindowSelection, useProjectWindowSelection } from './projectWindowSelection';
import { CheckpointRuleEditor, StudioEvaluateAuthoring } from './StudioEvaluateAuthoring';
import { StudioEvaluationAttemptDetails, StudioEvaluationResults } from './StudioEvaluationResults';
import { StudioTrainAuthoringForm } from './StudioTrainAuthoringForm';
import { TrialObjectiveInspector, TrialObjectiveTable, TrialTaskForm, TrialTaskPicker, useTrialAuthoring } from './StudioTrialAuthoring';
import { useManagedAnalyze } from './useManagedAnalyze';
import { useManagedReport } from './useManagedReport';

type EditorSession = ReturnType<typeof bindProjectEditor>;

function exceptionFieldLabel(path: string) {
  if (!path.startsWith('{')) return path;
  try {
    const address = ModelAddressSchema.safeParse(JSON.parse(path));
    const field = address.success && address.data.element.kind === 'field' ? address.data.element.path.at(-1) : undefined;
    return field == null ? path : humanizeLabel(String(field));
  } catch {
    return path;
  }
}
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
const noView = <StudioRegion state={{ kind: 'empty' }} />;

function ResultDetail({ name, result }: { name: 'currentResult' | 'previousResult'; result: NonNullable<SavedRow['result']> }) {
  const [copyError, setCopyError] = useState(false);
  return <section aria-label={studioName(name)} className="min-w-0">
    <h3 className="font-medium">{studioName(name)}</h3>
    {result.progress && <dl className="space-y-1">
      <div><dt>{studioName('updates')}</dt><dd>{result.progress.completed_updates}</dd></div>
    </dl>}
    <Button onClick={async () => {
      setCopyError(false);
      try {
        await navigator.clipboard.writeText(JSON.stringify(result, null, 2));
      } catch {
        setCopyError(true);
      }
    }}>{studioName('copyDetails')}</Button>
    {copyError && <p role="alert">{studioReason('copyFailed', studioName('details'))}</p>}
  </section>;
}

function TrainResults({ row, attempt, ...plot }: Omit<Parameters<typeof StudioTrainResults>[0], 'selection'> & {
  attempt: SavedRow['attempts'][number] | undefined;
}) {
  const [comparing, setComparing] = useState(false);
  const results: TrainLossResult[] = [
    ...(row.result ? [{ name: 'currentResult', result: row.result } as const] : []),
    ...(row.previous ? [{ name: 'previousResult', result: row.previous } as const] : []),
  ];
  const selection = comparing && results.length ? { kind: 'results', results } as const
    : attempt ? { kind: 'attempt', attempt } as const : undefined;
  return (
    <>
      {selection && <StudioTrainResults {...plot} row={row} selection={selection} />}
      {results.length > 0 && <details className="mt-3 rounded-surface border border-shell-divider p-3 text-sm"
        onToggle={(event) => setComparing(event.currentTarget.open)}>
        <summary className="cursor-pointer font-medium">
          {studioName(results.length === 2 ? 'compareResults' : results[0].name)}
        </summary>
        {row.change && (
          <p className="mt-2 break-all">
            <code>{row.change.path}</code>{' '}
            <code>{JSON.stringify(row.change.before)}</code> → <code>{JSON.stringify(row.change.after)}</code>
          </p>
        )}
        <div className="mt-2 grid gap-3 sm:grid-cols-2">
          {results.map(({ name, result }) => <ResultDetail key={name} name={name} result={result} />)}
        </div>
      </details>}
    </>
  );
}

export default function StudioProjectWorkspace({
  document, onNewStep = checkpointNewStep, onNewSweep,
}: { document: StudioProjectDocument } & ProjectWindowCommands) {
  const [load, setLoad] = useState<{ session: EditorSession } | { error: string }>();
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    try {
      const session = bindProjectEditor(document);
      setLoad({ session });
      return session.close;
    } catch (cause) {
      setLoad({ error: apiErrorMessage(cause) });
    }
  }, [document, retry]);

  if (!load) return <main aria-busy="true" />;
  if ('error' in load) return (
    <main className="p-4"><p role="alert">{load.error}</p>
      <Button onClick={() => setRetry((attempt) => attempt + 1)}>{studioName('retry')}</Button>
    </main>
  );
  return <ManagedProjectWindow session={load.session} onNewStep={onNewStep} onNewSweep={onNewSweep} />;
}

function ManagedProjectWindow({ session, onNewStep, onNewSweep }: { session: EditorSession } & ProjectWindowCommands) {
  const stoppedCheckpointId = useId();
  const { editor } = session;
  const snapshot = useSyncExternalStore(editor.subscribe, editor.getSnapshot);
  const { saved, configuration, pending, outside } = snapshot;
  const jobs = useStudioJobs();
  const connection = useStudioConnection();
  const queries = useQueryClient();
  const wasDisconnected = useRef(false);
  useEffect(() => {
    if (connection.isError) wasDisconnected.current = true;
    if (!connection.isSuccess || !wasDisconnected.current) return;
    wasDisconnected.current = false;
    void queries.refetchQueries({
      type: 'active',
      predicate: ({ state: { error } }) => error instanceof ApiRequestError &&
        (error.kind === 'network' || (error.status ?? 0) >= 500),
    });
  }, [connection.isError, connection.isSuccess, queries]);
  const actions = useStudioJobActions();
  const [windowState, setWindowState] = useProjectWindowState(saved.project_id);
  const modelWindow = useModelSelectionStore();
  const mapViewport = useRef<CanvasViewport>(null);
  const selection = useProjectWindowSelection(Object.keys(saved.project.steps ?? {})[0] ?? null);
  const [commandError, setCommandError] = useState<string>();
  const [machines, setMachines] = useState<Readonly<Record<string, string>>>({});
  const [machineQuote, setMachineQuote] = useState<StudioMachineQuote>();
  const [rename, setRename] = useState<{ name: string; revision: string }>();
  const [busy, setBusy] = useState(false);
  const [panel, setPanel] = useState<WindowPanel | null>(null);
  useEffect(() => {
    const openActivity = () => {
      delete document.documentElement.dataset.studioOpenActivity;
      setPanel({ kind: 'activity' });
    };
    window.addEventListener('studio-native-open-activity', openActivity);
    if (document.documentElement.dataset.studioOpenActivity) openActivity();
    return () => window.removeEventListener('studio-native-open-activity', openActivity);
  }, []);
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
  const insertMenu = useStepInsertMenu();
  const latestRevision = useRef(saved.revision);
  latestRevision.current = saved.revision;
  const [resultAttempt, setResultAttempt] = useState<{ runId: string; jobId: string }>();
  const [reportChoices, setReportChoices] = useState<{ revision: string; figures: ReportFigureSource[] }>();
  const inFlight = useRef(false);
  const [sweepStep, setSweepStep] = useState<string>();
  const [selectedTerm, setSelectedTerm] = useState<{ step: string; term: string }>();
  const submission = useRef<{ intent: string; token: string }>();
  const projection = useQuery({
    queryKey: ['saved-train-runs', saved.project_id, saved.revision, jobs.data?.cursor, jobs.data?.remote_version],
    queryFn: ({ signal }) => fetchSavedTrainRuns(saved.project_id, saved.revision, signal),
    placeholderData: keepPreviousData,
    gcTime: 0,
    retry: false,
  });
  const data = projection.data?.revision === saved.revision ? projection.data : undefined;
  const variants = useMemo(() => planVariants(projection.data), [projection.data]);
  const variant = selection.variantId ?? variants[0] ?? null;
  const missingVariant = selection.variantId !== null && !variants.includes(selection.variantId);
  const primaryVariant = variant !== null && variant === variants[0];
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
  const subjectCheckpoint = authored?.kind === 'evaluate' ? exactCheckpoint(authored) : undefined;
  const checkpoint = authored && authored.inputs?.checkpoint;
  const linked = wholeStep && authored?.kind === 'train' &&
    'step' in authored.base && authored.base.row == null && checkpoint &&
    'step' in checkpoint && checkpoint.step === authored.base.step && checkpoint.checkpoint === 'last';
  const authoring = step && (primaryVariant && directTrainStep(saved.project, step.id, true) !== null || linked) && state.selection ? {
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
    const document = field === 'graph' ? modelDocumentId(step?.documentIds)
      : step?.documentIds?.['payload.trial_documents.task'] ?? step?.documentIds?.['payload.trial_documents.python_task'];
    if (!document || !step || !singleStep || !data || step.kind !== 'train') return undefined;
    const localVariant = Boolean(saved.project.sweep && !primaryVariant && field === 'graph');
    if (!localVariant && !wholeStep) return undefined;
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
    setResultAttempt(undefined);
    setProblemReveal(undefined);
    selection.select(selectStep(state, id, extend));
  }
  const row = data?.rows.find(({ key }) => key.step === step?.id && savedRunId(key) === state.focus?.runId);
  const selectedAttempt = row && resultAttempt?.runId === savedRunId(row.key)
    ? row.attempts.find(({ job_id }) => job_id === resultAttempt?.jobId) : undefined;
  const attempt = selectedAttempt ?? row?.attempts.at(-1);
  const ready = Boolean(data && !pending && !outside && !connection.isError && !projection.error);
  const canAuthorContinuation = ready && primaryVariant;
  const checkpointAttempt = !saved.project.sweep && windowState.mode !== 'map' &&
    state.selection?.kind === 'rows' && state.selection.runIds.length === 1 &&
    (primaryView === 'results' || secondaryView === 'results') && row && attempt?.status === 'stopped' &&
    data?.window_map?.rows.some(({ key, evaluation_conditions }) =>
      evaluation_conditions != null && savedRunId(key) === savedRunId(row.key))
    ? attempt.job_id : undefined;
  const stoppedPlan = useQuery({
    queryKey: ['saved-stopped-checkpoint', saved.project_id, saved.revision, checkpointAttempt, jobs.data?.cursor, jobs.data?.remote_version],
    queryFn: ({ signal }) => fetchSavedTrainRuns(saved.project_id, saved.revision, signal, checkpointAttempt),
    enabled: ready && checkpointAttempt !== undefined,
    gcTime: 0,
    retry: false,
  });
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
  const authoredMap = data?.window_map;
  const lightSteps = new Set(plan.steps.filter(({ kind }) => kind === 'analyze' || kind === 'report').map(({ id }) => id));
  const lightRows = authoredMap?.rows.filter(({ key, status }) =>
    lightSteps.has(key.step) && runIds.has(savedRunId(key)) && status !== 'cannot_run') ?? [];
  const runnableTrainRows = runRows.filter(({ status }) => status !== 'cannot_run');
  const reuseCompleted = !state.selection;
  const runnableEvaluationRows = evaluationRows.filter(({ can_run, status }) => can_run &&
    (status === 'not_run' || status === 'stale' || status === 'failed' ||
      status === 'stopped' || status === 'interrupted' || status === 'corrupt' ||
      reuseCompleted && status === 'succeeded'));
  const runnableRows = [...runnableTrainRows, ...runnableEvaluationRows, ...lightRows];
  const runnableSteps = new Set(runnableRows.map(({ key }) => key.step));
  const machineKinds = [...new Set(plan.steps.filter(({ id }) => runnableSteps.has(id) || id === step?.id)
    .map(({ kind }) => kind))].sort();
  const machineCatalog = useQuery({
    queryKey: ['execution-targets', machineKinds],
    queryFn: ({ signal }) => fetchExecutionTargets(machineKinds, signal),
    enabled: machineKinds.length > 0,
    retry: false,
  });
  const machineCatalogVersion = jobs.data?.machine_catalog_version;
  useEffect(() => {
    if (machineCatalogVersion !== undefined) void queries.invalidateQueries({ queryKey: ['execution-targets'] });
  }, [machineCatalogVersion, queries]);
  const requestedMachine = step ? machines[step.id] : undefined;
  const localMachine = machineCatalog.data?.default ?? 'local';
  const remoteMachine = requestedMachine && requestedMachine !== localMachine ? requestedMachine : undefined;
  const machineReady = !remoteMachine || !machineCatalog.isError && machineCatalog.data?.targets.some(({ profile, availability }) =>
    profile.id === remoteMachine && availability.state === 'ready');
  const evaluationAuthoring = wholeStep && authored && ordinaryEvaluation(authored);
  const evaluationSource = evaluationAuthoring && plan.steps.find(({ id }) => id === evaluationAuthoring.subject.step);
  const subject = authoredMap?.steps.find(({ step_id, variant: id }) => step_id === step?.id && id === variant)
    ?.inputs.find(({ id }) => id === 'subject');
  const subjectSubset = subject?.subset;
  const rankGroups = bestRankGroups(subjectSubset);
  const rankedSubjects = rankGroups.length ? authoredMap?.evaluations?.flatMap(({ key, bound_subject }) =>
    key.variant === variant && key.step === step?.id && bound_subject ? [bound_subject] : []) ?? [] : [];
  const subjectsByRank = new Map(rankedSubjects.map((bound) => [bound.key, bound]));
  const freezeAttempts = rankGroups.length && rankGroups.every(({ ranks }) => {
    const snapshots = ranks.map((key) => subjectsByRank.get(key)?.selection_sha256);
    return snapshots.every(Boolean) && new Set(snapshots).size === 1;
  }) ? rankGroups.flatMap(({ ranks }) => ranks.map((key) => subjectsByRank.get(key)!.consumer_job_id)) : undefined;
  const subjectIds = rankGroups.length
    ? new Set(rankedSubjects.map((bound) => savedRunId({ ...bound, step: subject.source_step_id })))
    : subjectSubset?.members && new Set(subjectSubset.members.map(savedRunId));
  const evaluationEditor = evaluationSource && evaluationAuthoring && authoredMap?.rows.some(({ key, evaluation_conditions }) =>
    key.step === evaluationSource.id && key.variant === variant && evaluation_conditions != null)
    ? {
      authored: evaluationAuthoring, source: evaluationSource,
      domain: managedTrialDomain(projection.data, subjectIds ? evaluationSource.runs.filter(({ id }) => subjectIds.has(id))
        : evaluationAuthoring.subject.where == null ? evaluationSource.runs : []),
    } : undefined;
  const checkpointVariant = variant && saved.project.sweep?.rows?.[variant] ? variant : undefined;
  const localSubject = StepReferenceSchema.safeParse(authored?.inputs?.subject);
  const stepCheckpointRule = data && singleStep && (wholeStep && !saved.project.sweep || checkpointVariant)
    && authored?.kind === 'evaluate' && !authored.deltas?.length
    && localSubject.success && (localSubject.data.checkpoint != null || subject?.checkpoint_axis)
    ? editableCheckpointRule(subject?.checkpoint)
    : undefined;
  const exactSources = selectedEvaluation?.result?.checkpoint_sources;
  const exactSource = subjectCheckpoint !== undefined && exactSources?.length === 1 ? exactSources[0] : undefined;
  const evaluationReplay = useSavedEvaluationScene(saved.project_id, saved.revision, selectedEvaluation,
    windowState.mode !== 'map' && [primaryView, secondaryView].includes('scene'));
  const definitionRows = data?.window_map?.rows.filter(({ key }) => key.step === step?.id && key.variant === variant) ?? [];
  const selectedDefinitionRows = definitionRows.filter(({ key }) => runIds.has(savedRunId(key)));
  const namedRun = selectedDefinitionRows.find(({ key }) => savedRunId(key) === state.focus?.runId)?.key ??
    selectedDefinitionRows[0]?.key ?? definitionRows[0]?.key;
  const trialStep = step?.kind === 'train' || step?.kind === 'evaluate';
  const trialVisible = windowState.mode !== 'map' &&
    [primaryView, secondaryView].some((view) => view === 'task' || view === 'objective');
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
  const modelSource = step?.kind === 'evaluate' ? modelDocumentId(plan.steps.find(({ id }) =>
    id === step.inputs.find(({ id }) => id === 'subject')?.sourceStepId)?.documentIds) : undefined;
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
  const trial = useTrialAuthoring(editor, saved.project_id, saved.revision, namedRun, trialStep && trialVisible,
    !pending && !outside && !snapshot.saving && Boolean(evaluateScope ||
      wholeStep && !authored?.sweep && !authored?.deltas?.length && step?.kind === 'train'),
    evaluateScope || undefined);
  const chosenAxis = evaluateScope && evaluateScope.kind === 'axis_value'
    ? authored?.sweep?.axes?.[evaluateScope.axis] : undefined;
  const checkpointScope = !saved.project.sweep && !subject?.checkpoint_axis && evaluateScope && (
    evaluateScope.kind === 'axis_value' && chosenAxis &&
      ('values' in chosenAxis && !('path' in chosenAxis))
      ? evaluateScope
      : evaluateScope.kind === 'rows' || evaluateScope.kind === 'axis_value'
        ? { kind: 'rows', rows: selectedDefinitionRows.map(({ key }) => key.row) } as const
        : undefined
  );
  const selectedRules = new Map(sceneRows.map((item) => [JSON.stringify(item.checkpoint_rule), item.checkpoint_rule]));
  const scopedRule = checkpointScope && selectedRules.size === 1
    ? editableCheckpointRule(selectedRules.values().next().value) : undefined;
  const evaluationCheckpointRule = stepCheckpointRule || scopedRule;
  const canSweepCheckpoint = Boolean(namedRun && stepCheckpointRule && !checkpointVariant && !subject?.checkpoint_axis &&
    stepCheckpointRule !== 'last' && (Array.isArray(stepCheckpointRule) || 'every' in stepCheckpointRule));
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
    variantId: variant, evaluation: selectedEvaluation, evaluationReplay,
    namedRun, documentScopes, modelWindow, evaluateScope: evaluateScope || undefined, modelSource, modelProblems, selectedSweepValues,
    fieldRows: state.selection && state.selection.kind !== 'steps' ? selectedDefinitionRows.map(({ key }) => key.row) : undefined,
    fieldAxes: state.selection?.kind === 'values' ? Object.fromEntries(
      state.selection.terms.map(({ sweepId, valueIds }) => [sweepId, [...valueIds]]),
    ) : undefined,
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
    run: step?.kind === 'report' ? namedRun : undefined,
    wholeStep, focused: windowState.mode !== 'map' && focusView === 'document',
  });
  const jobsById = new Map(jobs.data?.jobs.map((job) => [job.job_id, job]));
  const evaluationRequests = new Map(evaluationRows.map(({ key, request_id }) => [savedRunId(key), request_id]));
  const selectedJobs = state.selection ? [
    ...runRows.flatMap((item) => {
      const latest = item.attempts.at(-1);
      const job = latest && jobsById.get(latest.job_id);
      return job ? [job] : [];
    }),
    ...(jobs.data?.jobs.filter((job) => job.kind === 'evaluate' && job.target?.kind === 'session' &&
      job.project_id === saved.project_id && evaluationRequests.get(job.run_key) === job.request_id) ?? []),
  ] : [];
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

  function applySubmission(result: StudioJobSubmission) {
    const refusal = result.items.find(({ disposition }) => disposition === 'rejected');
    if (refusal?.reason) setCommandError(refusal.reason);
    actions.refresh();
  }

  function run() {
    if (!ready || !data || !runnableRows.length || machineQuote || !machineReady) return;
    const keys = runnableRows.map(({ key }) => key);
    void perform(async () => {
      if (await editor.flush() !== data.revision) throw new Error('The project changed before the run was submitted.');
      const project = editor.getSnapshot().saved.project_id;
      if (remoteMachine) {
        setMachineQuote(await quoteSavedRun(project, { revision: data.revision, run_keys: keys, target: remoteMachine }));
        return;
      }
      const intent = JSON.stringify([data.revision, keys, reuseCompleted]);
      const token = submission.current?.intent === intent ? submission.current.token : crypto.randomUUID();
      submission.current = { intent, token };
      const result = await runSavedTrain(project, {
        revision: data.revision, run_keys: keys, token, ...(reuseCompleted ? { reuse_completed: true } : {}),
      });
      submission.current = undefined;
      applySubmission(result);
    });
  }

  function approveMachine() {
    if (!machineQuote) return;
    void perform(async () => {
      await approveSavedRun(saved.project_id, { quote: machineQuote.id });
      setMachineQuote(undefined);
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
  const graphRepair = revealed && removeGraphVariation(saved.project, revealed.problem);
  function removeRefusedVariation() {
    if (!graphRepair || !ready || snapshot.saving || busy) return;
    void perform(async () => {
      editor.editFiles(studioName('removeVariation'), [graphRepair], { reviewedRevision: saved.revision });
      await editor.flush();
    });
  }
  function showRun(key: StudioSavedTrainKey, jobId?: string, view: 'runs' | 'results' | 'scene' = 'results') {
    const target = { stepId: key.step, runId: savedRunId(key) };
    setResultAttempt(jobId ? { runId: target.runId, jobId } : undefined);
    selection.select(selectRun(state, target), key.variant);
    setWindowState((current) => showProjectWindowView(current, view));
    requestSlotFocus((count) => count + 1);
  }
  function checkpointLineage() {
    if (!exactSource || !selectedEvaluation || !ready) return undefined;
    const targetId = savedRunId(selectedEvaluation.key);
    const producerIds = new Set(data?.window_map?.edges
      .filter(({ to_key }) => savedRunId(to_key) === targetId)
      .map(({ from_key }) => savedRunId(from_key)));
    const producers = data?.rows.filter((candidate) => producerIds.has(savedRunId(candidate.key)) &&
      candidate.request_id === exactSource.request_id &&
      candidate.attempts.some(({ job_id }) => job_id === exactSource.job_id));
    const producer = producers?.length === 1 ? producers[0] : undefined;
    const label = checkpointLabel(exactSource.completed_updates);
    return <div className="ps-2">
      {producer ? <Button variant="ghost" className="-ml-control-inline" onClick={() => showRun(producer.key, exactSource.job_id)}>
        {label}
      </Button> : <span>{label}</span>}
    </div>;
  }

  async function openActivityRun(job: StudioActivityJob) {
    if (job.project_id !== saved.project_id) return false;
    if (job.kind === 'evaluate') {
      const match = data?.window_map?.evaluations?.find(({ key, request_id }) =>
        savedRunId(key) === job.run_key && request_id === job.request_id);
      if (!match) return false;
      const view = match.result?.job_id !== job.job_id ? 'runs' : match.checkpoint_series ? 'results' : 'scene';
      showRun(match.key, undefined, view);
      return true;
    }
    const match = data?.rows.find(({ key, attempts }) => savedRunId(key) === job.run_key &&
      attempts.some(({ job_id }) => job_id === job.job_id));
    if (!match) return false;
    showRun(match.key, job.job_id);
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

  const projectFailure = snapshot.error ?? stoppedPlan.error;
  const serviceIssue = connection.isError ? studioRegionIssue('service', connection.error, true) : undefined;
  const failure = serviceIssue?.reason ?? commandError ?? (projectFailure ? apiErrorMessage(projectFailure) : null);
  const retryPlan = () => {
    void projection.refetch();
  };
  const planError = (name: 'runs' | 'pipelineMap' | 'results'): StudioRegionState | undefined => projection.error
    ? { kind: 'failed', issue: studioRegionIssue(name, projection.error), retry: retryPlan }
    : undefined;
  const planUnavailable: StudioRegionState | undefined = planError('runs') ??
    (!data ? { kind: 'loading' } : undefined);
  const mapUnavailable = planError('pipelineMap');
  const renderMap = (body: ReactNode) => mapUnavailable
    ? <StudioRegion state={mapUnavailable} />
    : body;
  function retry() {
    setCommandError(undefined);
    if (connection.isError) {
      connection.refetch();
      return;
    }
    if (snapshot.error && snapshot.pending) void editor.flush().catch((error) => setCommandError(apiErrorMessage(error)));
    connection.refetch();
    void projection.refetch();
    if (checkpointAttempt) void stoppedPlan.refetch();
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

  function evaluateTrain(source: PipelineMapStepSource, projection: StudioSavedTrainRuns, stopped?: StoppedEvaluationSubject) {
    void perform(async () => {
      const reviewed = stopped
        ? await fetchSavedTrainRuns(saved.project_id, saved.revision, editor.signal, stopped.jobId)
        : projection;
      const id = await createEvaluation(editor, source, plan, reviewed, stopped);
      commandScope.selectCreatedStep(id);
    });
  }

  function openStepMenu(kinds: StepInsertion) {
    const revision = saved.revision;
    insertMenu.open(Object.fromEntries(Object.entries(kinds).map(([kind, create]) =>
      [kind, () => { if (latestRevision.current === revision) create(); }])));
  }

  const stoppedCoordinate = row && stoppedPlan.data?.rows.find(({ key }) => savedRunId(key) === savedRunId(row.key))
    ?.attempts.find(({ job_id }) => job_id === checkpointAttempt)?.stopped_checkpoint;
  const stoppedSubject = row && checkpointAttempt && stoppedCoordinate
    ? { runId: savedRunId(row.key), jobId: checkpointAttempt, at: stoppedCoordinate } : undefined;
  const stoppedNewStep = stoppedSubject && row && data && bindStudioCommand('newStep', () => openStepMenu({
    evaluate: () => evaluateTrain(
      { stepId: row.key.step, selection: { kind: 'rows', stepId: row.key.step, runIds: [stoppedSubject.runId] } },
      data, stoppedSubject),
  }), { enabled: canAuthorContinuation && !busy });

  function editEvaluation(name: string, patch: ProjectValueEdit) {
    if (!step || !evaluationEditor || !ready) return;
    const source = evaluationEditor.source.id;
    void perform(async () => {
      editor.editFiles(name, [{ address: { kind: 'step', id: step.id }, patches: [patch] }], {
        reviewedRevision: saved.revision, dependencies: [{ kind: 'step', id: source }],
      });
      await editor.flush();
    });
  }

  function removeSweepException(id: string) {
    if (!step || !ready || busy || saved.project.sweep || !authored?.sweep?.exceptions?.[id]) return;
    void perform(async () => {
      editor.editFiles(studioName('remove'), [{
        address: { kind: 'step', id: step.id },
        patches: [{ op: 'replace', path: authoredPointer(['sweep', 'exceptions', id]), value: null }],
      }], { reviewedRevision: saved.revision });
      await editor.flush();
    });
  }

  function freezeEvaluationSubset() {
    if (!step || !variant || !subject || !freezeAttempts || !ready) return;
    void perform(async () => {
      const reviewed = await prepareSubsetToRows(saved.project_id, {
        revision: saved.revision, variant, step: step.id, attempts: freezeAttempts,
      }, editor.signal);
      editor.editFiles(studioName('toRows'), reviewed.files, {
        reviewedRevision: reviewed.revision, dependencies: [{ kind: 'step', id: subject.source_step_id }],
      });
      await editor.flush();
    });
  }

  function editEvaluationCheckpoint(rule: EditableCheckpointRule) {
    if (!step || !subject || !evaluationCheckpointRule || !ready) return;
    if (checkpointScope) {
      void applyCheckpointSweepEdit({ kind: 'set', scope: { ...checkpointScope,
        ...('rows' in checkpointScope && { rows: [...checkpointScope.rows] }),
      }, rule });
      return;
    }
    const name = 'Set Evaluate checkpoints';
    const dependencies = [{ kind: 'step' as const, id: subject.source_step_id }];
    let address: { kind: 'step' | 'variant'; id: string } = { kind: 'step', id: step.id };
    let patch: ProjectValueEdit = { op: 'replace', path: '/inputs/subject/checkpoint', value: rule };
    if (checkpointVariant) {
      const row = saved.project.sweep?.rows?.[checkpointVariant];
      if (!row) return;
      const key = `steps.${step.id}.inputs.subject.checkpoint`;
      address = { kind: 'variant', id: checkpointVariant };
      dependencies.push({ kind: 'step', id: step.id });
      patch = row.patches
        ? { op: key in row.patches ? 'replace' : 'add', path: authoredPointer(['patches', key]), value: { value: rule } }
        : { op: 'add', path: '/patches', value: { [key]: { value: rule } } };
    }
    void perform(async () => {
      editor.editFiles(name, [{ address, patches: [patch] }], { reviewedRevision: saved.revision, dependencies });
      await editor.flush();
    });
  }

  async function applyCheckpointSweepEdit(edit: StudioCheckpointRuleRequest['edit']) {
    if (!step || !subject || !namedRun || !ready) return;
    await perform(async () => {
      const reviewed = await prepareCheckpointRuleEdit(saved.project_id, {
        revision: saved.revision, run: namedRun, edit,
      }, editor.signal);
      editor.editFiles(edit.kind === 'to_sweep' ? 'Sweep Evaluate checkpoints' : 'Set Evaluate checkpoints',
        reviewed.files, { reviewedRevision: reviewed.revision,
          dependencies: [{ kind: 'step', id: subject.source_step_id }] });
      await editor.flush();
    });
  }

  function chooseReportFigure(revision: string, figure: ReportFigureSource) {
    setReportChoices(undefined);
    if (!primaryVariant || !ready || revision !== saved.revision) return;
    void perform(async () => {
      const id = await createReport(editor, revision, figure, humanizeLabel(figure.name));
      commandScope.selectCreatedStep(id, 'document');
    });
  }

  function stepInsertion(source?: PipelineMapStepSource): { kinds: StepInsertion; enabled: boolean } | undefined {
    if (!primaryVariant) return undefined;
    if (!source) {
      if (step && (state.selection?.kind !== 'steps' || state.selection.stepIds.length === 1)) {
        return stepInsertion({
          stepId: step.id,
          selection: state.selection && 'stepId' in state.selection && state.selection.stepId === step.id
            ? state.selection : null,
        });
      }
      if (!canCreateFirstTrain(saved.project)) return undefined;
      const revision = saved.revision;
      return { kinds: { train: () => void perform(async () => {
        const id = await createFirstTrain(editor, revision);
        commandScope.selectCreatedStep(id, 'model');
      }) }, enabled: ready && !busy };
    }
    const sourceStep = plan.steps.find(({ id }) => id === source.stepId);
    if (sourceStep?.kind === 'evaluate' && data) {
      const rows = source.selection ? selectedRuns(sourceStep, source.selection) : sourceStep.runs;
      const input = analyzeInput(sourceStep, rows, data, !!saved.project.sweep);
      if (!input) return undefined;
      return { kinds: { analyze: () => void perform(async () => {
        const id = await createAnalyze(editor, data.revision, input);
        commandScope.selectCreatedStep(id, 'analysis');
      }) }, enabled: ready && !busy };
    }
    if (sourceStep?.kind === 'analyze' && data) {
      const rows = new Set((source.selection ? selectedRuns(sourceStep, source.selection) : sourceStep.runs).map(({ id }) => id));
      const figures = (data?.window_map?.analysis_elements ?? []).filter(isReportFigureSource)
        .filter(({ key }) => rows.has(savedRunId(key)));
      if (!figures.length) return undefined;
      const selected = step?.id === sourceStep.id && namedRun && figures.find(({ key, name }) =>
        savedRunId(key) === savedRunId(namedRun) && name === analyze.selectedFigure);
      return { kinds: { report: () => {
        const choice = selected || (figures.length === 1 ? figures[0] : undefined);
        if (choice) chooseReportFigure(saved.revision, choice);
        else setReportChoices({ revision: saved.revision, figures });
      } }, enabled: ready && !busy };
    }
    if (!onNewStep || sourceStep?.kind !== 'train') return undefined;
    const runs = source.selection ? selectedRuns(sourceStep, source.selection) : sourceStep.runs;
    return {
      kinds: {
        train: () => void perform(async () => { await onNewStep(source, commandScope); }),
        ...(data && managedTrialDomain(data, runs) && { evaluate: () => evaluateTrain(source, data) }),
      },
      enabled: canAuthorContinuation && !busy && runs.some(({ status }) => status !== 'cannot_run'),
    };
  }

  function newStepCommand(source?: PipelineMapStepSource, subject?: string) {
    const insertion = stepInsertion(source);
    return insertion && bindStudioCommand('newStep', () => openStepMenu(insertion.kinds),
      { enabled: insertion.enabled, subject });
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
      setResultAttempt(undefined);
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
  const runCommand = bindStudioCommand('run', run, {
    enabled: ready && !busy && !machineQuote && Boolean(machineReady) && runnableRows.length > 0,
    name: selectedJobs.length === 1 && selectedJobs[0].target?.kind === 'session' && selectedJobs[0].status === 'failed'
      ? 'rerun' : 'run',
  });
  const nextStep = newStepCommand();
  const runCommands: StudioCommand[] = [runCommand];
  if (step?.kind === 'train' || step?.kind === 'evaluate') {
    for (const [id, mode] of [['stop', 'checkpoint'], ['forceStop', 'force']] as const) {
      const eligible = selectedJobs.filter((job) => jobAllowsCommand(job, id)).map(({ job_id }) => job_id);
      if (!eligible.length && (id === 'forceStop' || step.kind === 'evaluate')) continue;
      runCommands.push(bindStudioCommand(id, () => stop(eligible, mode), { enabled: !busy && eligible.length > 0 }));
    }
  }
  if (step?.kind === 'train') {
    if (resumable && jobAllowsCommand(resumable, 'resume')) {
      runCommands.push(bindStudioCommand('resume', () => {
        void perform(async () => { await actions.resume(resumable.job_id); });
      }, { enabled: !busy }));
    }
    runCommands.push(bindStudioCommand('openRun', () => { if (exactRow) showRun(exactRow.key); }, { enabled: !!exactJob }));
    if (nextStep) runCommands.push(nextStep);
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
    bindStudioCommand('rename', () => {
      void perform(async () => {
        const revision = await editor.flush();
        const latest = editor.getSnapshot();
        if (latest.outside) throw new Error('Resolve the outside change before renaming.');
        setRename({ name: latest.configuration.project, revision });
      });
    }, {
      enabled: !busy && !outside,
    }),
    ...runCommands.filter(({ id }) => id !== 'newStep'),
    windowState.mode !== 'map' && focusView === 'results' && stoppedNewStep ? stoppedNewStep : nextStep,
    ...(focusedModel?.commands ?? []),
    ...analyze.commands,
    ...report.commands,
    step && selectedAxis && bindSweepFold(step.id, selectedAxis, folded?.id === selectedAxis.id, cardPresentation, changePresentation),
    step && newSweepCommand(step.id),
    windowState.mode !== 'map' && focusView === 'runs' && pivotCommand(focusSlot),
    bindStudioCommand('undo', () => { void moveHistory('undo'); }, {
      enabled: !!snapshot.undoName && !busy && !snapshot.saving && !outside,
      subject: snapshot.undoName === studioName('outsideChange') ? snapshot.undoName : undefined,
    }),
    bindStudioCommand('redo', () => { void moveHistory('redo'); }, {
      enabled: !!snapshot.redoName && !busy && !snapshot.saving && !outside,
      subject: snapshot.redoName === studioName('outsideChange') ? snapshot.redoName : undefined,
    }),
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
      map={renderMap(<StudioPipelineMap {...mapProps}
        activeStepId={selection.stepId} viewportRef={mapViewport} fitReady={!!data}
        presentation={presentation} onPresentationChange={changePresentation} />)}
      mapDrawer={renderMap(<StudioPipelineMapDrawer {...mapProps} />)}
      compactMap={renderMap(<StudioPipelineMap {...mapProps} mode="line" />)}
      variant={(variants.length > 1 || missingVariant) && <Select aria-label={studioName('variant')}
        value={variant} disabled={!data} onValueChange={selection.setVariant}
        items={[
          ...(missingVariant && variant ? [{ value: variant, label: variant, disabled: true }] : []),
          ...variants.map((id) => ({ value: id, label: id })),
        ]} />}
      overlay={<>
        {rename && <Dialog ariaLabel={studioName('rename')} manageFocus
          onClose={() => { if (!busy) setRename(undefined); }}
          panelClassName="w-[min(22rem,90vw)]">
          <form className="grid gap-2" onSubmit={(event) => {
            event.preventDefault();
            void perform(async () => {
              await editor.rename(rename.name, rename.revision);
              setRename(undefined);
            });
          }}>
            <label className="grid gap-1">
              {studioName('name')}
              <TextField data-dialog-autofocus value={rename.name} disabled={busy} required
                onValueChange={(name) => setRename({ ...rename, name })} />
            </label>
            {rename.revision !== saved.revision || outside
              ? <div role="alert">{studioReason('nameChangedDuringEdit', studioName('project'))}</div>
              : commandError && <div role="alert">{commandError}</div>}
            <Button type="submit" className="justify-self-end" disabled={busy || !!outside || rename.revision !== saved.revision}>
              {studioName('rename')}
            </Button>
          </form>
        </Dialog>}
        {focusedModel?.overlay}
        {analyze.overlay}
        {report.overlay}
        {machineQuote && <StudioMachineApproval quote={machineQuote} busy={busy} error={commandError}
          onApprove={approveMachine} onCancel={() => {
            setMachineQuote(undefined);
            setCommandError(undefined);
          }} />}
        {primaryVariant && reportChoices?.revision === saved.revision && <Dialog ariaLabel={studioName('report')} manageFocus
          onClose={() => setReportChoices(undefined)}
          panelClassName="grid w-[min(24rem,90vw)] gap-px p-1">
          {reportChoices.figures.map((figure) => <Button variant="ghost" align="start"
            key={JSON.stringify([figure.key, figure.name])}
            onClick={() => chooseReportFigure(reportChoices.revision, figure)}>
            {figure.name}{figure.key.row !== 'default' && ' · ' + figure.key.row}
          </Button>)}
        </Dialog>}
        {insertMenu.menu}
        {sweepStep && <Dialog ariaLabel={studioName('newSweep')} onClose={() => setSweepStep(undefined)} manageFocus
          panelClassName="grid max-h-[75dvh] w-[min(24rem,90vw)] gap-2 overflow-auto">
          <IconButton className="justify-self-end" label={studioName('close')} icon={<X aria-hidden="true" />}
            data-dialog-autofocus onClick={() => setSweepStep(undefined)} />
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
        <Button {...commandAttributes(retryCommand)} onClick={retryCommand.invoke}>{studioName('retry')}</Button>
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
      serviceFailure={serviceIssue && { kind: 'failed', issue: serviceIssue, retry: connection.refetch }}
      renderView={(view, slot) => {
        const details = slot === 'primary' ? primary : secondary;
        const withProblem = (body: ReactNode) => revealed?.target.view === view && slot === focusSlot
          ? <ProblemDestination reveal={revealed} action={graphRepair && <Button
            disabled={!ready || snapshot.saving || busy} onClick={removeRefusedVariation}>
            {studioName('removeVariation')}
          </Button>}>{body}</ProblemDestination> : body;
        if (view === 'analysis' || view === 'figures') {
          const detail = analyze[view];
          return { ...detail, body: withProblem(detail.body),
            drawer: <StudioRegion state={analyze.catalog}>{detail.drawer}</StudioRegion> };
        }
        if ((view === 'task' || view === 'objective') && trialStep) {
          if (!namedRun) return { body: noView };
          if (trial.error || !trial.view) return { body: withProblem(
            <StudioRegion state={trial.error
              ? { kind: 'failed', issue: studioRegionIssue(view, trial.error), retry: trial.retry }
              : { kind: 'loading' }} />,
          ) };
          const value = trial.view;
          const commit = trial.commit;
          if (view === 'task') return {
            header: <TrialTaskPicker view={value} commit={commit} />,
            body: withProblem(<TrialTaskForm view={value} commit={commit} />),
          };
          const selected = selectedTerm?.step === step?.id ? selectedTerm?.term : undefined;
          const choose = (term: string) => { if (step) setSelectedTerm({ step: step.id, term }); };
          return {
            header: value.objective_document,
            body: withProblem(<TrialObjectiveTable view={value} commit={commit} selected={selected} select={choose} />),
            inspector: <TrialObjectiveInspector view={value} commit={commit} selected={selected} />,
          };
        }
        if (view === 'model' || view === 'task' || view === 'scene') {
          const scope = slot === focusSlot && view === focusView ? activeDocumentScope : undefined;
          return { ...details[view], header: <>
            {details[view].header}
            {details[view].path}
            {scope && scope.users.size > 1 && <span className="whitespace-nowrap text-sm text-shell-muted" role="status">
              {' · '}{studioCountOf(scope.lit.size, scope.users.size)}
            </span>}
          </>, body: withProblem(details[view].body) };
        }
        if (!step) return { body: noView };
        if (view === 'document') return { body: withProblem(report.body ?? noView) };
        if (view === 'runs') {
          const runSelection = { step, state, onSelectionChange: mapProps.onSelectionChange };
          const pivot = pivotCommand(slot);
          const pivotActive = pivotSlots[slot] === step.id;
          const run = state.focus?.stepId === step.id ? step.runs.find(({ id }) => id === state.focus?.runId) : undefined;
          const exceptions = authoredMap?.steps.find(({ step_id, variant: id }) => step_id === step.id && id === variant)
            ?.exceptions?.map(({ id, count, rule }) => ({
              id, count,
              terms: [
                ...Object.entries(rule.where?.axes ?? {}).map(([axis, values]) => {
                  const options = step.sweeps.find(({ id }) => id === axis)?.values;
                  return `${axis} ${values.map((value) => options?.find(({ id }) => id === value)?.label ?? value).join(', ')}`;
                }),
                ...(rule.where?.rows ?? []),
              ],
              changes: Object.entries(rule.patches ?? {}).flatMap(([path, patch]) => {
                if (!patch) return [];
                return [{ id: path, label: exceptionFieldLabel(path),
                  value: 'value' in patch ? JSON.stringify(patch.value) : studioName('remove') }];
              }),
              removable: Boolean(!saved.project.sweep && authored?.sweep?.exceptions?.[id]),
            })) ?? [];
          return {
            header: <StudioRunControls commands={step.kind === 'train' || step.kind === 'evaluate' || lightSteps.has(step.id)
              ? runCommands : undefined}
              machine={<StudioRunMachine catalog={machineCatalog.data} value={requestedMachine}
                disabled={busy || machineCatalog.isPending}
                onChange={(value) => setMachines((current) => ({ ...current, [step.id]: value }))} />}
              pivotCommand={pivot} pivotActive={pivotActive} reason={run?.reason} />,
            drawer: (authoring || step.inputs.length > 0 || exceptions.length > 0) ? <div className="space-y-3 text-sm">
              {authoring && <StudioTrainAuthoringForm {...authoring} run={namedRun} fieldRows={detailInputs.fieldRows}
                  focusFirstField={ready && focusCreatedStep === step.id && focusSlot === slot}
                  onDetachedEditError={setCommandError}
                  onFirstFieldFocused={() => setFocusCreatedStep(undefined)} />}
              {exceptions.length > 0 && <StudioSweepOverview axes={[]} exceptions={exceptions}
                onRemoveException={ready && !busy ? removeSweepException : undefined} />}
              {step.inputs.length > 0 && <section aria-label={studioName('inputs')} className="space-y-1">
                <h3 className="font-semibold">{studioName('inputs')}</h3>
                <dl>{step.inputs.map(({ id, label, sourceStepId, subset }) => <div key={id} className="border-t border-shell-divider py-1">
                  <dt>{label}</dt>
                  <dd className="text-shell-muted">{plan.steps.find((item) => item.id === sourceStepId)?.label}</dd>
                  {id === 'subject' && subjectCheckpoint !== undefined &&
                    <dd className="text-shell-muted">{checkpointLabel(subjectCheckpoint)}</dd>}
                  {subset && <dd className="text-shell-muted">
                    {[...subset.terms, studioCountOf(subset.count, subset.total)].join(' · ')}
                    {subset.memberGroups?.map(({ label, members }) => <div key={members[0]?.id} className="mt-1">
                      {label && <div className="font-medium">{label}</div>}
                      <ol aria-label={label ? studioAccessibleName('members', label) : studioName('members')}
                        className="list-decimal space-y-0.5 ps-5">
                        {members.map(({ id, label }) => <li key={id}>{label}</li>)}
                      </ol>
                    </div>)}
                  </dd>}
                </div>)}</dl>
              </section>}
              {evaluationCheckpointRule &&
                <CheckpointRuleEditor key={JSON.stringify(['checkpoint', step.id, variant, saved.revision, checkpointScope])}
                  initial={evaluationCheckpointRule}
                  disabled={!ready || busy} readOnly={Boolean(stepCheckpointRule && subject?.checkpoint_axis)}
                  onSave={editEvaluationCheckpoint}
                  onToSweep={canSweepCheckpoint ? () => { void applyCheckpointSweepEdit({ kind: 'to_sweep' }); } : undefined} />}
              {evaluationEditor && data && variant && <StudioEvaluateAuthoring key={`${step.id}:${saved.revision}`}
                authored={{ ...evaluationEditor.authored, hasWhere: evaluationEditor.authored.subject.where !== undefined }}
                source={evaluationEditor.source} sourceRows={stepSourceRows(data, evaluationEditor.source.id, variant)}
                domain={evaluationEditor.domain}
                editSubject={evaluationEditor.authored.subject.checkpoint === 'last'}
                disabled={!ready || busy} focusTrialSet={focusCreatedStep === step.id && focusSlot === slot}
                onTrialSetFocused={() => setFocusCreatedStep(undefined)} onEdit={editEvaluation}
                onToRows={freezeAttempts ? freezeEvaluationSubset : undefined} />}
            </div> : undefined,
            inspector: selectedRuns(step, state.selection).length > 0
              ? <StudioRunDetails step={step} state={state}
                detail={selectedEvaluation?.checkpoint_series
                  ? <StudioEvaluationAttemptDetails evaluation={selectedEvaluation} /> : undefined} /> : undefined,
            body: withProblem(<StudioRuns plan={plan} {...runSelection}
              region={planUnavailable ?? (step.kind === 'train' || step.runs.length > 0
                ? { kind: 'result', missing: [] } : { kind: 'empty' })}
              pivotActive={Boolean(pivot) && pivotActive} upstreamDetail={checkpointLineage()} />),
          };
        }
        if (view === 'training' && authoring) {
          return { body: withProblem(<StudioTrainAuthoringForm {...authoring} view={view} run={namedRun}
            fieldRows={detailInputs.fieldRows} />) };
        }
        const resultsUnavailable = view === 'results' && (planError('results') ?? planUnavailable);
        if (resultsUnavailable) return {
          body: <StudioRegion state={resultsUnavailable} />,
        };
        if (view === 'results' && selectedEvaluation?.checkpoint_series && ready) return {
          body: <StudioEvaluationResults evaluation={selectedEvaluation} onReplay={(update) => {
            evaluationReplay.chooseCheckpoint(update);
            setWindowState((current) => showProjectWindowView(current, 'scene', slot));
            requestSlotFocus((count) => count + 1);
          }} />,
        };
        if (view === 'results' && row && ready) return { body: <div className="p-3">
          {stoppedSubject && stoppedNewStep && <section aria-labelledby={stoppedCheckpointId}
            className="flex items-center gap-2">
            <h3 id={stoppedCheckpointId} className="font-semibold">{checkpointLabel(stoppedSubject.at)}</h3>
            <Button {...commandAttributes(stoppedNewStep)}
              disabled={!stoppedNewStep.enabled} onClick={stoppedNewStep.invoke}>{studioName('newStep')}</Button>
          </section>}
          <TrainResults key={savedRunId(row.key)} project={saved.project_id} revision={saved.revision}
            evaluations={data?.window_map?.evaluations ?? []} mapRows={data?.window_map?.rows ?? []}
            rows={data!.rows} row={row} attempt={attempt} />
        </div> };
        return { body: withProblem(noView) };
      }}
    />
  );
}

function ProblemDestination({ reveal, action, children }: { reveal: ProblemReveal; action?: ReactNode; children: ReactNode }) {
  const reason = useRef<HTMLParagraphElement>(null);
  useEffect(() => {
    reason.current?.focus();
  }, [reveal]);
  return <div className="flex h-full min-h-0 flex-col">
    <div className="flex items-center justify-between gap-2 border-b border-shell-divider px-3 py-2 text-sm">
      <p ref={reason} tabIndex={-1}>
        <span className="font-medium">{reveal.target.label}</span>{' · '}
        {action && <>{reveal.problem.path}{' · '}</>}{problemReason(reveal.problem)}
      </p>
      {action}
    </div>
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
          ? <Button variant="ghost" align="start" className="block w-full !h-auto py-1 !whitespace-normal" onClick={() => onChoose(problem)}>{entry}</Button>
          : <div className="px-control-inline py-1">{entry}</div>}
      </li>;
    })}</ul>
  </section>)}</div>;
}
