import { useCallback, useEffect, useId, useMemo, useRef, useState, useSyncExternalStore, type ReactNode } from 'react';
import { useStore } from 'zustand';
import { keepPreviousData, useQuery, useQueryClient } from '@tanstack/react-query';
import { approveSavedRun, fetchExecutionTargets, fetchSavedTrainRuns, quoteSavedRun, runSavedTrain } from '@/api/jobs';
import { apiErrorMessage, isRefusal, isServiceFailure, isUnanswered, landsOnGesture } from '@/api/request';
import { prepareCheckpointRuleEdit, prepareSubsetToRows } from '@/api/studioAuthoring';
import { StudioRegion, studioRegionIssue, type StudioRegionState } from '@/components/activity/StudioRegion';
import {
  studioAccessibleName, studioCountOf, studioEvaluateValueSource, studioFormat, studioName, studioPlaceName, studioProblemDetail, studioReason, studioRunReason, type StudioPlace,
} from '@/components/activity/studioVocabulary';
import { HeaderActivity } from '@/components/layout/HeaderActivity';
import {
  StudioPipelineMap, StudioPipelineMapDrawer, type MapStepRename, type PipelineMapStepSource,
} from '@/components/pipeline/StudioPipelineMap';
import { isMemberName } from './projectMemberName';
import { KeyField } from '@/components/ui/KeyField';
import { stepElementName } from './analysisElements';
import { canvasViewportCommands, type CanvasViewport } from '@/components/canvas/StudioCanvas';
import { useStepInsertMenu, type StepInsertion } from '@/components/pipeline/StepInsertMenu';
import { InspectorRow, StudioRunControls, StudioRuns, StudioStepInspector, type RunMention } from '@/components/pipeline/StudioRuns';
import type { RunMenu, StepMenu } from '@/components/pipeline/CommandContextMenu';
import type { ActivityResultState } from '@/components/activity/StudioActivity';
import { useResultValidity } from './useResultValidity';
import { useFollowingQuery } from '@/hooks/useFollowingQuery';
import { StudioMachineApproval, studioRunMachineControl } from '@/components/pipeline/StudioRunMachine';
import { SweepExceptions } from '@/components/pipeline/SweepExceptions';
import { useDrawerSections, useFieldFocus, type FieldFocusRequest } from '@/features/projectWindow/DrawerForms';
import { X } from 'lucide-react';
import { Button, IconButton } from '@/components/ui/Button';
import { Dialog } from '@/components/ui/Dialog';
import { Select } from '@/components/ui/Select';
import { TextField } from '@/components/ui/TextField';
import { humanizeLabel } from '@/components/values/humanizeLabel';
import type { PipelineCardPresentation } from '@/components/pipeline/pipelineMapLayout';
import type { PlanRunView, PlanStepView, PlanView } from '@/features/pipeline/planView';
import { selectRun, selectStep, selectSweepValue, selectedRuns, type PipelineSelection, type PipelineSelectionState } from '@/features/pipeline/selection';
import { ProjectWindow } from '@/features/projectWindow/ProjectWindow';
import { useHeldGestures } from '@/features/projectWindow/heldGestures';
import { ProblemRow, StudioProblems } from '@/features/projectWindow/StudioProblems';
import { holds } from './model/modelGraph';
import { bindStudioCommand, commandAttributes, jobAllowsCommand, type StudioCommand } from '@/features/projectWindow/studioCommands';
import {
  projectWindowSlotView, showProjectWindowView, useProjectWindowState, type ProjectWindowSlot, type ProjectWindowView,
} from '@/features/projectWindow/projectWindowState';
import {
  ModelAddressSchema, StepReferenceSchema, type PresentationDocument, type PresentationPosition, type ProjectValueEdit,
  type StudioActivityJob, type StudioJobSubmission, type StudioMachineQuote, type StudioProjectDocument, type StudioProjectWindowMapProblem,
  type StudioSavedTrainKey, type StudioSavedTrainRuns,
  type StudioCheckpointRuleRequest, type JsonValue, type SweepExceptionTerms, type StudioPlanReviewSource, type StudioProjectWindowMapException, type ModelAddress, type PatchValue,
  type StudioClassField,
} from '@/generated/studioContracts';
import { useConnection } from '@/api/connection';
import { useStudioJobActions, useStudioJobs } from '@/hooks/useStudioJobs';
import { authoredPointer, directTrainStep, modelDocumentId } from './authoredEdits';
import { withBaseDescendants } from './baseDescendants';
import { useManagedProjectDetails, useSavedEvaluationScene, type NamedDocumentScope } from './managedProjectDetails';
import {
  choosePick, clearSelection, isModelAddress, selectElement, useElementSelectionStore, usePick, type PickTarget,
} from '@/features/projectWindow/elementSelection';
import { createCheckpointTrain } from './createCheckpointTrain';
import { analyzeInput, createAnalyze, createReport, isReportFigureSource, type ReportFigureSource } from './createManagedStep';
import { createTrain } from './createTrain';
import { trainingDocumentId } from './trainingDocument';
import { checkpointLabel, createEvaluation, editableCheckpointRule, exactCheckpoint, managedTrialDomain, ordinaryEvaluation, type EditableCheckpointRule, type StoppedEvaluationSubject } from './evaluateAuthoring';
import { bindProjectEditor } from './openProjectEditor';
import type { ProjectEditController } from './projectEditController';
import { bestRankGroups, projectWindowPlan, planVariants, savedRunId, stepSourceRows, unvalued } from './projectWindowPlan';
import { placeField, placeMember, placeView, removeGraphVariation, type Reveal } from './projectWindowProblems';
import { settledWindowSelection, useProjectWindowSelection } from './projectWindowSelection';
import { planWithRefusal, problemKey, type GestureFailures, type Refusal } from './refusals';
import { CheckpointRuleEditor, EvaluateSubjectFields, SubjectSubset, TrialSetEditor, type EvaluateField } from './StudioEvaluateAuthoring';
import { connection, selectionProblem } from './selectionEdits';
import { StudioEvaluationAttemptDetails } from './StudioEvaluationResults';
import { focusedResult, SavedResults } from './SavedResults';
import { filesRemoval, RunRemovalContext, useRunRemovalDialog } from './runRemoval';
import {
  jsonPointerPath, StudioTrainAuthoringForm, SweepFieldItems, useTrainClassFields, useTrainFieldLabels,
} from './StudioTrainAuthoringForm';
import {
  TrialObjectiveInspector, TrialObjectiveOutline, TrialObjectiveTable, TrialTaskForm, TrialTaskInspector, TrialTaskOutline, useTrialAuthoring,
  type TrialChangeScope,
} from './StudioTrialAuthoring';
import { documentFamily, useProjectDocuments, type DocumentRole } from './useProjectDocuments';
import { useManagedAnalyze } from './useManagedAnalyze';
import { useManagedReport } from './useManagedReport';
import { ScrollFade } from '@/components/ui/ScrollFade';
import type { AxisMenu } from '@/components/pipeline/PipelineRunMarks';
import { OfferedSets, setAuthoring } from './setAuthoring';

type EditorSession = ReturnType<typeof bindProjectEditor>;

function valueText(value: JsonValue) {
  return typeof value === 'string' ? value : JSON.stringify(value);
}

/**
 * The label of the field a sweep exception patches: a Train field's label from the server, or a model field's name.
 * A path no field of the step has (the exception outlived its field) is drawn by its last segment.
 */
function exceptionFieldLabel(path: string, trainLabels: ReadonlyMap<string, string>) {
  if (!path.startsWith('{')) return trainLabels.get(path) ?? humanizeLabel(path.split('.').at(-1)!);
  try {
    const address = ModelAddressSchema.safeParse(JSON.parse(path));
    const field = address.success && address.data.element.kind === 'field' ? address.data.element.path.at(-1) : undefined;
    return field == null ? path : humanizeLabel(String(field));
  } catch {
    return path;
  }
}
/**
 * What an exception's patches change, one text each. A class chosen for the runs reads by the field and the class's
 * label (`Optimizer AdamW`); the parameters choosing it wrote are the class's and are not listed.
 */
function exceptionPatchTexts(
  patches: Readonly<Record<string, PatchValue | null>>, trainLabels: ReadonlyMap<string, string>,
  classes: readonly StudioClassField[],
): string[] {
  const entries = Object.entries(patches);
  const chosen = classes.flatMap((field) => {
    const path = jsonPointerPath(field.id);
    for (const [at, patch] of entries) {
      const option = patch && 'value' in patch && at.startsWith(path + '.') && !at.slice(path.length + 1).includes('.')
        ? field.options.find(({ value }) => value === patch.value) : undefined;
      if (option) return [{ path, tag: at, text: `${field.label} ${option.label}` }];
    }
    return [];
  });
  return entries.flatMap(([path, patch]) => {
    const pick = chosen.find((entry) => path === entry.tag);
    if (pick) return [pick.text];
    if (!patch || chosen.some((entry) => path.startsWith(entry.path + '.'))) return [];
    return [`${exceptionFieldLabel(path, trainLabels)} ${'value' in patch ? valueText(patch.value) : studioName('remove')}`];
  });
}

type WindowPanel = { kind: 'activity' } | {
  kind: 'problems';
  scope: { step: string; variant: string | null; element?: ModelAddress } | null;
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

export default function StudioProjectWorkspace({
  document, onNewStep = checkpointNewStep,
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
  return <ManagedProjectWindow session={load.session} onNewStep={onNewStep} />;
}

function ManagedProjectWindow({ session, onNewStep }: { session: EditorSession } & ProjectWindowCommands) {
  const stoppedCheckpointId = useId();
  const { editor } = session;
  const snapshot = useSyncExternalStore(editor.subscribe, editor.getSnapshot);
  const { saved, configuration, pending, outside } = snapshot;
  const jobs = useStudioJobs();
  const connected = useConnection().kind === 'connected';
  const queries = useQueryClient();
  const actions = useStudioJobActions();
  const [windowState, setWindowState] = useProjectWindowState(saved.project_id);
  const drawerSections = useDrawerSections(windowState, setWindowState);
  const elements = useElementSelectionStore();
  const armedPick = usePick(elements);
  const mapPick = useMemo(() => armedPick && {
    accepts: armedPick.accepts, choose: (target: PickTarget) => { choosePick(elements, target); },
  }, [armedPick, elements]);
  const selectedElements = useStore(elements, (current) => current.selection);
  const mapViewport = useRef<CanvasViewport>(null);
  const selection = useProjectWindowSelection(Object.keys(saved.project.steps ?? {})[0] ?? null);
  const [commandError, setCommandError] = useState<string>();
  /** The last refused gesture, shown on the object it acted on. */
  const [refusal, setRefusal] = useState<Refusal>();
  /** The refusal of the selected Train step's authoring read, which is drawn on the step while it holds. */
  const [readRefusal, setReadRefusal] = useState<Refusal & { kind: 'step' }>();
  const commandFailed = (error: unknown) => {
    if (!isUnanswered(error)) setCommandError(apiErrorMessage(error));
  };
  const [machines, setMachines] = useState<Readonly<Record<string, string>>>({});
  const [machineQuote, setMachineQuote] = useState<StudioMachineQuote>();
  const [rename, setRename] = useState<{ name: string; revision: string }>();
  const [busy, setBusy] = useState(false);
  const [renamingStep, setRenamingStep] = useState<MapStepRename['editing']>(null);
  /** The step whose key the inspector is editing. */
  const [keyingStep, setKeyingStep] = useState<string>();
  const [stepFocus, setStepFocus] = useState<MapStepRename['focus']>(null);
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
    reveal: Reveal;
  }>();
  const [excludedUsers, setExcludedUsers] = useState<Readonly<Record<string, readonly string[]>>>({});
  /** The exception a field's source revealed, shown in its step's Exceptions while the revision stands. */
  const [exceptionReveal, setExceptionReveal] = useState<{ revision: string; step: string; id: string }>();
  /** A field of a step's definition to put the keyboard on once the Runs view's inspector shows it. */
  const [fieldFocus, setFieldFocus] = useState<FieldFocusRequest & { step: string }>();
  const [focusFirstModel, setFocusFirstModel] = useState<string>();
  const insertMenu = useStepInsertMenu();
  const latestRevision = useRef(saved.revision);
  latestRevision.current = saved.revision;
  const [resultAttempt, setResultAttempt] = useState<{ runId: string; jobId: string }>();
  const [reportChoices, setReportChoices] = useState<{ revision: string; figures: ReportFigureSource[] }>();
  const inFlight = useRef(false);
  const [sweepStep, setSweepStep] = useState<string>();
  const submission = useRef<{ intent: string; token: string }>();
  const jobsVersion = `${jobs.data?.cursor}:${jobs.data?.remote_version}`;
  const projection = useFollowingQuery({
    queryKey: ['saved-train-runs', saved.project_id, saved.revision],
    queryFn: ({ signal }) => fetchSavedTrainRuns(saved.project_id, saved.revision, signal),
    placeholderData: keepPreviousData,
    gcTime: 0,
  }, jobsVersion);
  const data = projection.data?.revision === saved.revision ? projection.data : undefined;
  /** The plan shown is of an earlier revision while the shown revision's plan is read. */
  const planPending = !!projection.data && !data;
  const refetchProjection = projection.refetch;
  const refreshProjection = useCallback(() => refetchProjection(), [refetchProjection]);
  const validity = useResultValidity({
    projectId: saved.project_id, revision: saved.revision, refresh: refreshProjection, onError: commandFailed,
  });
  const variants = useMemo(() => planVariants(projection.data), [projection.data]);
  const variant = selection.variantId ?? variants[0] ?? null;
  const missingVariant = selection.variantId !== null && !variants.includes(selection.variantId);
  const primaryVariant = variant !== null && variant === variants[0];
  const plan = useMemo(() => projectWindowPlan(saved.project, projection.data, variant, validity.checking),
    [saved.project, projection.data, variant, validity.checking]);
  const shownPlan = useMemo(() => planWithRefusal(planWithRefusal(plan, readRefusal), refusal), [plan, readRefusal, refusal]);
  const activityResultState = useMemo<ActivityResultState>(() => {
    const states = new Map(plan.steps.flatMap(({ runs }) => runs.flatMap(({ state, results }) =>
      results.map(({ jobId }) => [jobId, state] as const))));
    return (job) => states.get(job.job_id);
  }, [plan]);
  const runMenu: RunMenu = (runs, anchor) => {
    const owner = runs[0] && plan.steps.find(({ id }) => id === (JSON.parse(runs[0].id) as string[])[1]);
    return [...validity.commands(runs, anchor), ...owner ? [sets.duplicateRun(owner, runs)] : []];
  };
  const runRemoval = useRunRemovalDialog(data);
  const sets = setAuthoring({
    editor, project: saved.project, enabled: !!data && !pending && !outside && connected && primaryVariant && !busy,
    applies: !outside && connected && primaryVariant,
    perform, state: () => latestState.current, select: (next) => selection.select(next),
  });
  const axisMenu: AxisMenu = (stepId, axis) => ({
    picker: sets.axisPicker(stepId, axis), commands: sets.axisCommands(stepId, axis),
    refusal: refusal?.kind === 'axis' && refusal.step === stepId && refusal.axis === axis ? refusal.reason : undefined,
  });
  const step = plan.steps.find(({ id }) => id === selection.stepId);
  const state = data ? settledWindowSelection(plan, selection.pipeline) : selection.pipeline;
  const latestState = useRef(state);
  latestState.current = state;
  const singleStep = state.selection?.kind === 'steps' && state.selection.stepIds.length === 1 && state.selection.stepIds[0] === step?.id;
  const wholeStep = singleStep && primaryVariant;
  /** The selected values as an edit names them; the runs an optional axis keeps are named by row. */
  const valueTerms = state.selection?.kind === 'values' && !state.selection.terms.some(({ valueIds }) => valueIds.includes(unvalued))
    ? state.selection.terms : undefined;
  /** The one value of a named axis or set entry selected on the step, which a Model edit is stored on. */
  const selectedValue = primaryVariant && valueTerms?.length === 1 && valueTerms[0].valueIds.length === 1 &&
    state.selection?.kind === 'values' && state.selection.stepId === step?.id
    ? { axis: valueTerms[0].sweepId, value: valueTerms[0].valueIds[0] } : undefined;
  const canAuthorSweep = (id: string) => primaryVariant && directTrainStep(saved.project, id, true) !== null;
  const authored = step && saved.project.steps?.[step.id];
  const subjectCheckpoint = authored?.kind === 'evaluate' ? exactCheckpoint(authored) : undefined;
  const authoring = step && (primaryVariant && directTrainStep(saved.project, step.id, true) !== null ||
    wholeStep && authored?.kind === 'train') &&
    state.selection ? {
    editor, stepId: step.id, step, selection: state.selection, picking: elements,
    onRefused: (reason: string | undefined) => setReadRefusal((current) => reason ? { kind: 'step', step: step.id, reason }
      : current?.step === step.id ? undefined : current),
    trains: plan.steps.filter(({ kind }) => kind === 'train'),
    onChooseSweepValue: (sweepId: string, valueId: string, extend: boolean) => {
      selection.select(selectSweepValue(latestState.current, step.id, sweepId, valueId, extend));
    },
  } : null;
  const trainLabels = useTrainFieldLabels(editor, step?.kind === 'train' ? step.id : undefined);
  const trainingDocument = step?.kind === 'train' ? trainingDocumentId(step.documentIds) : undefined;
  const trainClasses = useTrainClassFields(editor, step?.kind === 'train' ? step.id : undefined);
  const availableViews = !step ? emptyViews : trainingDocument ? viewsByKind[step.kind]
    : viewsByKind[step.kind].filter((view) => view !== 'training');
  const primaryView = projectWindowSlotView(windowState, 'primary', availableViews);
  const secondaryView = projectWindowSlotView(windowState, 'secondary', availableViews);
  const focusSlot = secondaryView && windowState.focusedSlot === 'secondary' ? 'secondary' : 'primary';
  const focusView = focusSlot === 'secondary' ? secondaryView : primaryView;
  const revealed = problemReveal?.revision === saved.revision && problemReveal.variant === variant &&
    problemReveal.reveal.step === step?.id && problemReveal.runId === (state.focus?.runId ?? null)
    ? problemReveal.reveal : undefined;
  /** A revealed place among the step's own fields, which the inspector draws and reveals instead of a banner. */
  const fieldReveal = revealed?.view === 'runs' && placeField(revealed.place) ? revealed : undefined;
  /** The steps an edit of the selected step reaches outside a document view, drawn lit: it and the steps based on it. */
  const reachedSteps = wholeStep && step?.kind === 'train' ? withBaseDescendants(saved.project, [step.id]) : undefined;
  /**
   * The steps a view of the document the step reads in `field` lights: those that read it or a document based on it,
   * less those taken out. An edit there is made for the lit steps. From an Evaluate step the document's Train readers
   * are drawn as readers and are lit only when put in, and the step is lit even while it follows its subject's.
   */
  function documentScope(field: 'graph' | 'task' | 'objective' | 'training' | 'analysis'): NamedDocumentScope | undefined {
    const ids = step?.documentIds;
    const document = field === 'graph' ? modelDocumentId(ids)
      : field === 'task' ? ids?.['payload.trial_documents.task'] ?? ids?.['payload.trial_documents.python_task']
        : field === 'objective' ? ids?.['payload.trial_documents.objective']
          : field === 'analysis' ? ids?.payload : trainingDocumentId(ids);
    const evaluating = step?.kind === 'evaluate' && (field === 'graph' || field === 'task');
    const analyzing = step?.kind === 'analyze' && field === 'analysis';
    if (!document || !step || !singleStep || !data || step.kind !== 'train' && !evaluating && !analyzing) return undefined;
    const localVariant = Boolean(saved.project.sweep && !primaryVariant && field === 'graph');
    if (!localVariant && !wholeStep) return undefined;
    const family = documentFamily(saved.project, document);
    const users = new Set(plan.steps.filter(({ documentUses }) => documentUses?.some((used) => family.has(used)))
      .map(({ id }) => id));
    if (evaluating) users.add(step.id);
    if (!users.has(step.id)) return undefined;
    const key = JSON.stringify([step.id, document, variant]);
    if (localVariant) return { document, key, users, lit: users, localVariant };
    const toggled = excludedUsers[key] ?? [];
    const reader = (id: string) => evaluating && plan.steps.find((item) => item.id === id)?.kind !== 'evaluate';
    return { document, key, users, lit: new Set([...users].filter((id) =>
      id === step.id || (reader(id) ? toggled.includes(id) : !toggled.includes(id)))) };
  }
  /**
   * With one entry of a set selected, the steps that read the set or one based on it: an edit of the entry is made for
   * the lit ones, and a set based on it holds the edit when one is taken out.
   */
  function setScope(): NamedDocumentScope | undefined {
    const read = selectedValue && step && authored?.sweep?.axes?.[selectedValue.axis];
    if (!read || !('set' in read) || !step || !data) return undefined;
    const family = documentFamily(saved.project, read.set);
    const users = new Set(Object.entries(saved.project.steps ?? {}).flatMap(([id, item]) =>
      Object.values(item.sweep?.axes ?? {}).some((other) => other && 'set' in other && family.has(other.set)) ? [id] : []));
    const key = JSON.stringify([step.id, read.set, variant]);
    const toggled = excludedUsers[key] ?? [];
    return { document: read.set, key, users, lit: new Set([...users].filter((id) => id === step.id || !toggled.includes(id))) };
  }
  const documentScopes = {
    graph: setScope() ?? documentScope('graph'), task: documentScope('task'),
    objective: documentScope('objective'), training: documentScope('training'), analysis: documentScope('analysis'),
  };
  const activeDocumentScope = windowState.mode === 'map' ? undefined
    : focusView === 'model' ? documentScopes.graph
      : focusView === 'task' || focusView === 'objective' || focusView === 'training' ? documentScopes[focusView]
        : focusView === 'analysis' || focusView === 'figures' ? documentScopes.analysis
          : undefined;
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
  const runTarget: PipelineSelection = state.selection ?? { kind: 'steps', stepIds: plan.steps.map(({ id }) => id) };
  const runIds = new Set(plan.steps.flatMap((item) => selectedRuns(item, runTarget).map(({ id }) => id)));
  const { row, attempt, scene: sceneRows, evaluation: selectedEvaluation } =
    focusedResult(data, step?.id, state.focus?.runId, runIds, resultAttempt);
  const shownResult = focusedResult(projection.data, step?.id, state.focus?.runId, runIds, resultAttempt);
  const ready = Boolean(data && !pending && !outside && connected && !projection.error);
  const canAuthorContinuation = ready && primaryVariant;
  /** Whether the plan's commands apply. A gesture made while the workspace is unsettled is held until it settles. */
  const applies = Boolean(projection.data && !outside && connected && !projection.error);
  /** The stopped attempt of a focused single run whose checkpoints an Evaluate can use, in the plan it was read from. */
  const stoppedAttempt = ({ row, attempt }: ReturnType<typeof focusedResult>, source: StudioSavedTrainRuns | undefined) =>
    !saved.project.sweep && windowState.mode !== 'map' &&
    state.selection?.kind === 'rows' && state.selection.runIds.length === 1 &&
    (primaryView === 'results' || secondaryView === 'results') && row && attempt?.status === 'stopped' &&
    source?.window_map?.rows.some(({ key, evaluation_conditions }) =>
      evaluation_conditions != null && savedRunId(key) === savedRunId(row.key))
      ? attempt.job_id : undefined;
  const checkpointAttempt = stoppedAttempt({ row, attempt, scene: sceneRows, evaluation: selectedEvaluation }, data);
  const shownCheckpoint = stoppedAttempt(shownResult, projection.data);
  const stoppedPlan = useFollowingQuery({
    queryKey: ['saved-stopped-checkpoint', saved.project_id, saved.revision, checkpointAttempt],
    queryFn: ({ signal }) => fetchSavedTrainRuns(saved.project_id, saved.revision, signal, checkpointAttempt),
    enabled: ready && checkpointAttempt !== undefined,
    placeholderData: keepPreviousData,
    gcTime: 0,
  }, jobsVersion);
  useEffect(() => {
    if (!ready || !focusFirstModel || step?.id !== focusFirstModel) return;
    requestSlotFocus((count) => count + 1);
    setFocusFirstModel(undefined);
  }, [ready, focusFirstModel, step?.id]);
  const names = useMemo(() => new Map([[saved.project_id, configuration.project]]), [saved.project_id, configuration.project]);

  const runRows = data?.rows.filter(({ key }) => runIds.has(savedRunId(key))) ?? [];
  const evaluationRows = data?.window_map?.evaluations?.filter(({ key }) => runIds.has(savedRunId(key))) ?? [];
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
  const evaluationSources = evaluationAuthoring ? evaluationAuthoring.subjects.flatMap((selection) => {
    const source = plan.steps.find(({ id }) => id === selection.reference.step);
    return source ? [{ ...selection, source }] : [];
  }) : undefined;
  const mapInputs = authoredMap?.steps.find(({ step_id, variant: id }) => step_id === step?.id && id === variant)?.inputs;
  const subject = mapInputs?.find(({ id }) => id === 'subject');
  const subjectSubset = subject?.subset;
  const rankGroups = bestRankGroups(subjectSubset);
  const rankedSubjects = rankGroups.length ? authoredMap?.evaluations?.flatMap(({ key, bound_subject }) =>
    key.variant === variant && key.step === step?.id && bound_subject ? [bound_subject] : []) ?? [] : [];
  const boundByRank = new Map(authoredMap?.evaluations?.flatMap(({ key, bound_subject }) =>
    key.variant === variant && key.step === step?.id && bound_subject ? [[bound_subject.key, bound_subject]] : []));
  /** The attempts To Rows freezes a measured selection to: one per rank, each group's from one selection. */
  function freezeAttempts(inputId: string) {
    const groups = bestRankGroups(mapInputs?.find(({ id }) => id === inputId)?.subset);
    return groups.length && groups.every(({ ranks }) => {
      const snapshots = ranks.map((key) => boundByRank.get(key)?.selection_sha256);
      return snapshots.every(Boolean) && new Set(snapshots).size === 1;
    }) ? groups.flatMap(({ ranks }) => ranks.map((key) => boundByRank.get(key)!.consumer_job_id)) : undefined;
  }
  const subjectIds = subject && rankGroups.length
    ? new Set(rankedSubjects.map((bound) => savedRunId({ ...bound, step: subject.source_step_id })))
    : subjectSubset?.members && new Set(subjectSubset.members.map(savedRunId));
  const evaluationEditor = evaluationAuthoring && evaluationSources && evaluationSources.length === evaluationAuthoring.subjects.length &&
    evaluationSources.every(({ source }) => authoredMap?.rows.some(({ key, evaluation_conditions }) =>
      key.step === source.id && key.variant === variant && evaluation_conditions != null))
    ? {
      authored: evaluationAuthoring, subjects: evaluationSources,
      domain: managedTrialDomain(projection.data, evaluationSources.length === 1
        ? subjectIds ? evaluationSources[0].source.runs.filter(({ id }) => subjectIds.has(id))
          : evaluationSources[0].reference.where == null ? evaluationSources[0].source.runs : []
        : evaluationSources.flatMap(({ source }) => source.runs)),
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
  const definitionRows = projection.data?.window_map?.rows.filter(({ key }) => key.step === step?.id && key.variant === variant) ?? [];
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
  const trialSelection = trialStep && namedRun && state.selection;
  const modelSource = step?.kind === 'evaluate' ? modelDocumentId(plan.steps.find(({ id }) =>
    id === step.inputs.find(({ id }) => id === 'subject')?.sourceStepId)?.documentIds) : undefined;
  const modelProblems = data?.window_map?.problems.filter((problem) => placeMember(problem.place) &&
    (problem.step === step?.id || step?.kind === 'evaluate' && placeMember(problem.place)?.document === modelSource) &&
    (problem.variant === null || problem.variant === variant) &&
    (!problem.row || !namedRun || problem.step !== step?.id || problem.row === namedRun.row));
  const axis = trialSelection && valueTerms?.length === 1 ? valueTerms[0] : undefined;
  const trialScope = trialSelection && (state.selection?.kind === 'steps'
    ? singleStep ? (primaryVariant ? { kind: 'step' } as const : { kind: 'variant', variant: namedRun.variant } as const) : undefined
    : axis?.valueIds.length === 1 ? { kind: 'axis_value', axis: axis.sweepId, value: axis.valueIds[0] } as const
      : selectedDefinitionRows.length ? { kind: 'rows', rows: selectedDefinitionRows.map(({ key }) => key.row) } as const
        : undefined);
  const evaluateScope = step?.kind === 'evaluate' ? trialScope : undefined;
  const wholeTrialStep = trialScope && (trialScope.kind === 'step' || trialScope.kind === 'variant');
  // A whole-step edit is a step scope in every variant, the run naming the variant. A Train step's is unscoped, the
  // server writing its shared documents for every step of the variant that uses them, or made for the chosen steps.
  const trialEditScope = !trialScope || trialScope.kind === 'step' || trialScope.kind === 'variant'
    ? trialScope && step?.kind !== 'train' ? { kind: 'step' } as const : undefined
    : trialScope;
  const documentEditable = !pending && !outside && !snapshot.saving;
  const trialEditable = documentEditable && Boolean(trialScope);
  /** A whole-step edit of a Train step's Task or Objective is made for the steps its view chooses. */
  const trialChangeScope: TrialChangeScope = (change) => {
    if (step?.kind === 'evaluate' && wholeTrialStep && change.kind === 'task' && documentScopes.task) {
      return { kind: 'step', chosen_steps: [...documentScopes.task.lit] };
    }
    if (trialEditScope || step?.kind !== 'train' || !wholeTrialStep) return trialEditScope;
    const chosen = change.kind === 'task' ? documentScopes.task?.lit
      : change.kind === 'objective' || change.kind === 'expression' ? documentScopes.objective?.lit : undefined;
    return chosen && { kind: 'step', chosen_steps: [...chosen] };
  };
  const trial = useTrialAuthoring(editor, saved.project_id, saved.revision, namedRun, trialStep && trialVisible,
    trialEditable, trialChangeScope, runRemoval.confirm);
  const documentBase = authored && 'document' in authored.base && !authored.base.pointer ? authored.base.document : undefined;
  const pickDocument = useProjectDocuments({
    editor, stepId: step?.id, run: namedRun, scope: trialEditScope,
    editable: documentEditable,
    detachRefused: data?.window_map?.detach_refused,
    selectable: new Set<DocumentRole>([
      ...!trialEditable ? [] : step?.kind === 'train' ? ['model', 'task', 'objective', 'training'] as const : ['model', 'task'] as const,
      ...documentEditable && wholeStep ? ['analysis', 'report'] as const : [],
    ]),
    onError: setCommandError,
  });
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
    namedRun, documentScopes, planCurrent: !!data, elements, valueEditing: !!selectedValue, evaluateScope: evaluateScope || undefined, modelSource, modelProblems, selectedSweepValues, confirmRunRemoval: runRemoval.confirm,
    drawerSections,
    openModelProblems: (element: ModelAddress, opener: HTMLElement) => {
      const owner = modelProblems?.find((problem) => {
        const member = placeMember(problem.place);
        return member && holds(element, member.address);
      });
      if (owner?.step) setPanel({ kind: 'problems', scope: { step: owner.step, variant, element }, opener });
    },
    runProblems: namedRun && step?.runs.find(({ id }) => id === savedRunId(namedRun))?.problems,
    revealProblem,
    fieldRows: state.selection && state.selection.kind !== 'steps' ? selectedDefinitionRows.map(({ key }) => key.row) : undefined,
    fieldAxes: valueTerms && Object.fromEntries(valueTerms.map(({ sweepId, valueIds }) => [sweepId, [...valueIds]])),
    exceptions: exceptionMention,
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
  const focusedDetail = windowState.mode !== 'map' && (focusView === 'model' || focusView === 'scene')
    ? (focusSlot === 'primary' ? primary : secondary)[focusView] : undefined;
  const analyze = useManagedAnalyze({
    editor, key: step?.kind === 'analyze' ? namedRun : undefined, projection: data, wholeStep,
    chosen: documentScopes.analysis?.lit,
    focusedView: windowState.mode === 'map' ? null : focusView, reveal: revealed, savePosition: saveElementPosition, selection: elements,
    sections: drawerSections, steps: plan.steps, failures: stepFailures(step?.kind === 'analyze' ? step.id : undefined),
  });
  const report = useManagedReport({
    editor, snapshot, plan, projection: data, variant, stepId: step?.kind === 'report' ? step.id : undefined,
    run: step?.kind === 'report' ? namedRun : undefined,
    wholeStep, focused: windowState.mode !== 'map' && focusView === 'document',
    failures: stepFailures(step?.kind === 'report' ? step.id : undefined),
  });
  const workspaceSettled = !busy && !(pending && !snapshot.error) && !snapshot.saving && !planPending &&
    !projection.isPending && analyze.settled;
  useHeldGestures(workspaceSettled, !!snapshot.error);
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

  /** Runs a gesture; a refusal lands on the object `refuse` names, and only a service failure reaches the bottom bar. */
  async function perform(work: () => Promise<void>, refuse?: (reason: string) => Refusal | undefined) {
    if (inFlight.current) return;
    inFlight.current = true;
    setBusy(true);
    settled();
    try {
      await work();
    } catch (cause) {
      failed(cause, refuse);
    } finally {
      inFlight.current = false;
      setBusy(false);
    }
  }
  function settled() {
    setCommandError(undefined);
    setRefusal(undefined);
  }
  function failed(cause: unknown, refuse?: (reason: string) => Refusal | undefined) {
    const refused = landsOnGesture(cause) && refuse?.(apiErrorMessage(cause));
    if (refused) setRefusal(refused);
    else commandFailed(cause);
  }
  function refuseStep(stepId: string | undefined) {
    return (reason: string): Refusal | undefined => stepId ? { kind: 'step', step: stepId, reason } : undefined;
  }
  function stepFailures(stepId: string | undefined): GestureFailures {
    return {
      clear: settled,
      fail: (cause, land) => {
        if (land && landsOnGesture(cause)) land(apiErrorMessage(cause));
        else failed(cause, refuseStep(stepId));
      },
    };
  }

  const refuseRuns = (runIds: readonly string[]) => (reason: string): Refusal | undefined =>
    runIds.length ? { kind: 'runs', reasons: Object.fromEntries(runIds.map((id) => [id, reason])) } : undefined;
  const refuseField = (field: EvaluateField) => {
    const stepId = step?.id;
    return (reason: string): Refusal | undefined => stepId ? { kind: 'field', step: stepId, field, reason } : undefined;
  };

  function applySubmission(result: StudioJobSubmission) {
    const rejected = result.items.flatMap(({ disposition, run_key, reason }) =>
      disposition === 'rejected' && reason ? [[run_key, reason] as const] : []);
    if (rejected.length) setRefusal({ kind: 'runs', reasons: Object.fromEntries(rejected) });
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
    }, refuseRuns(keys.map(savedRunId)));
  }

  function approveMachine() {
    if (!machineQuote) return;
    void perform(async () => {
      await approveSavedRun(saved.project_id, { quote: machineQuote.id });
      setMachineQuote(undefined);
      actions.refresh();
    }, (reason) => ({ kind: 'quote', reason }));
  }

  function stop(jobIds: string[], mode: 'checkpoint' | 'force') {
    const runOf = (jobId: string) => jobsById.get(jobId)?.run_key;
    void perform(async () => {
      const result = await actions.stop(jobIds, mode);
      const refused = result.flatMap(({ job_id, error }) => {
        const run = error && runOf(job_id);
        return run ? [[run, error.message] as const] : [];
      });
      if (refused.length) setRefusal({ kind: 'runs', reasons: Object.fromEntries(refused) });
    }, refuseRuns(jobIds.flatMap((id) => runOf(id) ?? [])));
  }

  const problemPanel = panel?.kind === 'problems' ? panel : null;
  const problemScope = problemPanel?.scope;
  const problems = data?.window_map?.problems ?? [];
  const problemSteps = new Set(problems.flatMap((problem) =>
    problem.step && (problem.variant === null || problem.variant === variant) ? [problem.step] : []));
  const shownProblems = problemScope ? problems.filter((problem) =>
    problem.step === problemScope.step &&
    (problem.variant === null || problem.variant === problemScope.variant) &&
    (!problemScope.element || holdsMember(problemScope.element, problem))) : problems;
  function openProjectProblems(opener: HTMLElement | null = null) {
    setPanel({ kind: 'problems', scope: null, opener });
  }
  function holdsMember(element: ModelAddress, problem: StudioProjectWindowMapProblem) {
    const member = placeMember(problem.place);
    return !!member && holds(element, member.address);
  }
  function addressedStep(at: { step: string | null; variant: string | null }) {
    return data?.window_map?.steps.find(({ step_id, variant: scope }) =>
      step_id === at.step && scope === (at.variant ?? variant));
  }
  /** Show the place a reason names in its step's view, at the run the reason belongs to. */
  function reveal(at: { step: string | null; variant: string | null; row: string | null }, place: StudioPlace | null,
    reason: string, problem?: StudioProjectWindowMapProblem) {
    const destination = addressedStep(at);
    if (!destination) return;
    const { step_id: stepId, variant: scope, kind } = destination;
    const row = data?.window_map?.rows.find(({ key }) =>
      key.step === stepId && key.row === at.row && key.variant === scope);
    const named = placeView(place);
    const view = kind === 'analyze' ? 'analysis' : kind === 'report' ? 'document'
      : viewsByKind[kind].some((item) => item === named) ? named : 'runs';
    const runId = row ? savedRunId(row.key) : null;
    const field = view === 'runs' ? placeField(place) : null;
    selection.select(runId
      ? selectRun(state, { stepId, runId })
      : selectStep(state, stepId), scope);
    setExcludedUsers({});
    setProblemReveal({ revision: saved.revision, variant: scope, runId, reveal: { step: stepId, place, view, reason, problem } });
    setWindowState((current) => showProjectWindowView(field ? { ...current, inspectorOpen: true } : current, view, focusSlot));
    if (field) setFieldFocus({ key: crypto.randomUUID(), step: stepId, target: field.target });
    else requestSlotFocus((count) => count + 1);
    setPanel(null);
  }
  function revealProblem(problem: StudioProjectWindowMapProblem) {
    reveal(problem, problem.place, studioProblemDetail(problem), problem);
  }
  function revealRunMention(run: PlanRunView, mention: RunMention) {
    if (mention.problem) return revealProblem(mention.problem);
    const [variantId, stepId, rowId] = JSON.parse(run.id) as [string, string, string];
    reveal({ variant: variantId, step: stepId, row: rowId }, mention.place, mention.detail);
  }
  const graphRepair = revealed?.problem && removeGraphVariation(saved.project, revealed.problem);
  function removeRefusedVariation() {
    if (!graphRepair || !revealed?.problem || !ready || snapshot.saving || busy) return;
    const { step: stepId, variant: variantId } = revealed.problem;
    void perform(async () => {
      if (!await runRemoval.confirm((row) => row.key.step === stepId && (!variantId || row.key.variant === variantId))) return;
      editor.editFiles(studioName('removeVariation'), [graphRepair], { reviewedRevision: saved.revision });
      await editor.flush();
    }, (reason) => ({ kind: 'problem', problem: problemKey(revealed.problem!), reason }));
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
    id, { sweepOrder: value.sweep_order },
  ])), [present]);
  async function changePresentation(stepId: string, next: PipelineCardPresentation) {
    const current = editor.getSnapshot().saved.presentation ?? {};
    const updated: PresentationDocument = {
      ...current, steps: { ...current.steps, [stepId]: {
        ...current.steps?.[stepId], sweep_order: [...(next.sweepOrder ?? [])],
      } },
    };
    settled();
    try {
      await session.savePresentation(updated);
    } catch (cause) {
      failed(cause, refuseStep(stepId));
    }
  }

  const projectFailure = snapshot.error ?? stoppedPlan.error;
  const failure = commandError ?? (projectFailure && !isUnanswered(projectFailure) ? apiErrorMessage(projectFailure) : null);
  const retryPlan = () => {
    void projection.refetch();
  };
  const planError = (name: 'runs' | 'pipelineMap' | 'results'): StudioRegionState | undefined => projection.error
    ? { kind: 'failed', issue: studioRegionIssue(name, projection.error), retry: retryPlan }
    : undefined;
  const planUnavailable: StudioRegionState | undefined = planError('runs') ??
    (!projection.data ? { kind: 'loading' } : undefined);
  const mapUnavailable = planError('pipelineMap');
  const renderMap = (body: ReactNode) => mapUnavailable
    ? <StudioRegion state={mapUnavailable} />
    : body;
  function retry() {
    setCommandError(undefined);
    if (snapshot.error && snapshot.pending) void editor.flush().catch((error) => commandFailed(error));
    actions.refresh();
    void projection.refetch();
    if (checkpointAttempt) void stoppedPlan.refetch();
  }
  async function moveHistory(action: 'undo' | 'redo') {
    try {
      const before = Object.keys(editor.getSnapshot().saved.project.steps ?? {});
      const changed = await editor[action]();
      // One step name leaving as another arrives is a step rename moving through history.
      const after = Object.keys(editor.getSnapshot().saved.project.steps ?? {});
      const left = before.filter((id) => !after.includes(id));
      const arrived = after.filter((id) => !before.includes(id));
      if (left.length === 1 && arrived.length === 1) {
        selection.renameStep(left[0], arrived[0]);
        setStepFocus({ stepId: arrived[0], surface: windowState.mode === 'details' ? 'line' : 'card', force: true });
      }
      if (changed?.addresses.length) {
        const address = changed.addresses[0];
        elements.setState({
          reveal: { document: changed.document, address },
          selection: changed.addresses.map((item) => ({ document: changed.document, address: item })),
        });
        if (focusView !== 'model') setWindowState((current) => showProjectWindowView(current, 'model', focusSlot));
      }
      await editor.flush();
    } catch (error) {
      commandFailed(error);
    }
  }

  const commandScope: ProjectWindowCommandScope = {
    editor, selection: state, variantId: variant, onSelectionChange: selection.select,
    plan, projection: data,
    selectCreatedStep: (id, view = 'runs') => {
      selection.select(selectStep(state, id));
      if (view === 'model') setFocusFirstModel(id);
      if (view === 'runs') setFieldFocus({ key: crypto.randomUUID(), step: id });
      if (view === 'analysis' || view === 'document') requestSlotFocus((count) => count + 1);
      setWindowState((current) => showProjectWindowView(
        view === 'runs' ? { ...current, inspectorOpen: true } : current, view,
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
    }, refuseStep(source.stepId));
  }

  function openStepMenu(kinds: StepInsertion) {
    const revision = saved.revision;
    insertMenu.open(Object.fromEntries(Object.entries(kinds).map(([kind, create]) =>
      [kind, () => { if (latestRevision.current === revision) create(); }])));
  }

  const shownRow = shownResult.row;
  const stoppedCoordinate = shownRow && stoppedPlan.data?.rows.find(({ key }) => savedRunId(key) === savedRunId(shownRow.key))
    ?.attempts.find(({ job_id }) => job_id === shownCheckpoint)?.stopped_checkpoint;
  const stoppedSubject = shownRow && shownCheckpoint && stoppedCoordinate
    ? { runId: savedRunId(shownRow.key), jobId: shownCheckpoint, at: stoppedCoordinate } : undefined;
  // Shown from the plan on screen; it authors from the current plan, so it waits for that plan to be the shown one.
  const stoppedNewStep = stoppedSubject && bindStudioCommand('newStep', () => {
    if (!data || !row || checkpointAttempt !== stoppedSubject.jobId) return;
    openStepMenu({
      evaluate: () => evaluateTrain(
        { stepId: row.key.step, selection: { kind: 'rows', stepId: row.key.step, runIds: [stoppedSubject.runId] } },
        data, stoppedSubject),
    });
  }, { enabled: Boolean(applies && primaryVariant && checkpointAttempt === stoppedSubject.jobId) });

  const evaluateInputs = authored?.kind === 'evaluate' ? authored : undefined;
  const trainSteps = plan.steps.filter(({ kind }) => kind === 'train');
  /** A selection of the subject, whose detail is drawn on its row of the subject field. */
  const subjectSelection = (id: string) => !!evaluateInputs && (id === 'subject' || id.startsWith('subject.'));
  /** The steps this step reads, each a link, and the inputs no field of the step draws. */
  const listedInputs = step?.inputs.filter(({ id, sourceStepId }) =>
    sourceStepId || !evaluateInputs || !['subject', 'task', 'objective'].includes(id)) ?? [];
  function inputDetail({ id, subset }: PlanStepView['inputs'][number]) {
    return <>
      {id === 'subject' && subjectCheckpoint !== undefined &&
        <div className="px-control-inline text-shell-muted">{checkpointLabel(subjectCheckpoint)}</div>}
      {subset && <div className="px-control-inline text-shell-muted">
        {[...subset.terms, studioCountOf(subset.count, subset.total)].join(' · ')}
        {subset.memberGroups?.map(({ label, members }) => <div key={members[0]?.id} className="mt-1">
          {label && <div className="font-medium">{label}</div>}
          <ol aria-label={label ? studioAccessibleName('members', label) : studioName('members')}
            className="list-decimal space-y-0.5 ps-5">
            {members.map(({ id, label }) => <li key={id}>{label}</li>)}
          </ol>
        </div>)}
      </div>}
    </>;
  }

  const refusedAt = (field: EvaluateField) =>
    refusal?.kind === 'field' && refusal.step === step?.id && refusal.field === field ? refusal.reason : undefined;

  function editEvaluateSubject(name: string, patches: ProjectValueEdit[], reads: readonly string[]) {
    if (!step || !ready) return;
    const files = [{ address: { kind: 'step' as const, id: step.id }, patches }];
    const dependencies = reads.map((id) => ({ kind: saved.project.steps?.[id] ? 'step' as const : 'document' as const, id }));
    void perform(async () => {
      if (!await runRemoval.confirm(filesRemoval(saved.project_id, saved.revision, files, editor.signal))) return;
      editor.editFiles(name, files, { reviewedRevision: saved.revision, dependencies });
      await editor.flush();
    }, refuseField('subject'));
  }

  function editEvaluation(field: Exclude<EvaluateField, 'subject' | 'checkpoints'>, name: string, patch: ProjectValueEdit) {
    if (!step || !evaluationEditor || !ready) return;
    const dependencies = [...new Set(evaluationEditor.subjects.map(({ source }) => source.id))]
      .map((id) => ({ kind: 'step' as const, id }));
    const files = [{ address: { kind: 'step' as const, id: step.id }, patches: [patch] }];
    void perform(async () => {
      if (!await runRemoval.confirm(filesRemoval(saved.project_id, saved.revision, files, editor.signal))) return;
      editor.editFiles(name, files, { reviewedRevision: saved.revision, dependencies });
      await editor.flush();
    }, refuseField(field));
  }

  /**
   * The runs an exception applies to, by axis and values (`seed 2`). Rows are read through their coordinates; an axis
   * whose every value is included says nothing and is left out.
   */
  function exceptionTerms(target: PlanStepView, where: SweepExceptionTerms | null | undefined) {
    const chosen = new Map<string, Set<string>>();
    const add = (axis: string, value: string) => chosen.set(axis, (chosen.get(axis) ?? new Set()).add(value));
    for (const [axis, values] of Object.entries(where?.axes ?? {})) values.forEach((value) => add(axis, value));
    for (const row of where?.rows ?? []) {
      const coordinates = data?.window_map?.rows.find(({ key }) => key.step === target.id && key.row === row)?.coordinates;
      for (const [axis, { value_id }] of Object.entries(coordinates ?? {})) add(axis, value_id);
    }
    return [...chosen].filter(([axis, values]) => {
      const all = target.sweeps.find(({ id }) => id === axis)?.values;
      return !all || all.some(({ id }) => !values.has(id));
    }).map(([axis, values]) => `${axis} ${[...values].map((value) => axisValueText(axis, value)).join(', ')}`);
  }

  function stepExceptions(stepId: string, scope: string | null = variant) {
    return authoredMap?.steps.find(({ step_id, variant: id }) => step_id === stepId && id === scope)?.exceptions ?? [];
  }

  /** The runs an exception picks, as its row in Exceptions and a field's source read them (`seed 2 · 2 runs`). */
  function exceptionRuns(target: PlanStepView, { count, rule }: StudioProjectWindowMapException) {
    return [
      ...exceptionTerms(target, rule.where),
      count === 1 ? studioFormat('oneRun') : studioFormat('runCount', { count }),
    ].join(' · ');
  }

  function exceptionMention(source: StudioPlanReviewSource) {
    const target = plan.steps.find(({ id }) => id === source.step);
    const scope = source.variant ?? variant;
    const exception = target && stepExceptions(target.id, scope).find(({ id }) => id === source.name);
    if (!target || !exception) return undefined;
    return { runs: exceptionRuns(target, exception), reveal: () => {
      selection.select(selectStep(state, target.id), scope);
      setExceptionReveal({ revision: saved.revision, step: target.id, id: exception.id });
      setWindowState((current) => showProjectWindowView({ ...current, inspectorOpen: true }, 'runs', focusSlot));
    } };
  }

  /** A sweep value as the value it sets, never its id. */
  function axisValueText(axis: string, valueId: string) {
    const coordinate = data?.window_map?.rows.map(({ coordinates }) => coordinates[axis])
      .find((item) => item?.value_id === valueId);
    return coordinate ? valueText(coordinate.value) : valueId;
  }

  function removeSweepException(id: string) {
    if (!step || !ready || busy || saved.project.sweep || !authored?.sweep?.exceptions?.[id]) return;
    void perform(async () => {
      editor.editFiles(studioName('removeException'), [{
        address: { kind: 'step', id: step.id },
        patches: [{ op: 'remove', path: authoredPointer(['sweep', 'exceptions', id]) }],
      }], { reviewedRevision: saved.revision });
      await editor.flush();
    }, (reason) => ({ kind: 'exception', step: step.id, exception: id, reason }));
  }

  function freezeEvaluationSubset(inputId: string, source: string, attempts: string[], index: number) {
    if (!step || !variant || !ready) return;
    void perform(async () => {
      const reviewed = await prepareSubsetToRows(saved.project_id, {
        revision: saved.revision, variant, step: step.id, input: inputId, attempts,
      }, editor.signal);
      if (!await runRemoval.confirm(filesRemoval(saved.project_id, reviewed.revision, reviewed.files, editor.signal))) return;
      editor.editFiles(studioName('toRows'), reviewed.files, {
        reviewedRevision: reviewed.revision, dependencies: [{ kind: 'step', id: source }],
      });
      await editor.flush();
    }, refuseField(`subset.${index}`));
  }

  function editEvaluationCheckpoint(rule: EditableCheckpointRule) {
    if (!step || !subject || !evaluationCheckpointRule || !ready) return;
    if (checkpointScope) {
      void applyCheckpointSweepEdit({ kind: 'set', scope: { ...checkpointScope,
        ...('rows' in checkpointScope && { rows: [...checkpointScope.rows ?? []] }),
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
    const files = [{ address, patches: [patch] }];
    void perform(async () => {
      if (!await runRemoval.confirm(filesRemoval(saved.project_id, saved.revision, files, editor.signal))) return;
      editor.editFiles(name, files, { reviewedRevision: saved.revision, dependencies });
      await editor.flush();
    }, refuseField('checkpoints'));
  }

  async function applyCheckpointSweepEdit(edit: StudioCheckpointRuleRequest['edit']) {
    if (!step || !subject || !namedRun || !ready) return;
    await perform(async () => {
      const reviewed = await prepareCheckpointRuleEdit(saved.project_id, {
        revision: saved.revision, run: namedRun, edit,
      }, editor.signal);
      if (!await runRemoval.confirm(filesRemoval(saved.project_id, reviewed.revision, reviewed.files, editor.signal))) return;
      editor.editFiles(edit.kind === 'to_sweep' ? 'Sweep Evaluate checkpoints' : 'Set Evaluate checkpoints',
        reviewed.files, { reviewedRevision: reviewed.revision,
          dependencies: [{ kind: 'step', id: subject.source_step_id }] });
      await editor.flush();
    }, refuseField('checkpoints'));
  }

  function chooseReportFigure(revision: string, figure: ReportFigureSource) {
    setReportChoices(undefined);
    if (!primaryVariant || !ready || revision !== saved.revision) return;
    void perform(async () => {
      const id = await createReport(editor, revision, figure, humanizeLabel(figure.name));
      commandScope.selectCreatedStep(id, 'document');
    }, refuseStep(figure.key.step));
  }

  function stepInsertion(source?: PipelineMapStepSource): { kinds: StepInsertion; enabled: boolean } | undefined {
    if (!primaryVariant) return undefined;
    if (!source) {
      const chosen = state.selection?.kind === 'steps'
        ? state.selection.stepIds.length === 1 ? { stepId: state.selection.stepIds[0], selection: null } : undefined
        : state.selection && { stepId: state.selection.stepId, selection: state.selection };
      const fromChosen = chosen && stepInsertion(chosen);
      if (fromChosen) return fromChosen;
      const revision = saved.revision;
      return { kinds: { train: () => void perform(async () => {
        const id = await createTrain(editor, revision);
        commandScope.selectCreatedStep(id, 'model');
      }) }, enabled: applies };
    }
    const sourceStep = plan.steps.find(({ id }) => id === source.stepId);
    if (sourceStep?.kind === 'evaluate' && data) {
      const rows = source.selection ? selectedRuns(sourceStep, source.selection) : sourceStep.runs;
      const input = analyzeInput(sourceStep, rows, data, !!saved.project.sweep);
      if (!input) return undefined;
      return { kinds: { analyze: () => void perform(async () => {
        const id = await createAnalyze(editor, data.revision, input);
        commandScope.selectCreatedStep(id, 'analysis');
      }, refuseStep(source.stepId)) }, enabled: applies };
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
      } }, enabled: applies };
    }
    if (!onNewStep || sourceStep?.kind !== 'train') return undefined;
    const runs = source.selection ? selectedRuns(sourceStep, source.selection) : sourceStep.runs;
    return {
      kinds: {
        train: () => void perform(async () => { await onNewStep(source, commandScope); }, refuseStep(source.stepId)),
        ...(data && managedTrialDomain(data, runs) && { evaluate: () => evaluateTrain(source, data) }),
      },
      enabled: applies && primaryVariant,
    };
  }

  function newStepCommand(source?: PipelineMapStepSource, subject?: string) {
    const insertion = stepInsertion(source);
    return insertion && bindStudioCommand('newStep', () => openStepMenu(insertion.kinds),
      { enabled: insertion.enabled, subject, instance: JSON.stringify(source ?? null) });
  }

  function newSweepCommand(stepId: string, subject?: string) {
    if (!canAuthorSweep(stepId)) return undefined;
    return bindStudioCommand('newSweep', () => setSweepStep(stepId), { enabled: applies, subject, instance: stepId });
  }
  const sweepMenu = sweepStep ? {
    stepId: sweepStep,
    close: () => setSweepStep(undefined),
    items: <>
      <SweepFieldItems editor={editor} stepId={sweepStep} onSwept={() => {
        const current = latestState.current;
        if (current.selection?.kind !== 'values' || current.selection.stepId !== sweepStep) {
          selection.select(selectStep(current, sweepStep));
        }
        setSweepStep(undefined);
      }} />
      <OfferedSets editor={editor} step={sweepStep} choose={(set) => sets.applySet(sweepStep, set, () => {
        selection.select(selectStep(latestState.current, sweepStep));
      })} />
    </>,
  } : undefined;

  const mapSurface = windowState.mode === 'details' ? 'line' : 'card';
  const renameStep: MapStepRename | undefined = documentEditable ? {
    editing: renamingStep,
    setEditing: (next) => {
      if (!next && renamingStep) setStepFocus(renamingStep);
      setRenamingStep(next);
    },
    commit: (from, name) => {
      const label = name === from ? null : name;
      if (label === (saved.project.steps?.[from]?.label ?? null)) return true;
      const surface = renamingStep?.surface ?? mapSurface;
      void perform(async () => {
        await editor.memberCommand(studioAccessibleName('renameStep', from),
          { kind: 'rename', family: 'steps', member: from, label }, editor.getSnapshot().saved);
        setStepFocus({ stepId: from, surface });
      }, refuseStep(from));
      return true;
    },
    focus: stepFocus,
    focused: () => setStepFocus(null),
  } : undefined;
  const changeStepKey = documentEditable ? (from: string, key: string) => {
    if (!isMemberName(key)) return false;
    void perform(async () => {
      await editor.memberCommand(studioAccessibleName('changeKey', from),
        { kind: 'change_key', family: 'steps', member: from, key }, editor.getSnapshot().saved);
      selection.renameStep(from, key);
    }, (reason) => ({ kind: 'key', step: from, reason }));
    return true;
  } : undefined;
  const duplicateStep = documentEditable ? (from: string) => {
    void perform(async () => {
      const { member } = await editor.memberCommand(studioAccessibleName('duplicateStep', from),
        { kind: 'duplicate', family: 'steps', member: from }, editor.getSnapshot().saved);
      commandScope.selectCreatedStep(member);
    }, refuseStep(from));
  } : undefined;
  /**
   * Merge on a selection of two or more Analyze cards, offered on each of them: the others join the most upstream
   * card, the inverse of Split.
   */
  function mergeCommand(target: PlanStepView) {
    const chosen = state.selection?.kind === 'steps' ? state.selection.stepIds : [];
    const cards = plan.steps.filter(({ id }) => chosen.includes(id));
    const [member, ...others] = cards.map(({ id }) => id);
    const refused = new Set(data?.window_map?.regroup_refused);
    return target.kind !== 'analyze' || !chosen.includes(target.id) || cards.length < 2 ? [] : [
      bindStudioCommand('merge', () => {
        void perform(async () => {
          await editor.memberCommand(studioAccessibleName('merge', cards[0].label),
            { kind: 'merge', member, others }, editor.getSnapshot().saved);
          selection.select(selectStep(latestState.current, member));
        }, refuseStep(member));
      }, {
        subject: cards.map(({ label }) => label).join(', '),
        enabled: !outside && cards.every(({ kind, id }) => kind === 'analyze' && !refused.has(id)),
      }),
    ];
  }
  /** Going into a step puts the keyboard in the slot that shows its views; a window showing only the map opens Runs. */
  const enterStep = (stepId: string) => {
    selection.select(selectStep(state, stepId));
    if (windowState.mode === 'map') setWindowState((current) => showProjectWindowView(current, 'runs', focusSlot));
    requestSlotFocus((count) => count + 1);
  };
  /** The commands on a step: its context menu, and the palette's for the selected step. */
  const stepCommands: StepMenu = (target, rename = {
    start: () => setRenamingStep({ stepId: target.id, surface: mapSurface }), surface: mapSurface,
  }) => [
    bindStudioCommand('renameStep', rename.start, {
      subject: target.label, instance: JSON.stringify([target.id, rename.surface]), enabled: !outside,
    }),
    bindStudioCommand('changeKey', () => {
      selection.select(selectStep(state, target.id));
      if (windowState.mode === 'map') setWindowState((current) => showProjectWindowView(current, 'runs', focusSlot));
      setKeyingStep(target.id);
    }, { subject: target.label, instance: target.id, enabled: !outside }),
    bindStudioCommand('duplicateStep', () => duplicateStep?.(target.id), { subject: target.label, instance: target.id, enabled: !outside }),
    ...mergeCommand(target),
  ];
  /** A wire dragged on the map from `source` to `target` adds a selection of `source` to an input of `target`. */
  function connectSteps(source: string, target: string) {
    const wire = connection(saved.project.steps, source, target);
    if (!wire || !ready) return;
    const files = [{ address: { kind: 'step' as const, id: target }, patches: wire.edits }];
    void perform(async () => {
      if (!await runRemoval.confirm(filesRemoval(saved.project_id, saved.revision, files, editor.signal))) return;
      editor.editFiles(studioAccessibleName('add', humanizeLabel(wire.role)), files, {
        reviewedRevision: saved.revision, dependencies: wire.reads.map((id) => ({ kind: 'step' as const, id })),
      });
      await editor.flush();
    }, refuseStep(target));
  }
  const mapProps = {
    connect: ready && !busy ? {
      accepts: (source: string, target: string) => !!connection(saved.project.steps, source, target),
      connect: connectSteps,
    } : undefined,
    plan: shownPlan, state, problemSteps, rename: renameStep, onEnterStep: enterStep, stepMenu: stepCommands,
    onSelectionChange: (value: PipelineSelectionState) => {
      setResultAttempt(undefined);
      setProblemReveal(undefined);
      selection.select(value);
    },
    onProblems: (step: string, opener: HTMLButtonElement) => setPanel({ kind: 'problems', scope: { step, variant }, opener }),
    newStepCommand, newSweepCommand, onChooseStep: chooseMapStep, runMenu, axisMenu,
    documentScope: activeDocumentScope && activeDocumentScope.users.size > 1 ? activeDocumentScope : undefined,
    reached: activeDocumentScope ? undefined
      : reachedSteps && new Set([...reachedSteps].filter((id) => id !== step?.id)),
    pending: planPending,
  };
  const activity = bindStudioCommand('activity', () => setPanel((current) => current?.kind === 'activity' ? null : { kind: 'activity' }));
  const problemCommand = bindStudioCommand('problems', () => openProjectProblems());
  const retryCommand = bindStudioCommand('retry', retry);
  const runCommand = bindStudioCommand('run', run, {
    enabled: applies && !machineQuote && Boolean(machineReady) && runnableRows.length > 0,
    name: selectedJobs.length === 1 && selectedJobs[0].target?.kind === 'session' && selectedJobs[0].status === 'failed'
      ? 'rerun' : 'run',
  });
  const nextStep = newStepCommand();
  const runCommands: StudioCommand[] = [runCommand];
  for (const [id, mode] of [['stop', 'checkpoint'], ['forceStop', 'force']] as const) {
    const eligible = selectedJobs.filter((job) => jobAllowsCommand(job, id)).map(({ job_id }) => job_id);
    if (!eligible.length && id === 'forceStop') continue;
    runCommands.push(bindStudioCommand(id, () => stop(eligible, mode), { enabled: eligible.length > 0 }));
  }
  if (step?.kind === 'train') {
    if (resumable && jobAllowsCommand(resumable, 'resume')) {
      runCommands.push(bindStudioCommand('resume', () => {
        void perform(async () => { await actions.resume(resumable.job_id); }, refuseRuns([resumable.run_key]));
      }));
    }
    runCommands.push(bindStudioCommand('openRun', () => { if (exactRow) showRun(exactRow.key); }, { enabled: !!exactJob }));
  }
  function pivotCommand(slot: ProjectWindowSlot) {
    if (!step || step.sweeps.length < 2) return undefined;
    return bindStudioCommand('pivot', () => setPivotSlots((current) => ({
      ...current, [slot]: current[slot] === step.id ? undefined : step.id,
    })), { instance: slot });
  }
  const commands = [
    bindStudioCommand('rename', () => {
      void perform(async () => {
        // A refused pending save already went back to the gesture that made it.
        const revision = await editor.flush().catch((cause: unknown) => { if (isRefusal(cause)) return undefined; throw cause; });
        if (revision === undefined) return;
        setRename({ name: editor.getSnapshot().configuration.project, revision });
      });
    }, {
      enabled: !outside,
    }),
    ...runCommands,
    ...(step ? validity.commands(selectedRuns(step, state.selection)) : []),
    windowState.mode !== 'map' && focusView === 'results' && stoppedNewStep ? stoppedNewStep : nextStep,
    ...(focusedDetail?.commands ?? []),
    // After the focused view's commands, so its own F2 (a Model component's rename) comes first.
    ...(step ? stepCommands(step) : []),
    step && selectedRuns(step, state.selection).length > 0 && sets.duplicateRun(step),
    ...analyze.commands,
    ...report.commands,
    step && newSweepCommand(step.id),
    windowState.mode !== 'map' && focusView === 'runs' && pivotCommand(focusSlot),
    bindStudioCommand('undo', () => { void moveHistory('undo'); }, {
      enabled: !!snapshot.undoName && !outside,
      subject: snapshot.undoName === studioName('outsideChange') ? snapshot.undoName : undefined,
    }),
    bindStudioCommand('redo', () => { void moveHistory('redo'); }, {
      enabled: !!snapshot.redoName && !outside,
      subject: snapshot.redoName === studioName('outsideChange') ? snapshot.redoName : undefined,
    }),
    bindStudioCommand('clearFocus', () => selection.select({ ...state, focus: null }), { enabled: !!state.focus }),
    activity, problemCommand, failure && retryCommand,
  ].filter((command): command is StudioCommand => !!command);

  return (
    <RunRemovalContext.Provider value={runRemoval.confirm}>
      <ProjectWindow
        state={windowState} onStateChange={setWindowState} projectName={configuration.project}
        saveStatus={pending || snapshot.error ? studioName('edited') : undefined}
        availableViews={availableViews}
        commands={commands} mapCommands={canvasViewportCommands(() => mapViewport.current, !!data && plan.steps.length > 0)}
        focusRequest={focusRequest}
        onBack={selection.back} onForward={selection.forward}
        map={renderMap(<StudioPipelineMap {...mapProps}
          activeStepId={selection.stepId} viewportRef={mapViewport} fitReady={!!data} sweepMenu={sweepMenu} pick={mapPick}
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
              }, (reason) => ({ kind: 'projectName', reason }));
            }}>
              <label className="grid gap-1">
                {studioName('name')}
                <TextField data-dialog-autofocus value={rename.name} disabled={busy} required
                  onValueChange={(name) => setRename({ ...rename, name })} />
              </label>
              {rename.revision !== saved.revision || outside
                ? <div role="alert">{studioReason('nameChangedDuringEdit', studioName('project'))}</div>
                : refusal?.kind === 'projectName' && <div role="alert">{refusal.reason}</div>}
              <Button type="submit" className="justify-self-end" disabled={busy || !!outside || rename.revision !== saved.revision}>
                {studioName('rename')}
              </Button>
            </form>
          </Dialog>}
          {focusedDetail?.overlay}
          {analyze.overlay}
          {report.overlay}
          {machineQuote && <StudioMachineApproval quote={machineQuote} busy={busy}
            error={refusal?.kind === 'quote' ? refusal.reason : undefined}
            onApprove={approveMachine} onCancel={() => {
              setMachineQuote(undefined);
              setRefusal(undefined);
            }} />}
          {primaryVariant && reportChoices?.revision === saved.revision && <Dialog ariaLabel={studioName('report')} manageFocus
            onClose={() => setReportChoices(undefined)}
            panelClassName="grid w-[min(24rem,90vw)] gap-px p-1">
            {reportChoices.figures.map((figure) => <Button variant="ghost" align="start"
              key={JSON.stringify([figure.key, figure.name])}
              onClick={() => chooseReportFigure(reportChoices.revision, figure)}>
              {stepElementName(saved.project, saved.composed, figure.key.step, figure.name)}{figure.key.row !== 'default' && ' · ' + figure.key.row}
            </Button>)}
          </Dialog>}
          {insertMenu.menu}
          {validity.popover}
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
          <ProblemList problems={shownProblems} refusal={refusal?.kind === 'problem' ? refusal : undefined} canReveal={(problem) => !!addressedStep(problem)}
            onChoose={revealProblem} showVariant={variants.length > 1 && !problemPanel?.scope} />
        </StudioProblems>}
        activity={<HeaderActivity placement="above" open={panel?.kind === 'activity'} command={activity}
          onToggle={activity.invoke}
          onClose={() => setPanel(null)} onOpenRun={openActivityRun} projectNames={names} resultState={activityResultState} />}
        renderView={(view, slot) => {
          const details = slot === 'primary' ? primary : secondary;
          const withProblem = (body: ReactNode) => revealed?.view === view && slot === focusSlot && !fieldReveal
            ? <ProblemDestination reveal={revealed} action={graphRepair && <Button
              disabled={!ready || snapshot.saving || busy} onClick={removeRefusedVariation}>
              {studioName('removeVariation')}
            </Button>}>{body}</ProblemDestination> : body;
          if (view === 'analysis' || view === 'figures') {
            const detail = analyze[view];
            return { ...detail, body: withProblem(detail.body),
              picker: view === 'analysis' && authored?.kind === 'analyze'
                ? pickDocument('analysis', documentBase, documentScopes.analysis) : undefined,
            };
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
            const first = selectedElements[0];
            const named = (document: string) => first && !isModelAddress(first) && first.document === document ? first.element : undefined;
            const choose = (document: string) => (id: string | null, field?: string) => id === null
              ? clearSelection(elements) : selectElement(elements, { document, element: { kind: 'named', id, field } });
            if (view === 'task') {
              const document = value.task_document ?? step?.id ?? '';
              const selected = named(document)?.id;
              return {
                padded: true,
                picker: pickDocument('task', value.task_document ?? undefined, documentScopes.task),
                body: withProblem(<TrialTaskForm view={value} commit={commit} changed={step?.kind === 'evaluate'
                  ? studioEvaluateValueSource({ kind: 'evaluate', step: step.id }) : studioName('changed')} />),
                drawer: <TrialTaskOutline sections={drawerSections} view={value} selected={selected} select={choose(document)} />,
                inspector: selected && <TrialTaskInspector view={value} commit={commit} selected={selected} />,
              };
            }
            const document = value.objective_document ?? step?.id ?? '';
            const element = named(document);
            const selected = element && { term: element.id, field: element.field };
            return {
              picker: pickDocument('objective', value.objective_document ?? undefined, documentScopes.objective),
              body: withProblem(<TrialObjectiveTable view={value} commit={commit} selected={selected} select={choose(document)} />),
              drawer: <TrialObjectiveOutline sections={drawerSections} view={value} commit={commit} selected={selected?.term} select={choose(document)} />,
              inspector: <TrialObjectiveInspector view={value} commit={commit} selected={selected} />,
            };
          }
          if (view === 'training') {
            const scope = documentScopes.training;
            return {
              padded: true,
              picker: pickDocument('training', trainingDocument, scope),
              body: authoring ? withProblem(<StudioTrainAuthoringForm {...authoring} run={namedRun}
                runPending={!data && !projection.isError} trainingDocument chosen={scope ? [...scope.lit] : undefined}
                fieldRows={scope ? undefined : detailInputs.fieldRows} fieldAxes={scope ? undefined : detailInputs.fieldAxes} />)
                : noView,
            };
          }
          if (view === 'model' || view === 'task' || view === 'scene') {
            const scope = slot === focusSlot && view === focusView ? activeDocumentScope : undefined;
            const picker = pickDocument('model', details[view].documentId, scope);
            return { ...details[view], picker: picker && { ...picker, refusal: picker.refusal ?? details[view].refusal },
              header: details[view].path, body: withProblem(details[view].body) };
          }
          if (!step) return { body: noView };
          if (view === 'document') return { body: withProblem(report.body ?? noView),
            picker: authored?.kind === 'report' ? pickDocument('report', documentBase) : undefined };
          if (view === 'runs') {
            const runSelection = { step: shownPlan.steps.find(({ id }) => id === step.id) ?? step, state,
              onSelectionChange: mapProps.onSelectionChange };
            const pivot = pivotCommand(slot);
            const pivotActive = pivotSlots[slot] === step.id;
            const run = state.focus?.stepId === step.id ? step.runs.find(({ id }) => id === state.focus?.runId) : undefined;
            const exceptions = stepExceptions(step.id).map((exception) => ({ id: exception.id, text: [
              ...exceptionPatchTexts(exception.rule.patches ?? {}, trainLabels, trainClasses),
              exceptionRuns(step, exception),
            ].join(' · ') }));
            const revealedException = exceptionReveal?.revision === saved.revision && exceptionReveal.step === step.id
              ? exceptionReveal.id : undefined;
            return {
              header: <StudioRunControls commands={step.kind === 'train' || step.kind === 'evaluate' || lightSteps.has(step.id)
                ? runCommands : undefined}
                machine={studioRunMachineControl({ catalog: machineCatalog.data, value: requestedMachine,
                  disabled: busy || machineCatalog.isPending || !connected,
                  onChange: (value) => setMachines((current) => ({ ...current, [step.id]: value })) })}
                pivotCommand={pivot} pivotActive={pivotActive} reason={run && studioRunReason(run)}
                combination={sets.combination(step.id)} />,
              inspector: <StudioStepInspector plan={plan} step={step} state={state} onReveal={revealRunMention} stepMenu={stepCommands}
                stepKey={<KeyField value={step.id} name={step.label} editing={keyingStep === step.id}
                  onEditingChange={(editing) => setKeyingStep(editing ? step.id : undefined)}
                  onCommit={(key) => changeStepKey?.(step.id, key)}
                  refusal={refusal?.kind === 'key' && refusal.step === step.id ? refusal.reason : undefined} />}
                onChooseStep={(stepId) => selection.select(selectStep(state, stepId))}
                revealed={fieldReveal && { place: fieldReveal.place, detail: fieldReveal.reason, problem: fieldReveal.problem }}
                upstreamDetail={checkpointLineage()}
                detail={selectedEvaluation?.checkpoint_series
                  ? <StudioEvaluationAttemptDetails evaluation={selectedEvaluation}
                    onShowStep={(stepId) => selection.select(selectStep(state, stepId))} /> : undefined}
                fields={(authoring || evaluateInputs || evaluationCheckpointRule || evaluationEditor) &&
                  <FocusedFields request={slot === focusSlot && fieldFocus?.step === step.id ? fieldFocus : undefined}
                    onFocused={() => setFieldFocus(undefined)}>
                    {authoring && <StudioTrainAuthoringForm {...authoring} run={namedRun} runPending={!data && !projection.isError}
                      fieldRows={detailInputs.fieldRows} fieldAxes={detailInputs.fieldAxes} />}
                    {evaluateInputs && <EvaluateSubjectFields step={evaluateInputs} project={saved.project} trains={trainSteps}
                      picking={elements} refusal={refusedAt('subject')}
                      selection={(index, reference) => {
                        const listed = Array.isArray(evaluateInputs.inputs?.subject);
                        const inputId = listed ? `subject.${index}` : 'subject';
                        const input = step.inputs.find(({ id }) => id === inputId);
                        const subset = evaluationEditor?.subjects[index];
                        const attempts = freezeAttempts(inputId);
                        return {
                          problem: listed ? selectionProblem(authoredMap?.problems ?? [], step.id, variant, 'subject', index) : undefined,
                          detail: <>
                            {input && inputDetail(input)}
                            {subset?.terms && projection.data && variant && reference.checkpoint === 'last' && <SubjectSubset
                              key={JSON.stringify([step.id, index, subset.source.id])}
                              terms={subset.terms} where={subset.where} hasWhere={reference.where !== undefined}
                              source={subset.source} sourceRows={stepSourceRows(projection.data, subset.source.id, variant)}
                              disabled={!ready || busy} refusal={refusedAt(`subset.${index}`)}
                              onEdit={(name, patch) => editEvaluation(`subset.${index}`, name, patch)}
                              onToRows={attempts
                                ? () => freezeEvaluationSubset(inputId, reference.step, attempts, index) : undefined} />}
                          </>,
                        };
                      }}
                      onEdit={ready && !busy && wholeStep ? editEvaluateSubject : undefined} />}
                    {evaluationCheckpointRule &&
                      <CheckpointRuleEditor key={JSON.stringify(['checkpoint', step.id, variant, checkpointScope])}
                        initial={evaluationCheckpointRule}
                        disabled={!ready || busy} readOnly={Boolean(stepCheckpointRule && subject?.checkpoint_axis)}
                        refusal={refusedAt('checkpoints')}
                        onSave={editEvaluationCheckpoint}
                        onToSweep={canSweepCheckpoint ? () => { void applyCheckpointSweepEdit({ kind: 'to_sweep' }); } : undefined} />}
                    {evaluationEditor?.domain && <TrialSetEditor key={step.id} initial={evaluationEditor.authored.trialSet}
                      domain={evaluationEditor.domain} disabled={!ready || busy} refusal={refusedAt('trialSet')}
                      onSave={(rule) => editEvaluation('trialSet', studioName('applyTrials'),
                        { op: 'replace', path: '/base/inline/params/trial_set', value: rule })} />}
                  </FocusedFields>}
                attachments={(exceptions.length > 0 || listedInputs.length > 0) && <>
                  {exceptions.length > 0 && <SweepExceptions exceptions={exceptions} revealed={revealedException}
                    refusal={refusal?.kind === 'exception' && refusal.step === step.id
                      ? { id: refusal.exception, reason: refusal.reason } : undefined}
                    onRemove={ready && !busy && !saved.project.sweep ? removeSweepException : undefined} />}
                  {listedInputs.length > 0 && <ul aria-label={studioName('inputs')}>{listedInputs.map((input) => {
                    const source = input.sourceStepId && plan.steps.find((item) => item.id === input.sourceStepId);
                    return <li key={input.id}>
                      <InspectorRow onClick={input.sourceStepId ? () => selection.select(selectStep(state, input.sourceStepId!)) : undefined}>
                        <span className="text-shell-muted">{input.label}</span>
                        {source && <>{' · '}{source.label}</>}
                      </InspectorRow>
                      {!subjectSelection(input.id) && inputDetail(input)}
                    </li>;
                  })}</ul>}
                </>} />,
              body: withProblem(<StudioRuns plan={shownPlan} {...runSelection}
                region={planUnavailable ?? (step.kind === 'train' || step.runs.length > 0
                  ? { kind: 'result', missing: [] } : { kind: 'empty' })}
                pending={planPending} pivotActive={Boolean(pivot) && pivotActive} runMenu={runMenu} axisMenu={axisMenu} />),
            };
          }
          const resultsFailure = planError('results');
          if (view === 'results' && (resultsFailure || !projection.data || (!pending && !outside && connected))) return {
            padded: true,
            body: <SavedResults shown={projection.data} current={!!data} failure={resultsFailure} result={shownResult}
              project={saved.project_id} fallback={withProblem(noView)}
              stopped={stoppedSubject && stoppedNewStep && <section aria-labelledby={stoppedCheckpointId}
                className="flex items-center gap-2">
                <h3 id={stoppedCheckpointId} className="font-semibold">{checkpointLabel(stoppedSubject.at)}</h3>
                <Button {...commandAttributes(stoppedNewStep)}
                  disabled={!stoppedNewStep.enabled} onClick={stoppedNewStep.invoke}>{studioName('newStep')}</Button>
              </section>}
              onReplay={(update) => {
                evaluationReplay.chooseCheckpoint(update);
                setWindowState((current) => showProjectWindowView(current, 'scene', slot));
                requestSlotFocus((count) => count + 1);
              }} />,
          };
          return { body: withProblem(noView) };
        }}
      />
      {runRemoval.dialog}
    </RunRemovalContext.Provider>
  );
}

/** The step's fields in the inspector, putting the keyboard on the requested one once it shows. */
function FocusedFields({ request, onFocused, children }: {
  request: FieldFocusRequest | undefined; onFocused(): void; children: ReactNode;
}) {
  const root = useRef<HTMLDivElement>(null);
  useFieldFocus(root, request, onFocused);
  return <div ref={root} className="grid content-start gap-3">{children}</div>;
}

function ProblemDestination({ reveal, action, children }: { reveal: Reveal; action?: ReactNode; children: ReactNode }) {
  const reason = useRef<HTMLParagraphElement>(null);
  useEffect(() => {
    reason.current?.focus();
  }, [reveal]);
  return <div className="flex h-full min-h-0 flex-col">
    <div className="flex items-center justify-between gap-2 border-b border-shell-divider px-3 py-2 text-sm">
      <p ref={reason} tabIndex={-1}>
        <span className="font-medium">{studioPlaceName(reveal.place)}</span>{' · '}
        {action && reveal.problem && <>{reveal.problem.path}{' · '}</>}{reveal.reason}
      </p>
      {action}
    </div>
    <ScrollFade className="flex-1">{children}</ScrollFade>
  </div>;
}

/** The document a problem's place is in, or else its step: what the Problems panel groups it under. */
function problemHolder(problem: StudioProjectWindowMapProblem) {
  const place = problem.place;
  return place && 'document' in place && place.document ? place.document : problem.step;
}

/** Whether a place is its document as a whole, which the group heading naming that document already says. */
function namedByHolder(problem: StudioProjectWindowMapProblem, holder: string | null) {
  const place = problem.place;
  return !!place && 'document' in place && place.document === holder && !(place.part === 'task' && place.parameter);
}

function ProblemList({ problems, refusal, onChoose, canReveal, showVariant }: {
  problems: readonly StudioProjectWindowMapProblem[];
  /** The problem whose repair was refused, and why. */
  refusal?: { problem: string; reason: string };
  onChoose(problem: StudioProjectWindowMapProblem): void;
  canReveal(problem: StudioProjectWindowMapProblem): boolean;
  showVariant: boolean;
}) {
  const groups = new Map<string | null, StudioProjectWindowMapProblem[]>();
  for (const problem of problems) {
    const name = problemHolder(problem);
    if (!groups.has(name)) groups.set(name, []);
    groups.get(name)!.push(problem);
  }
  return <div className="space-y-2">{[...groups].map(([name, entries]) => {
    const first = entries.find(canReveal);
    return <section key={name ?? ''}>
      {name && <h2 className="font-medium">{first
        ? <Button variant="ghost" size="content" className="rounded-control px-control-inline text-left font-medium"
          onClick={() => onChoose(first)}>{name}</Button>
        : name}</h2>}
      <ul>{entries.map((problem, index) => <li key={index} className="break-words">
        <ProblemRow problem={problem} refusal={refusal?.problem === problemKey(problem) ? refusal.reason : undefined}
          context={[problem.step !== name ? problem.step : null, showVariant ? problem.variant : null].flatMap((part) => part ? [part] : [])}
          placeShown={!namedByHolder(problem, name)}
          onReveal={canReveal(problem) ? () => onChoose(problem) : undefined} />
      </li>)}</ul>
    </section>;
  })}</div>;
}
