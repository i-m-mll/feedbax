import { useCallback, useEffect, useId, useMemo, useRef, useState, useSyncExternalStore, type ReactNode } from 'react';
import { useStore } from 'zustand';
import { keepPreviousData, useQuery, useQueryClient } from '@tanstack/react-query';
import { approveSavedRun, fetchExecutionTargets, fetchSavedTrainRuns, quoteSavedRun, runSavedTrain } from '@/api/jobs';
import { isRefusal, isServiceFailure, isUnanswered, landsOnGesture } from '@/api/request';
import { isStudioDefect, shownError } from '@/api/shownErrors';
import {
  authoringCatalogQuery, prepareCheckpointRuleEdit, prepareSubsetToRows, prepareTrainStart, useAuthoringCatalog,
} from '@/api/studioAuthoring';
import { StudioRegion, studioRegionIssue, type StudioRegionState } from '@/components/activity/StudioRegion';
import {
  studioAccessibleName, studioCountOf, studioFormat, studioName, studioPlaceIsDocument, studioProblemName, studioReason, studioRunReason, studioRunStatus, studioStep, type SourceNames,
} from '@/components/activity/studioVocabulary';
import { HeaderActivity } from '@/components/layout/HeaderActivity';
import {
  StudioPipelineMap, StudioPipelineMapDrawer, type MapStepRename, type PipelineMapStepSource,
} from '@/components/pipeline/StudioPipelineMap';
import { keyUnavailable } from './projectMemberName';
import { KeyField } from '@/components/ui/KeyField';
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
import { FieldIntent } from '@/components/values/fieldIntent';
import { X } from 'lucide-react';
import { Button, IconButton } from '@/components/ui/Button';
import { editedElement, useChoiceMenu } from '@/components/ui/ChoiceMenu';
import { Dialog } from '@/components/ui/Dialog';
import { Select } from '@/components/ui/Select';
import { TextField } from '@/components/ui/TextField';
import { humanizeLabel } from '@/components/values/humanizeLabel';
import type { PipelineCardPresentation } from '@/components/pipeline/pipelineMapLayout';
import { type PlanRunView, type PlanStepView, type PlanView } from '@/features/pipeline/planView';
import { labelOf, nameIn, shownDocuments, variantNames, type ShownDocuments, type VariantNames } from './memberLabels';
import { Shown, type TextPart } from '@/components/ui/ShownText';
import {
  emptyPipelineSelection, selectRun, selectStep, selectSweepValue, selectedRuns, type PipelineSelection, type PipelineSelectionState,
} from '@/features/pipeline/selection';
import { ProjectWindow, type SlotPinning } from '@/features/projectWindow/ProjectWindow';
import { ProjectSettingRows } from '@/features/projectWindow/StudioSettings';
import { useHeldGestures } from '@/features/projectWindow/heldGestures';
import { StatusMark } from '@/components/ui/StatusMark';
import { ProblemRow, StudioProblems, problemEntries, type ProblemEntry } from '@/features/projectWindow/StudioProblems';
import { holds } from './model/modelGraph';
import { bindStudioCommand, commandAttributes, jobAllowsCommand, type StudioCommand } from '@/features/projectWindow/studioCommands';
import {
  noViews, pinProjectWindowSlot, projectWindowSlotView, showProjectWindowView, useProjectWindowState, type ProjectWindowSlot,
  type ProjectWindowView,
} from '@/features/projectWindow/projectWindowState';
import { pinnableViews, stepModelSource, usePinnedSubject } from './pinnedSlot';
import {
  ModelAddressSchema, StepReferenceSchema, StudioProjectRenameSchema, type PresentationDocument, type PresentationPosition, type ProjectValueEdit,
  type FigureLayoutMeta, type StudioActivityJob, type StudioJobSubmission, type StudioMachineQuote, type StudioProjectDocument, type StudioProjectWindowMapProblem,
  type LoadedModelReference, type StudioSavedTrainKey, type StudioSavedTrainRuns, type StepReference, type SubsetTerms,
  type StudioCheckpointRuleRequest, type JsonValue, type SweepExceptionTerms, type StudioPlanReviewSource, type StudioProjectWindowMapException, type ModelAddress, type PatchValue,
  type StudioClassField, type StudioProjectSnapshot,
} from '@/generated/studioContracts';
import { useConnection } from '@/api/connection';
import { useStudioJobActions, useStudioJobs } from '@/hooks/useStudioJobs';
import { authoredPointer, declarationAddress, ownDefinition, modelDocumentId, trialDocumentId } from './authoredEdits';
import { withBaseDescendants } from './baseDescendants';
import { stepDocument } from './documentPayload';
import type { SceneMarking, SceneRelate } from '@/components/scene/SceneRegion';
import { reasonOf } from '@/components/values/fieldModel';
import { useManagedProjectDetails, useRecordedTerms, useStoredModel, type NamedDocumentScope } from './managedProjectDetails';
import { useTrialClock } from './useTrialClock';
import { declaredEvaluation, FigureClockContext } from './figureClock';
import { drawnVariables } from './task/trialSample';
import { usePlayheadStore } from '@/components/timeline/trialClock';
import { viewOffer, viewsByKind } from './viewContent';
import {
  choosePick, clearSelection, isModelAddress, selectElement, useChoosing, useElementSelectionStore, usePick, type PickTarget,
} from '@/features/projectWindow/elementSelection';
import { createCheckpointTrain } from './createCheckpointTrain';
import { analyzeInput, createAnalyze, createReport, isReportFigureSource, type ReportFigureSource } from './createManagedStep';
import { createSynthesize, createTrain } from './createModelStep';
import { trainingDocumentId } from './trainingDocument';
import {
  authoredTrialDomain, checkpointLabel, createEvaluation, editableCheckpointRule, evaluateSubject, evaluateSubjectEdits, exactCheckpoint,
  managedTrialDomain, ordinaryEvaluation, selectionCheckpoints, type CheckpointRange, type EditableCheckpointRule, type EvaluateSubject,
  type SelectionCheckpoint, type StoppedEvaluationSubject,
} from './evaluateAuthoring';
import { bindProjectEditor } from './openProjectEditor';
import type { ProjectEditController } from './projectEditController';
import { axisProblems, bestRankGroups, projectWindowPlan, planVariants, savedRunId, stepSourceRows, unvalued } from './projectWindowPlan';
import { placeField, placeMember, placeView, removeGraphVariation, type Reveal } from './projectWindowProblems';
import { settledWindowSelection, useProjectWindowSelection } from './projectWindowSelection';
import { planWithRefusal, problemKey, type GestureFailures, type Refusal } from './refusals';
import { SubjectDocuments, TrialsEditor, type StepField } from './StudioEvaluateAuthoring';
import { SelectionRule } from './SelectionRule';
import { authoredSubset, bindingEdit, connection, exceptionRuleProblem, selectionProblems, stepSelections, subjectAxisEdits, withSubsetTerms } from './selectionEdits';
import { InputEditor, type InputGesture } from './InputEditor';
import { SynthesisForm, withMethodLabels } from './SynthesisForm';
import { documentChoice } from './referenceChoices';
import { documentsOfRole } from './useProjectDocuments';
import { StudioEvaluationAttemptDetails } from './StudioEvaluationResults';
import { holdsProducts } from './OperationProducts';
import { focusedResult, SavedResults } from './SavedResults';
import { exportLoadedModel } from '@/api/projectFolders';
import { saveFile } from '@/components/ui/chooseFile';
import { filesRemoval, RunRemovalContext, useRunRemovalDialog } from './runRemoval';
import { AskStoreContext, askStoreWith, sendWithStore } from './storeChoice';
import {
  jsonPointerPath, StudioStepField, StudioTrainAuthoringForm, SweepFieldItems, useTrainClassFields, useTrainFieldLabels,
} from './StudioTrainAuthoringForm';
import {
  TrialSweepItems, TrialObjectiveInspector, measureAnchor, TrialObjectiveOutline, TrialObjectiveTable, type TrialRuns, TrialTaskForm, TrialTaskInspector, TrialTaskOutline, useTrialAuthoring,
  type TrialChangeScope,
} from './StudioTrialAuthoring';
import { inGrid, TrialGridInspector } from './task/trialGrid';
import { selectedVariable, TrialTaskDocument, TrialVariableInspector, unfedTasks } from './task/trialTask';
import { documentFamily, fittedModel, useProjectDocuments, type DocumentRole } from './useProjectDocuments';
import { LoadedModelFacts } from './LoadedModelFacts';
import { useManagedAnalyze } from './useManagedAnalyze';
import { useSavedPlan } from './useSavedPlan';
import { figureName, stepReport, useManagedReport } from './useManagedReport';
import { ScrollFade } from '@/components/ui/ScrollFade';
import type { AxisMenu } from '@/components/pipeline/PipelineRunMarks';
import { OfferedSets, setAuthoring } from './setAuthoring';
import { useSceneGeometry } from '@/components/scene/useSceneGeometry';
import type { SceneTrial, StateValues } from '@/components/scene/trialScene';

type EditorSession = ReturnType<typeof bindProjectEditor>;
/** The Model's own initial state, at which the Scene's geometry tells whether the Model has anything to draw. */
const restPose: readonly StateValues[] = [{}];
/** What a Scene draws of a step the clock does not read: its Model at rest. */
const restTrial: SceneTrial = { picked: undefined, possible: [], reading: false, frame: 0, task: undefined, trails: false };
const slots = ['primary', 'secondary'] as const;
/** The anchor of an Evaluate's sampled number of trials, which the Trials editor draws. */
const trialCount = '/params/trial_set/count';

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
      setLoad({ error: shownError(cause) });
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

/** The key of a step being added, which no authored step can have. */
const addingStepId = ' adding';

function ManagedProjectWindow({ session, onNewStep }: { session: EditorSession } & ProjectWindowCommands) {
  const stoppedCheckpointId = useId();
  const { editor } = session;
  const snapshot = useSyncExternalStore(editor.subscribe, editor.getSnapshot);
  const { saved, configuration, pending, outside, planVersion } = snapshot;
  const jobs = useStudioJobs();
  const connectionKind = useConnection().kind;
  const connected = connectionKind === 'connected';
  /** A window that cannot reach the service; one still connecting may yet, and a gesture made then waits. */
  const reachable = connectionKind !== 'unreachable';
  const queries = useQueryClient();
  const actions = useStudioJobActions();
  const [windowState, setWindowState] = useProjectWindowState(saved.project_id);
  const drawerSections = useDrawerSections(windowState, setWindowState);
  const elements = useElementSelectionStore();
  const playhead = usePlayheadStore();
  const armedPick = usePick(elements);
  const choosing = useChoosing(elements);
  const mapPick = useMemo(() => armedPick && {
    accepts: armedPick.accepts, choose: (target: PickTarget) => { choosePick(elements, target); },
  }, [armedPick, elements]);
  const selectedElements = useStore(elements, (current) => current.selection);
  const mapViewport = useRef<CanvasViewport>(null);
  const plans = useSavedPlan(editor);
  const namedDocuments = shownDocuments(snapshot.draft, plans.project.documents);
  const selection = useProjectWindowSelection(Object.keys(saved.project.steps ?? {})[0] ?? null, plans.inTurn);
  /** The failure of the last gesture that has no object of its own to land on. */
  const [commandError, setCommandError] = useState<unknown>();
  /** The last refused gesture, shown on the object it acted on. */
  const [refusal, setRefusal] = useState<Refusal>();
  /** The refusal of the selected Train step's authoring read, which is drawn on the step while it holds. */
  const [readRefusal, setReadRefusal] = useState<Refusal & { kind: 'step' }>();
  const commandFailed = (error: unknown) => {
    if (!isUnanswered(error)) setCommandError(error);
  };
  const [machines, setMachines] = useState<Readonly<Record<string, string>>>({});
  const authoringCatalog = useAuthoringCatalog().data;
  const stepInputs = authoringCatalog?.step_inputs;
  const [machineQuote, setMachineQuote] = useState<StudioMachineQuote>();
  const [rename, setRename] = useState<{ name: string; revision: string }>();
  const projectNameProblemId = useId();
  const projectNameProblem = refusal?.kind === 'projectName' && rename?.revision === saved.revision && !outside
    ? refusal.reason : undefined;
  /** How many of its edits are waiting or in flight. */
  const [edits, setEdits] = useState(0);
  /** A step being added, drawn on the map from the gesture until its gestures settle on a plan that has it. */
  const [adding, setAdding] = useState<{ kind: PlanStepView['kind']; after?: string; revision: string }>();
  const busy = edits > 0;
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
  const [sweepStep, setSweepStep] = useState<string>();
  const submission = useRef<{ intent: string; token: string }>();
  const { shown } = plans;
  /** The plan shown is of an earlier revision while the shown revision's plan is read, which its regions mark. */
  const planPending = !!shown && !plans.current;
  useEffect(() => {
    if (adding && edits === 0 && plans.current) setAdding(undefined);
  }, [adding, edits, plans.current]);
  const refetchPlan = plans.refetch;
  const refreshProjection = useCallback(() => refetchPlan(), [refetchPlan]);
  const validity = useResultValidity({
    projectId: saved.project_id, revision: saved.revision, refresh: refreshProjection, onError: commandFailed,
  });
  const variants = useMemo(() => planVariants(shown), [shown]);
  const variant = selection.variantId ?? variants[0] ?? null;
  const missingVariant = selection.variantId !== null && !variants.includes(selection.variantId);
  const primaryVariant = variant !== null && variant === variants[0];
  /** Whether a command of the primary variant applies: until a plan names the variants it may, and a gesture waits. */
  const variantApplies = shown ? primaryVariant : true;
  const plan = useMemo(() => withMethodLabels(projectWindowPlan(plans.project, plans.names, shown, variant, validity.checking),
    plans.project, authoringCatalog), [plans.project, plans.names, shown, variant, validity.checking, authoringCatalog]);
  const shownPlan = useMemo(() => planWithRefusal(planWithRefusal(plan, readRefusal), refusal), [plan, readRefusal, refusal]);
  const activityResultState = useMemo<ActivityResultState>(() => {
    const states = new Map(plan.steps.flatMap(({ runs }) => runs.flatMap(({ state, results }) =>
      results.map(({ jobId }) => [jobId, state] as const))));
    return (job) => states.get(job.job_id);
  }, [plan]);
  const activityRunName = useMemo(() => {
    const map = shown?.window_map;
    const names = new Map(map?.rows.map(({ key, name }) => [savedRunId(key), name]));
    const requests = new Map(map?.requests?.filter(({ element }) => element == null)
      .map(({ key, request_id }) => [savedRunId(key), request_id]));
    return (job: StudioActivityJob) => job.project_id === saved.project_id && requests.get(job.run_key) === job.request_id
      ? names.get(job.run_key) : undefined;
  }, [shown, saved.project_id]);
  const runMenu: RunMenu = (runs, anchor) => {
    const owner = runs[0] && plan.steps.find(({ id }) => id === (JSON.parse(runs[0].id) as string[])[1]);
    return [
      ...validity.commands(runs, anchor), ...owner ? [sets.duplicateRun(owner, runs)] : [],
      ...owner?.kind === 'train' && runs.length === 1 ? [exportModel(runs[0])] : [],
    ];
  };
  /** Export: the Model and weights of a Train run's latest successful result, as a model file. */
  function exportModel(run: PlanRunView) {
    const row = shown?.rows.find(({ key }) => savedRunId(key) === run.id);
    const result = row?.result;
    const updates = result?.progress?.completed_updates;
    return bindStudioCommand('export', () => {
      if (!row || !result || !updates) return;
      void perform(async (current) => {
        const { archive, name } = await exportLoadedModel(current.project_id, {
          revision: current.revision, key: row.key, job_id: result.job_id, completed_updates: updates,
        }, editor.signal);
        await saveFile(archive, name);
      }, (reason) => ({ kind: 'runs', reasons: { [run.id]: reason } }));
    }, { subject: run.label, instance: run.id, enabled: !!updates });
  }
  const runRemoval = useRunRemovalDialog(plans.read);
  const choiceMenu = useChoiceMenu();
  const sets = setAuthoring({
    editor, project: saved.project, applies: !outside && reachable && variantApplies,
    perform, state: () => latestState.current, select: (next) => selection.select(next), confirmRemoval: runRemoval.confirm,
    ask: (anchor, choices) => askStore(anchor, choices),
    valueName: (step, axis, entry) => {
      const value = plan.steps.find(({ id }) => id === step)?.sweeps.find(({ id }) => id === axis)?.values.find(({ id }) => id === entry);
      if (!value) throw new Error(`The plan has no value ${entry} of axis ${axis} of step ${step}.`);
      return value.label;
    },
  });
  const axisMenu: AxisMenu = (stepId, axis) => {
    const planned = plan.steps.find(({ id }) => id === stepId);
    const values = planned?.sweeps.find(({ id }) => id === axis)?.values ?? [];
    const subject = planned?.kind === 'evaluate' ? planned.inputs.find(({ id }) => id === 'subject')?.sourceStepId : undefined;
    const model = modelDocumentId(planned?.documentIds) ??
      (subject ? modelDocumentId(plan.steps.find(({ id }) => id === subject)?.documentIds) : undefined);
    const [problem] = axisProblems(shown?.window_map, stepId, axis, variant);
    return {
      picker: sets.axisPicker(stepId, axis),
      commands: sets.axisCommands(stepId, axis, { model: !!model, empty: !values.some(({ id }) => id !== unvalued) }),
      removeValue: (value, label) => sets.removeValue(stepId, axis, value, label),
      refusal: refusal?.kind === 'axis' && refusal.step === stepId && refusal.axis === axis ? refusal.reason : problem?.detail,
    };
  };
  const step = plan.steps.find(({ id }) => id === selection.stepId);
  const state = shown ? settledWindowSelection(plan, selection.pipeline) : selection.pipeline;
  const latestState = useRef(state);
  latestState.current = state;
  const savedFocus = windowState.focus;
  const savedStep = windowState.stepId;
  const restoredSelection = useRef(false);
  useEffect(() => {
    if (restoredSelection.current || !savedFocus && !savedStep) return;
    restoredSelection.current = true;
    if (selection.edited) return;
    if (savedFocus) selection.select(selectRun(emptyPipelineSelection, savedFocus));
    else if (savedStep && saved.project.steps?.[savedStep]) selection.select(selectStep(emptyPipelineSelection, savedStep));
  }, [savedFocus, savedStep, selection, saved.project.steps]);
  const selectedStep = selection.stepId;
  useEffect(() => {
    if (!selection.edited) return;
    setWindowState((current) => current.stepId === (selectedStep ?? undefined) ? current
      : { ...current, stepId: selectedStep ?? undefined });
  }, [selectedStep, selection.edited, setWindowState]);
  const focusStep = state.focus?.stepId;
  const focusRun = state.focus?.runId;
  useEffect(() => {
    if (!selection.edited) return;
    setWindowState((current) => current.focus?.stepId === focusStep && current.focus?.runId === focusRun ? current
      : { ...current, focus: focusStep === undefined || focusRun === undefined ? undefined : { stepId: focusStep, runId: focusRun } });
  }, [focusStep, focusRun, selection.edited, setWindowState]);
  const singleStep = state.selection?.kind === 'steps' && state.selection.stepIds.length === 1 && state.selection.stepIds[0] === step?.id;
  const wholeStep = singleStep && primaryVariant;
  /** The selected values as an edit names them; the runs an optional axis keeps are named by row. */
  const valueTerms = state.selection?.kind === 'values' && !state.selection.terms.some(({ valueIds }) => valueIds.includes(unvalued))
    ? state.selection.terms : undefined;
  /** The one value of a named axis or set entry selected on the step, which a Model edit is stored on. */
  const selectedValue = primaryVariant && valueTerms?.length === 1 && valueTerms[0].valueIds.length === 1 &&
    state.selection?.kind === 'values' && state.selection.stepId === step?.id
    ? { axis: valueTerms[0].sweepId, value: valueTerms[0].valueIds[0] } : undefined;
  const canAuthorSweep = (id: string) => primaryVariant && ownDefinition(saved.project, id, true) !== null;
  const authored = step && saved.project.steps?.[step.id];
  const subjectCheckpoint = authored?.kind === 'evaluate' ? exactCheckpoint(authored) : undefined;
  const authoring = step && (primaryVariant && ownDefinition(saved.project, step.id, true) !== null ||
    wholeStep && authored?.kind === 'train') &&
    state.selection ? {
    editor, stepId: step.id, step, selection: state.selection,
    onRefused: (reason: string | undefined) => setReadRefusal((current) => reason ? { kind: 'step', step: step.id, reason }
      : current?.step === step.id ? undefined : current),
    onChooseSweepValue: (sweepId: string, valueId: string, extend: boolean) => {
      selection.select(selectSweepValue(latestState.current, step.id, sweepId, valueId, extend));
    },
  } : null;
  const trainLabels = useTrainFieldLabels(editor, step?.kind === 'train' ? step.id : undefined);
  const trainingDocument = step?.kind === 'train' ? trainingDocumentId(step.documentIds) : undefined;
  const trainClasses = useTrainClassFields(editor, step?.kind === 'train' ? step.id : undefined);
  const runTarget: PipelineSelection = state.selection ?? { kind: 'steps', stepIds: plan.steps.map(({ id }) => id) };
  const runIds = new Set(plan.steps.flatMap((item) => selectedRuns(item, runTarget).map(({ id }) => id)));
  const shownResult = focusedResult(shown, step?.id, state.focus?.runId, runIds, resultAttempt);
  const { row, scene: sceneRows, evaluation: selectedEvaluation } = shownResult;
  const definitionRows = shown?.window_map?.rows.filter(({ key }) => key.step === step?.id && key.variant === variant) ?? [];
  const selectedDefinitionRows = definitionRows.filter(({ key }) => runIds.has(savedRunId(key)));
  const namedRow = selectedDefinitionRows.find(({ key }) => savedRunId(key) === state.focus?.runId) ??
    selectedDefinitionRows[0] ?? definitionRows[0];
  const namedRun = namedRow?.key;
  const modelSource = stepModelSource(plan, step);
  const stored = useStoredModel(editor, step, modelSource);
  /** The Loaded Model the Model view shows in place of the step's Model, which the step does not read. */
  const [viewedLoaded, setViewedLoaded] = useState<{ step: string | undefined; document: string }>();
  const viewing = viewedLoaded?.step === step?.id ? viewedLoaded?.document : undefined;
  const viewedModel = useStoredModel(editor, undefined, fittedModel(viewing ? saved.project.documents?.[viewing] : undefined));
  const loaded = viewing && viewedModel ? { document: viewing, model: viewedModel } : undefined;
  const restGeometry = useSceneGeometry(saved.project_id, stored ? { document_id: stored.id, revision: saved.revision } : null, restPose)?.(0);
  const scoredBy = authored?.kind === 'evaluate' ? evaluateSubject(authored) : undefined;
  /** An Evaluate is scored by the Objective of the Train that made its checkpoint, or by the one it names itself. */
  const scored = authored?.kind !== 'evaluate' || scoredBy?.kind === 'steps' || scoredBy?.kind === 'document' && !!scoredBy.objective;
  const operationProducts = step?.kind === 'analyze' && !!namedRun && (shown?.window_map?.analysis_elements ?? [])
    .some(({ key, kind, requests }) => kind === 'analyze' && savedRunId(key) === savedRunId(namedRun) && requests.some(holdsProducts));
  const views = !step ? noViews : viewOffer({
    step, trainingForm: !!trainingDocument && !!authoring, planRead: !!shown, run: !namedRow ? 'none' : namedRow.resolved === false ? 'unresolved' : 'resolved',
    objective: scored, model: stored?.value,
    scenePlaces: restGeometry?.kind === 'ready' ? restGeometry.geometry.mechanisms.length > 0 : undefined,
    runProblems: (namedRun && step.runs.find(({ id }) => id === savedRunId(namedRun))?.problems) ?? [],
    results: !!plans.error || !!shownResult.row || !!shownResult.evaluation?.checkpoint_series ||
      !!shownResult.evaluation?.result?.origin || operationProducts,
    report: !!stepReport(snapshot, step?.id).report,
  });
  const pins = windowState.secondaryView ? windowState.pins ?? {} : {};
  const preferredView = (slot: ProjectWindowSlot) => slot === 'primary' ? windowState.primaryView : windowState.secondaryView;
  /** The slot pinned to a step other than the selected one whose Scene is shown, which the clock reads. */
  const clockSlot = slots.find((slot) => pins[slot] && pins[slot].stepId !== step?.id && preferredView(slot) === 'scene');
  const held = {
    primary: usePinnedSubject({ editor, plan, shown, variant, pin: pins.primary, selectedStep: step?.id, clocked: clockSlot === 'primary' }),
    secondary: usePinnedSubject({ editor, plan, shown, variant, pin: pins.secondary, selectedStep: step?.id, clocked: clockSlot === 'secondary' }),
  };
  const clockSubject = clockSlot && held[clockSlot].step ? held[clockSlot] : undefined;
  const slotOffer = (slot: ProjectWindowSlot) => held[slot].step ? held[slot].views : views;
  const primaryView = projectWindowSlotView(windowState, 'primary', slotOffer('primary').available);
  const secondaryView = projectWindowSlotView(windowState, 'secondary', slotOffer('secondary').available);
  const focusSlot = secondaryView && windowState.focusedSlot === 'secondary' ? 'secondary' : 'primary';
  const focusView = focusSlot === 'secondary' ? secondaryView : primaryView;
  /** Whether the focused slot shows a pinned step other than the selected one, which the selection's commands do not reach. */
  const focusHeld = !!held[focusSlot].step;
  const pinRun = namedRun && savedRunId(namedRun);
  useEffect(() => {
    setWindowState((current) => slots.reduce((next, slot) => {
      const pin = next.pins?.[slot];
      return pin && pin.stepId === step?.id && pinRun && pin.runId !== pinRun
        ? pinProjectWindowSlot(next, slot, { ...pin, runId: pinRun }) : next;
    }, current));
  }, [pinRun, setWindowState, step?.id]);
  const projectSteps = saved.project.steps;
  useEffect(() => {
    setWindowState((current) => slots.reduce((next, slot) => {
      const pin = next.pins?.[slot];
      return pin && !projectSteps?.[pin.stepId] ? pinProjectWindowSlot(next, slot, undefined) : next;
    }, current));
  }, [projectSteps, setWindowState]);
  const revealed = problemReveal?.revision === saved.revision && problemReveal.variant === variant &&
    problemReveal.reveal.step === step?.id && problemReveal.runId === (state.focus?.runId ?? null)
    ? problemReveal.reveal : undefined;
  /** A revealed field its view marks and focuses itself, instead of a banner; a step's own fields are in the inspector. */
  const fieldReveal = revealed && revealed.view === placeView(revealed.place) && placeField(revealed.place) ? revealed : undefined;
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
      : field === 'task' || field === 'objective' ? trialDocumentId(ids, field)
          : field === 'analysis' ? ids?.payload : trainingDocumentId(ids);
    const evaluating = step?.kind === 'evaluate' && (field === 'graph' || field === 'task');
    const analyzing = step?.kind === 'analyze' && field === 'analysis';
    const producing = step?.kind === 'train' || step?.kind === 'synthesize' && field !== 'training';
    if (!document || !step || !singleStep || !shown || !producing && !evaluating && !analyzing) return undefined;
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
    if (!read || !('set' in read) || !step || !shown) return undefined;
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
  const activeDocumentScope = windowState.mode === 'map' || focusHeld ? undefined
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
    // An element is selected within the step it was chosen in; choosing another step shows that step.
    if (id !== step?.id) clearSelection(elements);
    selection.select(selectStep(state, id, extend));
  }
  const ready = Boolean(shown && !pending && !outside && connected && !plans.error);
  /**
   * Whether the plan's commands apply. Only lasting conditions count: a plan still loading or a window still
   * connecting leaves the workspace unsettled, and a gesture made then is held until it settles.
   */
  const applies = Boolean((shown || plans.waiting) && !outside && reachable && !plans.error);
  /** The stopped attempt of a focused single run whose checkpoints an Evaluate can use, in the plan it was read from. */
  const stoppedAttempt = ({ row, attempt }: ReturnType<typeof focusedResult>, source: StudioSavedTrainRuns | undefined) =>
    !saved.project.sweep && windowState.mode !== 'map' &&
    state.selection?.kind === 'rows' && state.selection.runIds.length === 1 &&
    (primaryView === 'results' || secondaryView === 'results') && row && attempt?.status === 'stopped' &&
    source?.window_map?.rows.some(({ key, evaluation_conditions }) =>
      evaluation_conditions != null && savedRunId(key) === savedRunId(row.key))
      ? attempt.job_id : undefined;
  const checkpointAttempt = stoppedAttempt(shownResult, shown);
  const stoppedPlan = useFollowingQuery({
    queryKey: ['saved-stopped-checkpoint', saved.project_id, saved.revision, checkpointAttempt],
    queryFn: ({ signal }) => fetchSavedTrainRuns(saved.project_id, saved.revision, signal, checkpointAttempt),
    enabled: ready && checkpointAttempt !== undefined,
    placeholderData: keepPreviousData,
    gcTime: 0,
  }, String(planVersion));
  useEffect(() => {
    if (!ready || !focusFirstModel || step?.id !== focusFirstModel) return;
    requestSlotFocus((count) => count + 1);
    setFocusFirstModel(undefined);
  }, [ready, focusFirstModel, step?.id]);
  const names = useMemo(() => new Map([[saved.project_id, configuration.project]]), [saved.project_id, configuration.project]);

  const runRows = shown?.rows.filter(({ key }) => runIds.has(savedRunId(key))) ?? [];
  const evaluationRows = shown?.window_map?.evaluations?.filter(({ key }) => runIds.has(savedRunId(key))) ?? [];
  const authoredMap = shown?.window_map;
  /** The steps whose runs the Service saves as light results, which the window map's rows schedule. */
  const lightSteps = new Set(plan.steps.filter(({ kind }) => kind === 'synthesize' || kind === 'analyze' || kind === 'report')
    .map(({ id }) => id));
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
  /** Whether the chosen machine's catalog is still being read, which leaves the workspace unsettled. */
  const machinePending = !!remoteMachine && machineCatalog.isFetching;
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
  const evaluationEditor = evaluationAuthoring && evaluationSources && evaluationSources.length === evaluationAuthoring.subjects.length
    ? {
      authored: evaluationAuthoring, subjects: evaluationSources,
      domain: managedTrialDomain(shown, evaluationSources.length === 1
        ? subjectIds ? evaluationSources[0].source.runs.filter(({ id }) => subjectIds.has(id))
          : evaluationSources[0].reference.where == null ? evaluationSources[0].source.runs : []
        : evaluationSources.flatMap(({ source }) => source.runs)) ?? authoredTrialDomain(evaluationAuthoring.trialSet),
    } : undefined;
  const checkpointVariant = variant && saved.project.sweep?.rows?.[variant] ? variant : undefined;
  const evaluated = authored?.kind === 'evaluate' && !authored.deltas?.length ? authored : undefined;
  const subjectReferences = stepSelections(evaluated?.inputs?.subject);
  const listedSubject = Array.isArray(evaluated?.inputs?.subject);
  const localSubject = StepReferenceSchema.safeParse(evaluated?.inputs?.subject);
  /** A checkpoint axis or a variant's patch says the rule the plan read; otherwise the authored reference does. */
  const stepCheckpointRule = singleStep && (wholeStep && !saved.project.sweep || checkpointVariant)
    && localSubject.success && (localSubject.data.checkpoint != null || subject?.checkpoint_axis)
    ? editableCheckpointRule(subject?.checkpoint_axis || checkpointVariant ? subject?.checkpoint : localSubject.data.checkpoint)
    : undefined;
  const listedCheckpoints = wholeStep && !saved.project.sweep && listedSubject && subjectReferences &&
    !Object.values(evaluated?.sweep?.axes ?? {}).some((axis) => axis && 'checkpoint_rule' in axis)
    ? selectionCheckpoints(subjectReferences) : undefined;
  const stepLabel = (id: string) => {
    const planned = plan.steps.find((item) => item.id === id);
    if (!planned) throw new Error(`The plan has no step ${id}.`);
    return planned.label;
  };
  const plannedAxis = (step: string, axis: string) => {
    const planned = plan.steps.find((item) => item.id === step)?.sweeps.find((item) => item.id === axis);
    if (!planned) throw new Error(`The plan has no axis ${axis} of step ${step}.`);
    return planned;
  };
  const sourceNames: SourceNames = {
    step: stepLabel,
    document: (key) => labelOf(namedDocuments, key),
    axis: (step, axis) => plannedAxis(step, axis).label,
    value: (step, axis, value) => {
      const planned = plannedAxis(step, axis).values.find((item) => item.id === value);
      if (!planned) throw new Error(`The plan has no value ${value} of axis ${axis} of step ${step}.`);
      return planned.valueLabel;
    },
  };
  /** The rows of its producer that selection `index` of the step's input `role` reads. */
  function selectionRows(role: string, index: number) {
    const binding = authored?.inputs?.[role];
    const reference = stepSelections(binding)?.[index];
    if (!reference) return [];
    const members = mapInputs?.find(({ id }) => id === (Array.isArray(binding) ? `${role}.${index}` : role))?.subset?.members;
    const read = members && new Set(members.map(savedRunId));
    return shown?.window_map?.rows.filter(({ key }) =>
      key.step === reference.step && key.variant === variant && (!read || read.has(savedRunId(key)))) ?? [];
  }
  /** The updates every run that `selections` read has, among the runs whose extent the plan knows. */
  function checkpointRange(role: string, selections: readonly number[]): CheckpointRange | undefined {
    let range: CheckpointRange | undefined;
    for (const index of selections) {
      for (const { key, origin, target_total } of selectionRows(role, index)) {
        if (origin == null || target_total == null) continue;
        const step = stepLabel(key.step);
        range = {
          after: !range || origin > range.after.update ? { step, update: origin } : range.after,
          through: !range || target_total < range.through.update ? { step, update: target_total } : range.through,
        };
      }
    }
    return range;
  }
  const exactSources = selectedEvaluation?.result?.checkpoint_sources;
  const exactSource = subjectCheckpoint !== undefined && exactSources?.length === 1 ? exactSources[0] : undefined;
  const trialStep = step?.kind === 'train' || step?.kind === 'synthesize' || step?.kind === 'evaluate';
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
  const modelProblems = shown?.window_map?.problems.filter((problem) => placeMember(problem.place) &&
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
  const trialEditScope = !trialScope ? undefined : trialScope.kind === 'variant' ? { kind: 'step' } as const : trialScope;
  const documentEditable = !outside;
  const trialEditable = documentEditable && Boolean(trialScope);
  const trialChangeScope: TrialChangeScope = () => trialEditScope;
  const askStore = askStoreWith(choiceMenu.ask, {
    step: stepLabel,
    run: (key) => {
      const row = shown?.window_map?.rows.find((planned) => savedRunId(planned.key) === savedRunId(key));
      if (!row) throw new Error(`The plan has no run ${savedRunId(key)}.`);
      return row.name;
    },
  });
  // A run whose own definition is refused has no Task or Objective to read; its views are dimmed by its problems.
  const trial = useTrialAuthoring(editor, saved.project_id, saved.revision, namedRun, trialStep && namedRow?.resolved !== false,
    trialEditable, trialChangeScope, runRemoval.confirm, askStore);
  const clockView = clockSubject ? clockSubject.trial : trial.view;
  const clockStep = clockSubject ? clockSubject.step : step;
  /** The runs the shown figure declares its data was read from, whose recorded trials the clock offers for an Analyze. */
  const [figureRuns, declareFigureRuns] = useState<FigureLayoutMeta['runs']>();
  const figureEvaluation = step?.kind === 'analyze' ? declaredEvaluation(shown, figureRuns) : undefined;
  /** With no one run focused, an Evaluate's clock shows the first selected run that recorded trials. */
  const firstEvaluation = step?.kind === 'evaluate' ? sceneRows.find((row) => row.result ?? row.previous) : undefined;
  const trialClock = useTrialClock({
    windowState, setWindowState, projectId: saved.project_id, revision: saved.revision, view: clockView,
    taskDocument: clockView && (clockView.task_document ?? clockStep?.id),
    objectiveDocument: clockView && (clockView.objective_document ?? clockStep?.id),
    run: clockSubject ? clockSubject.namedRun : namedRun,
    stored: clockSubject ? clockSubject.stored : stored, evaluation: clockSubject ? clockSubject.evaluation : selectedEvaluation ?? firstEvaluation ?? figureEvaluation,
    sceneVisible: windowState.mode !== 'map' && [primaryView, secondaryView].includes('scene'), elements, playhead,
    shelfView: windowState.mode === 'map' ? undefined : primaryView ?? undefined,
  });
  const evaluationReplay = trialClock.replay;
  const scoredEvaluation = step?.kind === 'evaluate' && selectedEvaluation?.status === 'succeeded' ? selectedEvaluation : undefined;
  const recordedTerms = useRecordedTerms(saved.project_id, saved.revision, scoredEvaluation, evaluationReplay.checkpoint,
    windowState.mode !== 'map' && [primaryView, secondaryView].includes('objective'));
  const documentBase = authored && stepDocument(authored.base);
  const pickDocument = useProjectDocuments({
    editor, documents: namedDocuments, stepId: step?.id, run: namedRun, scope: trialEditScope,
    editable: documentEditable,
    selectable: new Set<DocumentRole>([
      ...!trialEditable ? [] : step?.kind === 'train' ? ['model', 'task', 'objective', 'training'] as const
        : step?.kind === 'synthesize' ? ['model', 'task', 'objective'] as const : ['model', 'task'] as const,
      ...documentEditable && wholeStep ? ['analysis', 'report'] as const : [],
    ]),
    onError: setCommandError,
    onLoaded: (document) => setViewedLoaded({ step: step?.id, document }),
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
  const unlisted = stepCheckpointRule || (!listedSubject ? scopedRule : undefined);
  const checkpointEditors: readonly SelectionCheckpoint[] | undefined = listedCheckpoints ??
    (unlisted ? [{ rule: unlisted, selections: [0] }] : undefined);
  const canSweepCheckpoint = Boolean(namedRun && subject?.can_sweep_checkpoint);
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
  const objectiveDocument = trial.view && (trial.view.objective_document ?? step?.id);
  const relate: SceneRelate | undefined = step?.kind === 'train' && trial.relate && objectiveDocument &&
    trial.view?.objective_source?.kind !== 'train' ? async (first, second) => {
      const answer = await trial.relate!(first, second);
      if (!answer || 'refused' in answer) return answer?.refused;
      selectElement(elements, { document: objectiveDocument, element: { kind: 'named', id: answer.term, field: 'measure' } });
      setFieldFocus({ key: crypto.randomUUID(), step: step.id, target: measureAnchor(answer.term) });
      return undefined;
    } : undefined;
  const taskDocument = trial.view && 'variables' in trial.view.task ? trial.view.task_document ?? step?.id : undefined;
  const marking = useMemo((): SceneMarking | undefined => trial.commit && taskDocument ? {
    document: taskDocument,
    mark: async (variable, marker) => reasonOf(await trial.commit!(studioAccessibleName('set', studioName('marker')),
      { kind: 'structure', changes: [{ op: 'set_variable_marker', id: variable, marker }] })),
  } : undefined, [trial.commit, taskDocument]);
  const detailInputs = {
    editor, step, stepId: step?.id ?? null, focus: state.focus, selection: state.selection, relate, marking,
    variantId: variant, evaluation: selectedEvaluation, sceneTrial: trialClock.scene, checkpoints: evaluationReplay.checkpoints,
    namedRun, documentScopes, planCurrent: plans.current, elements, trial: trial.view, valueEditing: !!selectedValue, evaluateScope: evaluateScope || undefined, stored, modelProblems, selectedSweepValues, confirmRunRemoval: runRemoval.confirm, askStore,
    drawerSections,
    openModelProblems: (element: ModelAddress, opener: HTMLElement) => {
      const owner = modelProblems?.find((problem) => {
        const member = placeMember(problem.place);
        return member && holds(element, member.address);
      });
      if (owner?.step) setPanel({ kind: 'problems', scope: { step: owner.step, variant, element }, opener });
    },
    fieldRows: state.selection && state.selection.kind !== 'steps' ? selectedDefinitionRows.map(({ key }) => key.row) : undefined,
    fieldAxes: valueTerms && Object.fromEntries(valueTerms.map(({ sweepId, valueIds }) => [sweepId, [...valueIds]])),
    sourceNaming: { names: sourceNames, exceptions: exceptionMention },
  };
  function setModelPath(slot: ProjectWindowSlot, document: string, path: string[]) {
    setWindowState((current) => ({
      ...current, modelPaths: { ...current.modelPaths, [slot]: { document, path } },
    }));
  }
  /** A slot's inputs: the selection's, or those of the step it is pinned to, which it shows without editing. */
  function slotDetailInputs(slot: ProjectWindowSlot) {
    const subject = held[slot];
    if (!subject.step && loaded) {
      return {
        ...detailInputs, focus: null, selection: null, namedRun: undefined, documentScopes: { graph: undefined, task: undefined },
        valueEditing: false, evaluateScope: undefined, stored: loaded.model, modelProblems: [],
        selectedSweepValues: new Map<string, Set<string>>(), fieldRows: undefined, fieldAxes: undefined,
      };
    }
    if (!subject.step) {
      return clockSubject ? { ...detailInputs, sceneTrial: restTrial, checkpoints: undefined } : detailInputs;
    }
    const clocked = slot === clockSlot;
    return {
      ...detailInputs, step: subject.step, stepId: subject.step.id, focus: null, selection: null, evaluation: subject.evaluation, relate: undefined, marking: undefined,
      sceneTrial: clocked ? trialClock.scene : restTrial, checkpoints: clocked ? evaluationReplay.checkpoints : undefined,
      namedRun: undefined, documentScopes: { graph: undefined, task: undefined }, valueEditing: false, evaluateScope: undefined,
      stored: subject.stored, modelProblems: [], selectedSweepValues: new Map<string, Set<string>>(), fieldRows: undefined,
      fieldAxes: undefined, trial: subject.trial,
    };
  }
  const primary = useManagedProjectDetails({ ...slotDetailInputs('primary'), activeView: windowState.mode === 'map' ? 'hidden' : primaryView ?? 'hidden',
    problemReveal: focusSlot === 'primary' && !held.primary.step ? revealed : undefined, modelPath: windowState.modelPaths?.primary,
    setModelPath: (document, path) => setModelPath('primary', document, path) });
  const secondary = useManagedProjectDetails({ ...slotDetailInputs('secondary'), activeView: windowState.mode === 'map' ? 'hidden' : secondaryView ?? 'hidden',
    problemReveal: focusSlot === 'secondary' && !held.secondary.step ? revealed : undefined, modelPath: windowState.modelPaths?.secondary,
    setModelPath: (document, path) => setModelPath('secondary', document, path) });
  const focusedDetail = windowState.mode !== 'map' && (focusView === 'model' || focusView === 'scene')
    ? (focusSlot === 'primary' ? primary : secondary)[focusView] : undefined;
  const analyze = useManagedAnalyze({
    editor, key: step?.kind === 'analyze' ? namedRun : undefined, projection: shown, wholeStep,
    chosen: documentScopes.analysis?.lit,
    focusedView: windowState.mode === 'map' || focusHeld ? null : focusView, reveal: revealed, savePosition: saveElementPosition, selection: elements,
    sections: drawerSections, steps: plan.steps, failures: stepFailures(step?.kind === 'analyze' ? step.id : undefined),
    selectStep: (source) => selection.select(selectStep(state, source)),
    mean: trialClock.mean,
  });
  const pinnedAnalyzeInputs = (slot: ProjectWindowSlot) => ({
    editor, key: held[slot].step?.kind === 'analyze' ? held[slot].namedRun : undefined, projection: shown, wholeStep: false,
    savePosition: saveElementPosition, selection: elements, sections: drawerSections, steps: plan.steps,
    failures: stepFailures(undefined), selectStep: (source: string) => selection.select(selectStep(state, source)),
  });
  const pinnedAnalyze = {
    primary: useManagedAnalyze(pinnedAnalyzeInputs('primary')),
    secondary: useManagedAnalyze(pinnedAnalyzeInputs('secondary')),
  };
  const report = useManagedReport({
    editor, snapshot, plan, names: plans.names, projection: shown, variant, stepId: step?.kind === 'report' ? step.id : undefined,
    run: step?.kind === 'report' ? namedRun : undefined,
    wholeStep, focused: windowState.mode !== 'map' && focusView === 'document',
    failures: stepFailures(step?.kind === 'report' ? step.id : undefined),
  });
  /** A window still connecting, at load or after a loss, has not settled; one that cannot reach the service has. */
  const workspaceSettled = connectionKind !== 'connecting' && !busy && !(pending && !snapshot.error) && !snapshot.saving &&
    !plans.waiting && !machinePending && analyze.settled;
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
  /**
   * Runs a gesture's edit once the edits before it have saved, with the project as then saved; the controls that made
   * it stay editable meanwhile.
   */
  async function perform(work: (current: StudioProjectSnapshot) => Promise<void>, refuse?: (reason: string) => Refusal | undefined) {
    setEdits((count) => count + 1);
    settled();
    try {
      await editor.command(work);
    } catch (cause) {
      failed(cause, refuse);
    } finally {
      setEdits((count) => count - 1);
    }
  }
  function settled() {
    setCommandError(undefined);
    setRefusal(undefined);
  }
  function failed(cause: unknown, refuse?: (reason: string) => Refusal | undefined) {
    const refused = landsOnGesture(cause) && refuse?.(shownError(cause));
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
        if (land && landsOnGesture(cause)) land(shownError(cause));
        else failed(cause, refuseStep(stepId));
      },
    };
  }

  const refuseRuns = (runIds: readonly string[]) => (reason: string): Refusal | undefined =>
    runIds.length ? { kind: 'runs', reasons: Object.fromEntries(runIds.map((id) => [id, reason])) } : undefined;
  const refuseField = (field: StepField) => {
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
    if (!ready || !shown || !runnableRows.length || machineQuote || !machineReady) return;
    const keys = runnableRows.map(({ key }) => key);
    const { revision } = shown;
    void perform(async () => {
      if (await editor.flush() !== revision) throw new Error('The project changed before the run was submitted.');
      const project = editor.getSnapshot().saved.project_id;
      if (remoteMachine) {
        setMachineQuote(await quoteSavedRun(project, { revision, run_keys: keys, target: remoteMachine }));
        return;
      }
      const intent = JSON.stringify([revision, keys, reuseCompleted]);
      const token = submission.current?.intent === intent ? submission.current.token : crypto.randomUUID();
      submission.current = { intent, token };
      const result = await runSavedTrain(project, {
        revision, run_keys: keys, token, ...(reuseCompleted ? { reuse_completed: true } : {}),
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
  const problems = shown?.window_map?.problems ?? [];
  const problemSteps = new Set(problems.flatMap((problem) =>
    problem.step && (problem.variant === null || problem.variant === variant) ? [problem.step] : []));
  const shownProblems = problemScope ? problems.filter((problem) =>
    problem.step === problemScope.step &&
    (problem.variant === null || problem.variant === problemScope.variant) &&
    (!problemScope.element || holdsMember(problemScope.element, problem))) : problems;
  const problemCount = problemEntries(problems).length;
  function openProjectProblems(opener: HTMLElement | null = null) {
    setPanel({ kind: 'problems', scope: null, opener });
  }
  function holdsMember(element: ModelAddress, problem: StudioProjectWindowMapProblem) {
    const member = placeMember(problem.place);
    return !!member && holds(element, member.address);
  }
  function addressedStep(at: { step: string | null; variant: string | null }) {
    return shown?.window_map?.steps.find(({ step_id, variant: scope }) =>
      step_id === at.step && scope === (at.variant ?? variant));
  }
  /** Show the place a reason names in its step's view, at the run the reason belongs to. */
  function reveal(at: { step: string | null; variant: string | null; row: string | null },
    { name, place, detail: reason, problem }: RunMention) {
    const destination = addressedStep(at);
    if (!destination) return;
    const { step_id: stepId, variant: scope, kind } = destination;
    const row = shown?.window_map?.rows.find(({ key }) =>
      key.step === stepId && key.row === at.row && key.variant === scope);
    const named = placeView(place);
    const view = kind === 'analyze' ? 'analysis' : kind === 'report' ? 'document'
      : viewsByKind[kind].some((item) => item === named) ? named : 'runs';
    const runId = row ? savedRunId(row.key) : null;
    const field = view === named ? placeField(place) : null;
    selection.select(runId
      ? selectRun(state, { stepId, runId })
      : selectStep(state, stepId), scope);
    setExcludedUsers({});
    setProblemReveal({ revision: saved.revision, variant: scope, runId, reveal: { step: stepId, name, place, view, reason, problem } });
    setWindowState((current) => showProjectWindowView(field && view === 'runs' ? { ...current, inspectorOpen: true } : current, view, focusSlot));
    if (field) setFieldFocus({ key: crypto.randomUUID(), step: stepId, target: field.target });
    else requestSlotFocus((count) => count + 1);
    setPanel(null);
  }
  function revealProblem(problem: StudioProjectWindowMapProblem) {
    reveal(problem, { name: studioProblemName(problem), place: problem.place, detail: problem.detail, problem });
  }
  function revealRunMention(run: PlanRunView, mention: RunMention) {
    if (mention.problem) return revealProblem(mention.problem);
    const [variantId, stepId, rowId] = JSON.parse(run.id) as [string, string, string];
    reveal({ variant: variantId, step: stepId, row: rowId }, mention);
  }
  const graphRepair = revealed?.problem && removeGraphVariation(saved.project, revealed.problem);
  function removeRefusedVariation() {
    if (!graphRepair || !revealed?.problem || !applies) return;
    const { step: stepId, variant: variantId } = revealed.problem;
    void perform(async (current) => {
      if (!await runRemoval.confirm((row) => row.key.step === stepId && (!variantId || row.key.variant === variantId))) return;
      editor.editFiles(studioName('removeVariation'), [graphRepair], { reviewedRevision: current.revision });
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
    const producerIds = new Set(shown?.window_map?.edges
      .filter(({ to_key }) => savedRunId(to_key) === targetId)
      .map(({ from_key }) => savedRunId(from_key)));
    const producers = shown?.rows.filter((candidate) => producerIds.has(savedRunId(candidate.key)) &&
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
      const match = shown?.window_map?.evaluations?.find(({ key, request_id }) =>
        savedRunId(key) === job.run_key && request_id === job.request_id);
      if (!match) return false;
      const view = match.result?.job_id !== job.job_id ? 'runs' : match.checkpoint_series ? 'results' : 'scene';
      showRun(match.key, undefined, view);
      return true;
    }
    const match = shown?.rows.find(({ key, attempts }) => savedRunId(key) === job.run_key &&
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
  const failureCause = commandError ?? (projectFailure && !isUnanswered(projectFailure) ? projectFailure : undefined);
  const failure = failureCause === undefined ? null : shownError(failureCause);
  const retryPlan = () => {
    void plans.refetch();
  };
  const planError = (name: 'runs' | 'pipelineMap' | 'results'): StudioRegionState | undefined => plans.error
    ? { kind: 'failed', issue: studioRegionIssue(name, plans.error), retry: retryPlan }
    : undefined;
  const planUnavailable: StudioRegionState | undefined = planError('runs') ??
    (!shown ? { kind: 'loading' } : undefined);
  const mapUnavailable = planError('pipelineMap');
  const renderMap = (body: ReactNode) => mapUnavailable
    ? <StudioRegion state={mapUnavailable} />
    : body;
  function retry() {
    setCommandError(undefined);
    if (snapshot.error && snapshot.pending) void editor.flush().catch((error) => commandFailed(error));
    actions.refresh();
    void plans.refetch();
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
    plan, projection: shown,
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

  function evaluateModel(source: PipelineMapStepSource, projection: StudioSavedTrainRuns, stopped?: StoppedEvaluationSubject) {
    void perform(async () => {
      const reviewed = stopped
        ? await fetchSavedTrainRuns(saved.project_id, saved.revision, editor.signal, stopped.jobId)
        : projection;
      const id = await createEvaluation(editor, source, plan, reviewed, stopped);
      commandScope.selectCreatedStep(id);
    }, refuseStep(source.stepId));
  }

  function openStepMenu(kinds: StepInsertion, after?: string) {
    const revision = saved.revision;
    insertMenu.open(Object.fromEntries(Object.entries(kinds).map(([kind, create]) =>
      [kind, () => {
        if (latestRevision.current !== revision) return;
        setAdding({ kind: kind as PlanStepView['kind'], after, revision });
        create();
      }])));
  }

  const shownRow = shownResult.row;
  const stoppedCoordinate = shownRow && stoppedPlan.data?.rows.find(({ key }) => savedRunId(key) === savedRunId(shownRow.key))
    ?.attempts.find(({ job_id }) => job_id === checkpointAttempt)?.stopped_checkpoint;
  const stoppedSubject = shownRow && checkpointAttempt && stoppedCoordinate
    ? { runId: savedRunId(shownRow.key), jobId: checkpointAttempt, at: stoppedCoordinate } : undefined;
  // A gesture on it is held until the shown revision's plan is read, and authors from that plan.
  const stoppedNewStep = stoppedSubject && shown && shownRow && bindStudioCommand('newStep', () => {
    openStepMenu({
      evaluate: () => evaluateModel(
        { stepId: shownRow.key.step, selection: { kind: 'rows', stepId: shownRow.key.step, runIds: [stoppedSubject.runId] } },
        shown, stoppedSubject),
    }, shownRow.key.step);
  }, { enabled: applies && variantApplies });

  const evaluateInputs = authored?.kind === 'evaluate' ? authored : undefined;
  /** A selection of the subject, whose detail is drawn on its row of the subject field. */
  const subjectSelection = (id: string) => !!evaluateInputs && (id === 'subject' || id.startsWith('subject.'));
  /** The steps this step reads, each a link, and the inputs no field of the step draws. */
  const listedInputs = step?.inputs.filter(({ id, sourceStepId }) => !subjectSelection(id) &&
    (sourceStepId || !evaluateInputs || !['task', 'objective'].includes(id))) ?? [];
  /** An input's checkpoint and selection; where a rule field draws the selection, only the ranks it has bound. */
  function inputDetail({ id, subset }: PlanStepView['inputs'][number], ruled = false) {
    return <>
      {id === 'subject' && subjectCheckpoint !== undefined &&
        <div className="px-control-inline text-shell-muted">{checkpointLabel(subjectCheckpoint)}</div>}
      {subset && (!ruled || subset.ranked) && <div className="px-control-inline text-shell-muted">
        {!ruled && [...subset.terms, studioCountOf(subset.count, subset.total)].join(' · ')}
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

  const refusedAt = (field: StepField) =>
    refusal?.kind === 'field' && refusal.step === step?.id && refusal.field === field ? refusal.reason : undefined;

  /** The problems the plan reports on selection `index` of the step's input `role`, each on the field it names. */
  const inputProblems = (role: string, index: number) => step ? selectionProblems(shown?.window_map?.problems ?? [], step.id,
    variant ?? null, role, index, Array.isArray(authored?.inputs?.[role])) : undefined;
  /** Selection `index` of input `role` as its rule field edits it: its terms, unless a named subset holds them. */
  function selectionRuns(role: string, index: number, reference: StepReference, onCommit: (terms: SubsetTerms) => void,
    onToRows?: () => void) {
    const refused = refusedAt(`subset.${index}`);
    const problem = refused !== undefined ? { reason: refused } : inputProblems(role, index)?.rule;
    const source = plan.steps.find(({ id }) => id === reference.step);
    const where = reference.where ?? {};
    if (!source || !shown || !variant || 'subset' in where) return undefined;
    const inputId = Array.isArray(authored?.inputs?.[role]) ? `${role}.${index}` : role;
    const subset = step?.inputs.find(({ id }) => id === inputId)?.subset;
    return <SelectionRule key={JSON.stringify([step?.id, role, index, source.id])}
      scope={{ project: saved.project_id, revision: shown.revision, variant }} source={source} terms={where}
      planned={subset} picking={elements} disabled={!applies}
      problem={problem} onToRows={onToRows}
      pick={stepSourceRows(shown, source.id, variant)} onCommit={onCommit} />;
  }
  const trainStart = authoring && authored?.kind === 'train' && step && readRefusal?.step !== step.id
    && modelDocumentId(step.documentIds);
  const startRole = stepInputs?.train?.checkpoint;
  const subjectRole = stepInputs?.evaluate?.subject;
  const startSelections = stepSelections(authored?.inputs?.checkpoint);
  const startCheckpoints = startSelections && selectionCheckpoints(startSelections);
  /** An input's drafts belong to the runs they were made on, so its editor is remade for another selection. */
  const inputEditorKey = JSON.stringify([variant, selection.pipeline]);
  const inputEditor = step && (trainStart && startRole ? <InputEditor key={inputEditorKey} label={studioName('start')} role={startRole}
    binding={authored?.inputs?.checkpoint} self={step.id} steps={plan.steps} project={saved.project} documents={namedDocuments} picking={elements}
    absent={documentChoice(namedDocuments, 'model', trainStart)} refusal={refusedAt('input')}
    onReveal={(source) => selection.select(selectStep(state, source))}
    selection={(index, reference) => ({
      problems: inputProblems('checkpoint', index),
      runs: wholeStep && selectionRuns('checkpoint', index, reference,
        (terms) => onCurrentPlan('startTrain')(withSubsetTerms(reference, terms), `subset.${index}`)),
    })}
    checkpoint={startCheckpoints && startSelections && {
      groups: startCheckpoints, disabled: !applies, refusal: refusedAt('checkpoint'),
      range: (selections) => checkpointRange('checkpoint', selections),
      onSave: (rule) => onCurrentPlan('startTrain')({ ...startSelections[0], checkpoint: rule }, 'checkpoint'),
    }}
    onChange={applies ? onCurrentPlan('changeTrainStart') : undefined} />
    : evaluateInputs && subjectRole ? <InputEditor key={inputEditorKey} label={studioName('subject')} role={subjectRole}
      binding={evaluateInputs.inputs?.subject} self={step.id} steps={plan.steps} project={saved.project} documents={namedDocuments} picking={elements}
      refusal={refusedAt('input')} onReveal={(source) => selection.select(selectStep(state, source))}
      selection={(index, reference) => {
        const inputId = Array.isArray(evaluateInputs.inputs?.subject) ? `subject.${index}` : 'subject';
        const input = step.inputs.find(({ id }) => id === inputId);
        const attempts = freezeAttempts(inputId);
        return {
          problems: inputProblems('subject', index),
          runs: <>
            {input && inputDetail(input, !!evaluationEditor)}
            {evaluationEditor && selectionRuns('subject', index, reference,
              (terms) => {
                const where = authoredSubset(terms);
                const path = evaluationEditor.subjects[index].where;
                const edit = onCurrentPlan('editEvaluation');
                if (where) edit(`subset.${index}`, studioName('runs'), {
                  op: reference.where !== undefined ? 'replace' : 'add', path, value: where,
                });
                else if (reference.where !== undefined) edit(`subset.${index}`, studioName('runs'), { op: 'remove', path });
              },
              attempts ? () => onCurrentPlan('freezeEvaluationSubset')(inputId, reference.step, attempts, index) : undefined)}
          </>,
        };
      }}
      checkpoint={checkpointEditors && {
        groups: checkpointEditors, disabled: !applies, readOnly: Boolean(stepCheckpointRule && subject?.checkpoint_axis),
        refusal: refusedAt('checkpoint'), range: (selections) => checkpointRange('subject', selections),
        onSave: onCurrentPlan('editEvaluationCheckpoint'),
        onToSweep: canSweepCheckpoint ? () => onCurrentPlan('applyCheckpointSweepEdit')({ kind: 'to_sweep' }) : undefined,
      }}
      onChange={applies && wholeStep ? onCurrentPlan('changeEvaluateSubject') : undefined} />
      : undefined);
  const evaluatedSubject = evaluateInputs && evaluateSubject(evaluateInputs);
  const subjectDocuments = evaluatedSubject?.kind === 'document' ? evaluatedSubject : undefined;

  function editEvaluateSubject(name: string, patches: ProjectValueEdit[], reads: readonly string[]) {
    if (!step || !applies) return;
    const files = [{ address: { kind: 'step' as const, id: step.id }, patches }];
    const dependencies = reads.map((id) => ({ kind: saved.project.steps?.[id] ? 'step' as const : 'document' as const, id }));
    void perform(async (current) => {
      if (!await runRemoval.confirm(filesRemoval(current.project_id, current.revision, files, editor.signal))) return;
      editor.editFiles(name, files, { reviewedRevision: current.revision, dependencies });
      await editor.flush();
    }, refuseField('input'));
  }

  /** The subject `gesture` asks for: Train steps' checkpoints, or an initial Model document with its Task and Objective. */
  function changeEvaluateSubject(gesture: InputGesture) {
    if (!evaluateInputs) return;
    const name = studioName('subject');
    const current = evaluateSubject(evaluateInputs);
    if (gesture.kind === 'document') {
      const reads = gesture.role === 'loaded_model' ? 'loaded_model' : 'initial_model';
      const next: EvaluateSubject = current?.kind === 'document' ? { ...current, reads, document: gesture.document }
        : { kind: 'document', reads, document: gesture.document, ...trainDocuments(current?.kind === 'steps' ? current.selections[0]?.step : undefined) };
      return editEvaluateSubject(name, evaluateSubjectEdits(evaluateInputs, next),
        [next.document, next.task, next.objective].filter((id): id is string => id !== undefined));
    }
    if (gesture.kind !== 'select') return;
    const selections = stepSelections(gesture.binding) ?? [];
    if (current?.kind === 'document') {
      const [first] = selections;
      return first && editEvaluateSubject(name, evaluateSubjectEdits(evaluateInputs, {
        kind: 'step', step: first.step, reads: first.checkpoint == null ? 'model' : 'checkpoint',
      }), [first.step]);
    }
    editEvaluateSubject(name, [
      bindingEdit(evaluateInputs.inputs, 'subject', gesture.binding), ...selections.length > 1 ? subjectAxisEdits(evaluateInputs) : [],
    ], [...new Set(selections.map(({ step }) => step))]);
  }

  /** The Task and Objective a model-producing step reads, which an initial Model subject made from its selection declares. */
  function trainDocuments(id: string | undefined) {
    const documents = plan.steps.find((source) => source.id === id)?.documentIds;
    return {
      task: trialDocumentId(documents, 'task') ?? documentsOfRole(saved.project, 'task')[0],
      objective: trialDocumentId(documents, 'objective'),
    };
  }

  function chooseSubjectDocument(kind: 'task' | 'objective', document: string) {
    const current = evaluateInputs && evaluateSubject(evaluateInputs);
    if (current?.kind !== 'document') return;
    const next = { ...current, [kind]: document };
    editEvaluateSubject(studioName(kind), evaluateSubjectEdits(evaluateInputs!, next),
      [next.document, next.task, next.objective].filter((id): id is string => id !== undefined));
  }

  /**
   * Starts the Train step from `source`'s checkpoint as that step's continuation, or from its Model when `source` is
   * null. The service writes the step so that its runs keep their definition.
   */
  function startTrain(source: StepReference | LoadedModelReference | null, field: StepField) {
    if (!step || !applies) return;
    const stepId = step.id;
    void perform(async (current) => {
      const own = current.project.steps?.[stepId];
      if (!own) return;
      const { files } = await prepareTrainStart(current.project_id, stepId, { revision: current.revision, source }, editor.signal);
      if (!files.length || !await runRemoval.confirm(filesRemoval(current.project_id, current.revision, files, editor.signal))) return;
      const based = 'step' in own.base ? [own.base.step] : [];
      editor.editFiles(studioName('start'), files, {
        reviewedRevision: current.revision,
        dependencies: [
          ...[...new Set([...based, ...source && 'step' in source ? [source.step] : []])].map((id) => ({ kind: 'step' as const, id })),
          ...source && 'kind' in source ? [{ kind: 'document' as const, id: source.source.document }] : [],
        ],
      });
      await editor.flush();
    }, refuseField(field));
  }

  function changeTrainStart(gesture: InputGesture) {
    if (gesture.kind === 'select') startTrain(stepSelections(gesture.binding)?.[0] ?? null, 'input');
    else if (gesture.kind === 'absent') startTrain(null, 'input');
    else if (gesture.role === 'loaded_model') startTrain({ kind: 'loaded_model', source: { document: gesture.document } }, 'input');
  }

  function editEvaluation(field: Exclude<StepField, 'input' | 'checkpoint'>, name: string, patch: ProjectValueEdit) {
    if (!step || !evaluationEditor || !applies) return;
    const dependencies = [...new Set(evaluationEditor.subjects.map(({ source }) => source.id))]
      .map((id) => ({ kind: 'step' as const, id }));
    const files = [{ address: { kind: 'step' as const, id: step.id }, patches: [patch] }];
    void perform(async (current) => {
      if (!await runRemoval.confirm(filesRemoval(current.project_id, current.revision, files, editor.signal))) return;
      editor.editFiles(name, files, { reviewedRevision: current.revision, dependencies });
      await editor.flush();
    }, refuseField(field));
  }

  /** Saves a change to the selected Synthesize step's method or settings, confirming the runs it removes. */
  function editSynthesis(name: string, patches: ProjectValueEdit[]) {
    if (!step || authored?.kind !== 'synthesize' || !applies) return;
    const files = [{ address: { kind: 'step' as const, id: step.id }, patches }];
    void perform(async (current) => {
      if (!await runRemoval.confirm(filesRemoval(current.project_id, current.revision, files, editor.signal))) return;
      editor.editFiles(name, files, { reviewedRevision: current.revision });
      await editor.flush();
    }, refuseField('method'));
  }

  /**
   * The runs an exception applies to: its rule, and its rows by axis and values (`Seed 2`). Rows are read through their
   * coordinates; an axis whose every value is included says nothing and is left out.
   */
  function exceptionTerms(target: PlanStepView, where: SweepExceptionTerms | null | undefined) {
    const chosen = new Map<string, Set<string>>();
    const add = (axis: string, value: string) => chosen.set(axis, (chosen.get(axis) ?? new Set()).add(value));
    for (const row of where?.rows ?? []) {
      const coordinates = shown?.window_map?.rows.find(({ key }) => key.step === target.id && key.row === row)?.coordinates;
      for (const [axis, { axis_value }] of Object.entries(coordinates ?? {})) add(axis, axis_value);
    }
    return [...where?.rule ? [where.rule] : [], ...[...chosen].filter(([axis, values]) => {
      const all = target.sweeps.find(({ id }) => id === axis)?.values;
      return !all || all.some(({ id }) => !values.has(id));
    }).map(([axis, values]) => {
      const sweep = plannedAxis(target.id, axis);
      const names = [...values].map((value) => sweep.values.find(({ id }) => id === value)?.valueLabel ?? axisValueText(axis, value));
      return `${sweep.label} ${names.join(', ')}`;
    })];
  }

  /** The name of the run a Report figure is read from, when its step has more than one run. */
  function stepExceptions(stepId: string, scope: string | null = variant) {
    return authoredMap?.steps.find(({ step_id, variant: id }) => step_id === stepId && id === scope)?.exceptions ?? [];
  }

  /** The runs an exception picks, as its row in Exceptions and a field's source read them (`Seed 2 · 2 runs`). */
  function exceptionRuns(target: PlanStepView, { count, rule }: StudioProjectWindowMapException) {
    return [
      ...exceptionTerms(target, rule.where),
      count === 1 ? studioFormat('oneRun') : studioFormat('runCount', { count }),
    ].join(' · ');
  }

  function exceptionMention(source: StudioPlanReviewSource) {
    const target = plan.steps.find(({ id }) => id === source.step);
    const scope = source.variant ?? variant;
    const exception = target && stepExceptions(target.id, scope).find(({ id }) => id === source.key);
    if (!target || !exception) return undefined;
    return { runs: exceptionRuns(target, exception), reveal: () => {
      selection.select(selectStep(state, target.id), scope);
      setExceptionReveal({ revision: saved.revision, step: target.id, id: exception.id });
      setWindowState((current) => showProjectWindowView({ ...current, inspectorOpen: true }, 'runs', focusSlot));
    } };
  }

  /** A sweep value as the value it sets, never its id. */
  function axisValueText(axis: string, valueId: string) {
    const coordinate = shown?.window_map?.rows.map(({ coordinates }) => coordinates[axis])
      .find((item) => item?.axis_value === valueId);
    return coordinate ? valueText(coordinate.value) : valueId;
  }

  function removeSweepException(id: string) {
    if (!step || !applies || saved.project.sweep || !authored?.sweep?.exceptions?.[id]) return;
    void perform(async (current) => {
      editor.editFiles(studioName('removeException'), [{
        address: { kind: 'step', id: step.id },
        patches: [{ op: 'remove', path: authoredPointer(['sweep', 'exceptions', id]) }],
      }], { reviewedRevision: current.revision });
      await editor.flush();
    }, (reason) => ({ kind: 'exception', step: step.id, exception: id, reason }));
  }

  /** Writes `rule` as the rule of the step's exception `id`, keeping the rows it names. */
  function editExceptionRule(id: string, rule: string) {
    const exception = authored?.sweep?.exceptions?.[id];
    if (!step || !applies || saved.project.sweep || !exception) return;
    const { rule: _written, ...rest } = exception.where ?? {};
    const files = [{ address: { kind: 'step' as const, id: step.id }, patches: [{
      op: exception.where ? 'replace' as const : 'add' as const, path: authoredPointer(['sweep', 'exceptions', id, 'where']),
      value: rule ? { ...rest, rule } : rest,
    }] }];
    void perform(async (current) => {
      if (!await runRemoval.confirm(filesRemoval(current.project_id, current.revision, files, editor.signal))) return;
      editor.editFiles(studioName('runs'), files, { reviewedRevision: current.revision });
      await editor.flush();
    }, (reason) => ({ kind: 'exception', step: step.id, exception: id, reason }));
  }

  function freezeEvaluationSubset(inputId: string, source: string, attempts: string[], index: number) {
    if (!step || !variant || !applies) return;
    void perform(async (current) => {
      const reviewed = await prepareSubsetToRows(current.project_id, {
        revision: current.revision, variant, step: step.id, input: inputId, attempts,
      }, editor.signal);
      if (!await runRemoval.confirm(filesRemoval(saved.project_id, reviewed.revision, reviewed.files, editor.signal))) return;
      editor.editFiles(studioName('toRows'), reviewed.files, {
        reviewedRevision: reviewed.revision, dependencies: [{ kind: 'step', id: source }],
      });
      await editor.flush();
    }, refuseField(`subset.${index}`));
  }

  function editEvaluationCheckpoint(rule: EditableCheckpointRule, selections: readonly number[]) {
    if (!step || !subjectReferences || !checkpointEditors || !applies) return;
    if (checkpointScope && !listedSubject) {
      void applyCheckpointSweepEdit({ kind: 'set', scope: { ...checkpointScope,
        ...('rows' in checkpointScope && { rows: [...checkpointScope.rows ?? []] }),
      }, rule });
      return;
    }
    const name = 'Set Evaluate checkpoints';
    const stepId = step.id;
    const sources = [...new Set(selections.map((index) => subjectReferences[index].step))];
    const listed = listedSubject;
    void perform(async (current) => {
      const dependencies = sources.map((id) => ({ kind: 'step' as const, id }));
      let address: { kind: 'step' | 'variant'; id: string } = { kind: 'step', id: stepId };
      let patches: ProjectValueEdit[] = listed
        ? selections.map((index) => ({ op: 'replace', path: `/inputs/subject/${index}/checkpoint`, value: rule }))
        : [{ op: 'replace', path: '/inputs/subject/checkpoint', value: rule }];
      if (checkpointVariant && !listed) {
        const row = current.project.sweep?.rows?.[checkpointVariant];
        if (!row) return;
        const key = `steps.${stepId}.inputs.subject.checkpoint`;
        address = { kind: 'variant', id: checkpointVariant };
        dependencies.push({ kind: 'step', id: stepId });
        patches = [row.patches
          ? { op: key in row.patches ? 'replace' : 'add', path: authoredPointer(['patches', key]), value: { value: rule } }
          : { op: 'add', path: '/patches', value: { [key]: { value: rule } } }];
      }
      const files = [{ address, patches }];
      if (!await runRemoval.confirm(filesRemoval(current.project_id, current.revision, files, editor.signal))) return;
      editor.editFiles(name, files, { reviewedRevision: current.revision, dependencies });
      await editor.flush();
    }, refuseField('checkpoint'));
  }

  async function applyCheckpointSweepEdit(edit: StudioCheckpointRuleRequest['edit']) {
    if (!step || !subject || !namedRun || !applies) return;
    const from = document.activeElement;
    await perform(async (current) => {
      const reviewed = await sendWithStore(askStore, () => editedElement(from, from), () => editor.flush(),
        (store_choice) => prepareCheckpointRuleEdit(current.project_id, {
          revision: current.revision, run: namedRun, edit, store_choice,
        }, editor.signal), (answer) => answer.store_choices);
      if (!reviewed) return;
      if (!await runRemoval.confirm(filesRemoval(saved.project_id, reviewed.revision, reviewed.files, editor.signal))) return;
      editor.editFiles(edit.kind === 'to_sweep' ? 'Sweep Evaluate checkpoints' : 'Set Evaluate checkpoints',
        reviewed.files, { reviewedRevision: reviewed.revision,
          dependencies: [{ kind: 'step', id: subject.source_step_id }] });
      await editor.flush();
    }, refuseField('checkpoint'));
  }

  function replay(update: number, slot: ProjectWindowSlot) {
    evaluationReplay.chooseCheckpoint(update);
    setWindowState((current) => showProjectWindowView(current, 'scene', slot));
    requestSlotFocus((count) => count + 1);
  }

  /** The inspector's gestures as last rendered: one made while the plan is read runs on the render of the current plan. */
  const latestGestures = useRef({ changeEvaluateSubject, chooseSubjectDocument, editEvaluation, editEvaluationCheckpoint, editSynthesis,
    applyCheckpointSweepEdit, freezeEvaluationSubset, removeSweepException, editExceptionRule, replay, startTrain, changeTrainStart });
  latestGestures.current = { changeEvaluateSubject, chooseSubjectDocument, editEvaluation, editEvaluationCheckpoint, editSynthesis,
    applyCheckpointSweepEdit, freezeEvaluationSubset, removeSweepException, editExceptionRule, replay, startTrain, changeTrainStart };
  type InspectorGestures = typeof latestGestures.current;
  /** `name`'s gesture, made now and run once the shown revision's plan is drawn, against that plan. */
  function onCurrentPlan<K extends keyof InspectorGestures>(name: K) {
    return (...args: Parameters<InspectorGestures[K]>) => plans.whenCurrent(() => {
      void (latestGestures.current[name] as (...gesture: Parameters<InspectorGestures[K]>) => unknown)(...args);
    });
  }

  function chooseReportFigure(revision: string, figure: ReportFigureSource) {
    setReportChoices(undefined);
    if (!primaryVariant || !ready || revision !== saved.revision) return;
    void perform(async () => {
      const id = await createReport(editor, revision, figure,
        nameIn(variantNames(saved.variant_names, figure.key.variant).elements[figure.key.step], figure.name));
      commandScope.selectCreatedStep(id, 'document');
    }, refuseStep(figure.key.step));
  }

  function stepInsertion(source?: PipelineMapStepSource): { kinds: StepInsertion; enabled: boolean } | undefined {
    if (!variantApplies) return undefined;
    /** The plan the insertion is read from: the last one while the current revision's loads, which is the current
     * one by the time a gesture made meanwhile runs. */
    const known = shown;
    if (!source) {
      const chosen = state.selection?.kind === 'steps'
        ? state.selection.stepIds.length === 1 ? { stepId: state.selection.stepIds[0], selection: null } : undefined
        : state.selection && { stepId: state.selection.stepId, selection: state.selection };
      const fromChosen = chosen && stepInsertion(chosen);
      if (fromChosen) return fromChosen;
      const revision = saved.revision;
      return { kinds: {
        train: () => void perform(async () => {
          const id = await createTrain(editor, revision);
          commandScope.selectCreatedStep(id, 'model');
        }),
        synthesize: () => void perform(async () => {
          const id = await createSynthesize(editor, revision, await queries.ensureQueryData(authoringCatalogQuery));
          commandScope.selectCreatedStep(id, 'model');
        }),
      }, enabled: applies };
    }
    const sourceStep = plan.steps.find(({ id }) => id === source.stepId);
    if (sourceStep?.kind === 'evaluate' && known) {
      const rows = source.selection ? selectedRuns(sourceStep, source.selection) : sourceStep.runs;
      const input = analyzeInput(sourceStep, rows, known);
      if (!input) return undefined;
      return { kinds: { analyze: () => void perform(async () => {
        const id = await createAnalyze(editor, known.revision, input);
        commandScope.selectCreatedStep(id, 'analysis');
      }, refuseStep(source.stepId)) }, enabled: applies };
    }
    if (sourceStep?.kind === 'analyze' && known) {
      const rows = new Set((source.selection ? selectedRuns(sourceStep, source.selection) : sourceStep.runs).map(({ id }) => id));
      const figures = (known.window_map?.analysis_elements ?? []).filter(isReportFigureSource)
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
    if (sourceStep?.kind === 'synthesize') {
      const runs = source.selection ? selectedRuns(sourceStep, source.selection) : sourceStep.runs;
      return known && managedTrialDomain(known, runs)
        ? { kinds: { evaluate: () => evaluateModel(source, known) }, enabled: applies && variantApplies } : undefined;
    }
    if (!onNewStep || sourceStep?.kind !== 'train') return undefined;
    const runs = source.selection ? selectedRuns(sourceStep, source.selection) : sourceStep.runs;
    return {
      kinds: {
        train: () => void perform(async () => { await onNewStep(source, commandScope); }, refuseStep(source.stepId)),
        ...(known && managedTrialDomain(known, runs) && { evaluate: () => evaluateModel(source, known) }),
      },
      enabled: applies && variantApplies,
    };
  }

  function newStepCommand(source?: PipelineMapStepSource, subject?: string) {
    const insertion = stepInsertion(source);
    return insertion && bindStudioCommand('newStep', () => openStepMenu(insertion.kinds, source?.stepId),
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
      <SweepFieldItems editor={editor} stepId={sweepStep} onChoose={(anchor) => {
        const current = latestState.current;
        if (current.selection?.kind !== 'values' || current.selection.stepId !== sweepStep) {
          selection.select(selectStep(current, sweepStep));
        }
        setSweepStep(undefined);
        setWindowState((window) => showProjectWindowView({ ...window, inspectorOpen: true }, 'runs', focusSlot));
        setFieldFocus({ key: crypto.randomUUID(), step: sweepStep, target: anchor, kind: 'sweep' });
      }} />
      <TrialSweepItems project={saved.project_id} revision={saved.revision}
        run={shown?.window_map?.rows.find(({ key }) => key.step === sweepStep && key.variant === variant)?.key}
        onChoose={(anchor, view) => {
          selection.select(selectStep(latestState.current, sweepStep));
          setSweepStep(undefined);
          setWindowState((window) => showProjectWindowView(window, view, focusSlot));
          setFieldFocus({ key: crypto.randomUUID(), step: sweepStep, target: anchor, kind: 'sweep' });
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
      const label = name;
      if (label === labelOf(saved.project.steps, from)) return true;
      const surface = renamingStep?.surface ?? mapSurface;
      void perform(async () => {
        await editor.memberCommand(studioAccessibleName('renameStep', labelOf(saved.project.steps, from)),
          { kind: 'rename', family: 'steps', member: from, label }, editor.getSnapshot().saved);
        setStepFocus({ stepId: from, surface });
      }, refuseStep(from));
      return true;
    },
    focus: stepFocus,
    focused: () => setStepFocus(null),
  } : undefined;
  const changeStepKey = documentEditable ? (from: string, key: string) => {
    void perform(async () => {
      await editor.memberCommand(studioAccessibleName('changeKey', labelOf(saved.project.steps, from)),
        { kind: 'change_key', family: 'steps', member: from, key }, editor.getSnapshot().saved);
      selection.renameStep(from, key);
    }, (reason) => ({ kind: 'key', step: from, reason }));
    return true;
  } : undefined;
  const duplicateStep = documentEditable ? (from: string) => {
    void perform(async () => {
      const { member } = await editor.memberCommand(studioAccessibleName('duplicateStep', labelOf(saved.project.steps, from)),
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
    const refused = new Set(shown?.window_map?.regroup_refused);
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
    if (!wire || !applies) return;
    const files = [{ address: { kind: 'step' as const, id: target }, patches: wire.edits }];
    void perform(async (current) => {
      if (!await runRemoval.confirm(filesRemoval(current.project_id, current.revision, files, editor.signal))) return;
      editor.editFiles(studioAccessibleName('add', humanizeLabel(wire.role)), files, {
        reviewedRevision: current.revision, dependencies: wire.reads.map((id) => ({ kind: 'step' as const, id })),
      });
      await editor.flush();
    }, refuseStep(target));
  }
  const mapPlan = useMemo(() => adding ? { ...shownPlan, steps: [...shownPlan.steps, {
    id: addingStepId, label: studioStep(adding.kind).name, kind: adding.kind, sweeps: [], runs: [], statusCounts: {},
    inputs: adding.after ? [{ id: 'source', label: 'source', sourceStepId: adding.after }] : [], adding: true,
  }] } : shownPlan, [adding, shownPlan]);
  const mapProps = {
    connect: applies ? {
      accepts: (source: string, target: string) => !!connection(saved.project.steps, source, target),
      connect: connectSteps,
    } : undefined,
    plan: mapPlan, state, problemSteps, rename: renameStep, onEnterStep: enterStep, stepMenu: stepCommands,
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
    enabled: applies && !machineQuote && (machinePending || Boolean(machineReady)) && (planPending || runnableRows.length > 0),
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
  /** A slot's pin: unpin when pinned; else, on a view a pin keeps, pin to the selected step and its run. */
  function slotPinning(slot: ProjectWindowSlot): SlotPinning | undefined {
    const pin = pins[slot];
    const view = slot === 'primary' ? primaryView : secondaryView;
    if (pin) {
      return {
        command: bindStudioCommand('pin', () => setWindowState((current) => pinProjectWindowSlot(current, slot, undefined)),
          { name: 'unpin', instance: slot }),
        pinned: { label: stepLabel(pin.stepId), views: held[slot].step ? held[slot].views : undefined },
      };
    }
    if (!step || !view || !pinnableViews.has(view)) return undefined;
    const target = { stepId: step.id, ...pinRun && { runId: pinRun } };
    return { command: bindStudioCommand('pin', () => setWindowState((current) => pinProjectWindowSlot(current, slot, target)),
      { instance: slot }) };
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
      subject: snapshot.undoName ?? undefined,
    }),
    bindStudioCommand('redo', () => { void moveHistory('redo'); }, {
      enabled: !!snapshot.redoName && !outside,
      subject: snapshot.redoName ?? undefined,
    }),
    bindStudioCommand('clearFocus', () => selection.select({ ...state, focus: null }), { enabled: !!state.focus }),
    activity, problemCommand, failure && !isStudioDefect(failureCause) && retryCommand,
  ].filter((command): command is StudioCommand => !!command);

  const projectSettings = <ProjectSettingRows settings={configuration.settings ?? {}} defaultDays={saved.default_keep_superseded_days}
    steps={Object.keys(saved.project.steps ?? {}).map((id) => ({ id, label: labelOf(saved.project.steps, id) }))}
    change={(name, setting, value) => editor.edit(declarationAddress, name, [{ op: 'add', path: `/settings/${setting}`, value }])} />;

  return (
    <RunRemovalContext.Provider value={runRemoval.confirm}><AskStoreContext.Provider value={askStore}>
    <FigureClockContext.Provider value={{ clock: trialClock.figure, declare: declareFigureRuns }}>
      <ProjectWindow
        projectSettings={projectSettings}
        state={windowState} onStateChange={setWindowState} projectName={configuration.project}
        saveStatus={pending || snapshot.error ? studioName('edited') : undefined}
        views={views}
        pins={{ primary: slotPinning('primary'), secondary: slotPinning('secondary') }}
        commands={commands} mapCommands={canvasViewportCommands(() => mapViewport.current, !!shown && plan.steps.length > 0)}
        focusRequest={focusRequest}
        onBack={selection.back} onForward={selection.forward}
        map={renderMap(<StudioPipelineMap {...mapProps}
          activeStepId={selection.stepId} viewportRef={mapViewport} fitReady={!!shown} sweepMenu={sweepMenu} pick={mapPick} choosing={choosing}
          presentation={presentation} onPresentationChange={changePresentation} />)}
        mapDrawer={renderMap(<StudioPipelineMapDrawer {...mapProps} />)}
        compactMap={renderMap(<StudioPipelineMap {...mapProps} mode="line" />)}
        variant={(variants.length > 1 || missingVariant) && <Select aria-label={studioName('variant')}
          value={variant} disabled={!shown} onValueChange={selection.setVariant}
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
              const checked = StudioProjectRenameSchema.safeParse(rename);
              if (!checked.success && checked.error.issues.some(({ path }) => path[0] === 'name')) {
                setRefusal({ kind: 'projectName', reason: studioReason('projectNameCharacters', studioName('project')) });
                return;
              }
              void perform(async () => {
                await editor.rename(rename.name, rename.revision);
                setRename(undefined);
              }, (reason) => ({ kind: 'projectName', reason }));
            }}>
              <label className="grid gap-1">
                {studioName('name')}
                <TextField data-dialog-autofocus value={rename.name} required
                  invalid={!!projectNameProblem}
                  aria-describedby={projectNameProblem ? projectNameProblemId : undefined}
                  onValueChange={(name) => {
                    setRename({ ...rename, name });
                    setRefusal((current) => current?.kind === 'projectName' ? undefined : current);
                  }} />
              </label>
              {rename.revision !== saved.revision || outside
                ? <div role="alert">{studioReason('nameChangedDuringEdit', studioName('project'))}</div>
                : projectNameProblem && <div id={projectNameProblemId} role="alert">{projectNameProblem}</div>}
              <Button type="submit" className="justify-self-end" disabled={!!outside || rename.revision !== saved.revision}>
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
              {figureName(plans.names, plan, figure, false)}
            </Button>)}
          </Dialog>}
          {insertMenu.menu}
          {validity.popover}
        </>}
        clock={trialClock.clock}
        shelf={trialClock.shelf}
        failure={failure && <div role="alert" className="flex min-w-0 items-center gap-1">
          <span className="min-w-0 break-words">{failure}</span>
          {!isStudioDefect(failureCause) &&
            <Button {...commandAttributes(retryCommand)} onClick={retryCommand.invoke}>{studioName('retry')}</Button>}
        </div>}
        problems={<StudioProblems command={problemCommand} count={problemCount} open={!!problemPanel}
          opener={problemPanel?.opener ?? null} onClose={() => setPanel(null)}
          onOpen={(opener) => {
            if (problemPanel && !problemPanel.scope) {
              setPanel(null);
              opener.focus();
            } else {
              openProjectProblems(opener);
            }
          }}>
          <ProblemList entries={problemEntries(shownProblems)} refusal={refusal?.kind === 'problem' ? refusal : undefined} canReveal={(problem) => !!addressedStep(problem)}
            onChoose={revealProblem} showVariant={variants.length > 1 && !problemPanel?.scope} stepName={stepLabel}
            documentName={(problem, id) => problemDocumentName(plans.names, namedDocuments, problem, variant, id)} />
        </StudioProblems>}
        activity={<HeaderActivity placement="above" open={panel?.kind === 'activity'} command={activity}
          onToggle={activity.invoke}
          onClose={() => setPanel(null)} onOpenRun={openActivityRun} projectNames={names}
          resultState={activityResultState} runName={activityRunName} />}
        renderView={(view, slot) => {
          const details = slot === 'primary' ? primary : secondary;
          if (held[slot].step) {
            if (view === 'analysis' || view === 'figures') return pinnedAnalyze[slot][view];
            if (view === 'model' || view === 'scene') return { ...details[view], header: details[view].path };
            return { body: null };
          }
          const withProblem = (body: ReactNode) => revealed?.view === view && slot === focusSlot && !fieldReveal
            ? <ProblemDestination reveal={revealed} action={graphRepair && <Button
              disabled={!applies} onClick={removeRefusedVariation}>
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
            if (!namedRun) return { body: <StudioRegion state={{ kind: 'loading' }} /> };
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
            const runs: TrialRuns = {
              selected: !wholeTrialStep,
              values: (sweep) => new Set(step && state.selection && state.selection.kind !== 'steps'
                ? selectedRuns(step, state.selection).map((row) => row.values[sweep]).filter(Boolean) : []),
              choose: authoring?.onChooseSweepValue,
            };
            const objectiveDocument = value.objective_document ?? step?.id ?? '';
            const termElement = named(objectiveDocument);
            const term = termElement && value.objective?.terms?.some(({ id }) => id === termElement.id)
              ? { term: termElement.id, field: termElement.field } : undefined;
            // An Evaluate of a checkpoint is scored by the Objective of the Train that made it, which it shows unedited.
            const train = value.objective_source?.kind === 'train' ? plan.steps.find(({ id }) => id === value.objective_source!.step) : undefined;
            const scoring = train ? undefined : commit;
            const termInspector = term && <TrialObjectiveInspector view={value} commit={scoring} selected={term} runs={runs}
              shown={trialClock.trial} />;
            if (view === 'task') {
              const document = value.task_document ?? step?.id ?? '';
              const selected = term ? undefined : named(document)?.id;
              const picker = pickDocument('task', value.task_document ?? undefined, documentScopes.task);
              const unfed = unfedTasks(value);
              const variable = selectedVariable(selected);
              const entries = { of: sets.entryTask, selected: step && sets.selectedEntry(step) };
              return {
                padded: true,
                picker: picker && { ...picker, documents: picker.documents.map((entry) => ({ ...entry, problem: unfed.get(entry.key) })) },
                body: withProblem('variables' in value.task
                  ? <FocusedFields className="contents" request={fieldFocus?.step === step?.id && fieldFocus?.target?.startsWith('task:') ? fieldFocus : undefined}
                    onFocused={() => setFieldFocus(undefined)}>
                    <TrialTaskDocument view={value} commit={commit} runs={runs} entries={entries} selected={selected} select={choose(document)}
                      setName={(key) => nameIn(variantNames(plans.names, value.run.variant).documents, key)}
                      toSweep={step && ((axis) => sets.toSweep(step.id, axis))}
                      weigh={step && ((axis) => sets.fieldWeight(step.id, axis))}
                      drawn={drawnVariables(trialClock.trial)} />
                  </FocusedFields>
                  : <TrialTaskForm view={value} commit={commit} />),
                drawer: <TrialTaskOutline sections={drawerSections} view={value} selected={selected} select={choose(document)} />,
                inspector: termInspector || (variable ? <TrialVariableInspector view={value} commit={commit} runs={runs} entries={entries} id={variable.id} />
                  : selected && 'variables' in value.task && inGrid(selected)
                    ? <TrialGridInspector view={value} commit={commit} runs={runs} selected={selected} shown={trialClock.trial} />
                    : selected && <TrialTaskInspector view={value} commit={commit} selected={selected} shown={trialClock.trial} />),
              };
            }
            const picker = train ? undefined : pickDocument('objective', value.objective_document ?? undefined, documentScopes.objective);
            const scores = scoredEvaluation && recordedTerms
              ? { shown: trialClock.trial, run: recordedTerms, trials: scoredEvaluation.trials.length } : undefined;
            return {
              picker: picker && step?.kind === 'evaluate'
                ? { ...picker, choose: trialEditable ? (next) => chooseSubjectDocument('objective', next) : undefined } : picker,
              header: train && <Button variant="bare" size="content" className="underline-offset-2 hover:underline"
                onClick={() => selection.select(selectStep(state, train.id))}>{train.label}</Button>,
              body: withProblem(<FocusedFields className="contents" request={fieldFocus?.step === step?.id && fieldFocus?.target?.startsWith('objective:') ? fieldFocus : undefined}
                onFocused={() => setFieldFocus(undefined)}>
                <TrialObjectiveTable view={value} commit={scoring} selected={term} select={choose(objectiveDocument)} runs={runs} scores={scores} />
              </FocusedFields>),
              drawer: <TrialObjectiveOutline sections={drawerSections} view={value} commit={scoring} selected={term?.term} select={choose(objectiveDocument)} />,
              inspector: termInspector,
            };
          }
          if (view === 'training') {
            const scope = documentScopes.training;
            return {
              padded: true,
              picker: pickDocument('training', trainingDocument, scope),
              body: authoring ? withProblem(<StudioTrainAuthoringForm {...authoring} run={namedRun}
                runPending={!shown && !plans.error} trainingDocument chosen={scope ? [...scope.lit] : undefined}
                fieldRows={scope ? undefined : detailInputs.fieldRows} fieldAxes={scope ? undefined : detailInputs.fieldAxes} />)
                : null,
            };
          }
          if (view === 'model' || view === 'scene') {
            const scope = slot === focusSlot && view === focusView ? activeDocumentScope : undefined;
            const own = stored?.id;
            const picker = pickDocument('model', view === 'model' && loaded ? loaded.document : details[view].documentId, scope);
            const shows = picker && view === 'model' ? {
              ...picker,
              documents: picker.documents.map((entry) => ({ ...entry, unavailable: !entry.loaded && entry.key !== own && !picker.choose })),
              choose: (next: string) => {
                const document = saved.project.documents?.[next];
                if (fittedModel(document) !== undefined) return setViewedLoaded({ step: step?.id, document: next });
                setViewedLoaded(undefined);
                if (next !== own) picker.choose?.(next);
              },
            } : picker;
            const first = selectedElements[0];
            // A term the Scene's Relate wrote is selected there, and shown in its Inspector with its Measure focused.
            const term = view === 'scene' && trial.view && first && !isModelAddress(first) && first.document === objectiveDocument &&
              trial.view.objective?.terms?.some(({ id }) => id === first.element.id) ? first.element : undefined;
            const inspector = term && trial.view ? <FocusedFields className="contents"
              request={fieldFocus?.step === step?.id && fieldFocus?.target?.startsWith('objective:') ? fieldFocus : undefined}
              onFocused={() => setFieldFocus(undefined)}>
              <TrialObjectiveInspector view={trial.view} commit={relate && trial.commit} selected={{ term: term.id, field: term.field }}
                shown={trialClock.trial} />
            </FocusedFields>
              : view === 'model' && loaded
                ? <><LoadedModelFacts project={saved.project} documents={namedDocuments} document={loaded.document} />{details[view].inspector}</>
                : details[view].inspector;
            return { ...details[view], inspector, picker: shows && { ...shows, refusal: shows.refusal ?? details[view].refusal },
              header: details[view].path, body: withProblem(details[view].body) };
          }
          if (!step) return { body: null };
          if (view === 'document') return { body: withProblem(report.body),
            picker: authored?.kind === 'report' ? pickDocument('report', documentBase) : undefined };
          if (view === 'runs') {
            const runSelection = { step: shownPlan.steps.find(({ id }) => id === step.id) ?? step, state,
              onSelectionChange: mapProps.onSelectionChange };
            const pivot = pivotCommand(slot);
            const pivotActive = pivotSlots[slot] === step.id;
            const run = state.focus?.stepId === step.id ? step.runs.find(({ id }) => id === state.focus?.runId) : undefined;
            const exceptions = stepExceptions(step.id).map((exception) => {
              const where = exception.rule.where;
              const editable = !where?.rows && shown && variant;
              return { id: exception.id, text: [
                ...exceptionPatchTexts(exception.rule.patches ?? {}, trainLabels, trainClasses),
                ...editable ? [] : [exceptionRuns(step, exception)],
              ].join(' · '), rule: editable && <SelectionRule scope={{ project: saved.project_id, revision: shown.revision, variant }}
                source={step} terms={{ rule: where?.rule ?? '' }} ranked={false} picking={elements}
                planned={{ count: exception.count, total: step.runs.length }}
                problem={exceptionRuleProblem(shown.window_map?.problems ?? [], step.id, variant, exception.id)}
                disabled={!applies || !!saved.project.sweep}
                onCommit={({ rule }) => onCurrentPlan('editExceptionRule')(exception.id, rule ?? '')} /> };
            });
            const revealedException = exceptionReveal?.revision === saved.revision && exceptionReveal.step === step.id
              ? exceptionReveal.id : undefined;
            const ownFieldProps = authoring && { ...authoring, run: namedRun, runPending: !shown && !plans.error,
              fieldRows: detailInputs.fieldRows, fieldAxes: detailInputs.fieldAxes };
            const ownFields = ownFieldProps && <StudioTrainAuthoringForm {...ownFieldProps}
              omitted={evaluationEditor?.authored.trialSet.kind === 'sample' ? [trialCount] : undefined} />;
            return {
              header: <StudioRunControls commands={runCommands}
                machine={studioRunMachineControl({ catalog: machineCatalog.data, value: requestedMachine,
                  disabled: machineCatalog.isPending || !connected,
                  onChange: (value) => setMachines((current) => ({ ...current, [step.id]: value })) })}
                pivotCommand={pivot} pivotActive={pivotActive} reason={run && studioRunReason(run)}
                combination={sets.combination(step.id)} />,
              inspector: <StudioStepInspector plan={plan} step={step} state={state} onReveal={revealRunMention} stepMenu={stepCommands}
                stepKey={<KeyField value={step.id} name={step.label} editing={keyingStep === step.id}
                  onEditingChange={(editing) => setKeyingStep(editing ? step.id : undefined)}
                  onCommit={(key) => changeStepKey?.(step.id, key)}
                  unavailable={(key) => keyUnavailable(key, (key) => key in (saved.project.steps ?? {}))}
                  refusal={refusal?.kind === 'key' && refusal.step === step.id ? refusal.reason : undefined} />}
                onChooseStep={(stepId) => selection.select(selectStep(state, stepId))}
                revealed={fieldReveal?.view === 'runs'
                  ? { name: fieldReveal.name, place: fieldReveal.place, detail: fieldReveal.reason, problem: fieldReveal.problem } : undefined}
                upstreamDetail={checkpointLineage()}
                detail={selectedEvaluation?.checkpoint_series
                  ? <StudioEvaluationAttemptDetails evaluation={selectedEvaluation} stepName={stepLabel}
                    onShowStep={(stepId) => selection.select(selectStep(state, stepId))} /> : undefined}
                fields={(authoring || inputEditor || evaluationEditor || authored?.kind === 'synthesize') &&
                  <FocusedFields request={slot === focusSlot && fieldFocus?.step === step.id ? fieldFocus : undefined}
                    onFocused={() => setFieldFocus(undefined)}>
                    {authored?.kind === 'train' && ownFields}
                    {authored?.kind === 'synthesize' && <SynthesisForm key={step.id} step={authored} refusal={refusedAt('method')}
                      onEdit={applies && wholeStep ? onCurrentPlan('editSynthesis') : undefined} />}
                    {inputEditor}
                    {subjectDocuments && <SubjectDocuments subject={subjectDocuments} project={saved.project} documents={namedDocuments}
                      onChoose={applies && wholeStep ? onCurrentPlan('chooseSubjectDocument') : undefined} />}
                    {evaluationEditor && trial.view && <TrialsEditor key={step.id} initial={evaluationEditor.authored.trialSet}
                      domain={evaluationEditor.domain} variables={trial.view.variables} disabled={!applies} refusal={refusedAt('trials')}
                      count={ownFieldProps && <StudioStepField {...ownFieldProps} anchor={trialCount} label={studioName('count')} />}
                      onSave={(rule) => onCurrentPlan('editEvaluation')('trials', studioName('trials'),
                        { op: 'replace', path: '/base/inline/params/trial_set', value: rule })} />}
                    {authored?.kind === 'evaluate' && ownFields}
                  </FocusedFields>}
                attachments={(exceptions.length > 0 || listedInputs.length > 0) && <>
                  {exceptions.length > 0 && <SweepExceptions exceptions={exceptions} revealed={revealedException}
                    refusal={refusal?.kind === 'exception' && refusal.step === step.id
                      ? { id: refusal.exception, reason: refusal.reason } : undefined}
                    onRemove={applies && !saved.project.sweep ? onCurrentPlan('removeSweepException') : undefined} />}
                  {listedInputs.length > 0 && <ul aria-label={studioName('inputs')}>{listedInputs.map((input) => {
                    const source = input.sourceStepId && plan.steps.find((item) => item.id === input.sourceStepId);
                    return <li key={input.id}>
                      <InspectorRow onClick={input.sourceStepId ? () => selection.select(selectStep(state, input.sourceStepId!)) : undefined}>
                        <span className="text-shell-muted">{input.label}</span>
                        {source && <>{' · '}{source.label}</>}
                      </InspectorRow>
                      {inputDetail(input)}
                    </li>;
                  })}</ul>}
                </>} />,
              body: withProblem(<StudioRuns plan={shownPlan} {...runSelection}
                region={planUnavailable ?? (step.kind === 'train' || step.runs.length > 0
                  ? { kind: 'result', missing: [] } : { kind: 'empty' })}
                pending={planPending} pivotActive={Boolean(pivot) && pivotActive} runMenu={runMenu} axisMenu={axisMenu}
                choosing={choosing} />),
            };
          }
          const resultsFailure = planError('results');
          if (view === 'results' && analyze.results) return { ...analyze.results, padded: true, body: withProblem(analyze.results.body) };
          if (view === 'results') return {
            padded: true,
            body: <SavedResults shown={shown} current={plans.current} failure={resultsFailure} result={shownResult}
              project={saved.project_id}
              stopped={stoppedSubject && stoppedNewStep && <section aria-labelledby={stoppedCheckpointId}
                className="flex items-center gap-2">
                <h3 id={stoppedCheckpointId} className="font-semibold">{checkpointLabel(stoppedSubject.at)}</h3>
                <Button {...commandAttributes(stoppedNewStep)}
                  disabled={!stoppedNewStep.enabled} onClick={stoppedNewStep.invoke}>{studioName('newStep')}</Button>
              </section>}
              onReplay={(update) => onCurrentPlan('replay')(update, slot)}
              stepLabel={stepLabel} onShowStep={(id) => selection.select(selectStep(state, id))} />,
          };
          return { body: null };
        }}
      />
      {runRemoval.dialog}
      {choiceMenu.menu}
    </FigureClockContext.Provider></AskStoreContext.Provider></RunRemovalContext.Provider>
  );
}

/** The step's fields in the inspector, putting the keyboard on the requested one once it shows. */
function FocusedFields({ request, onFocused, children, className = 'grid content-start gap-3' }: {
  request: FieldFocusRequest | undefined; onFocused(): void; children: ReactNode; className?: string;
}) {
  const root = useRef<HTMLDivElement>(null);
  useFieldFocus(root, request, onFocused);
  const { key, target, kind } = request ?? {};
  const intent = useMemo(() => key && target && kind ? { key, target, kind } : undefined, [key, target, kind]);
  return <FieldIntent.Provider value={intent}>
    <div ref={root} className={className}>{children}</div>
  </FieldIntent.Provider>;
}

function ProblemDestination({ reveal, action, children }: { reveal: Reveal; action?: ReactNode; children: ReactNode }) {
  const reason = useRef<HTMLParagraphElement>(null);
  useEffect(() => {
    reason.current?.focus();
  }, [reveal]);
  return <div className="flex h-full min-h-0 flex-col">
    <div className="flex items-center justify-between gap-2 border-b border-shell-divider px-3 py-2 text-sm">
      <p ref={reason} tabIndex={-1}>
        <span className="font-medium">{reveal.name}</span>{' · '}{reveal.reason}
      </p>
      {action}
    </div>
    <ScrollFade className="flex-1">{children}</ScrollFade>
  </div>;
}

/**
 * The name of document `id` that `problem` is placed in, as the problem's variant labels it, or the key it is missing
 * by when the project lacks it.
 */
function problemDocumentName(
  names: VariantNames, documents: ShownDocuments, problem: StudioProjectWindowMapProblem, variant: string | null, id: string,
): TextPart {
  const scope = problem.variant ?? variant;
  const label = scope === null ? documents[id]?.label : variantNames(names, scope).documents[id];
  if (label !== undefined) return label;
  if (problem.missing_subject !== id) throw new Error(`The project holds no document ${id}.`);
  return { missing: id };
}

/** The document a problem's place is in, or else its step: what the Problems panel groups it under. */
function problemHolder(problem: StudioProjectWindowMapProblem): { kind: 'document' | 'step'; id: string } | null {
  const place = problem.place;
  if (place && 'document' in place && place.document) return { kind: 'document', id: place.document };
  return problem.step ? { kind: 'step', id: problem.step } : null;
}

/** Whether a problem is about its document as a whole, which the group heading naming that document already says. */
function namedByHolder(problem: StudioProjectWindowMapProblem, holder: string | null) {
  const place = problem.place;
  return !!place && 'document' in place && place.document === holder && studioPlaceIsDocument(place) && !problem.document_path;
}

function ProblemList({ entries: listed, refusal, onChoose, canReveal, showVariant, stepName, documentName }: {
  entries: readonly ProblemEntry[];
  /** The problem whose repair was refused, and why. */
  refusal?: { problem: string; reason: string };
  onChoose(problem: StudioProjectWindowMapProblem): void;
  canReveal(problem: StudioProjectWindowMapProblem): boolean;
  showVariant: boolean;
  /** A step's label. */
  stepName(id: string): string;
  /** The name of document `id`, which `problem`'s place is in. */
  documentName(problem: StudioProjectWindowMapProblem, id: string): TextPart;
}) {
  const groups = new Map<string | null, { holder: ReturnType<typeof problemHolder>; entries: ProblemEntry[] }>();
  for (const entry of listed) {
    const holder = problemHolder(entry.problem);
    const key = holder && `${holder.kind}:${holder.id}`;
    if (!groups.has(key)) groups.set(key, { holder, entries: [] });
    groups.get(key)!.entries.push(entry);
  }
  return <div className="space-y-2">{[...groups].map(([key, { holder, entries }]) => {
    const first = entries.find(({ problem }) => canReveal(problem))?.problem;
    const named = holder && (holder.kind === 'step' ? stepName(holder.id) : documentName(entries[0].problem, holder.id));
    const name = named && <Shown text={[named]} />;
    return <section key={key ?? ''} data-holder={key ?? undefined}>
      {name && <h2 className="font-medium">{first
        ? <Button variant="ghost" size="content" className="rounded-control px-control-inline text-left font-medium"
          onClick={() => onChoose(first)}>{name}</Button>
        : name}</h2>}
      <ul>{entries.map(({ problem, runs }, index) => <li key={index} className="break-words">
        <ProblemRow problem={problem} refusal={refusal?.problem === problemKey(problem) ? refusal.reason : undefined}
          context={[problem.step && !(holder?.kind === 'step' && holder.id === problem.step) ? stepName(problem.step) : null,
            showVariant ? problem.variant : null,
            runs > 1 ? studioFormat('runCount', { count: runs }) : null].flatMap((part) => part ? [part] : [])}
          placeShown={(problem.place !== null || problem.step !== null) && !namedByHolder(problem, holder?.kind === 'document' ? holder.id : null)}
          onReveal={canReveal(problem) ? () => onChoose(problem) : undefined} />
      </li>)}</ul>
    </section>;
  })}</div>;
}
