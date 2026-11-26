import type { SourceNaming } from '@/components/values/adapters/fieldView';
import { documentPayload } from './documentPayload';
import { conditionLabels } from './evaluateAuthoring';
import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { useEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactNode } from 'react';
import { fetchSavedEvaluationReplay } from '@/api/jobs';
import type { MapDocumentScope } from '@/components/pipeline/StudioPipelineMap';
import { CanvasStrip } from '@/components/canvas/CanvasStrip';
import { canvasViewportCommands } from '@/components/canvas/StudioCanvas';
import { SceneCanvas, SceneDrawer, SceneElementInspector, SceneLayers, SceneTools, sceneContents, type SceneMarking, type SceneRelate } from '@/components/scene/SceneRegion';
import { recordedStates, trialScene, type SceneTrial, type StateValues } from '@/components/scene/trialScene';
import { emptyScene, fittedCamera, sceneElements, zoomScene, type Scene, type SceneCamera, type SceneLayer, type SceneTool, type SceneView } from '@/components/scene/sceneView';
import { useSceneGeometry } from '@/components/scene/useSceneGeometry';
import { isHttpConflict } from '@/api/request';
import { studioRegionIssue, type StudioRegionIssue } from '@/components/activity/StudioRegion';
import type { PlanStepView, RunAddress } from '@/features/pipeline/planView';
import type { PipelineSelection } from '@/features/pipeline/selection';
import { bindStudioCommand, type StudioCommand } from '@/features/projectWindow/studioCommands';
import {
  DocumentReferenceSchema, StoredModelDocumentSchema,
  type EvaluateModelRequest, type StoredModelDocument, type StudioSavedEvaluationRow,
  type StudioSavedTrainKey, type StudioProjectWindowMapProblem, type ModelAddress, type TrialRecord,
  type StudioSavedEvaluationReplayRequest, type StudioTrialAuthoring, type TrialMean, type TrialMeanOptions, type TrialMeanSelection,
} from '@/generated/studioContracts';
import { modelDocumentId, ownDefinition } from './authoredEdits';
import { placeMember, type Reveal } from './projectWindowProblems';
import type { ProjectEditController } from './projectEditController';
import { useStoredModelView } from './model/ModelView';
import type { DrawerSections } from '@/features/projectWindow/DrawerForms';
import type { ConfirmRunRemoval } from './runRemoval';
import type { AskStore } from './storeChoice';
import { modelDocumentSchemaId } from '@/generated/studioContracts';
import { elementKey, isModelAddress, selectedElement, type ElementAddress, type ElementSelectionStore } from '@/features/projectWindow/elementSelection';
import { useStore } from 'zustand';
import { usePlayhead, type PlayheadStore, type TrialShowing } from '@/components/timeline/trialClock';

type DetailName = 'model' | 'scene';
/** A step kind whose product is a model it determines from its authored Model. */
const producesModel = (kind: string | undefined) => kind === 'train' || kind === 'synthesize';

export type StoredModel = { id: string; value: StoredModelDocument };

/** A document view's steps: `lit` are those an edit is made for. */
export interface NamedDocumentScope extends MapDocumentScope {
  readonly key: string;
  readonly document: string;
  readonly localVariant?: boolean;
}

export interface ManagedProjectDetail {
  readonly documentId?: string;
  /** Why the service refuses the view's read of its document, shown on the document's picker. */
  readonly refusal?: string;
  readonly body: ReactNode;
  readonly drawer?: ReactNode;
  readonly inspector?: ReactNode;
  readonly commands?: readonly StudioCommand[];
  readonly overlay?: ReactNode;
  readonly path?: ReactNode;
}

interface DetailInputs {
  readonly editor: ProjectEditController;
  readonly stepId: string | null;
  readonly selection: PipelineSelection | null;
  readonly variantId: string | null;
  readonly focus: RunAddress | null;
  readonly activeView?: string;
  readonly step?: PlanStepView;
  readonly evaluation?: StudioSavedEvaluationRow;
  /** What the clock gives the Scene to draw. */
  readonly sceneTrial: SceneTrial;
  /** Writes an Objective term from the Scene's places, where the step's Objective can be edited. */
  readonly relate?: SceneRelate;
  /** Changes a task variable's marker from the Scene's list, where the step's Task can be edited. */
  readonly marking?: SceneMarking;
  /** The recorded checkpoints the Scene's trials can be read at, when there is a choice. */
  readonly checkpoints: ReturnType<typeof useSavedEvaluationScene>['checkpoints'];
  readonly namedRun?: StudioSavedTrainKey;
  readonly selectedSweepValues: ReadonlyMap<string, ReadonlySet<string>>;
  readonly confirmRunRemoval: ConfirmRunRemoval;
  readonly askStore: AskStore;
  readonly fieldRows?: readonly string[];
  /** Whether one value of a named axis or a set's entry is selected, which a Model edit is then stored on. */
  readonly valueEditing?: boolean;
  readonly fieldAxes?: Record<string, string[]>;
  readonly documentScopes: Readonly<Record<'graph' | 'task', NamedDocumentScope | undefined>>;
  /** Whether the plan read is of the shown revision; a field edit waits for it, since it carries the steps that plan lights. */
  readonly planCurrent: boolean;
  readonly problemReveal?: Reveal;
  readonly elements: ElementSelectionStore;
  readonly modelPath?: { document: string; path: string[] };
  readonly setModelPath: (document: string, path: string[]) => void;
  readonly evaluateScope?: EvaluateModelRequest['scope'];
  /** The stored Model the step's Model and Scene show, `useStoredModel`. */
  readonly stored: StoredModel | undefined;
  readonly modelProblems?: readonly StudioProjectWindowMapProblem[];
  readonly drawerSections: DrawerSections;
  readonly openModelProblems?: (element: ModelAddress, opener: HTMLElement) => void;
  readonly sourceNaming: SourceNaming;
  /** The step's trial, whose epochs and Task variables the Model's attachments read. */
  readonly trial?: StudioTrialAuthoring;
}

const rootModelPath: string[] = [];

const noStateValues: StateValues = {};
const shownLayers = new Set<SceneLayer>();

/**
 * The result a saved evaluation's views draw: its current one, else the one an edit put out of date, with the trials
 * that result recorded.
 */
export function drawnEvaluation(evaluation: StudioSavedEvaluationRow | undefined) {
  const result = evaluation?.result ?? evaluation?.previous;
  return result ? { result, trials: result.trials ?? evaluation!.trials, outOfDate: !evaluation!.result } : undefined;
}

/**
 * The trials a saved evaluation recorded, which the clock's picker lists, with each one's condition named by the Task's
 * `variables`; unnamed until the Task is read.
 */
export function recordedTrials(evaluation: StudioSavedEvaluationRow | undefined, variables: readonly { readonly id: string; readonly name: string }[] | undefined) {
  const drawn = drawnEvaluation(evaluation);
  if (!drawn) return { ids: [], labels: [], outOfDate: false };
  if (!variables) return { ids: drawn.trials, labels: [], outOfDate: drawn.outOfDate };
  const labels = conditionLabels(drawn.trials, variables);
  return { ids: drawn.trials, labels: drawn.trials.map((id) => labels.get(id) ?? id), outOfDate: drawn.outOfDate };
}

/**
 * The recorded trials the clock reads, as the Scene draws them: every trial read (the one shown, or all of them under
 * All and Mean), the picked one among them, and their state times; or why they are not read.
 */
export type RecordedTrialRead =
  | { readonly kind: 'read'; readonly records: readonly TrialRecord[]; readonly picked: TrialRecord; readonly stateTime: readonly number[] }
  | { readonly kind: 'loading' }
  | { readonly kind: 'empty' }
  | { readonly kind: 'failed'; readonly issue: StudioRegionIssue; readonly retry: () => void };

/**
 * What one replay reads for `showing`: the picked trial; every trial (All); every trial with each condition's Mean for
 * `mean` (Mean). Every reply carries the By and Align the trials offer; Mean's carries its summaries beside every
 * trial's record.
 */
function replayRequest(showing: TrialShowing, trials: readonly string[], mean: TrialMeanSelection):
  Pick<StudioSavedEvaluationReplayRequest, 'trial' | 'mean'> {
  switch (showing.kind) {
    case 'one': return { trial: trials[showing.index] };
    case 'all': return { trial: null };
    case 'mean': return { trial: null, mean };
  }
}

/**
 * The trials the saved evaluation recorded as `showing` shows them (the picked one; all of them with their aggregate;
 * all of them with each condition's Mean for `mean`), and their time on the window's playhead: the trial's time, or
 * under Mean the time from Align's epoch.
 */
export function useSavedEvaluationScene(projectId: string, revision: string, evaluation: StudioSavedEvaluationRow | undefined,
  showing: TrialShowing | undefined, mean: TrialMeanSelection, playhead: PlayheadStore) {
  const drawn = drawnEvaluation(evaluation);
  const result = drawn?.result;
  const jobId = result?.job_id;
  const coordinates = result?.checkpoint_sources?.map(({ completed_updates }) => completed_updates);
  const [chosenCheckpoint, setCheckpoint] = useState<{ jobId: string; coordinate: number }>();
  const chosen = chosenCheckpoint?.jobId === jobId ? chosenCheckpoint?.coordinate : undefined;
  const checkpoint = chosen !== undefined && coordinates?.includes(chosen) ? chosen : coordinates?.[0];
  const all = showing?.kind === 'all';
  const request = jobId && drawn && showing ? replayRequest(showing, drawn.trials, mean) : undefined;
  const readable = request?.trial !== undefined && (!evaluation?.checkpoint_series || checkpoint !== undefined);
  const read = (aggregate: boolean) => ({ signal }: { signal: AbortSignal }) => {
    if (!evaluation || !jobId || !request) throw new Error('Select a saved Evaluate trial.');
    return fetchSavedEvaluationReplay(projectId, { revision, key: evaluation.key, job_id: jobId, ...request, checkpoint, aggregate }, signal);
  };
  const key = ['saved-evaluation-replay', projectId, revision, evaluation?.key, jobId, request?.trial, checkpoint];
  // A changed By or Align keeps the Mean drawn, and its fields open, until the new Mean is read.
  const replay = useQuery({
    queryKey: request?.mean ? [...key, 'mean', request.mean] : key, queryFn: read(false), enabled: readable,
    placeholderData: request?.mean ? keepPreviousData : undefined,
  });
  const aggregate = useQuery({ queryKey: [...key, 'aggregate'], queryFn: read(true), enabled: readable && all });
  const records = replay.data?.drawn.kind === 'trials' ? replay.data.drawn.trials : undefined;
  const picked = records?.[showing && showing.kind !== 'one' ? showing.index : 0];
  const state: RecordedTrialRead = replay.error
    ? { kind: 'failed', issue: studioRegionIssue('scene', replay.error), retry: () => { void replay.refetch(); } }
    : replay.data && records && picked
      ? { kind: 'read', records, picked, stateTime: replay.data.state_time }
      : { kind: replay.isFetching ? 'loading' : 'empty' };
  const batch = aggregate.data?.drawn.kind === 'aggregate' ? aggregate.data.drawn : undefined;
  const averaged: TrialMean | undefined = request?.mean ? replay.data?.mean ?? undefined : undefined;
  const meanOptions: TrialMeanOptions | undefined = replay.data?.mean_options ?? undefined;
  const timeScope = JSON.stringify(['recorded', jobId, request?.mean?.align ?? null]);
  const [time, setTime] = usePlayhead(playhead, timeScope);
  const chooseCheckpoint = (coordinate: number) => {
    if (jobId && coordinates?.includes(coordinate)) setCheckpoint({ jobId, coordinate });
  };
  const checkpoints = jobId && coordinates && coordinates.length > 1 && checkpoint !== undefined ? {
    choices: coordinates,
    selected: checkpoint,
    onSelect: chooseCheckpoint,
  } : undefined;
  return {
    read: state, aggregate: { data: batch, error: aggregate.error, refetch: aggregate.refetch }, mean: averaged, meanOptions,
    time, setTime, checkpoints, chooseCheckpoint, checkpoint,
  };
}

/**
 * The Objective terms over every trial the saved evaluation recorded at `checkpoint`: each term's mean value over the
 * trials it covers and the share it covers, when `enabled` and the evaluation has a result.
 */
export function useRecordedTerms(projectId: string, revision: string, evaluation: StudioSavedEvaluationRow | undefined,
  checkpoint: number | undefined, enabled: boolean) {
  const jobId = drawnEvaluation(evaluation)?.result.job_id;
  const query = useQuery({
    queryKey: ['saved-evaluation-replay', projectId, revision, evaluation?.key, jobId, null, checkpoint, 'aggregate'],
    queryFn: ({ signal }) => fetchSavedEvaluationReplay(projectId,
      { revision, key: evaluation!.key, job_id: jobId!, trial: null, aggregate: true, checkpoint }, signal),
    enabled: enabled && !!jobId && (!evaluation?.checkpoint_series || checkpoint !== undefined),
  });
  return query.data?.drawn.kind === 'aggregate' ? query.data.drawn.terms ?? {} : undefined;
}

/** The Scene's part of the window's selection: the key of the selected element the Scene draws, a field as its element. */
function useSceneSelection(elements: ElementSelectionStore, scene: Scene | null) {
  const first = useStore(elements, (current) => current.selection[0]);
  const named = first && !isModelAddress(first) ? first.element.id : undefined;
  const key = first && elementKey(isModelAddress(first) ? { ...first, address: selectedElement(first.address) } : first);
  const drawn = scene && (sceneElements(scene).some((element) => element.key === key) || scene.groups.some((group) => group.key === key));
  return {
    selected: drawn ? key! : null,
    onSelect: (address: ElementAddress | null) => elements.setState({ selection: address ? [address] : [] }),
    selectedTerm: named,
  };
}

function useSceneDetail({ editor, activeView, sceneTrial, checkpoints, elements, drawerSections, relate, marking }: DetailInputs,
  model: StoredModel | undefined, scope: string, projectId: string, revision: string): ManagedProjectDetail {
  const visible = activeView === 'scene';
  const source = model && visible ? { document_id: model.id, revision } : null;
  const { picked, possible, frame, task, trails, failure, mean, reading } = sceneTrial;
  const recorded = useMemo(() => picked && recordedStates(picked), [picked]);
  const poses = useMemo(() => recorded ? [picked!.inits, ...recorded] : [picked?.inits ?? noStateValues], [picked, recorded]);
  const geometry = useSceneGeometry(projectId, source, poses);
  const posed = geometry?.(recorded ? 1 + Math.min(frame, recorded.length - 1) : 0);
  const rest = recorded ? geometry?.(0) : undefined;
  const sceneScope = JSON.stringify([scope, model?.id]);
  const [cameraState, setCamera] = useState<{ scope: string; value: SceneCamera }>();
  const [layerState, setLayers] = useState<{ scope: string; value: ReadonlySet<SceneLayer> }>();
  const camera = cameraState?.scope === sceneScope ? cameraState.value : fittedCamera;
  const onCameraChange = (value: SceneCamera) => setCamera({ scope: sceneScope, value });
  const hiddenLayers = layerState?.scope === sceneScope ? layerState.value : shownLayers;
  const [chosenTool, setTool] = useState<SceneTool>('select');
  const tool = relate ? chosenTool : 'select';
  const tools = relate && {
    select: bindStudioCommand('selectTool', () => setTool('select')),
    relate: bindStudioCommand('relate', () => setTool('relate')),
  };
  const read = useMemo(() => posed?.kind === 'ready' && !reading ? trialScene({
    posed: posed.geometry, rest: rest?.kind === 'ready' ? rest.geometry : undefined, task, picked, possible, frame, trails, mean,
  }) : null, [posed, rest, reading, task, picked, possible, frame, trails, mean]);
  /** The last Scene drawn of this subject, which stays drawn while an edit or a new pick is read. */
  const drawnLast = useRef<{ scope: string; scene: Scene }>();
  useEffect(() => { if (read) drawnLast.current = { scope: sceneScope, scene: read }; });
  const scene = read ?? (drawnLast.current?.scope === sceneScope ? drawnLast.current.scene : null);
  const view: SceneView = failure ?? (!posed ? { kind: 'empty' }
    : posed.kind === 'failed' ? {
      kind: 'failed', issue: studioRegionIssue('scene', posed.error),
      retry: isHttpConflict(posed.error) ? () => editor.connected() : posed.retry,
    }
      : scene ? { kind: 'drawn', scene } : { kind: 'loading' });
  const selection = useSceneSelection(elements, scene);
  const contents = sceneContents(view);
  const viewport = canvasViewportCommands(() => ({
    fitView: () => onCameraChange(fittedCamera),
    zoomIn: () => onCameraChange(zoomScene(camera, 1.2)),
    zoomOut: () => onCameraChange(zoomScene(camera, 1 / 1.2)),
  }), contents.navigable);
  return {
    documentId: model?.id,
    commands: tools ? [tools.select, tools.relate, ...viewport] : viewport,
    body: source ? <CanvasStrip viewport={viewport} tools={<>{tools && <SceneTools tool={tool} commands={tools} />}<SceneLayers layers={contents.layers}
      hidden={hiddenLayers} onToggle={(layer) => {
        const next = new Set(hiddenLayers);
        if (next.has(layer)) next.delete(layer);
        else next.add(layer);
        setLayers({ scope: sceneScope, value: next });
      }} /></>}>
      <div className="min-h-0 flex-1">
        <SceneCanvas view={view} camera={camera} onCameraChange={onCameraChange} hiddenLayers={hiddenLayers} fitToSlot tool={tool}
          onRelate={relate} {...selection} />
      </div>
    </CanvasStrip> : null,
    drawer: (scene || checkpoints) && <SceneDrawer sections={drawerSections} scene={scene ?? emptyScene} checkpoints={checkpoints}
      selected={selection.selected} onSelect={selection.onSelect} marking={marking} />,
    inspector: scene && selection.selected ? <SceneElementInspector scene={scene} selected={selection.selected} /> : null,
  };
}

/** The stored Model a step shows: its own, the Model its Evaluate subject trains, or the one its Train step names. */
export function useStoredModel(editor: ProjectEditController, step: PlanStepView | undefined, modelSource: string | undefined) {
  const { saved, draft } = useSyncExternalStore(editor.subscribe, editor.getSnapshot);
  const base = step ? ownDefinition(draft, step.id) : null;
  const trial = base?.trial_documents;
  const managedModel = trial && typeof trial === 'object' && !Array.isArray(trial) ? trial.model : undefined;
  const direct = DocumentReferenceSchema.safeParse(managedModel);
  const storedId = modelDocumentId(step?.documentIds) ?? modelSource ?? (direct.success && !direct.data.unresolved ? direct.data.document : undefined);
  const storedDocument = storedId ? saved.project.documents?.[storedId] : undefined;
  const storedPayload = storedId ? documentPayload(saved.project.documents, saved.composed, storedId) : undefined;
  return useMemo((): StoredModel | undefined => {
    if (!storedId || storedDocument?.schema !== modelDocumentSchemaId) return undefined;
    const result = StoredModelDocumentSchema.safeParse(storedPayload);
    return result.success ? { id: storedId, value: result.data } : undefined;
  }, [storedId, storedDocument, storedPayload]);
}

export function useManagedProjectDetails(input: DetailInputs): Record<DetailName, ManagedProjectDetail> {
  const { editor, stepId, selection, variantId, focus, activeView, stored } = input;
  const { saved, outside } = useSyncExternalStore(editor.subscribe, editor.getSnapshot);
  const subjectScope = JSON.stringify([saved.project_id, stepId, variantId]);
  const scope = JSON.stringify([subjectScope, selection, focus]);
  const storedId = stored?.id;
  const reveal = input.problemReveal;
  const revealedMember = placeMember(reveal?.place);
  const canEdit = selection?.kind === 'steps' && selection.stepIds.length === 1 && selection.stepIds[0] === stepId && !outside;
  const graphScope = input.documentScopes.graph;
  /** Whether the Model's topology commands apply; a gesture made while the workspace is unsettled waits for it. */
  const topologyApplies = !!saved.presentation && !outside && (input.valueEditing ||
    canEdit && producesModel(input.step?.kind) &&
    !!graphScope && !graphScope.localVariant && graphScope.lit.size === graphScope.users.size);
  const storedView = useStoredModelView({
    model: activeView === 'model' ? stored : undefined, editor, saved, elements: input.elements,
    active: activeView === 'model',
    selectedSweepValues: input.selectedSweepValues,
    confirmRunRemoval: input.confirmRunRemoval,
    askStore: input.askStore,
    fieldChosenSteps: graphScope?.lit,
    planCurrent: input.planCurrent,
    fieldRows: input.fieldRows,
    fieldAxes: input.fieldAxes,
    fieldEditable: !outside && Boolean(input.namedRun && (
      input.step?.kind === 'evaluate' ? input.evaluateScope
        : producesModel(input.step?.kind) && (saved.project.sweep ? canEdit && graphScope?.localVariant
          : canEdit || selection?.kind === 'values' || selection?.kind === 'rows')
    )),
    projectionRun: producesModel(input.step?.kind) ? input.namedRun : undefined,
    relate: input.relate && ((address) => input.relate!({ kind: 'port', address })),
    topologyApplies,
    evaluate: input.step?.kind === 'evaluate' && input.namedRun && input.evaluateScope
      ? { run: input.namedRun, scope: input.evaluateScope } : undefined,
    problem: revealedMember && revealedMember.document === storedId
      ? { address: revealedMember.address, reason: reveal!.problem?.detail ?? reveal!.reason } : undefined,
    problems: input.modelProblems?.flatMap((item) => {
      const member = placeMember(item.place);
      return member && member.document === storedId ? [{ address: member.address, reason: item.detail }] : [];
    }),
    path: input.modelPath && input.modelPath.document === storedId ? input.modelPath.path : rootModelPath,
    setPath: (path, document = storedId) => { if (document) input.setModelPath(document, path); },
    sections: input.drawerSections,
    openProblems: input.openModelProblems,
    sourceNaming: input.sourceNaming,
    trial: input.trial,
  });
  return {
    model: stored ? storedView : { body: null },
    scene: useSceneDetail(input, stored, JSON.stringify([scope, storedId]), saved.project_id, saved.revision),
  };
}
