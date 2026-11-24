import type { ExceptionMentions } from '@/components/values/adapters/fieldView';
import { documentPayload } from './documentPayload';
import { conditionLabels } from './evaluateAuthoring';
import { useQuery } from '@tanstack/react-query';
import { useMemo, useState, useSyncExternalStore, type ReactNode } from 'react';
import { fetchSavedEvaluationReplay } from '@/api/jobs';
import type { MapDocumentScope } from '@/components/pipeline/StudioPipelineMap';
import { CanvasStrip } from '@/components/canvas/CanvasStrip';
import { canvasViewportCommands } from '@/components/canvas/StudioCanvas';
import { SceneCanvas, SceneDrawer, SceneElementInspector, SceneLayers, sceneContents } from '@/components/scene/SceneRegion';
import { recordedState, trialScene, type SceneTrial, type StateValues } from '@/components/scene/trialScene';
import { emptyScene, fittedCamera, sceneElements, zoomScene, type Scene, type SceneCamera, type SceneLayer, type SceneView } from '@/components/scene/sceneView';
import { useSceneGeometry } from '@/components/scene/useSceneGeometry';
import { isHttpConflict } from '@/api/request';
import { studioRegionIssue, type StudioRegionIssue } from '@/components/activity/StudioRegion';
import type { PlanStepView, RunAddress } from '@/features/pipeline/planView';
import type { PipelineSelection } from '@/features/pipeline/selection';
import type { StudioCommand } from '@/features/projectWindow/studioCommands';
import {
  DocumentReferenceSchema, ModelDocumentSchema,
  type EvaluateModelRequest, type ModelDocument, type StudioSavedEvaluationRow,
  type StudioSavedTrainKey, type StudioProjectWindowMapProblem, type ModelAddress, type TrialRecord,
} from '@/generated/studioContracts';
import { modelDocumentId, ownDefinition } from './authoredEdits';
import { placeMember, type Reveal } from './projectWindowProblems';
import type { ProjectEditController } from './projectEditController';
import { useStoredModelView } from './model/ModelView';
import type { DrawerSections } from '@/features/projectWindow/DrawerForms';
import type { ConfirmRunRemoval } from './runRemoval';
import { modelSchema } from './model/modelGraph';
import { elementKey, isModelAddress, selectedElement, type ElementAddress, type ElementSelectionStore } from '@/features/projectWindow/elementSelection';
import { useStore } from 'zustand';
import { usePlayhead, type PlayheadStore } from '@/components/timeline/trialClock';

type DetailName = 'model' | 'scene';
export type StoredModel = { id: string; value: ModelDocument };

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
  /** The recorded checkpoints the Scene's trials can be read at, when there is a choice. */
  readonly checkpoints: ReturnType<typeof useSavedEvaluationScene>['checkpoints'];
  readonly namedRun?: StudioSavedTrainKey;
  readonly selectedSweepValues: ReadonlyMap<string, ReadonlySet<string>>;
  readonly confirmRunRemoval: ConfirmRunRemoval;
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
  readonly exceptions?: ExceptionMentions;
}

const rootModelPath: string[] = [];

const noStateValues: StateValues = {};
const shownLayers = new Set<SceneLayer>();

/** The trials a saved evaluation recorded, which the clock's picker lists, with each one's condition. */
export function recordedTrials(evaluation: StudioSavedEvaluationRow | undefined) {
  if (!evaluation?.result?.job_id) return { ids: [], labels: [] };
  const labels = conditionLabels(evaluation.trials);
  return { ids: evaluation.trials, labels: evaluation.trials.map((id) => labels.get(id) ?? id) };
}

/** A recorded trial as the Scene's clock reads it: its record and its state times, or why it is not read. */
export type RecordedTrialRead =
  | { readonly kind: 'read'; readonly record: TrialRecord; readonly stateTime: readonly number[] }
  | { readonly kind: 'loading' }
  | { readonly kind: 'empty' }
  | { readonly kind: 'failed'; readonly issue: StudioRegionIssue; readonly retry: () => void };

/** The `recorded`-th trial the saved evaluation recorded, and its time on the window's playhead. */
export function useSavedEvaluationScene(projectId: string, revision: string, evaluation: StudioSavedEvaluationRow | undefined,
  recorded: number | undefined, playhead: PlayheadStore) {
  const result = evaluation?.result;
  const jobId = result?.job_id;
  const coordinates = result?.checkpoint_sources?.map(({ completed_updates }) => completed_updates);
  const [chosenCheckpoint, setCheckpoint] = useState<{ jobId: string; coordinate: number }>();
  const chosen = chosenCheckpoint?.jobId === jobId ? chosenCheckpoint?.coordinate : undefined;
  const checkpoint = chosen !== undefined && coordinates?.includes(chosen) ? chosen : coordinates?.[0];
  const trial = jobId && recorded !== undefined ? evaluation?.trials[recorded] : undefined;
  const replay = useQuery({
    queryKey: ['saved-evaluation-replay', projectId, revision, evaluation?.key, jobId, trial, checkpoint],
    queryFn: ({ signal }) => {
      if (!evaluation || !jobId || !trial) throw new Error('Select a saved Evaluate trial.');
      return fetchSavedEvaluationReplay(projectId, { revision, key: evaluation.key, job_id: jobId, trial, checkpoint }, signal);
    },
    enabled: !!trial && (!evaluation?.checkpoint_series || checkpoint !== undefined),
  });
  const record = replay.data?.drawn.kind === 'trials' ? replay.data.drawn.trials[0] : undefined;
  const read: RecordedTrialRead = replay.error
    ? { kind: 'failed', issue: studioRegionIssue('scene', replay.error), retry: () => { void replay.refetch(); } }
    : replay.data && record
      ? { kind: 'read', record, stateTime: replay.data.state_time }
      : { kind: replay.isFetching ? 'loading' : 'empty' };
  const timeScope = JSON.stringify(['recorded', jobId, trial, checkpoint]);
  const [time, setTime] = usePlayhead(playhead, timeScope);
  const chooseCheckpoint = (coordinate: number) => {
    if (jobId && coordinates?.includes(coordinate)) setCheckpoint({ jobId, coordinate });
  };
  const checkpoints = jobId && coordinates && coordinates.length > 1 && checkpoint !== undefined ? {
    choices: coordinates,
    selected: checkpoint,
    onSelect: chooseCheckpoint,
  } : undefined;
  return { read, time, setTime, checkpoints, chooseCheckpoint };
}

/** The Scene's part of the window's selection: the key of the selected element the Scene draws, a field as its element. */
function useSceneSelection(elements: ElementSelectionStore, scene: Scene | null) {
  const first = useStore(elements, (current) => current.selection[0]);
  const key = first && elementKey(isModelAddress(first) ? { ...first, address: selectedElement(first.address) } : first);
  const drawn = scene && (sceneElements(scene).some((element) => element.key === key) || scene.groups.some((group) => group.key === key));
  return {
    selected: drawn ? key! : null,
    onSelect: (address: ElementAddress | null) => elements.setState({ selection: address ? [address] : [] }),
  };
}

function useSceneDetail({ editor, activeView, sceneTrial, checkpoints, elements, drawerSections }: DetailInputs,
  model: StoredModel | undefined, scope: string, projectId: string, revision: string): ManagedProjectDetail {
  const visible = activeView === 'scene';
  const source = model && visible ? { document_id: model.id, revision } : null;
  const { shown, possible, frame, task, failure } = sceneTrial;
  const record = shown?.kind === 'one' ? shown.record : undefined;
  const recorded = record && recordedState(record, frame);
  const posed = useSceneGeometry(projectId, source, recorded ?? record?.inits ?? noStateValues);
  const rest = useSceneGeometry(projectId, source, recorded && record.inits);
  const sceneScope = JSON.stringify([scope, model?.id]);
  const [cameraState, setCamera] = useState<{ scope: string; value: SceneCamera }>();
  const [layerState, setLayers] = useState<{ scope: string; value: ReadonlySet<SceneLayer> }>();
  const camera = cameraState?.scope === sceneScope ? cameraState.value : fittedCamera;
  const onCameraChange = (value: SceneCamera) => setCamera({ scope: sceneScope, value });
  const hiddenLayers = layerState?.scope === sceneScope ? layerState.value : shownLayers;
  const scene = useMemo(() => posed?.kind === 'ready' ? trialScene({
    posed: posed.geometry, rest: rest?.kind === 'ready' ? rest.geometry : undefined, task, shown, possible, frame,
  }) : null, [posed, rest, task, shown, possible, frame]);
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
    commands: viewport,
    body: source ? <CanvasStrip viewport={viewport} tools={<SceneLayers layers={contents.layers}
      hidden={hiddenLayers} onToggle={(layer) => {
        const next = new Set(hiddenLayers);
        if (next.has(layer)) next.delete(layer);
        else next.add(layer);
        setLayers({ scope: sceneScope, value: next });
      }} />}>
      <div className="min-h-0 flex-1">
        <SceneCanvas view={view} camera={camera} onCameraChange={onCameraChange} hiddenLayers={hiddenLayers} fitToSlot {...selection} />
      </div>
    </CanvasStrip> : null,
    drawer: (scene || checkpoints) && <SceneDrawer sections={drawerSections} scene={scene ?? emptyScene} checkpoints={checkpoints} {...selection} />,
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
    if (!storedId || storedDocument?.schema !== modelSchema) return undefined;
    const result = ModelDocumentSchema.safeParse(storedPayload);
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
    canEdit && input.step?.kind === 'train' &&
    !!graphScope && !graphScope.localVariant && graphScope.lit.size === graphScope.users.size);
  const storedView = useStoredModelView({
    model: activeView === 'model' ? stored : undefined, editor, saved, elements: input.elements,
    active: activeView === 'model',
    selectedSweepValues: input.selectedSweepValues,
    confirmRunRemoval: input.confirmRunRemoval,
    fieldChosenSteps: graphScope?.lit,
    planCurrent: input.planCurrent,
    fieldRows: input.fieldRows,
    fieldAxes: input.fieldAxes,
    fieldEditable: !outside && Boolean(input.namedRun && (
      input.step?.kind === 'evaluate' ? input.evaluateScope
        : input.step?.kind === 'train' && (saved.project.sweep ? canEdit && graphScope?.localVariant
          : canEdit || selection?.kind === 'values' || selection?.kind === 'rows')
    )),
    projectionRun: input.step?.kind === 'train' ? input.namedRun : undefined,
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
    exceptions: input.exceptions,
  });
  return {
    model: stored ? storedView : { body: null },
    scene: useSceneDetail(input, stored, JSON.stringify([scope, storedId]), saved.project_id, saved.revision),
  };
}
