import type { ExceptionMentions } from '@/components/values/adapters/fieldView';
import { documentPayload } from './documentPayload';
import { conditionLabels } from './evaluateAuthoring';
import { useQuery } from '@tanstack/react-query';
import { useMemo, useState, useSyncExternalStore, type ReactNode } from 'react';
import { fetchSavedEvaluationReplay } from '@/api/jobs';
import { studioRegionIssue } from '@/components/activity/StudioRegion';
import { studioName } from '@/components/activity/studioVocabulary';
import type { MapDocumentScope } from '@/components/pipeline/StudioPipelineMap';
import { CanvasStrip } from '@/components/canvas/CanvasStrip';
import { canvasViewportCommands } from '@/components/canvas/StudioCanvas';
import { SceneCanvas, SceneDrawer, SceneElementInspector, SceneLayers, sceneLayers } from '@/components/scene/SceneRegion';
import { SceneClockControls } from '@/components/scene/SceneClockControls';
import type { PlanarScenePose } from '@/components/scene/planarModel';
import { fittedCamera, sceneGeometry, sceneTimeline, zoomScene, type SceneCamera, type SceneDrawing, type SceneOverlay, type SceneView } from '@/components/scene/sceneView';
import { useProjectPlanarScene } from '@/components/scene/useProjectPlanarScene';
import type { PlanStepView, RunAddress } from '@/features/pipeline/planView';
import type { PipelineSelection } from '@/features/pipeline/selection';
import type { StudioCommand } from '@/features/projectWindow/studioCommands';
import type { ResolvedScene } from '@/features/scenario/projections';
import { primaryWorkspaceReplayTrack, workspaceReplayFrameIndex, workspaceReplaySampleAt } from '@/features/scenario/workspaceReplay';
import {
  DocumentReferenceSchema, ModelDocumentSchema,
  type EvaluateModelRequest, type ModelDocument, type StudioSavedEvaluationRow,
  type StudioSavedTrainKey, type WorkspaceReplayTrial, type StudioProjectWindowMapProblem, type ModelAddress,
} from '@/generated/studioContracts';
import { modelDocumentId, directTrainStep } from './authoredEdits';
import { placeMember, type Reveal } from './projectWindowProblems';
import { planarInteriors } from './viewContent';
import type { ProjectEditController } from './projectEditController';
import { useStoredModelView } from './model/ModelView';
import type { DrawerSections } from '@/features/projectWindow/DrawerForms';
import type { ConfirmRunRemoval } from './runRemoval';
import { modelSchema } from './model/modelGraph';
import { elementKey, isModelAddress, type ElementSelectionStore, type ModelLocation } from '@/features/projectWindow/elementSelection';
import { useStore } from 'zustand';
import { usePlayhead, type PlayheadStore } from '@/components/timeline/trialClock';
import { Select } from '@/components/ui/Select';

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
  readonly evaluationReplay: ReturnType<typeof useSavedEvaluationScene>;
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

const referencePose: PlanarScenePose = { kind: 'reference' };
const shownOverlays = new Set<SceneOverlay>();

const emptyReplayScene: ResolvedScene = {
  scenario_id: null, frame: 'world', units: 'm', entities: [], anchors: [], elements: [],
  required_selectors: [], validation: [],
};

function evaluationScene(trial: WorkspaceReplayTrial): SceneDrawing {
  const track = primaryWorkspaceReplayTrack(trial);
  if (!track) return { kind: 'replay', trial, scene: emptyReplayScene };
  const id = track.anchor_id;
  const anchorId = `${id}:position`;
  const label = studioName('effector');
  const frame = track.frame ?? emptyReplayScene.frame;
  const sceneAt = (time: number): ResolvedScene => {
    const position = workspaceReplaySampleAt(track, workspaceReplayFrameIndex(trial, time));
    return {
      ...emptyReplayScene, frame, units: track.units ?? emptyReplayScene.units,
      entities: [{
        id, label, kind: 'mechanics_object', summary: null, related_entity_ids: [],
        anchor_ids: [anchorId], element_ids: [id],
      }],
      anchors: [{
        id: anchorId, entity_id: id, local_id: 'position', label, semantic_role: 'effector',
        position, selectable: true, hoverable: true, interaction_roles: [], objective_roles: [],
        frame, selector: track.selector, metadata: {},
      }],
      elements: [{
        id, entity_id: id, local_id: 'point', archetype: 'point', anchor_ids: [anchorId], frame,
        scale_invariant: true, style: {}, metadata: {},
        geometry: { kind: 'points', points: position ? [position] : [] },
      }],
    };
  };
  return { kind: 'replay', trial, scene: sceneAt(0), sceneAt };
}

/** The trials a saved evaluation recorded, which the clock's picker lists, with each one's condition. */
export function recordedTrials(evaluation: StudioSavedEvaluationRow | undefined) {
  if (!evaluation?.result?.job_id) return { ids: [], labels: [] };
  const labels = conditionLabels(evaluation.trials);
  return { ids: evaluation.trials, labels: evaluation.trials.map((id) => labels.get(id) ?? id) };
}

/** The replay of the `recorded`-th trial the saved evaluation recorded, on the window's playhead. */
export function useSavedEvaluationScene(projectId: string, revision: string, evaluation: StudioSavedEvaluationRow | undefined,
  visible: boolean, recorded: number | undefined, playhead: PlayheadStore) {
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
    enabled: visible && !!trial && (!evaluation?.checkpoint_series || checkpoint !== undefined),
  });
  const replayDrawing = useMemo(() => replay.data && evaluationScene(replay.data), [replay.data]);
  const view: SceneView = replay.error
    ? { kind: 'failed', issue: studioRegionIssue('scene', replay.error), retry: () => { void replay.refetch(); } }
    : replayDrawing ?? { kind: trial && (!evaluation?.checkpoint_series || checkpoint !== undefined) ? 'loading' : 'empty' };
  const timeScope = JSON.stringify(['recorded', jobId, trial, checkpoint]);
  const [time, setTime] = usePlayhead(playhead, timeScope);
  const [speed, setSpeed] = useState(1);
  const chooseCheckpoint = (coordinate: number) => {
    if (jobId && coordinates?.includes(coordinate)) setCheckpoint({ jobId, coordinate });
  };
  const checkpoints = jobId && coordinates && coordinates.length > 1 && checkpoint !== undefined ? {
    choices: coordinates,
    selected: checkpoint,
    onSelect: chooseCheckpoint,
  } : undefined;
  return {
    view, time, checkpoints, chooseCheckpoint,
    transport: visible && view.kind === 'replay' && <SceneClockControls key={timeScope} timeline={sceneTimeline(view)} time={time}
      speed={speed} onSpeedChange={setSpeed} onTimeChange={setTime} />,
  };
}

/** The Model element a Scene entity shows, or an anchor that is itself a model element (a joint). */
function sceneAddress(id: string): ModelLocation | undefined {
  let parsed: unknown;
  try {
    parsed = JSON.parse(id);
  } catch (error) {
    if (error instanceof SyntaxError) return undefined;
    throw error;
  }
  if (!Array.isArray(parsed) || parsed.length !== 4) return undefined;
  const [, document, composites, element] = parsed as unknown[];
  return typeof document === 'string' && Array.isArray(composites) && composites.every((item) => typeof item === 'string') &&
    typeof element === 'string' ? { document, address: { composites, element: { kind: 'component', id: element } } } : undefined;
}

/**
 * The Scene draws the window's selection: an entity is the Model component it shows. An anchor that is not an element
 * of its own (a body's end, a muscle's path point) is selected as its entity, and the Scene keeps which anchor it was.
 * An entity with no Model element (a replay's effector) is a named element of the Scene, `document`.
 */
function useSceneSelection(elements: ElementSelectionStore, scene: ResolvedScene | null, document: string) {
  const selection = useStore(elements, (current) => current.selection);
  const [anchor, setAnchor] = useState<string>();
  const first = selection[0];
  const owner = (id: string) => sceneAddress(id) ?? sceneAddress(scene?.anchors.find((item) => item.id === id)?.entity_id ?? '');
  const anchorAddress = anchor ? owner(anchor) : undefined;
  const selectedKey = first && isModelAddress(first) ? elementKey(first) : undefined;
  const selectedId = !first || !scene ? null
    : !isModelAddress(first) ? first.document === document ? first.element.id : null
    : anchor && anchorAddress && elementKey(anchorAddress) === selectedKey ? anchor
    : [...scene.entities.map(({ id }) => id), ...scene.anchors.map(({ id }) => id)]
      .find((id) => { const address = sceneAddress(id); return address && elementKey(address) === selectedKey; }) ?? null;
  return {
    selectedId,
    onSelect: (id: string | null) => {
      const address = id === null ? undefined : owner(id);
      setAnchor(address && !sceneAddress(id!) ? id! : undefined);
      elements.setState({
        selection: address ? [address] : id === null ? [] : [{ document, element: { kind: 'named', id } }],
      });
    },
  };
}

function useSceneDetail({
  editor, activeView, step, evaluation, evaluationReplay, elements, drawerSections,
}: DetailInputs,
  model: StoredModel | undefined, scope: string, projectId: string, revision: string): ManagedProjectDetail {
  const interiors = useMemo(() => model ? planarInteriors(model.value.interior) : [], [model]);
  const [interiorChoice, chooseInterior] = useState<{ scope: string; path: string }>();
  const chosenInterior = interiorChoice?.scope === scope ? interiorChoice.path : undefined;
  const path = interiors.find((candidate) => JSON.stringify(candidate) === chosenInterior) ?? interiors[0];
  const source = model && path && activeView === 'scene'
    ? { document_id: model.id, node_path: path, revision } : null;
  const geometry = useProjectPlanarScene(projectId, source, referencePose, () => editor.connected());
  const evaluating = step?.kind === 'evaluate';
  const view = evaluating ? evaluationReplay.view : geometry;
  const time = evaluating ? evaluationReplay.time : 0;
  const checkpoints = evaluating ? evaluationReplay.checkpoints : undefined;
  const sceneScope = JSON.stringify([scope, path, evaluation?.result?.job_id]);
  const [cameraState, setCamera] = useState<{ scope: string; value: SceneCamera }>();
  const [overlayState, setOverlays] = useState<{ scope: string; value: ReadonlySet<SceneOverlay> }>();
  const camera = cameraState?.scope === sceneScope ? cameraState.value : fittedCamera;
  const onCameraChange = (value: SceneCamera) => setCamera({ scope: sceneScope, value });
  const hiddenOverlays = overlayState?.scope === sceneScope ? overlayState.value : shownOverlays;
  const scene = view.kind === 'geometry' || view.kind === 'replay' ? sceneGeometry(view, time) : null;
  const selection = useSceneSelection(elements, scene, sceneScope);
  const layers = sceneLayers(view);
  const viewport = canvasViewportCommands(() => ({
    fitView: () => onCameraChange(fittedCamera),
    zoomIn: () => onCameraChange(zoomScene(camera, 1.2)),
    zoomOut: () => onCameraChange(zoomScene(camera, 1 / 1.2)),
  }), layers.navigable);
  return {
    documentId: model?.id,
    path: interiors.length > 1 && path && <Select aria-label={studioName('interior')} value={JSON.stringify(path)}
      items={interiors.map((candidate) => ({ value: JSON.stringify(candidate), label: candidate.join(' / ') }))}
      onValueChange={(value) => chooseInterior({ scope, path: value })} />,
    commands: viewport,
    body: source || evaluating ? <CanvasStrip viewport={viewport} tools={<SceneLayers overlays={layers.overlays}
      hiddenOverlays={hiddenOverlays} onToggleOverlay={(overlay) => {
        const next = new Set(hiddenOverlays);
        if (next.has(overlay)) next.delete(overlay);
        else next.add(overlay);
        setOverlays({ scope: sceneScope, value: next });
      }} />}>
      <div className="min-h-0 flex-1">
        <SceneCanvas view={view} time={time} camera={camera} onCameraChange={onCameraChange} hiddenOverlays={hiddenOverlays} fitToSlot {...selection} />
      </div>
    </CanvasStrip> : null,
    drawer: (scene || checkpoints) && <SceneDrawer sections={drawerSections} scene={scene ?? emptyReplayScene} checkpoints={checkpoints} {...selection} />,
    inspector: scene && selection.selectedId ? <SceneElementInspector scene={scene} selectedId={selection.selectedId} /> : null,
  };
}

/** The stored Model a step shows: its own, the Model its Evaluate subject trains, or the one its Train step names. */
export function useStoredModel(editor: ProjectEditController, step: PlanStepView | undefined, modelSource: string | undefined) {
  const { saved, draft } = useSyncExternalStore(editor.subscribe, editor.getSnapshot);
  const base = step ? directTrainStep(draft, step.id) : null;
  const trial = base?.trial_documents;
  const managedModel = trial && typeof trial === 'object' && !Array.isArray(trial) ? trial.model : undefined;
  const direct = DocumentReferenceSchema.safeParse(managedModel);
  const storedId = modelDocumentId(step?.documentIds) ?? modelSource ?? (direct.success ? direct.data.document : undefined);
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
