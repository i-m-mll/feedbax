import type { ExceptionMentions } from '@/components/values/adapters/fieldView';
import { useQuery } from '@tanstack/react-query';
import { useMemo, useState, useSyncExternalStore, type ReactNode } from 'react';
import { fetchSavedEvaluationReplay } from '@/api/jobs';
import { studioRegionIssue } from '@/components/activity/StudioRegion';
import { studioName, studioReason } from '@/components/activity/studioVocabulary';
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
  type EvaluateAttachmentReadRequest, type GraphInterior, type ModelDocument, type StudioSavedEvaluationRow,
  type StudioSavedTrainKey, type WorkspaceReplayTrial, type StudioProjectWindowMapProblem, type ModelAddress,
} from '@/generated/studioContracts';
import { modelDocumentId, directTrainStep } from './authoredEdits';
import { placeMember, type Reveal } from './projectWindowProblems';
import { ProblemRow } from '@/features/projectWindow/StudioProblems';
import type { ProjectEditController } from './projectEditController';
import { useStoredModelView } from './model/ModelView';
import type { DrawerSections } from '@/features/projectWindow/DrawerForms';
import type { ConfirmRunRemoval } from './runRemoval';
import { modelSchema } from './model/modelGraph';
import { elementKey, isModelAddress, type ElementSelectionStore, type ModelLocation } from '@/features/projectWindow/elementSelection';
import { useStore } from 'zustand';
import { Select } from '@/components/ui/Select';

type DetailName = 'model' | 'task' | 'scene';
type StoredModel = { id: string; value: ModelDocument };

export interface NamedDocumentScope extends MapDocumentScope {
  readonly key: string;
  readonly document: string;
  readonly localVariant?: boolean;
}

export interface ManagedProjectDetail {
  readonly documentId?: string;
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
  readonly fieldAxes?: Record<string, string[]>;
  readonly documentScopes: Readonly<Record<'graph' | 'task', NamedDocumentScope | undefined>>;
  /** Whether the plan read is of the shown revision; a field edit waits for it, since it carries the steps that plan lights. */
  readonly planCurrent: boolean;
  readonly problemReveal?: Reveal;
  readonly elements: ElementSelectionStore;
  readonly modelPath?: { document: string; path: string[] };
  readonly setModelPath: (document: string, path: string[]) => void;
  readonly evaluateScope?: EvaluateAttachmentReadRequest['scope'];
  readonly modelSource?: string;
  readonly modelProblems?: readonly StudioProjectWindowMapProblem[];
  readonly drawerSections: DrawerSections;
  readonly openModelProblems?: (element: ModelAddress, opener: HTMLElement) => void;
  /** The problems that keep the named run from running. */
  readonly runProblems?: readonly StudioProjectWindowMapProblem[];
  readonly revealProblem?: (problem: StudioProjectWindowMapProblem) => void;
  readonly exceptions?: ExceptionMentions;
  /** The steps an edit of the selected step reaches outside a document view. */
  readonly reachedSteps?: ReadonlySet<string>;
}

const rootModelPath: string[] = [];

function planarInteriors(graph: GraphInterior, prefix: string[] = []): string[][] {
  return Object.entries(graph.components ?? {}).flatMap(([id, component]) => {
    if (component.kind !== 'composite') return [];
    const path = [...prefix, id];
    const child = component.interior;
    if (child.physical_domain === 'planar_multibody') return [path];
    return child.physical_domain == null ? planarInteriors(child as GraphInterior, path) : [];
  });
}

function DetailReason({ reason, name }: {
  reason: 'definitionUnavailable' | 'planarSceneUnavailable' | 'scenePlaybackUnavailable';
  name: 'model' | 'task' | 'scene' | 'playback';
}) {
  return <p className="p-3 text-sm" role="status">{studioReason(reason, studioName(name))}</p>;
}

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

export function useSavedEvaluationScene(projectId: string, revision: string, evaluation: StudioSavedEvaluationRow | undefined, visible: boolean) {
  const result = evaluation?.result;
  const jobId = result?.job_id;
  const coordinates = result?.checkpoint_sources?.map(({ completed_updates }) => completed_updates);
  const [chosenCheckpoint, setCheckpoint] = useState<{ jobId: string; coordinate: number }>();
  const chosen = chosenCheckpoint?.jobId === jobId ? chosenCheckpoint?.coordinate : undefined;
  const checkpoint = chosen !== undefined && coordinates?.includes(chosen) ? chosen : coordinates?.[0];
  const [chosenTrial, setTrial] = useState<{ jobId: string; trial: string }>();
  const trial = jobId && (chosenTrial?.jobId === jobId ? chosenTrial.trial : evaluation?.trials[0]);
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
  const timeScope = JSON.stringify([jobId, trial, checkpoint]);
  const [clock, setClock] = useState<{ scope: string; time: number }>();
  const [speed, setSpeed] = useState(1);
  const time = clock?.scope === timeScope ? clock.time : 0;
  const trials = jobId && trial && evaluation ? {
    choices: evaluation.trials.map((id) => ({ id, label: id })),
    selectedId: trial,
    onSelect: (value: string) => {
      setTrial({ jobId, trial: value });
      setClock(undefined);
    },
  } : undefined;
  const chooseCheckpoint = (coordinate: number) => {
    if (jobId && coordinates?.includes(coordinate)) {
      setCheckpoint({ jobId, coordinate });
      setClock(undefined);
    }
  };
  const checkpoints = jobId && coordinates && coordinates.length > 1 && checkpoint !== undefined ? {
    choices: coordinates,
    selected: checkpoint,
    onSelect: chooseCheckpoint,
  } : undefined;
  return {
    view, time, trials, checkpoints, chooseCheckpoint,
    clock: visible && view.kind === 'replay' && <SceneClockControls key={timeScope} timeline={sceneTimeline(view)} time={time}
      speed={speed} onSpeedChange={setSpeed} onTimeChange={(value) => setClock({ scope: timeScope, time: value })} />,
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
  editor, activeView, focus, step, evaluation, evaluationReplay, elements, drawerSections, runProblems = [], revealProblem,
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
  const trials = evaluating ? evaluationReplay.trials : undefined;
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
  const omission = model ? <DetailReason reason="planarSceneUnavailable" name="model" />
    : <DetailReason reason="definitionUnavailable" name="scene" />;
  return {
    documentId: model?.id,
    path: interiors.length > 1 && path && <Select aria-label={studioName('interior')} value={JSON.stringify(path)}
      items={interiors.map((candidate) => ({ value: JSON.stringify(candidate), label: candidate.join(' / ') }))}
      onValueChange={(value) => chooseInterior({ scope, path: value })} />,
    commands: viewport,
    body: source || trials ? <CanvasStrip viewport={viewport} tools={<SceneLayers overlays={layers.overlays}
      hiddenOverlays={hiddenOverlays} onToggleOverlay={(overlay) => {
        const next = new Set(hiddenOverlays);
        if (next.has(overlay)) next.delete(overlay);
        else next.add(overlay);
        setOverlays({ scope: sceneScope, value: next });
      }} />}>
      <div className="min-h-0 flex-1">
        <SceneCanvas view={view} time={time} camera={camera} onCameraChange={onCameraChange} hiddenOverlays={hiddenOverlays} fitToSlot {...selection} />
      </div>
      {!evaluating && focus && scene && <DetailReason reason="scenePlaybackUnavailable" name="playback" />}
    </CanvasStrip> : evaluating
      ? runProblems.length ? <ul role="status" className="p-3 text-sm">{runProblems.map((problem, index) => <li key={index}>
        <ProblemRow problem={problem} onReveal={revealProblem && (() => revealProblem(problem))} />
      </li>)}</ul>
        : evaluation?.reason ? <p role="status">{studioReason('observed', evaluation.reason)}</p>
        : <DetailReason reason="scenePlaybackUnavailable" name="playback" />
      : omission,
    drawer: (scene || trials) && <SceneDrawer sections={drawerSections} scene={scene ?? emptyReplayScene} trials={trials} checkpoints={checkpoints} {...selection} />,
    inspector: scene && selection.selectedId ? <SceneElementInspector scene={scene} selectedId={selection.selectedId} /> : null,
  };
}

export function useManagedProjectDetails(input: DetailInputs): Record<DetailName, ManagedProjectDetail> {
  const { editor, stepId, selection, variantId, focus, activeView } = input;
  const { saved, draft, outside, pending, saving } = useSyncExternalStore(editor.subscribe, editor.getSnapshot);
  const subjectScope = JSON.stringify([saved.project_id, stepId, variantId]);
  const scope = JSON.stringify([subjectScope, selection, focus]);
  const base = stepId ? directTrainStep(draft, stepId) : null;
  const trial = base?.trial_documents;
  const managedModel = trial && typeof trial === 'object' && !Array.isArray(trial) ? trial.model : undefined;
  const direct = DocumentReferenceSchema.safeParse(managedModel);
  const storedId = modelDocumentId(input.step?.documentIds) ?? input.modelSource ?? (direct.success ? direct.data.document : undefined);
  const storedDocument = storedId ? saved.project.documents?.[storedId] : undefined;
  const stored = useMemo(() => {
    if (!storedId || storedDocument?.schema !== modelSchema) return undefined;
    const result = ModelDocumentSchema.safeParse(storedDocument.payload);
    return result.success ? { id: storedId, value: result.data } : undefined;
  }, [storedId, storedDocument]);
  const reveal = input.problemReveal;
  const revealedMember = placeMember(reveal?.place);
  const canEdit = selection?.kind === 'steps' && selection.stepIds.length === 1 && selection.stepIds[0] === stepId && !outside;
  const graphScope = input.documentScopes.graph;
  const storedView = useStoredModelView({
    model: activeView === 'model' ? stored : undefined, editor, saved, elements: input.elements,
    active: activeView === 'model',
    selectedSweepValues: input.selectedSweepValues,
    confirmRunRemoval: input.confirmRunRemoval,
    fieldLitSteps: graphScope?.lit ?? input.reachedSteps,
    planCurrent: input.planCurrent,
    fieldRows: input.fieldRows,
    fieldAxes: input.fieldAxes,
    fieldEditable: !outside && !pending && !saving && Boolean(input.namedRun && (
      input.step?.kind === 'evaluate' ? input.evaluateScope
        : input.step?.kind === 'train' && (saved.project.sweep ? canEdit && graphScope?.localVariant
          : canEdit || selection?.kind === 'values' || selection?.kind === 'rows')
    )),
    projectionRun: input.step?.kind === 'train' ? input.namedRun : undefined,
    editable: canEdit && !pending && !saving && !!saved.presentation && input.step?.kind === 'train' &&
      !!graphScope && !graphScope.localVariant && graphScope.lit.size === graphScope.users.size,
    evaluate: input.step?.kind === 'evaluate' && input.namedRun && input.evaluateScope
      ? { run: input.namedRun, scope: input.evaluateScope } : undefined,
    problem: revealedMember && revealedMember.document === storedId
      ? { address: revealedMember.address, reason: reveal!.problem?.detail ?? reveal!.reason } : undefined,
    problems: input.modelProblems?.flatMap((item) => {
      const member = placeMember(item.place);
      return member && member.document === storedId ? [{ address: member.address, reason: item.detail }] : [];
    }),
    path: input.modelPath && input.modelPath.document === storedId ? input.modelPath.path : rootModelPath,
    setPath: (path) => { if (storedId) input.setModelPath(storedId, path); },
    sections: input.drawerSections,
    openProblems: input.openModelProblems,
    exceptions: input.exceptions,
  });
  return {
    model: stored ? storedView : { body: <DetailReason reason="definitionUnavailable" name="model" /> },
    task: { body: <DetailReason reason="definitionUnavailable" name="task" /> },
    scene: useSceneDetail(input, stored, JSON.stringify([scope, storedId]), saved.project_id, saved.revision),
  };
}
