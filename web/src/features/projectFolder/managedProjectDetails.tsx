import { useQuery } from '@tanstack/react-query';
import { useEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactNode } from 'react';
import { fetchComponents } from '@/api/client';
import { fetchSavedEvaluationReplay } from '@/api/jobs';
import { fetchDocumentAuthoring } from '@/api/projectFolders';
import { apiErrorMessage } from '@/api/request';
import { studioRegionIssue } from '@/components/activity/StudioRegion';
import { studioAccessibleName, studioName, studioReason, type StudioNameId, type StudioReasonId } from '@/components/activity/studioVocabulary';
import { StudioDefinitionForm, type StudioDefinitionField, type StudioFieldControl } from '@/components/pipeline/StudioDefinitionForm';
import type { MapDocumentScope } from '@/components/pipeline/StudioPipelineMap';
import { SceneCanvas, SceneDrawer, SceneElementInspector, SceneHeaderControls } from '@/components/scene/SceneRegion';
import { SceneClockControls } from '@/components/scene/SceneClockControls';
import type { PlanarScenePose } from '@/components/scene/planarModel';
import { fittedCamera, sceneGeometry, sceneTimeline, type SceneCamera, type SceneDrawing, type SceneOverlay, type SceneView } from '@/components/scene/sceneView';
import { useProjectPlanarScene } from '@/components/scene/useProjectPlanarScene';
import { humanizeLabel } from '@/components/values/ValueSpecField';
import type { PlanStepView, RunAddress } from '@/features/pipeline/planView';
import type { PipelineSelection } from '@/features/pipeline/selection';
import type { StudioCommand } from '@/features/projectWindow/studioCommands';
import { graphEdgeId } from '@/features/scenario/entities';
import type { ResolvedScene } from '@/features/scenario/projections';
import { primaryWorkspaceReplayTrack, workspaceReplayFrameIndex, workspaceReplaySampleAt } from '@/features/scenario/workspaceReplay';
import {
  DocumentReferenceSchema, GraphSpecSchema, TaskSpecSchema,
  type GraphSpec, type JsonPrimitive, type JsonValue, type ParamSchema, type StudioSavedEvaluationRow, type TaskSpec, type WireSpec,
  type PresentationPosition, type StudioDocumentAuthoring, type StudioDocumentAuthoringRequest,
  type StudioSavedTrainKey, type WorkspaceReplayTrial,
} from '@/generated/studioContracts';
import { authoredPointer, directTrainStep, type AuthoredProject } from './authoredEdits';
import type { ProblemReveal } from './projectWindowProblems';
import { StudioTrainAuthoringForm } from './StudioTrainAuthoringForm';
import { ProjectEditConflictError, type ProjectEditController } from './projectEditController';
import { useModelAuthoring } from './useModelAuthoring';
import { ManagedDocumentCanvas, useElementSelection } from './ManagedDocumentCanvas';

type DetailName = 'model' | 'task' | 'scene';
type DefinitionField = 'graph' | 'task';
type Definition = {
  readonly id?: string;
  readonly label: string;
  readonly payload: JsonValue;
  readonly shared: boolean;
  readonly authored?: StudioDocumentAuthoring;
};

export interface NamedDocumentScope extends MapDocumentScope {
  readonly key: string;
  readonly document: string;
  readonly localVariant?: boolean;
}

export interface ManagedProjectDetail {
  readonly documentId?: string;
  readonly header: ReactNode;
  readonly body: ReactNode;
  readonly drawer?: ReactNode;
  readonly inspector?: ReactNode;
  readonly commands?: readonly StudioCommand[];
  readonly overlay?: ReactNode;
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
  readonly onChooseSweepValue?: (sweepId: string, valueId: string, extend: boolean) => void;
  readonly namedRun?: StudioSavedTrainKey;
  readonly documentScopes: Readonly<Record<DefinitionField, NamedDocumentScope | undefined>>;
  readonly problemReveal?: ProblemReveal;
}

function authoredDefinition(project: AuthoredProject, stepId: string, field: DefinitionField): Definition | null {
  const inline = directTrainStep(project, stepId);
  if (!inline) return null;
  const raw = inline[field];
  if (raw == null) return null;
  const otherSteps = Object.entries(project.steps ?? {}).filter(([id]) => id !== stepId).map(([, step]) => step);
  const inherited = otherSteps.some(({ base }) => 'step' in base && base.step === stepId);
  const reference = DocumentReferenceSchema.safeParse(raw);
  if (!reference.success) return { label: studioName('inline'), payload: raw, shared: inherited };
  const { document: id, pointer } = reference.data;
  const document = project.documents?.[id];
  if (pointer || document?.schema !== (field === 'graph' ? 'feedbax.spec.graph' : 'feedbax.spec.task')) return null;
  return { id, label: id, payload: document.payload, shared: true };
}

function detailHeader(definition: Definition | null) {
  return definition ? <span className="studio-window-slot__document">{definition.label}</span> : null;
}

function DetailReason({ reason, name }: { reason: StudioReasonId; name: StudioNameId }) {
  return <p role="status" className="px-3 py-2 text-sm text-shell-muted">{studioReason(reason, studioName(name))}</p>;
}

function DetailList({ name, items, selectedId, onSelect }: ReturnType<typeof useElementSelection> & {
  name: StudioNameId;
  items: readonly { id: string; label: string }[];
}) {
  if (!items.length) return null;
  return (
    <section className="space-y-1" aria-label={studioName(name)}>
      <h3 className="px-2 text-xs font-medium text-shell-muted">{studioName(name)}</h3>
      <ul>
        {items.map(({ id, label }) => <li key={id}>
          <button type="button" aria-pressed={selectedId === id} onClick={() => onSelect(id)}
            className="w-full rounded px-2 py-1 text-left text-sm hover:bg-shell-hover aria-pressed:bg-shell-selected">
            {label}
          </button>
        </li>)}
      </ul>
    </section>
  );
}

const endpoint = (node: string, port: string) => `${node}.${port}`;

function WireDetails({ wire }: { wire: WireSpec }) {
  return <dl className="space-y-1 text-sm">
    <div><dt>{studioName('output')}</dt><dd>{endpoint(wire.source_node, wire.source_port)}</dd></div>
    <div><dt>{studioName('input')}</dt><dd>{endpoint(wire.target_node, wire.target_port)}</dd></div>
  </dl>;
}

const scalar = (value: unknown): value is JsonPrimitive => value === null ||
  typeof value === 'number' || typeof value === 'string' || typeof value === 'boolean';

function fieldControl(value: JsonPrimitive, schema?: ParamSchema): StudioFieldControl | undefined {
  if (typeof value === 'number' && (!schema || schema.type === 'float' || schema.type === 'int')) {
    return { kind: 'number', min: schema?.min ?? undefined, max: schema?.max ?? undefined,
      step: schema?.step ?? (schema?.type === 'int' ? 1 : undefined) };
  }
  if (typeof value === 'boolean' && (!schema || schema.type === 'bool')) return { kind: 'boolean' };
  if (typeof value === 'string' && schema?.type === 'enum' && schema.options?.length) {
    return { kind: 'choice', options: schema.options.map((option) => ({ value: option, label: option })) };
  }
}

function StructuredValue({ value }: { value: unknown }) {
  if (Array.isArray(value)) return <ol className="flex flex-wrap gap-2">{value.map((item, index) =>
    <li key={index}><StructuredValue value={item} /></li>)}</ol>;
  if (value && typeof value === 'object') return <dl className="pl-2">{Object.entries(value).map(([key, item]) =>
    <div key={key} className="flex flex-wrap gap-2"><dt>{key}</dt><dd><StructuredValue value={item} /></dd></div>)}</dl>;
  return <>{typeof value === 'boolean' ? studioName(value ? 'true' : 'false') : value == null ? '—' : String(value)}</>;
}

function ParameterDetails({ params, definition, selection, schemas, path, onEdit }: {
  params: Record<string, unknown>;
  definition: Definition;
  selection: PipelineSelection;
  schemas?: ParamSchema[];
  path: readonly string[];
  onEdit?: (field: string, value: JsonPrimitive) => void;
}) {
  const fields: StudioDefinitionField[] = [];
  const structured: [string, unknown][] = [];
  for (const [name, value] of Object.entries(params)) {
    if (!scalar(value)) { structured.push([name, value]); continue; }
    const schema = schemas?.find((candidate) => candidate.name === name);
    const pointer = authoredPointer([...path, name]);
    const source = definition.authored?.sources[pointer];
    const variantPaths = definition.authored?.variant_model_paths;
    const control = onEdit && (!schemas || schema) && (!definition.authored || source) &&
        (!variantPaths || variantPaths.includes(pointer))
      ? fieldControl(value, schema) : undefined;
    fields.push({ id: name, label: humanizeLabel(name), control, resolved: [{ id: name, value, source: source?.name ?? definition.label }],
      actions: control ? ['set'] : [] });
  }
  return <>
    {fields.length > 0 && <StudioDefinitionForm fields={fields} selection={selection}
      onEdit={({ fieldId, edit }) => { if (edit.kind === 'set') onEdit?.(fieldId, edit.value); }} />}
    {structured.length > 0 && <dl className="space-y-2 text-sm">{structured.map(([name, value]) =>
      <div key={name} className="rounded border border-shell-divider p-2">
        <dt className="font-medium">{humanizeLabel(name)}</dt><dd><StructuredValue value={value} /></dd>
      </div>)}</dl>}
  </>;
}

function useDefinitionEdit(input: DetailInputs, scope: string) {
  const { editor, stepId, selection, namedRun, documentScopes, activeView } = input;
  const reviewScope = JSON.stringify([scope, activeView, namedRun, ...Object.values(documentScopes).map((target) =>
    target && [target.key, [...target.lit], target.localVariant])]);
  const currentReview = useRef(reviewScope);
  currentReview.current = reviewScope;
  useEffect(() => {
    currentReview.current = reviewScope;
    return () => { currentReview.current = ''; };
  }, [reviewScope]);
  const [issue, setIssue] = useState<{ scope: string; field: DefinitionField; message: string }>();
  return {
    issue: (field: DefinitionField) => issue?.scope === scope && issue.field === field ? issue.message : null,
    async edit(field: DefinitionField, definition: Definition, label: string, path: string[], value: JsonPrimitive) {
      if (!stepId || selection?.kind !== 'steps') return;
      const id = definition.id;
      try {
        if (id) {
          const { saved, outside, pending } = editor.getSnapshot();
          const target = documentScopes[field];
          if (outside || pending || !target || target.document !== id || !namedRun || value === null ||
              definition.authored?.revision !== saved.revision) throw new ProjectEditConflictError();
          const reviewed = await fetchDocumentAuthoring(saved.project_id, {
            revision: saved.revision, run: namedRun, field,
            change: { lit_steps: [...target.lit], path: authoredPointer(path), value,
              ...(target.localVariant ? { scope: 'variant' } : {}) },
          });
          if (currentReview.current !== reviewScope || editor.getSnapshot().outside ||
              reviewed.document !== id || reviewed.revision !== saved.revision) throw new ProjectEditConflictError();
          const dependencies = target.localVariant ? saved.file_revisions.map(({ address }) => address).filter((address) =>
            (address.kind === 'document' && address.id === id) ||
            (address.kind === 'step' && target.users.has(address.id)) ||
            (address.kind === 'variant' && address.id !== namedRun.variant)) : [];
          editor.editFiles(label, reviewed.files, { reviewedRevision: reviewed.revision, dependencies });
        } else {
          const check = (project: AuthoredProject) => {
            const latest = authoredDefinition(project, stepId, field);
            if (!latest || latest.id || latest.shared) throw new ProjectEditConflictError();
          };
          check(editor.getSnapshot().draft);
          editor.edit({ kind: 'step', id: stepId }, label, [
            { op: 'replace', path: authoredPointer(['base', 'inline', field, ...path]), value },
          ], (before) => check(before));
        }
        setIssue(undefined);
      } catch (error) {
        setIssue({ scope, field, message: apiErrorMessage(error) });
      }
    },
  };
}

function planarInteriors(graph: GraphSpec, prefix: string[] = []): string[][] {
  return Object.entries(graph.subgraphs ?? {}).flatMap(([name, child]) => {
    const path = [...prefix, name];
    if ('physical_domain' in child) return child.physical_domain === 'planar_multibody' ? [path] : [];
    return planarInteriors(child, path);
  });
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
  const jobId = (result?.checkpoint_sources?.length ?? 0) <= 1 ? result?.job_id : undefined;
  const [chosenTrial, setTrial] = useState<{ jobId: string; trial: string }>();
  const trial = jobId && (chosenTrial?.jobId === jobId ? chosenTrial.trial : evaluation?.trials[0]);
  const replay = useQuery({
    queryKey: ['saved-evaluation-replay', projectId, revision, evaluation?.key, jobId, trial],
    queryFn: ({ signal }) => {
      if (!evaluation || !jobId || !trial) throw new Error('Select a saved Evaluate trial.');
      return fetchSavedEvaluationReplay(projectId, { revision, key: evaluation.key, job_id: jobId, trial }, signal);
    },
    enabled: visible && !!trial,
    retry: false,
  });
  const replayDrawing = useMemo(() => replay.data && evaluationScene(replay.data), [replay.data]);
  const view: SceneView = replay.error
    ? { kind: 'failed', issue: studioRegionIssue('scene', replay.error), retry: () => { void replay.refetch(); } }
    : replayDrawing ?? { kind: trial ? 'loading' : 'empty' };
  const timeScope = JSON.stringify([jobId, trial]);
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
  return {
    view, time, trials,
    clock: visible && view.kind === 'replay' && <SceneClockControls key={timeScope} timeline={sceneTimeline(view)} time={time}
      speed={speed} onSpeedChange={setSpeed} onTimeChange={(value) => setClock({ scope: timeScope, time: value })} />,
  };
}

function useSceneDetail({ editor, activeView, focus, step, evaluation, evaluationReplay }: DetailInputs, definition: Definition | null,
  graph: GraphSpec | null, scope: string, projectId: string, revision: string): ManagedProjectDetail {
  const interiors = useMemo(() => graph && definition?.id ? planarInteriors(graph) : [], [graph, definition?.id]);
  const interior = useElementSelection(scope);
  const path = interiors.find((candidate) => JSON.stringify(candidate) === interior.selectedId) ?? interiors[0];
  const source = definition?.id && path && activeView === 'scene'
    ? { document_id: definition.id, node_path: path, revision } : null;
  const geometry = useProjectPlanarScene(projectId, source, referencePose, () => editor.connected());
  const evaluating = step?.kind === 'evaluate';
  const view = evaluating ? evaluationReplay.view : geometry;
  const time = evaluating ? evaluationReplay.time : 0;
  const trials = evaluating ? evaluationReplay.trials : undefined;
  const sceneScope = JSON.stringify([scope, path, evaluation?.result?.job_id]);
  const selection = useElementSelection(sceneScope);
  const [cameraState, setCamera] = useState<{ scope: string; value: SceneCamera }>();
  const [overlayState, setOverlays] = useState<{ scope: string; value: ReadonlySet<SceneOverlay> }>();
  const camera = cameraState?.scope === sceneScope ? cameraState.value : fittedCamera;
  const onCameraChange = (value: SceneCamera) => setCamera({ scope: sceneScope, value });
  const hiddenOverlays = overlayState?.scope === sceneScope ? overlayState.value : shownOverlays;
  const scene = view.kind === 'geometry' || view.kind === 'replay' ? sceneGeometry(view, time) : null;
  const omission = !graph ? <DetailReason reason="definitionUnavailable" name="scene" />
    : definition?.id ? <DetailReason reason="planarSceneUnavailable" name="model" />
    : <DetailReason reason="sceneDocumentRequired" name="scene" />;
  return {
    documentId: definition?.id,
    header: <>
      {detailHeader(definition)}
      {interiors.length > 1 && path && <select aria-label={studioName('interior')} value={JSON.stringify(path)}
        onChange={({ currentTarget }) => interior.onSelect(currentTarget.value)}>
        {interiors.map((candidate) => <option key={JSON.stringify(candidate)} value={JSON.stringify(candidate)}>
          {candidate.join(' / ')}
        </option>)}
      </select>}
      <SceneHeaderControls view={view} camera={camera} onCameraChange={onCameraChange} hiddenOverlays={hiddenOverlays}
        onToggleOverlay={(overlay) => {
          const next = new Set(hiddenOverlays);
          if (next.has(overlay)) next.delete(overlay);
          else next.add(overlay);
          setOverlays({ scope: sceneScope, value: next });
        }} />
    </>,
    body: source || trials ? <div className="flex h-full min-h-0 flex-col">
      <div className="min-h-0 flex-1">
        <SceneCanvas view={view} time={time} camera={camera} onCameraChange={onCameraChange} hiddenOverlays={hiddenOverlays} fitToSlot {...selection} />
      </div>
      {!evaluating && focus && scene && <DetailReason reason="scenePlaybackUnavailable" name="playback" />}
    </div> : evaluating
      ? evaluation?.reason ? <p role="status">{studioReason('observed', evaluation.reason)}</p>
        : <DetailReason reason="scenePlaybackUnavailable" name="playback" />
      : omission,
    drawer: (scene || trials) && <SceneDrawer scene={scene ?? emptyReplayScene} trials={trials} {...selection} />,
    inspector: scene && selection.selectedId ? <SceneElementInspector scene={scene} selectedId={selection.selectedId} /> : null,
  };
}

export function useManagedProjectDetails(input: DetailInputs): Record<DetailName, ManagedProjectDetail> {
  const { editor, stepId, selection, variantId, focus, activeView } = input;
  const { saved, draft, outside, pending } = useSyncExternalStore(editor.subscribe, editor.getSnapshot);
  const subjectScope = JSON.stringify([saved.project_id, stepId, variantId]);
  const scope = JSON.stringify([subjectScope, selection, focus]);
  const namedField = activeView === 'model' ? 'graph' : activeView === 'task' ? 'task' : undefined;
  const namedDocument = namedField && input.step?.documentIds?.[`payload.${namedField}`];
  const reviewed: StudioDocumentAuthoringRequest | undefined = namedField && namedDocument && input.namedRun
    ? { revision: saved.revision, run: input.namedRun, field: namedField } : undefined;
  const named = useQuery({
    queryKey: ['named-definition', saved.project_id, reviewed],
    enabled: !!reviewed, retry: false,
    queryFn: ({ signal }) => fetchDocumentAuthoring(saved.project_id, reviewed!, signal),
  });
  const definitions = useMemo(() => {
    const train = stepId && !draft.sweep;
    const authored = named.data?.document === namedDocument ? named.data : undefined;
    const namedDefinition = authored && { id: authored.document, label: authored.document, payload: authored.payload, shared: true, authored };
    return {
      model: namedField === 'graph' && namedDefinition ? namedDefinition : train ? authoredDefinition(draft, stepId, 'graph') : null,
      task: namedField === 'task' && namedDefinition ? namedDefinition : train ? authoredDefinition(draft, stepId, 'task') : null,
    };
  }, [draft, stepId, named.data, namedDocument, namedField]);
  const graph = useMemo(() => {
    const result = GraphSpecSchema.safeParse(definitions.model?.payload);
    return result.success ? result.data : null;
  }, [definitions.model]);
  const task = useMemo(() => {
    const result = TaskSpecSchema.safeParse(definitions.task?.payload);
    return result.success ? result.data : null;
  }, [definitions.task]);
  const reveal = input.problemReveal;
  const element = reveal?.target.view === activeView ? reveal?.target.element : undefined;
  const document = reveal?.problem.document;
  const modelElement = !document || definitions.model?.id === document ? element : undefined;
  const wireTarget = modelElement?.kind === 'wire' ? graph?.wires?.[modelElement.index] : undefined;
  let revealedModelId: string | undefined;
  if (modelElement?.kind === 'component' && graph && Object.prototype.hasOwnProperty.call(graph.nodes ?? {}, modelElement.id)) {
    revealedModelId = `component:${modelElement.id}`;
  } else if (modelElement?.kind === 'wire' && wireTarget &&
      graph?.nodes?.[wireTarget.source_node] && graph.nodes?.[wireTarget.target_node]) {
    revealedModelId = graphEdgeId(wireTarget);
  }
  const taskId = element?.kind === 'taskParameter' && (!document || definitions.task?.id === document) &&
    task && Object.prototype.hasOwnProperty.call(task.params ?? {}, element.name) ? element.name : undefined;
  const canEdit = selection?.kind === 'steps' && selection.stepIds.length === 1 && selection.stepIds[0] === stepId && !outside;
  const modelSelection = useElementSelection(JSON.stringify([subjectScope, definitions.model?.id]), reveal, revealedModelId);
  const taskSelection = useElementSelection(JSON.stringify([subjectScope, definitions.task?.id]), reveal, taskId);
  const change = useDefinitionEdit(input, scope);
  function editable(field: DefinitionField, definition: Definition) {
    const target = input.documentScopes[field];
    return canEdit && (definition.id
      ? !pending && definition.authored?.revision === saved.revision && target?.document === definition.id &&
        (!target.localVariant || Boolean(definition.authored.variant_model_paths?.length))
      : !definition.shared);
  }
  const components = Object.entries(graph?.nodes ?? {});
  const modelItems = useMemo(() => Object.entries(graph?.nodes ?? {}).map(([id, node]) => ({
    id, selectionId: `component:${id}`, label: <>{id}<br />{node.type}</>,
  })), [graph]);
  const modelConnections = useMemo(() => (graph?.wires ?? []).map((wire) => ({
    id: graphEdgeId(wire), source: wire.source_node, target: wire.target_node,
  })), [graph]);
  const component = components.find(([id]) => modelSelection.selectedId === `component:${id}`);
  const wire = graph?.wires?.find((item) => modelSelection.selectedId === graphEdgeId(item));
  const registry = useQuery({ queryKey: ['components'], queryFn: fetchComponents, staleTime: 5 * 60 * 1000,
    enabled: activeView === 'model' });
  const model = definitions.model;
  const modelId = model?.id;
  const sharedPositions = modelId ? saved.presentation?.documents?.[modelId]?.positions : undefined;
  const localPositions = modelId && stepId ? saved.presentation?.steps?.[stepId]?.model_positions?.[modelId] : undefined;
  const positions = useMemo(() => localPositions ? { ...sharedPositions, ...localPositions } : sharedPositions,
    [sharedPositions, localPositions]);
  const graphScope = input.documentScopes.graph;
  const authorModel = useModelAuthoring({
    editor, graph, registry: registry.data, onSelect: modelSelection.onSelect,
    review: !pending && activeView === 'model' && model?.id && graphScope && !graphScope.localVariant && input.namedRun && editable('graph', model)
      ? { revision: saved.revision, presentationRevision: saved.presentation_revision,
        document: model.id, run: input.namedRun, litSteps: [...graphScope.lit] }
      : undefined,
  });
  const scene = useSceneDetail(input, model, graph, JSON.stringify([scope, model?.id]), saved.project_id, saved.revision);

  function componentParameters(definition: Definition, [id, component]: typeof components[number], selected: PipelineSelection) {
    const metadata = registry.data?.find((item) => (item.name === component.type || item.identity?.type_id === component.type) &&
      (component.revision == null || item.revision === component.revision));
    const schemas = metadata?.param_schema ?? [];
    const issue = change.issue('graph');
    const readOnlyReason = definition.authored?.variant_model_reason;
    const inline = stepId && !definition.id && !definition.shared;
    const raw = component.params ?? {};
    const path = ['nodes', id, 'params'];
    const parameterDetails = (projected: ReadonlySet<string>) => <ParameterDetails
      params={Object.fromEntries(Object.entries(raw).filter(([name]) => !projected.has(authoredPointer(['graph', 'nodes', id, 'params', name]))))}
      definition={definition} selection={selected} schemas={schemas} path={path}
      onEdit={editable('graph', definition) ? (field, value) => { void change.edit('graph', definition, field, [...path, field], value); } : undefined} />;
    return <section key={id} aria-label={studioName('details')} className="space-y-2">
      <h3 className="font-medium">{id}</h3>
      <p className="text-sm">{component.type}</p>
      {definition.shared && !editable('graph', definition) && (readOnlyReason
        ? <p className="text-sm">{readOnlyReason}</p> : <DetailReason reason="definitionReadOnly" name="model" />)}
      {named.error && <p role="alert">{apiErrorMessage(named.error)}</p>}
      {registry.error && <p role="alert">{apiErrorMessage(registry.error)}</p>}
      {inline ? <StudioTrainAuthoringForm editor={editor} stepId={stepId} component={id}
        step={input.step} selection={selected} onChooseSweepValue={input.onChooseSweepValue} renderAdditionalFields={parameterDetails} />
        : parameterDetails(new Set())}
      {issue && <p role="alert">{issue}</p>}
    </section>;
  }

  const taskDocument = definitions.task;
  const taskParam = Object.entries(task?.params ?? {}).find(([name]) => taskSelection.selectedId === name);
  return {
    model: {
      documentId: model?.id,
      header: detailHeader(model),
      body: graph ? <div className="flex h-full min-h-0 flex-col">
        {authorModel.issue}
        <div className="min-h-0 flex-1">
          <ManagedDocumentCanvas key={JSON.stringify([saved.project_id, model?.id ?? stepId])}
            items={modelItems} connections={modelConnections} positions={positions}
            openAt={authorModel.openAt} rememberPoint={authorModel.rememberPoint}
            placing={authorModel.placing} {...modelSelection} />
        </div>
      </div> : <DetailReason reason="definitionUnavailable" name="model" />,
      drawer: graph && (registry.error || authorModel.drawer || components.length > 0 || (graph.wires?.length ?? 0) > 0) ? <>
        {registry.error && <p role="alert">{apiErrorMessage(registry.error)}</p>}
        {authorModel.drawer}
        <DetailList name="components" items={components.map(([id]) => ({ id: `component:${id}`, label: id }))} {...modelSelection} />
        <DetailList name="wires" items={(graph.wires ?? []).map((item) => ({
          id: graphEdgeId(item), label: `${endpoint(item.source_node, item.source_port)} → ${endpoint(item.target_node, item.target_port)}`,
        }))} {...modelSelection} />
      </> : null,
      commands: authorModel.commands,
      overlay: authorModel.overlay,
      inspector: model && component && selection ? componentParameters(model, component, selection) : wire ? <WireDetails wire={wire} /> : null,
    },
    task: {
      documentId: taskDocument?.id,
      header: detailHeader(taskDocument),
      body: task && taskDocument && selection
        ? <TaskDetails key={JSON.stringify([scope, taskDocument.id])} task={task} definition={taskDocument} selection={selection}
          issue={change.issue('task') ?? (named.error ? apiErrorMessage(named.error) : null)} onEdit={editable('task', taskDocument)
            ? (field, value) => { void change.edit('task', taskDocument, field, ['params', field], value); } : undefined} />
        : <DetailReason reason="definitionUnavailable" name="task" />,
      drawer: task && Object.keys(task.params ?? {}).length > 0
        ? <DetailList name="parameters" items={Object.keys(task.params ?? {}).map((name) => ({ id: name, label: humanizeLabel(name) }))} {...taskSelection} />
        : null,
      inspector: taskParam ? <dl className="space-y-1 text-sm">
        <dt className="font-medium">{humanizeLabel(taskParam[0])}</dt><dd><StructuredValue value={taskParam[1]} /></dd>
      </dl> : null,
    },
    scene,
  };
}

function TaskDetails({ task, definition, selection, onEdit, issue }: {
  task: TaskSpec;
  definition: Definition;
  selection: PipelineSelection;
  onEdit?: (field: string, value: JsonPrimitive) => void;
  issue: string | null;
}) {
  return <section className="space-y-2 p-3" aria-label={studioName('task')}>
    <dl className="flex gap-2 text-sm"><dt>{studioName('type')}</dt><dd className="font-medium">{task.type}</dd></dl>
    {definition.shared && !onEdit && <DetailReason reason="definitionReadOnly" name="task" />}
    <ParameterDetails params={task.params ?? {}} definition={definition} selection={selection} path={['params']} onEdit={onEdit} />
    {task.timeline?.epochs?.length ? <section aria-label={studioName('epochs')}>
      <h3 className="font-medium">{studioName('epochs')}</h3>
      <ul>{task.timeline.epochs.map(({ id, label }) => <li key={id}>{label}</li>)}</ul>
    </section> : null}
    {issue && <p role="alert">{issue}</p>}
  </section>;
}
