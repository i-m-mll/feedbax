import dagre from '@dagrejs/dagre';
import { ReactFlow, ReactFlowProvider, getNodesBounds, useReactFlow, useStore, type Edge, type Node } from '@xyflow/react';
import { useQuery } from '@tanstack/react-query';
import { useEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactNode } from 'react';
import { fetchComponents } from '@/api/client';
import { fetchSavedEvaluationReplay } from '@/api/jobs';
import { apiErrorMessage } from '@/api/request';
import { studioRegionIssue } from '@/components/activity/StudioRegion';
import { studioAccessibleName, studioName, studioReason, type StudioNameId, type StudioReasonId } from '@/components/activity/studioVocabulary';
import { StudioDefinitionForm, type StudioDefinitionField, type StudioFieldControl } from '@/components/pipeline/StudioDefinitionForm';
import { SceneCanvas, SceneDrawer, SceneElementInspector, SceneHeaderControls } from '@/components/scene/SceneRegion';
import { SceneClockControls } from '@/components/scene/SceneClockControls';
import type { PlanarScenePose } from '@/components/scene/planarModel';
import { fittedCamera, sceneGeometry, sceneTimeline, type SceneCamera, type SceneDrawing, type SceneOverlay, type SceneView } from '@/components/scene/sceneView';
import { useProjectPlanarScene } from '@/components/scene/useProjectPlanarScene';
import { humanizeLabel } from '@/components/values/ValueSpecField';
import type { PlanStepView, RunAddress } from '@/features/pipeline/planView';
import type { PipelineSelection } from '@/features/pipeline/selection';
import type { ResolvedScene } from '@/features/scenario/projections';
import { primaryWorkspaceReplayTrack, workspaceReplayFrameIndex, workspaceReplaySampleAt } from '@/features/scenario/workspaceReplay';
import {
  DocumentReferenceSchema, GraphSpecSchema, TaskSpecSchema,
  type GraphSpec, type JsonPrimitive, type JsonValue, type ParamSchema, type StudioSavedEvaluationRow, type TaskSpec, type WireSpec,
  type WorkspaceReplayTrial,
} from '@/generated/studioContracts';
import { authoredPointer, directTrainStep, type AuthoredProject } from './authoredEdits';
import { StudioTrainAuthoringForm } from './StudioTrainAuthoringForm';
import { ProjectEditConflictError, type ProjectEditController } from './projectEditController';

type DetailName = 'model' | 'task' | 'scene';
type DefinitionField = 'graph' | 'task';
type Definition = {
  readonly id?: string;
  readonly label: string;
  readonly payload: JsonValue;
  readonly shared: boolean;
};

export interface ManagedProjectDetail {
  readonly documentId?: string;
  readonly header: ReactNode;
  readonly body: ReactNode;
  readonly drawer?: ReactNode;
  readonly inspector?: ReactNode;
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
}

function mentionsDocument(value: unknown, id: string): boolean {
  if (Array.isArray(value)) return value.some((child) => mentionsDocument(child, id));
  if (!value || typeof value !== 'object') return false;
  if ('document' in value && value.document === id && DocumentReferenceSchema.safeParse(value).success) return true;
  return Object.values(value).some((child) => mentionsDocument(child, id));
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
  const { [field]: _, ...otherFields } = inline;
  const shared = inherited || mentionsDocument(otherFields, id) ||
    otherSteps.some((other) => mentionsDocument(other, id)) ||
    Object.entries(project.documents ?? {}).some(([otherId, other]) => otherId !== id && mentionsDocument(other.payload, id));
  return { id, label: id, payload: document.payload, shared };
}

function detailHeader(definition: Definition | null) {
  return definition ? <span className="studio-window-slot__document">{definition.label}</span> : null;
}

function DetailReason({ reason, name }: { reason: StudioReasonId; name: StudioNameId }) {
  return <p role="status" className="px-3 py-2 text-sm text-slate-600">{studioReason(reason, studioName(name))}</p>;
}

function useElementSelection(scope: string) {
  const [selection, setSelection] = useState<{ scope: string; id: string | null }>();
  return {
    selectedId: selection?.scope === scope ? selection.id : null,
    onSelect: (id: string | null) => setSelection({ scope, id }),
  };
}

function DetailList({ name, items, selectedId, onSelect }: ReturnType<typeof useElementSelection> & {
  name: StudioNameId;
  items: readonly { id: string; label: string }[];
}) {
  if (!items.length) return null;
  return (
    <section className="space-y-1" aria-label={studioName(name)}>
      <h3 className="px-2 text-xs font-medium text-slate-700">{studioName(name)}</h3>
      <ul>
        {items.map(({ id, label }) => <li key={id}>
          <button type="button" aria-pressed={selectedId === id} onClick={() => onSelect(id)}
            className="w-full rounded px-2 py-1 text-left text-sm hover:bg-slate-100 aria-pressed:bg-violet-100">
            {label}
          </button>
        </li>)}
      </ul>
    </section>
  );
}

const endpoint = (node: string, port: string) => `${node}.${port}`;

function ModelCanvas({ graph, positions, selectedId, onSelect }: ReturnType<typeof useElementSelection> & {
  graph: GraphSpec;
  positions?: Record<string, { x: number; y: number }>;
}) {
  const { fitBounds, viewportInitialized } = useReactFlow();
  const width = useStore((state) => state.width);
  const height = useStore((state) => state.height);
  // The graph owns authored nodes. React Flow owns their rendered dimensions.
  const bounds = useStore(({ nodeLookup }) => {
    const rendered = [...nodeLookup.values()];
    return rendered.length > 0 && rendered.every((node) => !!node.measured.width && !!node.measured.height)
      ? getNodesBounds(rendered) : null;
  }, (first, second) => first?.x === second?.x && first?.y === second?.y &&
    first?.width === second?.width && first?.height === second?.height);
  const navigated = useRef(false);
  const { nodes, edges } = useMemo(() => {
    const entries = Object.entries(graph.nodes ?? {});
    const layout = new dagre.graphlib.Graph().setGraph({ rankdir: 'LR' }).setDefaultEdgeLabel(() => ({}));
    entries.forEach(([id]) => layout.setNode(id, { width: 150, height: 50 }));
    const edges = (graph.wires ?? []).flatMap((wire, index): Edge[] => {
      if (!layout.hasNode(wire.source_node) || !layout.hasNode(wire.target_node)) return [];
      layout.setEdge(wire.source_node, wire.target_node);
      return [{ id: `wire:${index}`, source: wire.source_node, target: wire.target_node }];
    });
    dagre.layout(layout);
    const nodes = entries.map(([id, node]): Node => {
      const { x, y } = layout.node(id);
      return { id, ariaLabel: studioAccessibleName('select', id),
        data: { label: <span className="break-words">{id}<br />{node.type}</span> },
        position: positions?.[id] ?? { x: x - 75, y: y - 25 } };
    });
    return { nodes, edges };
  }, [graph, positions]);
  const shownNodes = useMemo(() => nodes.map((node) => ({ ...node, selected: selectedId === `component:${node.id}` })), [nodes, selectedId]);
  const shownEdges = useMemo(() => edges.map((edge) => ({ ...edge, selected: selectedId === edge.id })), [edges, selectedId]);
  useEffect(() => {
    if (!bounds || !viewportInitialized || !width || !height) return;
    const frame = requestAnimationFrame(() => {
      if (!navigated.current) void fitBounds(bounds, { padding: 0.14 });
    });
    return () => cancelAnimationFrame(frame);
  }, [fitBounds, bounds, viewportInitialized, width, height]);
  return (
    <div className="h-full min-h-0">
      <ReactFlow nodes={shownNodes} edges={shownEdges}
        nodesDraggable={false} nodesConnectable={false} deleteKeyCode={null} proOptions={{ hideAttribution: true }}
        onMove={(event) => { if (event) navigated.current = true; }}
        onNodesChange={(changes) => {
          const change = changes.find((item) => item.type === 'select' && item.selected);
          if (change?.type === 'select') onSelect(`component:${change.id}`);
        }}
        onEdgesChange={(changes) => {
          const change = changes.find((item) => item.type === 'select' && item.selected);
          if (change?.type === 'select') onSelect(change.id);
        }}
        onPaneClick={() => onSelect(null)} />
    </div>
  );
}

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

function ParameterDetails({ params, definition, selection, schemas, onEdit }: {
  params: Record<string, unknown>;
  definition: Definition;
  selection: PipelineSelection;
  schemas?: ParamSchema[];
  onEdit?: (field: string, value: JsonPrimitive) => void;
}) {
  const fields: StudioDefinitionField[] = [];
  const structured: [string, unknown][] = [];
  for (const [name, value] of Object.entries(params)) {
    if (!scalar(value)) { structured.push([name, value]); continue; }
    const schema = schemas?.find((candidate) => candidate.name === name);
    const control = onEdit && (!schemas || schema) ? fieldControl(value, schema) : undefined;
    fields.push({ id: name, label: humanizeLabel(name), control, resolved: [{ id: name, value, source: definition.label }],
      actions: control ? ['set'] : [] });
  }
  return <>
    {fields.length > 0 && <StudioDefinitionForm fields={fields} selection={selection}
      onEdit={({ fieldId, edit }) => { if (edit.kind === 'set') onEdit?.(fieldId, edit.value); }} />}
    {structured.length > 0 && <dl className="space-y-2 text-sm">{structured.map(([name, value]) =>
      <div key={name} className="rounded border border-slate-200 p-2">
        <dt className="font-medium">{humanizeLabel(name)}</dt><dd><StructuredValue value={value} /></dd>
      </div>)}</dl>}
  </>;
}

function useDefinitionEdit(editor: ProjectEditController, scope: string, stepId: string | null, selection: PipelineSelection | null) {
  const [issue, setIssue] = useState<{ scope: string; field: DefinitionField; message: string }>();
  return {
    issue: (field: DefinitionField) => issue?.scope === scope && issue.field === field ? issue.message : null,
    edit(field: DefinitionField, definition: Definition, label: string, path: string[], value: JsonPrimitive) {
      if (!stepId || selection?.kind !== 'steps' || definition.shared) return;
      const id = definition.id;
      const check = (project: AuthoredProject) => {
        const latest = authoredDefinition(project, stepId, field);
        if (!latest || latest.id !== id || latest.shared) throw new ProjectEditConflictError();
      };
      try {
        check(editor.getSnapshot().draft);
        editor.edit(id ? { kind: 'document', id } : { kind: 'step', id: stepId }, label, [
          { op: 'replace', path: authoredPointer([...(id ? ['payload'] : ['base', 'inline', field]), ...path]), value },
        ], (before) => check(before));
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
  const jobId = evaluation?.result?.job_id;
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
  const { saved, draft, outside } = useSyncExternalStore(editor.subscribe, editor.getSnapshot);
  const subjectScope = JSON.stringify([saved.project_id, stepId, variantId]);
  const scope = JSON.stringify([subjectScope, selection, focus]);
  const definitions = useMemo(() => {
    const train = stepId && variantId === null;
    return {
      model: train ? authoredDefinition(draft, stepId, 'graph') : null,
      task: train ? authoredDefinition(draft, stepId, 'task') : null,
    };
  }, [draft, stepId, variantId]);
  const graph = useMemo(() => {
    const result = GraphSpecSchema.safeParse(definitions.model?.payload);
    return result.success ? result.data : null;
  }, [definitions.model]);
  const task = useMemo(() => {
    const result = TaskSpecSchema.safeParse(definitions.task?.payload);
    return result.success ? result.data : null;
  }, [definitions.task]);
  const canEdit = selection?.kind === 'steps' && selection.stepIds.length === 1 && selection.stepIds[0] === stepId && !outside;
  const modelSelection = useElementSelection(JSON.stringify([subjectScope, definitions.model?.id]));
  const taskSelection = useElementSelection(JSON.stringify([subjectScope, definitions.task?.id]));
  const change = useDefinitionEdit(editor, scope, stepId, selection);
  const components = Object.entries(graph?.nodes ?? {});
  const component = components.find(([id]) => modelSelection.selectedId === `component:${id}`);
  const wire = graph?.wires?.find((_, index) => modelSelection.selectedId === `wire:${index}`);
  const registry = useQuery({ queryKey: ['components'], queryFn: fetchComponents, staleTime: 5 * 60 * 1000,
    enabled: !!component && !definitions.model?.shared && activeView === 'model' });
  const model = definitions.model;
  const positions = model?.id ? saved.presentation?.documents?.[model.id]?.positions : undefined;
  const scene = useSceneDetail(input, model, graph, JSON.stringify([scope, model?.id]), saved.project_id, saved.revision);

  function componentParameters(definition: Definition, [id, component]: typeof components[number], selected: PipelineSelection) {
    const metadata = registry.data?.find((item) => (item.name === component.type || item.identity?.type_id === component.type) &&
      (component.revision == null || item.revision === component.revision));
    const schemas = metadata?.param_schema ?? [];
    const issue = change.issue('graph');
    const inline = stepId && !definition.id && !definition.shared;
    const raw = component.params ?? {};
    const parameterDetails = (projected: ReadonlySet<string>) => <ParameterDetails
      params={Object.fromEntries(Object.entries(raw).filter(([name]) => !projected.has(authoredPointer(['graph', 'nodes', id, 'params', name]))))}
      definition={definition} selection={selected} schemas={schemas}
      onEdit={canEdit && !definition.shared ? (field, value) => change.edit('graph', definition, field, ['nodes', id, 'params', field], value) : undefined} />;
    return <section key={id} aria-label={studioName('details')} className="space-y-2">
      <h3 className="font-medium">{id}</h3>
      <p className="text-sm">{component.type}</p>
      {definition.shared && <DetailReason reason="definitionReadOnly" name="model" />}
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
      body: graph ? <ReactFlowProvider key={JSON.stringify([saved.project_id, model?.id ?? stepId])}>
        <ModelCanvas graph={graph} positions={positions} {...modelSelection} />
      </ReactFlowProvider> : <DetailReason reason="definitionUnavailable" name="model" />,
      drawer: graph && (components.length > 0 || (graph.wires?.length ?? 0) > 0) ? <>
        <DetailList name="components" items={components.map(([id]) => ({ id: `component:${id}`, label: id }))} {...modelSelection} />
        <DetailList name="wires" items={(graph.wires ?? []).map((item, index) => ({
          id: `wire:${index}`, label: `${endpoint(item.source_node, item.source_port)} → ${endpoint(item.target_node, item.target_port)}`,
        }))} {...modelSelection} />
      </> : null,
      inspector: model && component && selection ? componentParameters(model, component, selection) : wire ? <WireDetails wire={wire} /> : null,
    },
    task: {
      documentId: taskDocument?.id,
      header: detailHeader(taskDocument),
      body: task && taskDocument && selection
        ? <TaskDetails key={JSON.stringify([scope, taskDocument.id])} task={task} definition={taskDocument} selection={selection}
          issue={change.issue('task')} onEdit={canEdit && !taskDocument.shared
            ? (field, value) => change.edit('task', taskDocument, field, ['params', field], value) : undefined} />
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
    {definition.shared && <DetailReason reason="definitionReadOnly" name="task" />}
    <ParameterDetails params={task.params ?? {}} definition={definition} selection={selection} onEdit={onEdit} />
    {task.timeline?.epochs?.length ? <section aria-label={studioName('epochs')}>
      <h3 className="font-medium">{studioName('epochs')}</h3>
      <ul>{task.timeline.epochs.map(({ id, label }) => <li key={id}>{label}</li>)}</ul>
    </section> : null}
    {issue && <p role="alert">{issue}</p>}
  </section>;
}
