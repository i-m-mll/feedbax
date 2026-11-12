import { useId, useMemo, useRef, type PointerEvent, type ReactNode, type SVGProps } from 'react';
import { LocateFixed, ZoomIn, ZoomOut } from 'lucide-react';
import { StudioRegion, StudioRegionBoundary } from '@/components/activity/StudioRegion';
import { studioAccessibleName, studioName, studioUnit, type StudioNameId } from '@/components/activity/studioVocabulary';
import { liveTrainingEffectorTrack, liveTrainingTargetTrack } from '@/features/scenario/liveTraining';
import type { ResolvedScene, ResolvedSceneElement } from '@/features/scenario/projections';
import { primaryWorkspaceReplayTrack, workspaceReplayFrameIndex, workspaceReplayPolyline, workspaceReplaySampleAt } from '@/features/scenario/workspaceReplay';
import { PlanarSceneElement } from './PlanarSceneElement';
import { WORKSPACE_SVG_HEIGHT, WORKSPACE_SVG_WIDTH, fitScale, includePoint, niceScaleLength, sceneBounds, type ScenePoint } from './planarProjection';
import { fittedCamera, sceneEntityOverlay, sceneGeometry, zoomScene, type SceneCamera, type SceneDrawing, type SceneOverlay, type SceneView } from './sceneView';

interface SceneSelection {
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}

function sceneTrack(view: SceneDrawing) {
  if (view.kind === 'replay') return primaryWorkspaceReplayTrack(view.trial);
  if (view.kind === 'live') return liveTrainingEffectorTrack(view.frame);
  return null;
}

function elementPosition(element: ResolvedSceneElement) {
  const { geometry } = element;
  if (geometry.kind === 'bounds') return geometry.min;
  return geometry.kind === 'none' ? undefined : geometry.points[0];
}

function selectionHandlers(id: string, label: string, selectedId: string | null, onSelect: SceneSelection['onSelect']): SVGProps<SVGGElement> {
  return {
    role: 'button',
    tabIndex: 0,
    'aria-label': studioAccessibleName('select', label),
    'aria-pressed': id === selectedId,
    className: 'cursor-pointer focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-violet-700',
    onPointerDown: (event) => event.stopPropagation(),
    onClick: (event) => { event.stopPropagation(); onSelect(id); },
    onKeyDown: (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        onSelect(id);
      }
    },
  };
}

type SceneDrawingProps = SceneSelection & {
  time: number;
  camera: SceneCamera;
  onCameraChange: (camera: SceneCamera) => void;
  hiddenOverlays: ReadonlySet<SceneOverlay>;
  problemEntityIds?: ReadonlySet<string>;
};

function SceneDrawingCanvas({
  view, time, camera, onCameraChange, hiddenOverlays, problemEntityIds, selectedId, onSelect,
}: SceneDrawingProps & { view: SceneDrawing }) {
  const gridId = useId();
  const pan = useRef<{ start: ScenePoint; camera: SceneCamera; ratio: number; moved: boolean } | null>(null);
  const scene = sceneGeometry(view, time);
  const track = sceneTrack(view);
  const path = useMemo(() => workspaceReplayPolyline(track), [track]);
  const frameIndex = view.kind === 'replay' ? workspaceReplayFrameIndex(view.trial, time) : null;
  const cursor = frameIndex == null ? path[path.length - 1] : workspaceReplaySampleAt(track, frameIndex);
  const goalTrack = view.kind === 'replay'
    ? view.trial.tracks?.find((track) => track.selector.role === 'target')
    : view.kind === 'live' ? liveTrainingTargetTrack(view.frame) : null;
  const goal = goalTrack && workspaceReplaySampleAt(goalTrack, frameIndex ?? goalTrack.samples.length - 1);
  const bounds = useMemo(() => {
    const bounds = sceneBounds(view.scene);
    path.forEach((point) => includePoint(bounds, point));
    workspaceReplayPolyline(goalTrack).forEach((point) => includePoint(bounds, point));
    return bounds;
  }, [view.scene, path, goalTrack]);
  const scale = fitScale(bounds) * camera.zoom;
  const center = [(bounds.min[0] + bounds.max[0]) / 2, (bounds.min[1] + bounds.max[1]) / 2];
  const project = ([x, y]: ScenePoint): ScenePoint => [
    WORKSPACE_SVG_WIDTH / 2 + (x - center[0]) * scale + camera.pan.x,
    WORKSPACE_SVG_HEIGHT / 2 - (y - center[1]) * scale + camera.pan.y,
  ];
  const meters = niceScaleLength(90 / Math.max(scale, 1));
  const entities = new Map(scene.entities.map((entity) => [entity.id, entity]));
  const entityIsVisible = (id: string) => {
    const entity = entities.get(id);
    const overlay = entity && sceneEntityOverlay(entity);
    return !overlay || !hiddenOverlays.has(overlay);
  };
  const visibleElements = scene.elements.filter((element) => entityIsVisible(element.entity_id));
  const diagnosedEntities = new Set(scene.validation.map((issue) => issue.entity_id));
  const marked = new Set<string>();

  const beginPan = (event: PointerEvent<SVGSVGElement>) => {
    if (event.button !== 0) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const ratio = Math.min(rect.width / WORKSPACE_SVG_WIDTH, rect.height / WORKSPACE_SVG_HEIGHT);
    if (!(ratio > 0)) return;
    pan.current = { start: [event.clientX, event.clientY], camera, ratio, moved: false };
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const movePan = (event: PointerEvent<SVGSVGElement>) => {
    const gesture = pan.current;
    if (!gesture) return;
    const dx = event.clientX - gesture.start[0];
    const dy = event.clientY - gesture.start[1];
    if (!gesture.moved && Math.hypot(dx, dy) < 3) return;
    gesture.moved = true;
    onCameraChange({
      ...gesture.camera,
      pan: { x: gesture.camera.pan.x + dx / gesture.ratio, y: gesture.camera.pan.y + dy / gesture.ratio },
    });
  };

  return (
    <svg
      viewBox={`0 0 ${WORKSPACE_SVG_WIDTH} ${WORKSPACE_SVG_HEIGHT}`}
      className="h-full min-h-0 w-full touch-none bg-white"
      role="group"
      aria-label={studioName('scene')}
      onPointerDown={beginPan}
      onPointerMove={movePan}
      onPointerUp={(event) => {
        if (!pan.current) return;
        if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
        if (!pan.current.moved) onSelect(null);
        pan.current = null;
      }}
      onPointerCancel={() => { pan.current = null; }}
      onWheel={(event) => onCameraChange(zoomScene(camera, event.deltaY > 0 ? 0.9 : 1.1))}
    >
      <defs>
        <pattern id={gridId} width="32" height="32" patternUnits="userSpaceOnUse">
          <path d="M 32 0 L 0 0 0 32" fill="none" stroke="#eef2f7" strokeWidth="1" />
        </pattern>
      </defs>
      <rect width={WORKSPACE_SVG_WIDTH} height={WORKSPACE_SVG_HEIGHT} fill={`url(#${gridId})`} />
      <g opacity={view.kind === 'replay' ? 0.6 : 1}>
        {visibleElements.map((element) => {
          const entity = entities.get(element.entity_id);
          if (!entity) return null;
          const point = elementPosition(element);
          const hasProblem = diagnosedEntities.has(entity.id) || problemEntityIds?.has(entity.id);
          const mark = point && hasProblem && !marked.has(entity.id) ? project(point) : null;
          if (mark) marked.add(entity.id);
          return (
            <g key={element.id}>
              <PlanarSceneElement
                element={element}
                active={selectedId === entity.id}
                project={project}
                scale={scale}
                interaction={selectionHandlers(entity.id, entity.label, selectedId, onSelect)}
              />
              {mark && <circle cx={mark[0]} cy={mark[1]} r={9} fill="none" stroke="#b91c1c" strokeWidth={2} pointerEvents="none" />}
            </g>
          );
        })}
      </g>
      <g pointerEvents="none" aria-hidden="true">
        {path.length > 1 && (
          <polyline points={path.map((point) => project(point).join(',')).join(' ')} fill="none"
            stroke="#0f766e" strokeWidth={2.5} vectorEffect="non-scaling-stroke" />
        )}
        {goal && <circle cx={project(goal)[0]} cy={project(goal)[1]} r={8} fill="none" stroke="#7c3aed" strokeWidth={2} />}
        {cursor && <circle data-scene-cursor cx={project(cursor)[0]} cy={project(cursor)[1]} r={6} fill="#0f766e" stroke="#fff" strokeWidth={2} />}
      </g>
      {scene.anchors.map((anchor) => {
        if (!anchor.position || !anchor.selectable || !entityIsVisible(anchor.entity_id) ||
          (!isJoint(anchor) && entities.get(anchor.entity_id)?.kind !== 'task_object')) return null;
        const [x, y] = project(anchor.position);
        return (
          <g key={anchor.id} {...selectionHandlers(anchor.id, anchor.label, selectedId, onSelect)}>
            <circle cx={x} cy={y} r={8} fill="transparent" />
            <circle cx={x} cy={y} r={selectedId === anchor.id ? 5 : 3} fill={selectedId === anchor.id ? '#7c3aed' : '#475569'} />
          </g>
        );
      })}
      <g transform={`translate(28 ${WORKSPACE_SVG_HEIGHT - 32})`} pointerEvents="none" aria-hidden="true">
        <path d={`M 0 -5 V 5 M 0 0 H ${meters * scale} M ${meters * scale} -5 V 5`} stroke="#334155" strokeWidth={2} />
        <text y={-9} className="fill-slate-600 text-[11px] font-medium">
          {meters >= 1 ? `${meters} ${studioUnit('meter')}` : `${Math.round(meters * 100)} ${studioUnit('centimeter')}`}
        </text>
      </g>
    </svg>
  );
}

function SceneIssues({ issues }: { issues: ResolvedScene['validation'] }) {
  if (!issues.length) return null;
  return (
    <ul role="alert" className="space-y-1 px-2 text-sm text-red-700">
      {issues.map((issue) => <li key={JSON.stringify([issue.type, issue.entity_id, issue.message])}>{issue.message}</li>)}
    </ul>
  );
}

export function SceneCanvas({ view, ...drawing }: SceneDrawingProps & { view: SceneView }) {
  switch (view.kind) {
    case 'geometry':
    case 'live':
    case 'replay':
      return (
        <StudioRegionBoundary name="scene">
          {view.kind === 'geometry' && !view.scene.elements.some(({ geometry }) => geometry.kind !== 'none') && view.scene.validation.length
            ? <SceneIssues issues={view.scene.validation} />
            : <SceneDrawingCanvas view={view} {...drawing} />}
        </StudioRegionBoundary>
      );
    case 'loading':
    case 'empty':
    case 'failed':
      return (
        <StudioRegion state={view} skeleton={<div className="h-full min-h-16 animate-pulse bg-slate-100" />}
          empty={<div className="h-full min-h-16 bg-slate-50" />}>
          {null}
        </StudioRegion>
      );
  }
}

export function SceneHeaderControls({
  view, camera, onCameraChange, hiddenOverlays, onToggleOverlay, renderControl,
}: {
  view: SceneView;
  camera: SceneCamera;
  onCameraChange: (camera: SceneCamera) => void;
  hiddenOverlays: ReadonlySet<SceneOverlay>;
  onToggleOverlay: (overlay: SceneOverlay) => void;
  renderControl?: (id: StudioNameId, button: ReactNode) => ReactNode;
}) {
  if (view.kind !== 'geometry' && view.kind !== 'live' && view.kind !== 'replay') return null;
  const drawableIds = new Set(view.scene.elements.filter((element) => element.geometry.kind !== 'none').map((element) => element.entity_id));
  if (!drawableIds.size && !view.scene.anchors.some((anchor) => anchor.position && anchor.selectable) &&
    !sceneTrack(view)?.samples.length) return null;
  const overlays = new Set(view.scene.entities.filter((entity) => drawableIds.has(entity.id)).map(sceneEntityOverlay));
  const control = (id: StudioNameId, button: ReactNode) => renderControl ? renderControl(id, button) : button;
  const buttonClass = 'rounded px-2 py-1 text-xs font-medium text-slate-700 hover:bg-slate-100 aria-pressed:bg-violet-100';
  return (
    <div className="flex flex-wrap items-center gap-1">
      {control('zoomIn', <button type="button" aria-label={studioName('zoomIn')} className={buttonClass} onClick={() => onCameraChange(zoomScene(camera, 1.2))}><ZoomIn className="size-4" aria-hidden="true" /></button>)}
      {control('zoomOut', <button type="button" aria-label={studioName('zoomOut')} className={buttonClass} onClick={() => onCameraChange(zoomScene(camera, 1 / 1.2))}><ZoomOut className="size-4" aria-hidden="true" /></button>)}
      {control('fitView', <button type="button" aria-label={studioName('fitView')} className={buttonClass} onClick={() => onCameraChange(fittedCamera)}><LocateFixed className="size-4" aria-hidden="true" /></button>)}
      {(['objectives', 'observables', 'overlays'] as const).filter((overlay) => overlays.has(overlay)).map((overlay) => (
        <span key={overlay}>
          {control(overlay, (
            <button type="button" className={buttonClass} aria-pressed={!hiddenOverlays.has(overlay)} onClick={() => onToggleOverlay(overlay)}>
              {studioName(overlay)}
            </button>
          ))}
        </span>
      ))}
    </div>
  );
}

const drawerGroups = ['bodies', 'joints', 'muscles', 'task', 'objectives', 'observables', 'overlays'] as const;
type DrawerGroup = typeof drawerGroups[number];

function isJoint(anchor: ResolvedScene['anchors'][number]) {
  return anchor.semantic_role === 'joint' || typeof anchor.metadata.joint_index === 'number';
}

function isDrawerAnchor(scene: ResolvedScene, anchor: ResolvedScene['anchors'][number]) {
  return isJoint(anchor) || scene.entities.some((entity) => entity.id === anchor.entity_id && entity.kind === 'task_object');
}

export function SceneDrawer({ scene, selectedId, onSelect, palette, trials }: SceneSelection & {
  scene: ResolvedScene;
  palette?: ReactNode;
  trials?: {
    choices: readonly { id: string; label: string }[];
    selectedId: string | null;
    onSelect: (id: string) => void;
  };
}) {
  const muscleIds = new Set(scene.elements.filter((element) => element.archetype === 'muscle_path').map((element) => element.entity_id));
  const issuesByEntity = new Map<string, ResolvedScene['validation']>();
  for (const issue of scene.validation) {
    if (issue.entity_id) issuesByEntity.set(issue.entity_id, [...(issuesByEntity.get(issue.entity_id) ?? []), issue]);
  }
  const groups = new Map<DrawerGroup, { id: string; label: string }[]>();
  function add(group: DrawerGroup, item: { id: string; label: string }) {
    const items = groups.get(group);
    if (items) items.push(item);
    else groups.set(group, [item]);
  }
  for (const entity of scene.entities) {
    const overlay = sceneEntityOverlay(entity);
    if (overlay) add(overlay, entity);
    else if (entity.kind === 'mechanics_object') add(muscleIds.has(entity.id) ? 'muscles' : 'bodies', entity);
    else if (entity.kind === 'task_object' || entity.kind === 'task_data' || entity.kind === 'task_binding') add('task', entity);
  }
  for (const anchor of scene.anchors) {
    if (anchor.selectable && isDrawerAnchor(scene, anchor)) add(isJoint(anchor) ? 'joints' : 'task', anchor);
  }
  return (
    <div className="space-y-3">
      {trials && trials.choices.length > 0 && (
        <label className="grid gap-1 px-2 text-sm">
          {studioName('trials')}
          <select value={trials.selectedId ?? ''} onChange={(event) => trials.onSelect(event.currentTarget.value)}>
            {trials.selectedId === null && <option value="" disabled>{studioName('choose')}</option>}
            {trials.choices.map(({ id, label }) => <option key={id} value={id}>{label}</option>)}
          </select>
        </label>
      )}
      {drawerGroups.map((group) => {
        const items = groups.get(group);
        if (!items) return null;
        return (
          <section key={group} aria-label={studioName(group)}>
            <h3 className="px-2 pb-1 text-xs font-medium text-slate-700">{studioName(group)}</h3>
            <ul>
              {items.map((item) => (
                <li key={item.id}>
                  <button type="button" aria-pressed={item.id === selectedId} onClick={() => onSelect(item.id)}
                    className="w-full rounded px-2 py-1 text-left text-sm text-slate-900 hover:bg-slate-100 aria-pressed:bg-violet-100">
                    {item.label}
                  </button>
                  <SceneIssues issues={issuesByEntity.get(item.id) ?? []} />
                </li>
              ))}
            </ul>
          </section>
        );
      })}
      {palette}
    </div>
  );
}

const coordinateFormat = new Intl.NumberFormat(undefined, { maximumFractionDigits: 4 });

function inspectorPosition(anchor: ResolvedScene['anchors'][number]) {
  const { authored_frame: frame, offset } = anchor.metadata;
  return typeof frame === 'string' && Array.isArray(offset) && offset.every(Number.isFinite) ? offset : anchor.position;
}

export function SceneElementInspector({ scene, selectedId }: { scene: ResolvedScene; selectedId: string | null }) {
  const entity = scene.entities.find((item) => item.id === selectedId);
  const anchor = scene.anchors.find((item) => item.id === selectedId);
  const selected = entity ?? anchor;
  if (!selected) return null;
  const anchors = anchor ? [anchor] : scene.anchors.filter((item) => item.entity_id === entity?.id);
  return (
    <section aria-label={studioName('details')} className="space-y-2">
      <h3 className="font-medium text-slate-900">{selected.label}</h3>
      <dl className="grid gap-2">
        {anchors.map((item) => {
          const position = inspectorPosition(item);
          return position && (
            <div key={item.id} className="flex flex-wrap justify-between gap-2 text-sm">
              <dt className="text-slate-700">{item.label}</dt>
              <dd className="tabular-nums">{position.map((coordinate) => coordinateFormat.format(coordinate)).join(', ')} {studioUnit('meter')}</dd>
            </div>
          );
        })}
      </dl>
    </section>
  );
}
