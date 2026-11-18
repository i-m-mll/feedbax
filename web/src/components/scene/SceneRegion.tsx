import { useId, useLayoutEffect, useMemo, useRef, type PointerEvent, type SVGProps } from 'react';
import { Layers } from 'lucide-react';
import { IconButton } from '@/components/ui/Button';
import { FieldRow } from '@/components/ui/FieldRow';
import { Inspector } from '@/components/ui/Inspector';
import { StatusMark } from '@/components/ui/StatusMark';
import { Tree, type TreeNode } from '@/components/ui/Tree';
import { ValueField } from '@/components/values/ValueField';
import { rawField } from '@/components/values/adapters/raw';
import { Drawer, DrawerSection, type DrawerSections } from '@/features/projectWindow/DrawerForms';
import { Menu, MenuCheckboxItem, MenuContent, MenuTrigger } from '@/components/ui/Menu';
import { Select } from '@/components/ui/Select';
import { StudioRegion, StudioRegionBoundary } from '@/components/activity/StudioRegion';
import { studioAccessibleName, studioName, studioUnit } from '@/components/activity/studioVocabulary';
import { liveTrainingEffectorTrack, liveTrainingTargetTrack } from '@/features/scenario/liveTraining';
import type { ResolvedScene, ResolvedSceneElement } from '@/features/scenario/projections';
import { primaryWorkspaceReplayTrack, workspaceReplayFrameIndex, workspaceReplayPolyline, workspaceReplaySampleAt } from '@/features/scenario/workspaceReplay';
import { useElementSize } from '@/hooks/useElementSize';
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
    className: 'cursor-pointer focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-shell-accent',
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
  fitToSlot?: boolean;
  problemEntityIds?: ReadonlySet<string>;
};

function SceneDrawingCanvas({
  view, time, camera, onCameraChange, hiddenOverlays, problemEntityIds, selectedId, onSelect, fitToSlot = false,
}: SceneDrawingProps & { view: SceneDrawing }) {
  const gridId = useId();
  const { ref: canvas, width, height } = useElementSize<SVGSVGElement>(fitToSlot);
  const viewportWidth = fitToSlot && width > 0 ? width : WORKSPACE_SVG_WIDTH;
  const viewportHeight = fitToSlot && height > 0 ? height : WORKSPACE_SVG_HEIGHT;
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
  const fit = fitScale(bounds, viewportWidth, viewportHeight);
  const autoFit = !fitToSlot || camera === fittedCamera;
  const navigatedFit = useRef<number>();
  useLayoutEffect(() => {
    if (autoFit) navigatedFit.current = undefined;
    else navigatedFit.current ??= fit;
  }, [autoFit, fit]);
  const scale = (autoFit ? fit : navigatedFit.current ?? fit) * camera.zoom;
  const center: ScenePoint = [(bounds.min[0] + bounds.max[0]) / 2, (bounds.min[1] + bounds.max[1]) / 2];
  const project = ([x, y]: ScenePoint): ScenePoint => [
    viewportWidth / 2 + (x - center[0]) * scale + camera.pan.x,
    viewportHeight / 2 - (y - center[1]) * scale + camera.pan.y,
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
  const selectedAnchor = scene.anchors.find((anchor) => anchor.id === selectedId);
  const selectedEntity = selectedAnchor?.entity_id ?? selectedId;

  const beginPan = (event: PointerEvent<SVGSVGElement>) => {
    if (event.button !== 0) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const ratio = Math.min(rect.width / viewportWidth, rect.height / viewportHeight);
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
      ref={canvas}
      viewBox={`0 0 ${viewportWidth} ${viewportHeight}`}
      className="h-full min-h-0 w-full touch-none bg-shell-surface"
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
          <path d="M 32 0 L 0 0 0 32" fill="none" stroke="var(--studio-scene-grid)" strokeWidth="1" />
        </pattern>
      </defs>
      <rect width={viewportWidth} height={viewportHeight} fill={`url(#${gridId})`} />
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
                active={selectedEntity === entity.id}
                project={project}
                scale={scale}
                interaction={selectionHandlers(entity.id, entity.label, selectedId, onSelect)}
              />
              {mark && <circle cx={mark[0]} cy={mark[1]} r={9} fill="none" stroke="var(--studio-status-failure)" strokeWidth={2} pointerEvents="none" />}
            </g>
          );
        })}
      </g>
      <g pointerEvents="none" aria-hidden="true">
        {path.length > 1 && (
          <polyline points={path.map((point) => project(point).join(',')).join(' ')} fill="none"
            stroke="var(--text-primary)" strokeWidth={2.5} vectorEffect="non-scaling-stroke" />
        )}
        {goal && <circle cx={project(goal)[0]} cy={project(goal)[1]} r={8} fill="none" stroke="var(--studio-scene-feature)" strokeWidth={2} />}
        {cursor && <circle data-scene-cursor cx={project(cursor)[0]} cy={project(cursor)[1]} r={6} fill="var(--text-primary)" stroke="var(--surface-main)" strokeWidth={2} />}
      </g>
      {scene.anchors.map((anchor) => {
        if (!anchor.position || !anchor.selectable || !entityIsVisible(anchor.entity_id) ||
          (!isJoint(anchor) && entities.get(anchor.entity_id)?.kind !== 'task_object')) return null;
        const [x, y] = project(anchor.position);
        return (
          <g key={anchor.id} {...selectionHandlers(anchor.id, anchor.label, selectedId, onSelect)}>
            <circle cx={x} cy={y} r={8} fill="transparent" />
            <circle cx={x} cy={y} r={selectedId === anchor.id ? 5 : 3} fill={selectedId === anchor.id ? 'var(--text-primary)' : 'var(--studio-scene-outline)'} />
          </g>
        );
      })}
      <g transform={`translate(28 ${viewportHeight - 32})`} pointerEvents="none" aria-hidden="true">
        <path d={`M 0 -5 V 5 M 0 0 H ${meters * scale} M ${meters * scale} -5 V 5`} stroke="var(--text-primary)" strokeWidth={2} />
        <text y={-9} className="fill-shell-muted text-xs font-medium">
          {meters >= 1 ? `${meters} ${studioUnit('meter')}` : `${Math.round(meters * 100)} ${studioUnit('centimeter')}`}
        </text>
      </g>
    </svg>
  );
}

function SceneIssues({ issues }: { issues: ResolvedScene['validation'] }) {
  if (!issues.length) return null;
  return (
    <ul role="alert" className="space-y-1 px-2 text-sm text-shell-status-failure">
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
        <StudioRegion state={view} skeleton={<div className="h-full min-h-16 motion-safe:animate-pulse bg-shell-subtle" />}
          empty={<div className="h-full min-h-16 bg-shell-subtle" />}>
          {null}
        </StudioRegion>
      );
  }
}

const overlayOrder = ['objectives', 'observables', 'overlays'] as const satisfies readonly SceneOverlay[];

/** What a view's strip can act on: whether there is anything to frame, and which layers it draws. */
export function sceneLayers(view: SceneView): { navigable: boolean; overlays: readonly SceneOverlay[] } {
  if (view.kind !== 'geometry' && view.kind !== 'live' && view.kind !== 'replay') return { navigable: false, overlays: [] };
  const drawableIds = new Set(view.scene.elements.filter((element) => element.geometry.kind !== 'none').map((element) => element.entity_id));
  const navigable = drawableIds.size > 0 || view.scene.anchors.some((anchor) => anchor.position && anchor.selectable) ||
    !!sceneTrack(view)?.samples.length;
  const present = new Set(view.scene.entities.filter((entity) => drawableIds.has(entity.id)).map(sceneEntityOverlay));
  return { navigable, overlays: overlayOrder.filter((overlay) => present.has(overlay)) };
}

export function SceneLayers({ overlays, hiddenOverlays, onToggleOverlay }: {
  overlays: readonly SceneOverlay[];
  hiddenOverlays: ReadonlySet<SceneOverlay>;
  onToggleOverlay: (overlay: SceneOverlay) => void;
}) {
  return (
    <Menu>
      <MenuTrigger disabled={!overlays.length}
        render={<IconButton label={studioName('layers')} icon={<Layers />} tipPlacement="right" />} />
      <MenuContent side="right" aria-label={studioName('layers')}>
        {overlays.map((overlay) => (
          <MenuCheckboxItem key={overlay} checked={!hiddenOverlays.has(overlay)} onCheckedChange={() => onToggleOverlay(overlay)}>
            {studioName(overlay)}
          </MenuCheckboxItem>
        ))}
      </MenuContent>
    </Menu>
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

export function SceneDrawer({ sections, scene, selectedId, onSelect, trials, checkpoints }: SceneSelection & {
  sections: DrawerSections;
  scene: ResolvedScene;
  trials?: {
    choices: readonly { id: string; label: string }[];
    selectedId: string | null;
    onSelect: (id: string) => void;
  };
  checkpoints?: {
    choices: readonly number[];
    selected: number;
    onSelect: (checkpoint: number) => void;
  };
}) {
  const checkpointLabel = useId();
  const trialLabel = useId();
  const muscleIds = new Set(scene.elements.filter((element) => element.archetype === 'muscle_path').map((element) => element.entity_id));
  const issueCounts = new Map<string, string[]>();
  for (const issue of scene.validation) {
    if (issue.entity_id) issueCounts.set(issue.entity_id, [...(issueCounts.get(issue.entity_id) ?? []), issue.message]);
  }
  const groups = new Map<DrawerGroup, TreeNode[]>();
  function add(group: DrawerGroup, item: { id: string; label: string }) {
    const messages = issueCounts.get(item.id);
    const node: TreeNode = {
      id: item.id, name: item.label,
      trailing: messages && <span className="inline-flex items-center gap-1 tabular-nums">
        <StatusMark status="cannot_run" label={studioName('problems')} />{messages.length > 1 ? messages.length : null}
      </span>,
    };
    const items = groups.get(group);
    if (items) items.push(node);
    else groups.set(group, [node]);
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
  const replay = checkpoints || (trials && trials.choices.length > 0);
  return (
    <Drawer>
      {replay && (
        <DrawerSection sections={sections} id="scene.trials" name={studioName('trials')} bodyClassName="grid gap-1 p-1">
          {checkpoints && <FieldRow label={studioName('checkpoint')} labelId={checkpointLabel}>
            <Select aria-labelledby={checkpointLabel} value={String(checkpoints.selected)}
              items={checkpoints.choices.map((coordinate) => ({ value: String(coordinate), label: String(coordinate) }))}
              onValueChange={(value) => checkpoints.onSelect(Number(value))} />
          </FieldRow>}
          {trials && trials.choices.length > 0 && <FieldRow label={studioName('trials')} labelId={trialLabel}>
            <Select aria-labelledby={trialLabel} value={trials.selectedId} placeholder={studioName('choose')}
              items={trials.choices.map(({ id, label }) => ({ value: id, label }))} onValueChange={trials.onSelect} />
          </FieldRow>}
        </DrawerSection>
      )}
      {drawerGroups.map((group) => {
        const nodes = groups.get(group);
        if (!nodes) return null;
        return (
          <DrawerSection key={group} sections={sections} id={`scene.${group}`} name={studioName(group)} count={nodes.length}>
            <Tree aria-label={studioName(group)} nodes={nodes} selected={selectedId && nodes.some(({ id }) => id === selectedId) ? [selectedId] : []}
              onSelectedChange={(ids) => onSelect(ids.at(-1) ?? null)} />
          </DrawerSection>
        );
      })}
    </Drawer>
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
  const issues = scene.validation.filter((issue) => issue.entity_id === (entity?.id ?? anchor?.entity_id));
  return (
    <Inspector name={selected.label}>
      {anchors.flatMap((item) => {
        const position = inspectorPosition(item);
        return position ? [<ValueField key={item.id} model={rawField({
          id: item.id, label: item.label, type: { kind: 'readOnly' }, unit: studioUnit('meter'),
          value: position.map((coordinate) => coordinateFormat.format(coordinate)).join(', '),
        })} />] : [];
      })}
      {issues.map((issue) => <p key={JSON.stringify([issue.type, issue.message])} role="status" className="text-shell-status-failure">
        {issue.message}
      </p>)}
    </Inspector>
  );
}
