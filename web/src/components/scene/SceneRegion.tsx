import { useId, useLayoutEffect, useMemo, useRef, type PointerEvent, type SVGProps } from 'react';
import { AudioWaveform, Boxes, Circle, Layers, RectangleHorizontal, Variable, Waypoints } from 'lucide-react';
import { IconButton } from '@/components/ui/Button';
import { FieldRow } from '@/components/ui/FieldRow';
import { Inspector } from '@/components/ui/Inspector';
import { StatusMark } from '@/components/ui/StatusMark';
import { Tree, type TreeNode } from '@/components/ui/Tree';
import { ValueField } from '@/components/values/ValueField';
import { rawField } from '@/components/values/adapters/raw';
import { Drawer, DrawerSection, type DrawerSections } from '@/features/projectWindow/DrawerForms';
import type { ElementAddress } from '@/features/projectWindow/elementSelection';
import { Menu, MenuCheckboxItem, MenuContent, MenuTrigger } from '@/components/ui/Menu';
import { Select } from '@/components/ui/Select';
import { StudioRegion, StudioRegionBoundary } from '@/components/activity/StudioRegion';
import { studioAccessibleName, studioName, studioUnit } from '@/components/activity/studioVocabulary';
import { useElementSize } from '@/hooks/useElementSize';
import { WORKSPACE_SVG_HEIGHT, WORKSPACE_SVG_WIDTH, fitScale, niceScaleLength, sceneBounds, shapePoints, type ScenePoint } from './planarProjection';
import {
  fittedCamera, sceneElements, sceneLayers, zoomScene,
  type Scene, type SceneCamera, type SceneElement, type SceneLayer, type SceneShape, type SceneView,
} from './sceneView';

/** The window's selection as the Scene sees it: the key of the selected element, and a choice of one by its address. */
interface SceneSelection {
  selected: string | null;
  onSelect: (address: ElementAddress | null) => void;
}

function selectionHandlers(element: SceneElement, selected: string | null, onSelect: SceneSelection['onSelect']): SVGProps<SVGGElement> {
  return {
    role: 'button',
    tabIndex: 0,
    'aria-label': studioAccessibleName('select', element.name),
    'aria-pressed': element.key === selected,
    className: 'cursor-pointer focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-shell-accent',
    onPointerDown: (event) => event.stopPropagation(),
    onClick: (event) => { event.stopPropagation(); onSelect(element.address); },
    onKeyDown: (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        onSelect(element.address);
      }
    },
  };
}

/**
 * Bodies under muscles under joints, so that a joint stays pressable where a body or a muscle crosses it; the task's
 * places above the mechanisms.
 */
const drawOrder: Record<SceneLayer, number> = { bodies: 0, muscles: 1, joints: 2, variables: 3, signals: 4 };

/** Where an element is drawn: its shape in the shown trial, and each place it may take. */
const elementPoints = (element: SceneElement): readonly ScenePoint[] =>
  [...element.shape ? shapePoints(element.shape) : [], ...element.possible ?? []];

/** What is drawn faint: the mechanisms at rest beneath a recorded pose. */
function RestDrawing({ shape, project }: { shape: SceneShape; project: (point: ScenePoint) => ScenePoint }) {
  if (shape.kind === 'point') {
    const [x, y] = project(shape.point);
    return <circle cx={x} cy={y} r={3} fill="var(--studio-scene-grid)" />;
  }
  return <polyline points={shape.points.map((point) => project(point).join(',')).join(' ')} fill="none"
    stroke="var(--studio-scene-grid)" strokeWidth={shape.kind === 'segment' ? 6 : 2} strokeLinecap="round"
    strokeLinejoin="round" vectorEffect="non-scaling-stroke" />;
}

function ElementDrawing({ element, active, project }: {
  element: SceneElement;
  active: boolean;
  project: (point: ScenePoint) => ScenePoint;
}) {
  const stroke = active ? 'var(--text-primary)' : 'var(--studio-scene-outline)';
  const possible = element.possible?.map(project).map(([x, y], index) => <circle key={index} data-scene-possible cx={x} cy={y}
    r={active ? 6 : 5} fill="transparent" stroke={stroke} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />);
  const { shape } = element;
  if (!shape) return <>{possible}</>;
  if (shape.kind === 'point') {
    const [x, y] = project(shape.point);
    return <>
      {possible}
      <circle data-scene-solid cx={x} cy={y} r={active ? 5 : 3} fill={stroke} />
      <circle cx={x} cy={y} r={8} fill="transparent" />
    </>;
  }
  const points = shape.points.map(project);
  const body = shape.kind === 'segment';
  return <>
    {possible}
    <polyline points={points.map((point) => point.join(',')).join(' ')} fill="none" stroke={stroke}
      strokeWidth={body ? active ? 8 : 6 : active ? 3 : 2} strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
    {points.map(([x, y], index) => <circle key={index} cx={x} cy={y} r={body ? 4.5 : 3}
      fill={body && index === points.length - 1 ? stroke : 'var(--surface-main)'} stroke={stroke} strokeWidth={1.5}
      vectorEffect="non-scaling-stroke" />)}
  </>;
}

type SceneDrawingProps = SceneSelection & {
  camera: SceneCamera;
  onCameraChange: (camera: SceneCamera) => void;
  hiddenLayers: ReadonlySet<SceneLayer>;
  fitToSlot?: boolean;
};

function SceneDrawingCanvas({
  scene, camera, onCameraChange, hiddenLayers, selected, onSelect, fitToSlot = false,
}: SceneDrawingProps & { scene: Scene }) {
  const gridId = useId();
  const { ref: canvas, width, height } = useElementSize<SVGSVGElement>(fitToSlot);
  const viewportWidth = fitToSlot && width > 0 ? width : WORKSPACE_SVG_WIDTH;
  const viewportHeight = fitToSlot && height > 0 ? height : WORKSPACE_SVG_HEIGHT;
  const pan = useRef<{ start: ScenePoint; camera: SceneCamera; ratio: number; moved: boolean } | null>(null);
  const bounds = useMemo(() => sceneBounds(scene, [
    ...sceneElements(scene).flatMap((element) => element.possible ?? []),
    ...scene.rest.flatMap(shapePoints),
    ...scene.tracks.flatMap((track) => track.points),
  ]), [scene]);
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
  const drawn = sceneElements(scene)
    .filter((element) => elementPoints(element).length > 0 && !hiddenLayers.has(element.layer))
    .sort((left, right) => drawOrder[left.layer] - drawOrder[right.layer]);

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
      <g pointerEvents="none" aria-hidden="true" data-scene-rest>
        {scene.rest.map((shape, index) => <RestDrawing key={index} shape={shape} project={project} />)}
      </g>
      <g pointerEvents="none" aria-hidden="true">
        {scene.tracks.map((track, index) => {
          const stroke = track.picked ? 'var(--text-primary)' : 'var(--studio-scene-outline)';
          return <g key={index}>
            {track.drawn > 0 && <polyline data-scene-track={track.picked ? 'picked' : 'other'} fill="none" stroke={stroke}
              points={track.points.slice(0, track.drawn).map((point) => project(point).join(',')).join(' ')}
              strokeWidth={track.picked ? 2 : 1} vectorEffect="non-scaling-stroke" />}
            {track.head && <circle data-scene-cursor cx={project(track.head)[0]} cy={project(track.head)[1]} r={track.picked ? 3 : 1.5}
              fill={stroke} />}
          </g>;
        })}
      </g>
      {drawn.map((element) => {
        const mark = element.problem && project(elementPoints(element)[0]);
        return (
          <g key={element.key} data-scene-layer={element.layer}>
            <g {...selectionHandlers(element, selected, onSelect)}>
              <ElementDrawing element={element} active={element.key === selected} project={project} />
            </g>
            {mark && <circle cx={mark[0]} cy={mark[1]} r={9} fill="none" stroke="var(--studio-status-failure)" strokeWidth={2} pointerEvents="none" />}
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

export function SceneCanvas({ view, ...drawing }: SceneDrawingProps & { view: SceneView }) {
  switch (view.kind) {
    case 'drawn':
      return (
        <StudioRegionBoundary name="scene">
          <SceneDrawingCanvas scene={view.scene} {...drawing} />
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

/** What a view's strip can act on: whether there is anything to frame, and the layers it draws. */
export function sceneContents(view: SceneView): { navigable: boolean; layers: readonly SceneLayer[] } {
  if (view.kind !== 'drawn') return { navigable: false, layers: [] };
  const placed = sceneElements(view.scene).filter((element) => elementPoints(element).length > 0);
  const present = new Set(placed.map((element) => element.layer));
  return {
    navigable: placed.length > 0 || view.scene.tracks.some((track) => track.points.length > 0),
    layers: sceneLayers.filter((layer) => present.has(layer)),
  };
}

export function SceneLayers({ layers, hidden, onToggle }: {
  layers: readonly SceneLayer[];
  hidden: ReadonlySet<SceneLayer>;
  onToggle: (layer: SceneLayer) => void;
}) {
  return (
    <Menu>
      <MenuTrigger disabled={!layers.length}
        render={<IconButton label={studioName('layers')} icon={<Layers />} tipPlacement="right" />} />
      <MenuContent side="right" aria-label={studioName('layers')}>
        {layers.map((layer) => (
          <MenuCheckboxItem key={layer} checked={!hidden.has(layer)} onCheckedChange={() => onToggle(layer)}>
            {studioName(layer)}
          </MenuCheckboxItem>
        ))}
      </MenuContent>
    </Menu>
  );
}

const layerGlyphs = {
  bodies: <RectangleHorizontal />, joints: <Circle />, muscles: <Waypoints />, variables: <Variable />, signals: <AudioWaveform />,
} satisfies Record<SceneLayer, unknown>;

function outlineMark(problem: string | undefined) {
  return problem && <StatusMark status="cannot_run" label={problem} />;
}

/**
 * Each mechanism with its bodies, joints and muscles, and the task with its variables and signals that have a place,
 * in Layers' order, as rows keyed by their element's key.
 */
function sceneOutline(scene: Scene) {
  const addresses = new Map<string, ElementAddress>();
  const nodes = scene.groups.map((group): TreeNode => {
    addresses.set(group.key, group.address);
    const children = sceneLayers.flatMap((layer) => group.elements.filter((element) => element.layer === layer)).map((element): TreeNode => {
      addresses.set(element.key, element.address);
      return { id: element.key, name: element.name, glyph: layerGlyphs[element.layer], trailing: outlineMark(element.problem) };
    });
    return { id: group.key, name: group.name, glyph: <Boxes />, trailing: outlineMark(group.problem), children: children.length ? children : undefined };
  });
  return { nodes, addresses };
}

export function SceneDrawer({ sections, scene, selected, onSelect, checkpoints }: SceneSelection & {
  sections: DrawerSections;
  scene: Scene;
  checkpoints?: {
    choices: readonly number[];
    selected: number;
    onSelect: (checkpoint: number) => void;
  };
}) {
  const checkpointLabel = useId();
  const outline = useMemo(() => sceneOutline(scene), [scene]);
  return (
    <Drawer>
      {checkpoints && (
        <DrawerSection sections={sections} id="scene.checkpoint" name={studioName('checkpoint')} bodyClassName="grid gap-1 p-1">
          <FieldRow label={studioName('checkpoint')} labelId={checkpointLabel}>
            <Select aria-labelledby={checkpointLabel} value={String(checkpoints.selected)}
              items={checkpoints.choices.map((coordinate) => ({ value: String(coordinate), label: String(coordinate) }))}
              onValueChange={(value) => checkpoints.onSelect(Number(value))} />
          </FieldRow>
        </DrawerSection>
      )}
      {outline.nodes.length > 0 && (
        <DrawerSection sections={sections} id="scene.outline" name={studioName('outline')}>
          <Tree key={outline.nodes.map(({ id }) => id).join(' ')} aria-label={studioName('outline')} nodes={outline.nodes}
            defaultExpanded={outline.nodes.map(({ id }) => id)}
            selected={selected && outline.addresses.has(selected) ? [selected] : []}
            onSelectedChange={(ids) => onSelect(outline.addresses.get(ids.at(-1) ?? '') ?? null)} />
        </DrawerSection>
      )}
    </Drawer>
  );
}

const coordinateFormat = new Intl.NumberFormat(undefined, { maximumFractionDigits: 4 });

/** The selected element's name and positions. */
export function SceneElementInspector({ scene, selected }: { scene: Scene; selected: string | null }) {
  const element = sceneElements(scene).find((item) => item.key === selected) ?? scene.groups.find((item) => item.key === selected);
  if (!element) return null;
  return (
    <Inspector name={element.name}>
      {'values' in element && element.values.map((value) => <ValueField key={value.label} model={rawField({
        id: `${element.key}/${value.label}`, label: value.label, type: { kind: 'readOnly' }, unit: studioUnit('meter'),
        value: value.point.map((coordinate) => coordinateFormat.format(coordinate)).join(', '),
      })} />)}
    </Inspector>
  );
}
