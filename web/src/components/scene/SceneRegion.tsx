import { useId, useLayoutEffect, useMemo, useRef, useState, type PointerEvent, type SVGProps } from 'react';
import { AudioWaveform, Boxes, Circle, Layers, MousePointer2, RectangleHorizontal, Spline, Variable, Waypoints } from 'lucide-react';
import { commandAttributes, type StudioCommand } from '@/features/projectWindow/studioCommands';
import { IconButton } from '@/components/ui/Button';
import { FieldRow } from '@/components/ui/FieldRow';
import { Inspector } from '@/components/ui/Inspector';
import { StatusMark } from '@/components/ui/StatusMark';
import { Tree, type TreeNode } from '@/components/ui/Tree';
import { ValueField } from '@/components/values/ValueField';
import { rawField } from '@/components/values/adapters/raw';
import { Drawer, DrawerSection, type DrawerSections } from '@/features/projectWindow/DrawerForms';
import { isModelAddress, type ElementAddress } from '@/features/projectWindow/elementSelection';
import type { Marker, StudioRelatePort } from '@/generated/studioContracts';
import { Menu, MenuCheckboxItem, MenuContent, MenuRadioGroup, MenuRadioItem, MenuSeparator, MenuTrigger } from '@/components/ui/Menu';
import { Select } from '@/components/ui/Select';
import { StudioRegion, StudioRegionBoundary } from '@/components/activity/StudioRegion';
import { studioAccessibleName, studioName, studioUnit } from '@/components/activity/studioVocabulary';
import { useElementSize } from '@/hooks/useElementSize';
import { WORKSPACE_SVG_HEIGHT, WORKSPACE_SVG_WIDTH, fitScale, niceScaleLength, sceneBounds, shapePoints, type ScenePoint } from './planarProjection';
import {
  fittedCamera, markMarkers, sceneElements, sceneLayers, zoomScene,
  type Scene, type SceneCamera, type SceneElement, type SceneEnd, type SceneEndPlace, type SceneLayer, type SceneMark, type SceneShape,
  type SceneTool, type SceneView,
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
 * places above the mechanisms, a variable's above a signal's, so that a goal cue held at the start leaves the start's
 * ring showing.
 */
const drawOrder: Record<SceneLayer, number> = { bodies: 0, muscles: 1, joints: 2, signals: 3, variables: 4 };

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

/** A marker centred at (x, y), `radius` across from its centre to its edge. */
export function MarkerShape({ marker, x, y, radius, ...props }: SVGProps<SVGElement> & {
  marker: Marker; x: number; y: number; radius: number;
}) {
  const drawn = props as SVGProps<SVGPathElement>;
  switch (marker.shape) {
    case 'circle': return <circle cx={x} cy={y} r={radius} {...props as SVGProps<SVGCircleElement>} />;
    case 'square': {
      const side = radius * 0.85;
      return <rect x={x - side} y={y - side} width={2 * side} height={2 * side} {...props as SVGProps<SVGRectElement>} />;
    }
    case 'triangle': {
      const [dx, dy] = [radius * Math.sqrt(3) / 2, radius / 2];
      return <path d={`M ${x} ${y - radius} L ${x + dx} ${y + dy} L ${x - dx} ${y + dy} Z`} strokeLinejoin="round" {...drawn} />;
    }
  }
}

function ElementDrawing({ element, active, project }: {
  element: SceneElement;
  active: boolean;
  project: (point: ScenePoint) => ScenePoint;
}) {
  const stroke = active ? 'var(--text-primary)' : 'var(--studio-scene-outline)';
  const possibleMarker = element.marker ? { ...element.marker, filled: false } : { shape: 'circle' as const, filled: false };
  const possible = element.possible?.map(project).map(([x, y], index) => <MarkerShape key={index} data-scene-possible
    marker={possibleMarker} x={x} y={y} radius={active ? 6 : 5} fill="transparent" stroke={stroke} strokeWidth={1.5}
    vectorEffect="non-scaling-stroke" />);
  const { shape } = element;
  if (!shape) return <>{possible}</>;
  if (shape.kind === 'point') {
    const [x, y] = project(shape.point);
    const colour = element.mark && `var(--studio-scene-${element.mark})`;
    return <>
      {possible}
      {element.marker && colour
        ? <MarkerShape data-scene-shown={`${element.marker.shape} ${element.marker.filled ? 'filled' : 'open'}`} marker={element.marker}
          x={x} y={y} radius={9} fill={element.marker.filled ? colour : 'transparent'} stroke={colour}
          strokeWidth={active ? 3 : 1.5} vectorEffect="non-scaling-stroke" />
        : <circle data-scene-shown cx={x} cy={y} r={active ? 7.5 : 6} fill="var(--text-primary)" />}
      <circle cx={x} cy={y} r={9} fill="transparent" />
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

/** Where a Relate ends besides another place: a fixed point of the world. */
export type SceneRelateEnd = SceneEndPlace | { kind: 'point'; position: ScenePoint };

/**
 * Writes the term relating `first` to `second`, or on `first` alone, a Scene place or a Model port; answers why when the
 * edit is refused.
 */
export type SceneRelate = (first: SceneEndPlace | StudioRelatePort, second?: SceneRelateEnd) => Promise<string | undefined>;

type SceneDrawingProps = SceneSelection & {
  camera: SceneCamera;
  onCameraChange: (camera: SceneCamera) => void;
  hiddenLayers: ReadonlySet<SceneLayer>;
  fitToSlot?: boolean;
  tool?: SceneTool;
  onRelate?: SceneRelate;
  /** The term whose relations are drawn heavier. */
  selectedTerm?: string;
};

/** A Relate drag: the end it started at, where the pointer is, and the end it is over. */
interface RelateDrag {
  from: SceneEnd;
  pointer: ScenePoint;
  over?: SceneEnd;
  moved: boolean;
}

/** How near, in canvas units, the pointer must be to a place for a drag to end at it. */
const endReach = 12;

function SceneDrawingCanvas({
  scene, camera, onCameraChange, hiddenLayers, selected, onSelect, fitToSlot = false, tool = 'select', onRelate, selectedTerm,
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
    ...scene.means.flatMap((path) => [...path.lower, ...path.upper]),
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
  const unproject = ([x, y]: ScenePoint): ScenePoint => [
    center[0] + (x - viewportWidth / 2 - camera.pan.x) / scale,
    center[1] - (y - viewportHeight / 2 - camera.pan.y) / scale,
  ];
  const relating = tool === 'relate' && onRelate !== undefined;
  const [drag, setDrag] = useState<RelateDrag>();
  const [refusal, setRefusal] = useState<{ end: string; reason: string }>();
  /** The pointer's place in the canvas's coordinates. */
  const canvasPoint = (event: PointerEvent<Element>, svg: SVGSVGElement): ScenePoint => {
    const rect = svg.getBoundingClientRect();
    const ratio = Math.min(rect.width / viewportWidth, rect.height / viewportHeight);
    return [
      (event.clientX - rect.left - (rect.width - viewportWidth * ratio) / 2) / ratio,
      (event.clientY - rect.top - (rect.height - viewportHeight * ratio) / 2) / ratio,
    ];
  };
  const endAt = (pointer: ScenePoint) => scene.ends.find((end) => {
    const [x, y] = project(end.point);
    return Math.hypot(x - pointer[0], y - pointer[1]) <= endReach;
  });
  const relate = (first: SceneEnd, second?: SceneRelateEnd) => {
    setRefusal(undefined);
    void onRelate?.(first.place, second).then((reason) => { if (reason) setRefusal({ end: first.key, reason }); });
  };
  const beginRelate = (event: PointerEvent<SVGGElement>, end: SceneEnd) => {
    if (event.button !== 0 || end.reason) return;
    event.stopPropagation();
    const svg = event.currentTarget.ownerSVGElement!;
    svg.setPointerCapture(event.pointerId);
    setDrag({ from: end, pointer: canvasPoint(event, svg), moved: false });
  };
  const meters = niceScaleLength(90 / Math.max(scale, 1));
  const drawn = sceneElements(scene)
    .filter((element) => elementPoints(element).length > 0 && !hiddenLayers.has(element.layer))
    .sort((left, right) => drawOrder[left.layer] - drawOrder[right.layer]);

  const beginPan = (event: PointerEvent<SVGSVGElement>) => {
    if (event.button !== 0 || drag) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const ratio = Math.min(rect.width / viewportWidth, rect.height / viewportHeight);
    if (!(ratio > 0)) return;
    pan.current = { start: [event.clientX, event.clientY], camera, ratio, moved: false };
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const movePan = (event: PointerEvent<SVGSVGElement>) => {
    if (drag) {
      const pointer = canvasPoint(event, event.currentTarget);
      const moved = drag.moved || Math.hypot(pointer[0] - project(drag.from.point)[0], pointer[1] - project(drag.from.point)[1]) > endReach;
      setDrag({ ...drag, pointer, moved, over: moved ? endAt(pointer) : undefined });
      return;
    }
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
        if (drag) {
          if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
          setDrag(undefined);
          if (!drag.moved) relate(drag.from);
          else if (!drag.over) relate(drag.from, { kind: 'point', position: unproject(drag.pointer) });
          else if (drag.over.key !== drag.from.key && !drag.over.reason) relate(drag.from, drag.over.place);
          return;
        }
        if (!pan.current) return;
        if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
        if (!pan.current.moved) onSelect(null);
        pan.current = null;
      }}
      onPointerCancel={() => { pan.current = null; setDrag(undefined); }}
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
          const stroke = track.picked ? 'var(--studio-scene-outline)' : 'var(--studio-scene-faint)';
          return <g key={index}>
            {track.drawn > 0 && <polyline data-scene-track={track.picked ? 'picked' : 'other'} fill="none" stroke={stroke}
              points={track.points.slice(0, track.drawn).map((point) => project(point).join(',')).join(' ')}
              strokeWidth={track.picked ? 1.5 : 1} strokeDasharray="0 3" strokeLinecap="round" vectorEffect="non-scaling-stroke" />}
            {track.head && <circle data-scene-cursor cx={project(track.head)[0]} cy={project(track.head)[1]} r={track.picked ? 3 : 1.5}
              fill={stroke} />}
            {track.ended.map(({ point, toward, by }, at) => {
              const [x, y] = project(point);
              const [tx, ty] = toward ? project(toward) : [x + 1, y];
              const length = Math.hypot(tx - x, ty - y) || 1;
              const [nx, ny] = [-(ty - y) / length * 5, (tx - x) / length * 5];
              return <line key={at} data-scene-ended={by} x1={x - nx} y1={y - ny} x2={x + nx} y2={y + ny} stroke={stroke}
                strokeWidth={by === 'condition' ? 2 : 1.5} strokeDasharray={by === 'limit' ? '2 2' : undefined} />;
            })}
          </g>;
        })}
      </g>
      <g pointerEvents="none" aria-hidden="true">
        {scene.means.map((path, index) => {
          const spread = path.lower.map((low, at) => {
            const [[x0, y0], [x1, y1]] = [project(low), project(path.upper[at])];
            return `M${x0},${y0}H${x1}V${y1}H${x0}Z`;
          }).join('');
          return <g key={index}>
            <path data-scene-spread d={spread} fill="var(--studio-scene-outline)" fillOpacity={0.12} stroke="none" />
            {path.drawn > 0 && <polyline data-scene-mean fill="none" stroke="var(--studio-scene-outline)"
              points={path.points.slice(0, path.drawn).map((point) => project(point).join(',')).join(' ')}
              strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />}
            {path.head && <circle data-scene-cursor cx={project(path.head)[0]} cy={project(path.head)[1]} r={3}
              fill="var(--studio-scene-outline)" />}
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
      <g pointerEvents="none" aria-hidden="true">
        {parallelRelations(scene.relations.map(({ term, ends: [a, b] }) => ({ term, ends: [project(a), project(b)] as const }))).map(({ term, ends }, index) => {
          const [[x1, y1], [x2, y2]] = ends;
          return <line key={index} data-scene-relation={term} x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--text-primary)"
            strokeWidth={term === selectedTerm ? 2.5 : 1.5} strokeDasharray="6 4" vectorEffect="non-scaling-stroke" />;
        })}
        {scene.places.map(({ term, point }, index) => {
          const [x, y] = project(point);
          const size = term === selectedTerm ? 6 : 4.5;
          return <path key={index} data-scene-place={term} d={`M ${x - size} ${y - size} L ${x + size} ${y + size} M ${x - size} ${y + size} L ${x + size} ${y - size}`}
            stroke="var(--text-primary)" strokeWidth={term === selectedTerm ? 2.5 : 1.5} vectorEffect="non-scaling-stroke" />;
        })}
      </g>
      {relating && <RelateEnds ends={scene.ends} project={project} drag={drag} refusal={refusal} onPointerDown={beginRelate}
        onRelate={(end) => relate(end)} />}
      <g transform={`translate(28 ${viewportHeight - 32})`} pointerEvents="none" aria-hidden="true">
        <path d={`M 0 -5 V 5 M 0 0 H ${meters * scale} M ${meters * scale} -5 V 5`} stroke="var(--text-primary)" strokeWidth={2} />
        <text y={-9} className="fill-shell-muted text-xs font-medium">
          {meters >= 1 ? `${meters} ${studioUnit('meter')}` : `${Math.round(meters * 100)} ${studioUnit('centimeter')}`}
        </text>
      </g>
    </svg>
  );
}

/**
 * Relate's places: each is pressable, and a drag from one marks every place it could end at, draws the line to the
 * pointer, and gives the reason beside a place it cannot end at. A refused edit's reason is beside the place it began.
 */
function RelateEnds({ ends, project, drag, refusal, onPointerDown, onRelate }: {
  ends: readonly SceneEnd[];
  project: (point: ScenePoint) => ScenePoint;
  drag: RelateDrag | undefined;
  refusal: { end: string; reason: string } | undefined;
  onPointerDown: (event: PointerEvent<SVGGElement>, end: SceneEnd) => void;
  onRelate: (end: SceneEnd) => void;
}) {
  const reasonBeside = (end: SceneEnd, reason: string) => {
    const [x, y] = project(end.point);
    return <text key={`${end.key} reason`} x={x + endReach + 4} y={y + 4} className="fill-shell-muted text-xs" pointerEvents="none">{reason}</text>;
  };
  const from = drag && project(drag.from.point);
  return <g data-scene-relate>
    {drag?.moved && from && <line x1={from[0]} y1={from[1]} x2={drag.pointer[0]} y2={drag.pointer[1]} stroke="var(--text-primary)"
      strokeWidth={1.5} strokeDasharray="6 4" pointerEvents="none" vectorEffect="non-scaling-stroke" />}
    {/* An anchor is drawn last, so that where a task place coincides with it, a press starts at the anchor. */}
    {[...ends].sort((left, right) => Number(left.place.kind === 'anchor') - Number(right.place.kind === 'anchor')).map((end) => {
      const [x, y] = project(end.point);
      const candidate = drag?.moved && !end.reason && end.key !== drag.from.key;
      return <g key={end.key} role="button" tabIndex={end.reason ? -1 : 0} aria-disabled={end.reason ? true : undefined}
        aria-label={studioAccessibleName('relate', end.name)} data-scene-end={end.key}
        className={end.reason ? 'cursor-not-allowed' : 'cursor-crosshair focus-visible:outline focus-visible:outline-2 focus-visible:outline-shell-accent'}
        onPointerDown={(event) => onPointerDown(event, end)}
        onKeyDown={(event) => {
          if (!end.reason && (event.key === 'Enter' || event.key === ' ')) {
            event.preventDefault();
            onRelate(end);
          }
        }}>
        <circle cx={x} cy={y} r={endReach} fill="transparent" />
        {candidate && <circle data-scene-candidate cx={x} cy={y} r={drag.over?.key === end.key ? endReach : endReach - 3}
          fill="none" stroke="var(--text-primary)" strokeWidth={drag.over?.key === end.key ? 2 : 1} vectorEffect="non-scaling-stroke" />}
        {end.reason && <circle cx={x} cy={y} r={endReach - 3} fill="none" stroke="var(--studio-scene-faint)" strokeWidth={1}
          strokeDasharray="2 2" vectorEffect="non-scaling-stroke" />}
      </g>;
    })}
    {drag?.over?.reason && reasonBeside(drag.over, drag.over.reason)}
    {!drag && refusal && ends.filter((end) => end.key === refusal.end).map((end) => reasonBeside(end, refusal.reason))}
  </g>;
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

/** The strip's tools, of which the one in use is pressed. */
export function SceneTools({ tool, commands }: { tool: SceneTool; commands: Readonly<Record<SceneTool, StudioCommand>> }) {
  const icons = { select: <MousePointer2 />, relate: <Spline /> } satisfies Record<SceneTool, unknown>;
  return <>{(['select', 'relate'] as const).map((id) => <IconButton key={id} label={commands[id].label} icon={icons[id]}
    {...commandAttributes(commands[id], 'right')} pressed={tool === id} disabled={!commands[id].enabled} onClick={commands[id].invoke} />)}</>;
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

const markerShapes = ['circle', 'square', 'triangle'] as const satisfies readonly Marker['shape'][];

/** A task element's marker as a row's glyph, in its mark's colour. */
function MarkerGlyph({ marker, mark }: { marker: Marker; mark: SceneMark }) {
  const colour = `var(--studio-scene-${mark})`;
  return <svg viewBox="0 0 16 16" aria-hidden data-marker={`${marker.shape} ${marker.filled ? 'filled' : 'open'}`}>
    <MarkerShape marker={marker} x={8} y={8.5} radius={6} fill={marker.filled ? colour : 'none'} stroke={colour} strokeWidth={1.5} />
  </svg>;
}

/** How the Scene's list changes a task variable's marker. */
export interface SceneMarking {
  /** The task document whose variables it changes. */
  readonly document: string;
  /** Stores the variable's marker, or none so that it is drawn by its mark's; answers why when the edit is refused. */
  mark(variable: string, marker: Marker | null): Promise<string | undefined>;
}

/** A variable's id, when the element is a variable of the task document `marking` changes. */
function markedVariable(element: SceneElement, marking: SceneMarking | undefined) {
  const { address } = element;
  if (!marking || isModelAddress(address) || address.document !== marking.document) return undefined;
  const [kind, ...id] = address.element.id.split(':');
  return kind === 'variable' ? id.join(':') : undefined;
}

/** The menu on a variable's marker: its shape, and whether it is filled. Its mark's marker is stored as none. */
function MarkerChoice({ element, marker, mark, choose }: {
  element: SceneElement; marker: Marker; mark: SceneMark; choose(marker: Marker | null): void;
}) {
  const change = (next: Marker) => {
    const own = markMarkers[mark];
    choose(next.shape === own.shape && next.filled === own.filled ? null : next);
  };
  return <span className="contents" onClick={(event) => event.stopPropagation()} onDoubleClick={(event) => event.stopPropagation()}
    onKeyDown={(event) => event.stopPropagation()}>
    <Menu>
      <MenuTrigger render={<IconButton size="mark" label={studioAccessibleName('marker', element.name)} icon={<MarkerGlyph marker={marker} mark={mark} />} tipPlacement="right" />} />
      <MenuContent side="right" aria-label={studioAccessibleName('marker', element.name)}>
        <MenuRadioGroup value={marker.shape} onValueChange={(shape: Marker['shape']) => change({ ...marker, shape })}>
          {markerShapes.map((shape) => <MenuRadioItem key={shape} value={shape}
            glyph={<MarkerGlyph marker={{ shape, filled: marker.filled }} mark={mark} />}>{studioName(shape)}</MenuRadioItem>)}
        </MenuRadioGroup>
        <MenuSeparator />
        <MenuCheckboxItem checked={marker.filled} closes onCheckedChange={(filled) => change({ ...marker, filled })}>
          {studioName('filled')}
        </MenuCheckboxItem>
      </MenuContent>
    </Menu>
  </span>;
}

type MarkerRefusal = { readonly key: string; readonly reason: string };

/**
 * Each mechanism with its bodies, joints and muscles, and the task with its variables and signals that have a place,
 * in Layers' order, as rows keyed by their element's key. A task element's row ends in its marker, which a variable of
 * the task document `marking` changes chooses from; a refused choice is marked on its row.
 */
function sceneOutline(scene: Scene, marking: SceneMarking | undefined, refusal: MarkerRefusal | undefined,
  choose: (element: SceneElement, variable: string, marker: Marker | null) => void) {
  const addresses = new Map<string, ElementAddress>();
  const nodes = scene.groups.map((group): TreeNode => {
    addresses.set(group.key, group.address);
    const children = sceneLayers.flatMap((layer) => group.elements.filter((element) => element.layer === layer)).map((element): TreeNode => {
      addresses.set(element.key, element.address);
      const variable = markedVariable(element, marking);
      const { marker, mark } = element;
      const shown = marker && mark && (variable === undefined ? <MarkerGlyph marker={marker} mark={mark} />
        : <MarkerChoice element={element} marker={marker} mark={mark} choose={(next) => choose(element, variable, next)} />);
      const problem = refusal?.key === element.key ? refusal.reason : element.problem;
      return { id: element.key, name: element.name, glyph: layerGlyphs[element.layer], trailing: <>{outlineMark(problem)}{shown}</> };
    });
    return { id: group.key, name: group.name, glyph: <Boxes />, trailing: outlineMark(group.problem), children: children.length ? children : undefined };
  });
  return { nodes, addresses };
}

export function SceneDrawer({ sections, scene, selected, onSelect, checkpoints, marking }: SceneSelection & {
  sections: DrawerSections;
  scene: Scene;
  checkpoints?: {
    choices: readonly number[];
    selected: number;
    onSelect: (checkpoint: number) => void;
  };
  marking?: SceneMarking;
}) {
  const checkpointLabel = useId();
  const [refusal, setRefusal] = useState<MarkerRefusal>();
  const outline = useMemo(() => sceneOutline(scene, marking, refusal, (element, variable, marker) => {
    setRefusal(undefined);
    void marking?.mark(variable, marker).then((reason) => { if (reason) setRefusal({ key: element.key, reason }); });
  }), [scene, marking, refusal]);
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

type Segment = readonly [ScenePoint, ScenePoint];

/** Pixels between relations drawn between the same two places. */
const relationGap = 6;

/** Relations between the same two places, side by side across their common line rather than drawn over each other. */
function parallelRelations<T extends { ends: Segment }>(relations: readonly T[]): T[] {
  const key = ([a, b]: Segment) => [a, b].map(([x, y]) => `${Math.round(x)},${Math.round(y)}`).sort().join(' ');
  const groups = new Map<string, number[]>();
  relations.forEach((relation, index) => groups.set(key(relation.ends), [...groups.get(key(relation.ends)) ?? [], index]));
  return relations.map((relation, index) => {
    const group = groups.get(key(relation.ends))!;
    const [[x1, y1], [x2, y2]] = relation.ends;
    const length = Math.hypot(x2 - x1, y2 - y1);
    if (group.length < 2 || !length) return relation;
    const shift = (group.indexOf(index) - (group.length - 1) / 2) * relationGap;
    const [dx, dy] = [(y1 - y2) / length * shift, (x2 - x1) / length * shift];
    return { ...relation, ends: [[x1 + dx, y1 + dy], [x2 + dx, y2 + dy]] satisfies Segment };
  });
}
