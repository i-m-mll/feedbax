import { useEffect, useMemo, useRef, useState } from 'react';
import type { KeyboardEvent, MouseEvent, PointerEvent, WheelEvent } from 'react';
import clsx from 'clsx';
import {
  Database,
  FoldVertical,
  LocateFixed,
  UnfoldVertical,
  GitBranch,
  ListChecks,
  Map as MapIcon,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  Settings2,
  Trash2,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import { Canvas } from '@/components/canvas/Canvas';
import {
  retainedObservableEntityId,
  buildScenarioEntityRegistry,
} from '@/features/scenario/entities';
import {
  buildResolvedScene,
  objectiveProjectionItems,
  relatedProjectionItems,
  type ResolvedScene,
  type ResolvedSceneElement,
  type ResolvedSceneEntity,
} from '@/features/scenario/projections';
import {
  ensureObjectiveSpec,
  OBJECTIVE_PENALTY_OPTIONS,
  OBJECTIVE_TEMPORAL_MODE_OPTIONS,
  objectiveTermEnabled,
  removeObjectiveTerm,
  setObjectiveTermEnabled,
  updateObjectiveTerm,
} from '@/features/scenario/objectives';
import {
  selectorDetail,
  selectorDisplayLabel,
  selectorGroupLabel,
  selectorOptionsForRegistry,
  type StudioSelectorOption,
} from '@/features/scenario/selectors';
import {
  createRetainedObservable,
  retainedObservableSelectorPatch,
  retainedObservableTargetKindLabel,
  selectorToRetainedObservableTarget,
} from '@/features/scenario/observables';
import { useComponents } from '@/hooks/useComponents';
import { useGraphStore } from '@/stores/graphStore';
import {
  getActiveStage,
  getScenario,
  getTopPaneState,
  useWorkspaceStore,
} from '@/stores/workspaceStore';
import { useLayoutStore } from '@/stores/layoutStore';
import { useStudioSchemaRegistry } from '@/hooks/useStudioSchemas';
import type {
  RetainedObservableSpec,
} from '@/types/graph';
import type {
  StudioObjectiveSpec,
  StudioObjectiveTermSpec,
  StudioScenarioEntityRegistry,
  StudioSchemaRegistry,
  StudioTopPaneProjection,
} from '@/types/workspace';
import type { TimeAggregationSpec } from '@/types/training';

const PROJECTIONS: Array<{
  id: StudioTopPaneProjection;
  label: string;
  icon: typeof GitBranch;
}> = [
  { id: 'model', label: 'Model', icon: GitBranch },
  { id: 'task', label: 'Task', icon: Settings2 },
  { id: 'workspace', label: 'Workspace', icon: MapIcon },
  { id: 'observables', label: 'Observables', icon: Database },
  { id: 'objectives', label: 'Objectives', icon: ListChecks },
];

const WORKSPACE_SVG_WIDTH = 860;
const WORKSPACE_SVG_HEIGHT = 520;
const WORKSPACE_PADDING = 56;

type ScenePoint = [number, number];

interface SceneBounds {
  min: ScenePoint;
  max: ScenePoint;
}

function includePoint(bounds: SceneBounds, point: ScenePoint) {
  bounds.min = [Math.min(bounds.min[0], point[0]), Math.min(bounds.min[1], point[1])];
  bounds.max = [Math.max(bounds.max[0], point[0]), Math.max(bounds.max[1], point[1])];
}

function sceneBounds(scene: ResolvedScene): SceneBounds {
  const bounds: SceneBounds = {
    min: [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY],
    max: [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY],
  };

  for (const anchor of scene.anchors) {
    if (anchor.position) includePoint(bounds, anchor.position);
  }
  for (const element of scene.elements) {
    if (element.geometry.kind === 'polyline' || element.geometry.kind === 'points' || element.geometry.kind === 'link') {
      for (const point of element.geometry.points) includePoint(bounds, point);
    } else if (element.geometry.kind === 'bounds') {
      includePoint(bounds, element.geometry.min);
      includePoint(bounds, element.geometry.max);
    }
  }

  if (!Number.isFinite(bounds.min[0]) || !Number.isFinite(bounds.min[1])) {
    return { min: [-1, -1], max: [1, 1] };
  }

  const width = Math.max(bounds.max[0] - bounds.min[0], 0.25);
  const height = Math.max(bounds.max[1] - bounds.min[1], 0.25);
  const pad = Math.max(width, height) * 0.12;
  return {
    min: [bounds.min[0] - pad, bounds.min[1] - pad],
    max: [bounds.max[0] + pad, bounds.max[1] + pad],
  };
}

function fitScale(bounds: SceneBounds): number {
  const width = Math.max(bounds.max[0] - bounds.min[0], 0.1);
  const height = Math.max(bounds.max[1] - bounds.min[1], 0.1);
  return Math.min(
    (WORKSPACE_SVG_WIDTH - WORKSPACE_PADDING * 2) / width,
    (WORKSPACE_SVG_HEIGHT - WORKSPACE_PADDING * 2) / height
  );
}

function niceScaleLength(targetMeters: number): number {
  if (!Number.isFinite(targetMeters) || targetMeters <= 0) return 0.1;
  const exponent = Math.floor(Math.log10(targetMeters));
  const base = 10 ** exponent;
  const normalized = targetMeters / base;
  const step = normalized >= 5 ? 5 : normalized >= 2 ? 2 : 1;
  return step * base;
}

export function ScenarioProjectionToolbar({ availableHeight }: { availableHeight: number }) {
  const workspace = useWorkspaceStore((state) => state.workspace);
  const setTopPaneProjection = useWorkspaceStore((state) => state.setTopPaneProjection);
  const {
    leftSidebarVisible,
    rightSidebarVisible,
    topCollapsed,
    toggleLeftSidebar,
    toggleRightSidebar,
    toggleTop,
  } = useLayoutStore();
  const topPane = getTopPaneState(workspace);
  const LeftIcon = leftSidebarVisible ? PanelLeftClose : PanelLeftOpen;
  const RightIcon = rightSidebarVisible ? PanelRightClose : PanelRightOpen;
  const TopIcon = topCollapsed ? UnfoldVertical : FoldVertical;

  return (
    <div className="flex h-11 shrink-0 items-end justify-between border-b border-slate-200 bg-white px-3">
      <div className="flex h-full min-w-0 items-end">
        <button
          type="button"
          onClick={toggleLeftSidebar}
          className="mb-1 mr-2 inline-flex h-8 w-8 items-center justify-center rounded-md text-slate-400 hover:bg-slate-100 hover:text-slate-700"
          title={leftSidebarVisible ? 'Hide component palette' : 'Show component palette'}
        >
          <LeftIcon className="h-4 w-4" />
        </button>
        {PROJECTIONS.map((projection) => {
          const Icon = projection.icon;
          const selected = projection.id === topPane.active_projection;
          return (
            <button
              key={projection.id}
              type="button"
              onClick={() => {
                setTopPaneProjection(projection.id);
                if (topCollapsed) toggleTop(availableHeight);
              }}
              className={clsx(
                'inline-flex h-10 items-center gap-2 border-b-2 px-4 text-xs font-semibold uppercase tracking-[0.12em] transition-colors',
                selected
                  ? 'border-brand-500 text-brand-600'
                  : 'border-transparent text-slate-400 hover:text-slate-600'
              )}
            >
              <Icon className="h-3.5 w-3.5" />
              {projection.label}
            </button>
          );
        })}
      </div>
      <div className="flex h-full shrink-0 items-end gap-1 pb-1">
        <button
          type="button"
          onClick={() => toggleTop(availableHeight)}
          className="inline-flex h-8 w-8 items-center justify-center rounded-md text-slate-400 hover:bg-slate-100 hover:text-slate-700"
          title={topCollapsed ? 'Expand top pane' : 'Collapse top pane'}
        >
          <TopIcon className="h-4 w-4" />
        </button>
        <button
          type="button"
          onClick={toggleRightSidebar}
          className="inline-flex h-8 w-8 items-center justify-center rounded-md text-slate-400 hover:bg-slate-100 hover:text-slate-700"
          title={rightSidebarVisible ? 'Hide properties panel' : 'Show properties panel'}
        >
          <RightIcon className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}

function ScenarioBadge({
  stageLabel,
  scenarioLabel,
  summary,
}: {
  stageLabel: string | null;
  scenarioLabel: string | null;
  summary: string | null;
}) {
  if (!stageLabel && !scenarioLabel) return null;
  return (
    <div className="pointer-events-none absolute bottom-4 left-20 z-10 max-w-[min(28rem,calc(100%-6rem))] rounded border border-slate-200 bg-white/90 px-3 py-2 shadow-sm backdrop-blur">
      <div className="truncate text-sm font-semibold text-slate-800">
        {scenarioLabel ?? stageLabel}
      </div>
      {summary && <div className="mt-0.5 truncate text-xs text-slate-500">{summary}</div>}
    </div>
  );
}

function WorkspaceProjection({
  registry,
  scene,
  selectedId,
  onSelect,
}: {
  registry: StudioScenarioEntityRegistry;
  scene: ResolvedScene;
  selectedId: string | null;
  onSelect: (entityId: string | null) => void;
}) {
  const [hoveredEntityId, setHoveredEntityId] = useState<string | null>(null);
  const [view, setView] = useState({ zoom: 1, pan: { x: 0, y: 0 } });
  const dragStartRef = useRef<{ x: number; y: number; pan: { x: number; y: number } } | null>(
    null
  );
  const relatedItems = relatedProjectionItems(registry, selectedId);
  const relatedIds = new Set(relatedItems.map((item) => item.entity_id));
  const hoveredRelatedIds = new Set(
    hoveredEntityId
      ? [
          ...(scene.entities.find((entity) => entity.id === hoveredEntityId)?.related_entity_ids ?? []),
          hoveredEntityId,
        ]
      : []
  );
  const bounds = sceneBounds(scene);
  const baseScale = fitScale(bounds);
  const scale = baseScale * view.zoom;
  const center: ScenePoint = [
    (bounds.min[0] + bounds.max[0]) / 2,
    (bounds.min[1] + bounds.max[1]) / 2,
  ];
  const project = (point: ScenePoint): ScenePoint => [
    WORKSPACE_SVG_WIDTH / 2 + (point[0] - center[0]) * scale + view.pan.x,
    WORKSPACE_SVG_HEIGHT / 2 - (point[1] - center[1]) * scale + view.pan.y,
  ];
  const scaleMeters = niceScaleLength(90 / Math.max(scale, 1));
  const scalePixels = scaleMeters * scale;
  const warningCount = scene.validation.filter((message) => message.severity === 'warning').length;

  const entityActive = (entity: Pick<ResolvedSceneEntity, 'id' | 'related_entity_ids'>) =>
    entity.id === selectedId ||
    relatedIds.has(entity.id) ||
    hoveredRelatedIds.has(entity.id) ||
    entity.related_entity_ids.some((id) => hoveredRelatedIds.has(id));

  const elementActive = (element: ResolvedSceneElement) => {
    const entity = scene.entities.find((candidate) => candidate.id === element.entity_id);
    return entity ? entityActive(entity) : false;
  };

  const zoomBy = (factor: number) => {
    setView((current) => ({
      ...current,
      zoom: Math.max(0.35, Math.min(8, current.zoom * factor)),
    }));
  };

  const resetView = () => setView({ zoom: 1, pan: { x: 0, y: 0 } });

  const beginPan = (event: PointerEvent<SVGSVGElement>) => {
    if (event.button !== 0) return;
    dragStartRef.current = {
      x: event.clientX,
      y: event.clientY,
      pan: view.pan,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const movePan = (event: PointerEvent<SVGSVGElement>) => {
    const start = dragStartRef.current;
    if (!start) return;
    setView((current) => ({
      ...current,
      pan: {
        x: start.pan.x + event.clientX - start.x,
        y: start.pan.y + event.clientY - start.y,
      },
    }));
  };

  const endPan = () => {
    dragStartRef.current = null;
  };

  const wheelZoom = (event: WheelEvent<SVGSVGElement>) => {
    event.preventDefault();
    zoomBy(event.deltaY > 0 ? 0.9 : 1.1);
  };

  const renderElement = (element: ResolvedSceneElement) => {
    const active = elementActive(element);
    const stroke = active ? '#0f766e' : '#64748b';
    const fill = active ? '#ccfbf1' : '#f8fafc';
    const markerRadius = element.scale_invariant ? 7 : Math.max(4, Math.min(18, scale * 0.02));
    const commonProps = {
      onMouseEnter: () => setHoveredEntityId(element.entity_id),
      onMouseLeave: () => setHoveredEntityId(null),
      onClick: (event: MouseEvent) => {
        event.stopPropagation();
        onSelect(element.entity_id);
      },
      className: 'cursor-pointer outline-none',
      role: 'button',
      tabIndex: 0,
      onKeyDown: (event: KeyboardEvent) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          onSelect(element.entity_id);
        }
      },
    };

    if (element.geometry.kind === 'bounds') {
      const [x0, y0] = project(element.geometry.min);
      const [x1, y1] = project(element.geometry.max);
      return (
        <g key={element.id} {...commonProps}>
          <rect
            x={Math.min(x0, x1)}
            y={Math.min(y0, y1)}
            width={Math.abs(x1 - x0)}
            height={Math.abs(y1 - y0)}
            fill={active ? '#dcfce7' : '#f0fdf4'}
            stroke={active ? '#16a34a' : '#86efac'}
            strokeWidth={1.5}
            vectorEffect="non-scaling-stroke"
          />
        </g>
      );
    }

    if (element.geometry.kind === 'polyline') {
      const geometry = element.geometry;
      const points = geometry.points.map(project).map((point) => point.join(',')).join(' ');
      return (
        <g key={element.id} {...commonProps}>
          <polyline
            points={points}
            fill="none"
            stroke={active ? '#b45309' : stroke}
            strokeWidth={active ? 8 : 6}
            strokeLinecap="round"
            strokeLinejoin="round"
            vectorEffect="non-scaling-stroke"
          />
          {geometry.points.map((point, index) => {
            const [x, y] = project(point);
            return (
              <circle
                key={`${element.id}:joint:${index}`}
                cx={x}
                cy={y}
                r={index === geometry.points.length - 1 ? markerRadius : 4.5}
                fill={index === geometry.points.length - 1 ? '#475569' : '#ffffff'}
                stroke={active ? '#92400e' : '#64748b'}
                strokeWidth={1.5}
                vectorEffect="non-scaling-stroke"
              />
            );
          })}
        </g>
      );
    }

    if (element.geometry.kind === 'points') {
      const glyph = String(element.style.glyph ?? element.metadata.glyph ?? '');
      return (
        <g key={element.id} {...commonProps}>
          {element.geometry.points.map((point, index) => {
            const [x, y] = project(point);
            const target = glyph === 'target' || element.metadata.canonical_goal === true;
            return (
              <g key={`${element.id}:point:${index}`}>
                {target ? (
                  <>
                    <circle
                      cx={x}
                      cy={y}
                      r={markerRadius + 3}
                      fill="none"
                      stroke={active ? '#059669' : '#34d399'}
                      strokeWidth={1.5}
                      vectorEffect="non-scaling-stroke"
                    />
                    <circle
                      cx={x}
                      cy={y}
                      r={markerRadius * 0.45}
                      fill={active ? '#059669' : '#a7f3d0'}
                    />
                  </>
                ) : (
                  <circle
                    cx={x}
                    cy={y}
                    r={markerRadius}
                    fill={fill}
                    stroke={stroke}
                    strokeWidth={1.5}
                    vectorEffect="non-scaling-stroke"
                  />
                )}
              </g>
            );
          })}
        </g>
      );
    }

    if (element.geometry.kind === 'link') {
      const points = element.geometry.points.map(project);
      if (points.length < 2) return null;
      return (
        <g key={element.id} {...commonProps}>
          <line
            x1={points[0][0]}
            y1={points[0][1]}
            x2={points[1][0]}
            y2={points[1][1]}
            stroke={active ? '#14b8a6' : '#cbd5e1'}
            strokeWidth={2}
            strokeDasharray="5 6"
            vectorEffect="non-scaling-stroke"
          />
        </g>
      );
    }

    return null;
  };

  return (
    <div className="grid h-full min-h-0 grid-cols-[minmax(0,1fr)_17rem] bg-slate-50">
      <div className="relative min-h-0 overflow-hidden">
        <svg
          viewBox={`0 0 ${WORKSPACE_SVG_WIDTH} ${WORKSPACE_SVG_HEIGHT}`}
          className="h-full w-full touch-none bg-white"
          role="img"
          aria-label="Workspace projection"
          onPointerDown={beginPan}
          onPointerMove={movePan}
          onPointerUp={endPan}
          onPointerCancel={endPan}
          onWheel={wheelZoom}
          onClick={() => onSelect(null)}
        >
          <defs>
            <pattern id="workspace-grid" width="32" height="32" patternUnits="userSpaceOnUse">
              <path d="M 32 0 L 0 0 0 32" fill="none" stroke="#eef2f7" strokeWidth="1" />
            </pattern>
          </defs>
          <rect width={WORKSPACE_SVG_WIDTH} height={WORKSPACE_SVG_HEIGHT} fill="url(#workspace-grid)" />
          <line
            x1={project([bounds.min[0], 0])[0]}
            y1={project([0, 0])[1]}
            x2={project([bounds.max[0], 0])[0]}
            y2={project([0, 0])[1]}
            stroke="#e2e8f0"
            strokeWidth="1"
            vectorEffect="non-scaling-stroke"
          />
          <line
            x1={project([0, 0])[0]}
            y1={project([0, bounds.min[1]])[1]}
            x2={project([0, 0])[0]}
            y2={project([0, bounds.max[1]])[1]}
            stroke="#e2e8f0"
            strokeWidth="1"
            vectorEffect="non-scaling-stroke"
          />
          {scene.elements.map(renderElement)}
          <g transform={`translate(28 ${WORKSPACE_SVG_HEIGHT - 32})`}>
            <line
              x1="0"
              y1="0"
              x2={scalePixels}
              y2="0"
              stroke="#334155"
              strokeWidth="2"
              vectorEffect="non-scaling-stroke"
            />
            <line x1="0" y1="-5" x2="0" y2="5" stroke="#334155" strokeWidth="2" />
            <line x1={scalePixels} y1="-5" x2={scalePixels} y2="5" stroke="#334155" strokeWidth="2" />
            <text x="0" y="-9" className="fill-slate-600 text-[11px] font-medium">
              {scaleMeters >= 1 ? `${scaleMeters} m` : `${Math.round(scaleMeters * 100)} cm`}
            </text>
          </g>
        </svg>
        <div className="absolute left-4 top-4 flex h-9 items-center gap-1 rounded border border-slate-200 bg-white/90 px-1.5 shadow-sm backdrop-blur">
          <button
            type="button"
            onClick={() => zoomBy(1.2)}
            className="inline-flex h-7 w-7 items-center justify-center rounded text-slate-500 hover:bg-slate-100 hover:text-slate-800"
            title="Zoom in"
          >
            <ZoomIn className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={() => zoomBy(0.85)}
            className="inline-flex h-7 w-7 items-center justify-center rounded text-slate-500 hover:bg-slate-100 hover:text-slate-800"
            title="Zoom out"
          >
            <ZoomOut className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={resetView}
            className="inline-flex h-7 w-7 items-center justify-center rounded text-slate-500 hover:bg-slate-100 hover:text-slate-800"
            title="Reset view"
          >
            <LocateFixed className="h-4 w-4" />
          </button>
        </div>
        <div className="pointer-events-none absolute bottom-4 right-4 rounded border border-slate-200 bg-white/90 px-3 py-2 text-xs text-slate-600 shadow-sm backdrop-blur">
          <div className="font-semibold text-slate-800">World frame</div>
          <div className="mt-0.5">meters, y-up · {scene.frame}</div>
        </div>
      </div>
      <div className="min-h-0 overflow-y-auto border-l border-slate-200 bg-white">
        <div className="sticky top-0 z-10 border-b border-slate-200 bg-white px-3 py-2">
          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">
            Scene
          </div>
          <div className="mt-1 text-xs text-slate-500">
            {scene.entities.length} entities · {warningCount} warnings
          </div>
        </div>
        <div className="divide-y divide-slate-100">
          {scene.entities.map((entity) => {
            const active = entity.id === selectedId;
            const related = relatedIds.has(entity.id) || hoveredRelatedIds.has(entity.id);
            return (
              <button
                key={entity.id}
                type="button"
                onClick={() => onSelect(entity.id)}
                onMouseEnter={() => setHoveredEntityId(entity.id)}
                onMouseLeave={() => setHoveredEntityId(null)}
                className={clsx(
                  'block w-full px-3 py-2.5 text-left text-xs transition-colors',
                  active
                    ? 'bg-brand-50 text-slate-900'
                    : related
                      ? 'bg-teal-50/70 text-slate-800'
                      : 'bg-white text-slate-600 hover:bg-slate-50'
                )}
              >
                <div className="truncate font-medium">{entity.label}</div>
                <div className="mt-0.5 truncate text-[11px] text-slate-400">
                  {entity.summary ?? entity.kind}
                </div>
              </button>
            );
          })}
        </div>
        {scene.validation.length > 0 && (
          <div className="border-t border-slate-200 px-3 py-2">
            <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-400">
              Validation
            </div>
            <div className="mt-2 space-y-1.5">
              {scene.validation.slice(0, 4).map((message, index) => (
                <div key={`${message.type}:${index}`} className="text-xs text-amber-600">
                  {message.message}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function temporalSelector(term: StudioObjectiveTermSpec): TimeAggregationSpec {
  const value = term.temporal_selector;
  if (!value || typeof value !== 'object' || !('mode' in value)) return { mode: 'all' };
  return value as TimeAggregationSpec;
}

function updateTemporalSelector(
  term: StudioObjectiveTermSpec,
  updates: Partial<TimeAggregationSpec>
): TimeAggregationSpec {
  const current = temporalSelector(term);
  return {
    ...current,
    ...updates,
  };
}

function optionLabel(option: StudioSelectorOption): string {
  return `${selectorGroupLabel(option.group)} / ${option.label}`;
}

function ObservableSelectorSelect({
  value,
  options,
  onChange,
  className,
}: {
  value: string | null | undefined;
  options: StudioSelectorOption[];
  onChange: (option: StudioSelectorOption | null) => void;
  className?: string;
}) {
  return (
    <select
      value={value ?? ''}
      onChange={(event) => {
        const option = options.find((candidate) => candidate.selector.compact === event.target.value);
        onChange(option ?? null);
      }}
      className={clsx('h-8 rounded border border-slate-200 bg-white px-2 text-xs', className)}
    >
      <option value="">Select source</option>
      {options.map((option) => (
        <option key={option.id} value={option.selector.compact}>
          {optionLabel(option)}
        </option>
      ))}
    </select>
  );
}

function ObservablesProjection({
  registry,
  selectedId,
  graph,
  objectiveSpec,
  schemaRegistry,
  onSelect,
  onAdd,
  onUpdate,
  onRemove,
}: {
  registry: StudioScenarioEntityRegistry;
  selectedId: string | null;
  graph: { retained_observables?: RetainedObservableSpec[] | null };
  objectiveSpec: StudioObjectiveSpec;
  schemaRegistry: StudioSchemaRegistry | null;
  onSelect: (entityId: string | null) => void;
  onAdd: (observable: RetainedObservableSpec) => void;
  onUpdate: (observableId: string, updates: Partial<RetainedObservableSpec>) => void;
  onRemove: (observableId: string) => void;
}) {
  const observables = graph.retained_observables ?? [];
  const selectorOptions = useMemo(
    () => selectorOptionsForRegistry({ registry, schemaRegistry, objectiveSpec }),
    [objectiveSpec, registry, schemaRegistry]
  );
  const captureOptions = useMemo(
    () =>
      selectorOptions.filter((option) => {
        if (option.selector.namespace === 'retained_observable') return false;
        if (option.selector.namespace === 'probe') return false;
        return selectorToRetainedObservableTarget(option.selector) !== null;
      }),
    [selectorOptions]
  );
  const [draftSelector, setDraftSelector] = useState<string>(() => captureOptions[0]?.selector.compact ?? '');

  useEffect(() => {
    if (!draftSelector && captureOptions[0]) {
      setDraftSelector(captureOptions[0].selector.compact);
    }
  }, [captureOptions, draftSelector]);

  const addObservable = () => {
    const option =
      captureOptions.find((candidate) => candidate.selector.compact === draftSelector) ??
      captureOptions[0];
    if (!option) return;
    const observable = createRetainedObservable({
      selector: option.selector,
      existingIds: new Set(observables.map((item) => item.id)),
    });
    if (!observable) return;
    onAdd(observable);
    onSelect(retainedObservableEntityId(observable.id));
  };

  return (
    <div className="h-full overflow-y-auto bg-slate-50 p-5">
      <div className="mx-auto max-w-6xl space-y-4">
        <div className="rounded-md border border-slate-200 bg-white p-4">
          <div className="grid grid-cols-[minmax(12rem,1fr)_9rem] gap-3">
            <ObservableSelectorSelect
              value={draftSelector}
              options={captureOptions}
              onChange={(option) => setDraftSelector(option?.selector.compact ?? '')}
              className="w-full"
            />
            <button
              type="button"
              onClick={addObservable}
              disabled={captureOptions.length === 0}
              className="inline-flex h-8 items-center justify-center rounded-md bg-slate-900 px-3 text-xs font-medium text-white hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              Add capture
            </button>
          </div>
        </div>

        <div className="overflow-hidden rounded-md border border-slate-200 bg-white">
          <div className="grid grid-cols-[minmax(10rem,1fr)_8rem_minmax(12rem,1.2fr)_4rem] border-b border-slate-200 bg-slate-50 px-4 py-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-400">
            <div>Observable</div>
            <div>Kind</div>
            <div>Source</div>
            <div />
          </div>
          {observables.map((observable) => {
            const active = selectedId === retainedObservableEntityId(observable.id);
            const source = observable.selector ?? observable.target?.selector ?? '';
            return (
              <div
                key={observable.id}
                onClick={() => onSelect(retainedObservableEntityId(observable.id))}
                className={clsx(
                  'grid grid-cols-[minmax(10rem,1fr)_8rem_minmax(12rem,1.2fr)_4rem] items-center gap-2 border-b border-slate-100 px-4 py-3 text-xs last:border-b-0',
                  active ? 'bg-brand-50 text-slate-900' : 'bg-white text-slate-600 hover:bg-slate-50'
                )}
              >
                <div className="min-w-0">
                  <input
                    value={observable.label ?? observable.id}
                    onChange={(event) => onUpdate(observable.id, { label: event.target.value })}
                    onClick={(event) => event.stopPropagation()}
                    className="h-8 w-full rounded border border-transparent bg-transparent px-2 font-medium text-slate-800 hover:border-slate-200 focus:border-brand-300 focus:bg-white focus:outline-none"
                  />
                </div>
                <div className="text-slate-500">
                  {retainedObservableTargetKindLabel(observable.target)}
                </div>
                <ObservableSelectorSelect
                  value={source}
                  options={captureOptions}
                  onChange={(option) => {
                    if (!option) return;
                    const patch = retainedObservableSelectorPatch(option.selector);
                    if (patch) onUpdate(observable.id, patch);
                  }}
                  className="w-full"
                />
                <button
                  type="button"
                  onClick={(event) => {
                    event.stopPropagation();
                    onRemove(observable.id);
                    if (active) onSelect(null);
                  }}
                  className="rounded p-1 text-slate-400 hover:bg-red-50 hover:text-red-600"
                  title="Delete retained observable"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            );
          })}
          {observables.length === 0 && (
            <div className="px-4 py-8 text-center text-sm text-slate-400">
              No explicit retained observables authored.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ObjectivesProjection({
  registry,
  selectedId,
  objectiveSpec,
  onSelect,
  onObjectiveSpecChange,
}: {
  registry: StudioScenarioEntityRegistry;
  selectedId: string | null;
  objectiveSpec: StudioObjectiveSpec;
  onSelect: (entityId: string | null) => void;
  onObjectiveSpecChange: (spec: StudioObjectiveSpec) => void;
}) {
  const items = objectiveProjectionItems(registry);
  const relatedItems = relatedProjectionItems(registry, selectedId);
  const relatedIds = new Set(relatedItems.map((item) => item.entity_id));
  const termByEntityId = new Map(
    objectiveSpec.terms.map((term) => [`objective_term:${term.id}`, term])
  );

  const updateTerm = (termId: string, updates: Partial<StudioObjectiveTermSpec>) => {
    onObjectiveSpecChange(updateObjectiveTerm(objectiveSpec, termId, updates));
  };

  return (
    <div className="h-full overflow-y-auto bg-slate-50 p-5">
      <div className="mx-auto max-w-5xl overflow-hidden rounded-md border border-slate-200 bg-white">
        <div className="grid grid-cols-[minmax(10rem,1.4fr)_6.5rem_5.5rem_7.25rem_8.5rem_minmax(8rem,1fr)_4rem] border-b border-slate-200 bg-slate-50 px-4 py-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-400">
          <div>Term</div>
          <div>Role</div>
          <div>Weight</div>
          <div>Penalty</div>
          <div>Time</div>
          <div>Source</div>
          <div />
        </div>
        {items.map((item) => {
          const term = termByEntityId.get(item.entity_id);
          const source = term?.source_selector;
          const time = term ? temporalSelector(term) : { mode: 'all' as const };
          const active = item.entity_id === selectedId;
          const related = relatedIds.has(item.entity_id);
          if (!term) return null;
          return (
            <div
              key={item.entity_id}
              onClick={() => onSelect(item.entity_id)}
              className={clsx(
                'grid w-full grid-cols-[minmax(10rem,1.4fr)_6.5rem_5.5rem_7.25rem_8.5rem_minmax(8rem,1fr)_4rem] items-center gap-2 border-b border-slate-100 px-4 py-3 text-left text-xs last:border-b-0',
                active
                  ? 'bg-brand-50 text-slate-900'
                  : related
                    ? 'bg-violet-50/60 text-slate-800'
                    : 'bg-white text-slate-600 hover:bg-slate-50'
              )}
            >
              <div className="min-w-0">
                <input
                  value={term.label}
                  onChange={(event) => updateTerm(term.id, { label: event.target.value })}
                  onClick={(event) => event.stopPropagation()}
                  className="h-8 w-full rounded border border-transparent bg-transparent px-2 font-medium text-slate-800 hover:border-slate-200 focus:border-brand-300 focus:bg-white focus:outline-none"
                />
                {item.summary && <div className="mt-0.5 truncate text-slate-400">{item.summary}</div>}
              </div>
              <select
                value={term.role}
                onChange={(event) => updateTerm(term.id, { role: event.target.value })}
                onClick={(event) => event.stopPropagation()}
                className="h-8 rounded border border-slate-200 bg-white px-2 text-xs"
              >
                <option value="loss">Loss</option>
                <option value="metric">Metric</option>
                <option value="constraint">Constraint</option>
                <option value="reward">Reward</option>
                <option value="regularizer">Regularizer</option>
              </select>
              <input
                type="number"
                min={0}
                step={0.01}
                value={term.weight}
                onChange={(event) => {
                  const weight = Number.parseFloat(event.target.value);
                  if (Number.isFinite(weight)) updateTerm(term.id, { weight });
                }}
                onClick={(event) => event.stopPropagation()}
                className="h-8 rounded border border-slate-200 bg-white px-2 text-xs"
              />
              <select
                value={term.penalty ?? 'squared_l2'}
                onChange={(event) => updateTerm(term.id, { penalty: event.target.value })}
                onClick={(event) => event.stopPropagation()}
                className="h-8 rounded border border-slate-200 bg-white px-2 text-xs"
              >
                {OBJECTIVE_PENALTY_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              <select
                value={time.mode}
                onChange={(event) =>
                  updateTerm(term.id, {
                    temporal_selector: updateTemporalSelector(term, {
                      mode: event.target.value as TimeAggregationSpec['mode'],
                    }),
                  })
                }
                onClick={(event) => event.stopPropagation()}
                className="h-8 rounded border border-slate-200 bg-white px-2 text-xs"
              >
                {OBJECTIVE_TEMPORAL_MODE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              <div className="min-w-0">
                <div className="truncate text-slate-600" title={source?.compact}>
                  {selectorDisplayLabel(source)}
                </div>
                {selectorDetail(source) && (
                  <div className="mt-0.5 truncate text-[11px] text-slate-400">
                    {selectorDetail(source)}
                  </div>
                )}
              </div>
              <div className="flex items-center gap-1">
                <input
                  type="checkbox"
                  checked={objectiveTermEnabled(term)}
                  onChange={(event) =>
                    onObjectiveSpecChange(
                      setObjectiveTermEnabled(objectiveSpec, term.id, event.target.checked)
                    )
                  }
                  onClick={(event) => event.stopPropagation()}
                  className="h-4 w-4 rounded border-slate-300"
                  title="Enabled"
                />
                <button
                  type="button"
                  onClick={(event) => {
                    event.stopPropagation();
                    onObjectiveSpecChange(removeObjectiveTerm(objectiveSpec, term.id));
                    if (selectedId === item.entity_id) onSelect(null);
                  }}
                  className="rounded p-1 text-slate-400 hover:bg-red-50 hover:text-red-600"
                  title="Delete objective"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          );
        })}
        {items.length === 0 && (
          <div className="px-4 py-8 text-center text-sm text-slate-400">
            No objective terms recorded.
          </div>
        )}
      </div>
    </div>
  );
}

export function ScenarioProjectionWorkspace() {
  const workspace = useWorkspaceStore((state) => state.workspace);
  const selectTopPaneEntity = useWorkspaceStore((state) => state.selectTopPaneEntity);
  const rightSidebarVisible = useLayoutStore((state) => state.rightSidebarVisible);
  const toggleRightSidebar = useLayoutStore((state) => state.toggleRightSidebar);
  const updateActiveScenarioObjectiveSpec = useWorkspaceStore(
    (state) => state.updateActiveScenarioObjectiveSpec
  );
  const graph = useGraphStore((state) => state.graph);
  const addRetainedObservable = useGraphStore((state) => state.addRetainedObservable);
  const updateRetainedObservable = useGraphStore((state) => state.updateRetainedObservable);
  const removeRetainedObservable = useGraphStore((state) => state.removeRetainedObservable);
  const { components } = useComponents();
  const topPane = getTopPaneState(workspace);
  const activeStage = getActiveStage(workspace);
  const activeScenario = getScenario(workspace, activeStage?.scenario_id);
  const objectiveSpec = ensureObjectiveSpec(activeScenario?.objective_spec);
  const schemaQuery = useStudioSchemaRegistry(
    workspace,
    activeStage?.scenario_id ?? activeScenario?.id ?? null
  );
  const registry = useMemo(
    () => buildScenarioEntityRegistry({ scenario: activeScenario, graph }),
    [activeScenario, graph]
  );
  const scene = useMemo(
    () => buildResolvedScene({ scenario: activeScenario, graph, registry, components }),
    [activeScenario, components, graph, registry]
  );
  const stageSummary =
    typeof activeStage?.metadata.summary === 'string' ? activeStage.metadata.summary : null;

  useEffect(() => {
    if (topPane.selected_entity_id && !rightSidebarVisible) {
      toggleRightSidebar();
    }
  }, [rightSidebarVisible, toggleRightSidebar, topPane.selected_entity_id]);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="relative min-h-0 flex-1">
        {(topPane.active_projection === 'model' || topPane.active_projection === 'task') && (
          <div className="absolute inset-0">
            <Canvas />
          </div>
        )}
        {topPane.active_projection === 'workspace' && (
          <WorkspaceProjection
            registry={registry}
            scene={scene}
            selectedId={topPane.selected_entity_id}
            onSelect={selectTopPaneEntity}
          />
        )}
        {topPane.active_projection === 'observables' && (
          <ObservablesProjection
            registry={registry}
            selectedId={topPane.selected_entity_id}
            graph={graph}
            objectiveSpec={objectiveSpec}
            schemaRegistry={schemaQuery.data ?? null}
            onSelect={selectTopPaneEntity}
            onAdd={addRetainedObservable}
            onUpdate={updateRetainedObservable}
            onRemove={removeRetainedObservable}
          />
        )}
        {topPane.active_projection === 'objectives' && (
          <ObjectivesProjection
            registry={registry}
            selectedId={topPane.selected_entity_id}
            objectiveSpec={objectiveSpec}
            onSelect={selectTopPaneEntity}
            onObjectiveSpecChange={updateActiveScenarioObjectiveSpec}
          />
        )}
        <ScenarioBadge
          stageLabel={activeStage?.label ?? null}
          scenarioLabel={activeScenario?.label ?? null}
          summary={stageSummary}
        />
      </div>
    </div>
  );
}
