import { useEffect, useMemo, useRef, useState } from 'react';
import type { KeyboardEvent, MouseEvent, PointerEvent, WheelEvent } from 'react';
import clsx from 'clsx';
import {
  Database,
  Eye,
  EyeOff,
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
  RefreshCw,
  Settings2,
  Trash2,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import type { SampledTaskTrial } from '@/api/client';
import { sampleTaskTrials } from '@/api/client';
import { Canvas } from '@/components/canvas/Canvas';
import { PlaybackControls } from '@/components/viewer/PlaybackControls';
import {
  objectiveEntityId,
  retainedObservableEntityId,
  buildScenarioEntityRegistry,
} from '@/features/scenario/entities';
import {
  buildResolvedScene,
  objectiveProjectionItems,
  relatedProjectionItems,
  type ResolvedScene,
  type ResolvedSceneAnchor,
  type ResolvedSceneElement,
  type ResolvedSceneEntity,
} from '@/features/scenario/projections';
import {
  addObjectiveTerm,
  createObjectiveTermFromAnchors,
  ensureObjectiveSpec,
  objectiveAnchorCanSource,
  objectiveAnchorCanTarget,
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
import {
  compactTimelineCues,
  sampledPreviewTrialLabel,
  timelineCueOffset,
} from '@/features/scenario/samplePreview';
import {
  objectiveLossWindowBands,
  primaryWorkspaceReplayTrack,
  resolveWorkspaceReplayComparison,
  resolveWorkspaceReplayModel,
  resolveWorkspaceReplaySources,
  selectWorkspaceReplayTrial,
  workspaceReplayDuration,
  workspaceReplayEventTicks,
  workspaceReplayFrameIndex,
  workspaceReplayFrameTimes,
  workspaceReplayPolyline,
  workspaceReplayProvenance,
  workspaceReplaySampleAt,
  workspaceReplayTimelineBands,
  workspaceReplayTrialLabel,
  workspaceReplayTrialRef,
  type WorkspaceReplayComparisonMember,
} from '@/features/scenario/workspaceReplay';
import { semanticTokens } from '@/components/ui/semanticTokens';
import { useComponents } from '@/hooks/useComponents';
import { useGraphStore } from '@/stores/graphStore';
import {
  getActiveStage,
  getProjectedScenario,
  getTopPaneState,
  getWorkspaceViewState,
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
  StudioStageSpec,
  StudioTopPaneProjection,
  StudioWorkspaceSpec,
  WorkspaceViewState,
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

function includeSampledTrials(bounds: SceneBounds, trials: SampledTaskTrial[]) {
  for (const trial of trials) {
    includePoint(bounds, trial.start);
    includePoint(bounds, trial.goal);
  }
}

function includeReplayTrials(
  bounds: SceneBounds,
  trials: NonNullable<ReturnType<typeof resolveWorkspaceReplayModel>['product']['trials']>
) {
  for (const trial of trials) {
    const track = primaryWorkspaceReplayTrack(trial);
    for (const point of workspaceReplayPolyline(track)) includePoint(bounds, point);
  }
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
  workspace,
  activeStage,
  registry,
  scene,
  selectedId,
  objectiveSpec,
  schemaRegistry,
  viewState,
  sampledTrials,
  previewStatus,
  previewError,
  previewMode,
  previewSeed,
  previewCount,
  onPreviewModeChange,
  onPreviewSeedChange,
  onPreviewCountChange,
  onReseed,
  onSelect,
  onObjectiveSpecChange,
  onViewStateChange,
}: {
  workspace: StudioWorkspaceSpec | null;
  activeStage: StudioStageSpec | null;
  registry: StudioScenarioEntityRegistry;
  scene: ResolvedScene;
  selectedId: string | null;
  objectiveSpec: StudioObjectiveSpec;
  schemaRegistry: StudioSchemaRegistry | null;
  viewState: WorkspaceViewState;
  sampledTrials: SampledTaskTrial[];
  previewStatus: 'idle' | 'loading' | 'ready' | 'error';
  previewError: string | null;
  previewMode: 'authoring' | 'sampled' | 'playback';
  previewSeed: number;
  previewCount: number;
  onPreviewModeChange: (mode: 'authoring' | 'sampled' | 'playback') => void;
  onPreviewSeedChange: (seed: number) => void;
  onPreviewCountChange: (count: number) => void;
  onReseed: () => void;
  onSelect: (entityId: string | null) => void;
  onObjectiveSpecChange: (spec: StudioObjectiveSpec) => void;
  onViewStateChange: (patch: Partial<WorkspaceViewState>) => void;
}) {
  const [hoveredEntityId, setHoveredEntityId] = useState<string | null>(null);
  const [objectiveDrag, setObjectiveDrag] = useState<{
    sourceAnchorId: string;
    pointer: ScenePoint;
  } | null>(null);
  const [objectiveNotice, setObjectiveNotice] = useState<string | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const suppressNextClickRef = useRef(false);
  const view = viewState.camera;
  const dragStartRef = useRef<{ x: number; y: number; pan: { x: number; y: number } } | null>(
    null
  );
  const overlayVisible = (overlayClass: string) =>
    viewState.overlay_visibility[overlayClass] !== false;
  const entityOverlayClass = (kind: ResolvedSceneEntity['kind']) => {
    if (kind === 'objective_term') return 'objectives';
    if (kind === 'retained_observable' || kind === 'probe') return 'observables';
    if (kind === 'artifact_overlay') return 'artifacts';
    if (kind === 'task_object' || kind === 'task_data' || kind === 'task_binding') return 'task';
    return 'mechanics';
  };
  const visibleEntities = scene.entities.filter((entity) =>
    overlayVisible(entityOverlayClass(entity.kind))
  );
  const visibleEntityIds = new Set(visibleEntities.map((entity) => entity.id));
  const visibleElements = scene.elements.filter((element) =>
    visibleEntityIds.has(element.entity_id)
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
  const replaySources = useMemo(
    () => resolveWorkspaceReplaySources(workspace, activeStage),
    [activeStage, workspace]
  );
  const replayModel = useMemo(
    () => replaySources[0] ?? resolveWorkspaceReplayModel(workspace, activeStage),
    [activeStage, replaySources, workspace]
  );
  const replayTrials = replayModel.product.trials ?? [];
  const selectedReplayTrial = selectWorkspaceReplayTrial(
    replayModel.product,
    viewState.selected_trial_ref
  );
  const replayComparison = useMemo(
    () =>
      resolveWorkspaceReplayComparison(
        replaySources,
        viewState.comparison_selection,
        viewState.selected_trial_ref
      ),
    [replaySources, viewState.comparison_selection, viewState.selected_trial_ref]
  );
  const comparisonActive =
    overlayVisible('comparisons') && replayComparison.members.length >= 2;
  const displayedReplayTrial = comparisonActive
    ? replayComparison.primaryTrial
    : selectedReplayTrial;
  const playbackTrialOptions = comparisonActive
    ? replayComparison.members[0]?.product.trials ?? replayTrials
    : replayTrials;
  const selectedReplayTrack = primaryWorkspaceReplayTrack(displayedReplayTrial);
  const replayFrameTimes = workspaceReplayFrameTimes(displayedReplayTrial);
  const replayDuration = workspaceReplayDuration(displayedReplayTrial);
  const replayFrameIndex = displayedReplayTrial
    ? workspaceReplayFrameIndex(displayedReplayTrial, viewState.playback.position)
    : 0;
  const replayCursorPoint = workspaceReplaySampleAt(selectedReplayTrack, replayFrameIndex);
  const replayTimelineBands = [
    ...workspaceReplayTimelineBands(displayedReplayTrial),
    ...objectiveLossWindowBands(objectiveSpec, replayDuration),
  ];
  const replayEventTicks = workspaceReplayEventTicks(displayedReplayTrial);
  const bounds = sceneBounds(scene);
  if (previewMode === 'sampled') includeSampledTrials(bounds, sampledTrials);
  if (previewMode === 'playback') includeReplayTrials(bounds, replayTrials);
  if (previewMode === 'playback' && comparisonActive) {
    includeReplayTrials(bounds, replayComparison.members.map((member) => member.trial));
  }
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
  const selectorOptions = useMemo(
    () => selectorOptionsForRegistry({ registry, schemaRegistry, objectiveSpec }),
    [objectiveSpec, registry, schemaRegistry]
  );
  const spatialObjectiveTermIds = new Set(
    scene.elements
      .filter((element) => element.archetype === 'objective_link')
      .map((element) => element.metadata.term_id)
      .filter((termId): termId is string => typeof termId === 'string')
  );
  const nonSpatialObjectiveCount = objectiveSpec.terms.filter(
    (term) => !spatialObjectiveTermIds.has(term.id)
  ).length;

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
    onViewStateChange({
      camera: {
        ...view,
        zoom: Math.max(0.35, Math.min(8, view.zoom * factor)),
      },
    });
  };

  const resetView = () => onViewStateChange({ camera: { zoom: 1, pan: { x: 0, y: 0 } } });

  const toggleOverlay = (overlayClass: string) => {
    onViewStateChange({
      overlay_visibility: {
        ...viewState.overlay_visibility,
        [overlayClass]: !overlayVisible(overlayClass),
      },
    });
  };

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
    onViewStateChange({
      camera: {
        ...view,
        pan: {
          x: start.pan.x + event.clientX - start.x,
          y: start.pan.y + event.clientY - start.y,
        },
      },
    });
  };

  const endPan = () => {
    dragStartRef.current = null;
  };

  const wheelZoom = (event: WheelEvent<SVGSVGElement>) => {
    event.preventDefault();
    zoomBy(event.deltaY > 0 ? 0.9 : 1.1);
  };

  const svgPointFromEvent = (event: PointerEvent<SVGSVGElement> | PointerEvent<SVGGElement>): ScenePoint => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return [event.clientX, event.clientY];
    return [
      ((event.clientX - rect.left) / rect.width) * WORKSPACE_SVG_WIDTH,
      ((event.clientY - rect.top) / rect.height) * WORKSPACE_SVG_HEIGHT,
    ];
  };

  const objectiveRelevantAnchor = (anchor: ResolvedSceneAnchor) =>
    objectiveAnchorCanSource(anchor) ||
    objectiveAnchorCanTarget(anchor) ||
    anchor.objective_roles.includes('illustrative') ||
    anchor.objective_roles.some((role) => role.startsWith('canonical-for:'));

  const nearestObjectiveAnchor = (point: ScenePoint): ResolvedSceneAnchor | null => {
    let best: { anchor: ResolvedSceneAnchor; distance: number } | null = null;
    for (const anchor of scene.anchors) {
      if (!anchor.position || !objectiveRelevantAnchor(anchor)) continue;
      const [x, y] = project(anchor.position);
      const distance = Math.hypot(point[0] - x, point[1] - y);
      if (distance > 26) continue;
      if (!best || distance < best.distance) best = { anchor, distance };
    }
    return best?.anchor ?? null;
  };

  const beginObjectiveDrag = (anchor: ResolvedSceneAnchor, event: PointerEvent<SVGGElement>) => {
    if (!objectiveRelevantAnchor(anchor)) return;
    event.preventDefault();
    event.stopPropagation();
    setObjectiveNotice(null);
    setObjectiveDrag({ sourceAnchorId: anchor.id, pointer: svgPointFromEvent(event) });
  };

  const moveObjectiveDrag = (event: PointerEvent<SVGSVGElement>) => {
    if (!objectiveDrag) return false;
    setObjectiveDrag({ ...objectiveDrag, pointer: svgPointFromEvent(event) });
    return true;
  };

  const completeObjectiveDrag = (event: PointerEvent<SVGSVGElement>) => {
    const drag = objectiveDrag;
    if (!drag) return false;
    const sourceAnchor = scene.anchors.find((anchor) => anchor.id === drag.sourceAnchorId);
    const targetAnchor = nearestObjectiveAnchor(svgPointFromEvent(event));
    setObjectiveDrag(null);
    if (!sourceAnchor || !targetAnchor || sourceAnchor.id === targetAnchor.id) {
      setObjectiveNotice('Drop on a different objective anchor.');
      return true;
    }
    const result = createObjectiveTermFromAnchors({
      spec: objectiveSpec,
      sourceAnchor,
      targetAnchor,
      anchors: scene.anchors,
      selectorOptions,
    });
    if (!result) {
      setObjectiveNotice('Those anchors do not expose compatible objective selectors.');
      return true;
    }
    onObjectiveSpecChange(addObjectiveTerm(objectiveSpec, result.term));
    onSelect(objectiveEntityId(result.term.id));
    suppressNextClickRef.current = true;
    const notices = [result.source.message, result.target.message].filter(Boolean);
    setObjectiveNotice(notices.join(' ') || 'Objective created from workspace anchors.');
    return true;
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
            stroke={active ? semanticTokens.objective.strongStroke : semanticTokens.objective.stroke}
            strokeWidth={active ? 3 : 2.25}
            strokeDasharray="6 5"
            opacity={active ? 0.95 : 0.72}
            vectorEffect="non-scaling-stroke"
          />
        </g>
      );
    }

    return null;
  };

  const renderObjectiveAnchor = (anchor: ResolvedSceneAnchor) => {
    if (!anchor.position || !objectiveRelevantAnchor(anchor)) return null;
    const [x, y] = project(anchor.position);
    const source = objectiveAnchorCanSource(anchor);
    const target = objectiveAnchorCanTarget(anchor);
    const illustrative = anchor.objective_roles.includes('illustrative');
    const active = anchor.entity_id === selectedId || hoveredRelatedIds.has(anchor.entity_id);
    const radius = target ? 8 : 6;
    return (
      <g
        key={`${anchor.id}:objective-anchor`}
        onPointerDown={(event) => beginObjectiveDrag(anchor, event)}
        onClick={(event) => event.stopPropagation()}
        onMouseEnter={() => setHoveredEntityId(anchor.entity_id)}
        onMouseLeave={() => setHoveredEntityId(null)}
        className="cursor-crosshair"
      >
        <circle
          cx={x}
          cy={y}
          r={radius + (active ? 4 : 2)}
          fill={target ? semanticTokens.objective.paleFill : '#ffffff'}
          stroke={semanticTokens.objective.stroke}
          strokeWidth={illustrative ? 1 : 1.5}
          strokeDasharray={illustrative ? '3 3' : undefined}
          opacity={illustrative ? 0.7 : 0.92}
          vectorEffect="non-scaling-stroke"
        />
        {source && (
          <circle
            cx={x}
            cy={y}
            r={2.75}
            fill={semanticTokens.objective.strongStroke}
          />
        )}
        {target && (
          <path
            d={`M ${x - 5} ${y} L ${x + 5} ${y} M ${x} ${y - 5} L ${x} ${y + 5}`}
            stroke={semanticTokens.objective.strongStroke}
            strokeWidth={1.4}
            strokeLinecap="round"
            vectorEffect="non-scaling-stroke"
          />
        )}
        <title>
          {target
            ? `Objective target: ${anchor.label}`
            : source
              ? `Objective source: ${anchor.label}`
              : `Objective anchor: ${anchor.label}`}
        </title>
      </g>
    );
  };

  const renderSampledTrial = (trial: SampledTaskTrial) => {
    const [x0, y0] = project(trial.start);
    const [x1, y1] = project(trial.goal);
    const hue = trial.index % 2 === 0 ? '#0f766e' : '#7c3aed';
    return (
      <g key={trial.id} opacity="0.72">
        <line
          x1={x0}
          y1={y0}
          x2={x1}
          y2={y1}
          stroke={hue}
          strokeWidth={2}
          strokeDasharray="5 5"
          vectorEffect="non-scaling-stroke"
        />
        <circle
          cx={x0}
          cy={y0}
          r={6}
          fill="#ffffff"
          stroke={hue}
          strokeWidth={1.75}
          vectorEffect="non-scaling-stroke"
        />
        <circle
          cx={x1}
          cy={y1}
          r={9}
          fill="none"
          stroke={hue}
          strokeWidth={1.75}
          vectorEffect="non-scaling-stroke"
        />
        <circle cx={x1} cy={y1} r={3.5} fill={hue} />
        <text
          x={x1 + 10}
          y={y1 - 8}
          className="select-none fill-slate-700 text-[11px] font-semibold"
        >
          {sampledPreviewTrialLabel(trial)}
        </text>
      </g>
    );
  };

  const renderReplayTrace = (
    trial: NonNullable<typeof selectedReplayTrial>,
    index: number
  ) => {
    const track = primaryWorkspaceReplayTrack(trial);
    const points = workspaceReplayPolyline(track);
    if (points.length === 0) return null;
    const selected = selectedReplayTrial
      ? workspaceReplayTrialRef(trial) === workspaceReplayTrialRef(selectedReplayTrial)
      : index === 0;
    const projected = points.map(project).map((point) => point.join(',')).join(' ');
    const color = index % 2 === 0 ? '#0f766e' : '#7c3aed';
    return (
      <g key={workspaceReplayTrialRef(trial)} opacity={selected ? 0.95 : 0.38}>
        <polyline
          points={projected}
          fill="none"
          stroke={color}
          strokeWidth={selected ? 3 : 2}
          strokeLinecap="round"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />
        {!selected && <title>{workspaceReplayTrialLabel(trial)}</title>}
      </g>
    );
  };

  const renderComparisonMemberTrace = (member: WorkspaceReplayComparisonMember) => {
    const points = workspaceReplayPolyline(member.track);
    if (points.length === 0) return null;
    const projected = points.map(project).map((point) => point.join(',')).join(' ');
    return (
      <g key={`comparison:${member.role}:${member.ref}`} opacity={0.92}>
        <polyline
          points={projected}
          fill="none"
          stroke={member.color}
          strokeWidth={member.role === 'baseline' ? 3.25 : 2.75}
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeDasharray={member.role === 'baseline' ? undefined : '7 5'}
          vectorEffect="non-scaling-stroke"
        />
        <title>{`${member.role}: ${member.label}`}</title>
      </g>
    );
  };

  const renderComparisonMemberCursor = (member: WorkspaceReplayComparisonMember) => {
    const frameIndex = workspaceReplayFrameIndex(member.trial, viewState.playback.position);
    const point = workspaceReplaySampleAt(member.track, frameIndex);
    if (!point) return null;
    const [x, y] = project(point);
    return (
      <g key={`comparison-cursor:${member.role}:${member.ref}`}>
        <circle
          cx={x}
          cy={y}
          r={member.role === 'baseline' ? 10 : 8}
          fill="#ffffff"
          stroke={member.color}
          strokeWidth={2.25}
          vectorEffect="non-scaling-stroke"
        />
        <circle cx={x} cy={y} r={4} fill={member.color} />
        <text
          x={x + 12}
          y={member.role === 'baseline' ? y - 12 : y + 18}
          className="select-none fill-slate-800 text-[11px] font-semibold capitalize"
        >
          {member.role}
        </text>
      </g>
    );
  };

  const renderReplayCursor = () => {
    if (!replayCursorPoint || !displayedReplayTrial) return null;
    const [x, y] = project(replayCursorPoint);
    return (
      <g>
        <circle
          cx={x}
          cy={y}
          r={10}
          fill="#ffffff"
          stroke="#0f172a"
          strokeWidth={2.25}
          vectorEffect="non-scaling-stroke"
        />
        <circle cx={x} cy={y} r={4.5} fill="#0f172a" />
        <text
          x={x + 12}
          y={y - 12}
          className="select-none fill-slate-800 text-[11px] font-semibold"
        >
          {workspaceReplayTrialLabel(displayedReplayTrial)}
        </text>
      </g>
    );
  };

  const renderedSamples = previewMode === 'sampled' ? sampledTrials.slice(0, previewCount) : [];
  const modeLabel =
    previewMode === 'sampled'
      ? 'Sampled preview'
      : previewMode === 'playback'
        ? comparisonActive
          ? 'Comparison playback'
          : 'Playback'
        : 'Authoring rest pose';

  return (
    <div className="grid h-full min-h-0 grid-cols-[minmax(0,1fr)_17rem] bg-slate-50">
      <div className="relative min-h-0 overflow-hidden">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${WORKSPACE_SVG_WIDTH} ${WORKSPACE_SVG_HEIGHT}`}
          className="h-full w-full touch-none bg-white"
          role="img"
          aria-label="Workspace projection"
          onPointerDown={beginPan}
          onPointerMove={(event) => {
            if (moveObjectiveDrag(event)) return;
            movePan(event);
          }}
          onPointerUp={(event) => {
            if (completeObjectiveDrag(event)) return;
            endPan();
          }}
          onPointerCancel={() => {
            setObjectiveDrag(null);
            endPan();
          }}
          onWheel={wheelZoom}
          onClick={() => {
            if (suppressNextClickRef.current) {
              suppressNextClickRef.current = false;
              return;
            }
            onSelect(null);
          }}
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
          <g opacity={previewMode === 'playback' ? 0.32 : 1}>
            {visibleElements.map(renderElement)}
          </g>
          {renderedSamples.map(renderSampledTrial)}
          {previewMode === 'playback' && comparisonActive
            ? replayComparison.members.map(renderComparisonMemberTrace)
            : previewMode === 'playback'
              ? replayTrials.map(renderReplayTrace)
              : null}
          {previewMode === 'playback' && comparisonActive
            ? replayComparison.members.map(renderComparisonMemberCursor)
            : previewMode === 'playback'
              ? renderReplayCursor()
              : null}
          {objectiveDrag && (() => {
            const sourceAnchor = scene.anchors.find((anchor) => anchor.id === objectiveDrag.sourceAnchorId);
            if (!sourceAnchor?.position) return null;
            const [x0, y0] = project(sourceAnchor.position);
            return (
              <line
                x1={x0}
                y1={y0}
                x2={objectiveDrag.pointer[0]}
                y2={objectiveDrag.pointer[1]}
                stroke={semanticTokens.objective.strongStroke}
                strokeWidth={2}
                strokeDasharray="4 5"
                opacity="0.82"
                vectorEffect="non-scaling-stroke"
              />
            );
          })()}
          {scene.anchors.map(renderObjectiveAnchor)}
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
        <div className="absolute left-4 top-16 flex w-[27rem] max-w-[calc(100%-2rem)] flex-col gap-2 rounded border border-slate-200 bg-white/95 p-2 shadow-sm backdrop-blur">
          <div className="flex items-center justify-between gap-3">
            <div className="min-w-0">
              <div className="truncate text-xs font-semibold text-slate-800">{modeLabel}</div>
              <div className="truncate text-[11px] text-slate-500">
                {previewMode === 'sampled'
                  ? `${renderedSamples.length} trials - seed ${previewSeed}`
                  : previewMode === 'playback'
                    ? workspaceReplayProvenance(displayedReplayTrial)
                    : 'Authored geometry without sampled trial instances'}
                {nonSpatialObjectiveCount > 0
                  ? ` - ${nonSpatialObjectiveCount} non-spatial objectives`
                  : ''}
              </div>
            </div>
            <div className="grid h-7 grid-cols-3 rounded border border-slate-200 bg-slate-50 p-0.5 text-[11px] font-medium">
              {(['authoring', 'sampled', 'playback'] as const).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  onClick={() => onPreviewModeChange(mode)}
                  className={clsx(
                    'h-6 w-16 rounded px-1 capitalize',
                    previewMode === mode
                      ? 'bg-white text-slate-900 shadow-sm'
                      : 'text-slate-500 hover:text-slate-800'
                  )}
                >
                  {mode === 'authoring' ? 'Rest' : mode}
                </button>
              ))}
            </div>
          </div>
          {previewMode === 'sampled' && (
            <div className="grid grid-cols-[5.5rem_4.5rem_2rem_minmax(0,1fr)] items-center gap-2">
              <input
                type="number"
                value={previewSeed}
                onChange={(event) => {
                  const value = Number.parseInt(event.target.value, 10);
                  if (Number.isFinite(value)) onPreviewSeedChange(value);
                }}
                className="h-8 rounded border border-slate-200 px-2 text-xs"
                aria-label="Preview seed"
                title="Preview seed"
              />
              <input
                type="number"
                min={1}
                max={16}
                value={previewCount}
                onChange={(event) => {
                  const value = Number.parseInt(event.target.value, 10);
                  if (Number.isFinite(value)) onPreviewCountChange(Math.max(1, Math.min(16, value)));
                }}
                className="h-8 rounded border border-slate-200 px-2 text-xs"
                aria-label="Preview trial count"
                title="Preview trial count"
              />
              <button
                type="button"
                onClick={onReseed}
                className="inline-flex h-8 w-8 items-center justify-center rounded border border-slate-200 text-slate-500 hover:bg-slate-100 hover:text-slate-800"
                title="Reseed preview"
              >
                <RefreshCw className="h-4 w-4" />
              </button>
              <div className="truncate text-[11px] text-slate-500">
                {previewStatus === 'loading'
                  ? 'Sampling...'
                  : previewStatus === 'error'
                    ? previewError
                    : 'Ghosted start and goal pairs are sampled from the task only'}
              </div>
            </div>
          )}
          {previewMode === 'sampled' && renderedSamples.length > 0 && (
            <div className="grid max-h-20 grid-cols-2 gap-1 overflow-hidden">
              {renderedSamples.slice(0, 4).map((trial) => (
                <div
                  key={`${trial.id}:timeline`}
                  className="grid grid-cols-[1.75rem_minmax(0,1fr)] items-center gap-1 text-[10px] text-slate-500"
                >
                  <div className="font-semibold text-slate-600">{sampledPreviewTrialLabel(trial)}</div>
                  <div className="relative h-2 rounded-full bg-slate-100">
                    {compactTimelineCues(trial).map((cue) => (
                      <span
                        key={`${cue.kind}:${cue.label}:${cue.step}`}
                        className={clsx(
                          'absolute top-0 h-2 w-1 rounded-full',
                          cue.kind === 'event' ? 'bg-amber-500' : 'bg-teal-500'
                        )}
                        style={{ left: `${timelineCueOffset(cue, trial.n_steps) * 100}%` }}
                        title={`${cue.label} · step ${cue.step}`}
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
          {previewMode === 'playback' && (
            <div className="space-y-2">
              <div className="grid grid-cols-[minmax(0,1fr)_7.5rem] items-center gap-2">
                <select
                  value={displayedReplayTrial ? workspaceReplayTrialRef(displayedReplayTrial) : ''}
                  onChange={(event) => {
                    onViewStateChange({
                      selected_trial_ref: event.target.value || null,
                      playback: { ...viewState.playback, position: 0 },
                    });
                  }}
                  className="h-8 min-w-0 rounded border border-slate-200 bg-white px-2 text-xs"
                  aria-label="Playback trial"
                  title="Playback trial"
                >
                  {playbackTrialOptions.map((trial) => (
                    <option
                      key={workspaceReplayTrialRef(trial)}
                      value={workspaceReplayTrialRef(trial)}
                    >
                      {workspaceReplayTrialLabel(trial)}
                    </option>
                  ))}
                </select>
                <div className="truncate text-right text-[11px] font-medium text-slate-500">
                  {comparisonActive
                    ? 'comparison'
                    : replayModel.source === 'fixture'
                      ? 'fixture data'
                      : 'artifact data'}
                </div>
              </div>
              {replaySources.length > 0 && (
                <div className="grid grid-cols-2 gap-2">
                  <label className="min-w-0 text-[10px] font-semibold uppercase text-slate-500">
                    Baseline
                    <select
                      value={viewState.comparison_selection.baseline_ref ?? ''}
                      onChange={(event) =>
                        onViewStateChange({
                          comparison_selection: {
                            ...viewState.comparison_selection,
                            baseline_ref: event.target.value || null,
                          },
                        })
                      }
                      className="mt-1 h-8 w-full min-w-0 rounded border border-slate-200 bg-white px-2 text-xs font-normal normal-case text-slate-700"
                      aria-label="Comparison baseline replay"
                    >
                      <option value="">None</option>
                      {replaySources.map((source) => (
                        <option key={`baseline:${source.ref}`} value={source.ref}>
                          {source.label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="min-w-0 text-[10px] font-semibold uppercase text-slate-500">
                    Candidate
                    <select
                      value={viewState.comparison_selection.candidate_ref ?? ''}
                      onChange={(event) =>
                        onViewStateChange({
                          comparison_selection: {
                            ...viewState.comparison_selection,
                            candidate_ref: event.target.value || null,
                          },
                        })
                      }
                      className="mt-1 h-8 w-full min-w-0 rounded border border-slate-200 bg-white px-2 text-xs font-normal normal-case text-slate-700"
                      aria-label="Comparison candidate replay"
                    >
                      <option value="">None</option>
                      {replaySources.map((source) => (
                        <option key={`candidate:${source.ref}`} value={source.ref}>
                          {source.label}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
              )}
              {comparisonActive && (
                <div className="grid grid-cols-2 gap-2">
                  {replayComparison.members.map((member) => (
                    <div
                      key={`legend:${member.role}:${member.ref}`}
                      className="grid min-w-0 grid-cols-[0.75rem_minmax(0,1fr)] items-center gap-1 rounded border border-slate-200 bg-slate-50 px-2 py-1"
                    >
                      <span
                        className="h-2.5 w-2.5 rounded-full"
                        style={{ backgroundColor: member.color }}
                      />
                      <span className="truncate text-[11px] text-slate-600">
                        <span className="font-semibold capitalize text-slate-700">
                          {member.role}
                        </span>{' '}
                        {member.label}
                      </span>
                    </div>
                  ))}
                </div>
              )}
              <PlaybackControls
                position={viewState.playback.position}
                duration={replayDuration}
                speed={viewState.playback.speed}
                frameTimes={replayFrameTimes}
                bands={replayTimelineBands}
                eventTicks={replayEventTicks}
                cursorLabel={workspaceReplayProvenance(displayedReplayTrial)}
                disabled={!displayedReplayTrial || !selectedReplayTrack}
                onPositionChange={(position) =>
                  onViewStateChange({ playback: { ...viewState.playback, position } })
                }
                onSpeedChange={(speed) =>
                  onViewStateChange({ playback: { ...viewState.playback, speed } })
                }
              />
              <div className="max-h-10 overflow-hidden text-[11px] leading-5 text-slate-500">
                {replayModel.message}
                {replayModel.warnings.length > 0 ? ` ${replayModel.warnings[0]}` : ''}
              </div>
            </div>
          )}
          {objectiveNotice && (
            <div
              className={clsx(
                'rounded border px-2 py-1 text-[11px]',
                semanticTokens.objective.border,
                semanticTokens.objective.background,
                semanticTokens.objective.text
              )}
            >
              {objectiveNotice}
            </div>
          )}
        </div>
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
          {(['objectives', 'observables', 'artifacts'] as const).map((overlayClass) => {
            const Icon = overlayVisible(overlayClass) ? Eye : EyeOff;
            return (
              <button
                key={overlayClass}
                type="button"
                onClick={() => toggleOverlay(overlayClass)}
                className="inline-flex h-7 w-7 items-center justify-center rounded text-slate-500 hover:bg-slate-100 hover:text-slate-800"
                title={`${overlayVisible(overlayClass) ? 'Hide' : 'Show'} ${overlayClass}`}
              >
                <Icon className="h-4 w-4" />
              </button>
            );
          })}
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
          {visibleEntities.map((entity) => {
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
                      ? clsx(semanticTokens.objective.softBackground, 'text-slate-800')
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
  const updateActiveWorkspaceViewState = useWorkspaceStore(
    (state) => state.updateActiveWorkspaceViewState
  );
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
  const activeScenario = getProjectedScenario(workspace, activeStage);
  const workspaceViewState = getWorkspaceViewState(workspace, activeStage);
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
  const [previewMode, setPreviewMode] = useState<'authoring' | 'sampled' | 'playback'>('sampled');
  const [previewSeed, setPreviewSeed] = useState(0);
  const [previewCount, setPreviewCount] = useState(6);
  const [sampledTrials, setSampledTrials] = useState<SampledTaskTrial[]>([]);
  const [previewStatus, setPreviewStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  const [previewError, setPreviewError] = useState<string | null>(null);
  const stageSummary =
    typeof activeStage?.metadata.summary === 'string' ? activeStage.metadata.summary : null;

  useEffect(() => {
    if (previewMode !== 'sampled' || !activeScenario?.task_spec) {
      setPreviewStatus('idle');
      return;
    }
    let cancelled = false;
    setPreviewStatus('loading');
    setPreviewError(null);
    sampleTaskTrials({
      task_spec: activeScenario.task_spec,
      seed: previewSeed,
      count: previewCount,
    })
      .then((response) => {
        if (cancelled) return;
        setSampledTrials(response.trials);
        setPreviewStatus('ready');
      })
      .catch((error) => {
        if (cancelled) return;
        setSampledTrials([]);
        setPreviewStatus('error');
        setPreviewError(error instanceof Error ? error.message : 'Preview sampling failed');
      });
    return () => {
      cancelled = true;
    };
  }, [activeScenario?.task_spec, previewCount, previewMode, previewSeed]);

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
            workspace={workspace}
            activeStage={activeStage}
            registry={registry}
            scene={scene}
            selectedId={topPane.selected_entity_id}
            objectiveSpec={objectiveSpec}
            schemaRegistry={schemaQuery.data ?? null}
            viewState={workspaceViewState}
            sampledTrials={sampledTrials}
            previewStatus={previewStatus}
            previewError={previewError}
            previewMode={previewMode}
            previewSeed={previewSeed}
            previewCount={previewCount}
            onPreviewModeChange={setPreviewMode}
            onPreviewSeedChange={setPreviewSeed}
            onPreviewCountChange={setPreviewCount}
            onReseed={() => setPreviewSeed((seed) => seed + 1)}
            onSelect={selectTopPaneEntity}
            onObjectiveSpecChange={updateActiveScenarioObjectiveSpec}
            onViewStateChange={updateActiveWorkspaceViewState}
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
