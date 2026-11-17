import { ArrowLeft, ArrowRight, ChevronRight, Plus } from 'lucide-react';
import { ReactFlowProvider, ViewportPortal, useReactFlow } from '@xyflow/react';
import {
  memo,
  useCallback,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type DragEvent,
  type ReactNode,
  type Ref,
} from 'react';
import { StudioRunMark } from '@/components/activity/StudioRunStatus';
import {
  studioAccessibleName,
  studioCountOf,
  studioName,
  studioRunStatus,
  studioStep,
  type StudioRunStatusId,
} from '@/components/activity/studioVocabulary';
import { bindStudioCommand, commandAttributes, type StudioCommand } from '@/features/projectWindow/studioCommands';
import { StudioCanvas, type CanvasViewport } from '@/components/canvas/StudioCanvas';
import { useElementSize } from '@/hooks/useElementSize';
import type { PlanStepView, PlanSweepView, PlanView } from '@/features/pipeline/planView';
import {
  selectionButton,
  indexPlanLineage,
  selectedRuns,
  selectionTerms,
  selectStep,
  type PipelineSelection,
  type PipelineSelectionState,
  type RunSetsByStep,
} from '@/features/pipeline/selection';
import {
  MapSweepValue,
  PipelineRunMarks,
  type ApplyMapSelection,
} from './PipelineRunMarks';
import {
  foldedMapSweep,
  compactMapCardWidth,
  focusedMapEdgePairs,
  fullMapFits,
  layoutPipelineMap,
  mapAnchor,
  mapCardWidth,
  mapColumnGap,
  mapEdgePair,
  mapPadding,
  mapRowGap,
  mapStepLineage,
  orderedMapSweeps,
  type PipelineCardPresentation,
} from './pipelineMapLayout';

type StepRunSelection = Exclude<PipelineSelection, { kind: 'steps' }>;

export interface PipelineMapStepSource {
  stepId: string;
  selection: StepRunSelection | null;
}

export interface MapDocumentScope {
  readonly users: ReadonlySet<string>;
  readonly lit: ReadonlySet<string>;
}

type ChooseStep = (id: string, extend: boolean) => void;

function documentLevel(scope: MapDocumentScope | undefined, step: string) {
  return !scope ? undefined : scope.lit.has(step) ? 'lit' : scope.users.has(step) ? 'plain' : 'faint';
}

const documentLevelStyle = 'data-[document-scope=faint]:border-dashed';
const documentLitStyle = 'data-[document-scope=lit]:ring-1 data-[document-scope=lit]:ring-shell-muted';

function chooseMapStep(apply: ApplyMapSelection, toggle?: ChooseStep): ChooseStep {
  return (id, extend) => {
    if (toggle) toggle(id, extend);
    else apply((state) => selectStep(state, id, extend));
  };
}

export interface StudioPipelineMapProps {
  plan: PlanView;
  state: PipelineSelectionState;
  onSelectionChange: (state: PipelineSelectionState) => void;
  documentScope?: MapDocumentScope;
  onChooseStep?: ChooseStep;
  mode?: 'map' | 'line';
  drawerOpen?: boolean;
  presentation?: Readonly<Record<string, PipelineCardPresentation>>;
  onPresentationChange?: (stepId: string, presentation: PipelineCardPresentation) => void;
  newStepCommand?: (source?: PipelineMapStepSource, subject?: string) => StudioCommand | undefined;
  newSweepCommand?: (stepId: string, subject?: string) => StudioCommand | undefined;
  problemSteps?: ReadonlySet<string>;
  onProblems?: (stepId: string, opener: HTMLButtonElement) => void;
  activeStepId?: string | null;
  viewportRef?: Ref<CanvasViewport>;
  fitReady?: boolean;
}

const emptyPresentation: PipelineCardPresentation = {};
const smallButton = 'rounded border border-shell-control bg-shell-surface p-1 text-shell-muted hover:bg-shell-hover focus-visible:outline-2 focus-visible:outline-shell-accent';

function stepSelection(selection: PipelineSelection | null, stepId: string) {
  if (!selection) return null;
  if (selection.kind === 'steps') return selection.stepIds.includes(stepId) ? selection : null;
  return selection.stepId === stepId ? selection : null;
}

function StatusCounts({ id, counts, onProblems }: {
  id?: string;
  counts: PlanStepView['statusCounts'];
  onProblems?: (opener: HTMLButtonElement) => void;
}) {
  const className = 'inline-flex items-center gap-1 whitespace-nowrap';
  const statuses = Object.entries(counts) as [StudioRunStatusId, number][];
  if (onProblems && counts.cannot_run === undefined) statuses.push(['cannot_run', 0]);
  return (
    <span id={id} className="flex flex-wrap gap-x-2 gap-y-1 text-xs text-shell-muted">
      {statuses.map(([status, count]) => {
        const problem = status === 'cannot_run' && onProblems;
        if (count <= 0 && !problem) return null;
        const mark = <><StudioRunMark status={status} />
          {count > 0 ? `${count} ` : ''}{studioRunStatus(status)}</>;
        return problem
          ? <button type="button" key={status} className={`${className} underline`} aria-haspopup="dialog"
            onClick={(event) => problem(event.currentTarget)}>{mark}</button>
          : <span key={status} className={className}>{mark}</span>;
      })}
    </span>
  );
}

function stepLabel(step: PlanStepView, count?: string | number) {
  return [step.label, studioStep(step.kind).name, count].filter((part) => part !== undefined).join(' ');
}

function StepName({ step }: { step: PlanStepView }) {
  const { Icon, name } = studioStep(step.kind);
  return (
    <span className="flex min-w-0 items-center gap-1.5">
      <Icon aria-hidden="true" className="h-4 w-4 shrink-0" />
      <span className="truncate font-medium">{step.label}</span>{' '}
      <span data-map-kind-label className="shrink-0 text-xs text-shell-muted">{name}</span>
    </span>
  );
}

export function bindSweepFold(
  stepId: string, sweep: PlanSweepView, folded: boolean, presentation: PipelineCardPresentation,
  save: NonNullable<StudioPipelineMapProps['onPresentationChange']>,
) {
  return bindStudioCommand(folded ? 'unfold' : 'fold',
    () => save(stepId, { ...presentation, foldedSweepId: folded ? null : sweep.id }), { subject: sweep.label });
}

function FoldSweep({ command }: { command: StudioCommand }) {
  return <button type="button" className={smallButton} {...commandAttributes(command)} onClick={command.invoke}>
    {studioName(command.name)}
  </button>;
}

function StepShape({
  step,
  sweeps,
  folded,
  selection,
  presentation,
  applySelection,
  onPresentationChange,
  newSweep,
}: {
  step: PlanStepView;
  sweeps: readonly PlanSweepView[];
  folded?: PlanSweepView;
  selection: PipelineSelection | null;
  presentation: PipelineCardPresentation;
  applySelection: ApplyMapSelection;
  onPresentationChange?: StudioPipelineMapProps['onPresentationChange'];
  newSweep?: StudioCommand;
}) {
  const dragging = useRef<string | null>(null);
  const [openSweep, setOpenSweep] = useState<string | null>(null);

  function moveSweep(from: number, to: number) {
    if (!onPresentationChange || from === to || from < 0 || to < 0 || to >= sweeps.length) return;
    const order = sweeps.map(({ id }) => id);
    const [moved] = order.splice(from, 1);
    order.splice(to, 0, moved);
    onPresentationChange(step.id, { ...presentation, sweepOrder: order });
  }

  function dropSweep(event: DragEvent, index: number) {
    if (!dragging.current) return;
    event.preventDefault();
    moveSweep(sweeps.findIndex(({ id }) => id === dragging.current), index);
    dragging.current = null;
  }

  return (
    <div className="flex flex-wrap items-center gap-1 text-xs text-shell-muted">
      {sweeps.map((sweep, index) => {
        const isFolded = sweep.id === folded?.id;
        const label = `${sweep.label} ${sweep.values.length}`;
        return (
          <div key={sweep.id} className="inline-flex items-center gap-1">
            {index > 0 && <span aria-hidden="true">×</span>}
            <div className="relative">
              <button
                type="button"
                className="rounded px-1 py-0.5 hover:bg-shell-hover focus-visible:outline-2 focus-visible:outline-shell-accent aria-expanded:bg-shell-subtle"
                aria-label={studioAccessibleName('sweep', label)}
                aria-expanded={openSweep === sweep.id}
                draggable={Boolean(onPresentationChange)}
                onClick={() => setOpenSweep((open) => open === sweep.id ? null : sweep.id)}
                onDragStart={(event) => {
                  dragging.current = sweep.id;
                  event.dataTransfer.effectAllowed = 'move';
                  event.dataTransfer.setData('text/plain', sweep.id);
                }}
                onDragEnd={() => { dragging.current = null; }}
                onDragOver={(event) => {
                  if (dragging.current) event.preventDefault();
                }}
                onDrop={(event) => dropSweep(event, index)}
              >
                {isFolded ? `(${label})` : label}
              </button>
              {openSweep === sweep.id && (
                <div className="absolute left-0 top-full z-20 w-max max-w-56 rounded-md border border-shell-divider bg-shell-surface p-2 shadow-md">
                  <ul className="nowheel max-h-36 space-y-0.5 overflow-y-auto">
                    {sweep.values.map((value) => (
                      <li key={value.id}>
                        <MapSweepValue
                          sweep={sweep}
                          value={value}
                          stepId={step.id}
                          selection={selection}
                          applySelection={applySelection}
                        />
                      </li>
                    ))}
                  </ul>
                  {onPresentationChange && (
                    <div className="mt-2 flex items-center gap-1 border-t border-shell-divider pt-1">
                      <FoldSweep command={bindSweepFold(step.id, sweep, isFolded, presentation, onPresentationChange)} />
                      <button
                        type="button"
                        className={smallButton}
                        aria-label={studioAccessibleName('moveEarlier', sweep.label)}
                        disabled={index === 0}
                        onClick={() => moveSweep(index, index - 1)}
                      >
                        <ArrowLeft aria-hidden="true" className="h-3 w-3" />
                      </button>
                      <button
                        type="button"
                        className={smallButton}
                        aria-label={studioAccessibleName('moveLater', sweep.label)}
                        disabled={index === sweeps.length - 1}
                        onClick={() => moveSweep(index, index + 1)}
                      >
                        <ArrowRight aria-hidden="true" className="h-3 w-3" />
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        );
      })}
      {newSweep && (
        <button type="button" className={smallButton} {...commandAttributes(newSweep)}
          disabled={!newSweep.enabled} onClick={newSweep.invoke}>
          <Plus aria-hidden="true" className="h-3 w-3" />
        </button>
      )}
    </div>
  );
}

const StepCard = memo(function StepCard({
  step,
  selection,
  presentation,
  applySelection,
  chooseStep,
  documentScope,
  onPresentationChange,
  newStep,
  newSweep,
  onProblems,
  compact,
  dimmed,
  upstream,
  downstream,
}: {
  step: PlanStepView;
  selection: PipelineSelection | null;
  presentation: PipelineCardPresentation;
  applySelection: ApplyMapSelection;
  chooseStep: ChooseStep;
  documentScope?: MapDocumentScope;
  onPresentationChange?: StudioPipelineMapProps['onPresentationChange'];
  newStep?: StudioCommand;
  newSweep?: StudioCommand;
  onProblems?: StudioPipelineMapProps['onProblems'];
  compact: boolean;
  dimmed: boolean;
  upstream?: ReadonlySet<string>;
  downstream?: ReadonlySet<string>;
}) {
  const id = useId();
  const sweeps = useMemo(() => orderedMapSweeps(step, presentation), [step, presentation]);
  const folded = foldedMapSweep(step, sweeps, presentation);
  const level = documentLevel(documentScope, step.id);
  const selected = useMemo(() => new Set(
    (compact ? [] : level === 'lit' ? step.runs : selectedRuns(step, selection)).map(({ id }) => id),
  ), [step, selection, level, compact]);
  const terms = selection?.kind === 'steps' ? [] : selectionTerms(step, selection);
  const { tint } = studioStep(step.kind);
  const related = (upstream?.size ?? 0) + (downstream?.size ?? 0) -
    (upstream && downstream ? [...upstream].filter((id) => downstream.has(id)).length : 0);
  const problem = onProblems && ((opener: HTMLButtonElement) => onProblems(step.id, opener));
  return (
    <article
      data-map-card
      data-canvas-item
      data-map-step={step.id}
      data-map-density={compact ? 'compact' : 'full'}
      data-map-selection={selection?.kind}
      data-map-related={related || undefined}
      data-map-dimmed={dimmed || undefined}
      data-document-scope={level}
      aria-label={studioAccessibleName(step.kind, step.label)}
      className={`studio-map-card nopan pointer-events-auto relative space-y-2 rounded-lg border-2 bg-shell-surface px-3 py-2 text-sm shadow-sm focus-within:z-20 data-[map-dimmed=true]:border-dashed ${documentLevelStyle} ${!selection ? documentLitStyle : ''}`}
      style={{ borderColor: selection?.kind === 'steps' ? 'var(--text-primary)' : tint, width: compact ? compactMapCardWidth : mapCardWidth }}
    >
      <span id={`${id}-summary`} className="sr-only">
        {sweeps.map((sweep) => `${sweep.label} ${sweep.values.length}`).join(' · ')}
      </span>
      <header className="relative">
        <span aria-hidden="true" data-map-anchor={mapAnchor(step.id, 'in')} className="pointer-events-none absolute -left-3 top-1/2" />
        <span aria-hidden="true" data-map-anchor={mapAnchor(step.id, 'out')} className="pointer-events-none absolute -right-3 top-1/2" />
        <button
          type="button"
          data-canvas-focus
          className="studio-map-card__head w-full rounded px-1 py-1 text-left hover:bg-shell-hover"
          aria-label={studioAccessibleName('select', stepLabel(step, compact ? step.runs.length : undefined))}
          aria-describedby={[`${id}-summary`, `${id}-status`, related ? `${id}-related` : ''].filter(Boolean).join(' ')}
          aria-current={selection?.kind === 'steps' ? 'step' : undefined}
          aria-pressed={documentScope ? level === 'lit' : selection?.kind === 'steps'}
          {...selectionButton((extend) => chooseStep(step.id, extend))}
        >
          {compact ? <span className="flex items-center gap-1"><StepName step={step} />{' '}
            <span className="ml-auto tabular-nums text-xs">{step.runs.length}</span></span> : <StepName step={step} />}
        </button>
      </header>
      {related > 0 && <span id={`${id}-related`} className="sr-only">{studioCountOf(related, step.runs.length)}</span>}
      {compact ? <>
        <StatusCounts id={`${id}-status`} counts={step.statusCounts} onProblems={problem} />
        {related > 0 && <span aria-hidden="true" className="block border-t-2 border-dotted border-shell-foreground"
          style={{ width: `${100 * related / step.runs.length}%` }} />}
      </> : <>
      <StepShape
        step={step}
        sweeps={sweeps}
        folded={folded}
        selection={selection}
        presentation={presentation}
        applySelection={applySelection}
        onPresentationChange={onPresentationChange}
        newSweep={newSweep}
      />
      <StatusCounts id={`${id}-status`} counts={step.statusCounts} onProblems={problem} />
      <PipelineRunMarks
        step={step}
        sweeps={sweeps}
        folded={folded}
        selection={selection}
        selected={selected}
        upstream={upstream}
        downstream={downstream}
        applySelection={applySelection}
      />
      {step.inputs.some((input) => input.subset) && (
        <ul className="-mx-3 divide-y divide-shell-divider border-t border-shell-divider">
          {step.inputs.map(({ id, label, subset }) => subset && (
            <li key={id} className="relative px-3 py-1 text-xs text-shell-muted">
              <span aria-hidden="true" data-map-anchor={mapAnchor(step.id, 'in', id)} className="pointer-events-none absolute left-0 top-1/2" />
              <strong className="block font-medium">{label}</strong>
              <div className="flex min-w-0 flex-wrap items-baseline gap-x-1">
                {subset.terms.length > 0 && <span className="min-w-0 break-words">{subset.terms.join(' · ')}</span>}
                <span className="shrink-0 whitespace-nowrap tabular-nums">{studioCountOf(subset.count, subset.total)}</span>
              </div>
            </li>
          ))}
        </ul>
      )}
      {selection && (
        <p className="flex items-baseline justify-between gap-x-1 text-xs text-shell-muted">
          {terms.length > 0 && <span className="min-w-0 truncate">{terms.join(' · ')}</span>}
          <span className="shrink-0 whitespace-nowrap tabular-nums">{studioCountOf(selected.size, step.runs.length)}</span>
        </p>
      )}
      {newStep && (
        <button type="button" className={`${smallButton} absolute -right-3 top-1/2 -translate-y-1/2`}
          {...commandAttributes(newStep)} disabled={!newStep.enabled} onClick={newStep.invoke}>
          <Plus aria-hidden="true" className="h-3 w-3" />
        </button>
      )}
      </>}
    </article>
  );
});

type MapLayout = ReturnType<typeof layoutPipelineMap>;

function MapDiagram({
  layout,
  children,
  clearFocus,
  strongEdges,
  viewportRef,
  fitReady,
}: {
  layout: MapLayout;
  children: (compact: boolean) => ReactNode;
  clearFocus: () => void;
  strongEdges: ReadonlySet<string>;
  viewportRef?: Ref<CanvasViewport>;
  fitReady?: boolean;
}) {
  const [grid, setGrid] = useState<HTMLDivElement | null>(null);
  const { ref, width, height } = useElementSize<HTMLDivElement>();
  const { getViewport } = useReactFlow();
  const fullHeights = useRef(new Map<string, number>());
  const visibleSize = useRef({ width: 0, height: 0 });
  if (width && height) visibleSize.current = { width, height };
  const compact = !fullMapFits(
    layout.columns, visibleSize.current.width, visibleSize.current.height, fullHeights.current,
  );
  const [geometry, setGeometry] = useState<{
    width: number; height: number; compact: boolean;
    paths: readonly { id: string; d: string }[];
  }>();
  useLayoutEffect(() => {
    if (!grid) return;
    const measure = () => {
      const origin = grid.getBoundingClientRect();
      const { zoom } = getViewport();
      const positions = new Map<string, { x: number; y: number }>();
      for (const card of grid.querySelectorAll<HTMLElement>('[data-map-density="full"]')) {
        if (card.offsetHeight) fullHeights.current.set(card.dataset.mapStep!, card.offsetHeight);
      }
      for (const anchor of grid.querySelectorAll<HTMLElement>('[data-map-anchor]')) {
        const bounds = anchor.getBoundingClientRect();
        positions.set(anchor.dataset.mapAnchor!, { x: (bounds.left - origin.left) / zoom, y: (bounds.top - origin.top) / zoom });
      }
      const paths = layout.edges.flatMap(({ id, from, to, inputId }) => {
        const a = positions.get(mapAnchor(from, 'out'));
        const b = positions.get(mapAnchor(to, 'in', inputId)) ?? positions.get(mapAnchor(to, 'in'));
        if (!a || !b) return [];
        const bend = Math.max(16, (b.x - a.x) / 2);
        return [{ id, d: `M ${a.x} ${a.y} C ${a.x + bend} ${a.y}, ${b.x - bend} ${b.y}, ${b.x} ${b.y}` }];
      });
      setGeometry({ width: grid.offsetWidth, height: grid.offsetHeight, compact, paths });
    };
    measure();
    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(measure);
    observer.observe(grid);
    for (const card of grid.querySelectorAll('[data-map-card]')) observer.observe(card);
    return () => observer.disconnect();
  }, [grid, layout.edges, compact, getViewport]);
  const bounds = useMemo(() => geometry?.width && geometry.height
    ? { x: 0, y: 0, width: geometry.width, height: geometry.height } : null,
  [geometry?.width, geometry?.height]);
  return (
    <div ref={ref} className="min-h-0 min-w-0 flex-1" data-map-canvas>
      <StudioCanvas contentBounds={bounds} viewportRef={viewportRef} navigateContent minZoom={0.02}
        fitReady={fitReady !== false && geometry?.compact === compact && !!width && !!height}
        elementsSelectable={false} nodesFocusable={false} edgesFocusable={false} onPaneClick={clearFocus}>
        <ViewportPortal>
          <div ref={setGrid} className="pointer-events-none relative isolate grid w-max items-start"
            style={{ gridTemplateColumns: `repeat(${layout.columns.length}, max-content)`,
              columnGap: mapColumnGap, padding: mapPadding }}>
            <svg aria-hidden="true" className="pointer-events-none absolute inset-0 h-full w-full">
              {geometry?.paths.map(({ id, d }) => <path key={id} d={d} stroke="currentColor" fill="none"
                data-map-edge={id} data-map-emphasis={strongEdges.has(id) ? 'strong' : 'light'}
                className={strongEdges.has(id) ? 'text-shell-muted' : 'text-shell-divider'}
                strokeWidth={strongEdges.has(id) ? 2 : 1.5} />)}
            </svg>
            {children(compact)}
          </div>
        </ViewportPortal>
      </StudioCanvas>
    </div>
  );
}

function MapDrawer({
  steps,
  selection,
  related,
  focusedStepId,
  chooseStep,
  documentScope,
  newStep,
  problemSteps,
  onProblems,
  framed = false,
}: {
  steps: readonly PlanStepView[];
  selection: PipelineSelection | null;
  related: RunSetsByStep | null;
  focusedStepId?: string;
  chooseStep: ChooseStep;
  documentScope?: MapDocumentScope;
  newStep?: StudioCommand;
  problemSteps?: StudioPipelineMapProps['problemSteps'];
  onProblems?: StudioPipelineMapProps['onProblems'];
  framed?: boolean;
}) {
  const groups = new Map<string | null, PlanStepView[]>();
  for (const step of steps) {
    const group = step.group ?? null;
    const entries = groups.get(group);
    if (entries) entries.push(step);
    else groups.set(group, [step]);
  }
  return (
    <nav aria-label={studioName('steps')} className={`p-2 ${framed ? '' : 'w-52 shrink-0 overflow-auto border-r border-shell-divider bg-shell-chrome'}`}>
      <ul className="space-y-1">
        {Array.from(groups).map(([group, entries]) => (
          <li key={group ?? ''}>
            {group && <h3 className="px-1.5 text-xs font-medium text-shell-muted">{group}</h3>}
            <ul>{entries.map((step) => {
          const scope = stepSelection(selection, step.id);
          const count = related?.get(step.id)?.size ?? 0;
          const summary = related && focusedStepId !== step.id ? studioCountOf(count, step.runs.length) : undefined;
          const level = documentLevel(documentScope, step.id);
          return (
            <li key={step.id}>
              <button
                type="button"
                className={`w-full space-y-1 rounded border border-transparent p-1.5 text-left text-sm hover:bg-shell-hover aria-[current=step]:bg-shell-divider data-[map-dimmed=true]:border-dashed data-[map-dimmed=true]:border-shell-divider data-[document-scope=faint]:border-shell-divider ${documentLevelStyle} ${!scope ? documentLitStyle : ''}`}
                aria-current={scope ? 'step' : undefined}
                aria-label={studioAccessibleName('select', stepLabel(step, summary))}
                aria-pressed={documentScope ? level === 'lit' : undefined}
                data-map-dimmed={Boolean(related && !count && focusedStepId !== step.id) || undefined}
                data-document-scope={level}
                {...selectionButton((extend) => chooseStep(step.id, extend))}
              >
                <StepName step={step} />{' '}
                {summary && <span className="block text-xs tabular-nums">{summary}</span>}
              </button>
              <div className="px-1.5"><StatusCounts counts={step.statusCounts}
                onProblems={problemSteps?.has(step.id) && onProblems ? (opener) => onProblems(step.id, opener) : undefined} /></div>
            </li>
          );
        })}</ul></li>))}
        {newStep && (
          <li>
            <button type="button" className={`${smallButton} inline-flex items-center gap-1`}
              {...commandAttributes(newStep)} disabled={!newStep.enabled} onClick={newStep.invoke}>
              <Plus aria-hidden="true" className="h-3 w-3" />
              {newStep.label}
            </button>
          </li>
        )}
      </ul>
    </nav>
  );
}

export function StudioPipelineMapDrawer({ plan, state, onSelectionChange, newStepCommand, problemSteps, onProblems, documentScope, onChooseStep }: Pick<
  StudioPipelineMapProps, 'plan' | 'state' | 'onSelectionChange' | 'newStepCommand' | 'problemSteps' | 'onProblems' | 'documentScope' | 'onChooseStep'
>) {
  const lineage = useMemo(() => indexPlanLineage(plan.edges), [plan.edges]);
  return <MapDrawer steps={plan.steps} selection={state.selection} related={state.focus ? lineage.related(state.focus) : null}
    focusedStepId={state.focus?.stepId} documentScope={documentScope}
    chooseStep={chooseMapStep((change) => onSelectionChange(change(state)), onChooseStep)}
    newStep={plan.steps.length ? newStepCommand?.() : undefined} problemSteps={problemSteps} onProblems={onProblems} framed />;
}

function MapLine({
  plan,
  layout,
  selection,
  chooseStep,
  documentScope,
}: {
  plan: PlanView;
  layout: MapLayout;
  selection: PipelineSelection | null;
  chooseStep: ChooseStep;
  documentScope?: MapDocumentScope;
}) {
  const selected = plan.steps.filter(({ id }) => stepSelection(selection, id));
  const center = selected.length === 1 ? selected[0] : null;
  const before = center ? plan.steps.filter(({ id }) => layout.upstream.get(center.id)?.has(id)) : [];
  const after = center ? plan.steps.filter(({ id }) => layout.downstream.get(center.id)?.has(id)) : [];
  const groups = documentScope ? [plan.steps] : [before, selected.length ? selected : plan.steps, after].filter((steps) => steps.length);
  return (
    <div className="flex items-center gap-2 overflow-auto p-2">
      {groups.map((steps, index) => (
        <div key={index} className="flex shrink-0 items-center gap-2">
          {index > 0 && <ChevronRight aria-hidden="true" className="h-4 w-4 text-shell-muted" />}
          {steps.map((step) => (
            <button
              key={step.id}
              type="button"
              className={`studio-map-line-step ${smallButton} border-2 text-sm ${documentLevelStyle} ${!stepSelection(selection, step.id) ? documentLitStyle : ''}`}
              style={{ borderColor: stepSelection(selection, step.id)?.kind === 'steps' ? 'var(--text-primary)' : studioStep(step.kind).tint }}
              aria-label={studioAccessibleName('select', stepLabel(step))}
              aria-current={stepSelection(selection, step.id) ? 'step' : undefined}
              aria-pressed={documentScope ? documentScope.lit.has(step.id) : undefined}
              data-map-selection={stepSelection(selection, step.id)?.kind}
              data-document-scope={documentLevel(documentScope, step.id)}
              {...selectionButton((extend) => chooseStep(step.id, extend))}
            >
              <StepName step={step} />
            </button>
          ))}
        </div>
      ))}
    </div>
  );
}

export function StudioPipelineMap({
  plan,
  state,
  onSelectionChange,
  onChooseStep,
  documentScope,
  mode = 'map',
  drawerOpen = false,
  presentation,
  onPresentationChange,
  newStepCommand,
  newSweepCommand,
  problemSteps,
  onProblems,
  activeStepId,
  viewportRef,
  fitReady,
}: StudioPipelineMapProps) {
  const current = useRef(state);
  useLayoutEffect(() => { current.current = state; }, [state]);
  const applySelection = useCallback<ApplyMapSelection>((change) => {
    onSelectionChange(change(current.current));
  }, [onSelectionChange]);
  const chooseStep = chooseMapStep(applySelection, onChooseStep);
  const clearFocus = useCallback(() => {
    if (current.current.focus) onSelectionChange({ ...current.current, focus: null });
  }, [onSelectionChange]);
  const layout = useMemo(() => layoutPipelineMap(plan), [plan]);
  const selectedSteps = state.selection?.kind === 'steps' ? state.selection.stepIds
    : state.selection ? [state.selection.stepId] : [];
  const primaryStep = activeStepId ?? selectedSteps[0];
  const firstStep = newStepCommand?.();
  const drawDrawer = drawerOpen && mode === 'map';
  const lineage = useMemo(() => indexPlanLineage(plan.edges), [plan.edges]);
  const focus = state.focus;
  const upstream = useMemo(() => focus ? lineage.upstream([focus]) : null, [lineage, focus]);
  const downstream = useMemo(() => focus ? lineage.downstream([focus]) : null, [lineage, focus]);
  const related = useMemo(
    () => drawDrawer && focus ? lineage.related(focus) : null,
    [drawDrawer, lineage, focus],
  );
  const ancestry = mapStepLineage(selectedSteps, layout.upstream, layout.downstream);
  const focusPairs = useMemo(
    () => focus && upstream && downstream ? focusedMapEdgePairs(plan.edges, focus, upstream, downstream) : null,
    [focus, upstream, downstream, plan.edges],
  );
  const strongEdges = new Set(layout.edges.filter(({ from, to }) =>
    focusPairs ? focusPairs.has(mapEdgePair(from, to)) : !documentScope &&
      (ancestry.upstream.has(from) && ancestry.upstream.has(to) ||
        ancestry.downstream.has(from) && ancestry.downstream.has(to)),
  ).map(({ id }) => id));
  return (
    <section
      aria-label={studioName('pipelineMap')}
      className="flex h-full min-h-0 min-w-0 overflow-hidden bg-shell-subtle"
      onKeyDown={(event) => {
        if (event.key === 'Escape' && state.focus) {
          event.stopPropagation();
          clearFocus();
        }
      }}
    >
      {!plan.steps.length ? (
        firstStep && (
          <button type="button" className={`${smallButton} m-4 grid h-11 w-[150px] shrink-0 place-items-center self-start border border-dashed border-shell-muted`}
            {...commandAttributes(firstStep)} disabled={!firstStep.enabled} onClick={firstStep.invoke}>
            <Plus aria-hidden="true" className="h-4 w-4" />
          </button>
        )
      ) : mode === 'line' ? (
        <MapLine plan={plan} layout={layout} selection={state.selection} chooseStep={chooseStep} documentScope={documentScope} />
      ) : (
        <>
          {drawDrawer && (
            <MapDrawer
              steps={layout.columns.flat()}
              selection={state.selection}
              related={related}
              focusedStepId={state.focus?.stepId}
              chooseStep={chooseStep}
              documentScope={documentScope}
              newStep={firstStep}
              problemSteps={problemSteps}
              onProblems={onProblems}
            />
          )}
          <ReactFlowProvider><MapDiagram layout={layout} clearFocus={clearFocus}
            strongEdges={strongEdges} viewportRef={viewportRef} fitReady={fitReady}>
            {(compact) => layout.columns.map((steps, index) => (
              <div key={index} data-canvas-column={index} className="relative flex flex-col items-start" style={{ gap: mapRowGap }}>
                {steps.map((step) => (
                  <StepCard
                    key={step.id}
                    step={step}
                    selection={stepSelection(state.selection, step.id)}
                    presentation={presentation?.[step.id] ?? emptyPresentation}
                    applySelection={applySelection}
                    chooseStep={chooseStep}
                    documentScope={documentScope}
                    compact={compact && step.id !== primaryStep}
                    dimmed={!documentScope && selectedSteps.length > 0 &&
                      !ancestry.upstream.has(step.id) && !ancestry.downstream.has(step.id)}
                    upstream={upstream?.get(step.id)}
                    downstream={downstream?.get(step.id)}
                    onPresentationChange={onPresentationChange}
                    newStep={newStepCommand?.({ stepId: step.id,
                      selection: state.selection?.kind !== 'steps' && state.selection?.stepId === step.id ? state.selection : null,
                    }, step.label)}
                    newSweep={newSweepCommand?.(step.id, step.label)}
                    onProblems={problemSteps?.has(step.id) ? onProblems : undefined}
                  />
                ))}
              </div>
            ))}
          </MapDiagram></ReactFlowProvider>
        </>
      )}
    </section>
  );
}
