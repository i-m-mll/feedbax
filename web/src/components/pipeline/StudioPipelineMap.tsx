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
import { appTip } from '@window-shell/browser/tooltip';
import { Card, CardHead, type CardStatus } from '@/components/ui/Card';
import { ScrollFade } from '@/components/ui/ScrollFade';
import { StatusMark, worstStatus } from '@/components/ui/StatusMark';
import {
  studioAccessibleName,
  studioCountOf,
  studioName,
  studioRunStatus,
  studioStep,
  type StudioResultState,
} from '@/components/activity/studioVocabulary';
import { bindStudioCommand, commandAttributes, type StudioCommand } from '@/features/projectWindow/studioCommands';
import { StudioCanvas, type CanvasViewport } from '@/components/canvas/StudioCanvas';
import { Button, IconButton } from '@/components/ui/Button';
import { ListRow } from '@/components/ui/ListRow';
import { useElementSize } from '@/hooks/useElementSize';
import type { PlanStepView, PlanSweepView, PlanView } from '@/features/pipeline/planView';
import {
  selectionButton,
  indexPlanLineage,
  selectedRuns,
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
import type { RunMenu } from './RunContextMenu';
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
  runMenu?: RunMenu;
}

const emptyPresentation: PipelineCardPresentation = {};

function stepRunSelection(selection: PipelineSelection | null, stepId: string) {
  return selection?.kind !== 'steps' && selection?.stepId === stepId ? selection : null;
}

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
  const className = 'inline-flex items-center gap-1 whitespace-nowrap tabular-nums';
  const statuses = Object.entries(counts) as [StudioResultState, number][];
  if (onProblems && counts.cannot_run === undefined) statuses.push(['cannot_run', 0]);
  return (
    <span id={id} className="flex flex-wrap gap-x-2 gap-y-1 text-xs text-shell-muted">
      {statuses.map(([status, count]) => {
        const problem = status === 'cannot_run' && onProblems;
        if (count <= 0 && !problem) return null;
        const word = studioRunStatus(status);
        const name = count > 0 ? `${count} ${word}` : word;
        const mark = <><StatusMark status={status} />{count > 0 && count}</>;
        return problem
          ? <Button variant="bare" size="content" key={status} className={className} aria-haspopup="dialog"
            aria-label={name} {...appTip(word)} onClick={(event) => problem(event.currentTarget)}>{mark}</Button>
          : <span key={status} role="img" aria-label={name} className={className} {...appTip(word)}>{mark}</span>;
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
    <span className="studio-card__title flex min-w-0 items-center gap-1.5">
      <span className="studio-card__glyph" {...appTip(name)}><Icon aria-hidden="true" className="size-4" /></span>
      <span className="truncate font-medium">{step.label}</span>
    </span>
  );
}

function cardStatus(counts: PlanStepView['statusCounts']): CardStatus | undefined {
  const has = (...states: StudioResultState[]) => states.some((state) => (counts[state] ?? 0) > 0);
  if (has('running', 'starting', 'collecting', 'stopping', 'checking')) return 'busy';
  if (has('failed', 'corrupt', 'cannot_run', 'blocked', 'interrupted')) return 'failure';
  const states = Object.keys(counts);
  return states.length && states.every((state) => state === 'succeeded') ? 'success' : undefined;
}

export function bindSweepFold(
  stepId: string, sweep: PlanSweepView, folded: boolean, presentation: PipelineCardPresentation,
  save: NonNullable<StudioPipelineMapProps['onPresentationChange']>,
) {
  return bindStudioCommand(folded ? 'unfold' : 'fold',
    () => save(stepId, { ...presentation, foldedSweepId: folded ? null : sweep.id }), { subject: sweep.label });
}

function FoldSweep({ command }: { command: StudioCommand }) {
  return <Button size="mark" {...commandAttributes(command)} onClick={command.invoke}>
    {studioName(command.name)}
  </Button>;
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
              <Button
                variant="ghost"
                size="mark"
                className="aria-expanded:bg-shell-selected"
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
              </Button>
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
                      <IconButton variant="default" size="mark" label={studioAccessibleName('moveEarlier', sweep.label)}
                        icon={<ArrowLeft aria-hidden="true" />} disabled={index === 0}
                        onClick={() => moveSweep(index, index - 1)} />
                      <IconButton variant="default" size="mark" label={studioAccessibleName('moveLater', sweep.label)}
                        icon={<ArrowRight aria-hidden="true" />} disabled={index === sweeps.length - 1}
                        onClick={() => moveSweep(index, index + 1)} />
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        );
      })}
      {newSweep && (
        <IconButton variant="default" size="mark" label={newSweep.label} icon={<Plus aria-hidden="true" />}
          {...commandAttributes(newSweep)} disabled={!newSweep.enabled} onClick={newSweep.invoke} />
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
  runMenu,
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
  runMenu?: RunMenu;
}) {
  const id = useId();
  const sweeps = useMemo(() => orderedMapSweeps(step, presentation), [step, presentation]);
  const folded = foldedMapSweep(step, sweeps, presentation);
  const level = documentLevel(documentScope, step.id);
  const selected = useMemo(() => new Set(
    (compact ? [] : level === 'lit' ? step.runs : selectedRuns(step, selection)).map(({ id }) => id),
  ), [step, selection, level, compact]);
  const related = (upstream?.size ?? 0) + (downstream?.size ?? 0) -
    (upstream && downstream ? [...upstream].filter((id) => downstream.has(id)).length : 0);
  const problem = onProblems && ((opener: HTMLButtonElement) => onProblems(step.id, opener));
  return (
    <Card
      data-map-card
      data-canvas-item
      data-map-step={step.id}
      data-map-density={compact ? 'compact' : 'full'}
      data-map-selection={selection?.kind}
      data-map-related={related || undefined}
      data-map-dimmed={dimmed || undefined}
      data-document-scope={level}
      selected={selection?.kind === 'steps'}
      status={cardStatus(step.statusCounts)}
      role="article"
      aria-label={studioAccessibleName(step.kind, step.label)}
      className={`studio-map-card nopan pointer-events-auto text-sm focus-within:z-20 data-[map-dimmed=true]:border-dashed ${documentLevelStyle} ${!selection ? documentLitStyle : ''}`}
      style={{ width: compact ? compactMapCardWidth : mapCardWidth }}
    >
      <span id={`${id}-summary`} className="sr-only">
        {sweeps.map((sweep) => `${sweep.label} ${sweep.values.length}`).join(' · ')}
      </span>
      <CardHead className="relative">
        <span aria-hidden="true" data-map-anchor={mapAnchor(step.id, 'in')} className="pointer-events-none absolute -left-px top-1/2" />
        <span aria-hidden="true" data-map-anchor={mapAnchor(step.id, 'out')} className="pointer-events-none absolute -right-px top-1/2" />
        <Button
          variant="bare"
          size="content"
          align="start"
          data-canvas-focus
          className="studio-map-card__head flex w-full px-3 py-2"
          aria-label={studioAccessibleName('select', stepLabel(step))}
          aria-describedby={[`${id}-summary`, compact ? `${id}-status` : '', related ? `${id}-related` : ''].filter(Boolean).join(' ')}
          aria-current={selection?.kind === 'steps' ? 'step' : undefined}
          aria-pressed={documentScope ? level === 'lit' : selection?.kind === 'steps'}
          {...selectionButton((extend) => chooseStep(step.id, extend))}
        >
          <StepName step={step} />
        </Button>
      </CardHead>
      {related > 0 && <span id={`${id}-related`} className="sr-only">{studioCountOf(related, step.runs.length)}</span>}
      {compact ? <div className="space-y-2 px-3 pb-2">
        <StatusCounts id={`${id}-status`} counts={step.statusCounts} onProblems={problem} />
        {related > 0 && <span aria-hidden="true" className="block border-t-[1.5px] border-dotted border-shell-foreground"
          style={{ width: `${100 * related / step.runs.length}%` }} />}
      </div> : <>
      <div className="space-y-2 px-3 pb-2 empty:hidden">
        {(sweeps.length > 0 || newSweep) && <StepShape
          step={step}
          sweeps={sweeps}
          folded={folded}
          selection={selection}
          presentation={presentation}
          applySelection={applySelection}
          onPresentationChange={onPresentationChange}
          newSweep={newSweep}
        />}
        <PipelineRunMarks
          step={step}
          sweeps={sweeps}
          folded={folded}
          selection={selection}
          selected={selected}
          upstream={upstream}
          downstream={downstream}
          applySelection={applySelection}
          runMenu={runMenu}
        />
      </div>
      {step.inputs.some((input) => input.subset) && (
        <ul className="divide-y divide-shell-divider border-t border-shell-divider">
          {step.inputs.map(({ id, label, subset }) => subset && (
            <li key={id} className="relative px-3 py-1 text-xs text-shell-muted">
              <span aria-hidden="true" data-map-anchor={mapAnchor(step.id, 'in', id)} className="pointer-events-none absolute -left-px top-1/2" />
              <strong className="block font-medium">{label}</strong>
              <div className="flex min-w-0 flex-wrap items-baseline gap-x-1">
                {subset.terms.length > 0 && <span className="min-w-0 break-words">{subset.terms.join(' · ')}</span>}
                <span className="shrink-0 whitespace-nowrap tabular-nums">{studioCountOf(subset.count, subset.total)}</span>
              </div>
            </li>
          ))}
        </ul>
      )}
      {newStep && (
        <IconButton variant="default" size="mark" label={newStep.label} icon={<Plus aria-hidden="true" />}
          className="studio-card__attached"
          {...commandAttributes(newStep)} disabled={!newStep.enabled} onClick={newStep.invoke} />
      )}
      </>}
    </Card>
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
              <defs>
                <linearGradient id="studio-map-turbo-edge" gradientUnits="userSpaceOnUse" x1={0} x2={geometry?.width ?? 1} y1={0} y2={0}>
                  <stop offset="0%" style={{ stopColor: 'var(--turbo-start)' }} />
                  <stop offset="100%" style={{ stopColor: 'var(--turbo-end)' }} />
                </linearGradient>
              </defs>
              {geometry?.paths.map(({ id, d }) => <path key={id} d={d} stroke="currentColor" fill="none"
                data-map-edge={id} data-map-emphasis={strongEdges.has(id) ? 'strong' : 'light'}
                className={`studio-map-edge ${strongEdges.has(id) ? 'text-shell-muted' : 'text-shell-divider'}`}
                strokeWidth={strongEdges.has(id) ? 2 : 1.5} />)}
            </svg>
            {children(compact)}
          </div>
        </ViewportPortal>
      </StudioCanvas>
    </div>
  );
}

interface DrawerProps {
  selection: PipelineSelection | null;
  related: RunSetsByStep | null;
  focusedStepId?: string;
  chooseStep: ChooseStep;
  applySelection: ApplyMapSelection;
  documentScope?: MapDocumentScope;
}

function StepRow({ step, selection, related, focusedStepId, chooseStep, applySelection, documentScope }: DrawerProps & {
  step: PlanStepView;
}) {
  const scope = stepSelection(selection, step.id);
  const count = related?.get(step.id)?.size ?? 0;
  const summary = related && focusedStepId !== step.id ? studioCountOf(count, step.runs.length) : undefined;
  const level = documentLevel(documentScope, step.id);
  const worst = worstStatus(step.statusCounts);
  const choose = selectionButton((extend) => chooseStep(step.id, extend));
  const { Icon } = studioStep(step.kind);
  const word = worst && studioRunStatus(worst.status);
  const shown = worst && step.runs.length > 1 ? worst.count : undefined;
  return (
    <ListRow
      selected={!!scope}
      data-map-dimmed={Boolean(related && !count && focusedStepId !== step.id) || undefined}
      data-document-scope={level}
      className={`data-[document-scope=faint]:text-shell-muted data-[map-dimmed=true]:text-shell-muted ${!scope ? documentLitStyle : ''}`}
      glyph={<Icon aria-hidden="true" />}
      onClick={choose.onClick}
      name={<Button variant="bare" size="content" align="start" className="block w-full truncate"
        aria-current={scope ? 'step' : undefined}
        aria-label={studioAccessibleName('select', stepLabel(step, summary))}
        aria-pressed={documentScope ? level === 'lit' : undefined}
        onKeyDown={choose.onKeyDown}>{step.label}</Button>}
      trailing={worst && word && (
        <Button variant="bare" size="content" className="inline-flex items-center gap-1 rounded-sm tabular-nums"
          aria-label={studioAccessibleName('select', [shown, word].filter(Boolean).join(' '))} {...appTip(word)}
          onClick={(event) => {
            event.stopPropagation();
            const runIds = step.runs.filter(({ state }) => state === worst.status).map(({ id }) => id);
            applySelection(() => ({ selection: { kind: 'rows', stepId: step.id, runIds }, focus: null }));
          }}>
          <StatusMark status={worst.status} />{shown}
        </Button>
      )}
    />
  );
}

function MapDrawer({ steps, newStep, framed = false, ...props }: DrawerProps & {
  steps: readonly PlanStepView[];
  newStep?: StudioCommand;
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
    <nav aria-label={studioName('steps')} className={`flex min-h-0 flex-col ${framed ? 'h-full' : 'w-52 shrink-0 border-r border-shell-divider bg-shell-chrome'}`}>
      {newStep && (
        <div className="shrink-0 px-2 pt-2">
          <ListRow role="button" tabIndex={0} glyph={<Plus aria-hidden="true" />} name={studioName('newStep')}
            {...commandAttributes(newStep)} disabled={!newStep.enabled}
            onClick={newStep.invoke}
            onKeyDown={(event) => {
              if (event.key !== 'Enter' && event.key !== ' ') return;
              event.preventDefault();
              newStep.invoke();
            }} />
        </div>
      )}
      <ScrollFade axis="y" className="min-h-0 flex-1 p-2">
        <ul className="space-y-1">
          {Array.from(groups).map(([group, entries]) => (
            <li key={group ?? ''}>
              {group && <h3 className="px-row-inline text-xs font-medium text-shell-muted">{group}</h3>}
              <ul>{entries.map((step) => <li key={step.id}><StepRow step={step} {...props} /></li>)}</ul>
            </li>
          ))}
        </ul>
      </ScrollFade>
    </nav>
  );
}

export function StudioPipelineMapDrawer({ plan, state, onSelectionChange, newStepCommand, documentScope, onChooseStep }: Pick<
  StudioPipelineMapProps, 'plan' | 'state' | 'onSelectionChange' | 'newStepCommand' | 'documentScope' | 'onChooseStep'
>) {
  const lineage = useMemo(() => indexPlanLineage(plan.edges), [plan.edges]);
  const applySelection: ApplyMapSelection = (change) => onSelectionChange(change(state));
  return <MapDrawer steps={plan.steps} selection={state.selection} related={state.focus ? lineage.related(state.focus) : null}
    focusedStepId={state.focus?.stepId} documentScope={documentScope} applySelection={applySelection}
    chooseStep={chooseMapStep(applySelection, onChooseStep)}
    newStep={newStepCommand?.() ?? bindStudioCommand('newStep', () => {}, { enabled: false })} framed />;
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
            <Button
              key={step.id}
              variant="bare"
              size="content"
              className={`studio-card studio-map-line-step px-2 py-1 text-sm ${documentLevelStyle} ${!stepSelection(selection, step.id) ? documentLitStyle : ''}`}
              data-selected={stepSelection(selection, step.id)?.kind === 'steps' || undefined}
              aria-label={studioAccessibleName('select', stepLabel(step))}
              aria-current={stepSelection(selection, step.id) ? 'step' : undefined}
              aria-pressed={documentScope ? documentScope.lit.has(step.id) : undefined}
              data-map-selection={stepSelection(selection, step.id)?.kind}
              data-document-scope={documentLevel(documentScope, step.id)}
              {...selectionButton((extend) => chooseStep(step.id, extend))}
            >
              <StepName step={step} />
            </Button>
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
  runMenu,
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
  function insertAt(target: EventTarget) {
    const card = target instanceof Element ? target.closest<HTMLElement>('[data-map-step]') : null;
    const stepId = card?.dataset.mapStep;
    const command = stepId ? newStepCommand?.({ stepId, selection: stepRunSelection(state.selection, stepId) },
      plan.steps.find(({ id }) => id === stepId)?.label) : firstStep;
    command?.invoke();
  }
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
      className="relative flex h-full min-h-0 min-w-0 overflow-hidden bg-shell-subtle"
      onKeyDown={(event) => {
        if (event.key === 'Escape' && state.focus) {
          event.stopPropagation();
          clearFocus();
        }
        if (mode === 'map' && (event.key === 'ContextMenu' || (event.key === 'F10' && event.shiftKey))) {
          event.preventDefault();
          insertAt(event.target);
        }
      }}
      onContextMenu={mode === 'map' ? (event) => {
        event.preventDefault();
        insertAt(event.target);
      } : undefined}
    >
      {!plan.steps.length ? null : mode === 'line' ? (
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
              applySelection={applySelection}
              documentScope={documentScope}
              newStep={firstStep}
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
                    newStep={newStepCommand?.({ stepId: step.id, selection: stepRunSelection(state.selection, step.id) }, step.label)}
                    newSweep={newSweepCommand?.(step.id, step.label)}
                    onProblems={problemSteps?.has(step.id) ? onProblems : undefined}
                    runMenu={runMenu}
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
