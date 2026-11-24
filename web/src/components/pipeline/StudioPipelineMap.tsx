import { ChevronRight, Plus } from 'lucide-react';
import { ReactFlowProvider, ViewportPortal, useReactFlow } from '@xyflow/react';
import {
  memo,
  useCallback,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type MouseEvent,
  type PointerEvent,
  type ReactNode,
  type Ref,
} from 'react';
import { appTip } from '@window-shell/browser/tooltip';
import { useRegionPending } from '@/components/activity/StudioRegion';
import { Card, CardHead, cardTint, type CardStatus } from '@/components/ui/Card';
import { ScrollFade } from '@/components/ui/ScrollFade';
import { StepKindMark } from './StepKindMark';
import { StatusMark, worstStatus } from '@/components/ui/StatusMark';
import {
  studioAccessibleName,
  studioCountOf,
  studioName,
  studioRunReason,
  studioRunStatus,
  studioStep,
  type StudioResultState,
} from '@/components/activity/studioVocabulary';
import { useCommandSource } from '@/features/projectWindow/heldGestures';
import { commandAttributes, unboundStudioCommand, type StudioCommand } from '@/features/projectWindow/studioCommands';
import type { ReferencePick, RuleChoice } from '@/features/projectWindow/elementSelection';
import { StudioCanvas, type CanvasViewport } from '@/components/canvas/StudioCanvas';
import { Button, IconButton } from '@/components/ui/Button';
import { InlineEdit } from '@/components/ui/InlineEdit';
import { ListRow } from '@/components/ui/ListRow';
import { Tree, type TreeHandle, type TreeNode } from '@/components/ui/Tree';
import { Menu, MenuContent, MenuTrigger } from '@/components/ui/Menu';
import { useElementSize } from '@/hooks/useElementSize';
import type { PlanStepView, PlanView } from '@/features/pipeline/planView';
import {
  extendsSelection,
  selectionButton,
  indexPlanLineage,
  selectedRuns,
  selectStep,
  type PipelineSelection,
  type PipelineSelectionState,
  type RunSetsByStep,
} from '@/features/pipeline/selection';
import { PipelineRunMarks, type ApplyMapSelection, type AxisMenu } from './PipelineRunMarks';
import { CommandContextMenu, CommandMenuItems, type RunMenu, type StepMenu } from './CommandContextMenu';
import {
  compactMapCardWidth,
  focusedMapEdgePairs,
  fullMapFits,
  layoutPipelineMap,
  mapAnchor,
  mapCardWidth,
  mapEdgePair,
  mapStepLineage,
  orderedMapSweeps,
  placePipelineMap,
  type MapLayout,
  type MapPlacement,
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

/** How a step is drawn against the document view's scope, or, outside one, whether an edit of the selection reaches it. */
function documentLevel(scope: MapDocumentScope | undefined, step: string, reached = false) {
  return !scope ? reached ? 'lit' : undefined : scope.lit.has(step) ? 'lit' : scope.users.has(step) ? 'plain' : 'faint';
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
  /** The runs a rule field being written chooses, boxed on their step's card. */
  choosing?: RuleChoice | null;
  state: PipelineSelectionState;
  onSelectionChange: (state: PipelineSelectionState) => void;
  documentScope?: MapDocumentScope;
  /** Outside a document view, the steps an edit of the selected step reaches besides it: those based on it. */
  reached?: ReadonlySet<string>;
  onChooseStep?: ChooseStep;
  mode?: 'map' | 'line';
  drawerOpen?: boolean;
  presentation?: Readonly<Record<string, PipelineCardPresentation>>;
  onPresentationChange?: (stepId: string, presentation: PipelineCardPresentation) => void;
  newStepCommand?: (source?: PipelineMapStepSource, subject?: string) => StudioCommand | undefined;
  newSweepCommand?: (stepId: string, subject?: string) => StudioCommand | undefined;
  /** The open menu of a step's sweepable fields, anchored at Add Sweep in its card's head. */
  sweepMenu?: SweepMenu;
  problemSteps?: ReadonlySet<string>;
  onProblems?: (stepId: string, opener: HTMLButtonElement) => void;
  activeStepId?: string | null;
  viewportRef?: Ref<CanvasViewport>;
  fitReady?: boolean;
  runMenu?: RunMenu;
  axisMenu?: AxisMenu;
  stepMenu?: StepMenu;
  /** An armed reference pick: a card it accepts fills the reference when its head is chosen. */
  pick?: ReferencePick | null;
  /** Wires dragged from a card's output to a card that takes it as one more selection of an input. */
  connect?: MapConnect;
  rename?: MapStepRename;
  /** Go into a step: its views, with the keyboard in the slot that shows its Runs. */
  onEnterStep?: (stepId: string) => void;
  /** The plan shown is of an earlier revision while the shown revision's plan is read. */
  pending?: boolean;
}

export interface MapConnect {
  accepts(source: string, target: string): boolean;
  connect(source: string, target: string): void;
}

export type MapRenameSurface = 'card' | 'line';

/** A step renamed in place: `editing` names the step whose name is being edited, and the surface it is edited on. */
export interface MapStepRename {
  readonly editing: { readonly stepId: string; readonly surface: MapRenameSurface } | null;
  setEditing(editing: MapStepRename['editing']): void;
  /** Returns false to refuse the name and stay in editing. */
  commit(stepId: string, name: string): boolean;
  /**
   * A step name to take keyboard focus once it is drawn: after a rename ends, when focus was not moved
   * elsewhere, or after history moves a rename (`force`).
   */
  readonly focus: { readonly stepId: string; readonly surface: MapRenameSurface; readonly force?: boolean } | null;
  focused(): void;
}

const renameFocusTarget: Record<MapRenameSurface, (stepId: string) => string> = {
  card: (stepId) => `[data-map-step="${stepId}"] button[data-canvas-focus]`,
  line: (stepId) => `[data-map-line-step="${stepId}"]`,
};

/** Focus the step name `rename.focus` asks for when this region draws it. */
function useRenameFocus(rename: MapStepRename | undefined, root: { readonly current: HTMLElement | null }) {
  useLayoutEffect(() => {
    const target = rename?.focus;
    const element = target && root.current?.querySelector<HTMLElement>(renameFocusTarget[target.surface](target.stepId));
    if (!element) return;
    const active = document.activeElement;
    if (target.force || !active || active === document.body) element.focus();
    rename.focused();
  });
}

function renamingStep(rename: MapStepRename | undefined, stepId: string, surface: MapRenameSurface) {
  return rename?.editing?.stepId === stepId && rename.editing.surface === surface ? rename : undefined;
}

/** Return goes into the step and F2 renames it; a double-click on its name renames it, elsewhere goes in. */
function stepGestures(
  rename: MapStepRename | undefined, enter: ((stepId: string) => void) | undefined,
  stepId: string, surface: MapRenameSurface, onKeyDown: ReturnType<typeof selectionButton>['onKeyDown'],
) {
  return {
    onKeyDown: (event: KeyboardEvent) => {
      if (event.key === 'F2' && rename) {
        event.preventDefault();
        rename.setEditing({ stepId, surface });
      } else if (event.key === 'Enter' && enter && !extendsSelection(event)) {
        event.preventDefault();
        enter(stepId);
      } else onKeyDown(event);
    },
    onDoubleClick: (event: MouseEvent) => {
      if (rename && event.target instanceof Element && event.target.closest('[data-step-name]')) rename.setEditing({ stepId, surface });
      else enter?.(stepId);
    },
  };
}

export interface SweepMenu {
  readonly stepId: string;
  readonly items: ReactNode;
  close(): void;
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

function StatusCounts({ id, counts, refusal, onProblems }: {
  id?: string;
  counts: PlanStepView['statusCounts'];
  refusal?: string;
  onProblems?: (opener: HTMLButtonElement) => void;
}) {
  const className = 'inline-flex items-center gap-1 whitespace-nowrap tabular-nums';
  const statuses = Object.entries(counts) as [StudioResultState, number][];
  if ((onProblems || refusal) && counts.cannot_run === undefined) statuses.push(['cannot_run', 0]);
  return (
    <span id={id} className="flex flex-wrap gap-x-2 gap-y-1 text-xs text-shell-muted">
      {statuses.map(([status, count]) => {
        const problem = status === 'cannot_run' && onProblems;
        const refused = status === 'cannot_run' ? refusal : undefined;
        if (count <= 0 && !problem && !refused) return null;
        const mark = <>{count > 0 && <>{count}{' '}</>}<StatusMark status={status} label={studioRunStatus(status)} /></>;
        return problem
          ? <Button variant="bare" size="content" key={status} className={className} aria-haspopup="dialog"
            aria-description={refused} {...refused && appTip(studioRunReason({ reason: refused }))}
            onClick={(event) => problem(event.currentTarget)}>{mark}</Button>
          : <span key={status} className={className} aria-description={refused} {...refused && appTip(studioRunReason({ reason: refused }))}>{mark}</span>;
      })}
    </span>
  );
}

function stepLabel(step: PlanStepView, count?: string | number) {
  return [step.label, studioStep(step.kind).name, count].filter((part) => part !== undefined).join(' ');
}

function StepName({ step, rename }: { step: PlanStepView; rename?: MapStepRename }) {
  return (
    <span className="studio-card__title flex min-w-0 items-center gap-1.5">
      <span className="studio-card__glyph"><StepKindMark kind={step.kind} /></span>
      {rename ? <StepNameEdit step={step} rename={rename} /> : <span data-step-name className="truncate font-medium">{step.label}</span>}
    </span>
  );
}

function StepNameEdit({ step, rename }: { step: PlanStepView; rename: MapStepRename }) {
  return <InlineEdit value={step.label} editing className="font-medium"
    aria-label={studioAccessibleName('rename', step.label)}
    onEditingChange={(next) => { if (!next) rename.setEditing(null); }}
    onCommit={(name) => rename.commit(step.id, name)} />;
}

function cardStatus(counts: PlanStepView['statusCounts']): CardStatus | undefined {
  const has = (...states: StudioResultState[]) => states.some((state) => (counts[state] ?? 0) > 0);
  if (has('running', 'starting', 'collecting', 'stopping', 'checking')) return 'busy';
  if (has('failed', 'corrupt', 'cannot_run', 'blocked', 'interrupted')) return 'failure';
  const states = Object.keys(counts);
  return states.length && states.every((state) => state === 'succeeded') ? 'success' : undefined;
}

function NewSweep({ command, menu }: { command: StudioCommand; menu?: SweepMenu }) {
  // The items stay while the menu closes: removing the chosen one would send focus back to the trigger, away from
  // where the choice put it.
  const items = useRef<ReactNode>();
  if (menu) items.current = menu.items;
  return (
    <Menu open={!!menu} onOpenChange={(open) => { if (open) command.invoke(); else menu?.close(); }}>
      <MenuTrigger render={<IconButton variant="default" size="mark" label={command.label} icon={<Plus aria-hidden="true" />}
        className="studio-map-attention" {...commandAttributes(command)} disabled={!command.enabled} />} />
      <MenuContent aria-label={studioName('newSweep')}>{items.current}</MenuContent>
    </Menu>
  );
}

/** A press on a card that lands on none of its own controls is a press on its head. */
const cardControl = 'button, a, input, select, textarea, [role="button"], [role="menuitem"], [role="menu"], [role="dialog"]';

const StepCard = memo(function StepCard({
  step,
  selection,
  presentation,
  applySelection,
  chooseStep,
  documentScope,
  reached,
  onPresentationChange,
  newStep,
  newSweep,
  sweepMenu,
  onProblems,
  compact,
  dimmed,
  upstream,
  downstream,
  boxed,
  runMenu,
  axisMenu,
  stepMenu,
  rename,
  pick,
  connectable,
  enterStep,
}: {
  step: PlanStepView;
  selection: PipelineSelection | null;
  presentation: PipelineCardPresentation;
  applySelection: ApplyMapSelection;
  chooseStep: ChooseStep;
  documentScope?: MapDocumentScope;
  reached?: boolean;
  onPresentationChange?: StudioPipelineMapProps['onPresentationChange'];
  newStep?: StudioCommand;
  newSweep?: StudioCommand;
  sweepMenu?: SweepMenu;
  onProblems?: StudioPipelineMapProps['onProblems'];
  compact: boolean;
  dimmed: boolean;
  upstream?: ReadonlySet<string>;
  downstream?: ReadonlySet<string>;
  boxed?: ReadonlySet<string>;
  runMenu?: RunMenu;
  axisMenu?: AxisMenu;
  stepMenu?: StepMenu;
  rename?: MapStepRename;
  pick?: ReferencePick | null;
  /** A wire can be dragged from the card's output. */
  connectable?: boolean;
  enterStep?: (stepId: string) => void;
}) {
  const id = useId();
  const picked = { kind: 'step', step: step.id } as const;
  const pickable = pick ? pick.accepts(picked) : undefined;
  const sweeps = useMemo(() => orderedMapSweeps(step, presentation), [step, presentation]);
  const arrange = useMemo(() => onPresentationChange &&
    ((order: string[]) => onPresentationChange(step.id, { ...presentation, sweepOrder: order })),
  [onPresentationChange, step.id, presentation]);
  const level = documentLevel(documentScope, step.id, reached);
  const selected = useMemo(() => new Set(
    (compact ? [] : level === 'lit' ? step.runs : selectedRuns(step, selection)).map(({ id }) => id),
  ), [step, selection, level, compact]);
  const related = (upstream?.size ?? 0) + (downstream?.size ?? 0) -
    (upstream && downstream ? [...upstream].filter((id) => downstream.has(id)).length : 0);
  const problem = onProblems && ((opener: HTMLButtonElement) => onProblems(step.id, opener));
  const headSelection = selectionButton((extend) => {
    if (pick) {
      if (pickable) pick.choose(picked);
    } else chooseStep(step.id, extend);
  });
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
      data-pick={pickable === undefined ? undefined : pickable ? 'allowed' : 'refused'}
      selected={selection?.kind === 'steps'}
      status={cardStatus(step.statusCounts)}
      tint={studioStep(step.kind).tint}
      role="article"
      aria-label={studioAccessibleName(step.kind, step.label)}
      className={`studio-map-card nopan pointer-events-auto text-sm focus-within:z-20 data-[map-dimmed=true]:border-dashed ${documentLevelStyle} ${!selection ? documentLitStyle : ''}`}
      style={{ width: compact ? compactMapCardWidth : mapCardWidth }}
      onClick={(event) => {
        if (event.target instanceof Element && event.target.closest(cardControl)) return;
        headSelection.onClick(event);
      }}
    >
      <span id={`${id}-summary`} className="sr-only">
        {sweeps.map((sweep) => `${sweep.label} ${sweep.values.length}`).join(' · ')}
      </span>
      <CardHead className="relative flex items-center">
        <span aria-hidden="true" data-map-anchor={mapAnchor(step.id, 'in')} className="pointer-events-none absolute -left-px top-1/2" />
        <span aria-hidden="true" data-map-anchor={mapAnchor(step.id, 'out')} className="pointer-events-none absolute -right-px top-1/2" />
        {connectable && <span aria-hidden="true" data-map-handle={step.id}
          className="studio-map-attention nopan absolute -right-1.5 top-1/2 z-10 size-3 -translate-y-1/2 cursor-crosshair rounded-full border border-shell-muted bg-shell-chrome" />}
        {renamingStep(rename, step.id, 'card') ? (
          <div className="studio-map-card__head flex w-full px-3 py-2"><StepName step={step} rename={rename} /></div>
        ) : <CommandContextMenu commands={stepMenu && (() => stepMenu(step, { start: () => rename?.setEditing({ stepId: step.id, surface: 'card' }), surface: 'card' }))}
          trigger={<Button
          variant="bare"
          size="content"
          align="start"
          data-canvas-focus
          className="studio-map-card__head flex min-w-0 flex-1 px-3 py-2"
          aria-label={studioAccessibleName('select', stepLabel(step))}
          aria-describedby={[`${id}-summary`, compact ? `${id}-status` : '', related ? `${id}-related` : ''].filter(Boolean).join(' ')}
          aria-current={selection?.kind === 'steps' ? 'step' : undefined}
          aria-pressed={documentScope ? level === 'lit' : selection?.kind === 'steps'}
          {...headSelection}
          {...stepGestures(rename, pick ? undefined : enterStep, step.id, 'card', headSelection.onKeyDown)}
        >
          <StepName step={step} />
        </Button>} />}
        {!compact && newSweep && !renamingStep(rename, step.id, 'card') && (
          <span className="flex shrink-0 pr-2"><NewSweep command={newSweep} menu={sweepMenu} /></span>
        )}
      </CardHead>
      {related > 0 && <span id={`${id}-related`} className="sr-only">{studioCountOf(related, step.runs.length)}</span>}
      {compact ? <div className="space-y-2 px-3 pb-2">
        <StatusCounts id={`${id}-status`} counts={step.statusCounts} refusal={step.refusal} onProblems={problem} />
        {related > 0 && <span aria-hidden="true" className="block border-t-[1.5px] border-dotted border-shell-foreground"
          style={{ width: `${100 * related / step.runs.length}%` }} />}
      </div> : <>
      <div className="space-y-2 px-3 pb-2 empty:hidden">
        <PipelineRunMarks
          step={step}
          sweeps={sweeps}
          selection={selection}
          selected={selected}
          upstream={upstream}
          downstream={downstream}
          boxed={boxed}
          applySelection={applySelection}
          runMenu={runMenu}
          axisMenu={axisMenu}
          arrange={arrange}
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
          className="studio-card__attached studio-map-attention"
          {...commandAttributes(newStep)} disabled={!newStep.enabled} onClick={newStep.invoke} />
      )}
      </>}
    </Card>
  );
});

/** Where a wire is dropped: the card under the pointer, or none. */
function dropTarget(event: PointerEvent<Element>) {
  const card = document.elementFromPoint(event.clientX, event.clientY)?.closest<HTMLElement>('[data-map-step]');
  return card?.dataset.mapStep ?? null;
}

/** A card's height before it is measured; it is measured before the first paint. */
const unmeasuredCardHeight = 96;

function sameHeights(a: ReadonlyMap<string, number>, b: ReadonlyMap<string, number>) {
  return a.size === b.size && [...a].every(([id, height]) => b.get(id) === height);
}

function MapDiagram({
  layout,
  card,
  primaryStep,
  clearFocus,
  strongEdges,
  viewportRef,
  fitReady,
  wiring,
}: {
  layout: MapLayout;
  card: (step: PlanStepView, compact: boolean) => ReactNode;
  /** In compact density, the step whose card stays full. */
  primaryStep?: string;
  clearFocus: () => void;
  strongEdges: ReadonlySet<string>;
  viewportRef?: Ref<CanvasViewport>;
  fitReady?: boolean;
  /** A wire dragged from a card's handle: `start` with its card, `end` with the card it was dropped on. */
  wiring?: { start(source: string): void; end(source: string, target: string | null): void };
}) {
  const [grid, setGrid] = useState<HTMLDivElement | null>(null);
  const [wire, setWire] = useState<{ source: string; from: { x: number; y: number }; to: { x: number; y: number } }>();
  const { ref, width, height } = useElementSize<HTMLDivElement>();
  const { getViewport } = useReactFlow();
  const fullHeights = useRef(new Map<string, number>());
  const [heights, setHeights] = useState<ReadonlyMap<string, number>>(() => new Map());
  const visibleSize = useRef({ width: 0, height: 0 });
  if (width && height) visibleSize.current = { width, height };
  const compact = !fullMapFits(
    placePipelineMap(layout, ({ id }) => ({ width: mapCardWidth, height: fullHeights.current.get(id) ?? unmeasuredCardHeight })),
    visibleSize.current.width, visibleSize.current.height,
  );
  const compactCard = (step: PlanStepView) => compact && step.id !== primaryStep;
  const placement: MapPlacement = useMemo(() => placePipelineMap(layout, ({ id }) => ({
    width: compact && id !== primaryStep ? compactMapCardWidth : mapCardWidth,
    height: heights.get(id) ?? unmeasuredCardHeight,
  })), [layout, heights, compact, primaryStep]);
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
      const measured = new Map<string, number>();
      for (const card of grid.querySelectorAll<HTMLElement>('[data-map-card]')) {
        if (!card.offsetHeight) continue;
        measured.set(card.dataset.mapStep!, card.offsetHeight);
        if (card.dataset.mapDensity === 'full') fullHeights.current.set(card.dataset.mapStep!, card.offsetHeight);
      }
      setHeights((prior) => sameHeights(prior, measured) ? prior : measured);
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
  }, [grid, layout.edges, compact, placement, getViewport]);
  const bounds = useMemo(() => geometry?.width && geometry.height
    ? { x: 0, y: 0, width: geometry.width, height: geometry.height } : null,
  [geometry?.width, geometry?.height]);
  const local = (x: number, y: number) => {
    const origin = grid!.getBoundingClientRect();
    const { zoom } = getViewport();
    return { x: (x - origin.left) / zoom, y: (y - origin.top) / zoom };
  };
  const wireEvents = wiring && grid ? {
    onPointerDown: (event: PointerEvent<HTMLDivElement>) => {
      const handle = event.target instanceof Element ? event.target.closest<HTMLElement>('[data-map-handle]') : null;
      if (!handle || event.button !== 0) return;
      event.preventDefault();
      event.stopPropagation();
      handle.setPointerCapture(event.pointerId);
      const bounds = handle.getBoundingClientRect();
      const from = local(bounds.left + bounds.width / 2, bounds.top + bounds.height / 2);
      const source = handle.dataset.mapHandle!;
      setWire({ source, from, to: from });
      wiring.start(source);
    },
    onPointerMove: (event: PointerEvent<HTMLDivElement>) => {
      if (wire) setWire({ ...wire, to: local(event.clientX, event.clientY) });
    },
    onPointerUp: (event: PointerEvent<HTMLDivElement>) => {
      if (!wire) return;
      setWire(undefined);
      wiring.end(wire.source, dropTarget(event));
    },
    onPointerCancel: () => {
      if (!wire) return;
      setWire(undefined);
      wiring.end(wire.source, null);
    },
  } : undefined;
  return (
    <div ref={ref} className="min-h-0 min-w-0 flex-1" data-map-canvas>
      <StudioCanvas contentBounds={bounds} viewportRef={viewportRef} navigateContent minZoom={0.02}
        fitReady={fitReady !== false && geometry?.compact === compact && !!width && !!height}
        elementsSelectable={false} nodesFocusable={false} edgesFocusable={false} onPaneClick={clearFocus}>
        <ViewportPortal>
          <div ref={setGrid} {...wireEvents} className="pointer-events-none relative isolate"
            style={{ width: placement.width, height: placement.height }}>
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
              {wire && <path data-map-wire={wire.source} stroke="currentColor" fill="none" strokeWidth={2}
                className="text-shell-muted" d={`M ${wire.from.x} ${wire.from.y} L ${wire.to.x} ${wire.to.y}`} />}
            </svg>
            {layout.columns.map((steps, index) => (
              <div key={index} data-canvas-column={index} className="absolute top-0 h-full"
                style={{ left: placement.columns[index].x, width: placement.columns[index].width }}>
                {steps.map((step) => (
                  <div key={step.id} className="absolute left-0" style={{ top: placement.cards.get(step.id)!.y }}>
                    {card(step, compactCard(step))}
                  </div>
                ))}
              </div>
            ))}
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
  rename?: MapStepRename;
  enterStep?: (stepId: string) => void;
  stepMenu?: StepMenu;
}

function stepTrailing(step: PlanStepView, applySelection: ApplyMapSelection) {
  const worst = step.refusal ? { status: 'cannot_run' as const, count: step.statusCounts.cannot_run ?? 0 }
    : worstStatus(step.statusCounts);
  if (!worst) return undefined;
  const shown = step.runs.length > 1 ? worst.count : undefined;
  return (
    <Button variant="bare" size="content" className="inline-flex items-center gap-1 rounded-sm tabular-nums"
      aria-description={step.refusal} {...step.refusal && appTip(studioRunReason({ reason: step.refusal }))}
      onClick={(event) => {
        event.stopPropagation();
        const runIds = step.runs.filter(({ state }) => state === worst.status).map(({ id }) => id);
        applySelection(() => ({ selection: { kind: 'rows', stepId: step.id, runIds }, focus: null }));
      }}>
      {shown !== undefined && <>{shown}{' '}</>}<StatusMark status={worst.status} label={studioRunStatus(worst.status)} />
    </Button>
  );
}

const groupId = (group: string) => '\u0000' + group;

function MapDrawer({ steps, newStep, framed = false, pending = false, selection, related, focusedStepId, chooseStep, applySelection, documentScope, rename, enterStep, stepMenu }: DrawerProps & {
  steps: readonly PlanStepView[];
  newStep?: StudioCommand;
  framed?: boolean;
  pending?: boolean;
}) {
  const region = useRegionPending(pending);
  const tree = useRef<TreeHandle>(null);
  const commands = stepMenu && ((step: PlanStepView) =>
    stepMenu(step, { start: () => tree.current?.rename(step.id), surface: 'drawer' }));
  useCommandSource(commands && (() => steps.forEach(commands)));
  const menu = commands && ((node: TreeNode) => {
    const step = steps.find(({ id }) => id === node.id);
    return step && <CommandMenuItems commands={() => commands(step)} />;
  });
  const rows = (entries: readonly PlanStepView[]): TreeNode[] => entries.map((step) => {
    const { Icon } = studioStep(step.kind);
    const count = related?.get(step.id)?.size ?? 0;
    const linked = related && focusedStepId !== step.id;
    return {
      id: step.id, name: step.label, glyph: <Icon aria-hidden="true" />,
      description: linked ? studioCountOf(count, step.runs.length) : undefined,
      muted: Boolean(linked && !count) || documentLevel(documentScope, step.id) === 'faint',
      trailing: stepTrailing(step, applySelection),
    };
  });
  const grouped = new Map<string | null, PlanStepView[]>();
  for (const step of steps) {
    const group = step.group ?? null;
    const entries = grouped.get(group);
    if (entries) entries.push(step);
    else grouped.set(group, [step]);
  }
  const nodes = [...grouped].flatMap(([group, entries]): TreeNode[] => group === null ? rows(entries)
    : [{ id: groupId(group), name: group, disabled: true, children: rows(entries) }]);
  const selected = steps.filter(({ id }) => stepSelection(selection, id)).map(({ id }) => id);
  const choose = (ids: string[]) => {
    const before = new Set(selected);
    const after = new Set(ids);
    const changed = [...ids.filter((id) => !before.has(id)), ...selected.filter((id) => !after.has(id))];
    if (changed.length === 1 && before.size + after.size > 1) return chooseStep(changed[0], true);
    if (ids.length === 1) return chooseStep(ids[0], false);
    applySelection(() => ids.length
      ? { selection: { kind: 'steps', stepIds: ids }, focus: null } : { selection: null, focus: null });
  };
  return (
    <nav aria-label={studioName('steps')} {...region} className={`flex min-h-0 flex-col ${framed ? 'h-full' : 'w-52 shrink-0 border-r border-shell-divider bg-shell-sidebar'}`}>
      {newStep && (
        <div className="shrink-0 px-1 pt-1">
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
      <ScrollFade axis="y" className="min-h-0 flex-1">
        <Tree ref={tree} aria-label={studioName('steps')} nodes={nodes} menu={menu} selected={selected} onSelectedChange={choose}
          onRename={rename && ((stepId, name) => rename.commit(stepId, name))} renameLabel={studioName('rename')}
          onActivate={enterStep}
          defaultExpanded={[...grouped.keys()].flatMap((group) => group === null ? [] : [groupId(group)])} />
      </ScrollFade>
    </nav>
  );
}

export function StudioPipelineMapDrawer({ plan, state, onSelectionChange, newStepCommand, documentScope, onChooseStep, rename, onEnterStep, stepMenu, pending }: Pick<
  StudioPipelineMapProps, 'plan' | 'state' | 'onSelectionChange' | 'newStepCommand' | 'documentScope' | 'onChooseStep' | 'rename' | 'onEnterStep' | 'stepMenu' | 'pending'
>) {
  const lineage = useMemo(() => indexPlanLineage(plan.edges), [plan.edges]);
  const applySelection: ApplyMapSelection = (change) => onSelectionChange(change(state));
  return <MapDrawer steps={plan.steps} selection={state.selection} related={state.focus ? lineage.related(state.focus) : null}
    focusedStepId={state.focus?.stepId} documentScope={documentScope} applySelection={applySelection}
    chooseStep={chooseMapStep(applySelection, onChooseStep)} rename={rename} enterStep={onEnterStep} stepMenu={stepMenu}
    newStep={newStepCommand?.() ?? unboundStudioCommand('newStep')} framed pending={pending} />;
}

function MapLine({
  plan,
  layout,
  selection,
  chooseStep,
  documentScope,
  reached,
  rename,
  enterStep,
  stepMenu,
}: {
  plan: PlanView;
  layout: MapLayout;
  selection: PipelineSelection | null;
  chooseStep: ChooseStep;
  documentScope?: MapDocumentScope;
  reached?: ReadonlySet<string>;
  rename?: MapStepRename;
  enterStep?: (stepId: string) => void;
  stepMenu?: StepMenu;
}) {
  const selected = plan.steps.filter(({ id }) => stepSelection(selection, id));
  const center = selected.length === 1 ? selected[0] : null;
  const before = center ? plan.steps.filter(({ id }) => layout.upstream.get(center.id)?.has(id)) : [];
  const after = center ? plan.steps.filter(({ id }) => layout.downstream.get(center.id)?.has(id)) : [];
  const groups = documentScope ? [plan.steps] : [before, selected.length ? selected : plan.steps, after].filter((steps) => steps.length);
  const lineSelection = (stepId: string) => {
    const choose = selectionButton((extend) => chooseStep(stepId, extend));
    return { ...choose, ...stepGestures(rename, enterStep, stepId, 'line', choose.onKeyDown) };
  };
  return (
    <ScrollFade axis="x" className="flex items-center gap-2 p-2">
      {groups.map((steps, index) => (
        <div key={index} className="flex shrink-0 items-center gap-2">
          {index > 0 && <ChevronRight aria-hidden="true" className="h-4 w-4 text-shell-muted" />}
          {steps.map((step) => renamingStep(rename, step.id, 'line') ? (
            <span key={step.id} className="studio-card studio-map-line-step px-2 py-1 text-sm" style={cardTint(studioStep(step.kind).tint)}><StepName step={step} rename={rename} /></span>
          ) : (
            <CommandContextMenu key={step.id}
              commands={stepMenu && (() => stepMenu(step, { start: () => rename?.setEditing({ stepId: step.id, surface: 'line' }), surface: 'line' }))}
              trigger={<Button
              variant="bare"
              size="content"
              className={`studio-card studio-map-line-step px-2 py-1 text-sm ${documentLevelStyle} ${!stepSelection(selection, step.id) ? documentLitStyle : ''}`}
              data-map-line-step={step.id}
              style={cardTint(studioStep(step.kind).tint)}
              data-selected={stepSelection(selection, step.id)?.kind === 'steps' || undefined}
              aria-label={studioAccessibleName('select', stepLabel(step))}
              aria-current={stepSelection(selection, step.id) ? 'step' : undefined}
              aria-pressed={documentScope ? documentScope.lit.has(step.id) : undefined}
              data-map-selection={stepSelection(selection, step.id)?.kind}
              data-document-scope={documentLevel(documentScope, step.id, reached?.has(step.id))}
              {...lineSelection(step.id)}
            >
              <StepName step={step} />
            </Button>} />
          ))}
        </div>
      ))}
    </ScrollFade>
  );
}

export function StudioPipelineMap({
  plan,
  state,
  onSelectionChange,
  onChooseStep,
  documentScope,
  reached,
  mode = 'map',
  drawerOpen = false,
  presentation,
  onPresentationChange,
  newStepCommand,
  newSweepCommand,
  sweepMenu,
  problemSteps,
  onProblems,
  activeStepId,
  viewportRef,
  fitReady,
  runMenu,
  axisMenu,
  stepMenu,
  rename,
  pick,
  choosing,
  connect,
  onEnterStep,
  pending = false,
}: StudioPipelineMapProps) {
  const region = useRegionPending(pending);
  const [wiringFrom, setWiringFrom] = useState<string | null>(null);
  const wirePick = useMemo<ReferencePick | null>(() => wiringFrom && connect ? {
    accepts: (target) => target.kind === 'step' && connect.accepts(wiringFrom, target.step),
    choose: () => {},
  } : null, [wiringFrom, connect]);
  const wiring = useMemo(() => connect && {
    start: setWiringFrom,
    end: (source: string, target: string | null) => {
      setWiringFrom(null);
      if (target && connect.accepts(source, target)) connect.connect(source, target);
    },
  }, [connect]);
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
  const root = useRef<HTMLElement>(null);
  useRenameFocus(rename, root);
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
      ref={root}
      aria-label={studioName('pipelineMap')}
      {...region}
      className="relative flex h-full min-h-0 min-w-0 overflow-hidden bg-shell-surface"
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
        <MapLine plan={plan} layout={layout} selection={state.selection} chooseStep={chooseStep} documentScope={documentScope} reached={reached} rename={rename} enterStep={onEnterStep} stepMenu={stepMenu} />
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
              rename={rename}
              enterStep={onEnterStep}
              stepMenu={stepMenu}
            />
          )}
          <ReactFlowProvider><MapDiagram layout={layout} clearFocus={clearFocus} primaryStep={primaryStep}
            strongEdges={strongEdges} viewportRef={viewportRef} fitReady={fitReady} wiring={wiring}
            card={(step, compact) => (
              <StepCard
                step={step}
                selection={stepSelection(state.selection, step.id)}
                presentation={presentation?.[step.id] ?? emptyPresentation}
                applySelection={applySelection}
                chooseStep={chooseStep}
                documentScope={documentScope}
                reached={reached?.has(step.id)}
                compact={compact}
                dimmed={!documentScope && selectedSteps.length > 0 &&
                  !ancestry.upstream.has(step.id) && !ancestry.downstream.has(step.id)}
                upstream={upstream?.get(step.id)}
                downstream={downstream?.get(step.id)}
                boxed={choosing?.step === step.id ? choosing.runs : undefined}
                onPresentationChange={onPresentationChange}
                newStep={newStepCommand?.({ stepId: step.id, selection: stepRunSelection(state.selection, step.id) }, step.label)}
                newSweep={newSweepCommand?.(step.id, step.label)}
                sweepMenu={sweepMenu?.stepId === step.id ? sweepMenu : undefined}
                onProblems={problemSteps?.has(step.id) ? onProblems : undefined}
                runMenu={runMenu}
                axisMenu={axisMenu}
                stepMenu={stepMenu}
                rename={rename}
                pick={wirePick ?? pick}
                connectable={!!connect && !pick && !compact}
                enterStep={onEnterStep}
              />
            )} /></ReactFlowProvider>
        </>
      )}
    </section>
  );
}
