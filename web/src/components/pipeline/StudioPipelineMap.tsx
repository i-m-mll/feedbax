import { ArrowLeft, ArrowRight, ChevronRight, Plus } from 'lucide-react';
import {
  memo,
  useCallback,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type DragEvent,
  type ReactNode,
} from 'react';
import { studioRunMarkClass } from '@/components/activity/StudioRunStatus';
import {
  studioAccessibleName,
  studioCountOf,
  studioName,
  studioRunStatus,
  studioStep,
  type StudioRunStatusId,
} from '@/components/activity/studioVocabulary';
import { bindStudioCommand, commandAttributes, type StudioCommand } from '@/features/projectWindow/studioCommands';
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
  layoutPipelineMap,
  mapAnchor,
  orderedMapSweeps,
  type PipelineCardPresentation,
} from './pipelineMapLayout';

type StepRunSelection = Exclude<PipelineSelection, { kind: 'steps' }>;

export interface PipelineMapStepSource {
  stepId: string;
  selection: StepRunSelection | null;
}

export interface StudioPipelineMapProps {
  plan: PlanView;
  state: PipelineSelectionState;
  onSelectionChange: (state: PipelineSelectionState) => void;
  mode?: 'map' | 'line';
  drawerOpen?: boolean;
  presentation?: Readonly<Record<string, PipelineCardPresentation>>;
  onPresentationChange?: (stepId: string, presentation: PipelineCardPresentation) => void;
  newStepCommand?: (source?: PipelineMapStepSource, subject?: string) => StudioCommand | undefined;
  newSweepCommand?: (stepId: string, subject?: string) => StudioCommand | undefined;
  onProblems?: (stepId: string) => void;
}

const emptyPresentation: PipelineCardPresentation = {};
const smallButton = 'rounded border border-slate-300 bg-white p-1 text-slate-700 hover:bg-slate-100 focus-visible:outline-2 focus-visible:outline-slate-900';

function stepSelection(selection: PipelineSelection | null, stepId: string) {
  if (!selection) return null;
  if (selection.kind === 'steps') return selection.stepIds.includes(stepId) ? selection : null;
  return selection.stepId === stepId ? selection : null;
}

function StatusCounts({ id, counts, onProblems }: { id?: string; counts: PlanStepView['statusCounts']; onProblems?: () => void }) {
  return (
    <span id={id} className="flex flex-wrap gap-x-2 gap-y-1 text-xs text-slate-700">
      {(Object.entries(counts) as [StudioRunStatusId, number][]).map(([status, count]) => {
        if (count <= 0) return null;
        const mark = <><span aria-hidden="true" className={`h-1.5 w-1.5 rounded-full ${studioRunMarkClass(status)}`} />{count} {studioRunStatus(status)}</>;
        const className = 'inline-flex items-center gap-1 whitespace-nowrap';
        return status === 'cannot_run' && onProblems
          ? <button type="button" key={status} className={`${className} underline`} onClick={onProblems}>{mark}</button>
          : <span key={status} className={className}>{mark}</span>;
      })}
    </span>
  );
}

function StepName({ step }: { step: PlanStepView }) {
  const { Icon, name } = studioStep(step.kind);
  return (
    <span className="flex min-w-0 items-center gap-1.5">
      <Icon aria-hidden="true" className="h-4 w-4 shrink-0" />
      <span className="truncate font-medium">{step.label}</span>
      <span className="shrink-0 text-xs text-slate-600">{name}</span>
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
    <div className="flex flex-wrap items-center gap-1 text-xs text-slate-700">
      {sweeps.map((sweep, index) => {
        const isFolded = sweep.id === folded?.id;
        return (
          <div key={sweep.id} className="inline-flex items-center gap-1">
            {index > 0 && <span aria-hidden="true">×</span>}
            <div className="relative">
              <button
                type="button"
                className="rounded px-1 py-0.5 hover:bg-slate-100 focus-visible:outline-2 focus-visible:outline-slate-900 aria-expanded:bg-slate-100"
                aria-label={studioAccessibleName('sweep', sweep.label)}
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
                {isFolded ? '(' : ''}{sweep.label} {sweep.values.length}{isFolded ? ')' : ''}
              </button>
              {openSweep === sweep.id && (
                <div className="absolute left-0 top-full z-20 w-max max-w-56 rounded-md border border-slate-200 bg-white p-2 shadow-md">
                  <ul className="max-h-36 space-y-0.5 overflow-y-auto">
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
                    <div className="mt-2 flex items-center gap-1 border-t border-slate-100 pt-1">
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
  onPresentationChange,
  newStep,
  newSweep,
  onProblems,
}: {
  step: PlanStepView;
  selection: PipelineSelection | null;
  presentation: PipelineCardPresentation;
  applySelection: ApplyMapSelection;
  onPresentationChange?: StudioPipelineMapProps['onPresentationChange'];
  newStep?: StudioCommand;
  newSweep?: StudioCommand;
  onProblems?: StudioPipelineMapProps['onProblems'];
}) {
  const sweeps = useMemo(() => orderedMapSweeps(step, presentation), [step, presentation]);
  const folded = foldedMapSweep(step, sweeps, presentation);
  const selected = useMemo(() => new Set(
    selectedRuns(step, selection).map(({ id }) => id),
  ), [step, selection]);
  const terms = selection?.kind === 'steps' ? [] : selectionTerms(step, selection);
  const { tint, name: kind } = studioStep(step.kind);
  return (
    <article
      data-map-card
      aria-label={studioAccessibleName(step.kind, step.label)}
      className="relative w-64 space-y-2 rounded-lg border-2 bg-white px-3 py-2 text-sm shadow-sm focus-within:z-20"
      style={{ borderColor: tint }}
    >
      <span id={`map-step-${step.id}-summary`} className="sr-only">
        {[kind, ...sweeps.map((sweep) => `${sweep.label} ${sweep.values.length}`)].join(' · ')}
      </span>
      <header className="relative">
        <span aria-hidden="true" data-map-anchor={mapAnchor(step.id, 'in')} className="pointer-events-none absolute -left-3 top-1/2" />
        <span aria-hidden="true" data-map-anchor={mapAnchor(step.id, 'out')} className="pointer-events-none absolute -right-3 top-1/2" />
        <button
          type="button"
          className="w-full rounded px-1 py-1 text-left text-slate-900 hover:bg-slate-100 aria-pressed:bg-slate-200"
          aria-label={studioAccessibleName('select', step.label)}
          aria-describedby={`map-step-${step.id}-summary map-step-${step.id}-status`}
          aria-pressed={selection?.kind === 'steps'}
          {...selectionButton((extend) => applySelection((state) => selectStep(state, step.id, extend)))}
        >
          <StepName step={step} />
        </button>
      </header>
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
      <StatusCounts id={`map-step-${step.id}-status`} counts={step.statusCounts} onProblems={onProblems && (() => onProblems(step.id))} />
      {selection && (
        <p className="truncate rounded bg-slate-100 px-1 text-xs font-medium text-slate-800">
          {[...terms, studioCountOf(selected.size, step.runs.length)].join(' · ')}
        </p>
      )}
      <PipelineRunMarks
        step={step}
        sweeps={sweeps}
        folded={folded}
        selection={selection}
        selected={selected}
        applySelection={applySelection}
      />
      {step.inputs.some((input) => input.subset) && (
        <ul className="-mx-3 divide-y divide-slate-100 border-t border-slate-200">
          {step.inputs.map(({ id, label, subset }) => subset && (
            <li key={id} className="relative px-3 py-1 text-xs text-slate-700">
              <span aria-hidden="true" data-map-anchor={mapAnchor(step.id, 'in', id)} className="pointer-events-none absolute left-0 top-1/2" />
              <strong className="block font-medium">{label}</strong>
              {[...subset.terms, studioCountOf(subset.count, subset.total)].join(' · ')}
            </li>
          ))}
        </ul>
      )}
      {newStep && (
        <button type="button" className={`${smallButton} absolute -right-3 top-1/2 -translate-y-1/2`}
          {...commandAttributes(newStep)} disabled={!newStep.enabled} onClick={newStep.invoke}>
          <Plus aria-hidden="true" className="h-3 w-3" />
        </button>
      )}
    </article>
  );
});

type MapLayout = ReturnType<typeof layoutPipelineMap>;

function MapDiagram({
  layout,
  children,
  clearFocus,
}: {
  layout: MapLayout;
  children: ReactNode;
  clearFocus: () => void;
}) {
  const grid = useRef<HTMLDivElement>(null);
  const [paths, setPaths] = useState<readonly { id: string; d: string }[]>([]);
  useLayoutEffect(() => {
    const element = grid.current;
    if (!element) return;
    const measure = () => {
      const origin = element.getBoundingClientRect();
      const positions = new Map<string, { x: number; y: number }>();
      for (const anchor of element.querySelectorAll<HTMLElement>('[data-map-anchor]')) {
        const bounds = anchor.getBoundingClientRect();
        positions.set(anchor.dataset.mapAnchor!, { x: bounds.left - origin.left, y: bounds.top - origin.top });
      }
      setPaths(layout.edges.flatMap(({ id, from, to, inputId }) => {
        const a = positions.get(mapAnchor(from, 'out'));
        const b = positions.get(mapAnchor(to, 'in', inputId));
        if (!a || !b) return [];
        const bend = Math.max(16, (b.x - a.x) / 2);
        return [{ id, d: `M ${a.x} ${a.y} C ${a.x + bend} ${a.y}, ${b.x - bend} ${b.y}, ${b.x} ${b.y}` }];
      }));
    };
    measure();
    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    for (const card of element.querySelectorAll('[data-map-card]')) observer.observe(card);
    return () => observer.disconnect();
  }, [layout.edges]);
  return (
    <div
      className="min-w-0 flex-1 overflow-auto"
      onClick={(event) => { if (event.target === event.currentTarget) clearFocus(); }}
    >
      <div
        ref={grid}
        className="relative isolate grid w-max min-w-full items-start gap-9 p-4"
        style={{ gridTemplateColumns: `repeat(${layout.columns.length}, max-content)` }}
        onClick={(event) => { if (event.target === event.currentTarget) clearFocus(); }}
      >
        <svg aria-hidden="true" className="pointer-events-none absolute inset-0 h-full w-full text-slate-300">
          {paths.map(({ id, d }) => <path key={id} d={d} stroke="currentColor" strokeWidth={1.5} fill="none" />)}
        </svg>
        {children}
      </div>
    </div>
  );
}

function MapDrawer({
  steps,
  selection,
  related,
  focusedStepId,
  applySelection,
  newStep,
  onProblems,
  framed = false,
}: {
  steps: readonly PlanStepView[];
  selection: PipelineSelection | null;
  related: RunSetsByStep | null;
  focusedStepId?: string;
  applySelection: ApplyMapSelection;
  newStep?: StudioCommand;
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
          const count = (related?.get(step.id)?.size ?? 0) + Number(focusedStepId === step.id);
          return (
            <li key={step.id}>
              <button
                type="button"
                className={`w-full space-y-1 rounded p-1.5 text-left text-sm hover:bg-slate-100 aria-[current=step]:bg-slate-200 ${related && !count ? 'opacity-60' : ''}`}
                aria-label={studioAccessibleName('select', step.label)}
                aria-current={scope ? 'step' : undefined}
                {...selectionButton((extend) => applySelection((state) => selectStep(state, step.id, extend)))}
              >
                <StepName step={step} />
                {related && <span className="block text-xs tabular-nums">{studioCountOf(count, step.runs.length)}</span>}
              </button>
              <div className="px-1.5"><StatusCounts counts={step.statusCounts} onProblems={onProblems && (() => onProblems(step.id))} /></div>
            </li>
          );
        })}</ul></li>))}
        {newStep && (
          <li>
            <button type="button" className={`${smallButton} inline-flex items-center gap-1`}
              {...commandAttributes(newStep)} disabled={!newStep.enabled} onClick={newStep.invoke}>
              <Plus aria-hidden="true" className="h-3 w-3" />
              {studioName('newStep')}
            </button>
          </li>
        )}
      </ul>
    </nav>
  );
}

export function StudioPipelineMapDrawer({ plan, state, onSelectionChange, newStepCommand, onProblems }: Pick<
  StudioPipelineMapProps, 'plan' | 'state' | 'onSelectionChange' | 'newStepCommand' | 'onProblems'
>) {
  const lineage = useMemo(() => indexPlanLineage(plan.edges), [plan.edges]);
  return <MapDrawer steps={plan.steps} selection={state.selection} related={state.focus ? lineage.related(state.focus) : null}
    focusedStepId={state.focus?.stepId} applySelection={(change) => onSelectionChange(change(state))}
    newStep={newStepCommand?.()} onProblems={onProblems} framed />;
}

function MapLine({
  plan,
  layout,
  selection,
  applySelection,
}: {
  plan: PlanView;
  layout: MapLayout;
  selection: PipelineSelection | null;
  applySelection: ApplyMapSelection;
}) {
  const selected = plan.steps.filter(({ id }) => stepSelection(selection, id));
  const center = selected.length === 1 ? selected[0] : null;
  const before = center ? plan.steps.filter(({ id }) => layout.upstream.get(center.id)?.has(id)) : [];
  const after = center ? plan.steps.filter(({ id }) => layout.downstream.get(center.id)?.has(id)) : [];
  const groups = [before, selected.length ? selected : plan.steps, after].filter((steps) => steps.length);
  return (
    <div className="flex items-center gap-2 overflow-auto p-2">
      {groups.map((steps, index) => (
        <div key={index} className="flex shrink-0 items-center gap-2">
          {index > 0 && <ChevronRight aria-hidden="true" className="h-4 w-4 text-slate-500" />}
          {steps.map((step) => (
            <button
              key={step.id}
              type="button"
              className={`${smallButton} border-2 text-sm aria-[current=step]:bg-slate-200`}
              style={{ borderColor: studioStep(step.kind).tint }}
              aria-label={studioAccessibleName('select', step.label)}
              aria-current={stepSelection(selection, step.id) ? 'step' : undefined}
              {...selectionButton((extend) => applySelection((state) => selectStep(state, step.id, extend)))}
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
  mode = 'map',
  drawerOpen = false,
  presentation,
  onPresentationChange,
  newStepCommand,
  newSweepCommand,
  onProblems,
}: StudioPipelineMapProps) {
  const current = useRef(state);
  useLayoutEffect(() => { current.current = state; }, [state]);
  const applySelection = useCallback<ApplyMapSelection>((change) => {
    onSelectionChange(change(current.current));
  }, [onSelectionChange]);
  const clearFocus = useCallback(() => {
    if (current.current.focus) onSelectionChange({ ...current.current, focus: null });
  }, [onSelectionChange]);
  const layout = useMemo(() => layoutPipelineMap(plan), [plan]);
  const firstStep = newStepCommand?.();
  const drawDrawer = drawerOpen && mode === 'map';
  const lineage = useMemo(() => drawDrawer ? indexPlanLineage(plan.edges) : null, [drawDrawer, plan.edges]);
  const related = useMemo(
    () => state.focus ? lineage?.related(state.focus) ?? null : null,
    [lineage, state.focus],
  );
  return (
    <section
      aria-label={studioName('pipelineMap')}
      className="flex min-h-0 min-w-0 overflow-hidden bg-slate-50"
      onKeyDown={(event) => {
        if (event.key === 'Escape' && state.focus) {
          event.stopPropagation();
          clearFocus();
        }
      }}
    >
      {!plan.steps.length ? (
        firstStep && (
          <button type="button" className={`${smallButton} m-4 inline-flex items-center gap-1.5`}
            {...commandAttributes(firstStep)} disabled={!firstStep.enabled} onClick={firstStep.invoke}>
            <Plus aria-hidden="true" className="h-4 w-4" />
            {studioName('newStep')}
          </button>
        )
      ) : mode === 'line' ? (
        <MapLine plan={plan} layout={layout} selection={state.selection} applySelection={applySelection} />
      ) : (
        <>
          {drawDrawer && (
            <MapDrawer
              steps={layout.columns.flat()}
              selection={state.selection}
              related={related}
              focusedStepId={state.focus?.stepId}
              applySelection={applySelection}
              newStep={firstStep}
              onProblems={onProblems}
            />
          )}
          <MapDiagram layout={layout} clearFocus={clearFocus}>
            {layout.columns.map((steps, index) => (
              <div key={index} className="relative flex flex-col gap-3">
                {steps.map((step) => (
                  <StepCard
                    key={step.id}
                    step={step}
                    selection={stepSelection(state.selection, step.id)}
                    presentation={presentation?.[step.id] ?? emptyPresentation}
                    applySelection={applySelection}
                    onPresentationChange={onPresentationChange}
                    newStep={newStepCommand?.({ stepId: step.id,
                      selection: state.selection?.kind !== 'steps' && state.selection?.stepId === step.id ? state.selection : null,
                    }, step.label)}
                    newSweep={newSweepCommand?.(step.id, step.label)}
                    onProblems={onProblems}
                  />
                ))}
              </div>
            ))}
          </MapDiagram>
        </>
      )}
    </section>
  );
}
