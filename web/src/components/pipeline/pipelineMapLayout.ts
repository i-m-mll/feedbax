import type {
  PlanRunView,
  PlanStepView,
  PlanSweepView,
  PlanView,
} from '@/features/pipeline/planView';
import type { PipelineSelection, RunSetsByStep } from '@/features/pipeline/selection';
import type { RunAddress } from '@/features/pipeline/planView';

export interface PipelineCardPresentation {
  sweepOrder?: readonly string[];
}

interface MapEdge {
  id: string;
  from: string;
  to: string;
  inputId?: string;
}

export const mapCardWidth = 256;
export const compactMapCardWidth = 176;
export const mapColumnGap = 36;
export const mapRowGap = 12;
export const mapPadding = 16;

/** Whether the cards, placed at full size, fit a region of this size at natural scale. */
export function fullMapFits(placement: MapPlacement, width: number, height: number) {
  return !width || !height || (placement.width <= width && placement.height <= height);
}

export function mapStepLineage(
  roots: readonly string[], upstream: ReadonlyMap<string, ReadonlySet<string>>,
  downstream: ReadonlyMap<string, ReadonlySet<string>>,
) {
  const visit = (neighbors: typeof upstream) => {
    const reached = new Set(roots);
    const pending = [...roots];
    for (const step of pending) for (const next of neighbors.get(step) ?? []) {
      if (reached.has(next)) continue;
      reached.add(next);
      pending.push(next);
    }
    return reached;
  };
  return { upstream: visit(upstream), downstream: visit(downstream) };
}

export function focusedMapEdgePairs(
  edges: PlanView['edges'], focus: RunAddress,
  upstream: RunSetsByStep, downstream: RunSetsByStep,
) {
  function contains(steps: RunSetsByStep, run: RunAddress) {
    return (run.stepId === focus.stepId && run.runId === focus.runId) ||
      steps.get(run.stepId)?.has(run.runId);
  }
  const pairs = new Set<string>();
  for (const { from, to } of edges) {
    if ((contains(upstream, from) && contains(upstream, to)) ||
        (contains(downstream, from) && contains(downstream, to))) {
      pairs.add(mapEdgePair(from.stepId, to.stepId));
    }
  }
  return pairs;
}

export function mapEdgePair(from: string, to: string) {
  return JSON.stringify([from, to]);
}

export function mapAnchor(stepId: string, direction: 'in' | 'out', inputId?: string): string {
  return JSON.stringify([stepId, direction, inputId ?? null]);
}

export function layoutPipelineMap(plan: PlanView) {
  const upstream = new Map(plan.steps.map(({ id }) => [id, new Set<string>()]));
  const downstream = new Map(plan.steps.map(({ id }) => [id, new Set<string>()]));
  const edges = new Map<string, MapEdge>();

  function connect(from: string, to: string, inputId?: string) {
    if (from === to || !upstream.has(from) || !upstream.has(to)) return;
    const id = JSON.stringify([from, to, inputId ?? null]);
    edges.set(id, { id, from, to, inputId });
    upstream.get(to)!.add(from);
    downstream.get(from)!.add(to);
  }

  for (const step of plan.steps) {
    for (const input of step.inputs) {
      if (input.sourceStepId) connect(input.sourceStepId, step.id, input.subset ? input.id : undefined);
    }
  }
  for (const { from, to } of plan.edges) {
    if (!upstream.get(to.stepId)?.has(from.stepId)) connect(from.stepId, to.stepId);
  }

  const remaining = new Map(Array.from(upstream, ([id, sources]) => [id, sources.size]));
  const depth = new Map(plan.steps.map(({ id }) => [id, 0]));
  const ready = plan.steps.filter(({ id }) => remaining.get(id) === 0).map(({ id }) => id);
  for (let index = 0; index < ready.length; index += 1) {
    const id = ready[index];
    for (const target of downstream.get(id)!) {
      depth.set(target, Math.max(depth.get(target)!, depth.get(id)! + 1));
      const count = remaining.get(target)! - 1;
      remaining.set(target, count);
      if (count === 0) ready.push(target);
    }
  }
  const columns: PlanStepView[][] = [];
  for (const step of plan.steps) (columns[depth.get(step.id)!] ??= []).push(step);
  const order = new Map<string, number>();
  for (const column of columns) {
    const first = (step: PlanStepView) => Math.min(...[...upstream.get(step.id)!].map((id) => order.get(id) ?? Infinity));
    column.sort((a, b) => first(a) - first(b) || byKey(a.id, b.id));
    for (const step of column) order.set(step.id, order.size);
  }
  return { columns, edges: [...edges.values()], upstream, downstream };
}

function byKey(a: string, b: string) {
  return a < b ? -1 : a > b ? 1 : 0;
}

export type MapLayout = ReturnType<typeof layoutPipelineMap>;

export interface MapPoint {
  readonly x: number;
  readonly y: number;
}

/** Where the cards sit: each column's left edge and width, each card's top-left corner, and the extent with padding. */
export interface MapPlacement {
  readonly columns: readonly { readonly x: number; readonly width: number }[];
  readonly cards: ReadonlyMap<string, MapPoint>;
  readonly width: number;
  readonly height: number;
}

/**
 * A card's place is a function of the cards upstream of it, so that a changed connection moves the card that changed
 * and what is downstream of it, and nothing upstream. In each column, in order, a card's top is its first upstream
 * card's top, or just below the cards placed before it in its column when that place is taken.
 */
export function placePipelineMap(
  { columns, upstream }: Pick<MapLayout, 'columns' | 'upstream'>,
  size: (step: PlanStepView) => { readonly width: number; readonly height: number },
): MapPlacement {
  const cards = new Map<string, MapPoint>();
  const order = new Map(columns.flat().map(({ id }, index) => [id, index]));
  const placed: { x: number; width: number }[] = [];
  let x = mapPadding;
  let bottom = mapPadding;
  for (const column of columns) {
    const width = Math.max(0, ...column.map((step) => size(step).width));
    let next = mapPadding;
    for (const step of column) {
      const first = [...upstream.get(step.id) ?? []]
        .sort((a, b) => order.get(a)! - order.get(b)!)
        .find((id) => cards.has(id));
      const y = Math.max(first ? cards.get(first)!.y : mapPadding, next);
      cards.set(step.id, { x, y });
      next = y + size(step).height + mapRowGap;
      bottom = Math.max(bottom, next - mapRowGap);
    }
    placed.push({ x, width });
    x += width + mapColumnGap;
  }
  return {
    columns: placed,
    cards,
    width: (placed.length ? x - mapColumnGap : mapPadding) + mapPadding,
    height: bottom + mapPadding,
  };
}

/**
 * The card's sweeps in label order: the first runs down, the second across, the rest are folded. Unless arranged, the
 * longer of the first two runs across.
 */
export function orderedMapSweeps(step: PlanStepView, presentation: PipelineCardPresentation) {
  const byId = new Map(step.sweeps.map((sweep) => [sweep.id, sweep]));
  const order = new Set([...(presentation.sweepOrder ?? []), ...byId.keys()]);
  const sweeps = [...order].flatMap((id) => {
    const sweep = byId.get(id);
    return sweep ? [sweep] : [];
  });
  const [down, across] = sweeps;
  if (presentation.sweepOrder || !across || down.values.length <= across.values.length) return sweeps;
  return [across, down, ...sweeps.slice(2)];
}

/** Where a card draws its sweeps: one sweep runs across; with two or more, the first runs down, the second across, and the rest are folded into each mark. */
export interface MapAxes {
  rows?: PlanSweepView;
  columns?: PlanSweepView;
  folded: readonly PlanSweepView[];
}

export function mapAxes(sweeps: readonly PlanSweepView[]): MapAxes {
  const [first, second, ...folded] = sweeps;
  return second ? { rows: first, columns: second, folded } : { columns: first, folded: [] };
}

/** The sweep order with two sweeps exchanged. */
export function exchangeMapSweeps(sweeps: readonly PlanSweepView[], a: string, b: string) {
  const order = sweeps.map(({ id }) => id);
  const from = order.indexOf(a);
  const to = order.indexOf(b);
  if (from < 0 || to < 0 || from === to) return undefined;
  [order[from], order[to]] = [order[to], order[from]];
  return order;
}

export interface MapMarkSize {
  readonly id: 'full' | 'small' | 'smallest';
  readonly mark: number;
  readonly gap: number;
}

export const mapMarkSizes: readonly MapMarkSize[] = [
  { id: 'full', mark: 16, gap: 4 },
  { id: 'small', mark: 8, gap: 2 },
  { id: 'smallest', mark: 4, gap: 1 },
];

/** The marks' region inside a full card: its width less the card's and the region's padding; its height cap less padding. */
export const mapMarkRegion = { width: mapCardWidth - 2 * 12 - 2 * 4, height: 224 - 2 * 4 };

function extent(count: number, { mark, gap }: MapMarkSize) {
  return count * mark + Math.max(0, count - 1) * gap;
}

/** The largest size at which the marks fit the region in both directions; the smallest when none does, and the region scrolls. */
export function mapMarkSize(columns: number, rows: number): MapMarkSize {
  return mapMarkSizes.find((size) =>
    extent(columns, size) <= mapMarkRegion.width && extent(rows, size) <= mapMarkRegion.height,
  ) ?? mapMarkSizes[mapMarkSizes.length - 1];
}

type SweepTerms = Extract<PipelineSelection, { kind: 'values' }>['terms'];
/** One mark per cell: a run, or every run in the cell, which a press selects by its shown values when it has them all. */
export type RunMark =
  | { kind: 'run'; run: PlanRunView }
  | { kind: 'group'; runs: PlanRunView[]; terms: SweepTerms | null };

export interface MarkCell {
  /** Zero-based; a run with no value on the sweep is in the trailing row or column. */
  row: number;
  column: number;
  mark: RunMark;
}

export function mapRunCells(runs: readonly PlanRunView[], { rows, columns }: MapAxes) {
  const positions = (sweep?: PlanSweepView) => new Map(sweep?.values.map(({ id }, index) => [id, index]));
  const rowPositions = positions(rows);
  const columnPositions = positions(columns);
  const grouped = new Map<string, { row: number; column: number; runs: PlanRunView[] }>();
  for (const run of runs) {
    const row = rows ? rowPositions.get(run.values[rows.id]) ?? rowPositions.size : 0;
    const column = columns ? columnPositions.get(run.values[columns.id]) ?? columnPositions.size : 0;
    const key = `${row}:${column}`;
    const cell = grouped.get(key);
    if (cell) cell.runs.push(run);
    else grouped.set(key, { row, column, runs: [run] });
  }
  const shown = [rows, columns].filter((sweep): sweep is PlanSweepView => !!sweep);
  const cells: MarkCell[] = [...grouped.values()].map(({ row, column, runs }) => {
    if (runs.length === 1) return { row, column, mark: { kind: 'run', run: runs[0] } };
    const terms = shown.flatMap((sweep, index) => {
      const value = sweep.values[index === 0 && rows ? row : column]?.id;
      return value === undefined ? [] : [{ sweepId: sweep.id, valueIds: [value] }];
    });
    return { row, column, mark: { kind: 'group', runs, terms: shown.length && terms.length === shown.length ? terms : null } };
  });
  const span = (axis: 'row' | 'column', sweep?: PlanSweepView) =>
    Math.max(sweep ? sweep.values.length : 1, ...cells.map((cell) => cell[axis] + 1));
  return { cells, rowCount: cells.length ? span('row', rows) : 0, columnCount: cells.length ? span('column', columns) : 0 };
}
