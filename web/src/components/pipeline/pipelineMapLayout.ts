import type {
  PlanRunView,
  PlanStepView,
  PlanSweepView,
  PlanView,
} from '@/features/pipeline/planView';
import type { PipelineSelection } from '@/features/pipeline/selection';

export interface PipelineCardPresentation {
  sweepOrder?: readonly string[];
  foldedSweepId?: string | null;
}

interface MapEdge {
  id: string;
  from: string;
  to: string;
  inputId?: string;
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
  for (const step of plan.steps) {
    const column = depth.get(step.id)!;
    (columns[column] ??= []).push(step);
  }
  return { columns, edges: [...edges.values()], upstream, downstream };
}

const unfoldedMarkCapacity = 1_000;

export function orderedMapSweeps(step: PlanStepView, presentation: PipelineCardPresentation) {
  const byId = new Map(step.sweeps.map((sweep) => [sweep.id, sweep]));
  const order = new Set([...(presentation.sweepOrder ?? []), ...byId.keys()]);
  return [...order].flatMap((id) => {
    const sweep = byId.get(id);
    return sweep ? [sweep] : [];
  });
}

export function foldedMapSweep(
  step: PlanStepView,
  sweeps: readonly PlanSweepView[],
  presentation: PipelineCardPresentation,
) {
  if (presentation.foldedSweepId === null) return undefined;
  const specified = sweeps.find((sweep) => sweep.id === presentation.foldedSweepId);
  if (specified || step.runs.length <= unfoldedMarkCapacity) return specified;
  const candidates = sweeps.slice(2);
  if (!candidates.length) return sweeps[1] ?? sweeps[0];
  return candidates.reduce((largest, sweep) =>
    sweep.values.length > largest.values.length ? sweep : largest,
  );
}

type SweepTerms = Extract<PipelineSelection, { kind: 'values' }>['terms'];
export type RunMark =
  | { kind: 'run'; run: PlanRunView }
  | { kind: 'group'; runs: PlanRunView[]; terms: SweepTerms };

interface MarkCell {
  row: string | undefined;
  column: string | undefined;
  marks: Map<string, RunMark>;
}

function markTerms(run: PlanRunView, visibleSweeps: readonly PlanSweepView[]): SweepTerms | null {
  const terms = visibleSweeps.flatMap((sweep) => {
    const valueId = run.values[sweep.id];
    return valueId === undefined ? [] : [{ sweepId: sweep.id, valueIds: [valueId] }];
  });
  return terms.length === visibleSweeps.length ? terms : null;
}

export function mapRunCells(
  runs: readonly PlanRunView[],
  sweeps: readonly PlanSweepView[],
  folded: PlanSweepView | undefined,
) {
  const visibleSweeps = sweeps.filter((sweep) => sweep.id !== folded?.id);
  const [rowSweep, columnSweep] = visibleSweeps;
  const rowIds = new Set(rowSweep?.values.map(({ id }) => id));
  const columnIds = new Set(columnSweep?.values.map(({ id }) => id));
  const singleFoldValues = new Set(folded && !visibleSweeps.length ? folded.values.map(({ id }) => id) : []);
  const singleFoldTerms = folded && singleFoldValues.size
    ? [{ sweepId: folded.id, valueIds: [...singleFoldValues] }]
    : null;
  const cells = new Map<string, MarkCell>();
  for (const run of runs) {
    const row = rowSweep && rowIds.has(run.values[rowSweep.id]) ? run.values[rowSweep.id] : undefined;
    const column = columnSweep && columnIds.has(run.values[columnSweep.id])
      ? run.values[columnSweep.id]
      : undefined;
    const cellKey = JSON.stringify([row ?? null, column ?? null]);
    let cell = cells.get(cellKey);
    if (!cell) {
      cell = { row, column, marks: new Map() };
      cells.set(cellKey, cell);
    }
    const terms = folded && (visibleSweeps.length
      ? markTerms(run, visibleSweeps)
      : singleFoldValues.has(run.values[folded.id]) && singleFoldTerms);
    if (!terms) {
      cell.marks.set(JSON.stringify(['run', run.id]), { kind: 'run', run });
      continue;
    }
    const markKey = terms === singleFoldTerms ? 'folded' : JSON.stringify(['group', ...terms]);
    const mark = cell.marks.get(markKey);
    if (mark?.kind === 'group') mark.runs.push(run);
    else cell.marks.set(markKey, { kind: 'group', terms, runs: [run] });
  }
  return { rowSweep, columnSweep, cells };
}
