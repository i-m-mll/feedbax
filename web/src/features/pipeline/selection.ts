import type { PlanRunEdge, PlanRunView, PlanStepView, RunAddress } from "./planView";

export interface SweepSelectionTerm {
  readonly sweepId: string;
  readonly valueIds: readonly string[];
}

export type PipelineSelection =
  | { readonly kind: "steps"; readonly stepIds: readonly string[] }
  | { readonly kind: "values"; readonly stepId: string; readonly terms: readonly SweepSelectionTerm[] }
  | { readonly kind: "rows"; readonly stepId: string; readonly runIds: readonly string[] };

export interface PipelineSelectionState {
  readonly selection: PipelineSelection | null;
  readonly focus: RunAddress | null;
}

export const emptyPipelineSelection: PipelineSelectionState = { selection: null, focus: null };

export function extendsSelection(event: Pick<MouseEvent, 'metaKey' | 'ctrlKey' | 'shiftKey'>) {
  return event.metaKey || event.ctrlKey || event.shiftKey;
}

export function selectionButton(choose: (extend: boolean) => void) {
  return {
    onClick: (event: Pick<MouseEvent, 'metaKey' | 'ctrlKey' | 'shiftKey'>) => choose(extendsSelection(event)),
    onKeyDown: (event: Pick<KeyboardEvent, 'key' | 'metaKey' | 'ctrlKey' | 'shiftKey' | 'repeat' | 'preventDefault'>) => {
      if ((event.key === 'Enter' || event.key === ' ') && extendsSelection(event)) {
        event.preventDefault();
        if (!event.repeat) choose(true);
      }
    },
  };
}

function toggle(ids: readonly string[], id: string): string[] {
  return ids.includes(id) ? ids.filter((item) => item !== id) : [...ids, id];
}

export function selectStep(state: PipelineSelectionState, stepId: string, extend = false): PipelineSelectionState {
  const stepIds = extend && state.selection?.kind === "steps"
    ? toggle(state.selection.stepIds, stepId)
    : [stepId];
  return stepIds.length ? { selection: { kind: "steps", stepIds }, focus: null } : emptyPipelineSelection;
}

export function selectSweepGroup(stepId: string, terms: readonly SweepSelectionTerm[]): PipelineSelectionState {
  return terms.length ? { selection: { kind: "values", stepId, terms }, focus: null } : emptyPipelineSelection;
}

export function selectSweepValue(
  state: PipelineSelectionState,
  stepId: string,
  sweepId: string,
  valueId: string,
  extend = false,
): PipelineSelectionState {
  const { selection } = state;
  const terms = extend && selection?.kind === "values" && selection.stepId === stepId ? selection.terms : [];
  const existing = terms.find((term) => term.sweepId === sweepId);
  if (!existing) return selectSweepGroup(stepId, [...terms, { sweepId, valueIds: [valueId] }]);
  const valueIds = toggle(existing.valueIds, valueId);
  return selectSweepGroup(stepId, valueIds.length
    ? terms.map((term) => term === existing ? { sweepId, valueIds } : term)
    : terms.filter((term) => term !== existing));
}

export function selectRun(state: PipelineSelectionState, address: RunAddress, extend = false): PipelineSelectionState {
  const { stepId, runId } = address;
  const { selection } = state;
  const runIds = extend && selection?.kind === "rows" && selection.stepId === stepId
    ? toggle(selection.runIds, runId)
    : [runId];
  if (!runIds.length) return emptyPipelineSelection;
  const focus = runIds.includes(runId) ? address : { stepId, runId: runIds[runIds.length - 1] };
  return { selection: { kind: "rows", stepId, runIds }, focus };
}

export function selectedRuns(step: PlanStepView, selection: PipelineSelection | null): readonly PlanRunView[] {
  if (!selection) return [];
  if (selection.kind === "steps") return selection.stepIds.includes(step.id) ? step.runs : [];
  if (selection.stepId !== step.id) return [];
  if (selection.kind === "rows") {
    const ids = new Set(selection.runIds);
    return step.runs.filter((run) => ids.has(run.id));
  }
  const values = selection.terms.map(({ sweepId, valueIds }) => ({ sweepId, allowed: new Set(valueIds) }));
  return step.runs.filter((run) => values.every(({ sweepId, allowed }) => allowed.has(run.values[sweepId])));
}

export function selectionTerms(step: PlanStepView, selection: PipelineSelection | null): readonly string[] {
  if (!selection || selection.kind === "steps" || selection.stepId !== step.id) return [];
  if (selection.kind === "rows") return selectedRuns(step, selection).map((run) => run.label);
  return step.sweeps.flatMap((sweep) => {
    const term = selection.terms.find(({ sweepId }) => sweepId === sweep.id);
    if (!term) return [];
    const included = new Set(term.valueIds);
    const values = sweep.values.filter(({ id }) => included.has(id));
    return values.length ? [`${sweep.label} ${values.map(({ label }) => label).join(", ")}`] : [];
  });
}

export type RunSetsByStep = ReadonlyMap<string, ReadonlySet<string>>;
type RunNeighbors = Map<string, Map<string, RunAddress[]>>;

function appendNeighbor(index: RunNeighbors, key: RunAddress, value: RunAddress) {
  let byRun = index.get(key.stepId);
  if (!byRun) index.set(key.stepId, byRun = new Map());
  let neighbors = byRun.get(key.runId);
  if (!neighbors) byRun.set(key.runId, neighbors = []);
  neighbors.push(value);
}

function visit(index: RunNeighbors, roots: readonly RunAddress[]) {
  const seen = new Map<string, Set<string>>();
  const add = ({ stepId, runId }: RunAddress) => {
    let ids = seen.get(stepId);
    if (!ids) seen.set(stepId, ids = new Set());
    if (ids.has(runId)) return false;
    ids.add(runId);
    return true;
  };
  const pending = roots.filter(add);
  while (pending.length) {
    const { stepId, runId } = pending.pop()!;
    for (const neighbor of index.get(stepId)?.get(runId) ?? []) {
      if (add(neighbor)) pending.push(neighbor);
    }
  }
  for (const { stepId, runId } of roots) {
    const ids = seen.get(stepId);
    ids?.delete(runId);
    if (!ids?.size) seen.delete(stepId);
  }
  return seen;
}

export function indexPlanLineage(edges: readonly PlanRunEdge[]) {
  const incoming: RunNeighbors = new Map();
  const outgoing: RunNeighbors = new Map();
  for (const { from, to } of edges) {
    appendNeighbor(outgoing, from, to);
    appendNeighbor(incoming, to, from);
  }
  return {
    upstream: (roots: readonly RunAddress[]): RunSetsByStep => visit(incoming, roots),
    downstream: (roots: readonly RunAddress[]): RunSetsByStep => visit(outgoing, roots),
    related(focus: RunAddress | null): RunSetsByStep {
      if (!focus) return new Map();
      const related = visit(incoming, [focus]);
      for (const [stepId, ids] of visit(outgoing, [focus])) {
        const known = related.get(stepId);
        if (known) for (const id of ids) known.add(id);
        else related.set(stepId, ids);
      }
      return related;
    },
  };
}
