import type { PlanRunView, PlanStepView, PlanView } from '@/features/pipeline/planView';
import type { Project, StudioSavedTrainRuns, SubsetTerms } from '@/generated/studioContracts';

export type SavedTrainRow = StudioSavedTrainRuns['rows'][number];

export function savedRunId({ variant, step, row }: SavedTrainRow['key']) {
  return JSON.stringify([variant, step, row]);
}

export function planVariants(saved: StudioSavedTrainRuns | undefined) {
  return [...new Set([
    ...(saved?.window_map?.variants ?? []),
    ...(saved?.rows.map(({ key }) => key.variant) ?? []),
  ])];
}

function subsetTerms(terms: SubsetTerms, sweeps?: ReadonlyMap<string, ReadonlyMap<string, string>>): string[] {
  return [
    ...Object.entries(terms.axes ?? {}).map(([axis, ids]) =>
      `${axis} ${ids.map((id) => sweeps?.get(axis)?.get(id) ?? id).join(', ')}`),
    ...(terms.rows ?? []),
    ...(terms.best ? [terms.best.by] : []),
  ];
}

export function projectWindowPlan(project: Project, saved: StudioSavedTrainRuns | undefined, variant: string | null): PlanView {
  const map = saved?.window_map;
  const scope = variant ?? planVariants(saved)[0] ?? null;
  const byRun = new Map<string, PlanRunView['values']>();
  const sweepsByStep = new Map<string, Map<string, Map<string, string>>>();
  for (const row of map?.rows ?? []) {
    if (row.key.variant !== scope) continue;
    let sweeps = sweepsByStep.get(row.key.step);
    if (!sweeps) sweepsByStep.set(row.key.step, sweeps = new Map());
    const values: Record<string, string> = {};
    for (const [sweep, coordinate] of Object.entries(row.coordinates)) {
      let entries = sweeps.get(sweep);
      if (!entries) sweeps.set(sweep, entries = new Map());
      entries.set(coordinate.value_id, coordinate.label ?? coordinate.value_id);
      values[sweep] = coordinate.value_id;
    }
    byRun.set(savedRunId(row.key), values);
  }
  const runsByStep = new Map<string, PlanRunView[]>();
  for (const row of saved?.rows ?? []) {
    if (row.key.variant !== scope) continue;
    const id = savedRunId(row.key);
    let runs = runsByStep.get(row.key.step);
    if (!runs) runsByStep.set(row.key.step, runs = []);
    runs.push({ id, label: row.label, values: byRun.get(id) ?? {}, status: row.status, reason: row.reason ?? undefined,
      progress: row.attempts.at(-1)?.progress ?? undefined, variant: row.key.variant });
  }
  const source = map
    ? map.steps.filter((step) => step.variant === scope).map((step) => ({
      id: step.step_id, kind: step.kind, group: step.group, projected: step,
    }))
    : Object.entries(project.steps ?? {}).map(([id, step]) => ({
      id, kind: step.kind, group: step.group, projected: undefined,
    }));
  const steps = source.map(({ id, kind, group, projected }): PlanStepView => {
    return {
      id, label: id, kind, group,
      inputs: projected?.inputs.map((input) => ({
        id: input.id, label: input.id, sourceStepId: input.source_step_id ?? undefined,
        subset: input.subset ? {
          ...input.subset, terms: subsetTerms(input.subset.terms, sweepsByStep.get(input.source_step_id)),
        } : undefined,
      })) ?? [],
      sweeps: [...(sweepsByStep.get(id) ?? [])].map(([id, values]) => ({
        id, label: id, values: [...values].map(([id, label]) => ({ id, label })),
      })),
      statusCounts: projected?.status_counts ?? {}, runs: runsByStep.get(id) ?? [],
    };
  });
  return {
    steps,
    edges: (map?.edges ?? []).filter(({ from_key, to_key }) => from_key.variant === scope && to_key.variant === scope)
      .map(({ from_key, to_key }) => ({
        from: { stepId: from_key.step, runId: savedRunId(from_key) },
        to: { stepId: to_key.step, runId: savedRunId(to_key) },
      })),
  };
}
