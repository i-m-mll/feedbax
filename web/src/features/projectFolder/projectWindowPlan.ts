import { studioFormat, studioName } from '@/components/activity/studioVocabulary';
import type { PlanRunView, PlanStepView, PlanView } from '@/features/pipeline/planView';
import type { Project, StudioSavedTrainRuns, SubsetTerms } from '@/generated/studioContracts';

export type SavedTrainRow = StudioSavedTrainRuns['rows'][number];

export function savedRunId({ variant, step, row }: SavedTrainRow['key']) {
  return JSON.stringify([variant, step, row]);
}

export function planVariants(saved: StudioSavedTrainRuns | undefined) {
  return saved?.window_map?.variants ?? [...new Set(saved?.rows.map(({ key }) => key.variant))];
}

function subsetTerms(terms: SubsetTerms, sweeps?: ReadonlyMap<string, ReadonlyMap<string, string>>): string[] {
  const best = terms.best;
  return [
    ...Object.entries(terms.axes ?? {}).map(([axis, ids]) =>
      `${axis} ${ids.map((id) => sweeps?.get(axis)?.get(id) ?? id).join(', ')}`),
    ...(terms.rows ?? []),
    ...(best ? [
      studioFormat('bestSubset', {
        order: studioName(best.order === 'min' ? 'lowest' : 'highest'), metric: best.by, count: best.k,
      }),
      ...(best.per?.length ? [studioFormat('subsetPer', { axes: best.per.join(', ') })] : []),
    ] : []),
  ];
}

export function projectWindowPlan(project: Project, saved: StudioSavedTrainRuns | undefined, variant: string | null): PlanView {
  const map = saved?.window_map;
  const scope = variant ?? planVariants(saved)[0] ?? null;
  const evaluations = new Map(map?.evaluations?.map((row) => [savedRunId(row.key), row]));
  const memberLabels = new Map(map?.rows.map(({ key, label }) => [savedRunId(key), label]));
  const train = new Map(saved?.rows.map((row) => [savedRunId(row.key), row]));
  const sweepsByStep = new Map<string, Map<string, Map<string, string>>>();
  const runsByStep = new Map<string, PlanRunView[]>();
  function addRun(step: string, run: PlanRunView) {
    const runs = runsByStep.get(step);
    if (runs) runs.push(run);
    else runsByStep.set(step, [run]);
  }
  for (const row of map?.rows ?? []) {
    if (row.key.variant !== scope) continue;
    let sweeps = sweepsByStep.get(row.key.step);
    if (!sweeps) sweepsByStep.set(row.key.step, sweeps = new Map());
    const values: Record<string, string> = {};
    for (const [sweep, coordinate] of Object.entries(row.coordinates)) {
      let entries = sweeps.get(sweep);
      if (!entries) sweeps.set(sweep, entries = new Map());
      const number = typeof coordinate.value === 'number' ? String(coordinate.value) : undefined;
      const label = coordinate.label;
      entries.set(coordinate.value_id, number && label && label !== number
        ? `${label} · ${number}` : label ?? number ?? coordinate.value_id);
      values[sweep] = coordinate.value_id;
    }
    const id = savedRunId(row.key);
    const training = train.get(id);
    const change = training?.change;
    const reason = change && row.status === 'stale'
      ? [row.reason, `${JSON.stringify(change.before)} → ${JSON.stringify(change.after)}`].filter(Boolean).join(' ')
      : row.reason ?? undefined;
    addRun(row.key.step, {
      id, values, label: row.label, status: row.status, reason, variant: row.key.variant,
      progress: training?.attempts.at(-1)?.progress ?? undefined,
      sizeBytes: evaluations.get(id)?.result?.size_bytes,
    });
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
      documentIds: projected?.document_ids,
      documentUses: projected?.document_uses,
      inputs: projected?.inputs.map((input) => ({
        id: input.id, label: input.id, sourceStepId: input.source_step_id ?? undefined,
        subset: input.subset ? {
          ...input.subset, terms: subsetTerms(input.subset.terms, sweepsByStep.get(input.source_step_id)),
          members: input.subset.members?.map((key) => {
            const id = savedRunId(key);
            return { id, label: memberLabels.get(id) ?? key.row };
          }),
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
