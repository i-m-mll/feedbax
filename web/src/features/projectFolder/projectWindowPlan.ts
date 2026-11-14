import { studioFormat, studioName } from '@/components/activity/studioVocabulary';
import type { PlanRunView, PlanStepView, PlanView } from '@/features/pipeline/planView';
import type {
  Project, SelectedRow, StudioBestRankGroup, StudioBoundBestSubject, StudioProjectWindowMapSubset, StudioSavedTrainRuns,
  SubsetTerms,
} from '@/generated/studioContracts';

export type SavedTrainRow = StudioSavedTrainRuns['rows'][number];

export function savedRunId({ variant, step, row }: SavedTrainRow['key']) {
  return JSON.stringify([variant, step, row]);
}

export function stepSourceRows(saved: StudioSavedTrainRuns, step: string, variant: string): SelectedRow[] {
  return saved.window_map?.rows.flatMap(({ key, member_key }) => key.step === step && key.variant === variant && member_key != null
    ? [{ key: member_key, variant: key.variant, row: key.row }] : []) ?? [];
}

export function planVariants(saved: StudioSavedTrainRuns | undefined) {
  return saved?.window_map?.variants ?? [...new Set(saved?.rows.map(({ key }) => key.variant))];
}

export function bestRankGroups(subset: StudioProjectWindowMapSubset | null | undefined): readonly StudioBestRankGroup[] {
  if (subset?.rank_groups?.length) return subset.rank_groups;
  return subset?.ranks?.length ? [{ coordinates: {}, ranks: subset.ranks }] : [];
}

function subsetTerms(terms: SubsetTerms, sweeps?: ReadonlyMap<string, ReadonlyMap<string, string>>): string[] {
  const best = terms.best;
  return [
    ...Object.entries(terms.axes ?? {}).map(([axis, ids]) =>
      `${axis} ${ids.map((id) => sweeps?.get(axis)?.get(id) ?? id).join(', ')}`),
    ...(terms.rows ?? []).map(({ key }) => key),
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
  const subjectsByStep = new Map<string, Map<string, StudioBoundBestSubject>>();
  for (const evaluation of map?.evaluations ?? []) {
    if (evaluation.key.variant !== scope || !evaluation.bound_subject) continue;
    let subjects = subjectsByStep.get(evaluation.key.step);
    if (!subjects) subjectsByStep.set(evaluation.key.step, subjects = new Map());
    subjects.set(evaluation.bound_subject.key, evaluation.bound_subject);
  }
  const sourceRows = new Map(map?.rows.map((row) => [savedRunId(row.key), row]));
  const stepSources = new Map(map?.steps.filter(({ variant }) => variant === scope).map(({ step_id, inputs }) =>
    [step_id, inputs.find(({ id }) => id === 'subject')?.source_step_id]));
  const train = new Map(saved?.rows.map((row) => [savedRunId(row.key), row]));
  const sweepsByStep = new Map<string, Map<string, Map<string, string>>>();
  const runsByStep = new Map<string, PlanRunView[]>();
  function addRun(step: string, run: PlanRunView) {
    const runs = runsByStep.get(step);
    if (runs) runs.push(run);
    else runsByStep.set(step, [run]);
  }
  function inputMemberGroups(
    subset: StudioProjectWindowMapSubset,
    consumer: string,
    producer: string,
  ): NonNullable<PlanStepView['inputs'][number]['subset']>['memberGroups'] {
    const groups = bestRankGroups(subset);
    if (groups.length) {
      const sweeps = sweepsByStep.get(producer);
      return groups.map(({ coordinates, ranks }) => {
        const label = Object.entries(coordinates).map(([axis, value]) =>
          `${axis} ${sweeps?.get(axis)?.get(value) ?? value}`).join(' · ');
        return { label, members: ranks.map((id, index) => {
          const bound = subjectsByStep.get(consumer)?.get(id);
          const rank = index + 1;
          return { id, label: bound ? studioFormat('boundRankedMember', {
            rank, producer: sourceRows.get(savedRunId({ ...bound, step: producer }))?.label ?? bound.row,
          }) : studioFormat('rankedMember', { rank }) };
        }) };
      });
    }
    return subset.members?.length ? [{ members: subset.members.map((key) => {
      const id = savedRunId(key);
      return { id, label: sourceRows.get(id)?.label ?? key.row };
    }) }] : undefined;
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
    const bound = evaluations.get(id)?.bound_subject;
    const sourceStep = stepSources.get(row.key.step);
    const producer = sourceStep && bound && sourceRows.get(savedRunId({ ...bound, step: sourceStep }));
    const change = training?.change;
    const reason = change && row.status === 'stale'
      ? [row.reason, `${JSON.stringify(change.before)} → ${JSON.stringify(change.after)}`].filter(Boolean).join(' ')
      : row.reason ?? undefined;
    addRun(row.key.step, {
      id, values, label: producer?.label ?? row.label, status: row.status, reason, variant: row.key.variant,
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
    const inputs = projected?.inputs.map((input): PlanStepView['inputs'][number] => ({
      id: input.id, label: input.id, sourceStepId: input.source_step_id,
      subset: input.subset ? {
        ...input.subset,
        terms: subsetTerms(input.subset.terms, sweepsByStep.get(input.source_step_id)),
        memberGroups: inputMemberGroups(input.subset, id, input.source_step_id),
      } : undefined,
    })) ?? [];
    const subjectGroups = inputs.find((input) => input.id === 'subject')?.subset?.memberGroups ?? [];
    const groupedLabels = new Map<string, string>();
    for (const { label, members } of subjectGroups) {
      if (!label) continue;
      for (const member of members) {
        groupedLabels.set(member.id, studioFormat('groupedRank', { group: label, rank: member.label }));
      }
    }
    const runs = runsByStep.get(id) ?? [];
    return {
      id, label: id, kind, group,
      documentIds: projected?.document_ids,
      documentUses: projected?.document_uses,
      inputs,
      sweeps: [...(sweepsByStep.get(id) ?? [])].map(([id, values]) => ({
        id, label: id, values: [...values].map(([id, label]) => ({ id, label })),
      })),
      statusCounts: projected?.status_counts ?? {},
      runs: groupedLabels.size ? runs.map((run) => {
        const rank = evaluations.get(run.id)?.subject_rank_key;
        const label = rank && groupedLabels.get(rank);
        return label ? { ...run, label } : run;
      }) : runs,
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
