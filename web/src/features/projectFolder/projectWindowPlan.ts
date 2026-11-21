import { studioFormat, studioName, studioOpeningStage, studioOpeningStep } from '@/components/activity/studioVocabulary';
import {
  runResultState, shownName, type PlanResultView, type PlanRunView, type PlanStepView, type PlanSweepView, type PlanView,
} from '@/features/pipeline/planView';
import type {
  Project, SelectedRow, StudioSavedLightResult, StudioBestRankGroup, StudioBoundBestSubject, StudioProjectWindowMapCoordinate, StudioProjectWindowMapSubset, StudioSavedTrainRuns,
  SubsetTerms,
} from '@/generated/studioContracts';

export type SavedTrainRow = StudioSavedTrainRuns['rows'][number];

export function savedRunId({ variant, step, row }: SavedTrainRow['key']) {
  return JSON.stringify([variant, step, row]);
}

/** The run's ID once its step is named `to`. */
export function renamedSavedRunId(id: string, to: string) {
  const [variant, , row] = JSON.parse(id) as [string, string, string];
  return savedRunId({ variant, step: to, row });
}

export function stepSourceRows(saved: StudioSavedTrainRuns, step: string, variant: string): SelectedRow[] {
  return saved.window_map?.rows.flatMap(({ key, member_key }) => key.step === step && key.variant === variant && member_key != null
    ? [{ key: member_key, variant: key.variant, row: key.row }] : []) ?? [];
}

export function sharedRowTerms(
  step: PlanStepView, members: readonly PlanRunView[],
): SubsetTerms | undefined {
  if (!members.length) return undefined;
  const axes = Object.fromEntries(step.sweeps.flatMap(({ id, values }) => {
    const selected = new Set<string | undefined>(members.map((member) => member.values[id]));
    return selected.has(undefined) || selected.has(unvalued) ? [] : [[id, values.filter((value) => selected.has(value.id)).map(({ id }) => id)]];
  }));
  const selected = new Set(members.map(({ id }) => id));
  const matches = step.runs.filter(({ values }) => Object.entries(axes).every(([axis, ids]) => ids.includes(values[axis])));
  return matches.length === selected.size && matches.every(({ id }) => selected.has(id)) ? { axes } : undefined;
}

export function planVariants(saved: StudioSavedTrainRuns | undefined) {
  return saved?.window_map?.variants ?? [...new Set(saved?.rows.map(({ key }) => key.variant))];
}

export function bestRankGroups(subset: StudioProjectWindowMapSubset | null | undefined): readonly StudioBestRankGroup[] {
  if (subset?.rank_groups?.length) return subset.rank_groups;
  return subset?.ranks?.length ? [{ coordinates: {}, ranks: subset.ranks }] : [];
}

/** A coordinate is drawn as its label, else its single value, else its name. */
function coordinateText({ value, label, value_id }: StudioProjectWindowMapCoordinate) {
  if (label != null) return label;
  if (typeof value === 'number' || typeof value === 'string' || typeof value === 'boolean') return String(value);
  return value_id;
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

function savedResult(result: { request_id: string; job_id: string; validity?: PlanResultView['validity'] | null } | null | undefined) {
  return result?.validity ? [{ requestId: result.request_id, jobId: result.job_id, validity: result.validity }] : [];
}

const noChecks: ReadonlySet<string> = new Set();

/**
 * The value an optional axis gives the runs it keeps beside their copies: the axis's first mark, with no name of its
 * own. It is no value of the authored axis, so no edit names it.
 */
export const unvalued = '';

function optionalAxes(project: Project, step: string) {
  return Object.entries(project.steps?.[step]?.sweep?.axes ?? {}).flatMap(([axis, read]) =>
    read && 'optional' in read && read.optional ? [axis] : []);
}

/** The row an optional axis keeps beside the row with `value` on it: its id without that value. */
function keptRow(row: string, axis: string, value: string) {
  return row.split('--').filter((part) => part !== `${axis}=${value}`).join('--') || 'default';
}

/** Whether every run has a value of both axes, and each axis's value determines the other's. */
function paired(a: string, b: string, runs: readonly PlanRunView[]) {
  const forward = new Map<string, string>();
  const backward = new Map<string, string>();
  for (const { values } of runs) {
    const [x, y] = [values[a], values[b]];
    if (x === undefined || y === undefined || x === unvalued || y === unvalued) return false;
    if ((forward.get(x) ?? y) !== y || (backward.get(y) ?? x) !== x) return false;
    forward.set(x, y);
    backward.set(y, x);
  }
  return forward.size > 1;
}

/**
 * The step's axes as the map draws them: input axes that vary together over its runs, as those of zipped list
 * inputs do, are one axis, whose values are labelled by each. It is named by the axis the step authors, when one does;
 * two authored axes stay apart.
 */
function pairedAxesJoined(sweeps: readonly PlanSweepView[], runs: readonly PlanRunView[], authored: ReadonlySet<string>) {
  const groups: PlanSweepView[][] = [];
  for (const sweep of sweeps) {
    const group = groups.find((group) => paired(group[0].id, sweep.id, runs) &&
      !(authored.has(sweep.id) && group.some(({ id }) => authored.has(id))));
    if (!group) groups.push([sweep]);
    else if (authored.has(sweep.id)) group.unshift(sweep);
    else group.push(sweep);
  }
  const absorbed = groups.flatMap((group) => group.slice(1).map(({ id }) => id));
  if (!absorbed.length) return { sweeps, runs };
  return {
    sweeps: groups.map(([first, ...rest]): PlanSweepView => !rest.length ? first : {
      id: first.id,
      label: [first, ...rest].map(({ label }) => label).join(' · '),
      values: first.values.map((value) => {
        const run = runs.find(({ values }) => values[first.id] === value.id);
        return { id: value.id, label: [value.label, ...rest.map((sweep) =>
          sweep.values.find(({ id }) => id === run?.values[sweep.id])?.label ?? '')].join(' · ') };
      }),
    }),
    runs: runs.map((run) => ({
      ...run, values: Object.fromEntries(Object.entries(run.values).filter(([axis]) => !absorbed.includes(axis))),
    })),
  };
}

export function projectWindowPlan(
  project: Project, saved: StudioSavedTrainRuns | undefined, variant: string | null, checking = noChecks,
): PlanView {
  const map = saved?.window_map;
  const elementResults = new Map<string, StudioSavedLightResult[]>();
  for (const element of map?.analysis_elements ?? []) {
    const results = element.requests.flatMap(({ result }) => result ? [result] : []);
    if (!results.length) continue;
    const id = savedRunId(element.key);
    elementResults.set(id, [...elementResults.get(id) ?? [], ...results]);
  }
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
  const stepRowCounts = new Map<string, number>();
  for (const { key } of map?.rows ?? []) {
    const step = JSON.stringify([key.variant, key.step]);
    stepRowCounts.set(step, (stepRowCounts.get(step) ?? 0) + 1);
  }
  /** A step's only run is named by its step: the row key of a step without a sweep distinguishes nothing. */
  const rowName = (id: string, fallback: string) => {
    const row = sourceRows.get(id);
    if (!row) return fallback;
    return stepRowCounts.get(JSON.stringify([row.key.variant, row.key.step])) === 1 ? row.key.step : row.label;
  };
  const stepSources = new Map(map?.steps.filter(({ variant }) => variant === scope).map(({ step_id, inputs }) =>
    [step_id, inputs.find(({ id }) => id === 'subject')?.source_step_id]));
  const train = new Map(saved?.rows.map((row) => [savedRunId(row.key), row]));
  /** The Train steps an Evaluate step's subject gathers from, when it gathers from several. */
  const gathered = new Map(map?.steps.filter(({ variant }) => variant === scope).flatMap(({ step_id, inputs }) => {
    const sources = new Set(inputs.filter(({ id }) => id === 'subject' || id.startsWith('subject.'))
      .map(({ source_step_id }) => source_step_id));
    return sources.size > 1 ? [[step_id, sources] as const] : [];
  }));
  /** The Train step that made each gathered Evaluate run's model, the run's `model` label. */
  const models = new Map((map?.edges ?? []).flatMap(({ from_key, to_key }) =>
    to_key.variant === scope && gathered.get(to_key.step)?.has(from_key.step)
      ? [[savedRunId(to_key), from_key.step] as const] : []));
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
            rank, producer: rowName(savedRunId({ ...bound, step: producer }), bound.row),
          }) : studioFormat('rankedMember', { rank }) };
        }) };
      });
    }
    return subset.members?.length ? [{ members: subset.members.map((key) => {
      const id = savedRunId(key);
      return { id, label: rowName(id, key.row) };
    }) }] : undefined;
  }
  const kept = new Map<string, Set<string>>();
  for (const row of map?.rows ?? []) {
    if (row.key.variant !== scope) continue;
    for (const axis of optionalAxes(project, row.key.step)) {
      const coordinate = row.coordinates[axis];
      if (!coordinate) continue;
      const key = JSON.stringify([row.key.step, axis]);
      kept.set(key, (kept.get(key) ?? new Set()).add(keptRow(row.key.row, axis, coordinate.value_id)));
    }
  }
  for (const row of map?.rows ?? []) {
    if (row.key.variant !== scope) continue;
    let sweeps = sweepsByStep.get(row.key.step);
    if (!sweeps) sweepsByStep.set(row.key.step, sweeps = new Map());
    const values: Record<string, string> = {};
    const model = models.get(savedRunId(row.key));
    if (model !== undefined && !('model' in row.coordinates)) {
      let entries = sweeps.get('model');
      if (!entries) sweeps.set('model', entries = new Map());
      entries.set(model, shownName(model, project.steps?.[model]));
      values.model = model;
    }
    for (const [sweep, coordinate] of Object.entries(row.coordinates)) {
      let entries = sweeps.get(sweep);
      if (!entries) sweeps.set(sweep, entries = new Map());
      entries.set(coordinate.value_id, coordinateText(coordinate));
      values[sweep] = coordinate.value_id;
    }
    for (const axis of optionalAxes(project, row.key.step)) {
      if (axis in row.coordinates || !kept.get(JSON.stringify([row.key.step, axis]))?.has(row.key.row)) continue;
      const entries = sweeps.get(axis) ?? new Map<string, string>();
      sweeps.set(axis, new Map([[unvalued, ''], ...entries]));
      values[axis] = unvalued;
    }
    const id = savedRunId(row.key);
    const training = train.get(id);
    const bound = evaluations.get(id)?.bound_subject;
    const sourceStep = stepSources.get(row.key.step);
    const producer = sourceStep && bound ? sourceRows.get(savedRunId({ ...bound, step: sourceStep })) : undefined;
    const change = training?.change ?? row.light?.changed ?? undefined;
    const acquiringFailed = row.status === 'failed' && row.opening_step === 'acquiring' && row.reason;
    const [summary, ...openingErrors] = acquiringFailed ? acquiringFailed.split('\n') : [];
    const openingReason = row.status === 'starting' && row.opening_step
      ? studioFormat('runOpeningMachine', { stage: studioOpeningStage(row.opening_step) })
      : acquiringFailed ? studioFormat('runAcquisitionFailure', { reason: summary })
      : row.status === 'failed' && row.opening_step
        ? [studioOpeningStep(row.opening_step), row.reason].filter(Boolean).join(' · ')
        : undefined;
    const reason = openingReason ? openingReason
      : row.opening_step ? [studioOpeningStep(row.opening_step), row.reason].filter(Boolean).join(' · ')
      : row.reason ?? undefined;
    const evaluation = evaluations.get(id);
    const results = [...new Map([
      ...savedResult(training?.result),
      ...savedResult(evaluation?.result && { ...evaluation.result, request_id: evaluation.request_id }),
      ...savedResult(row.light?.result),
      ...(elementResults.get(id) ?? []).flatMap(savedResult),
    ].map((result) => [result.requestId, result])).values()];
    addRun(row.key.step, {
      id, values, label: rowName(producer ? savedRunId(producer.key) : id, row.label), status: row.status, reason, variant: row.key.variant,
      problems: map?.problems.filter((problem) => problem.step === row.key.step &&
        (problem.variant === null || problem.variant === row.key.variant) && (problem.row == null || problem.row === row.key.row)),
      change: change && row.status === 'stale' ? change : undefined,
      state: runResultState(row.status, results, checking), results,
      openingReason,
      openingErrors: openingErrors.length ? openingErrors : undefined,
      progress: training?.attempts.at(-1)?.progress ?? undefined,
      collectionProgress: row.collection_progress ?? undefined,
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
      id, label: shownName(id, project.steps?.[id]), kind, group,
      documentIds: projected?.document_ids,
      documentUses: projected?.document_uses,
      inputs,
      ...pairedAxesJoined([...(sweepsByStep.get(id) ?? [])].map(([axis, values]) => {
        const read = project.steps?.[id]?.sweep?.axes?.[axis];
        return { id: axis, label: read && 'set' in read ? read.set : axis, values: [...values].map(([id, label]) => ({ id, label })) };
      }), groupedLabels.size ? runs.map((run) => {
        const rank = evaluations.get(run.id)?.subject_rank_key;
        const label = rank && groupedLabels.get(rank);
        return label ? { ...run, label } : run;
      }) : runs, new Set(Object.keys(project.steps?.[id]?.sweep?.axes ?? {}))),
      statusCounts: resultCounts(projected?.status_counts ?? {}, runs),
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

function resultCounts(counts: Readonly<Record<string, number>>, runs: readonly PlanRunView[]): PlanStepView['statusCounts'] {
  const folded: Record<string, number> = { ...counts };
  for (const { status, state } of runs) {
    if (status === state) continue;
    folded[status] = (folded[status] ?? 0) - 1;
    folded[state] = (folded[state] ?? 0) + 1;
  }
  return Object.fromEntries(Object.entries(folded).filter(([, count]) => count > 0));
}
