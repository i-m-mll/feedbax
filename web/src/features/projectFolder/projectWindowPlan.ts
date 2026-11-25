import { studioFormat, studioOpeningStage, studioOpeningStep } from '@/components/activity/studioVocabulary';
import {
  runResultState, shownName, type PlanResultView, type PlanRunView, type PlanStepView, type PlanSweepView, type PlanView,
} from '@/features/pipeline/planView';
import type {
  AxisSelectionClause, Project, SelectedAxisValue, StudioProjectWindowMapProblem, StudioProjectWindowMapRow, StudioSavedLightResult,
  StudioBestRankGroup, StudioBoundBestSubject, StudioProjectWindowMapSubset, StudioSavedTrainRuns,
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

export function stepSourceRows(saved: StudioSavedTrainRuns, step: string, variant: string): StudioProjectWindowMapRow[] {
  return saved.window_map?.rows.filter(({ key, member_key }) =>
    key.step === step && key.variant === variant && member_key != null) ?? [];
}

type SourceCoordinate = StudioProjectWindowMapRow['coordinates'][string] |
  NonNullable<StudioProjectWindowMapRow['input_axes']>[string];

function pinValue(coordinate: SourceCoordinate): SelectedAxisValue {
  const { value_id: id, label } = coordinate;
  const value = 'value' in coordinate ? coordinate.value : null;
  return { id, label: label ?? (typeof value === 'number' || typeof value === 'boolean' || typeof value === 'string'
    ? String(value) : id) };
}

const compareIds = (left: string, right: string) => left < right ? -1 : left > right ? 1 : 0;

/** Pins a product of axis values if it matches the pick, else gives each picked row its own clause. */
export function selectedRowTerms(rows: readonly StudioProjectWindowMapRow[], picked: ReadonlySet<string>): SubsetTerms | undefined {
  const members = rows.map(({ key, coordinates, input_axes }) =>
    ({ id: savedRunId(key), values: { ...coordinates, ...input_axes } as Record<string, SourceCoordinate> }));
  const selected = members.filter(({ id }) => picked.has(id));
  if (!selected.length || selected.length !== picked.size) return undefined;
  const others = members.filter(({ id }) => !picked.has(id));
  const matches = (values: Record<string, SourceCoordinate>, clause: AxisSelectionClause) => Object.entries(clause)
    .every(([axis, wanted]) => wanted.some(({ id }) => values[axis]?.value_id === id));
  const product: AxisSelectionClause = {};
  const names = new Set([
    ...rows.flatMap(({ coordinates }) => Object.keys(coordinates)),
    ...rows.flatMap(({ input_axes }) => Object.keys(input_axes ?? {})),
  ]);
  let outside = others;
  for (const axis of names) {
    const values = selected.map((member) => member.values[axis]);
    if (values.some((value) => value === undefined)) continue;
    const chosen = new Map(values.map((value) => [value.value_id, pinValue(value)]));
    const stillOutside = outside.filter((member) => chosen.has(member.values[axis]?.value_id));
    if (stillOutside.length === outside.length) continue;
    product[axis] = [...chosen.values()].sort((left, right) => compareIds(left.id, right.id));
    outside = stillOutside;
  }
  if (!outside.length) return Object.keys(product).length ? { clauses: storedClauses([product]) } : {};

  const clauses = selected.map(({ values }) =>
    Object.fromEntries(Object.entries(values).map(([axis, value]) => [axis, [pinValue(value)]])));
  if (clauses.some((clause) => others.some(({ values }) => matches(values, clause)))) return undefined;
  return { clauses: storedClauses(clauses) };
}

/** The runs of `rows` a choice of clauses reads: those any clause matches on every axis it names. */
export function clauseRuns(rows: readonly StudioProjectWindowMapRow[], clauses: readonly AxisSelectionClause[]): ReadonlySet<string> {
  return new Set(rows.filter(({ coordinates, input_axes }) => clauses.some((clause) => Object.entries(clause).every(([axis, wanted]) =>
    wanted.some(({ id }) => (input_axes?.[axis] ?? coordinates[axis])?.value_id === id)))).map(({ key }) => savedRunId(key)));
}

/** A clause in its stored order: axes by id, and each axis's values by id. */
function canonicalClause(clause: AxisSelectionClause): AxisSelectionClause {
  return Object.fromEntries(Object.keys(clause).sort(compareIds).map((axis) =>
    [axis, [...clause[axis]].sort((left, right) => compareIds(left.id, right.id))]));
}

const clauseKey = (clause: AxisSelectionClause) =>
  JSON.stringify(Object.entries(clause).map(([axis, values]) => [axis, values.map(({ id }) => id)]));

/** Clauses as they are stored: each in canonical order, equal ones once, and in order of their axes' and values' ids. */
export function storedClauses(clauses: readonly AxisSelectionClause[]): AxisSelectionClause[] {
  const stored = new Map(clauses.map(canonicalClause).map((clause) => [clauseKey(clause), clause]));
  return [...stored.keys()].sort(compareIds).map((key) => stored.get(key)!);
}

export function planVariants(saved: StudioSavedTrainRuns | undefined) {
  return saved?.window_map?.variants ?? [...new Set(saved?.rows.map(({ key }) => key.variant))];
}

export function bestRankGroups(subset: StudioProjectWindowMapSubset | null | undefined): readonly StudioBestRankGroup[] {
  if (subset?.rank_groups?.length) return subset.rank_groups;
  return subset?.ranks?.length ? [{ coordinates: {}, ranks: subset.ranks }] : [];
}

/** A clause as its line reads: each axis it constrains, with the values it wants. */
export function clauseText(clause: AxisSelectionClause) {
  return Object.entries(clause).map(([axis, values]) => `${axis}=${values.map(({ label }) => label).join(', ')}`).join(' ');
}

export function pinnedSelectionLabels(terms: SubsetTerms): string[] | undefined {
  const clauses = (terms.clauses ?? []).flatMap((clause) => Object.keys(clause).length ? [clauseText(clause)] : []);
  return clauses.length ? clauses : terms.bound_ranks?.map(({ key }) => key);
}

function subsetTerms(terms: SubsetTerms): string[] {
  return [...terms.rule ? [terms.rule] : [], ...(pinnedSelectionLabels(terms) ?? [])];
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

/**
 * Whether the runs' values on `known` determine their value on `axis`; a run with no value on `axis` is not counted.
 * Unless `absent` is a value, a run with no value on one of `known` is told apart by nothing.
 */
function determined(axis: string, known: readonly string[], runs: readonly PlanRunView[], absent = false) {
  const by = new Map<string, string>();
  for (const { values } of runs) {
    const value = values[axis];
    if (value === undefined) continue;
    if (!absent && known.some((id) => values[id] === undefined)) return false;
    const key = JSON.stringify(known.map((id) => values[id] ?? null));
    if ((by.get(key) ?? value) !== value) return false;
    by.set(key, value);
  }
  return true;
}

/** An axis of a step's runs: one of their coordinates, the step that made each run's input, or an input's members. */
export type StepAxis = PlanSweepView & (
  | { readonly kind: 'coordinate' }
  | { readonly kind: 'step' }
  | { readonly kind: 'input'; readonly carries: ReadonlySet<string> }
);

/**
 * The step's axes as Studio draws them, so that each dimension is drawn once. Every coordinate is drawn but one an
 * input carries with a single value, which tells none of its members apart. An input's members are told apart first
 * by the coordinates they carry, then by the step that made them, and the input's own axis is drawn only when neither
 * does, as for a Best rank; a member that carries none of a coordinate is not told apart by it. An input axis that
 * varies one-to-one with an input axis already drawn, as those of zipped list inputs do, labels that axis's values
 * beside its own.
 */
export function drawnAxes(axes: readonly StepAxis[], runs: readonly PlanRunView[]): PlanSweepView[] {
  const carried = new Set(axes.flatMap((axis) => axis.kind === 'input' ? [...axis.carries] : []));
  const drawn = new Set(axes.flatMap((axis) =>
    axis.kind === 'coordinate' && !(carried.has(axis.id) && axis.values.length === 1) ? [axis.id] : []));
  const identifying: string[] = [];
  const steps = axes.filter((axis) => axis.kind === 'step');
  const joined = new Map<string, StepAxis[]>();
  function draw(axis: StepAxis) {
    drawn.add(axis.id);
    identifying.push(axis.id);
  }
  for (const axis of axes) {
    if (axis.kind !== 'input') continue;
    const known = [...[...axis.carries].filter((id) => drawn.has(id)), ...identifying];
    if (determined(axis.id, known, runs)) {
      const partner = identifying.find((id) => determined(id, [axis.id], runs) && determined(axis.id, [id], runs));
      if (partner) joined.set(partner, [...joined.get(partner) ?? [], axis]);
      continue;
    }
    for (const step of steps) {
      if (drawn.has(step.id) || determined(step.id, known, runs)) continue;
      draw(step);
      known.push(step.id);
    }
    if (!determined(axis.id, known, runs, true)) draw(axis);
  }
  for (const step of steps) if (!drawn.has(step.id) && !determined(step.id, [...drawn], runs)) draw(step);
  return axes.flatMap((axis): PlanSweepView[] => {
    if (!drawn.has(axis.id)) return [];
    const view = { id: axis.id, label: axis.label, values: axis.values };
    const rest = joined.get(axis.id);
    if (!rest) return [view];
    return [{
      ...view,
      label: [axis, ...rest].map(({ label }) => label).join(' · '),
      values: axis.values.map((value) => {
        const run = runs.find(({ values }) => values[axis.id] === value.id);
        return { id: value.id, label: [value.label, ...rest.map((other) =>
          other.values.find(({ id }) => id === run?.values[other.id])?.label ?? '')].join(' · ') };
      }),
    }];
  });
}

/** The Problems the plan reports at a step's authored sweep axis or anything in it, such as its Set or a value's target. */
export function axisProblems(
  map: StudioSavedTrainRuns['window_map'], step: string, axis: string, variant: string | null,
): StudioProjectWindowMapProblem[] {
  const at = `sweep.axes.${axis}`;
  return map?.problems.filter((problem) => problem.step === step && (problem.variant === null || problem.variant === variant) &&
    (problem.path === at || problem.path.startsWith(at + '.'))) ?? [];
}

/** The axis naming the Train step that made each run's model, for an Evaluate step whose subject gathers several. */
const gatheredModelAxis = 'model';

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
  /** Each step's axes as the window map gathers them, each value's label by its ID. */
  const sweepsByStep = new Map(map?.steps.filter(({ variant }) => variant === scope).map(({ step_id, axes }) =>
    [step_id, new Map(axes?.map(({ id, values }) => [id, new Map(values.map((value) => [value.id, value.label]))]))]));
  /** Each gathered Evaluate step's model axis, its Train steps' names by ID. */
  const modelsByStep = new Map<string, Map<string, string>>();
  const runsByStep = new Map<string, PlanRunView[]>();
  /** Each input axis of a step, with the coordinates its members carry. */
  const carriesByStep = new Map<string, Map<string, Set<string>>>();
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
      const coordinate = row.coordinates[axis] ?? row.input_axes?.[axis];
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
    if (model !== undefined && !(gatheredModelAxis in row.coordinates)) {
      let entries = modelsByStep.get(row.key.step);
      if (!entries) modelsByStep.set(row.key.step, entries = new Map());
      entries.set(model, shownName(model, project.steps?.[model]));
      values[gatheredModelAxis] = model;
    }
    for (const [sweep, coordinate] of Object.entries(row.coordinates)) values[sweep] = coordinate.axis_value;
    for (const [axis, member] of Object.entries(row.input_axes ?? {})) {
      values[axis] = member.axis_value;
      let carried = carriesByStep.get(row.key.step);
      if (!carried) carriesByStep.set(row.key.step, carried = new Map());
      carried.set(axis, new Set([...carried.get(axis) ?? [], ...member.carries]));
    }
    for (const axis of optionalAxes(project, row.key.step)) {
      if (axis in row.coordinates || axis in (row.input_axes ?? {}) || !kept.get(JSON.stringify([row.key.step, axis]))?.has(row.key.row)) continue;
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
    const rowProblems = map?.problems.filter((problem) => problem.step === row.key.step &&
      (problem.variant === null || problem.variant === row.key.variant) && (problem.row == null || problem.row === row.key.row));
    const blocking = rowProblems?.find(({ code, blocked_by }) => code === 'blocked_input' && blocked_by)?.blocked_by;
    /** A run's own problems; the run it waits on is its link, `blockedBy`. */
    const problems = rowProblems?.filter(({ code }) => code !== 'blocked_input');
    addRun(row.key.step, {
      id, values, label: rowName(producer ? savedRunId(producer.key) : id, row.label), status: row.status, reason, variant: row.key.variant,
      problems,
      blockedBy: blocking ? { stepId: blocking.step, label: studioFormat('stepRun', {
        step: shownName(blocking.step, project.steps?.[blocking.step]),
        run: sourceRows.get(savedRunId(blocking))?.label ?? blocking.row,
      }) } : undefined,
      change: change && row.status === 'stale' ? change : undefined,
      state: runResultState(row.status, results, checking), results,
      openingReason,
      openingErrors: openingErrors.length ? openingErrors : undefined,
      progress: training?.attempts.at(-1)?.progress ?? undefined,
      collectionProgress: row.collection_progress ?? undefined,
      sizeBytes: (evaluation?.result ?? evaluation?.previous)?.size_bytes,
    });
  }
  const source = map
    ? map.steps.filter((step) => step.variant === scope).map((step) => ({
      id: step.step_id, kind: step.kind, group: step.group, projected: step,
    }))
    : Object.entries(project.steps ?? {}).map(([id, step]) => ({
      id, kind: step.kind, group: step.group, projected: undefined,
    }));
  /**
   * The authored axes of a step that no planned run has a value on and that have a Problem, such as a required Set left
   * with no entries: each is drawn with no values, where its Problem is marked and repaired.
   */
  function unplannedAxes(step: string): [string, Map<string, string>][] {
    const planned = sweepsByStep.get(step);
    return Object.keys(project.steps?.[step]?.sweep?.axes ?? {}).flatMap((axis): [string, Map<string, string>][] =>
      planned?.has(axis) || !axisProblems(map, step, axis, scope).length ? [] : [[axis, new Map()]]);
  }
  const steps = source.map(({ id, kind, group, projected }): PlanStepView => {
    const inputs = projected?.inputs.map((input): PlanStepView['inputs'][number] => ({
      id: input.id, label: input.id, sourceStepId: input.source_step_id,
      subset: input.subset ? {
        ...input.subset,
        terms: subsetTerms(input.subset.terms),
        ranked: !!input.subset.ranks?.length,
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
    const labelled = groupedLabels.size ? runs.map((run) => {
      const rank = evaluations.get(run.id)?.subject_rank_key;
      const label = rank && groupedLabels.get(rank);
      return label ? { ...run, label } : run;
    }) : runs;
    const carries = carriesByStep.get(id);
    const modelAxis = modelsByStep.get(id);
    return {
      id, label: shownName(id, project.steps?.[id]), kind, group,
      documentIds: projected?.document_ids,
      documentUses: projected?.document_uses,
      inputs,
      sweeps: drawnAxes([
        ...modelAxis ? [[gatheredModelAxis, modelAxis] as const] : [], ...sweepsByStep.get(id) ?? [],
        ...unplannedAxes(id),
      ].map(([axis, values]): StepAxis => {
        const read = project.steps?.[id]?.sweep?.axes?.[axis];
        const view = { id: axis, label: read && 'set' in read ? read.set : axis, values: [...values].map(([id, label]) => ({ id, label })) };
        const carried = carries?.get(axis);
        return carried ? { ...view, kind: 'input', carries: carried }
          : axis === gatheredModelAxis && modelAxis ? { ...view, kind: 'step' } : { ...view, kind: 'coordinate' };
      }), labelled),
      runs: labelled,
      statusCounts: resultCounts(projected?.status_counts ?? {}, runs),
      ruleAxes: projected?.axes && Object.fromEntries(projected.axes.map(({ id, values }) =>
        [id, Object.fromEntries(values.map((value) => [value.id, value.rule]))])),
      ruleMetrics: projected?.rule_metrics,
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
