import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { useEffect, useMemo, useState } from 'react';
import { checkRunRule } from '@/api/studioAuthoring';
import { studioName } from '@/components/activity/studioVocabulary';
import { RuleField, type RuleFieldCommand, type RuleFieldProblem } from '@/components/values/RuleField';
import { pickedTerms } from '@/components/values/runRule';
import type { PlanStepView } from '@/features/pipeline/planView';
import type { ElementSelectionStore } from '@/features/projectWindow/elementSelection';
import type { SelectedRow, StudioRunRuleRequest, SubsetTerms } from '@/generated/studioContracts';
import { pinnedSelectionLabels, savedRunId } from './projectWindowPlan';
import { replacedAxisValue, type RuleProblem } from './selectionEdits';

/** Where a rule is checked: the saved revision and variant whose plan holds the producer's runs. */
export interface RuleScope {
  readonly project: string;
  readonly revision: string;
  readonly variant: string;
}

const check = (scope: RuleScope, step: string, rule: string, ranked: boolean) => ({
  queryKey: ['run-rule', scope.project, scope.revision, scope.variant, step, rule, ranked] as const,
  queryFn: ({ signal }: { signal: AbortSignal }) => checkRunRule(scope.project, {
    revision: scope.revision, variant: scope.variant, step, rule, ranked,
  } satisfies StudioRunRuleRequest, signal),
  staleTime: Infinity,
});

/**
 * The rule of a selection of `source`'s runs, or of a sweep exception (`ranked` false). The count is the plan's for
 * the saved rule and the check's while another is written; while the field has focus the runs it chooses are boxed on
 * `source`'s card. Enter saves the rule as written, whether or not it chooses runs, and the plan's problem with it is
 * shown on the saved rule. Pick Runs starts from none; each press on `source`'s card toggles runs, and the field shows the shortest rule choosing those picked.
 */
export function SelectionRule({ scope, source, terms, planned, ranked = true, label = studioName('runs'), picking,
  disabled, problem, onToRows, pick, onCommit }: {
  scope: RuleScope;
  source: PlanStepView;
  /** The saved selection: its rule, and the rows or axis values pinned beside it. */
  terms: SubsetTerms;
  /** The plan's count of the saved rule, of its candidates; an empty rule counts every run of `source`. */
  planned?: { readonly count: number; readonly total: number };
  ranked?: boolean;
  label?: string;
  picking?: ElementSelectionStore;
  disabled: boolean;
  /** The plan's problem on the saved rule, or why the last edit of it was refused. */
  problem?: RuleProblem;
  onToRows?: () => void;
  /** Where runs can be picked on `source`'s card for the selection: its rows. */
  pick?: { readonly rows: readonly SelectedRow[] };
  /** Saves the selection: the rule as written, what was picked, or a value offered in place of a missing one. */
  onCommit(terms: SubsetTerms): void;
}) {
  const rule = terms.rule ?? '';
  const pinned = pinnedSelectionLabels(terms);
  const [draft, setDraft] = useState<string | null>(null);
  const checked = useQuery({
    ...check(scope, source.id, draft?.trim() ?? '', ranked), enabled: draft !== null, placeholderData: keepPreviousData,
  });
  const result = checked.data?.result;
  const runs = useMemo(() => result?.kind === 'chosen'
    ? new Set(result.runs.map((row) => savedRunId({ variant: scope.variant, step: source.id, row }))) : undefined,
  [result, scope.variant, source.id]);
  const [picked, setPicked] = useState<ReadonlySet<string>>();
  const rowOf = useMemo(() => new Map(pick?.rows.map((row) => [savedRunId({ variant: row.variant, step: source.id, row: row.row }), row])),
    [pick?.rows, source.id]);
  const pickedWrites = useMemo(() => picked && pickedTerms(source.runs, picked, source.ruleAxes ?? {}, (id) => rowOf.get(id)),
    [picked, source.runs, source.ruleAxes, rowOf]);
  useEffect(() => {
    if (!picking) return;
    picking.setState({ choosing: picked ? {
      step: source.id, runs: picked,
      toggle: (toggled) => setPicked((current) => {
        const next = new Set(current);
        const all = toggled.every((id) => next.has(id));
        for (const id of toggled) if (all) next.delete(id); else next.add(id);
        return next;
      }),
    } : draft !== null && runs ? { step: source.id, runs } : null });
  }, [picking, picked, draft, runs, source.id]);
  useEffect(() => () => picking?.setState({ choosing: null }), [picking]);
  /** An empty rule reads every run, which the plan does not count as a selection. */
  const shown = planned ?? (!rule.trim() && !pinned?.length ? { count: source.runs.length, total: source.runs.length } : undefined);
  const writing = draft !== null && draft.trim() !== rule.trim();
  const count = picked ? `${picked.size}/${source.runs.length}`
    : writing ? result?.kind === 'chosen' && checked.data ? `${result.count}/${checked.data.total}` : undefined
      : shown && `${shown.count}/${shown.total}`;
  const shownPicks = useMemo(() => pickedWrites && ('rule' in pickedWrites ? { text: pickedWrites.rule }
    : { text: '', pinned: pickedWrites.rows.map(({ key }) => key) }), [pickedWrites]);
  function endPick(save: boolean) {
    setPicked(undefined);
    if (!save || !pick || !pickedWrites || !picked?.size) return;
    onCommit('rows' in pickedWrites ? { rows: [...pickedWrites.rows] } : pickedWrites.rule ? { rule: pickedWrites.rule } : {});
  }
  const offered = problem?.value;
  const shownProblem: RuleFieldProblem | undefined = !writing && problem ? {
    reason: problem.reason, ...problem.span && { term: { text: rule, ...problem.span } },
    ...offered && { offers: offered.remaining.map((value) => ({
      label: value.label, disabled, invoke: () => onCommit(replacedAxisValue(terms, offered, value)),
    })) },
  } : undefined;
  const commands: RuleFieldCommand[] = [
    ...pick ? [{ label: studioName('pickRuns'), disabled: disabled || !source.runs.length, invoke: () => setPicked(new Set()) }] : [],
    ...onToRows ? [{ label: studioName('toRows'), disabled: disabled || writing, invoke: onToRows }] : [],
  ];
  return <RuleField label={label} value={rule} disabled={disabled} count={count} pinned={pinned}
    vocabulary={{ axes: Object.fromEntries(Object.entries(source.ruleAxes ?? {}).map(([axis, values]) => [axis, Object.values(values)])),
      metrics: ranked ? source.ruleMetrics ?? [] : undefined }}
    problem={shownProblem} picked={shownPicks}
    commands={commands} onPickEnd={endPick} onDraft={setDraft} onCommit={(written) => {
      const { rule: _written, ...rest } = terms;
      onCommit(written ? { ...rest, rule: written } : rest);
    }} />;
}
