import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { useEffect, useMemo, useState } from 'react';
import { checkRunRule } from '@/api/studioAuthoring';
import { studioName } from '@/components/activity/studioVocabulary';
import { RuleField, type RuleFieldCommand, type RuleFieldProblem } from '@/components/values/RuleField';
import type { PlanStepView } from '@/features/pipeline/planView';
import type { ElementSelectionStore } from '@/features/projectWindow/elementSelection';
import type { StudioProjectWindowMapRow, StudioRunRuleRequest, SubsetTerms } from '@/generated/studioContracts';
import { pinnedSelectionLabels, savedRunId, selectedRowTerms } from './projectWindowPlan';
import type { RuleProblem } from './selectionEdits';

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
 * shown on the saved rule. Pick Runs starts from none; each press on `source`'s card toggles runs, and the field shows the axes that distinguish the picked runs.
 */
export function SelectionRule({ scope, source, rule, pinned, planned, ranked = true, label = studioName('runs'), picking,
  disabled, problem, onToRows, pick, onCommit }: {
  scope: RuleScope;
  source: PlanStepView;
  rule: string;
  pinned?: readonly string[];
  /** The plan's count of the saved rule, of its candidates; an empty rule counts every run of `source`. */
  planned?: { readonly count: number; readonly total: number };
  ranked?: boolean;
  label?: string;
  picking?: ElementSelectionStore;
  disabled: boolean;
  /** The plan's problem on the saved rule, or why the last edit of it was refused. */
  problem?: RuleProblem;
  onToRows?: () => void;
  /** Where runs can be picked on `source`'s card for the selection: its rows, and the save of what is picked. */
  pick?: { readonly rows: readonly StudioProjectWindowMapRow[]; onCommit(terms: SubsetTerms): void };
  onCommit(rule: string): void;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const checked = useQuery({
    ...check(scope, source.id, draft?.trim() ?? '', ranked), enabled: draft !== null, placeholderData: keepPreviousData,
  });
  const result = checked.data?.result;
  const runs = useMemo(() => result?.kind === 'chosen'
    ? new Set(result.runs.map((row) => savedRunId({ variant: scope.variant, step: source.id, row }))) : undefined,
  [result, scope.variant, source.id]);
  const [picked, setPicked] = useState<ReadonlySet<string>>();
  const pickedWrites = useMemo(() => picked && pick && selectedRowTerms(pick.rows, picked), [picked, pick?.rows]);
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
  const shownPicks = useMemo(() => picked && {
    text: '', pinned: pickedWrites && pinnedSelectionLabels(pickedWrites),
    representable: picked.size > 0 && pickedWrites !== undefined,
  }, [picked, pickedWrites]);
  function endPick(save: boolean) {
    setPicked(undefined);
    if (!save || !pick || !pickedWrites || !picked?.size) return;
    pick.onCommit(pickedWrites);
  }
  const shownProblem: RuleFieldProblem | undefined = picked?.size && !pickedWrites
    ? { reason: 'These axis values also select unpicked runs.' }
    : !writing && problem ? { reason: problem.reason, ...problem.span && { term: { text: rule, ...problem.span } } } : undefined;
  const commands: RuleFieldCommand[] = [
    ...pick ? [{ label: studioName('pickRuns'), disabled: disabled || !source.runs.length, invoke: () => setPicked(new Set()) }] : [],
    ...onToRows ? [{ label: studioName('toRows'), disabled: disabled || writing, invoke: onToRows }] : [],
  ];
  return <RuleField label={label} value={rule} disabled={disabled} count={count} pinned={pinned}
    vocabulary={{ axes: Object.fromEntries(Object.entries(source.ruleAxes ?? {}).map(([axis, values]) => [axis, Object.values(values)])),
      metrics: ranked ? source.ruleMetrics ?? [] : undefined }}
    problem={shownProblem} picked={shownPicks}
    commands={commands} onPickEnd={endPick} onDraft={setDraft} onCommit={onCommit} />;
}
