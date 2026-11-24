import { keepPreviousData, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useMemo, useState } from 'react';
import { checkRunRule } from '@/api/studioAuthoring';
import { studioName } from '@/components/activity/studioVocabulary';
import { RuleField, type RuleFieldCommand } from '@/components/values/RuleField';
import type { PlanStepView } from '@/features/pipeline/planView';
import type { ElementSelectionStore } from '@/features/projectWindow/elementSelection';
import type { StudioRunRuleRequest } from '@/generated/studioContracts';
import { savedRunId } from './projectWindowPlan';

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
 * `source`'s card. A rule the check refuses is not saved, and the refusal is shown on the field.
 */
export function SelectionRule({ scope, source, rule, pinned, planned, ranked = true, label = studioName('runs'), picking,
  disabled, problem, onToRows, onCommit }: {
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
  problem?: string;
  onToRows?: () => void;
  onCommit(rule: string): void;
}) {
  const queries = useQueryClient();
  const [draft, setDraft] = useState<string | null>(null);
  const [refused, setRefused] = useState<{ rule: string; reason: string }>();
  const checked = useQuery({
    ...check(scope, source.id, draft?.trim() ?? '', ranked), enabled: draft !== null, placeholderData: keepPreviousData,
  });
  const result = checked.data?.result;
  const runs = useMemo(() => result?.kind === 'chosen'
    ? new Set(result.runs.map((row) => savedRunId({ variant: scope.variant, step: source.id, row }))) : undefined,
  [result, scope.variant, source.id]);
  useEffect(() => {
    if (!picking) return;
    picking.setState({ choosing: draft !== null && runs ? { step: source.id, runs } : null });
  }, [picking, draft, runs, source.id]);
  useEffect(() => () => picking?.setState({ choosing: null }), [picking]);
  /** An empty rule reads every run, which the plan does not count as a selection. */
  const shown = planned ?? (!rule.trim() && !pinned?.length ? { count: source.runs.length, total: source.runs.length } : undefined);
  const writing = draft !== null && draft.trim() !== rule.trim();
  const count = writing ? result?.kind === 'chosen' && checked.data ? `${result.count}/${checked.data.total}` : undefined
    : shown && `${shown.count}/${shown.total}`;
  async function commit(next: string) {
    const answer = await queries.fetchQuery(check(scope, source.id, next, ranked));
    if (answer.result.kind === 'refused') return setRefused({ rule: next, reason: answer.result.reason });
    setRefused(undefined);
    onCommit(next);
  }
  const commands: RuleFieldCommand[] = onToRows
    ? [{ label: studioName('toRows'), disabled: disabled || writing, invoke: onToRows }] : [];
  return <RuleField label={label} value={rule} disabled={disabled} count={count} pinned={pinned}
    vocabulary={{ axes: Object.fromEntries(Object.entries(source.ruleAxes ?? {}).map(([axis, values]) => [axis, Object.values(values)])),
      metrics: ranked ? source.ruleMetrics ?? [] : undefined }}
    problem={refused && (draft === null || draft.trim() === refused.rule) ? refused.reason : problem}
    commands={commands} onDraft={setDraft} onCommit={(next) => void commit(next)} />;
}
