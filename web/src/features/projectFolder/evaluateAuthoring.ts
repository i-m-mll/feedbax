import type { PipelineMapStepSource } from '@/components/pipeline/StudioPipelineMap';
import { studioFormat, studioName, studioReason } from '@/components/activity/studioVocabulary';
import type { PlanRunView, PlanView } from '@/features/pipeline/planView';
import {
  BatchCheckpointSchema, StepReferenceSchema, UpdateRuleSchema,
  type BatchCheckpoint, type CheckpointRule, type Step, type StepReference, type StudioSavedTrainRuns,
  type SubsetTerms, type UpdateRule,
} from '@/generated/studioContracts';
import { trainSourceSubset } from './createCheckpointTrain';
import { ProjectEditConflictError, type ProjectEditController } from './projectEditController';
import { savedRunId } from './projectWindowPlan';

const rollout = 'feedbax.rollout';
export const checkpointLabel = (update: number) => studioFormat('checkpointAtUpdate', { update });

export type EditableCheckpointRule = 'last' | BatchCheckpoint | UpdateRule;

export function editableCheckpointRule(rule: CheckpointRule | null | undefined): EditableCheckpointRule | undefined {
  if (rule === 'last') return rule;
  const point = BatchCheckpointSchema.safeParse(rule);
  if (point.success) return point.data;
  const series = UpdateRuleSchema.safeParse(rule);
  return series.success ? series.data : undefined;
}

export function formatCheckpointRule(rule: EditableCheckpointRule): string {
  if (rule === 'last') return rule;
  if (Array.isArray(rule)) return rule.map(({ at }) => at).join(', ');
  if ('at' in rule) return `at ${rule.at}`;
  return [
    `every ${rule.every}`,
    rule.from == null ? '' : `from ${rule.from}`,
    rule.through == null ? '' : `through ${rule.through}`,
  ].filter(Boolean).join(' ');
}

export function parseCheckpointRule(text: string): { rule: EditableCheckpointRule } | { reason: string } {
  const shorthand = text.trim().toLowerCase();
  if (shorthand === 'last') return { rule: 'last' };
  const point = /^at\s+(\d+)$/.exec(shorthand);
  if (point) {
    const parsed = BatchCheckpointSchema.safeParse({ at: Number(point[1]) });
    if (parsed.success) return { rule: parsed.data };
    return { reason: studioReason('wholeUpdate', point[1]) };
  }
  const cadence = /^every\s+(\d+)(?:\s+from\s+(\d+))?(?:\s+through\s+(\d+))?$/.exec(shorthand);
  if (cadence) {
    const [, every, from, through] = cadence;
    const rule = {
      every: Number(every),
      ...(from !== undefined && { from: Number(from) }),
      ...(through !== undefined && { through: Number(through) }),
    };
    const parsed = UpdateRuleSchema.safeParse(rule);
    if (!parsed.success) return { reason: studioReason('positiveUpdates', text.trim()) };
    if (rule.from !== undefined && rule.through !== undefined && rule.from > rule.through) {
      return { reason: studioReason('reversedUpdates', text.trim()) };
    }
    return { rule: parsed.data };
  }
  if (/^\d+(?:\s*,\s*\d+)*$/.test(shorthand)) {
    const updates = shorthand.split(',').map(Number);
    const parsed = UpdateRuleSchema.safeParse(updates.map((at) => ({ at })));
    if (!parsed.success || new Set(updates).size !== updates.length) {
      return { reason: studioReason('invalidUpdateList', studioName('checkpointRule')) };
    }
    return { rule: parsed.data };
  }
  return { reason: studioReason('invalidCheckpointRule', studioName('checkpointRule')) };
}

export type TrialSet =
  | { kind: 'every_condition' }
  | { kind: 'conditions'; ids: string[] }
  | { kind: 'sample'; count: number };

export type TrialDomain =
  | { kind: 'fixed'; conditions: string[] }
  | { kind: 'sampler' };

export type StoppedEvaluationSubject = {
  runId: string;
  jobId: string;
  at: number;
};

export function exactCheckpoint(step: Step | undefined): number | undefined {
  const subject = StepReferenceSchema.safeParse(step?.inputs?.subject);
  if (!subject.success) return undefined;
  const checkpoint = subject.data.checkpoint;
  return checkpoint && typeof checkpoint === 'object' && !Array.isArray(checkpoint) && 'at' in checkpoint
    ? checkpoint.at : undefined;
}

export function managedTrialDomain(projection: StudioSavedTrainRuns | undefined, members: readonly PlanRunView[]): TrialDomain | undefined {
  if (!members.length) return undefined;
  const domains = new Map(projection?.window_map?.rows
    .filter(({ evaluation_conditions }) => evaluation_conditions != null)
    .map(({ key, evaluation_conditions }) => [savedRunId(key), evaluation_conditions]));
  const cohorts = members.map(({ id }) => domains.get(id));
  const first = cohorts[0];
  if (!first || cohorts.some((cohort) => !cohort || (cohort.length === 0) !== (first.length === 0))) return undefined;
  return first.length
    ? { kind: 'fixed', conditions: first.filter((id) => cohorts.every((cohort) => cohort?.includes(id))) }
    : { kind: 'sampler' };
}

export function validTrialSet(rule: TrialSet, domain: TrialDomain | undefined): boolean {
  if (rule.kind === 'sample') {
    return domain?.kind === 'sampler' && Number.isInteger(rule.count) && rule.count >= 1 && rule.count <= 10_000;
  }
  return domain?.kind === 'fixed' && (rule.kind === 'every_condition' ||
    (rule.ids.length > 0 && rule.ids.every((id) => domain.conditions.includes(id))));
}

export function ordinaryEvaluation(step: Step): { subject: StepReference; terms: SubsetTerms; trialSet: TrialSet } | undefined {
  if (step.kind !== 'evaluate' || !('inline' in step.base) || step.deltas?.length) return undefined;
  const parsed = StepReferenceSchema.safeParse(step.inputs?.subject);
  if (!parsed.success) return undefined;
  const subject = parsed.data;
  const axes = Object.values(step.sweep?.axes ?? {}).filter((axis) => axis != null);
  if ((editableCheckpointRule(subject.checkpoint) === undefined && !axes.some((axis) => 'checkpoint_rule' in axis)) ||
      subject.row != null || subject.variants != null) return undefined;
  const terms = subject.where ?? {};
  if ('subset' in terms || terms.when) return undefined;
  const sweep = step.sweep;
  const axis = sweep?.axes?.subject;
  if (!axis || !('input' in axis) || axis.input !== 'subject' ||
      axes.some((axis) => !('input' in axis) && !('checkpoint_rule' in axis)) ||
      sweep?.rows || sweep?.combination) return undefined;
  const { evaluation_type, params } = step.base.inline;
  if (evaluation_type !== rollout || !params || typeof params !== 'object' || Array.isArray(params) ||
      !Array.isArray(params.variation) || params.variation.length || 'trials' in params) return undefined;
  const rule = params.trial_set;
  if (!rule || typeof rule !== 'object' || Array.isArray(rule)) return undefined;
  let trialSet: TrialSet;
  if (rule.kind === 'every_condition') trialSet = { kind: rule.kind };
  else if (rule.kind === 'conditions' && Array.isArray(rule.ids) && rule.ids.every((id) => typeof id === 'string')) {
    trialSet = { kind: rule.kind, ids: rule.ids };
  } else if (rule.kind === 'sample' && typeof rule.count === 'number') trialSet = { kind: rule.kind, count: rule.count };
  else return undefined;
  return { subject, terms, trialSet };
}

export async function createEvaluation(
  editor: ProjectEditController, source: PipelineMapStepSource, plan: PlanView, projection: StudioSavedTrainRuns,
  stopped?: StoppedEvaluationSubject,
) {
  const { saved } = editor.getSnapshot();
  if (saved.revision !== projection.revision || saved.project.sweep) throw new ProjectEditConflictError();
  const { step, members, where } = trainSourceSubset(source, plan, projection);
  if (stopped && (
    members.length !== 1 || members[0].id !== stopped.runId ||
    !projection.rows.find(({ key }) => savedRunId(key) === stopped.runId)?.attempts.some(
      ({ job_id, stopped_checkpoint }) => job_id === stopped.jobId && stopped_checkpoint === stopped.at,
    )
  )) throw new ProjectEditConflictError();
  const domain = managedTrialDomain(projection, members);
  if (!domain) throw new Error('Choose managed Train runs that share a trial-set rule.');
  const trialSet: TrialSet = domain.kind === 'fixed' ? { kind: 'every_condition' } : { kind: 'sample', count: 1 };
  let id = 'evaluate';
  for (let number = 2; Object.prototype.hasOwnProperty.call(saved.project.steps, id); number += 1) id = 'evaluate-' + number;
  await editor.createFile({
    kind: 'step', id,
    step: {
      kind: 'evaluate',
      base: { inline: {
        evaluation_type: rollout,
        params: { trial_set: trialSet, seed: 0, trial_policy: 'paired', variation: [] },
      } },
      inputs: { subject: { step: step.id, checkpoint: stopped ? { at: stopped.at } : 'last', ...(where && { where }) } },
      sweep: { axes: { subject: { input: 'subject' } } },
    },
  }, projection.revision);
  return id;
}
