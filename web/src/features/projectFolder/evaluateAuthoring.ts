import type { PipelineMapStepSource } from '@/components/pipeline/StudioPipelineMap';
import { studioFormat, studioReason } from '@/components/activity/studioVocabulary';
import type { PlanRunView, PlanView } from '@/features/pipeline/planView';
import {
  BatchCheckpointSchema, StepReferenceSchema, UpdateRuleSchema,
  type Axis, type BatchCheckpoint, type CheckpointRule, type JsonValue, type ProjectValueEdit, type Step, type StepReference,
  type StudioSavedTrainRuns,
  type SubsetTerms, type UpdateRule,
} from '@/generated/studioContracts';
import { trainSourceSubset } from './createCheckpointTrain';
import { ProjectEditConflictError, type ProjectEditController } from './projectEditController';
import { savedRunId } from './projectWindowPlan';
import { stepSelections, subjectAxisEdits } from './selectionEdits';
import { generatedName, holds } from './generatedName';

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

/** The kinds of checkpoint rule the Evaluate inspector edits: the last update, one update, a list, or a series. */
export type CheckpointKind = 'last' | 'at' | 'list' | 'every';

/** A checkpoint rule as its editor holds it: the kind chosen and the numbers typed for it, valid or not. */
export type CheckpointDraft =
  | { kind: 'last' }
  | { kind: 'at'; at: number | null }
  | { kind: 'list'; updates: string }
  | { kind: 'every'; every: number | null; from: number | null; through: number | null };

/** The draft field a reason belongs on. */
export type CheckpointDraftField = 'at' | 'updates' | 'every' | 'from' | 'through';

export function checkpointDraft(rule: EditableCheckpointRule): CheckpointDraft {
  if (rule === 'last') return { kind: 'last' };
  if (Array.isArray(rule)) return { kind: 'list', updates: rule.map(({ at }) => at).join(', ') };
  if ('at' in rule) return { kind: 'at', at: rule.at };
  return { kind: 'every', every: rule.every, from: rule.from ?? null, through: rule.through ?? null };
}

/** An empty draft of `kind`, holding nothing until its fields are filled. */
export function emptyCheckpointDraft(kind: CheckpointKind): CheckpointDraft {
  switch (kind) {
    case 'last': return { kind };
    case 'at': return { kind, at: null };
    case 'list': return { kind, updates: '' };
    case 'every': return { kind, every: null, from: null, through: null };
  }
}

/**
 * The rule a draft says, or why it says none: a reason on the field that is wrong, or none while a required field is
 * empty.
 */
export function checkpointDraftRule(draft: CheckpointDraft):
  { rule: EditableCheckpointRule } | { field?: CheckpointDraftField; reason?: string } {
  switch (draft.kind) {
    case 'last': return { rule: 'last' };
    case 'at': {
      if (draft.at === null) return {};
      const parsed = BatchCheckpointSchema.safeParse({ at: draft.at });
      return parsed.success ? { rule: parsed.data } : { field: 'at', reason: studioReason('wholeUpdate', String(draft.at)) };
    }
    case 'list': {
      const text = draft.updates.trim();
      if (!text) return {};
      if (!/^\d+(?:\s*,\s*\d+)*$/.test(text)) return { field: 'updates', reason: studioReason('invalidCheckpointRule', text) };
      const updates = text.split(',').map(Number);
      const parsed = UpdateRuleSchema.safeParse(updates.map((at) => ({ at })));
      return parsed.success && new Set(updates).size === updates.length
        ? { rule: parsed.data } : { field: 'updates', reason: studioReason('invalidUpdateList', text) };
    }
    case 'every': {
      if (draft.every === null) return {};
      for (const field of ['every', 'from', 'through'] as const) {
        const value = draft[field];
        if (value !== null && !(Number.isSafeInteger(value) && value > 0)) {
          return { field, reason: studioReason('positiveUpdates', String(value)) };
        }
      }
      const rule = {
        every: draft.every,
        ...(draft.from !== null && { from: draft.from }),
        ...(draft.through !== null && { through: draft.through }),
      };
      if (rule.from !== undefined && rule.through !== undefined && rule.from > rule.through) {
        return { field: 'through', reason: studioReason('reversedUpdates', `${rule.from}–${rule.through}`) };
      }
      return { rule: UpdateRuleSchema.parse(rule) };
    }
  }
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

/**
 * Each condition of a cohort named by the task variables that vary across it, as `name=value`. A condition is
 * identified by the canonical JSON of its variables; an id that is not a JSON object is its own name.
 */
export function conditionLabels(ids: readonly string[]): ReadonlyMap<string, string> {
  const parsed = ids.map((id) => {
    try {
      const value: unknown = JSON.parse(id);
      return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : undefined;
    } catch {
      return undefined;
    }
  });
  const text = (value: unknown) => typeof value === 'string' ? value : JSON.stringify(value);
  const names = [...new Set(parsed.flatMap((variables) => Object.keys(variables ?? {})))];
  const varying = names.filter((name) => new Set(parsed.map((variables) => text(variables?.[name]))).size > 1);
  const shown = varying.length ? varying : names;
  return new Map(ids.map((id, index) => {
    const variables = parsed[index];
    if (!variables) return [id, id];
    const terms = shown.filter((name) => name in variables).map((name) => `${name}=${text(variables[name])}`);
    return [id, terms.join(' ') || id];
  }));
}

export function validTrialSet(rule: TrialSet, domain: TrialDomain | undefined): boolean {
  if (rule.kind === 'sample') {
    return domain?.kind === 'sampler' && Number.isInteger(rule.count) && rule.count >= 1 && rule.count <= 10_000;
  }
  return domain?.kind === 'fixed' && (rule.kind === 'every_condition' ||
    (rule.ids.length > 0 && rule.ids.every((id) => domain.conditions.includes(id))));
}

function addressedModelAxis(axis: Axis) {
  const paths = 'path' in axis ? (axis.path === undefined ? [] : [axis.path]) : 'values' in axis && !Array.isArray(axis.values)
    ? Object.values(axis.values).flatMap((value) => Object.keys(value?.patches ?? {})) : [];
  return paths.length > 0 && paths.every((path) => path.startsWith('{'));
}

/** A selection of an Evaluate subject that Studio edits: its reference, its restriction, and where that is written. */
export interface EvaluationSubject {
  readonly reference: StepReference;
  /** The selection's own terms; a named subset has none to edit. */
  readonly terms?: SubsetTerms;
  /** The pointer of the selection's `where`. */
  readonly where: string;
}

export function ordinaryEvaluation(step: Step): { subjects: EvaluationSubject[]; trialSet: TrialSet } | undefined {
  if (step.kind !== 'evaluate' || !('inline' in step.base) || step.deltas?.length) return undefined;
  const selections = stepSelections(step.inputs?.subject);
  if (!selections?.length) return undefined;
  const axes = Object.values(step.sweep?.axes ?? {}).filter((axis) => axis != null);
  const listed = Array.isArray(step.inputs?.subject);
  const subjects: EvaluationSubject[] = [];
  for (const [index, reference] of selections.entries()) {
    if ((editableCheckpointRule(reference.checkpoint) === undefined && !axes.some((axis) => 'checkpoint_rule' in axis)) ||
        reference.row != null || reference.variants != null) return undefined;
    const where = reference.where ?? {};
    if (!('subset' in where) && where.when) return undefined;
    const terms = 'subset' in where ? undefined : where;
    subjects.push({ reference, terms, where: listed ? `/inputs/subject/${index}/where` : '/inputs/subject/where' });
  }
  const sweep = step.sweep;
  const axis = sweep?.axes?.subject;
  if (!axis || !('input' in axis) || axis.input !== 'subject' ||
      axes.some((axis) => !('input' in axis) && !('checkpoint_rule' in axis) && !addressedModelAxis(axis)) ||
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
  return { subjects, trialSet };
}

export async function createEvaluation(
  editor: ProjectEditController, source: PipelineMapStepSource, plan: PlanView, projection: StudioSavedTrainRuns,
  stopped?: StoppedEvaluationSubject,
) {
  const { saved } = editor.getSnapshot();
  if (saved.revision !== projection.revision || stopped && saved.project.sweep) throw new ProjectEditConflictError();
  const { step, members, where } = trainSourceSubset(source, plan, projection, !!saved.project.sweep);
  if (stopped && (
    members.length !== 1 || members[0].id !== stopped.runId ||
    !projection.rows.find(({ key }) => savedRunId(key) === stopped.runId)?.attempts.some(
      ({ job_id, stopped_checkpoint }) => job_id === stopped.jobId && stopped_checkpoint === stopped.at,
    )
  )) throw new ProjectEditConflictError();
  const domain = managedTrialDomain(projection, members);
  if (!domain) throw new Error('Choose managed Train runs that share a trial-set rule.');
  const trialSet: TrialSet = domain.kind === 'fixed' ? { kind: 'every_condition' } : { kind: 'sample', count: 1 };
  const id = generatedName('evaluate', holds(saved.project.steps));
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

/** What an Evaluate step's subject reads: an initial Model document, or checkpoints of Train steps. */
export type AuthoredSubject =
  | { kind: 'document'; document: string; task?: string; objective?: string }
  | { kind: 'steps'; selections: StepReference[] };

/** The subject an edit points an Evaluate step at: an initial Model document, or one Train step's checkpoint. */
export type EvaluateSubject =
  | { kind: 'document'; document: string; task?: string; objective?: string }
  | { kind: 'step'; step: string };

export function evaluateSubject(step: Step): AuthoredSubject | undefined {
  const inputs = step.inputs ?? {};
  const subject = inputs.subject;
  const source = (binding: unknown, kind: string) => {
    const parsed = binding as { kind?: string; source?: { document?: unknown } } | null | undefined;
    return parsed?.kind === kind && typeof parsed.source?.document === 'string' ? parsed.source.document : undefined;
  };
  const document = source(subject, 'initial_model');
  if (document !== undefined) {
    return { kind: 'document', document, task: source(inputs.task, 'task'), objective: source(inputs.objective, 'objective') };
  }
  const selections = stepSelections(subject);
  return selections ? { kind: 'steps', selections } : undefined;
}

const documentReference = (kind: 'initial_model' | 'task' | 'objective', document: string) =>
  ({ kind, source: { document } }) as const;

/**
 * The edits that point an Evaluate step's subject at `next`. A checkpoint subject reads its Task and Objective from
 * its Train step and fans out over the subject's members; an initial Model subject declares its Task and Objective
 * and has no subject or checkpoint axis.
 */
export function evaluateSubjectEdits(step: Step, next: EvaluateSubject): ProjectValueEdit[] {
  const current = evaluateSubject(step);
  const inputs = step.inputs ?? {};
  const set = (path: string, present: boolean, value: JsonValue): ProjectValueEdit =>
    ({ op: present ? 'replace' : 'add', path, value });
  if (next.kind === 'step') {
    if (current?.kind === 'steps') {
      const [first] = current.selections;
      if (current.selections.length === 1 && first.step === next.step) return [];
      return [set('/inputs/subject', true, { step: next.step, checkpoint: first?.checkpoint ?? 'last' })];
    }
    return [
      set('/inputs/subject', 'subject' in inputs, { step: next.step, checkpoint: 'last' }),
      ...Object.keys(inputs).filter((role) => role === 'task' || role === 'objective').reverse()
        .map((role) => ({ op: 'remove', path: `/inputs/${role}` }) as const),
      ...subjectAxisEdits(step),
    ];
  }
  const edits: ProjectValueEdit[] = [];
  if (current?.kind === 'document') {
    if (current.document !== next.document) edits.push(set('/inputs/subject/source/document', true, next.document));
  } else {
    edits.push(set('/inputs/subject', 'subject' in inputs, documentReference('initial_model', next.document)));
    const axes = Object.entries(step.sweep?.axes ?? {});
    const kept = axes.filter(([, axis]) => !axis || !('input' in axis && axis.input === 'subject') && !('checkpoint_rule' in axis));
    const { axes: _axes, ...rest } = step.sweep ?? {};
    if (!kept.length && !Object.values(rest).some((value) => value != null && Object.keys(value).length)) {
      if (step.sweep) edits.push({ op: 'remove', path: '/sweep' });
    } else if (kept.length !== axes.length) {
      edits.push(set('/sweep/axes', true, Object.fromEntries(kept) as JsonValue));
    }
  }
  for (const role of ['task', 'objective'] as const) {
    const document = next[role];
    if (document !== undefined && (current?.kind !== 'document' || current[role] !== document)) {
      edits.push(set(`/inputs/${role}`, role in inputs, documentReference(role, document)));
    }
  }
  return edits;
}
