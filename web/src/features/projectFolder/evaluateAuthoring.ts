import { joinText, referentName, type ShownText } from '@/components/ui/ShownText';
import type { PipelineMapStepSource } from '@/components/pipeline/StudioPipelineMap';
import { studioFormat, studioReason } from '@/components/activity/studioVocabulary';
import type { PlanRunView, PlanView } from '@/features/pipeline/planView';
import {
  BatchCheckpointSchema, StepReferenceSchema, UnresolvedValueSchema, UpdateRuleSchema,
  type Axis, type BatchCheckpoint, type CheckpointRule, type JsonValue, type ProjectValueEdit, type Step, type StepReference,
  type StudioSavedTrainRuns,
  type SubsetTerms, type UnresolvedValue, type UpdateRule,
} from '@/generated/studioContracts';
import { modelSourceSubset } from './createCheckpointTrain';
import { ProjectEditConflictError, type ProjectEditController } from './projectEditController';
import { savedRunId } from './projectWindowPlan';
import { stepSelections, subjectAxisEdits } from './selectionEdits';

const rollout = 'feedbax.rollout';
export const checkpointLabel = (update: number) => studioFormat('checkpointAtUpdate', { update });

/** A checkpoint rule the Evaluate inspector edits, or the text of one written that does not say a rule. */
export type EditableCheckpointRule = 'last' | BatchCheckpoint | UpdateRule | UnresolvedValue;

export function editableCheckpointRule(rule: CheckpointRule | null | undefined): EditableCheckpointRule | undefined {
  if (rule === 'last') return rule;
  const point = BatchCheckpointSchema.safeParse(rule);
  if (point.success) return point.data;
  const series = UpdateRuleSchema.safeParse(rule);
  if (series.success) return series.data;
  const held = UnresolvedValueSchema.safeParse(rule);
  return held.success ? held.data : undefined;
}

const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

/** A checkpoint rule and the selections of a subject that hold it, by index. */
export interface SelectionCheckpoint {
  readonly rule: EditableCheckpointRule;
  readonly selections: readonly number[];
}

/**
 * The checkpoint rules of a subject's selections as Studio edits them: one rule for every selection when they share
 * it, else one per selection. A selection whose rule Studio does not edit leaves none.
 */
export function selectionCheckpoints(selections: readonly StepReference[]): SelectionCheckpoint[] | undefined {
  const rules = selections.map(({ checkpoint }) => editableCheckpointRule(checkpoint));
  if (!rules.length || rules.some((rule) => rule === undefined)) return undefined;
  const [first] = rules as EditableCheckpointRule[];
  return rules.every((rule) => same(rule, first))
    ? [{ rule: first, selections: selections.map((_, index) => index) }]
    : (rules as EditableCheckpointRule[]).map((rule, index) => ({ rule, selections: [index] }));
}

/** An update of a step's run that bounds the checkpoints a rule can read. */
export interface CheckpointBound {
  readonly step: string;
  readonly update: number;
}

/**
 * The updates every run a checkpoint rule reads has: past the largest origin, and up to the fewest batches. Each end
 * names the step whose run sets it.
 */
export interface CheckpointRange {
  readonly after: CheckpointBound;
  readonly through: CheckpointBound;
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
  if ('text' in rule) return { kind: 'list', updates: rule.text };
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

/** Why a checkpoint draft is wrong, on the field that is. */
export interface CheckpointDraftProblem {
  readonly field: CheckpointDraftField;
  readonly reason: string;
}

/**
 * What a draft saves and why it is wrong. A list that says no rule saves its text; a rule outside `range` saves with
 * its reason. A draft with a required field empty, or a number no rule takes, saves nothing.
 */
export function checkpointDraftRule(draft: CheckpointDraft, range?: CheckpointRange):
  { readonly rule?: EditableCheckpointRule; readonly problem?: CheckpointDraftProblem } {
  /** The end of `range` that `update` lies beyond, if any. */
  const beyond = (update: number) => !range ? undefined
    : update <= range.after.update ? range.after : update > range.through.update ? range.through : undefined;
  const outside = (field: CheckpointDraftField, bound: CheckpointBound | undefined) => bound && range && {
    field, reason: studioFormat('checkpointRange', { step: bound.step, from: range.after.update + 1, through: range.through.update }),
  };
  switch (draft.kind) {
    case 'last': return { rule: 'last' };
    case 'at': {
      if (draft.at === null) return {};
      const parsed = BatchCheckpointSchema.safeParse({ at: draft.at });
      if (!parsed.success) return { problem: { field: 'at', reason: studioReason('wholeUpdate', String(draft.at)) } };
      return { rule: parsed.data, problem: outside('at', beyond(parsed.data.at)) };
    }
    case 'list': {
      const text = draft.updates.trim();
      if (!text) return {};
      const held = { rule: { kind: 'unresolved', text } as const };
      if (!/^\d+(?:\s*,\s*\d+)*$/.test(text)) {
        return { ...held, problem: { field: 'updates', reason: studioReason('invalidCheckpointRule', text) } };
      }
      const updates = text.split(',').map(Number);
      const parsed = UpdateRuleSchema.safeParse(updates.map((at) => ({ at })));
      if (!parsed.success || new Set(updates).size !== updates.length) {
        return { ...held, problem: { field: 'updates', reason: studioReason('invalidUpdateList', text) } };
      }
      return { rule: parsed.data, problem: outside('updates', updates.map(beyond).find((end) => end !== undefined)) };
    }
    case 'every': {
      if (draft.every === null) return {};
      for (const field of ['every', 'from', 'through'] as const) {
        const value = draft[field];
        if (value !== null && !(Number.isSafeInteger(value) && value > 0)) {
          return { problem: { field, reason: studioReason('positiveUpdates', String(value)) } };
        }
      }
      const rule = {
        every: draft.every,
        ...(draft.from !== null && { from: draft.from }),
        ...(draft.through !== null && { through: draft.through }),
      };
      if (rule.from !== undefined && rule.through !== undefined && rule.from > rule.through) {
        return { problem: { field: 'through', reason: studioReason('reversedUpdates', `${rule.from}–${rule.through}`) } };
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
 * Each condition of a cohort named by the Task variables that vary across it, as `name = value`, each variable by its
 * name in `variables` and marked missing when the Task lacks it. A condition is identified by the canonical JSON of its
 * variables, keyed by variable id; an id that is not a JSON object is its own name.
 */
export function conditionLabels(ids: readonly string[], variables: readonly { readonly id: string; readonly name: string }[]): ReadonlyMap<string, ShownText> {
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
  return new Map(ids.map((id, index): [string, ShownText] => {
    const values = parsed[index];
    if (!values) return [id, id];
    const terms = shown.filter((key) => key in values)
      .map((key) => joinText([referentName(variables.find((variable) => variable.id === key), key), text(values[key])], ' = '));
    return [id, terms.length ? joinText(terms, ', ') : id];
  }));
}

/** The trials an authored rule names by itself, which are what is offered while the subject's Task is unknown. */
export function authoredTrialDomain(rule: TrialSet): TrialDomain {
  return rule.kind === 'sample' ? { kind: 'sampler' } : { kind: 'fixed', conditions: rule.kind === 'conditions' ? rule.ids : [] };
}

export function validTrialSet(rule: TrialSet, domain: TrialDomain | undefined): boolean {
  if (rule.kind === 'sample') {
    return domain?.kind === 'sampler' && Number.isInteger(rule.count) && rule.count >= 1 && rule.count <= 10_000;
  }
  return domain?.kind === 'fixed' && (rule.kind === 'every_condition' ||
    (rule.ids.length > 0 && rule.ids.every((id) => domain.conditions.includes(id))));
}

/** A path of the Evaluate's own fields: its Model's by address, its Task's, its Seed and its sampled Trials. */
const ownField = (path: string) => path.startsWith('{') || path.startsWith('trial_documents.') ||
  path === 'params.seed' || path === 'params.trial_set.count';

function ownFieldAxis(axis: Axis) {
  const paths = 'path' in axis ? (axis.path === undefined ? [] : [axis.path]) : 'values' in axis && !Array.isArray(axis.values)
    ? Object.values(axis.values).flatMap((value) => Object.keys(value?.patches ?? {})) : [];
  return paths.length > 0 && paths.every(ownField);
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
    // A selection with no checkpoint reads a Synthesize step's model.
    if ((reference.checkpoint != null && editableCheckpointRule(reference.checkpoint) === undefined &&
        !axes.some((axis) => 'checkpoint_rule' in axis)) || reference.variants != null) return undefined;
    const where = reference.where ?? {};
    const terms = 'subset' in where ? undefined : where;
    subjects.push({ reference, terms, where: listed ? `/inputs/subject/${index}/where` : '/inputs/subject/where' });
  }
  const sweep = step.sweep;
  const axis = sweep?.axes?.subject;
  if (!axis || !('input' in axis) || axis.input !== 'subject' ||
      axes.some((axis) => !('input' in axis) && !('checkpoint_rule' in axis) && !ownFieldAxis(axis)) ||
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
  const { step, members, where } = modelSourceSubset(source, plan, projection, ['train', 'synthesize']);
  const checkpoint = step.kind === 'train' && { checkpoint: stopped ? { at: stopped.at } : 'last' as const };
  if (stopped && (
    members.length !== 1 || members[0].id !== stopped.runId ||
    !projection.rows.find(({ key }) => savedRunId(key) === stopped.runId)?.attempts.some(
      ({ job_id, stopped_checkpoint }) => job_id === stopped.jobId && stopped_checkpoint === stopped.at,
    )
  )) throw new ProjectEditConflictError();
  const domain = managedTrialDomain(projection, members);
  if (!domain) throw new Error('Choose managed runs that share a trial-set rule.');
  const trialSet: TrialSet = domain.kind === 'fixed' ? { kind: 'every_condition' } : { kind: 'sample', count: 1 };
  const { id, label } = await editor.nameMember({ kind: 'step', step: 'evaluate' }, projection.revision);
  await editor.createFile({
    kind: 'step', id,
    step: {
      kind: 'evaluate', label,
      base: { inline: {
        evaluation_type: rollout,
        params: { trial_set: trialSet, seed: 0, trial_policy: 'paired', variation: [] },
      } },
      inputs: { subject: { step: step.id, ...checkpoint, ...(where && { where }) } },
      sweep: { axes: { subject: { input: 'subject' } } },
    },
  }, projection.revision);
  return id;
}

/** How a document subject reads its document: as an initial Model, or as a Loaded Model with its weights. */
export type SubjectDocumentKind = 'initial_model' | 'loaded_model';

/** A document an Evaluate step evaluates, on the Task and Objective it declares. */
export interface DocumentSubject {
  readonly kind: 'document';
  readonly reads: SubjectDocumentKind;
  readonly document: string;
  readonly task?: string;
  readonly objective?: string;
}

/** What an Evaluate step's subject reads: an initial Model or Loaded Model document, or checkpoints of Train steps. */
export type AuthoredSubject = DocumentSubject | { kind: 'steps'; selections: StepReference[] };

/** The subject an edit points an Evaluate step at: a document, or one Train step's checkpoint or Synthesize step's model. */
export type EvaluateSubject = DocumentSubject | { kind: 'step'; step: string; reads: 'checkpoint' | 'model' };

export function evaluateSubject(step: Step): AuthoredSubject | undefined {
  const inputs = step.inputs ?? {};
  const subject = inputs.subject;
  const source = (binding: unknown, kind: string) => {
    const parsed = binding as { kind?: string; source?: { document?: unknown } } | null | undefined;
    return parsed?.kind === kind && typeof parsed.source?.document === 'string' ? parsed.source.document : undefined;
  };
  for (const reads of ['initial_model', 'loaded_model'] as const) {
    const document = source(subject, reads);
    if (document !== undefined) {
      return { kind: 'document', reads, document, task: source(inputs.task, 'task'), objective: source(inputs.objective, 'objective') };
    }
  }
  const selections = stepSelections(subject);
  return selections ? { kind: 'steps', selections } : undefined;
}

const documentReference = (kind: SubjectDocumentKind | 'task' | 'objective', document: string) =>
  ({ kind, source: { document } }) as const;

/**
 * The edits that point an Evaluate step's subject at `next`. A step subject reads its Task and Objective from the
 * step that made its model and fans out over the subject's members; a document subject declares its Task and Objective and has
 * no subject or checkpoint axis.
 */
export function evaluateSubjectEdits(step: Step, next: EvaluateSubject): ProjectValueEdit[] {
  const current = evaluateSubject(step);
  const inputs = step.inputs ?? {};
  const set = (path: string, present: boolean, value: JsonValue): ProjectValueEdit =>
    ({ op: present ? 'replace' : 'add', path, value });
  if (next.kind === 'step') {
    const read = (checkpoint: CheckpointRule | null | undefined): JsonValue =>
      next.reads === 'checkpoint' ? { step: next.step, checkpoint: checkpoint ?? 'last' } : { step: next.step };
    if (current?.kind === 'steps') {
      const [first] = current.selections;
      if (current.selections.length === 1 && first.step === next.step) return [];
      return [set('/inputs/subject', true, read(current.selections.find(({ checkpoint }) => checkpoint != null)?.checkpoint))];
    }
    return [
      set('/inputs/subject', 'subject' in inputs, read(undefined)),
      ...Object.keys(inputs).filter((role) => role === 'task' || role === 'objective').reverse()
        .map((role) => ({ op: 'remove', path: `/inputs/${role}` }) as const),
      ...subjectAxisEdits(step),
    ];
  }
  const edits: ProjectValueEdit[] = [];
  if (current?.kind === 'document') {
    if (current.reads !== next.reads) edits.push(set('/inputs/subject', true, documentReference(next.reads, next.document)));
    else if (current.document !== next.document) edits.push(set('/inputs/subject/source/document', true, next.document));
  } else {
    edits.push(set('/inputs/subject', 'subject' in inputs, documentReference(next.reads, next.document)));
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
