import { useId, useRef, type ReactNode } from 'react';
import { useDraft } from '@/hooks/useDraft';
import { studioName, studioReason } from '@/components/activity/studioVocabulary';
import { Button } from '@/components/ui/Button';
import { Checkbox } from '@/components/ui/Checkbox';
import { NumberField } from '@/components/ui/NumberField';
import { Select } from '@/components/ui/Select';
import { TextField } from '@/components/ui/TextField';
import { FieldRow } from '@/components/ui/FieldRow';
import { Section } from '@/components/ui/Section';
import { ReferenceField, type ReferenceChoice } from '@/components/values/ReferenceField';
import type { PlanStepView } from '@/features/pipeline/planView';
import type { ElementSelectionStore } from '@/features/projectWindow/elementSelection';
import {
  BestSchema, type Best, type InputBinding, type ProjectValueEdit, type SelectedRow, type Step, type StepReference,
  type SubsetTerms,
} from '@/generated/studioContracts';
import type { AuthoredProject } from './authoredEdits';
import {
  checkpointDraft, checkpointDraftRule, conditionLabels, emptyCheckpointDraft, evaluateSubject, evaluateSubjectEdits, validTrialSet,
  type CheckpointDraft, type CheckpointDraftField, type CheckpointKind, type EditableCheckpointRule, type EvaluateSubject,
  type TrialDomain, type TrialSet,
} from './evaluateAuthoring';
import {
  documentChoice, documentChoiceId, documentGroup, stepChoice, stepChoiceId, stepGroup,
} from './referenceChoices';
import { documentsOfRole } from './useProjectDocuments';
import { savedRunId } from './projectWindowPlan';
import { InputSelections } from './InputSelections';
import { bindingEdit, changedSelections, stepSelections, subjectAxisEdits } from './selectionEdits';
import { ScrollFade } from '@/components/ui/ScrollFade';
import '@/components/values/valueField.css';

type Choice = { id: string; label: string };

/** The Evaluate fields an edit can be refused on; a subset is named by its selection's index. */
export type EvaluateField = 'subject' | 'checkpoint' | `subset.${number}` | 'trials';

/** A field group's members, one step in; no space at all when the group's control is all there is. */
const members = 'value-field-parts grid gap-1 empty:hidden';
const checkpointKinds: readonly CheckpointKind[] = ['last', 'at', 'list', 'every'];
const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
/** The field of each kind that a refusal of the step lands on. */
const firstMember: Record<CheckpointKind, CheckpointDraftField | undefined> = {
  last: undefined, at: 'at', list: 'updates', every: 'every',
};

/**
 * Which updates of the subject's training the Evaluate tests: the kind of rule as the group's control, its updates
 * beneath. A kind with updates saves once they say a rule; Escape restores the saved one.
 */
export function CheckpointEditor({ initial, disabled, readOnly, refusal, onSave, onToSweep }: {
  initial: EditableCheckpointRule;
  disabled: boolean;
  readOnly?: boolean;
  refusal?: string;
  onSave(rule: EditableCheckpointRule): void;
  onToSweep?(): void;
}) {
  const saved = checkpointDraft(initial);
  const [draft, setDraft] = useDraft(saved);
  const ids = useId();
  const first = useRef<HTMLInputElement>(null);
  const held = checkpointDraftRule(draft);
  const fixed = disabled || readOnly;
  const reason = (field: CheckpointDraftField) => !('rule' in held) && held.field === field ? held.reason
    : field === firstMember[draft.kind] ? refusal : undefined;
  function commit(next: CheckpointDraft) {
    setDraft(next);
    const result = checkpointDraftRule(next);
    if (!fixed && 'rule' in result && !same(result.rule, initial)) onSave(result.rule);
  }
  function number(field: 'at' | 'every' | 'from' | 'through', value: number | null, next: (value: number | null) => CheckpointDraft) {
    const why = reason(field);
    return <FieldRow key={field} label={studioName(field === 'at' ? 'update' : field)} htmlFor={`${ids}-${field}`}
      below={why && <p role="alert">{why}</p>}>
      <NumberField ref={field === firstMember[draft.kind] ? first : undefined} id={`${ids}-${field}`}
        min={field === 'at' ? 0 : 1} step={1} value={value} invalid={why !== undefined} disabled={disabled} readOnly={readOnly}
        onValueChange={(changed) => setDraft(next(changed))} onValueCommitted={(changed) => commit(next(changed))} />
    </FieldRow>;
  }
  const updatesReason = reason('updates');
  return <div className="contents" onKeyDown={(event) => {
    if (event.key !== 'Escape' || same(draft, saved)) return;
    event.preventDefault();
    event.stopPropagation();
    setDraft(saved);
  }}><Section level="field" name={studioName('checkpoint')} bodyClassName={members}
    control={<>
      <Select aria-label={studioName('checkpoint')} value={draft.kind} disabled={fixed}
        aria-invalid={draft.kind === 'last' && refusal !== undefined}
        items={checkpointKinds.map((kind) => ({ value: kind, label: studioName(kind) }))}
        finalFocus={() => first.current}
        onValueChange={(kind) => commit(kind === saved.kind ? saved : emptyCheckpointDraft(kind))} />
      {draft.kind === 'last' && refusal && <p role="alert">{refusal}</p>}
    </>}>
      {draft.kind === 'at' && number('at', draft.at, (at) => ({ kind: 'at', at }))}
      {draft.kind === 'list' && <FieldRow label={studioName('updates')} htmlFor={`${ids}-updates`}
        below={updatesReason && <p role="alert">{updatesReason}</p>}>
        <TextField ref={first} id={`${ids}-updates`} value={draft.updates} spellCheck={false} readOnly={readOnly}
          disabled={disabled} aria-invalid={updatesReason !== undefined}
          onChange={(event) => setDraft({ kind: 'list', updates: event.currentTarget.value })}
          onBlur={() => commit(draft)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.nativeEvent.isComposing) {
              event.preventDefault();
              commit(draft);
            }
          }} />
      </FieldRow>}
      {draft.kind === 'every' && <div className="grid grid-cols-3 gap-1">
        {number('every', draft.every, (every) => ({ ...draft, every }))}
        {number('from', draft.from, (from) => ({ ...draft, from }))}
        {number('through', draft.through, (through) => ({ ...draft, through }))}
      </div>}
      {onToSweep && <Button className="justify-self-start" disabled={disabled || !same(draft, saved)} onClick={onToSweep}>
        {studioName('toSweep')}
      </Button>}
  </Section></div>;
}

function Checkboxes({ options, selected, disabled, onToggle }: {
  options: readonly Choice[];
  selected: readonly string[];
  disabled?: boolean;
  onToggle(id: string, checked: boolean): void;
}) {
  return <ScrollFade className="grid max-h-40">
    {options.map(({ id, label }) => <Checkbox key={id} label={label} checked={selected.includes(id)} disabled={disabled}
      onCheckedChange={(checked) => onToggle(id, checked)} />)}
  </ScrollFade>;
}

function toggle(ids: readonly string[], id: string, checked: boolean) {
  return checked ? [...ids, id] : ids.filter((entry) => entry !== id);
}

function BestEditor({ rule, sweeps, disabled, onChange }: {
  rule: Best;
  sweeps: PlanStepView['sweeps'];
  disabled: boolean;
  onChange(rule: Best): void;
}) {
  return <div className="value-field-parts grid gap-1">
    <label className="grid gap-1">
      {studioName('metric')}
      <TextField value={rule.by} spellCheck={false} onValueChange={(by) => onChange({ ...rule, by })} />
    </label>
    <div className="grid grid-cols-2 gap-1">
      <label className="grid gap-1">
        {studioName('order')}
        <Select aria-label={studioName('order')} value={rule.order ?? 'max'} disabled={disabled}
          items={[{ value: 'min', label: studioName('lowest') }, { value: 'max', label: studioName('highest') }]}
          onValueChange={(order) => onChange({ ...rule, order })} />
      </label>
      <label className="grid gap-1">
        {studioName('count')}
        <NumberField className="min-w-0" min={1} step={1}
          value={Number.isNaN(rule.k) ? null : rule.k}
          onValueChange={(k) => onChange({ ...rule, k: k ?? Number.NaN })} />
      </label>
    </div>
    {sweeps.length > 0 && <fieldset>
      <legend>{studioName('per')}</legend>
      <Checkboxes options={sweeps} selected={rule.per ?? []} disabled={disabled}
        onToggle={(id, checked) => onChange({ ...rule, per: toggle(rule.per ?? [], id, checked) })} />
    </fieldset>}
  </div>;
}

/**
 * Which trials the Evaluate runs: every condition, chosen conditions, or a number sampled from the task. A choice
 * saves once it names some trials.
 */
export function TrialsEditor({ initial, domain, disabled, refusal, onSave }: {
  initial: TrialSet; domain: TrialDomain; disabled: boolean; refusal?: string; onSave(rule: TrialSet): void;
}) {
  const [rule, setRule] = useDraft(initial);
  const count = useId();
  const first = useRef<HTMLInputElement>(null);
  function commit(next: TrialSet) {
    setRule(next);
    if (!disabled && validTrialSet(next, domain) && !same(next, initial)) onSave(next);
  }
  const choices = domain.kind === 'fixed'
    ? [
      { value: 'every_condition', label: studioName('everyCondition') },
      { value: 'conditions', label: studioName('selectedConditions'), disabled: !domain.conditions.length },
      ...rule.kind === 'sample' ? [{ value: 'sample', label: studioReason('unavailable', studioName('sample')), disabled: true }] : [],
    ]
    : [
      ...!validTrialSet(rule, domain) && rule.kind !== 'sample'
        ? [{ value: rule.kind, label: studioReason('unavailable', studioName('conditions')), disabled: true }] : [],
      { value: 'sample', label: studioName('sample') },
    ];
  const sampling = rule.kind === 'sample' && domain.kind === 'sampler';
  return <Section level="field" name={studioName('trials')} bodyClassName={members}
    control={<>
      <Select aria-label={studioName('trials')} value={rule.kind} disabled={disabled} items={choices}
        aria-invalid={!sampling && refusal !== undefined} finalFocus={() => first.current}
        onValueChange={(kind) => {
          if (kind === initial.kind) return commit(initial);
          switch (kind) {
            case 'every_condition': return commit({ kind: 'every_condition' });
            case 'conditions': return setRule({ kind: 'conditions', ids: [] });
            case 'sample': return setRule({ kind: 'sample', count: Number.NaN });
          }
        }} />
      {!sampling && refusal && <p role="alert">{refusal}</p>}
    </>}>
    {rule.kind === 'conditions' && domain.kind === 'fixed' && <fieldset disabled={disabled}>
      <Checkboxes options={[...conditionLabels(domain.conditions)].map(([id, label]) => ({ id, label }))} selected={rule.ids} disabled={disabled}
        onToggle={(id, checked) => commit({ kind: 'conditions', ids: toggle(rule.ids, id, checked) })} />
    </fieldset>}
    {sampling && <FieldRow label={studioName('count')} htmlFor={count} below={refusal && <p role="alert">{refusal}</p>}>
      <NumberField ref={first} id={count} min={1} max={10_000} step={1} disabled={disabled} invalid={refusal !== undefined}
        value={Number.isNaN(rule.count) ? null : rule.count}
        onValueChange={(next) => setRule({ kind: 'sample', count: next ?? Number.NaN })}
        onValueCommitted={(next) => commit({ kind: 'sample', count: next ?? Number.NaN })} />
    </FieldRow>}
  </Section>;
}

/** A Train selection's restriction: its rows, its sweep values, and Best, written at `where`. */
export function SubjectSubset({ terms, where, hasWhere, source, sourceRows, disabled, refusal, onEdit, onToRows }: {
  terms: SubsetTerms;
  /** The pointer of the selection's `where`. */
  where: string;
  hasWhere: boolean;
  source: PlanStepView;
  sourceRows: readonly SelectedRow[];
  disabled: boolean;
  refusal?: string;
  onEdit(name: string, edit: ProjectValueEdit): void;
  onToRows?(): void;
}) {
  const [draft, setDraft] = useDraft(terms);
  const changed = JSON.stringify(draft) !== JSON.stringify(terms);
  const best = draft.best ? BestSchema.safeParse({ ...draft.best, by: draft.best.by.trim() }) : undefined;
  const hasMembers = draft.rows?.length !== 0 && Object.values(draft.axes ?? {}).every((values) => values.length > 0);
  const canApplySubject = changed && hasMembers && (!best || best.success);
  const address = ({ variant, row }: SelectedRow) => savedRunId({ step: source.id, variant, row });
  const labels = new Map(source.runs.map(({ id, label }) => [id, label]));
  const rows = sourceRows.map((row) => ({ id: address(row), label: labels.get(address(row)) ?? row.row }));
  const selectedRows = draft.rows?.map(address) ?? [];
  const editRows = terms.rows !== undefined;
  const uniqueRows = !draft.rows || new Set(draft.rows.map(({ key }) => key)).size === draft.rows.length;
  function toggleRow(id: string, checked: boolean) {
    const selected = new Set(toggle(selectedRows, id, checked));
    const picked = sourceRows.filter((row) => selected.has(address(row)));
    const initial = terms.rows ?? [];
    const entries = new Set(initial.map(address));
    const ordinary = sourceRows.filter((row) => entries.has(address(row)));
    const hasAuthoredOrder = ordinary.length !== initial.length || ordinary.some((row, index) =>
      row.key !== initial[index].key || address(row) !== address(initial[index]));
    const rows = hasAuthoredOrder
      ? [...initial.filter((row) => selected.has(address(row))), ...picked.filter((row) => !entries.has(address(row)))]
      : picked;
    setDraft({ ...draft, rows });
  }
  const sweeps = editRows ? source.sweeps.filter(({ id }) => terms.axes?.[id] !== undefined) : source.sweeps;
  return <Section level="field" name={studioName('runs')} bodyClassName={members}><fieldset className="contents" disabled={disabled}>
    {editRows && rows.length > 0 && <Checkboxes options={rows} selected={selectedRows} disabled={disabled} onToggle={toggleRow} />}
    {sweeps.map(({ id, label, values }) => {
      const selected = draft.axes?.[id] ?? values.map((value) => value.id);
      return <fieldset key={id}>
        <legend>{label}</legend>
        <Checkboxes options={values} selected={selected} disabled={disabled}
          onToggle={(valueId, checked) => setDraft({
            ...draft, axes: { ...draft.axes, [id]: toggle(selected, valueId, checked) },
          })} />
      </fieldset>;
    })}
    <Checkbox label={studioName('best')} checked={draft.best != null} disabled={disabled} onCheckedChange={(checked) => {
      const { best: _best, ...staticTerms } = draft;
      setDraft(checked ? { ...staticTerms, best: { by: '', k: 1, order: 'min' } } : staticTerms);
    }} />
    {draft.best && <BestEditor rule={draft.best} sweeps={source.sweeps} disabled={disabled}
      onChange={(rule) => setDraft({ ...draft, best: rule })} />}
    <Button className="justify-self-start" disabled={disabled || !canApplySubject || !uniqueRows}
      onClick={() => onEdit(studioName('applySubset'), {
        op: hasWhere ? 'replace' : 'add', path: where,
        value: { ...draft, ...(best?.success && { best: best.data }) },
      })}>{studioName('applySubset')}</Button>
    {onToRows && <Button className="justify-self-start" disabled={disabled || changed} onClick={onToRows}>{studioName('toRows')}</Button>}
    {changed && !hasMembers && <p role="status">{studioReason('emptySelection', studioName('runs'))}</p>}
    {refusal && <p role="alert">{refusal}</p>}
  </fieldset></Section>;
}

/**
 * An Evaluate step's subject, and the Task and Objective an initial Model subject declares. The subject is an
 * initial Model document or a Train step's checkpoint; a checkpoint subject reads its Task and Objective from its
 * Train step.
 */
export function EvaluateSubjectFields({ step, project, trains, picking, selection, refusal, onEdit }: {
  step: Step;
  project: AuthoredProject;
  trains: readonly PlanStepView[];
  picking: ElementSelectionStore;
  /** A Train selection's restriction, and why the plan refuses it. */
  selection?: (index: number, reference: StepReference) => { detail?: ReactNode; problem?: string };
  /** Why the last edit of the subject was refused. */
  refusal?: string;
  onEdit?: (name: string, edits: ProjectValueEdit[], documents: readonly string[]) => void;
}) {
  const labels = useId();
  const subject = evaluateSubject(step);
  const choose = (name: string, next: EvaluateSubject) => {
    const edits = evaluateSubjectEdits(step, next);
    if (edits.length) onEdit?.(name, edits, next.kind === 'document'
      ? [next.document, next.task, next.objective].filter((id): id is string => id !== undefined) : [next.step]);
  };
  const fromTrain = (id: string | undefined) => {
    const documents = trains.find((train) => train.id === id)?.documentIds;
    return {
      task: documents?.['payload.trial_documents.task'] ?? documents?.['payload.trial_documents.python_task']
        ?? documentsOfRole(project, 'task')[0],
      objective: documents?.['payload.trial_documents.objective'],
    };
  };
  const name = studioName('subject');
  const trainChoice = (id: string): ReferenceChoice => {
    const train = trains.find((train) => train.id === id);
    return train ? stepChoice(train) : { id: stepChoiceId(id), label: id };
  };
  const trainOf = (choice: ReferenceChoice) => trains.find(({ id }) => stepChoiceId(id) === choice.id);
  const toDocument = (choice: ReferenceChoice, current: string | undefined) => {
    if (choice.id !== documentChoiceId(choice.label)) return;
    choose(name, subject?.kind === 'document' ? { ...subject, document: choice.label }
      : { kind: 'document', document: choice.label, ...fromTrain(current) });
  };
  const subjectId = `${labels}-subject`;
  if (subject?.kind === 'steps') {
    const { selections } = subject;
    const change = (next: InputBinding) => {
      const reads = stepSelections(next)?.map(({ step }) => step) ?? [];
      const fans = (stepSelections(next)?.length ?? 0) > 1;
      onEdit?.(name, [bindingEdit(step.inputs, 'subject', next), ...fans ? subjectAxisEdits(step) : []], [...new Set(reads)]);
    };
    return <InputSelections label={name} labelId={subjectId} picking={picking} refusal={refusal}
      groups={[stepGroup(trains)]}
      rows={selections.map((reference, index) => ({
        value: trainChoice(reference.step),
        ...selections.length === 1 && { groups: [documentGroup(project, 'model'), stepGroup(trains)] },
        ...selection?.(index, reference),
      }))}
      onChange={onEdit && ((gesture) => {
        if (gesture.kind === 'remove') return change(changedSelections(selections, gesture));
        const train = trainOf(gesture.choice);
        if (gesture.kind === 'add') {
          if (train) change(changedSelections(selections, {
            kind: 'add', reference: { step: train.id, checkpoint: selections[0].checkpoint ?? 'last' },
          }));
        } else if (!train) toDocument(gesture.choice, selections[0].step);
        else if (selections.length === 1) choose(name, { kind: 'step', step: train.id });
        else if (train.id !== selections[gesture.index].step) {
          change(changedSelections(selections, { kind: 'replace', index: gesture.index, reference: { step: train.id } }));
        }
      })} />;
  }
  const value: ReferenceChoice | null = subject?.kind === 'document' ? documentChoice('model', subject.document) : null;
  const role = (kind: 'task' | 'objective') => subject?.kind === 'document' && <FieldRow label={studioName(kind)}
    labelId={`${labels}-${kind}`}>
    <ReferenceField label={studioName(kind)} aria-labelledby={`${labels}-${kind}`}
      groups={[documentGroup(project, kind)]}
      value={subject[kind] === undefined ? null : documentChoice(kind, subject[kind]!)}
      onChoose={onEdit && ((choice) => choose(studioName(kind), { ...subject, [kind]: choice.label }))} />
  </FieldRow>;
  return <>
    <FieldRow label={name} labelId={subjectId} below={refusal && <p role="alert">{refusal}</p>}>
      <ReferenceField label={name} aria-labelledby={subjectId} picking={picking} value={value}
        groups={[documentGroup(project, 'model'), stepGroup(trains)]}
        onChoose={onEdit && ((choice) => {
          const train = trainOf(choice);
          if (train) choose(name, { kind: 'step', step: train.id });
          else toDocument(choice, undefined);
        })} />
    </FieldRow>
    {role('task')}
    {role('objective')}
  </>;
}
