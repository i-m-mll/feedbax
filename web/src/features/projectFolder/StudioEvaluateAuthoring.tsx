import { useId, type ReactNode } from 'react';
import { useDraft } from '@/hooks/useDraft';
import { studioName, studioReason } from '@/components/activity/studioVocabulary';
import { Button } from '@/components/ui/Button';
import { Checkbox } from '@/components/ui/Checkbox';
import { NumberField } from '@/components/ui/NumberField';
import { Select } from '@/components/ui/Select';
import { TextField } from '@/components/ui/TextField';
import { FieldRow } from '@/components/ui/FieldRow';
import { ReferenceField, type ReferenceChoice } from '@/components/values/ReferenceField';
import type { PlanStepView } from '@/features/pipeline/planView';
import type { ElementSelectionStore } from '@/features/projectWindow/elementSelection';
import {
  BestSchema, type Best, type ProjectValueEdit, type SelectedRow, type Step, type SubsetTerms,
} from '@/generated/studioContracts';
import type { AuthoredProject } from './authoredEdits';
import {
  evaluateSubject, evaluateSubjectEdits, formatCheckpointRule, parseCheckpointRule, validTrialSet,
  type EditableCheckpointRule, type EvaluateSubject, type TrialDomain, type TrialSet,
} from './evaluateAuthoring';
import {
  documentChoice, documentChoiceId, documentGroup, stepChoice, stepChoiceId, stepGroup,
} from './referenceChoices';
import { documentsOfRole } from './useProjectDocuments';
import { savedRunId } from './projectWindowPlan';
import { ScrollFade } from '@/components/ui/ScrollFade';

type Choice = { id: string; label: string };

/** The Evaluate fields an edit can be refused on. */
export type EvaluateField = 'subject' | 'checkpoints' | 'subset' | 'trialSet';

export function CheckpointRuleEditor({ initial, disabled, readOnly, refusal, onSave, onToSweep }: {
  initial: EditableCheckpointRule;
  disabled: boolean;
  readOnly?: boolean;
  refusal?: string;
  onSave(rule: EditableCheckpointRule): void;
  onToSweep?(): void;
}) {
  const saved = formatCheckpointRule(initial);
  const [draft, setDraft] = useDraft(saved);
  const errorId = useId();
  const parsed = parseCheckpointRule(draft);
  const reason = 'reason' in parsed ? parsed.reason : undefined;
  const changed = 'rule' in parsed && formatCheckpointRule(parsed.rule) !== saved;

  function save() {
    if (!disabled && !readOnly && changed && 'rule' in parsed) onSave(parsed.rule);
  }

  return <fieldset className="grid gap-1" disabled={disabled}>
    <label className="grid gap-1 font-semibold">
      {studioName('checkpoints')}
      <TextField className="font-normal" value={draft} spellCheck={false} readOnly={readOnly} disabled={disabled}
        aria-invalid={reason !== undefined}
        aria-describedby={reason ? errorId : undefined}
        onChange={(event) => setDraft(event.currentTarget.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter' && !event.nativeEvent.isComposing) {
            event.preventDefault();
            save();
          }
        }} />
    </label>
    {reason && <p id={errorId} className="text-shell-muted" role="status">{reason}</p>}
    {refusal && <p role="alert">{refusal}</p>}
    {!readOnly && <Button className="justify-self-start" disabled={disabled || !changed} onClick={save}>{studioName('applyCheckpoints')}</Button>}
    {onToSweep && <Button className="justify-self-start" disabled={disabled || draft !== saved} onClick={onToSweep}>{studioName('toSweep')}</Button>}
  </fieldset>;
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
  return <div className="grid gap-1">
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

function TrialSetEditor({ initial, domain, disabled, refusal, onSave }: {
  initial: TrialSet; domain: TrialDomain; disabled: boolean; refusal?: string; onSave(rule: TrialSet): void;
}) {
  const [rule, setRule] = useDraft(initial);
  const select = useId();
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
  return <fieldset aria-label={studioName('trialSet')} className="grid gap-1" disabled={disabled}>
    <label className="grid gap-1 font-semibold">
      {studioName('trialSet')}
      <Select id={select} className="font-normal" aria-label={studioName('trialSet')} value={rule.kind} disabled={disabled}
        items={choices}
        onValueChange={(kind) => {
          switch (kind) {
            case 'every_condition': setRule({ kind: 'every_condition' }); break;
            case 'conditions': setRule({ kind: 'conditions', ids: [] }); break;
            case 'sample': setRule({ kind: 'sample', count: 1 }); break;
          }
        }} />
    </label>
    {rule.kind === 'conditions' && domain.kind === 'fixed' && <Checkboxes
      options={domain.conditions.map((id) => ({ id, label: id }))}
      selected={rule.ids} disabled={disabled} onToggle={(id, checked) => setRule({ kind: 'conditions', ids: toggle(rule.ids, id, checked) })} />}
    {rule.kind === 'sample' && domain.kind === 'sampler' && <label className="grid gap-1">
      {studioName('sampleCount')}
      <NumberField min={1} max={10_000}
        value={Number.isNaN(rule.count) ? null : rule.count}
        onValueChange={(count) => setRule({ kind: 'sample', count: count ?? Number.NaN })} />
    </label>}
    <Button className="justify-self-start" disabled={disabled || !validTrialSet(rule, domain) || JSON.stringify(rule) === JSON.stringify(initial)}
      onClick={() => onSave(rule)}>{studioName('applyTrials')}</Button>
    {refusal && <p role="alert">{refusal}</p>}
  </fieldset>;
}

export function StudioEvaluateAuthoring({
  authored, source, sourceRows, domain, disabled, editSubject, refusals, onEdit, onToRows,
}: {
  authored: { terms: SubsetTerms; trialSet: TrialSet; hasWhere: boolean };
  source: PlanStepView;
  sourceRows: readonly SelectedRow[];
  domain: TrialDomain | undefined;
  disabled: boolean;
  editSubject: boolean;
  refusals: { subset?: string; trialSet?: string };
  onEdit(field: 'subset' | 'trialSet', name: string, edit: ProjectValueEdit): void;
  onToRows?(): void;
}) {
  const { terms, trialSet } = authored;
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
  return <div className="grid gap-3">
    {editSubject && <fieldset className="grid gap-1" disabled={disabled}>
      <legend className="font-semibold">{studioName('subjectSubset')}</legend>
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
        onClick={() => onEdit('subset', studioName('applySubset'), {
          op: authored.hasWhere ? 'replace' : 'add', path: '/inputs/subject/where',
          value: { ...draft, ...(best?.success && { best: best.data }) },
        })}>{studioName('applySubset')}</Button>
      {onToRows && <Button className="justify-self-start" disabled={disabled || changed} onClick={onToRows}>{studioName('toRows')}</Button>}
      {changed && !hasMembers && <p role="status">{studioReason('emptySelection', studioName('runs'))}</p>}
      {refusals.subset && <p role="alert">{refusals.subset}</p>}
    </fieldset>}
    {domain && <TrialSetEditor initial={trialSet} domain={domain} disabled={disabled} refusal={refusals.trialSet}
      onSave={(rule) => onEdit('trialSet', studioName('applyTrials'), { op: 'replace', path: '/base/inline/params/trial_set', value: rule })} />}
  </div>;
}

/**
 * An Evaluate step's subject, and the Task and Objective an initial Model subject declares. The subject is an
 * initial Model document or a Train step's checkpoint; a checkpoint subject reads its Task and Objective from its
 * Train step.
 */
export function EvaluateSubjectFields({ step, project, trains, picking, subset, refusal, onEdit }: {
  step: Step;
  project: AuthoredProject;
  trains: readonly PlanStepView[];
  picking: ElementSelectionStore;
  /** The subject's subset, under its field. */
  subset?: ReactNode;
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
  const value: ReferenceChoice | null = subject?.kind === 'document' ? documentChoice('model', subject.document)
    : subject ? trains.find(({ id }) => id === subject.step) ? stepChoice(trains.find(({ id }) => id === subject.step)!)
      : { id: stepChoiceId(subject.step), label: subject.step } : null;
  const role = (kind: 'task' | 'objective') => subject?.kind === 'document' && <FieldRow label={studioName(kind)}
    labelId={`${labels}-${kind}`}>
    <ReferenceField label={studioName(kind)} aria-labelledby={`${labels}-${kind}`}
      groups={[documentGroup(project, kind)]}
      value={subject[kind] === undefined ? null : documentChoice(kind, subject[kind]!)}
      onChoose={onEdit && ((choice) => choose(studioName(kind), { ...subject, [kind]: choice.label }))} />
  </FieldRow>;
  return <>
    <FieldRow label={name} labelId={`${labels}-subject`}
      below={(refusal || subset) && <>{refusal && <p role="alert">{refusal}</p>}{subset}</>}>
      <ReferenceField label={name} aria-labelledby={`${labels}-subject`} picking={picking} value={value}
        groups={[documentGroup(project, 'model'), stepGroup(trains)]}
        onChoose={onEdit && ((choice) => {
          const train = trains.find(({ id }) => stepChoiceId(id) === choice.id);
          if (train) choose(name, { kind: 'step', step: train.id });
          else if (choice.id === documentChoiceId(choice.label)) {
            choose(name, subject?.kind === 'document' ? { ...subject, document: choice.label }
              : { kind: 'document', document: choice.label, ...fromTrain(subject?.step) });
          }
        })} />
    </FieldRow>
    {role('task')}
    {role('objective')}
  </>;
}
