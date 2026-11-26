import { holdsMissing, Shown, type ShownText } from '@/components/ui/ShownText';
import { useId, useRef, type ReactNode } from 'react';
import { useDraft } from '@/hooks/useDraft';
import { studioAccessibleName, studioFormat, studioName, studioReason } from '@/components/activity/studioVocabulary';
import { Button } from '@/components/ui/Button';
import { Checkbox } from '@/components/ui/Checkbox';
import { NumberField } from '@/components/ui/NumberField';
import { Select } from '@/components/ui/Select';
import { TextField } from '@/components/ui/TextField';
import { FieldProblem } from '@/components/ui/FieldProblem';
import { FieldRow } from '@/components/ui/FieldRow';
import { Section } from '@/components/ui/Section';
import { ReferenceField } from '@/components/values/ReferenceField';
import {
  checkpointDraft, checkpointDraftRule, emptyCheckpointDraft, validTrialSet,
  type CheckpointDraft, type CheckpointDraftField, type CheckpointKind, type CheckpointRange, type EditableCheckpointRule,
  type TrialDomain, type TrialSet,
} from './evaluateAuthoring';
import type { AuthoredProject } from './authoredEdits';
import { chosenDocument, documentChoice, documentGroup } from './referenceChoices';
import type { ShownDocuments } from './memberLabels';
import { ScrollFade } from '@/components/ui/ScrollFade';
import '@/components/values/valueField.css';

type Choice = { id: string; label: ShownText };

/** The inspector fields of a step an edit can be refused on: its input, a selection's Checkpoint or Runs by index, and Trials. */
export type StepField = 'input' | 'checkpoint' | `subset.${number}` | 'trials' | 'method';

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
 * beneath. A kind with updates saves once they are written, a list that says no rule as its text; Escape restores the
 * saved one.
 */
export function CheckpointEditor({ initial, step, kinds = checkpointKinds, range, disabled, readOnly, problem, onSave, onToSweep }: {
  initial: EditableCheckpointRule;
  /** The kinds of rule the input takes. */
  kinds?: readonly CheckpointKind[];
  /** The name of the one step whose selection this editor holds, beside other selections' editors. */
  step?: string;
  range?: CheckpointRange;
  disabled: boolean;
  readOnly?: boolean;
  /** Why the saved rule, or the last one saved, is refused, or the checkpoint it asks for does not exist. */
  problem?: string;
  onSave(rule: EditableCheckpointRule): void;
  onToSweep?(): void;
}) {
  const saved = checkpointDraft(initial);
  const [draft, setDraft] = useDraft(saved);
  const ids = useId();
  const first = useRef<HTMLInputElement>(null);
  const held = checkpointDraftRule(draft, range);
  const name = studioName('checkpoint');
  const fixed = disabled || readOnly;
  /** The draft's own problem while it is written; once saved, the plan's problem on the saved rule. */
  const reason = (field: CheckpointDraftField) => held.problem?.field === field ? held.problem.reason
    : field === firstMember[draft.kind] ? problem : undefined;
  /** The rule last saved from this editor, and the rule it replaced: until the saved rule changes, it is not saved again. */
  const sent = useRef<{ initial: EditableCheckpointRule; rule: EditableCheckpointRule }>();
  function commit(next: CheckpointDraft) {
    setDraft(next);
    const { rule } = checkpointDraftRule(next, range);
    if (fixed || rule === undefined || same(rule, initial)) return;
    if (sent.current && same(sent.current.initial, initial) && same(sent.current.rule, rule)) return;
    sent.current = { initial, rule };
    onSave(rule);
  }
  function number(field: 'at' | 'every' | 'from' | 'through', value: number | null, next: (value: number | null) => CheckpointDraft) {
    const why = reason(field);
    return <FieldRow key={field} label={studioName(field === 'at' ? 'update' : field)} htmlFor={`${ids}-${field}`}>
      <NumberField ref={field === firstMember[draft.kind] ? first : undefined} id={`${ids}-${field}`}
        min={field === 'at' ? 0 : 1} step={1} value={value} invalid={why !== undefined} disabled={disabled} readOnly={readOnly}
        aria-describedby={why ? `${ids}-${field}-problem` : undefined}
        trailing={why && <FieldProblem reason={why} id={`${ids}-${field}-problem`} />}
        onValueChange={(changed) => setDraft(next(changed))} onValueCommitted={(changed) => commit(next(changed))} />
    </FieldRow>;
  }
  const updatesReason = reason('updates');
  return <div className="contents" onKeyDown={(event) => {
    if (event.key !== 'Escape' || same(draft, saved)) return;
    event.preventDefault();
    event.stopPropagation();
    setDraft(saved);
  }}><Section level="field" name={step ?? name} bodyClassName={members}
    control={<>
      <Select aria-label={studioAccessibleName('checkpoint', step)} value={draft.kind}
        disabled={fixed}
        aria-invalid={draft.kind === 'last' && problem !== undefined}
        aria-describedby={draft.kind === 'last' && problem ? `${ids}-last-problem` : undefined}
        trailing={draft.kind === 'last' && problem && <FieldProblem reason={problem} id={`${ids}-last-problem`} />}
        items={kinds.map((kind) => ({ value: kind, label: studioName(kind) }))}
        finalFocus={() => first.current}
        onValueChange={(kind) => commit(kind === saved.kind ? saved : emptyCheckpointDraft(kind))} />
    </>}>
      {draft.kind === 'at' && number('at', draft.at, (at) => ({ kind: 'at', at }))}
      {draft.kind === 'list' && <FieldRow label={studioName('updates')} htmlFor={`${ids}-updates`}>
        <TextField ref={first} id={`${ids}-updates`} value={draft.updates} spellCheck={false} readOnly={readOnly}
          disabled={disabled} invalid={updatesReason !== undefined}
          aria-describedby={updatesReason ? `${ids}-updates-problem` : undefined}
          trailing={updatesReason && <FieldProblem reason={updatesReason} id={`${ids}-updates-problem`} />}
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
    {options.map(({ id, label }) => <Checkbox key={id} label={<Shown text={label} />} checked={selected.includes(id)} disabled={disabled}
      aria-invalid={holdsMissing(label) || undefined}
      onCheckedChange={(checked) => onToggle(id, checked)} />)}
  </ScrollFade>;
}

function toggle(ids: readonly string[], id: string, checked: boolean) {
  return checked ? [...ids, id] : ids.filter((entry) => entry !== id);
}

/**
 * Which trials the Evaluate runs: every condition, chosen conditions, or a number sampled from the task. A choice
 * saves once it names some trials. A saved sample's number is `count`, the step's own field, which takes a sweep.
 */
export function TrialsEditor({ initial, domain, disabled, refusal, count: savedCount, onSave }: {
  initial: TrialSet; domain: TrialDomain; disabled: boolean; refusal?: string; count?: ReactNode; onSave(rule: TrialSet): void;
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
        aria-describedby={!sampling && refusal ? `${count}-kind-problem` : undefined}
        trailing={!sampling && refusal && <FieldProblem reason={refusal} id={`${count}-kind-problem`} />}
        onValueChange={(kind) => {
          if (kind === initial.kind) return commit(initial);
          switch (kind) {
            case 'every_condition': return commit({ kind: 'every_condition' });
            case 'conditions': return setRule({ kind: 'conditions', ids: [] });
            case 'sample': return setRule({ kind: 'sample', count: Number.NaN });
          }
        }} />
    </>}>
    {rule.kind === 'conditions' && domain.kind === 'fixed' && <fieldset disabled={disabled}>
      <Checkboxes options={domain.conditions.map(({ id, name }, index) => ({
        id, label: name ?? studioFormat('trialOf', { index: index + 1, count: domain.conditions.length }) }))} selected={rule.ids} disabled={disabled}
        onToggle={(id, checked) => commit({ kind: 'conditions', ids: toggle(rule.ids, id, checked) })} />
    </fieldset>}
    {sampling && (initial.kind === 'sample' && savedCount || <FieldRow label={studioName('count')} htmlFor={count}>
      <NumberField ref={first} id={count} min={1} max={10_000} step={1} disabled={disabled} invalid={refusal !== undefined}
        aria-describedby={refusal ? `${count}-problem` : undefined}
        trailing={refusal && <FieldProblem reason={refusal} id={`${count}-problem`} />}
        value={Number.isNaN(rule.count) ? null : rule.count}
        onValueChange={(next) => setRule({ kind: 'sample', count: next ?? Number.NaN })}
        onValueCommitted={(next) => commit({ kind: 'sample', count: next ?? Number.NaN })} />
    </FieldRow>)}
  </Section>;
}

/** The Task and Objective an initial Model subject declares; a checkpoint subject reads them from its Train step. */
export function SubjectDocuments({ subject, project, documents, onChoose }: {
  subject: { task?: string; objective?: string };
  project: AuthoredProject;
  documents: ShownDocuments;
  onChoose?: (kind: 'task' | 'objective', document: string) => void;
}) {
  const labels = useId();
  return <>{(['task', 'objective'] as const).map((kind) => <FieldRow key={kind} label={studioName(kind)} labelId={`${labels}-${kind}`}>
    <ReferenceField label={studioName(kind)} aria-labelledby={`${labels}-${kind}`} groups={[documentGroup(project, kind)]}
      value={subject[kind] === undefined ? null : documentChoice(documents, kind, subject[kind]!)}
      onChoose={onChoose && ((choice) => { const document = chosenDocument(choice); if (document) onChoose(kind, document); })} />
  </FieldRow>)}</>;
}
