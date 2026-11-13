import { useId, useLayoutEffect, useRef, useState } from 'react';
import { studioName, studioReason } from '@/components/activity/studioVocabulary';
import type { PlanStepView } from '@/features/pipeline/planView';
import type { ProjectValueEdit, StudioSavedTrainRuns, SubsetTerms } from '@/generated/studioContracts';
import {
  formatCheckpointRule, parseCheckpointRule, validTrialSet,
  type EditableCheckpointRule, type TrialDomain, type TrialSet,
} from './evaluateAuthoring';
import { savedRunId } from './projectWindowPlan';

type Choice = { id: string; label: string };
const field = 'rounded border border-shell-divider bg-shell-chrome px-2 py-1';
const apply = field + ' justify-self-start hover:bg-shell-hover disabled:opacity-50';

export function CheckpointRuleEditor({ initial, disabled, readOnly, onSave, onToSweep }: {
  initial: EditableCheckpointRule;
  disabled: boolean;
  readOnly?: boolean;
  onSave(rule: EditableCheckpointRule): void;
  onToSweep?(): void;
}) {
  const saved = formatCheckpointRule(initial);
  const [draft, setDraft] = useState(saved);
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
      <input className={field + ' font-normal'} type="text" value={draft} spellCheck={false} readOnly={readOnly}
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
    {!readOnly && <button type="button" className={apply} disabled={!changed} onClick={save}>{studioName('applyCheckpoints')}</button>}
    {onToSweep && <button type="button" className={apply} disabled={draft !== saved} onClick={onToSweep}>{studioName('toSweep')}</button>}
  </fieldset>;
}

function Checkboxes({ options, selected, onToggle }: {
  options: readonly Choice[];
  selected: readonly string[];
  onToggle(id: string, checked: boolean): void;
}) {
  return <div className="grid max-h-40 gap-1 overflow-auto">
    {options.map(({ id, label }) => <label key={id} className="flex gap-2">
      <input type="checkbox" checked={selected.includes(id)}
        onChange={(event) => onToggle(id, event.currentTarget.checked)} />
      <span className="break-all">{label}</span>
    </label>)}
  </div>;
}

function toggle(ids: readonly string[], id: string, checked: boolean) {
  return checked ? [...ids, id] : ids.filter((entry) => entry !== id);
}

function TrialSetEditor({ initial, domain, disabled, focus, onFocused, onSave }: {
  initial: TrialSet; domain: TrialDomain; disabled: boolean; onSave(rule: TrialSet): void;
  focus: boolean; onFocused(): void;
}) {
  const [rule, setRule] = useState(initial);
  const select = useRef<HTMLSelectElement>(null);
  useLayoutEffect(() => {
    if (focus && !disabled && select.current) {
      select.current.focus();
      onFocused();
    }
  }, [focus, disabled, onFocused]);
  return <fieldset aria-label={studioName('trialSet')} className="grid gap-1" disabled={disabled}>
    <label className="grid gap-1 font-semibold">
      {studioName('trialSet')}
      <select ref={select} className={field + ' font-normal'} value={rule.kind}
        onChange={(event) => {
          switch (event.currentTarget.value) {
            case 'every_condition': setRule({ kind: 'every_condition' }); break;
            case 'conditions': setRule({ kind: 'conditions', ids: [] }); break;
            case 'sample': setRule({ kind: 'sample', count: 1 }); break;
          }
        }}>
        {!validTrialSet(rule, domain) && domain.kind === 'sampler' && rule.kind !== 'sample' &&
          <option value={rule.kind} disabled>{studioReason('unavailable', studioName('conditions'))}</option>}
        {domain.kind === 'fixed' && rule.kind === 'sample' && <option value="sample" disabled>{studioReason('unavailable', studioName('sample'))}</option>}
        {domain.kind === 'fixed' ? <>
          <option value="every_condition">{studioName('everyCondition')}</option>
          <option value="conditions" disabled={!domain.conditions.length}>{studioName('selectedConditions')}</option>
        </> : <option value="sample">{studioName('sample')}</option>}
      </select>
    </label>
    {rule.kind === 'conditions' && domain.kind === 'fixed' && <Checkboxes
      options={domain.conditions.map((id) => ({ id, label: id }))}
      selected={rule.ids} onToggle={(id, checked) => setRule({ kind: 'conditions', ids: toggle(rule.ids, id, checked) })} />}
    {rule.kind === 'sample' && domain.kind === 'sampler' && <label className="grid gap-1">
      {studioName('sampleCount')}
      <input type="number" className={field} min={1} max={10_000}
        value={Number.isNaN(rule.count) ? '' : rule.count}
        onChange={(event) => setRule({ kind: 'sample', count: event.currentTarget.valueAsNumber })} />
    </label>}
    <button type="button" className={apply} disabled={!validTrialSet(rule, domain) || JSON.stringify(rule) === JSON.stringify(initial)}
      onClick={() => onSave(rule)}>{studioName('applyTrials')}</button>
  </fieldset>;
}

export function StudioEvaluateAuthoring({
  authored, source, projection, domain, disabled, editSubject, focusTrialSet, onTrialSetFocused, onEdit,
}: {
  authored: { terms: SubsetTerms; trialSet: TrialSet; hasWhere: boolean };
  source: PlanStepView;
  projection: StudioSavedTrainRuns;
  domain: TrialDomain | undefined;
  disabled: boolean;
  editSubject: boolean;
  focusTrialSet: boolean;
  onTrialSetFocused(): void;
  onEdit(name: string, edit: ProjectValueEdit): void;
}) {
  const { terms, trialSet } = authored;
  const [draft, setDraft] = useState(terms);
  const changed = JSON.stringify(draft) !== JSON.stringify(terms);
  const canApplySubject = changed && draft.rows?.length !== 0 &&
    Object.values(draft.axes ?? {}).every((values) => values.length > 0);
  const rowKeys = new Map(projection.rows.map(({ key }) => [savedRunId(key), key.row]));
  const rows = source.runs.flatMap(({ id, label }) => {
    const key = rowKeys.get(id);
    return key === undefined ? [] : [{ id: key, label }];
  });
  const editRows = terms.rows !== undefined;
  const sweeps = editRows ? source.sweeps.filter(({ id }) => terms.axes?.[id] !== undefined) : source.sweeps;
  return <div className="grid gap-3">
    {editSubject && (editRows || sweeps.length > 0) && <fieldset className="grid gap-1" disabled={disabled}>
      <legend className="font-semibold">{studioName('subjectSubset')}</legend>
      {editRows && <Checkboxes options={rows} selected={draft.rows ?? []}
        onToggle={(id, checked) => setDraft({ ...draft, rows: toggle(draft.rows ?? [], id, checked) })} />}
      {sweeps.map(({ id, label, values }) => {
        const selected = draft.axes?.[id] ?? values.map((value) => value.id);
        return <fieldset key={id}>
          <legend>{label}</legend>
          <Checkboxes options={values} selected={selected}
            onToggle={(valueId, checked) => setDraft({
              ...draft, axes: { ...draft.axes, [id]: toggle(selected, valueId, checked) },
            })} />
        </fieldset>;
      })}
      <button type="button" className={apply} disabled={!canApplySubject}
        onClick={() => onEdit(studioName('applySubset'), {
          op: authored.hasWhere ? 'replace' : 'add', path: '/inputs/subject/where', value: { ...draft },
        })}>{studioName('applySubset')}</button>
      {changed && !canApplySubject && <p role="status">{studioReason('emptySelection', studioName('runs'))}</p>}
    </fieldset>}
    {domain && <TrialSetEditor initial={trialSet} domain={domain} disabled={disabled} focus={focusTrialSet} onFocused={onTrialSetFocused}
      onSave={(rule) => onEdit(studioName('applyTrials'), { op: 'replace', path: '/base/inline/params/trial_set', value: rule })} />}
  </div>;
}
