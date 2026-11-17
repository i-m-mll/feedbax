import { useId, useRef, useState } from 'react';
import { FunctionSquare } from 'lucide-react';
import type { ModelFieldRequest, StudioEvaluateValueSource, StudioFieldView, StudioPlanReviewSource } from '@/generated/studioContracts';
import { appTip } from '@window-shell/browser/tooltip';
import { studioAccessibleName, studioEvaluateValueSource, studioFieldKindLabel, studioFieldLabel, studioFieldValueLabel, studioName, studioPlanValueSource } from '@/components/activity/studioVocabulary';
import { selectionButton } from '@/features/pipeline/selection';
import { Button, IconButton } from '@/components/ui/Button';
import { Menu, MenuContent, MenuItem, MenuTrigger } from '@/components/ui/Menu';
import { Select } from '@/components/ui/Select';
import { TextField } from '@/components/ui/TextField';

type Edit = NonNullable<ModelFieldRequest['edit']>;
type Commit = (edit: Edit) => Promise<string | undefined>;
type FieldKind = StudioFieldView['site']['kinds'][number];

export function StudioFieldControl({ field, commit, source, planSources, selectedValues, rowSelection = false, multipleRows = false, onChooseValue, valueKind = 'value' }: {
  field: StudioFieldView;
  commit?: Commit;
  source?: StudioEvaluateValueSource;
  planSources?: readonly StudioPlanReviewSource[];
  selectedValues?: ReadonlySet<string>;
  rowSelection?: boolean;
  multipleRows?: boolean;
  onChooseValue?: (id: string, extend: boolean) => void;
  valueKind?: StudioFieldView['site']['kinds'][number];
}) {
  const { label, sweep } = field;
  const fieldId = useId();
  const reasonId = useId();
  const sourceId = useId();
  const textInput = useRef<HTMLInputElement>(null);
  const [draft, setDraft] = useState<string>();
  const [reason, setReason] = useState<string>();
  const [menu, setMenu] = useState(false);
  const [busy, setBusy] = useState(false);
  const [keep, setKeep] = useState<string>();
  const [adding, setAdding] = useState(false);
  const [newValue, setNewValue] = useState('');
  const [members, setMembers] = useState<Record<string, string>>({});
  const readOnly = !commit || busy;
  const ruleReadOnly = readOnly || rowSelection || !field.can_change_axis;
  const kind = sweep ? 'sweep' : valueKind;
  const list = sweep?.kind === 'list' ? sweep : undefined;
  const choices = list?.values ?? field.members ?? [];
  const range = sweep?.kind === 'range' ? sweep.rule : undefined;
  const rangeUnit = field.unit ?? '';
  const sampler = sweep?.kind === 'sampler' ? sweep.rule : undefined;
  const authored = list ? list.values.map(({ value }) => value).join(', ') + (list.values.length === 1 ? ',' : '')
    : field.text;
  const picked = selectedValues ?? new Set<string>();
  const effective = multipleRows ? undefined : field.effective;
  const sourceLabel = planSources?.length ? [...new Set(planSources.map(studioPlanValueSource))].join(', ')
    : source && studioEvaluateValueSource(source);

  async function submit(edit: Edit) {
    if (!commit || busy) return false;
    setBusy(true);
    const failure = await commit(edit);
    setBusy(false);
    setReason(failure);
    if (failure) return false;
    setDraft(undefined);
    setMembers({});
    setAdding(false);
    setNewValue('');
    setKeep(undefined);
    return true;
  }

  function choose(kind: FieldKind) {
    setMenu(false);
    if (kind === 'sweep' && !sweep) {
      void submit({ kind: 'to_sweep' });
    } else if (kind === 'value' && sweep) {
      if (choices.length === 1) void submit({ kind: 'to_value', value_id: choices[0].id });
      else if (choices.length) setKeep(field.selected_value ?? choices[0].id);
    } else {
      textInput.current?.focus();
      textInput.current?.select();
    }
  }

  function updateRange(name: 'start' | 'stop' | 'count' | 'scale', value: string) {
    if (!range) return;
    const next = { ...range, [name]: value };
    void submit({ kind: 'text', text: `${next.scale === 'log' ? 'geomspace' : 'linspace'}(${next.start} ${rangeUnit}, ${next.stop} ${rangeUnit}, ${next.count})` });
  }

  return <section className="grid min-w-0 gap-1 tabular-nums" aria-label={studioFieldLabel('field', label)}>
    <label htmlFor={fieldId}>{label}</label>
    {sourceLabel && <span id={sourceId} className="text-xs text-shell-muted">{sourceLabel}</span>}
    <div className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-1">
      <Menu open={menu} onOpenChange={setMenu}>
        <MenuTrigger render={<IconButton label={studioFieldKindLabel(label, kind)} className={sweep ? undefined : 'text-shell-muted'}
          icon={valueKind === 'formula' && !sweep
            ? <FunctionSquare aria-hidden="true" />
            : <span aria-hidden="true">{sweep ? '◇' : valueKind === 'link' ? '↗' : '·'}</span>} />} />
        <MenuContent aria-label={studioFieldLabel('kinds', label)} finalFocus={textInput}>
          {field.site.kinds.map((value) => <MenuItem key={value}
            disabled={readOnly || rowSelection && value !== kind ||
              value === 'sweep' && !sweep && !field.can_sweep ||
              value === 'value' && !!sweep && (!choices.length || !field.can_change_axis)}
            onClick={() => choose(value)}>{studioName(value)}</MenuItem>)}
        </MenuContent>
      </Menu>
      <TextField ref={textInput} id={fieldId} type="text"
        value={draft ?? (rowSelection && !sweep ? multipleRows ? '' : effective ?? authored : authored)}
        placeholder={multipleRows && !sweep ? studioName('selectedRuns') : undefined}
        readOnly={readOnly} aria-invalid={!!reason || undefined}
        aria-describedby={[sourceLabel && sourceId, reason && reasonId].filter(Boolean).join(' ') || undefined}
        onValueChange={(value) => { setDraft(value); setReason(undefined); }}
        onKeyDown={(event) => {
          if (event.key === 'ContextMenu' || event.key === 'F10' && event.shiftKey) {
            event.preventDefault();
            setMenu(true);
            return;
          }
          if (event.key === 'Escape') {
            event.stopPropagation();
            setDraft(undefined);
            setReason(undefined);
            setMenu(false);
            setKeep(undefined);
          }
          if (event.key === 'Enter' && draft !== undefined) {
            event.preventDefault();
            void submit({ kind: 'text', text: draft });
          }
        }} />
      {effective != null && <output className="text-shell-muted" {...appTip(studioName('selectedValue'))}
        aria-label={studioFieldLabel('selectedValue', label)} aria-description={String(effective)}>◆</output>}
    </div>
    {reason && <p role="alert" id={reasonId} className="text-xs">{reason}</p>}
    {list && <div className="grid gap-1 border-l border-shell-divider pl-2">
      {list.values.map(({ id, label: memberLabel, value }, index) => <div key={id} className="grid grid-cols-[auto_minmax(0,1fr)_minmax(0,1fr)] items-center gap-1">
        {onChooseValue ? <IconButton pressed={picked.has(id)}
          label={studioAccessibleName('select', `${label} ${memberLabel ?? value}`)}
          {...selectionButton((extend) => onChooseValue(id, extend))}
          icon={picked.has(id) ? '◆' : '·'} /> : <span aria-hidden="true" className={picked.has(id) ? 'font-bold' : 'text-shell-muted'}>{picked.has(id) ? '◆' : '·'}</span>}
        <TextField type="text" aria-label={memberLabel ? studioAccessibleName('value', memberLabel) : studioFieldValueLabel(label, index + 1)}
          value={members[id] ?? String(value)} readOnly={readOnly || !list.editable}
          onValueChange={(next) => { setMembers({ ...members, [id]: next }); setReason(undefined); }}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && members[id] !== undefined) {
              event.preventDefault();
              void submit({ kind: 'text', text: members[id], value_id: id });
            }
            if (event.key === 'Escape') { setMembers({}); setReason(undefined); }
          }} />
        {list.editable && <TextField type="text" readOnly={readOnly}
          aria-label={studioFieldValueLabel(label, index + 1, true)} placeholder={studioName('name')} key={memberLabel ?? ''}
          defaultValue={memberLabel ?? ''} onKeyDown={(event) => {
            if (event.key === 'Enter') {
              event.preventDefault();
              void submit({ kind: 'name_value', value_id: id, label: event.currentTarget.value });
            }
            if (event.key === 'Escape') event.currentTarget.value = memberLabel ?? '';
          }} />}
      </div>)}
      {adding && <form className="grid grid-cols-[1fr_auto] gap-1" onSubmit={(event) => {
        event.preventDefault();
        void submit({ kind: 'text', text: `${authored.replace(/,$/, '')}, ${newValue}` });
      }}>
        <TextField aria-label={studioAccessibleName('newValue', label)} autoFocus value={newValue} onValueChange={setNewValue} />
        <Button type="submit" disabled={readOnly || !newValue.trim()}>{studioName('add')}</Button>
      </form>}
      {!adding && field.can_change_axis && !rowSelection && <Button variant="ghost" className="justify-self-start"
        disabled={readOnly} onClick={() => setAdding(true)}>{studioName('addValue')}</Button>}
    </div>}
    {range && <div className="grid gap-1 border-l border-shell-divider pl-2">
      {([['start', 'first'], ['stop', 'last'], ['count', 'count']] as const).map(([key, name]) =>
        <label key={key} className="grid grid-cols-[auto_1fr] items-center gap-2">{studioName(name)}
          <TextField type="text" key={range[key]} defaultValue={range[key]} readOnly={ruleReadOnly}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !ruleReadOnly) { event.preventDefault(); updateRange(key, event.currentTarget.value); }
              if (event.key === 'Escape') event.currentTarget.value = String(range[key]);
            }} />
        </label>)}
      <div className="grid grid-cols-[auto_1fr] items-center gap-2">{studioName('spacing')}
        <Select aria-label={studioName('spacing')} value={range.scale ?? 'linear'} disabled={ruleReadOnly}
          items={[{ value: 'linear', label: studioName('linear') }, { value: 'log', label: studioName('logarithmic') }]}
          onValueChange={(scale) => updateRange('scale', scale)} />
      </div>
    </div>}
    {sampler && typeof sampler.distribution !== 'string' && <div className="grid gap-1 border-l border-shell-divider pl-2">
      <label className="grid grid-cols-[auto_1fr] items-center gap-2">{studioName('count')}
        <TextField key={sampler.count} defaultValue={sampler.count} readOnly={ruleReadOnly}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !ruleReadOnly) {
              const text = field.text.replace(/\d+\)$/, `${event.currentTarget.value})`);
              void submit({ kind: 'text', text });
            }
          }} />
      </label>
      <label className="grid grid-cols-[auto_1fr] items-center gap-2">{studioName('seed')}
        <TextField key={sampler.seed} type="number" min={0} max={2 ** 32 - 1} step={1}
          defaultValue={sampler.seed} readOnly={ruleReadOnly} onKeyDown={(event) => {
            if (event.key === 'Enter' && !ruleReadOnly && event.currentTarget.validity.valid) {
              void submit({ kind: 'text', text: field.text, seed: event.currentTarget.valueAsNumber });
            }
          }} />
      </label>
    </div>}
    {sweep && sweep.kind !== 'list' && sweep.can_list && !rowSelection && <Button variant="ghost"
      className="justify-self-start" disabled={readOnly}
      onClick={() => void submit({ kind: 'to_list' })}>{studioName('toList')}</Button>}
    {keep !== undefined && choices.length > 0 && <form className="grid grid-cols-[1fr_auto] gap-1" onSubmit={(event) => {
      event.preventDefault();
      void submit({ kind: 'to_value', value_id: keep });
    }}>
      <Select aria-label={studioAccessibleName('keepValue', label)} value={keep} onValueChange={setKeep}
        items={choices.map(({ id, value }) => ({ value: id, label: String(value) }))} />
      <Button type="submit" disabled={readOnly}>{studioName('keepValue')}</Button>
    </form>}
  </section>;
}
