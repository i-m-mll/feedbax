import { useEffect, useId, useRef, useState } from 'react';
import { FunctionSquare } from 'lucide-react';
import type { ModelFieldRequest, StudioFieldView } from '@/generated/studioContracts';
import { appTip } from '@window-shell/browser/tooltip';
import { studioAccessibleName, studioFieldKindLabel, studioFieldLabel, studioFieldValueLabel, studioName } from '@/components/activity/studioVocabulary';
import { selectionButton } from '@/features/pipeline/selection';

type Edit = NonNullable<ModelFieldRequest['edit']>;
type Commit = (edit: Edit) => Promise<string | undefined>;
const inputClass = 'min-w-0 rounded border border-shell-divider bg-shell-chrome px-1 py-0.5 tabular-nums';
const buttonClass = 'rounded px-1.5 py-0.5 hover:bg-shell-hover disabled:opacity-50';
type FieldKind = StudioFieldView['site']['kinds'][number];

export function StudioFieldControl({ field, commit, selectedValues, rowSelection = false, multipleRows = false, onChooseValue, valueKind = 'value' }: {
  field: StudioFieldView;
  commit?: Commit;
  selectedValues?: ReadonlySet<string>;
  rowSelection?: boolean;
  multipleRows?: boolean;
  onChooseValue?: (id: string, extend: boolean) => void;
  valueKind?: StudioFieldView['site']['kinds'][number];
}) {
  const { label, sweep } = field;
  const fieldId = useId();
  const reasonId = useId();
  const textInput = useRef<HTMLInputElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
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

  useEffect(() => {
    if (menu) menuRef.current?.querySelector<HTMLButtonElement>('button:not(:disabled)')?.focus();
  }, [menu]);

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

  return <section className="grid min-w-0 gap-1" aria-label={studioFieldLabel('field', label)}>
    <label htmlFor={fieldId}>{label}</label>
    <div className="relative grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-1">
      <button type="button" className={`${buttonClass} ${sweep ? '' : 'text-shell-muted'}`}
        aria-label={studioFieldKindLabel(label, kind)} aria-haspopup="menu" aria-expanded={menu}
        onClick={() => setMenu(!menu)}>{valueKind === 'formula' && !sweep
          ? <FunctionSquare aria-hidden="true" size={12} />
          : <span aria-hidden="true">{sweep ? '◇' : valueKind === 'link' ? '↗' : '·'}</span>}</button>
      {menu && <div ref={menuRef} role="menu" aria-label={studioFieldLabel('kinds', label)}
        className="absolute left-0 top-full z-10 grid rounded border border-shell-divider bg-shell-chrome p-1 shadow-md"
        onKeyDown={(event) => {
          if (event.key === 'Escape') { event.stopPropagation(); setMenu(false); textInput.current?.focus(); }
          if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
            event.preventDefault();
            const items = [...event.currentTarget.querySelectorAll<HTMLButtonElement>('button:not(:disabled)')];
            const index = items.indexOf(document.activeElement as HTMLButtonElement);
            const next = event.key === 'ArrowDown' ? index + 1 : index < 0 ? items.length - 1 : index - 1;
            items[(next + items.length) % items.length]?.focus();
          }
        }}>
        {field.site.kinds.map((value) => <button key={value} type="button" role="menuitem"
          className={buttonClass}
          disabled={readOnly || rowSelection && value !== kind ||
            value === 'sweep' && !sweep && !field.can_sweep ||
            value === 'value' && !!sweep && (!choices.length || !field.can_change_axis)}
          onClick={() => choose(value)}>{studioName(value)}</button>)}
      </div>}
      <input ref={textInput} id={fieldId} type="text" className={inputClass}
        value={draft ?? (rowSelection && !sweep ? multipleRows ? '' : effective ?? authored : authored)}
        placeholder={multipleRows && !sweep ? studioName('selectedRuns') : undefined}
        readOnly={readOnly} aria-invalid={!!reason || undefined} aria-describedby={reason ? reasonId : undefined}
        onChange={(event) => { setDraft(event.currentTarget.value); setReason(undefined); }}
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
        {onChooseValue ? <button type="button" className={buttonClass} aria-pressed={picked.has(id)}
          aria-label={studioAccessibleName('select', `${label} ${memberLabel ?? value}`)}
          {...selectionButton((extend) => onChooseValue(id, extend))}>
          {picked.has(id) ? '◆' : '·'}
        </button> : <span aria-hidden="true" className={picked.has(id) ? 'font-bold' : 'text-shell-muted'}>{picked.has(id) ? '◆' : '·'}</span>}
        <input type="text" className={inputClass} aria-label={memberLabel ? studioAccessibleName('value', memberLabel) : studioFieldValueLabel(label, index + 1)}
          value={members[id] ?? String(value)} readOnly={readOnly || !list.editable}
          onChange={(event) => { setMembers({ ...members, [id]: event.currentTarget.value }); setReason(undefined); }}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && members[id] !== undefined) {
              event.preventDefault();
              void submit({ kind: 'text', text: members[id], value_id: id });
            }
            if (event.key === 'Escape') { setMembers({}); setReason(undefined); }
          }} />
        {list.editable && <input className={inputClass} type="text" readOnly={readOnly}
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
        <input aria-label={studioAccessibleName('newValue', label)} autoFocus className={inputClass} value={newValue}
          onChange={(event) => setNewValue(event.currentTarget.value)} />
        <button type="submit" className={buttonClass} disabled={readOnly || !newValue.trim()}>{studioName('add')}</button>
      </form>}
      {!adding && field.can_change_axis && !rowSelection && <button type="button" className={`${buttonClass} justify-self-start`}
        disabled={readOnly} onClick={() => setAdding(true)}>{studioName('addValue')}</button>}
    </div>}
    {range && <div className="grid gap-1 border-l border-shell-divider pl-2">
      {([['start', 'first'], ['stop', 'last'], ['count', 'count']] as const).map(([key, name]) =>
        <label key={key} className="grid grid-cols-[auto_1fr] gap-2">{studioName(name)}
          <input type="text" className={inputClass} key={range[key]} defaultValue={range[key]} readOnly={ruleReadOnly}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !ruleReadOnly) { event.preventDefault(); updateRange(key, event.currentTarget.value); }
              if (event.key === 'Escape') event.currentTarget.value = String(range[key]);
            }} />
        </label>)}
      <label className="grid grid-cols-[auto_1fr] gap-2">{studioName('spacing')}
        <select className={inputClass} value={range.scale ?? 'linear'} disabled={ruleReadOnly}
          onChange={(event) => updateRange('scale', event.currentTarget.value)}>
          <option value="linear">{studioName('linear')}</option><option value="log">{studioName('logarithmic')}</option>
        </select>
      </label>
    </div>}
    {sampler && typeof sampler.distribution !== 'string' && <div className="grid gap-1 border-l border-shell-divider pl-2">
      <label className="grid grid-cols-[auto_1fr] gap-2">{studioName('count')}
        <input className={inputClass} key={sampler.count} defaultValue={sampler.count} readOnly={ruleReadOnly}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !ruleReadOnly) {
              const text = field.text.replace(/\d+\)$/, `${event.currentTarget.value})`);
              void submit({ kind: 'text', text });
            }
          }} />
      </label>
      <label className="grid grid-cols-[auto_1fr] gap-2">{studioName('seed')}
        <input className={inputClass} key={sampler.seed} type="number" min={0} max={2 ** 32 - 1} step={1}
          defaultValue={sampler.seed} readOnly={ruleReadOnly} onKeyDown={(event) => {
            if (event.key === 'Enter' && !ruleReadOnly && event.currentTarget.validity.valid) {
              void submit({ kind: 'text', text: field.text, seed: event.currentTarget.valueAsNumber });
            }
          }} />
      </label>
    </div>}
    {sweep && sweep.kind !== 'list' && sweep.can_list && !rowSelection && <button type="button"
      className={`${buttonClass} justify-self-start`} disabled={readOnly}
      onClick={() => void submit({ kind: 'to_list' })}>{studioName('toList')}</button>}
    {keep !== undefined && choices.length > 0 && <form className="grid grid-cols-[1fr_auto] gap-1" onSubmit={(event) => {
      event.preventDefault();
      void submit({ kind: 'to_value', value_id: keep });
    }}>
      <select aria-label={studioAccessibleName('keepValue', label)} className={inputClass} value={keep}
        onChange={(event) => setKeep(event.currentTarget.value)}>
        {choices.map(({ id, value }) => <option key={id} value={id}>{value}</option>)}
      </select>
      <button className={buttonClass} type="submit" disabled={readOnly}>{studioName('keepValue')}</button>
    </form>}
  </section>;
}
