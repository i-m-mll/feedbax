import { useEffect, useId, useRef, useState } from 'react';
import type { ModelFieldRequest, StudioFieldView } from '@/generated/studioContracts';
import { selectionButton } from '@/features/pipeline/selection';

type Edit = NonNullable<ModelFieldRequest['edit']>;
type Commit = (edit: Edit) => Promise<string | undefined>;
const inputClass = 'min-w-0 rounded border border-shell-divider bg-shell-chrome px-1 py-0.5 tabular-nums';
const buttonClass = 'rounded px-1.5 py-0.5 hover:bg-shell-hover disabled:opacity-50';
const kindLabel = (kind: string) => kind[0].toUpperCase() + kind.slice(1);

export function StudioFieldControl({ field, commit, selectedValues, rowSelection = false, onChooseValue }: {
  field: StudioFieldView;
  commit?: Commit;
  selectedValues?: ReadonlySet<string>;
  rowSelection?: boolean;
  onChooseValue?: (id: string, extend: boolean) => void;
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
  const kind = sweep ? 'Sweep' : 'Value';
  const list = sweep?.kind === 'list' ? sweep : undefined;
  const choices = list?.values ?? field.members ?? [];
  const range = sweep?.kind === 'range' ? sweep.rule : undefined;
  const sampler = sweep?.kind === 'sampler' ? sweep.rule : undefined;
  const authored = list ? list.values.map(({ value }) => value).join(', ') + (list.values.length === 1 ? ',' : '')
    : field.text;
  const picked = selectedValues ?? new Set<string>();

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

  function choose(kind: string) {
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
    void submit({ kind: 'text', text: `${next.scale === 'log' ? 'geomspace' : 'linspace'}(${next.start}, ${next.stop}, ${next.count})` });
  }

  return <section className="grid min-w-0 gap-1" aria-label={`${label} field`}>
    <label htmlFor={fieldId}>{label}</label>
    <div className="relative grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-1">
      <button type="button" className={`${buttonClass} ${sweep ? '' : 'text-shell-muted'}`}
        aria-label={`${label} kind: ${kind}`} aria-haspopup="menu" aria-expanded={menu}
        onClick={() => setMenu(!menu)}>{sweep ? '◇' : '·'}</button>
      {menu && <div ref={menuRef} role="menu" aria-label={`${label} kinds`}
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
          disabled={readOnly || rowSelection && value !== kind.toLowerCase() ||
            value === 'sweep' && !sweep && !field.can_sweep ||
            value === 'value' && !!sweep && (!choices.length || !field.can_change_axis)}
          onClick={() => choose(value)}>{kindLabel(value)}</button>)}
      </div>}
      <input ref={textInput} id={fieldId} type="text" className={inputClass}
        value={draft ?? (rowSelection && !sweep ? field.effective ?? authored : authored)}
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
      {field.effective != null && <output className="text-shell-muted" title={`Selected value ${field.effective}`}
        aria-label={`${label} selected value ${field.effective}`}>◆</output>}
    </div>
    {reason && <p role="alert" id={reasonId} className="text-xs">{reason}</p>}
    {list && <div className="grid gap-1 border-l border-shell-divider pl-2">
      {list.values.map(({ id, label: memberLabel, value }, index) => <div key={id} className="grid grid-cols-[auto_minmax(0,1fr)_minmax(0,1fr)] items-center gap-1">
        {onChooseValue ? <button type="button" className={buttonClass} aria-pressed={picked.has(id)}
          aria-label={`Select ${label} ${memberLabel ?? value}`}
          {...selectionButton((extend) => onChooseValue(id, extend))}>
          {picked.has(id) ? '◆' : '·'}
        </button> : <span aria-hidden="true" className={picked.has(id) ? 'font-bold' : 'text-shell-muted'}>{picked.has(id) ? '◆' : '·'}</span>}
        <input type="text" className={inputClass} aria-label={memberLabel ?? `${label} value ${index + 1}`}
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
          aria-label={`${label} value ${index + 1} name`} placeholder="Name" key={memberLabel ?? ''}
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
        <input aria-label={`New ${label} value`} autoFocus className={inputClass} value={newValue}
          onChange={(event) => setNewValue(event.currentTarget.value)} />
        <button type="submit" className={buttonClass} disabled={readOnly || !newValue.trim()}>Add</button>
      </form>}
      {!adding && field.can_change_axis && !rowSelection && <button type="button" className={`${buttonClass} justify-self-start`}
        disabled={readOnly} onClick={() => setAdding(true)}>Add Value</button>}
    </div>}
    {range && <div className="grid gap-1 border-l border-shell-divider pl-2">
      {([['start', 'First'], ['stop', 'Last'], ['count', 'Count']] as const).map(([key, name]) =>
        <label key={key} className="grid grid-cols-[auto_1fr] gap-2">{name}
          <input type="text" className={inputClass} key={range[key]} defaultValue={range[key]} readOnly={readOnly}
            onKeyDown={(event) => {
              if (event.key === 'Enter') { event.preventDefault(); updateRange(key, event.currentTarget.value); }
              if (event.key === 'Escape') event.currentTarget.value = String(range[key]);
            }} />
        </label>)}
      <label className="grid grid-cols-[auto_1fr] gap-2">Spacing
        <select className={inputClass} value={range.scale ?? 'linear'} disabled={readOnly}
          onChange={(event) => updateRange('scale', event.currentTarget.value)}>
          <option value="linear">Linear</option><option value="log">Logarithmic</option>
        </select>
      </label>
    </div>}
    {sampler && typeof sampler.distribution !== 'string' && <div className="grid gap-1 border-l border-shell-divider pl-2">
      <label className="grid grid-cols-[auto_1fr] gap-2">Count
        <input className={inputClass} key={sampler.count} defaultValue={sampler.count} readOnly={readOnly}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              const text = field.text.replace(/\d+\)$/, `${event.currentTarget.value})`);
              void submit({ kind: 'text', text });
            }
          }} />
      </label>
      <label className="grid grid-cols-[auto_1fr] gap-2">Seed
        <input className={inputClass} key={sampler.seed} type="number" min={0} max={2 ** 32 - 1} step={1}
          defaultValue={sampler.seed} readOnly={readOnly} onKeyDown={(event) => {
            if (event.key === 'Enter' && event.currentTarget.validity.valid) {
              void submit({ kind: 'text', text: field.text, seed: event.currentTarget.valueAsNumber });
            }
          }} />
      </label>
    </div>}
    {sweep && sweep.kind !== 'list' && sweep.can_list && !rowSelection && <button type="button"
      className={`${buttonClass} justify-self-start`} disabled={readOnly}
      onClick={() => void submit({ kind: 'to_list' })}>To List</button>}
    {keep !== undefined && choices.length > 0 && <form className="grid grid-cols-[1fr_auto] gap-1" onSubmit={(event) => {
      event.preventDefault();
      void submit({ kind: 'to_value', value_id: keep });
    }}>
      <select aria-label={`Keep ${label} value`} className={inputClass} value={keep}
        onChange={(event) => setKeep(event.currentTarget.value)}>
        {choices.map(({ id, value }) => <option key={id} value={id}>{value}</option>)}
      </select>
      <button className={buttonClass} type="submit" disabled={readOnly}>Keep Value</button>
    </form>}
  </section>;
}
