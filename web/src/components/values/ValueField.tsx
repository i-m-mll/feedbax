import clsx from 'clsx';
import { FunctionSquare } from 'lucide-react';
import { useId, useLayoutEffect, useRef, useState, type KeyboardEvent, type ReactNode, type RefObject } from 'react';
import { createPortal } from 'react-dom';
import type { FieldKind } from '@/generated/studioContracts';
import { appTip } from '@window-shell/browser/tooltip';
import {
  studioAccessibleName, studioFieldKindLabel, studioFieldLabel, studioFieldValueLabel, studioName,
} from '@/components/activity/studioVocabulary';
import { selectionButton } from '@/features/pipeline/selection';
import { Button, IconButton } from '@/components/ui/Button';
import { Checkbox } from '@/components/ui/Checkbox';
import { FieldRow } from '@/components/ui/FieldRow';
import { Menu, MenuContent, MenuItem, MenuTrigger } from '@/components/ui/Menu';
import { usePopupLayer } from '@/components/ui/popupLayer';
import { Select } from '@/components/ui/Select';
import { popupItem, popupSurface } from '@/components/ui/styles';
import { TextField } from '@/components/ui/TextField';
import { completions, hasMark, type Completion, type FieldEdit, type FieldModel } from './fieldModel';
import './valueField.css';

export type ValueFieldForm = 'row' | 'cell';

const kindGlyphs: Record<FieldKind, ReactNode> = {
  value: '·', sweep: '◇', draw: '~', formula: <FunctionSquare aria-hidden="true" />, shape: '∿', link: '↗',
};

function useSubmit(model: FieldModel) {
  const [busy, setBusy] = useState(false);
  const [reason, setReason] = useState<string>();
  async function submit(edit: FieldEdit) {
    if (!model.commit || busy) return false;
    setBusy(true);
    const failure = await model.commit(edit);
    setBusy(false);
    setReason(failure);
    return !failure;
  }
  return { busy, reason, setReason, submit };
}

/**
 * The one field control: a value of any kind its site offers, as an inspector row or a table cell (90 px at its
 * narrowest). The control computes nothing; it sends text or an edit to the model's source and shows the reason a
 * source refuses.
 */
export function ValueField({ model, form = 'row' }: { model: FieldModel; form?: ValueFieldForm }) {
  const { label, type } = model;
  const id = useId();
  const labelId = useId();
  const reasonId = useId();
  const originId = useId();
  const { busy, reason, setReason, submit } = useSubmit(model);
  const [keep, setKeep] = useState<string>();
  const readOnly = !model.commit || busy;
  const describedBy = [model.origin && originId, (reason ?? model.reason) && reasonId].filter(Boolean).join(' ') || undefined;
  const below = <>
    {model.origin && <span id={originId} className="text-shell-muted">{model.origin}</span>}
    {(reason ?? model.reason) && <p role="alert" id={reasonId}>{reason ?? model.reason}</p>}
  </>;

  function chooseKind(kind: FieldKind, focusText: () => void) {
    if (kind === 'sweep' && model.kind !== 'sweep') void submit({ kind: 'to_sweep' });
    else if (kind === 'value' && model.kind === 'sweep') {
      const values = model.keep ?? [];
      if (values.length === 1) void submit({ kind: 'to_value', value_id: values[0].id });
      else if (values.length) setKeep(values.find(({ selected }) => selected)?.id ?? values[0].id);
    } else focusText();
  }

  const widget = (() => {
    switch (type.kind) {
      case 'boolean':
        return <Checkbox aria-labelledby={labelId} checked={model.value === true} disabled={readOnly}
          aria-describedby={describedBy} onCheckedChange={(checked) => void submit({ kind: 'set', value: checked })} />;
      case 'choice': {
        const index = type.options.findIndex(({ value }) => JSON.stringify(value) === JSON.stringify(model.value));
        return <Select aria-labelledby={labelId} aria-describedby={describedBy} className="w-full" disabled={readOnly}
          value={index < 0 ? null : String(index)} placeholder={studioName('mixed')}
          items={type.options.map((option, position) => ({ value: String(position), label: option.label }))}
          onValueChange={(position) => void submit({ kind: 'set', value: type.options[Number(position)].value })} />;
      }
      case 'vector':
        return <div className="grid min-w-0 auto-cols-[minmax(0,1fr)] grid-flow-col gap-1">
          {model.parts?.map((part) => <ValueField key={part.id} model={part} form="cell" />)}
        </div>;
      case 'group': case 'list':
        return null;
      default:
        return <ValueText model={model} id={id} readOnly={readOnly} describedBy={describedBy} invalid={!!reason}
          submit={submit} clearReason={() => setReason(undefined)} chooseKind={chooseKind}
          onEscape={() => setKeep(undefined)} />;
    }
  })();

  const textual = !['boolean', 'choice', 'vector', 'group', 'list'].includes(type.kind);
  if (form === 'cell') {
    return <div className="value-field value-field-cell grid min-w-0 gap-0.5 tabular-nums" role="group"
      aria-label={studioFieldLabel('field', label)}>
      {textual ? <label id={labelId} htmlFor={id} className="sr-only">{label}</label> : <span id={labelId} className="sr-only">{label}</span>}
      {widget}
      {(reason ?? model.reason) && <p role="alert" id={reasonId} className="text-xs">{reason ?? model.reason}</p>}
    </div>;
  }

  const indented = type.kind !== 'vector' && !!model.parts?.length;
  return <section className="value-field grid min-w-0 gap-1 tabular-nums" aria-label={studioFieldLabel('field', label)}>
    <FieldRow label={label} labelId={labelId} htmlFor={textual ? id : undefined}
      below={model.origin || reason || model.reason ? below : undefined}>
      {widget}
    </FieldRow>
    {(indented || model.members || model.canList || keep !== undefined) && <div className="value-field-parts grid gap-1">
      {indented && model.parts!.map((part) => <ValueField key={part.id} model={part} />)}
      {model.members && <Members model={model} readOnly={readOnly} submit={submit} clearReason={() => setReason(undefined)} />}
      {model.canList && <Button variant="ghost" className="justify-self-start" disabled={readOnly}
        onClick={() => void submit({ kind: 'to_list' })}>{studioName('toList')}</Button>}
      {keep !== undefined && !!model.keep?.length && <form className="grid grid-cols-[1fr_auto] gap-1" onSubmit={(event) => {
        event.preventDefault();
        void submit({ kind: 'to_value', value_id: keep }).then((done) => { if (done) setKeep(undefined); });
      }}>
        <Select aria-label={studioAccessibleName('keepValue', label)} value={keep} onValueChange={setKeep}
          items={model.keep.map(({ id: value, text }) => ({ value, label: text }))} />
        <Button type="submit" disabled={readOnly}>{studioName('keepValue')}</Button>
      </form>}
    </div>}
  </section>;
}

function ValueText({ model, id, readOnly, describedBy, invalid, submit, clearReason, chooseKind, onEscape }: {
  model: FieldModel; id: string; readOnly: boolean; describedBy?: string; invalid: boolean;
  submit(edit: FieldEdit): Promise<boolean>; clearReason(): void;
  chooseKind(kind: FieldKind, focusText: () => void): void; onEscape(): void;
}) {
  const { label, kind } = model;
  const input = useRef<HTMLInputElement>(null);
  const [draft, setDraft] = useState<string>();
  const [menu, setMenu] = useState(false);
  const [caret, setCaret] = useState<number>();
  const [highlight, setHighlight] = useState(0);
  const [dismissed, setDismissed] = useState(false);
  const listId = useId();
  const text = draft ?? model.text ?? '';
  const offered = draft !== undefined && !dismissed && caret !== undefined
    ? completions(model, text.slice(0, caret)) : { token: '', items: [] };
  const open = offered.items.length > 0;
  const mark = hasMark(model);
  const focusText = () => { input.current?.focus(); input.current?.select(); };

  function accept(item: Completion) {
    const start = (caret ?? text.length) - offered.token.length;
    const next = text.slice(0, start) + item.insert + text.slice(caret ?? text.length);
    const position = start + item.insert.length;
    setDraft(next);
    setCaret(position);
    setHighlight(0);
    requestAnimationFrame(() => input.current?.setSelectionRange(position, position));
  }

  function onKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (open && (event.key === 'ArrowDown' || event.key === 'ArrowUp')) {
      event.preventDefault();
      const step = event.key === 'ArrowDown' ? 1 : -1;
      setHighlight((highlight + step + offered.items.length) % offered.items.length);
      return;
    }
    if (open && event.key === 'Tab' && !event.shiftKey) {
      event.preventDefault();
      accept(offered.items[Math.min(highlight, offered.items.length - 1)]);
      return;
    }
    if (mark && (event.key === 'ContextMenu' || event.key === 'F10' && event.shiftKey)) {
      event.preventDefault();
      setMenu(true);
      return;
    }
    if (event.key === 'Escape') {
      event.stopPropagation();
      if (open) { setDismissed(true); return; }
      setDraft(undefined);
      clearReason();
      setMenu(false);
      onEscape();
    }
    if (event.key === 'Enter' && draft !== undefined) {
      event.preventDefault();
      void submit({ kind: 'text', text: draft }).then((done) => { if (done) setDraft(undefined); });
    }
  }

  const markMenu = mark ? <Menu open={menu} onOpenChange={setMenu}>
    <MenuTrigger render={<IconButton size="mark" label={studioFieldKindLabel(label, kind)}
      className={kind === 'value' ? 'value-field-constant-mark' : undefined}
      icon={<span aria-hidden="true" className="flex">{kindGlyphs[kind]}</span>} />} />
    <MenuContent aria-label={studioFieldLabel('kinds', label)} finalFocus={input}>
      {model.kinds.map((offer) => <MenuItem key={offer.kind} disabled={readOnly || !offer.enabled}
        onClick={() => { setMenu(false); chooseKind(offer.kind, focusText); }}>{studioName(offer.kind)}</MenuItem>)}
    </MenuContent>
  </Menu> : undefined;
  const differs = model.differs ? <output {...appTip(studioName('selectedValue'))}
    aria-label={studioFieldLabel('selectedValue', label)}>◆</output> : undefined;
  return <>
    <TextField ref={input} id={id} type="text" value={text} leading={markMenu} trailing={differs} className="min-w-0"
      placeholder={model.text === null && draft === undefined ? studioName('mixed') : undefined}
      readOnly={readOnly || model.type.kind === 'readOnly'} invalid={invalid}
      aria-describedby={describedBy}
      role={open ? 'combobox' : undefined} aria-expanded={open || undefined}
      aria-controls={open ? listId : undefined} aria-autocomplete={model.names.length || model.functions.length ? 'list' : undefined}
      aria-activedescendant={open ? `${listId}-${highlight}` : undefined}
      onValueChange={(value) => { setDraft(value); setDismissed(false); setHighlight(0); clearReason(); }}
      onSelect={(event) => setCaret(event.currentTarget.selectionStart ?? undefined)}
      onBlur={() => setDismissed(true)}
      onKeyDown={onKeyDown} />
    {open && <CompletionList id={listId} anchor={input} items={offered.items} highlight={highlight} onChoose={accept} />}
  </>;
}

function CompletionList({ id, anchor, items, highlight, onChoose }: {
  id: string; anchor: RefObject<HTMLInputElement>; items: readonly Completion[]; highlight: number;
  onChoose(item: Completion): void;
}) {
  const layer = usePopupLayer() ?? document.body;
  const [box, setBox] = useState<DOMRect>();
  useLayoutEffect(() => {
    const frame = anchor.current?.parentElement;
    if (frame) setBox(frame.getBoundingClientRect());
  }, [anchor, items]);
  if (!box) return null;
  return createPortal(<div id={id} role="listbox" aria-label={studioName('completions')}
    className={clsx(popupSurface, 'fixed z-50 max-h-64 overflow-y-auto')}
    style={{ left: box.left, top: box.bottom + 4, minWidth: box.width }}>
    {items.map((item, index) => <div key={item.id} id={`${id}-${index}`} role="option" aria-selected={index === highlight}
      data-highlighted={index === highlight ? '' : undefined} className={clsx(popupItem, 'justify-between')}
      onMouseDown={(event) => { event.preventDefault(); onChoose(item); }}>
      <span className="truncate">{item.label}</span>
      <span className="truncate pl-4 text-shell-muted">{item.detail}</span>
    </div>)}
  </div>, layer);
}

function Members({ model, readOnly, submit, clearReason }: {
  model: FieldModel; readOnly: boolean; submit(edit: FieldEdit): Promise<boolean>; clearReason(): void;
}) {
  const { label } = model;
  const members = model.members!;
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [adding, setAdding] = useState(false);
  const [newValue, setNewValue] = useState('');
  const settle = (done: boolean) => { if (done) { setDrafts({}); setAdding(false); setNewValue(''); } };
  return <>
    {members.values.map(({ id, label: name, text, selected }, index) =>
      <div key={id} className="grid grid-cols-[auto_minmax(0,1fr)_minmax(0,1fr)] items-center gap-1">
        {members.choose ? <IconButton pressed={selected}
          label={studioAccessibleName('select', `${label} ${name ?? text}`)}
          {...selectionButton((extend) => members.choose!(id, extend))}
          icon={selected ? '◆' : '·'} /> : <span aria-hidden="true" className={selected ? 'font-bold' : 'text-shell-muted'}>{selected ? '◆' : '·'}</span>}
        <TextField type="text" aria-label={name ? studioAccessibleName('value', name) : studioFieldValueLabel(label, index + 1)}
          value={drafts[id] ?? text} readOnly={readOnly || !members.editable}
          onValueChange={(next) => { setDrafts({ ...drafts, [id]: next }); clearReason(); }}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && drafts[id] !== undefined) {
              event.preventDefault();
              void submit({ kind: 'text', text: drafts[id], value_id: id }).then(settle);
            }
            if (event.key === 'Escape') { setDrafts({}); clearReason(); }
          }} />
        {members.editable && <TextField type="text" readOnly={readOnly}
          aria-label={studioFieldValueLabel(label, index + 1, true)} placeholder={studioName('name')} key={name ?? ''}
          defaultValue={name ?? ''} onKeyDown={(event) => {
            if (event.key === 'Enter') {
              event.preventDefault();
              void submit({ kind: 'name_value', value_id: id, label: event.currentTarget.value });
            }
            if (event.key === 'Escape') event.currentTarget.value = name ?? '';
          }} />}
      </div>)}
    {adding && <form className="grid grid-cols-[1fr_auto] gap-1" onSubmit={(event) => {
      event.preventDefault();
      void submit({ kind: 'text', text: `${(model.text ?? '').replace(/,$/, '')}, ${newValue}` }).then(settle);
    }}>
      <TextField aria-label={studioAccessibleName('newValue', label)} autoFocus value={newValue} onValueChange={setNewValue} />
      <Button type="submit" disabled={readOnly || !newValue.trim()}>{studioName('add')}</Button>
    </form>}
    {!adding && members.extendable && <Button variant="ghost" className="justify-self-start"
      disabled={readOnly} onClick={() => setAdding(true)}>{studioName('addValue')}</Button>}
  </>;
}
