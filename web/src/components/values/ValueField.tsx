import clsx from 'clsx';
import { Dices, Link, Sigma, Spline } from 'lucide-react';
import { useEffect, useId, useLayoutEffect, useRef, useState, type KeyboardEvent, type ReactNode, type RefObject } from 'react';
import { createPortal } from 'react-dom';
import type { FieldKind } from '@/generated/studioContracts';
import { appTip } from '@window-shell/browser/tooltip';
import {
  studioAccessibleName, studioFieldKindLabel, studioFieldLabel, studioName, studioParameterLabel,
} from '@/components/activity/studioVocabulary';
import { selectionButton } from '@/features/pipeline/selection';
import { Button, IconButton } from '@/components/ui/Button';
import { Checkbox } from '@/components/ui/Checkbox';
import { Combobox } from '@/components/ui/Combobox';
import { FieldRow } from '@/components/ui/FieldRow';
import { Menu, MenuContent, MenuItem, MenuTrigger } from '@/components/ui/Menu';
import { usePopupLayer } from '@/components/ui/popupLayer';
import { Select } from '@/components/ui/Select';
import { Section } from '@/components/ui/Section';
import { popupItem, popupSurface, primaryText, secondaryText } from '@/components/ui/styles';
import { TextField } from '@/components/ui/TextField';
import { completions, hasMark, type Completion, type FieldEdit, type FieldModel, type FieldOption } from './fieldModel';
import { useFieldGroup } from './fieldGroups';
import './valueField.css';

export type ValueFieldForm = 'row' | 'cell';

function Outline({ children }: { children: ReactNode }) {
  return <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}
    strokeLinecap="round" strokeLinejoin="round">{children}</svg>;
}

const kindGlyphs: Record<FieldKind, ReactNode> = {
  value: <Outline><circle cx="12" cy="12" r="2.5" /></Outline>,
  sweep: <Outline><circle cx="3.5" cy="12" r="2.5" /><circle cx="12" cy="12" r="2.5" /><circle cx="20.5" cy="12" r="2.5" /></Outline>,
  draw: <Dices aria-hidden="true" />,
  formula: <Sigma aria-hidden="true" />,
  shape: <Spline aria-hidden="true" />,
  link: <Link aria-hidden="true" />,
};

/**
 * Sends edits to the model's source. An edit made while the source is briefly unavailable (a save in flight, a
 * projection being replaced) waits for it rather than being dropped.
 */
function useSubmit(model: FieldModel) {
  const [busy, setBusy] = useState(false);
  const [reason, setReason] = useState<string>();
  const waiting = useRef<{ edit: FieldEdit; resolve(done: boolean): void }>();
  async function send(commit: NonNullable<FieldModel['commit']>, edit: FieldEdit) {
    setBusy(true);
    const failure = await commit(edit);
    setBusy(false);
    setReason(failure);
    return !failure;
  }
  function submit(edit: FieldEdit) {
    if (busy) return Promise.resolve(false);
    if (!model.commit) return new Promise<boolean>((resolve) => { waiting.current = { edit, resolve }; });
    return send(model.commit, edit);
  }
  useEffect(() => {
    const next = waiting.current;
    if (!model.commit || !next) return;
    waiting.current = undefined;
    void send(model.commit, next.edit).then(next.resolve);
  });
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
  const readOnly = !model.commit || busy;
  const describedBy = [model.origin && originId, (reason ?? model.reason) && reasonId].filter(Boolean).join(' ') || undefined;
  const below = <>
    {model.origin && <span id={originId} className="text-shell-muted">{model.origin}</span>}
    {(reason ?? model.reason) && <p role="alert" id={reasonId}>{reason ?? model.reason}</p>}
  </>;

  function chooseKind(kind: FieldKind, focusText: () => void) {
    if (kind === 'sweep' && model.kind !== 'sweep') void submit({ kind: 'to_sweep' });
    else if (kind === 'value' && model.kind === 'sweep' && model.keeps) void submit({ kind: 'to_value', value_id: model.keeps });
    else focusText();
  }

  const parameter = type.kind === 'class' && model.parts?.length === 1 ? model.parts[0] : undefined;
  const boxed = !!parameter && textualKind(parameter.type.kind);
  const widget = (() => {
    switch (type.kind) {
      case 'boolean':
        return <Checkbox aria-labelledby={labelId} checked={model.value === true} disabled={readOnly}
          aria-describedby={describedBy} onCheckedChange={(checked) => void submit({ kind: 'set', value: checked })} />;
      case 'choice': {
        const index = optionIndex(type.options, model.value);
        return <Select aria-labelledby={labelId} aria-describedby={describedBy} className="w-full" disabled={readOnly}
          value={index < 0 ? null : String(index)} placeholder={studioName('mixed')}
          items={type.options.map((option, position) => ({ value: String(position), label: option.label }))}
          onValueChange={(position) => void submit({ kind: 'set', value: type.options[Number(position)].value })} />;
      }
      case 'class': {
        const index = optionIndex(type.options, model.value);
        return <Combobox aria-labelledby={labelId} className={boxed ? undefined : 'w-full'} inline={boxed} disabled={readOnly}
          value={index < 0 ? null : String(index)}
          items={type.options.map((option, position) => ({ value: String(position), label: option.label }))}
          onValueChange={(position) => {
            if (position !== null && Number(position) !== index) void submit({ kind: 'set', value: type.options[Number(position)].value });
          }} />;
      }
      case 'vector':
        return <div className="grid min-w-0 auto-cols-[minmax(0,1fr)] grid-flow-col gap-1">
          {model.parts?.map((part) => <ValueField key={part.id} model={part} form="cell" />)}
        </div>;
      case 'group': case 'list':
        return null;
      default:
        return <ValueText model={model} id={id} readOnly={readOnly} describedBy={describedBy} invalid={!!reason}
          submit={submit} clearReason={() => setReason(undefined)} chooseKind={chooseKind} />;
    }
  })();

  const textual = textualKind(type.kind);
  if (form === 'cell') {
    return <div className="value-field value-field-cell grid min-w-0 gap-0.5 tabular-nums" role="group"
      aria-label={studioFieldLabel('field', label)}>
      {textual ? <label id={labelId} htmlFor={id} className="sr-only">{label}</label> : <span id={labelId} className="sr-only">{label}</span>}
      {widget}
      {(reason ?? model.reason) && <p role="alert" id={reasonId} className="text-xs">{reason ?? model.reason}</p>}
    </div>;
  }

  const parts = (indented: boolean) => (indented || model.members || model.canList) && <div className="value-field-parts grid gap-1">
    {indented && model.parts!.map((part) => <ValueField key={part.id} model={part} />)}
    {model.members && <Members model={model} />}
    {model.canList && <Button variant="ghost" className="justify-self-start" disabled={readOnly}
      onClick={() => void submit({ kind: 'to_list' })}>{studioName('toList')}</Button>}
  </div>;
  if (boxed) return <ClassValue model={model} parameter={parameter!} labelId={labelId} picker={widget} />;
  if (type.kind === 'class' && !model.parts?.length) {
    return <section className="value-field grid min-w-0 gap-1" aria-label={studioFieldLabel('field', label)}>
      <FieldRow label={label} labelId={labelId} below={model.origin || reason || model.reason ? below : undefined}>
        {widget}
      </FieldRow>
    </section>;
  }
  if (type.kind === 'group' || type.kind === 'class' || type.kind === 'list') {
    return <FieldGroup model={model} labelId={labelId} picker={widget}>
      {(model.origin || reason || model.reason) && <div className="field-row-below">{below}</div>}
      {parts(!!model.parts?.length)}
    </FieldGroup>;
  }
  return <section className="value-field grid min-w-0 tabular-nums" aria-label={studioFieldLabel('field', label)}>
    <FieldRow label={label} labelId={labelId} htmlFor={textual ? id : undefined}
      below={model.origin || reason || model.reason ? below : undefined}>
      {widget}
    </FieldRow>
    {parts(type.kind !== 'vector' && !!model.parts?.length)}
  </section>;
}

const optionIndex = (options: readonly FieldOption[], value: unknown) =>
  options.findIndex((option) => JSON.stringify(option.value) === JSON.stringify(value));

const textualKind = (kind: FieldModel['type']['kind']) =>
  !['boolean', 'choice', 'class', 'vector', 'group', 'list'].includes(kind);

/** A field whose value is a group of fields: a folding header in the label style; a class's picker under it, kept when folded. */
function FieldGroup({ model, labelId, picker, children }: {
  model: FieldModel; labelId: string; picker: ReactNode; children: ReactNode;
}) {
  return <Section level="field" name={model.label} nameId={labelId} control={picker} bodyClassName="grid gap-1"
    className="value-field tabular-nums" {...useFieldGroup(model.id)}>
    {children}
  </Section>;
}

/**
 * A class with one textual parameter: the parameter is the field's value, in one box with the class's picker at its
 * end. The parameter's name is not drawn; it names the input, after the field's name.
 */
function ClassValue({ model, parameter, labelId, picker }: {
  model: FieldModel; parameter: FieldModel; labelId: string; picker: ReactNode;
}) {
  const id = useId();
  const reasonId = useId();
  const originId = useId();
  const { busy, reason, setReason, submit } = useSubmit(parameter);
  const readOnly = !parameter.commit || busy;
  const shown = reason ?? parameter.reason ?? model.reason;
  const origin = parameter.origin ?? model.origin;
  const describedBy = [origin && originId, shown && reasonId].filter(Boolean).join(' ') || undefined;
  function chooseKind(kind: FieldKind, focusText: () => void) {
    if (kind === 'sweep' && parameter.kind !== 'sweep') void submit({ kind: 'to_sweep' });
    else if (kind === 'value' && parameter.kind === 'sweep' && parameter.keeps) void submit({ kind: 'to_value', value_id: parameter.keeps });
    else focusText();
  }
  return <section className="value-field value-field-class grid min-w-0 tabular-nums" aria-label={studioFieldLabel('field', model.label)}>
    <FieldRow label={model.label} labelId={labelId} below={origin || shown ? <>
      {origin && <span id={originId} className="text-shell-muted">{origin}</span>}
      {shown && <p role="alert" id={reasonId}>{shown}</p>}
    </> : undefined}>
      <ValueText model={parameter} id={id} readOnly={readOnly} describedBy={describedBy} invalid={!!reason}
        submit={submit} clearReason={() => setReason(undefined)} chooseKind={chooseKind}
        end={picker} parameterOf={[model.label, parameter.parameter!]} />
    </FieldRow>
    {(parameter.members || parameter.canList) && <div className="value-field-parts grid gap-1">
      {parameter.members && <Members model={parameter} />}
      {parameter.canList && <Button variant="ghost" className="justify-self-start" disabled={readOnly}
        onClick={() => void submit({ kind: 'to_list' })}>{studioName('toList')}</Button>}
    </div>}
  </section>;
}

function ValueText({ model, id, readOnly, describedBy, invalid, submit, clearReason, chooseKind, end, parameterOf }: {
  model: FieldModel; id: string; readOnly: boolean; describedBy?: string; invalid: boolean;
  submit(edit: FieldEdit): Promise<boolean>; clearReason(): void;
  chooseKind(kind: FieldKind, focusText: () => void): void;
  end?: ReactNode;
  /** The field and parameter whose names make the input's accessible name, for a chosen class's only parameter. */
  parameterOf?: readonly [field: string, parameter: string];
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
    }
    if (event.key === 'Enter' && draft !== undefined) {
      event.preventDefault();
      void submit({ kind: 'text', text: draft }).then((done) => { if (done) setDraft(undefined); });
    }
  }

  const markMenu = mark ? <Menu open={menu} onOpenChange={setMenu}>
    <MenuTrigger render={<IconButton size="content" label={studioFieldKindLabel(label, kind)}
      className="flex size-full items-center rounded-control data-[popup-open]:bg-shell-hover"
      icon={kindGlyphs[kind]} />} />
    <MenuContent aria-label={studioFieldLabel('kinds', label)} finalFocus={input}>
      {model.kinds.map((offer) => <MenuItem key={offer.kind} disabled={readOnly || !offer.enabled}
        glyph={<span className="flex text-shell-muted">{kindGlyphs[offer.kind]}</span>}
        onClick={() => { setMenu(false); chooseKind(offer.kind, focusText); }}>{studioName(offer.kind)}</MenuItem>)}
    </MenuContent>
  </Menu> : undefined;
  const differs = model.differs !== undefined ? <output {...appTip(studioName('selectedValue'))} aria-description={model.differs}
    aria-label={studioFieldLabel('selectedValue', label)}>◆</output> : undefined;
  return <>
    <TextField ref={input} id={id} type="text" value={text} leading={markMenu} trailing={differs} end={end} className="min-w-0"
      aria-label={parameterOf ? studioParameterLabel(...parameterOf) : undefined} size={end ? Math.max(6, text.length + 1) : undefined}
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
      data-highlighted={index === highlight ? '' : undefined} className={popupItem}
      onMouseDown={(event) => { event.preventDefault(); onChoose(item); }}>
      <span className={primaryText}>{item.label}</span>
      <span className={secondaryText}>{item.detail}</span>
    </div>)}
  </div>, layer);
}

function Members({ model }: { model: FieldModel }) {
  const { label } = model;
  const { values, choose } = model.members!;
  return <div className="flex flex-wrap gap-1">
    {values.map(({ id, text, selected }) => choose
      ? <Button key={id} variant="ghost" size="mark" pressed={selected}
        aria-label={studioAccessibleName('select', `${label} ${text}`)}
        {...selectionButton((extend) => choose(id, extend))}>{text}</Button>
      : <span key={id} className={clsx('px-1 text-xs', selected ? 'font-medium' : 'text-shell-muted')}>{text}</span>)}
  </div>;
}
