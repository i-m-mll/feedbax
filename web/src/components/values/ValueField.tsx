import clsx from 'clsx';
import { ArrowRightToLine, Dices, Hourglass, Link, Sigma, Spline } from 'lucide-react';
import { useContext, useEffect, useId, useLayoutEffect, useRef, useState, type KeyboardEvent, type ReactNode, type RefObject } from 'react';
import { createPortal } from 'react-dom';
import { appTip } from '@window-shell/browser/tooltip';
import {
  studioAccessibleName, studioEvaluateValueSource, studioFieldKindLabel, studioFieldLabel, studioFormat, studioName,
  studioParameterLabel,
} from '@/components/activity/studioVocabulary';
import { selectionButton } from '@/features/pipeline/selection';
import { Button, IconButton } from '@/components/ui/Button';
import { ChangeMark } from '@/components/ui/ChangeMark';
import { Checkbox } from '@/components/ui/Checkbox';
import { Combobox } from '@/components/ui/Combobox';
import { FieldProblem, FieldProblemReason } from '@/components/ui/FieldProblem';
import { FieldRow } from '@/components/ui/FieldRow';
import { Menu, MenuContent, MenuRadioGroup, MenuRadioItem, MenuTrigger } from '@/components/ui/Menu';
import { usePopupLayer } from '@/components/ui/popupLayer';
import { Select } from '@/components/ui/Select';
import { Section } from '@/components/ui/Section';
import { popupItem, popupSurface, primaryText, secondaryText } from '@/components/ui/styles';
import { TextField, type TextTerm } from '@/components/ui/TextField';
import {
  callOf, callText, completions, drawCentre, drawOf, hasMark, kindEdit, kindOffers, shapeFrom, valueKind,
  type ChangeMark as FieldChangeMark, type ChangeSource, type Completion, type FieldEdit, type FieldModel, type FieldOption,
  type FieldRefusal, type ValueKind,
} from './fieldModel';
import { DrawParts, startDraw } from './DrawParts';
import { FormulaParts } from './FormulaParts';
import { ItemParts, listRead } from './ItemParts';
import { LinkParts } from './LinkParts';
import { partDraft, setPartDraft, usePartDraft, type PartDraft } from './partDrafts';
import { ShapeParts } from './ShapeParts';
import { FieldIntent } from './fieldIntent';
import { useFieldGroup } from './fieldGroups';
import './valueField.css';

export type ValueFieldForm = 'row' | 'cell';

/** Where the value comes from, when that is not a change; a link to that object when it names one. */
function Origin({ id, text, reveal }: { id: string; text: string; reveal?: () => void }) {
  return reveal
    ? <Button id={id} variant="ghost" size="content" className="justify-self-start text-left text-shell-muted" onClick={reveal}>
      {text}
    </Button>
    : <span id={id} className="text-shell-muted">{text}</span>;
}

/** What a changed value differs from, as its mark's tip. */
export function changeTip(from: ChangeSource) {
  switch (from.kind) {
    case 'base': return studioFormat('changedFrom', { name: from.document });
    case 'value': return studioFormat('changedFrom', { name: from.text });
    case 'selected': return studioFormat('fieldSource', { kind: studioName('selectedValue'), name: from.text });
    case 'evaluate': return studioEvaluateValueSource({ kind: 'evaluate', step: from.step });
  }
}

/**
 * A field's changed mark, inside its box. `id` names the tip's text, which describes the field's control; `onReset` is
 * the field's Reset, where the change offers one and the field takes edits.
 */
export function FieldChange({ change, id, onReset }: { change?: FieldChangeMark; id: string; onReset?: () => void }) {
  if (!change) return null;
  return <ChangeMark tip={changeTip(change.from)} id={id} reset={change.reset && onReset} />;
}

function Outline({ children }: { children: ReactNode }) {
  return <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}
    strokeLinecap="round" strokeLinejoin="round">{children}</svg>;
}

const kindGlyphs: Record<ValueKind, ReactNode> = {
  value: <Outline><circle cx="12" cy="12" r="2.5" /></Outline>,
  sweep: <Outline><circle cx="3.5" cy="12" r="2.5" /><circle cx="12" cy="12" r="2.5" /><circle cx="20.5" cy="12" r="2.5" /></Outline>,
  draw: <Dices aria-hidden="true" />,
  formula: <Sigma aria-hidden="true" />,
  shape: <Spline aria-hidden="true" />,
  link: <Link aria-hidden="true" />,
  until: <Hourglass aria-hidden="true" />,
  rest: <ArrowRightToLine aria-hidden="true" />,
};

/**
 * Sends edits to the model's source, one at a time in the order they are made. An edit made while the source is briefly
 * unavailable (a projection being replaced) or while the field's last edit saves waits for it rather than being
 * dropped, and the field stays editable meanwhile.
 */
function useSubmit(model: FieldModel) {
  const [refusal, setRefusal] = useState<Refused>();
  const tail = useRef<Promise<unknown>>(Promise.resolve());
  const waiting = useRef<{ edit: FieldEdit; resolve(done: boolean): void }>();
  const latest = useRef(model);
  latest.current = model;
  async function send(commit: NonNullable<FieldModel['commit']>, edit: FieldEdit) {
    const failure = await commit(edit);
    setRefusal(failure === undefined ? undefined : refused(failure, edit));
    return failure === undefined;
  }
  const setReason = (reason: FieldRefusal | undefined) =>
    setRefusal(reason === undefined ? undefined : { reason: typeof reason === 'string' ? reason : reason.reason });
  function submit(edit: FieldEdit) {
    const result = tail.current.then(() => {
      const { commit } = latest.current;
      if (commit) return send(commit, edit);
      return new Promise<boolean>((resolve) => { waiting.current = { edit, resolve }; });
    });
    tail.current = result.catch(() => undefined);
    return result;
  }
  useEffect(() => {
    const next = waiting.current;
    if (!model.commit || !next) return;
    waiting.current = undefined;
    void send(model.commit, next.edit).then(next.resolve);
  });
  return { reason: refusal?.reason, refusal, setReason, submit };
}

/** A refusal, with the term it is about in the text it is about. */
interface Refused {
  readonly reason: string;
  readonly term?: { readonly text: string; readonly start: number; readonly end: number };
}

function refused(failure: FieldRefusal, edit: FieldEdit): Refused {
  if (typeof failure === 'string') return { reason: failure };
  const { reason, span } = failure;
  return span && edit.kind === 'text' ? { reason, term: { text: edit.text, ...span } } : { reason };
}

/** The term the field's problem is about, in the text it is about: the refused draft's, or the stored value's. */
function problemTerm(model: FieldModel, refusal: Refused | undefined) {
  if (refusal) return refusal.term;
  return model.reason && model.span && model.text !== null ? { text: model.text, ...model.span } : undefined;
}

/**
 * The one field control: a value of any kind its site offers, as an inspector row or a table cell (90 px at its
 * narrowest). The control computes nothing; it sends text or an edit to the model's source and shows the reason a
 * source refuses.
 */
export function ValueField({ model, form = 'row' }: { model: FieldModel; form?: ValueFieldForm }) {
  const field = <Field model={model} form={form} />;
  return model.anchor ? <div data-field={model.anchor} className="contents">{field}</div> : field;
}

function Field({ model, form }: { model: FieldModel; form: ValueFieldForm }) {
  const { label, type } = model;
  const id = useId();
  const labelId = useId();
  const reasonId = useId();
  const originId = useId();
  const changeId = useId();
  const { reason, refusal, setReason, submit } = useSubmit(model);
  const readOnly = !model.commit;
  const shownReason = reason ?? model.reason;
  const term = shownReason ? problemTerm(model, refusal) : undefined;
  const describedBy = [model.change && changeId, model.origin && originId, shownReason && reasonId]
    .filter(Boolean).join(' ') || undefined;
  // A problem about one term is drawn on the term; its reason still describes the control.
  const problem = shownReason && (term ? <FieldProblemReason reason={shownReason} id={reasonId} />
    : <FieldProblem reason={shownReason} id={reasonId} />);
  const change = model.change && <FieldChange change={model.change} id={changeId}
    onReset={readOnly ? undefined : () => void submit({ kind: 'reset' })} />;
  /** The field's marks at its box's trailing end: its problem, then its change. */
  const marks = (problem || change) && <>{problem}{change}</>;
  const below = model.origin && <Origin id={originId} text={model.origin} reveal={model.revealOrigin} />;

  function chooseKind(kind: ValueKind) {
    const edit = kindEdit(model, kind);
    if (edit) void submit(edit);
  }

  const parameter = type.kind === 'class' && model.kind !== 'sweep' && model.parts?.length === 1 ? model.parts[0] : undefined;
  const boxed = !!parameter && textualKind(parameter.type.kind);
  const widget = (() => {
    switch (type.kind) {
      case 'boolean':
        return <span className="value-field-check">
          <Checkbox aria-labelledby={labelId} checked={model.value === true} disabled={readOnly}
            aria-describedby={describedBy} onCheckedChange={(checked) => void submit({ kind: 'set', value: checked })} />
          {marks}
        </span>;
      case 'choice': {
        const index = optionIndex(type.options, model.value);
        return <Select aria-labelledby={labelId} aria-describedby={describedBy} className="w-full" disabled={readOnly}
          value={index < 0 ? null : String(index)} placeholder={model.text === null ? studioName('mixed') : undefined}
          trailing={problem || change}
          items={type.options.map((option, position) => ({ value: String(position), label: option.label }))}
          onValueChange={(position) => void submit({ kind: 'set', value: type.options[Number(position)].value })} />;
      }
      case 'class':
        return <ClassPicker model={model} labelId={labelId} inline={boxed} readOnly={readOnly} submit={submit}
          chooseKind={chooseKind} mark={boxed ? undefined : marks} describedBy={describedBy} />;
      case 'vector':
        return <div className="grid min-w-0 auto-cols-[minmax(0,1fr)] grid-flow-col gap-1">
          {model.parts?.map((part) => <ValueField key={part.id} model={part} form="cell" />)}
        </div>;
      case 'group': case 'list':
        return null;
      default:
        return <ValueText model={model} id={id} readOnly={readOnly} describedBy={describedBy} invalid={!!shownReason}
          term={term && shownReason ? { ...term, reason: shownReason } : undefined}
          submit={submit} clearReason={() => setReason(undefined)} chooseKind={chooseKind} changeMark={marks} />;
    }
  })();

  const textual = textualKind(type.kind);
  const drafted = usePartDraft(model.id);
  const valueParts = textual && model.kind !== 'sweep' ? partsOf(model, drafted) : undefined;
  if (form === 'cell') {
    return <div className="value-field value-field-cell grid min-w-0 gap-0.5 tabular-nums" role="group"
      aria-label={studioFieldLabel('field', label)}>
      {textual ? <label id={labelId} htmlFor={id} className="sr-only">{label}</label> : <span id={labelId} className="sr-only">{label}</span>}
      {widget}
    </div>;
  }

  const parts = (indented: boolean) => (indented || model.members || model.canList || valueParts) && <div className="value-field-parts grid gap-1">
    {valueParts}
    {model.members && <Members model={model} />}
    {indented && model.parts!.map((part) => <ValueField key={part.id} model={part} />)}
    {model.canList && <Button variant="ghost" className="justify-self-start" disabled={readOnly}
      onClick={() => void submit({ kind: 'to_list' })}>{studioName('toList')}</Button>}
  </div>;
  if (boxed) return <ClassValue model={model} parameter={parameter!} labelId={labelId} picker={widget} pickReason={reason} />;
  const headerMarks = (problem || change) && <span className="inline-flex items-center">{problem}{change}</span>;
  if (type.kind === 'class' && !model.parts?.length) {
    return <section className="value-field grid min-w-0 gap-1" aria-label={studioFieldLabel('field', label)}>
      <FieldRow label={label} labelId={labelId} below={below}>
        {widget}
      </FieldRow>
      {parts(false)}
    </section>;
  }
  if (type.kind === 'group' || type.kind === 'class' || type.kind === 'list') {
    return <FieldGroup model={model} labelId={labelId} picker={widget} mark={type.kind === 'class' ? undefined : headerMarks}>
      {below && <div className="field-row-below">{below}</div>}
      {parts(!!model.parts?.length)}
    </FieldGroup>;
  }
  return <section className="value-field grid min-w-0 tabular-nums" aria-label={studioFieldLabel('field', label)}>
    <FieldRow label={label} labelId={labelId} htmlFor={textual ? id : undefined}
      actions={type.kind === 'vector' ? headerMarks : undefined}
      below={below}>
      {widget}
    </FieldRow>
    {parts(type.kind !== 'vector' && !!model.parts?.length)}
  </section>;
}

const optionIndex = (options: readonly FieldOption[], value: unknown) =>
  options.findIndex((option) => JSON.stringify(option.value) === JSON.stringify(value));

const textualKind = (kind: FieldModel['type']['kind']) =>
  !['boolean', 'choice', 'class', 'vector', 'group', 'list'].includes(kind);

/**
 * The parts of the field's value, or of its unfilled draft: a Draw's, a list item's (a root that reads a list), a
 * Formula's, a Shape's or a Link's.
 */
function partsOf(model: FieldModel, draft: PartDraft | undefined) {
  switch (draft?.kind) {
    case 'draw': return <DrawParts model={model} draw={draft.draw} />;
    case 'formula': return <FormulaParts model={model} call={draft.call} />;
    case 'shape': return <ShapeParts model={model} call={draft.call} />;
    case 'link': return <LinkParts model={model} opened />;
  }
  const value = model.structure?.value;
  if (!value) return undefined;
  const read = listRead(model);
  if (read) return <ItemParts model={model} read={read} />;
  const call = callOf(model);
  switch (valueKind(value, model.functions)) {
    case 'draw': return <DrawParts model={model} draw={drawOf(model)!} />;
    case 'formula': return call && <FormulaParts model={model} call={call} />;
    case 'shape': return call && <ShapeParts model={model} call={call} />;
    case 'link': return value.kind === 'name' && <LinkParts model={model} reference={value.reference} opened={false} />;
    default: return undefined;
  }
}

/**
 * A field whose value is a group of fields: a folding header in the label style, its changed mark at the header's end;
 * a class's picker under it, kept when folded, holds the class's mark.
 */
function FieldGroup({ model, labelId, picker, mark, children }: {
  model: FieldModel; labelId: string; picker: ReactNode; mark?: ReactNode; children: ReactNode;
}) {
  return <Section level="field" name={model.label} nameId={labelId} control={picker} actions={mark} bodyClassName="grid gap-1"
    className="value-field tabular-nums" {...useFieldGroup(model.id)}>
    {children}
  </Section>;
}

/**
 * A class with one textual parameter: the parameter is the field's value, in one box with the class's picker at its
 * end. The parameter's name is not drawn; it names the input, after the field's name.
 */
function ClassValue({ model, parameter, labelId, picker, pickReason }: {
  model: FieldModel; parameter: FieldModel; labelId: string; picker: ReactNode;
  /** Why the source refused the last class chosen in the picker. */
  pickReason?: string;
}) {
  const id = useId();
  const reasonId = useId();
  const originId = useId();
  const changeId = useId();
  const { reason, setReason, submit } = useSubmit(parameter);
  const readOnly = !parameter.commit;
  const shown = reason ?? pickReason ?? parameter.reason ?? model.reason;
  const origin = parameter.origin ?? model.origin;
  const change = parameter.change ?? model.change;
  const reset = readOnly ? undefined : parameter.change ? () => void submit({ kind: 'reset' })
    : model.commit ? () => void model.commit!({ kind: 'reset' }).then(setReason) : undefined;
  const describedBy = [change && changeId, origin && originId, shown && reasonId].filter(Boolean).join(' ') || undefined;
  const problem = shown && <FieldProblem reason={shown} id={reasonId} />;
  function chooseKind(kind: ValueKind) {
    const edit = kindEdit(parameter, kind);
    if (edit) void submit(edit);
  }
  return <section className={clsx('value-field value-field-class grid min-w-0 tabular-nums', parameter.members && 'value-field-class-swept')}
    aria-label={studioFieldLabel('field', model.label)}>
    <FieldRow label={model.label} labelId={labelId}
      below={origin && <Origin id={originId} text={origin} reveal={parameter.origin ? parameter.revealOrigin : model.revealOrigin} />}>
      <div data-field={parameter.anchor} className="contents">
        <ValueText model={parameter} id={id} readOnly={readOnly} describedBy={describedBy} invalid={!!shown}
          submit={submit} clearReason={() => setReason(undefined)} chooseKind={chooseKind}
          changeMark={(problem || change) && <>{problem}{change && <FieldChange change={change} id={changeId} onReset={reset} />}</>}
          end={picker} parameterOf={[model.label, parameter.parameter!]} />
      </div>
    </FieldRow>
    {(parameter.members || parameter.canList) && <div className="value-field-parts grid gap-1">
      {parameter.members && <Members model={parameter} />}
      {parameter.canList && <Button variant="ghost" className="justify-self-start" disabled={readOnly}
        onClick={() => void submit({ kind: 'to_list' })}>{studioName('toList')}</Button>}
    </div>}
  </section>;
}

function ValueText({ model, id, readOnly, describedBy, invalid, term, submit, clearReason, chooseKind, changeMark, end, parameterOf }: {
  model: FieldModel; id: string; readOnly: boolean; describedBy?: string; invalid: boolean;
  /** The term of a text the problem is about; drawn while the field shows that text. */
  term?: TextTerm & { readonly text: string };
  submit(edit: FieldEdit): Promise<boolean>; clearReason(): void;
  chooseKind(kind: ValueKind): void;
  /** The field's problem and changed marks, at the box's trailing end. */
  changeMark?: ReactNode;
  end?: ReactNode;
  /** The field and parameter whose names make the input's accessible name, for a chosen class's only parameter. */
  parameterOf?: readonly [field: string, parameter: string];
}) {
  const { label } = model;
  const input = useRef<HTMLInputElement>(null);
  const [draft, setDraft] = useState<string>();
  /** The text the field committed and is saving; Escape does not take it back. */
  const saving = useRef<string>();
  const [menu, setMenu] = useState(false);
  const [caret, setCaret] = useState<number>();
  const [highlight, setHighlight] = useState(0);
  const [dismissed, setDismissed] = useState(false);
  const listId = useId();
  const intent = useContext(FieldIntent);
  /** A kind a command put the field in that no edit has made yet: it is made by the text the field commits. */
  const [pending, setPending] = useState<ValueKind>();
  function enter(kind: ValueKind) {
    setPending(kind);
    const end = input.current?.value.length ?? 0;
    input.current?.setSelectionRange(end, end);
  }
  useEffect(() => {
    if (intent && intent.target === model.anchor) enter(intent.kind);
  }, [intent?.key]);
  /** A kind chosen in the kind menu that the field takes when the menu gives the keyboard back to it. */
  const chosen = useRef<ValueKind>();
  const drafted = usePartDraft(model.id);
  const text = draft ?? model.text ?? '';
  const shownKind = pending ?? drafted?.kind ?? model.kind;
  /**
   * Replaces the value with `kind`'s form made from it: a Draw around it, a Formula holding its text with the caret in
   * it, a Shape of it, a Link's choice opened, a Draw's centre as the Value. Sweep and Until are made by the text the
   * field commits, Rest by its text, a sweep's Value by its own edit.
   */
  function choose(kind: ValueKind) {
    if (kind === shownKind) return;
    const held = partDraft(model.id);
    const shownDraw = held?.kind === 'draw' ? held.draw : drawOf(model);
    const centre = shownDraw && drawCentre(model, shownDraw);
    const present = centre ?? text;
    if (kind !== 'until' && kind !== 'sweep') {
      setPartDraft(model.id, undefined);
      setDraft(undefined);
      setPending(undefined);
    }
    switch (kind) {
      case 'sweep': case 'until': chosen.current = kind; return;
      case 'rest': void submit({ kind: 'text', text: 'rest' }); return;
      case 'draw': startDraw(model, present); return;
      case 'link': setPartDraft(model.id, { kind: 'link' }); return;
      case 'shape': {
        const call = shapeFrom(model, present);
        const written = call && callText({ ...model, structure: undefined }, call);
        if (written) void submit({ kind: 'text', text: written });
        else if (call) setPartDraft(model.id, { kind: 'shape', call });
        return;
      }
      case 'value':
        if (kindEdit(model, kind)) { chooseKind(kind); return; }
        if (centre !== undefined && !held) { void submit({ kind: 'text', text: centre }); return; }
        setDraft(present);
        chosen.current = kind;
        return;
      case 'formula':
        setDraft(present);
        chosen.current = kind;
    }
  }
  const offered = draft !== undefined && !dismissed && caret !== undefined
    ? completions(model, text.slice(0, caret)) : { token: '', items: [] };
  const open = offered.items.length > 0;
  const mark = hasMark(model);
  const offers = kindOffers(model);

  function accept(item: Completion) {
    const start = (caret ?? text.length) - offered.token.length;
    const next = text.slice(0, start) + item.insert + text.slice(caret ?? text.length);
    const position = start + item.insert.length;
    setDraft(next);
    setCaret(position);
    setHighlight(0);
    placed.current = position;
  }
  /** The caret an accepted completion leaves, placed as the field draws its new text, before the next key is read. */
  const placed = useRef<number>();
  useLayoutEffect(() => {
    if (placed.current === undefined) return;
    input.current?.setSelectionRange(placed.current, placed.current);
    placed.current = undefined;
  });

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
    if (mark && offers.length && (event.key === 'ContextMenu' || event.key === 'F10' && event.shiftKey)) {
      event.preventDefault();
      setMenu(true);
      return;
    }
    if (event.key === 'Escape') {
      event.stopPropagation();
      if (open) { setDismissed(true); return; }
      setDraft((current) => current === saving.current ? current : undefined);
      setPending(undefined);
      setPartDraft(model.id, undefined);
      clearReason();
      setMenu(false);
    }
    if (event.key === 'Enter' && draft !== undefined) {
      event.preventDefault();
      commit(draft);
    }
  }
  function commit(committed: string) {
    saving.current = committed;
    void submit({ kind: 'text', text: committed }).then((done) => {
      if (saving.current === committed) saving.current = undefined;
      if (!done) return;
      setDraft((current) => current === committed ? undefined : current);
      setPending(undefined);
    });
  }

  const markMenu = mark ? <KindMark model={model} kind={shownKind} readOnly={readOnly} open={menu}
    onOpenChange={setMenu} finalFocus={input} onChoose={choose} /> : undefined;
  return <>
    <TextField ref={input} id={id} type="text" value={text} leading={markMenu} trailing={changeMark} end={end} className="min-w-0"
      term={term && term.text === text ? term : undefined}
      aria-label={parameterOf ? studioParameterLabel(...parameterOf) : undefined} size={end ? Math.max(6, text.length + 1) : undefined}
      placeholder={model.text === null && draft === undefined ? studioName('mixed') : undefined}
      readOnly={readOnly || model.type.kind === 'readOnly'} invalid={invalid}
      aria-describedby={describedBy}
      role={open ? 'combobox' : undefined} aria-expanded={open || undefined}
      aria-controls={open ? listId : undefined} aria-autocomplete={model.names.length || model.functions.length ? 'list' : undefined}
      aria-activedescendant={open ? `${listId}-${highlight}` : undefined}
      onValueChange={(value) => { setDraft(value); setDismissed(false); setHighlight(0); clearReason(); }}
      onSelect={(event) => setCaret(event.currentTarget.selectionStart ?? undefined)}
      onFocus={() => {
        if (chosen.current) enter(chosen.current);
        chosen.current = undefined;
      }}
      onBlur={(event) => {
        setDismissed(true);
        if (draft === undefined) { setPending(undefined); return; }
        const within = event.relatedTarget instanceof Node && event.currentTarget.parentElement?.contains(event.relatedTarget);
        if (!within && draft !== saving.current) commit(draft);
      }}
      onKeyDown={onKeyDown} />
    {open && <CompletionList id={listId} anchor={input} items={offered.items} highlight={highlight} onChoose={accept} />}
  </>;
}

/**
 * A field's kind as a glyph that opens the menu of every kind its site takes, the present one checked. With one, or
 * on a field that cannot be changed, the glyph is only the kind.
 */
function KindMark({ model, kind = model.kind, label = model.label, readOnly, open, onOpenChange, finalFocus, onChoose }: {
  model: FieldModel; kind?: ValueKind; label?: string; readOnly: boolean; open: boolean; onOpenChange(open: boolean): void;
  finalFocus: RefObject<HTMLInputElement>; onChoose(kind: ValueKind): void;
}) {
  const offers = kindOffers(model);
  if (offers.length < 2 || readOnly) {
    return <span role="img" aria-label={studioFieldKindLabel(label, kind)} {...appTip(studioFieldKindLabel(label, kind))}
      className="flex size-full items-center justify-center text-shell-muted">
      {kindGlyphs[kind]}
    </span>;
  }
  return <Menu open={open} onOpenChange={onOpenChange}>
    <MenuTrigger render={<IconButton size="content" label={studioFieldKindLabel(label, kind)}
      className="flex size-full items-center rounded-control data-[popup-open]:bg-shell-hover"
      icon={kindGlyphs[kind]} />} />
    <MenuContent aria-label={studioFieldLabel('kinds', label)} finalFocus={finalFocus}>
      <MenuRadioGroup value={kind} onValueChange={(chosen: ValueKind) => { onOpenChange(false); onChoose(chosen); }}>
        {offers.map((offer) => <MenuRadioItem key={offer.kind} value={offer.kind} disabled={!offer.enabled}
          glyph={<span className="flex text-shell-muted">{kindGlyphs[offer.kind]}</span>}>{studioName(offer.kind)}</MenuRadioItem>)}
      </MenuRadioGroup>
    </MenuContent>
  </Menu>;
}

/**
 * A class-valued field's filter-select of classes, its kind mark inside the box. A swept field's members are the
 * classes checked in its list: choosing a class adds it as a member or removes it, except the last member.
 */
function ClassPicker({ model, labelId, inline, readOnly, submit, chooseKind, mark, describedBy }: {
  model: FieldModel; labelId: string; inline: boolean; readOnly: boolean;
  submit(edit: FieldEdit): Promise<boolean>; chooseKind(kind: ValueKind): void;
  /** The class's problem and changed marks, at the box's trailing end. */
  mark?: ReactNode;
  describedBy?: string;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [menu, setMenu] = useState(false);
  if (model.type.kind !== 'class') return null;
  const { options } = model.type;
  // Inline, the picker shares its box with its class's parameter, whose own mark is named by the field.
  const markLabel = inline ? studioParameterLabel(model.label, studioName('type')) : model.label;
  const leading = hasMark(model) ? <KindMark model={model} label={markLabel} readOnly={readOnly} open={menu} onOpenChange={setMenu}
    finalFocus={input} onChoose={chooseKind} /> : undefined;
  const common = {
    'aria-labelledby': labelId, 'aria-describedby': inline ? undefined : describedBy,
    className: inline ? undefined : 'w-full', inline, disabled: readOnly, leading, trailing: mark, inputRef: input,
  };
  const members = model.members?.values;
  if (members) {
    const checked = members.map(({ option }) => String(optionIndex(options, option)));
    return <Combobox {...common} checked={checked} text={model.text ?? ''}
      items={options.map((option, position) => ({
        value: String(position), label: option.label, disabled: checked.length === 1 && checked[0] === String(position),
      }))}
      onToggle={(position) => {
        const member = members.find((_, index) => checked[index] === position);
        void submit(member ? { kind: 'remove_member', value_id: member.id }
          : { kind: 'add_member', key: String(options[Number(position)].value) });
      }} />;
  }
  const index = optionIndex(options, model.value);
  return <Combobox {...common} value={index < 0 ? null : String(index)}
    placeholder={model.text === null ? studioName('mixed') : undefined}
    items={options.map((option, position) => ({ value: String(position), label: option.label }))}
    onValueChange={(position) => {
      if (position !== null && Number(position) !== index) void submit({ kind: 'set', value: options[Number(position)].value });
    }} />;
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
