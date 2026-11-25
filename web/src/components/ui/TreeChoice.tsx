import { Popover } from '@base-ui/react/popover';
import clsx from 'clsx';
import { Command, useCommandState } from 'cmdk';
import { Check, ChevronDown, ChevronLeft, ChevronRight } from 'lucide-react';
import { useId, useRef, useState, type InputHTMLAttributes, type KeyboardEvent, type ReactNode } from 'react';
import { usePopupLayer } from './popupLayer';
import { fieldBox, fieldInput, glyph, popupItem, popupSurface, primaryText, secondaryText } from './styles';

/** An option at the end of its path: the names of the levels that hold it, then its own. */
export interface TreeOption<Value extends string> {
  readonly value: Value;
  readonly path: readonly string[];
}

/** One option chosen; or a set of options checked, which choosing an option checks or unchecks, the box showing `text`. */
type TreeSelection<Value extends string> =
  | { value: NoInfer<Value> | null; onValueChange(value: Value): void; checked?: undefined }
  | { checked: readonly NoInfer<Value>[]; onToggle(value: Value): void; text: string };

export type TreeChoiceProps<Value extends string> = TreeSelection<Value> & {
  options: readonly TreeOption<Value>[];
  disabled?: boolean;
  invalid?: boolean;
  'aria-label': string;
  className?: string;
  /** A control at the box's trailing end, before the list's glyph: the problem and the changed mark. */
  trailing?: ReactNode;
  /** The list is open when the choice is first drawn. */
  defaultOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
};

/** A row of the open list: the level above, a level below, or an option. */
type Row<Value extends string> =
  | { key: string; kind: 'up'; label: string }
  | { key: string; kind: 'branch'; label: string; segment: string }
  | { key: string; kind: 'option'; label: string; detail?: string; value: Value; path: readonly string[] };

const within = (path: readonly string[], at: readonly string[]) => at.every((segment, index) => path[index] === segment);

function rows<Value extends string>(options: readonly TreeOption<Value>[], at: readonly string[], query: string): Row<Value>[] {
  const words = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  if (words.length) {
    return options.filter(({ path }) => words.every((word) => path.join(' ').toLowerCase().includes(word)))
      .map(({ value, path }) => ({ key: `option:${value}`, kind: 'option', value, path, label: path.at(-1)!, detail: path.slice(0, -1).join(' · ') }));
  }
  const here = options.filter(({ path }) => path.length > at.length && within(path, at));
  const branches = [...new Set(here.filter(({ path }) => path.length > at.length + 1).map(({ path }) => path[at.length]))];
  return [
    ...at.length ? [{ key: 'up', kind: 'up' as const, label: at.join(' · ') }] : [],
    ...branches.map((segment) => ({ key: `branch:${segment}`, kind: 'branch' as const, label: segment, segment })),
    ...here.filter(({ path }) => path.length === at.length + 1)
      .map(({ value, path }) => ({ key: `option:${value}`, kind: 'option' as const, value, path, label: path.at(-1)! })),
  ];
}

/**
 * The box's type-ahead. cmdk's own input names its list and active row even while the list is closed and unmounted,
 * so this input names them only while the list is open.
 */
function TreeInput({ open, list, ...props }: InputHTMLAttributes<HTMLInputElement> & { open: boolean; list: string }) {
  const active = useCommandState((state) => state.selectedItemId);
  return <input {...props} type="text" role="combobox" autoComplete="off" spellCheck={false} aria-autocomplete="list"
    aria-expanded={open} aria-controls={open ? list : undefined} aria-activedescendant={open ? active : undefined} />;
}

/**
 * A choice among options that have paths (P44): the list opens on the top level and descends a level at a time;
 * typing filters across every level and shows each match with its path. A level with no option under it is not drawn.
 */
export function TreeChoice<Value extends string>(props: TreeChoiceProps<Value>) {
  const { options, disabled, invalid, className, trailing, defaultOpen = false, onOpenChange } = props;
  const label = props['aria-label'];
  const box = useRef<HTMLDivElement>(null);
  const list = useId();
  const [open, setOpen] = useState(defaultOpen);
  const [at, setAt] = useState<readonly string[]>([]);
  const [query, setQuery] = useState('');
  const [highlighted, setHighlighted] = useState('');
  const shown = rows(options, at, query);
  const active = shown.some(({ key }) => key === highlighted) ? highlighted : shown.find(({ kind }) => kind !== 'up')?.key ?? '';
  const chosen = props.checked ?? (props.value === null ? [] : [props.value]);
  const text = props.checked ? props.text : options.find(({ value }) => value === props.value)?.path.join(' · ') ?? '';

  const show = (next: boolean) => {
    if (next === open) return;
    setOpen(next);
    onOpenChange?.(next);
    setAt([]);
    setQuery('');
  };
  const choose = (row: Row<Value>) => {
    if (row.kind === 'up') setAt(at.slice(0, -1));
    else if (row.kind === 'branch') setAt([...at, row.segment]);
    else if (props.checked) props.onToggle(row.value);
    else { props.onValueChange(row.value); show(false); }
  };
  const key = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!open) {
      if (event.key === 'ArrowDown' || event.key === 'Enter') { event.preventDefault(); show(true); }
      return;
    }
    const row = shown.find((item) => item.key === active);
    if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); show(false); }
    else if (event.key === 'ArrowRight' && !query && row?.kind === 'branch') { event.preventDefault(); choose(row); }
    else if ((event.key === 'ArrowLeft' || event.key === 'Backspace') && !query && at.length) { event.preventDefault(); setAt(at.slice(0, -1)); }
  };

  return <Command shouldFilter={false} loop value={active} onValueChange={setHighlighted} label={label} onKeyDown={key}
    ref={box} data-field-box="" className={clsx(fieldBox, 'overflow-hidden', className)}>
    <TreeInput value={query} disabled={disabled} placeholder={text} aria-label={label} aria-invalid={invalid || undefined}
      open={open && shown.length > 0} list={list} className={clsx(fieldInput, 'text-ellipsis placeholder:text-shell-foreground')}
      onChange={(event) => { setQuery(event.target.value); show(true); }} onClick={() => show(true)} onBlur={(event) => {
        if (!event.relatedTarget || !box.current?.contains(event.relatedTarget)) show(false);
      }} />
    {trailing ? <span className="flex shrink-0" onMouseDown={(event) => event.stopPropagation()}>{trailing}</span> : null}
    <button type="button" tabIndex={-1} aria-label={label} disabled={disabled}
      className={clsx('flex h-full w-control shrink-0 items-center justify-center text-shell-muted outline-none', glyph)}
      onMouseDown={(event) => { event.preventDefault(); box.current?.querySelector('input')?.focus(); show(!open); }}>
      <ChevronDown />
    </button>
    <Popover.Root open={open && shown.length > 0} modal={false}
      onOpenChange={(next, { event }) => { if (!next && !box.current?.contains(event?.target as Node | null)) show(false); }}>
      <Popover.Portal container={usePopupLayer()}>
        <Popover.Positioner className="z-50 outline-none" anchor={box} side="bottom" align="start" sideOffset={4}>
          <Popover.Popup id={list} initialFocus={false} finalFocus={false}
            className={clsx(popupSurface, 'w-max min-w-[var(--anchor-width)] max-w-[var(--available-width)]')}
            onMouseDown={(event) => event.preventDefault()}>
            <Command.List className="max-h-[min(22rem,var(--available-height))] overflow-y-auto">
              {shown.map((row) => <Command.Item key={row.key} value={row.key} onSelect={() => choose(row)}
                aria-label={row.kind === 'option' ? row.path.join(' · ') : undefined}
                className={clsx(popupItem, 'data-[selected=true]:bg-shell-hover', row.kind === 'up' && 'text-shell-muted')}>
                <span className="flex w-[1.15em] justify-center">
                  {row.kind === 'up' ? <ChevronLeft /> : row.kind === 'option' && chosen.includes(row.value) && <Check />}
                </span>
                <span className={primaryText}>{row.label}</span>
                {row.kind === 'option' && row.detail ? <span className={secondaryText}>{row.detail}</span> : null}
                {row.kind === 'branch' && <ChevronRight className="text-shell-muted" />}
              </Command.Item>)}
            </Command.List>
          </Popover.Popup>
        </Popover.Positioner>
      </Popover.Portal>
    </Popover.Root>
  </Command>;
}
