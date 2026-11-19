import clsx from 'clsx';
import { Ellipsis, type LucideIcon } from 'lucide-react';
import { Fragment, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { IconButton, type IconButtonProps } from './Button';
import { Menu, MenuContent, MenuItem, MenuTrigger } from './Menu';

/** A bound command as a bar draws it: the records `bindStudioCommand` produces satisfy this. */
export interface BarCommand {
  readonly id: string;
  /** The name, shown in the trailing menu. */
  readonly title: string;
  /** The accessible name, which may add the command's subject. */
  readonly label: string;
  readonly glyph?: LucideIcon;
  readonly enabled: boolean;
  invoke(): void;
}

/** A control that is not a single command, such as a picker: drawn inline in the bar, or as rows of the trailing menu. */
export interface BarControl {
  readonly id: string;
  readonly bar: ReactNode;
  readonly menu: ReactNode;
}

/** Which entries leave the bar first: `low` before `normal`, the trailing one first within a priority. */
export type BarPriority = 'normal' | 'low';

export type BarEntry<C extends BarCommand> =
  | { readonly command: GlyphCommand<C>; readonly priority: BarPriority }
  | { readonly control: BarControl; readonly priority: BarPriority };

/** A command a bar can hold: a bar draws every command as its glyph, with its name as tooltip. */
export type GlyphCommand<C extends BarCommand> = C & { readonly glyph: LucideIcon };

/** The command as a bar holds it; a command without a glyph has no place in a bar. */
export function withGlyph<C extends BarCommand>(command: C): GlyphCommand<C> {
  if (!command.glyph) throw new Error(`Bar command ${command.id} has no glyph`);
  return command as GlyphCommand<C>;
}

/** The bar's one pinned action and its counterpart, such as Run and Stop. They never move to the trailing menu. */
export interface PinnedCommands<C extends BarCommand> {
  readonly action: GlyphCommand<C>;
  readonly counterpart?: GlyphCommand<C>;
}

type ItemProps = Omit<IconButtonProps, 'label' | 'icon' | 'onClick' | 'className'>;

export interface CommandBarProps<C extends BarCommand> {
  /** In display order, before the pinned commands, such as the machine Run runs on. */
  leading?: readonly BarEntry<C>[];
  pinned?: PinnedCommands<C>;
  /** In display order, after the pinned commands. */
  entries: readonly BarEntry<C>[];
  /** Accessible name of the trailing menu. */
  moreLabel: string;
  itemProps?: (command: C) => ItemProps;
  /** Classes for every drawn command, applied alike when measuring. */
  itemClassName?: string;
  className?: string;
  'aria-label'?: string;
}

export interface BarWidths {
  readonly available: number;
  readonly gap: number;
  readonly trigger: number;
  readonly pinned: readonly number[];
  readonly entries: readonly { readonly width: number; readonly priority: BarPriority }[];
}


const rank: Record<BarPriority, number> = { low: 0, normal: 1 };

/** Entry indices in the order they leave the bar. */
export function overflowOrder(priorities: readonly BarPriority[]) {
  return priorities.map((priority, index) => ({ priority, index }))
    .sort((a, b) => rank[a.priority] - rank[b.priority] || b.index - a.index)
    .map(({ index }) => index);
}

/** Indices of the entries that are rows of the trailing menu. */
export function fitBar({ available, gap, trigger, pinned, entries }: BarWidths): ReadonlySet<number> {
  const span = (widths: readonly number[]) =>
    widths.reduce((sum, width) => sum + width, 0) + gap * Math.max(0, widths.length - 1);
  const hidden = new Set<number>();
  const shown = () => entries.flatMap(({ width }, index) => hidden.has(index) ? [] : [width]);
  if (span([...pinned, ...shown()]) <= available) return hidden;
  for (const index of overflowOrder(entries.map(({ priority }) => priority))) {
    hidden.add(index);
    if (span([...pinned, ...shown(), trigger]) <= available) return hidden;
  }
  return hidden;
}

function Command<C extends BarCommand>({ command, props, className }: {
  command: GlyphCommand<C>; props?: ItemProps; className?: string;
}) {
  return <IconButton label={command.label} icon={<command.glyph aria-hidden="true" />} disabled={!command.enabled}
    onClick={command.invoke} className={className} {...props} />;
}

function entryId<C extends BarCommand>(entry: BarEntry<C>) {
  return 'command' in entry ? entry.command.id : entry.control.id;
}

/** A row of commands that never scrolls: the entries that do not fit are rows of a trailing menu. */
export function CommandBar<C extends BarCommand>({
  leading = [], pinned, entries: trailing, moreLabel, itemProps, itemClassName, className, ...aria
}: CommandBarProps<C>) {
  const entries = [...leading, ...trailing];
  const bar = useRef<HTMLDivElement>(null);
  const measure = useRef<HTMLDivElement>(null);
  const [hidden, setHidden] = useState<ReadonlySet<number>>(new Set());
  const pins = pinned ? [pinned.action, ...pinned.counterpart ? [pinned.counterpart] : []] : [];
  const signature = [
    ...pins.map(({ id }) => `pin:${id}`),
    ...entries.map((entry) => 'command' in entry
      ? `${entry.command.id}:${entry.priority}`
      : `${entry.control.id}:${entry.priority}`),
  ].join('|');

  useLayoutEffect(() => {
    const node = bar.current;
    const ruler = measure.current;
    if (!node || !ruler) return;
    const update = () => {
      const width = (role: string) => [...ruler.querySelectorAll<HTMLElement>(`:scope > [data-measure="${role}"]`)]
        .map((item) => item.offsetWidth);
      const gap = parseFloat(getComputedStyle(ruler).columnGap) || 0;
      setHidden(fitBar({
        available: node.clientWidth, gap, trigger: width('trigger')[0] ?? 0, pinned: width('pinned'),
        entries: width('entry').map((itemWidth, index) => ({ width: itemWidth, priority: entries[index]!.priority })),
      }));
    };
    const sizes = new ResizeObserver(update);
    sizes.observe(node);
    sizes.observe(ruler);
    update();
    return () => sizes.disconnect();
  }, [signature]);

  const overflow = entries.filter((_, index) => hidden.has(index));
  const drawn = (entry: BarEntry<C>, index: number) => hidden.has(index) ? null : 'command' in entry
    ? <Command key={entry.command.id} command={entry.command} props={itemProps?.(entry.command)} className={itemClassName} />
    : <div key={entry.control.id} className="flex shrink-0">{entry.control.bar}</div>;
  return (
    <div ref={bar} role="toolbar" className={clsx('relative flex min-w-0 flex-1 items-center gap-1 overflow-hidden', className)} {...aria}>
      <div ref={measure} aria-hidden="true"
        className="pointer-events-none invisible absolute start-0 top-0 flex w-max items-center gap-1">
        {pins.map((command) => <IconButton key={command.id} data-measure="pinned" label="" icon={<command.glyph />}
          className={itemClassName} />)}
        {entries.map((entry) => 'command' in entry
          ? <IconButton key={entry.command.id} data-measure="entry" label="" icon={<entry.command.glyph />} className={itemClassName} />
          : <div key={entry.control.id} data-measure="entry" className="flex shrink-0">{entry.control.bar}</div>)}
        <IconButton data-measure="trigger" label="" icon={<Ellipsis />} className={itemClassName} />
      </div>
      {entries.slice(0, leading.length).map(drawn)}
      {pins.map((command) => <Command key={command.id} command={command} props={itemProps?.(command)} className={itemClassName} />)}
      {entries.slice(leading.length).map((entry, index) => drawn(entry, leading.length + index))}
      {overflow.length > 0 && <Menu>
        <MenuTrigger render={<IconButton label={moreLabel} icon={<Ellipsis aria-hidden="true" />} className={itemClassName} />} />
        <MenuContent align="end" aria-label={moreLabel}>
          {overflow.map((entry) => {
            if ('control' in entry) return <Fragment key={entryId(entry)}>{entry.control.menu}</Fragment>;
            const { command } = entry;
            return <MenuItem key={command.id} disabled={!command.enabled} onClick={command.invoke}
              glyph={<command.glyph aria-hidden="true" />}>
              {command.title}
            </MenuItem>;
          })}
        </MenuContent>
      </Menu>}
    </div>
  );
}
