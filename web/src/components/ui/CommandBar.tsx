import clsx from 'clsx';
import { Ellipsis, type LucideIcon } from 'lucide-react';
import { Fragment, useLayoutEffect, useRef, useState, type ComponentPropsWithoutRef, type ReactNode } from 'react';
import { Button, IconButton } from './Button';
import { Menu, MenuContent, MenuItem, MenuTrigger } from './Menu';

/** A bound command as a bar draws it: the records `bindStudioCommand` produces satisfy this. */
export interface BarCommand {
  readonly id: string;
  /** The visible name. */
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
  | { readonly command: C; readonly priority: BarPriority }
  | { readonly control: BarControl; readonly priority: BarPriority };

/** A command that may be drawn as its glyph alone. */
export type GlyphCommand<C extends BarCommand> = C & { readonly glyph: LucideIcon };

/** The bar's one pinned action and its counterpart, such as Run and Stop. They never move to the trailing menu. */
export interface PinnedCommands<C extends BarCommand> {
  readonly action: GlyphCommand<C>;
  readonly counterpart?: GlyphCommand<C>;
}

type ItemProps = Omit<ComponentPropsWithoutRef<typeof Button>, 'children' | 'onClick' | 'className'>;

export interface CommandBarProps<C extends BarCommand> {
  /** Drawn first, by name when they fit and by glyph otherwise. */
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
  readonly pinnedNamed: readonly number[];
  readonly pinnedGlyph: readonly number[];
  readonly entries: readonly { readonly width: number; readonly priority: BarPriority }[];
}

export interface BarFit {
  /** The pinned commands are drawn as glyphs. */
  readonly compact: boolean;
  /** Indices of the entries that are rows of the trailing menu. */
  readonly hidden: ReadonlySet<number>;
}

const rank: Record<BarPriority, number> = { low: 0, normal: 1 };

/** Entry indices in the order they leave the bar. */
export function overflowOrder(priorities: readonly BarPriority[]) {
  return priorities.map((priority, index) => ({ priority, index }))
    .sort((a, b) => rank[a.priority] - rank[b.priority] || b.index - a.index)
    .map(({ index }) => index);
}

export function fitBar({ available, gap, trigger, pinnedNamed, pinnedGlyph, entries }: BarWidths): BarFit {
  const span = (widths: readonly number[]) =>
    widths.reduce((sum, width) => sum + width, 0) + gap * Math.max(0, widths.length - 1);
  const order = overflowOrder(entries.map(({ priority }) => priority));
  for (const compact of [false, true]) {
    const pinned = compact ? pinnedGlyph : pinnedNamed;
    const hidden = new Set<number>();
    const shown = () => entries.flatMap(({ width }, index) => hidden.has(index) ? [] : [width]);
    if (span([...pinned, ...shown()]) <= available) return { compact, hidden };
    for (const index of order) {
      hidden.add(index);
      if (span([...pinned, ...shown(), trigger]) <= available) return { compact, hidden };
    }
  }
  return { compact: true, hidden: new Set(entries.keys()) };
}

function Command<C extends BarCommand>({ command, props, className, named }: {
  command: C; props?: ItemProps; className?: string; named: boolean;
}) {
  const Glyph = command.glyph;
  return Glyph && !named
    ? <IconButton label={command.label} icon={<Glyph aria-hidden="true" />} disabled={!command.enabled}
      onClick={command.invoke} className={className} {...props} />
    : <Button aria-label={command.label === command.title ? undefined : command.label} disabled={!command.enabled}
      onClick={command.invoke} className={className} {...props}>{command.title}</Button>;
}

function entryId<C extends BarCommand>(entry: BarEntry<C>) {
  return 'command' in entry ? entry.command.id : entry.control.id;
}

/** A row of commands that never scrolls: the entries that do not fit are rows of a trailing menu. */
export function CommandBar<C extends BarCommand>({
  pinned, entries, moreLabel, itemProps, itemClassName, className, ...aria
}: CommandBarProps<C>) {
  const bar = useRef<HTMLDivElement>(null);
  const measure = useRef<HTMLDivElement>(null);
  const [fit, setFit] = useState<BarFit>({ compact: false, hidden: new Set() });
  const pins = pinned ? [pinned.action, ...pinned.counterpart ? [pinned.counterpart] : []] : [];
  const signature = [
    ...pins.map(({ id, title }) => `pin:${id}:${title}`),
    ...entries.map((entry) => 'command' in entry
      ? `${entry.command.id}:${entry.command.title}:${entry.command.glyph ? 1 : 0}:${entry.priority}`
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
      setFit(fitBar({
        available: node.clientWidth, gap, trigger: width('trigger')[0] ?? 0,
        pinnedNamed: width('named'), pinnedGlyph: width('glyph'),
        entries: width('entry').map((itemWidth, index) => ({ width: itemWidth, priority: entries[index]!.priority })),
      }));
    };
    const sizes = new ResizeObserver(update);
    sizes.observe(node);
    sizes.observe(ruler);
    update();
    return () => sizes.disconnect();
  }, [signature]);

  const overflow = entries.filter((_, index) => fit.hidden.has(index));
  return (
    <div ref={bar} role="toolbar" className={clsx('relative flex min-w-0 flex-1 items-center gap-1 overflow-hidden', className)} {...aria}>
      <div ref={measure} aria-hidden="true"
        className="pointer-events-none invisible absolute start-0 top-0 flex w-max items-center gap-1">
        {pins.map((command) => <Button key={`named:${command.id}`} data-measure="named" className={itemClassName}>{command.title}</Button>)}
        {pins.map((command) => <IconButton key={`glyph:${command.id}`} data-measure="glyph" label="" icon={<command.glyph />}
          className={itemClassName} />)}
        {entries.map((entry) => 'command' in entry
          ? entry.command.glyph
            ? <IconButton key={entry.command.id} data-measure="entry" label="" icon={<entry.command.glyph />} className={itemClassName} />
            : <Button key={entry.command.id} data-measure="entry" className={itemClassName}>{entry.command.title}</Button>
          : <div key={entry.control.id} data-measure="entry" className="flex shrink-0">{entry.control.bar}</div>)}
        <IconButton data-measure="trigger" label="" icon={<Ellipsis />} className={itemClassName} />
      </div>
      {pins.map((command) => <Command key={command.id} command={command} props={itemProps?.(command)} className={itemClassName}
        named={!fit.compact} />)}
      {entries.map((entry, index) => fit.hidden.has(index) ? null : 'command' in entry
        ? <Command key={entry.command.id} command={entry.command} props={itemProps?.(entry.command)} className={itemClassName} named={false} />
        : <div key={entry.control.id} className="flex shrink-0">{entry.control.bar}</div>)}
      {overflow.length > 0 && <Menu>
        <MenuTrigger render={<IconButton label={moreLabel} icon={<Ellipsis aria-hidden="true" />} className={itemClassName} />} />
        <MenuContent align="end" aria-label={moreLabel}>
          {overflow.map((entry) => {
            if ('control' in entry) return <Fragment key={entryId(entry)}>{entry.control.menu}</Fragment>;
            const { command } = entry;
            const Glyph = command.glyph;
            return <MenuItem key={command.id} disabled={!command.enabled} onClick={command.invoke}
              glyph={Glyph ? <Glyph aria-hidden="true" /> : <span aria-hidden="true" className="size-[1.15em] shrink-0" />}>
              {command.title}
            </MenuItem>;
          })}
        </MenuContent>
      </Menu>}
    </div>
  );
}
