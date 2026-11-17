import clsx from 'clsx';
import { Ellipsis, type LucideIcon } from 'lucide-react';
import { useLayoutEffect, useRef, useState, type ComponentPropsWithoutRef } from 'react';
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

type ItemProps = Omit<ComponentPropsWithoutRef<typeof Button>, 'children' | 'onClick' | 'className'>;

export interface CommandBarProps<C extends BarCommand> {
  /** In priority order: the last ones move into the trailing menu first. */
  commands: readonly C[];
  /** Accessible name of the trailing menu. */
  moreLabel: string;
  itemProps?: (command: C) => ItemProps;
  /** Classes for every drawn command, applied alike when measuring. */
  itemClassName?: string;
  className?: string;
  'aria-label'?: string;
}

function fitting(available: number, widths: readonly number[], trigger: number, gap: number) {
  const span = (count: number) => widths.slice(0, count).reduce((sum, width) => sum + width, 0) + gap * Math.max(0, count - 1);
  if (span(widths.length) <= available) return widths.length;
  let count = widths.length - 1;
  while (count > 0 && span(count) + gap + trigger > available) count -= 1;
  return count;
}

function Command<C extends BarCommand>({ command, props, className }: { command: C; props?: ItemProps; className?: string }) {
  const Glyph = command.glyph;
  return Glyph
    ? <IconButton label={command.label} icon={<Glyph aria-hidden="true" />} disabled={!command.enabled}
      onClick={command.invoke} className={className} {...props} />
    : <Button aria-label={command.label === command.title ? undefined : command.label} disabled={!command.enabled}
      onClick={command.invoke} className={className} {...props}>{command.title}</Button>;
}

/** A row of commands that never scrolls: the commands that do not fit are rows of a trailing menu. */
export function CommandBar<C extends BarCommand>({ commands, moreLabel, itemProps, itemClassName, className, ...aria }: CommandBarProps<C>) {
  const bar = useRef<HTMLDivElement>(null);
  const measure = useRef<HTMLDivElement>(null);
  const [shown, setShown] = useState(commands.length);
  const signature = commands.map(({ id, title, glyph }) => `${id}:${title}:${glyph ? 1 : 0}`).join('|');

  useLayoutEffect(() => {
    const node = bar.current;
    const ruler = measure.current;
    if (!node || !ruler) return;
    const update = () => {
      const items = [...ruler.children] as HTMLElement[];
      const trigger = items.pop()!;
      const gap = parseFloat(getComputedStyle(ruler).columnGap) || 0;
      setShown(fitting(node.clientWidth, items.map((item) => item.offsetWidth), trigger.offsetWidth, gap));
    };
    const sizes = new ResizeObserver(update);
    sizes.observe(node);
    sizes.observe(ruler);
    update();
    return () => sizes.disconnect();
  }, [signature]);

  const overflow = commands.slice(shown);
  return (
    <div ref={bar} role="toolbar" className={clsx('relative flex min-w-0 flex-1 items-center gap-1 overflow-hidden', className)} {...aria}>
      <div ref={measure} aria-hidden="true"
        className="pointer-events-none invisible absolute start-0 top-0 flex w-max items-center gap-1">
        {commands.map((command) => command.glyph
          ? <IconButton key={command.id} label="" icon={<command.glyph />} className={itemClassName} />
          : <Button key={command.id} className={itemClassName}>{command.title}</Button>)}
        <IconButton label="" icon={<Ellipsis />} className={itemClassName} />
      </div>
      {commands.slice(0, shown).map((command) => <Command key={command.id} command={command} props={itemProps?.(command)} className={itemClassName} />)}
      {overflow.length > 0 && <Menu>
        <MenuTrigger render={<IconButton label={moreLabel} icon={<Ellipsis aria-hidden="true" />} className={itemClassName} />} />
        <MenuContent align="end" aria-label={moreLabel}>
          {overflow.map((command) => {
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
