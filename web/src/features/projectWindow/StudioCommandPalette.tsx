import { Command } from 'cmdk';
import { useLayoutEffect, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import { Dialog } from '@/components/ui/Dialog';
import { studioName } from '@/components/activity/studioVocabulary';
import { commandAttributes, shortcutLabel, type StudioCommand } from './studioCommands';

export function StudioCommandPalette({ commands, close }: {
  commands: readonly StudioCommand[];
  close(): void;
}) {
  const [value, setValue] = useState('');
  const list = useRef<HTMLDivElement>(null);
  const enabled = commands.filter((command) => command.enabled).map(({ id }) => id).join();
  useLayoutEffect(() => {
    if (list.current?.querySelector('[cmdk-item][aria-selected="true"]:not([aria-disabled="true"])')) return;
    const first = list.current?.querySelector('[cmdk-item]:not([aria-disabled="true"])')?.getAttribute('data-value');
    if (first) setValue(first);
  }, [enabled]);
  return (
    <Dialog ariaLabel={studioName('commandPalette')} onClose={close} manageFocus
      panelClassName="w-[min(26rem,92vw)] overflow-hidden rounded-shell-control border border-shell-divider bg-shell-chrome text-shell-foreground shadow-xl">
      <Command label={studioName('commandPalette')} loop value={value} onValueChange={setValue}>
        <Command.Input data-dialog-autofocus aria-label={studioName('findCommand')} placeholder={studioName('findCommand')}
          className="w-full border-b border-shell-divider bg-transparent px-3 py-2" />
        <Command.List ref={list} className="max-h-[55dvh] overflow-auto p-1.5 text-sm">
          <Command.Empty className="p-2 text-shell-muted">{studioName('noCommands')}</Command.Empty>
          {commands.map((command) => (
            <Command.Item key={command.id} value={command.id} keywords={[command.label]} disabled={!command.enabled}
              {...commandAttributes(command)} aria-label={undefined}
              className="flex h-row cursor-default items-center justify-between gap-3 rounded px-row-inline aria-selected:bg-shell-hover aria-disabled:opacity-50"
              onSelect={() => { flushSync(close); command.invoke(); }}>
              {command.label}{' '}
              {command.shortcut && <kbd aria-hidden="true" className="text-shell-muted">{shortcutLabel(command.shortcut)}</kbd>}
            </Command.Item>
          ))}
        </Command.List>
      </Command>
    </Dialog>
  );
}
