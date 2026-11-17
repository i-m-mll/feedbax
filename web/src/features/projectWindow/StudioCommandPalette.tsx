import { Command } from 'cmdk';
import { flushSync } from 'react-dom';
import { Dialog } from '@/components/ui/Dialog';
import { studioName } from '@/components/activity/studioVocabulary';
import { commandAttributes, shortcutLabel, type StudioCommand } from './studioCommands';

export function StudioCommandPalette({ commands, close }: {
  commands: readonly StudioCommand[];
  close(): void;
}) {
  return (
    <Dialog ariaLabel={studioName('commandPalette')} onClose={close} manageFocus
      panelClassName="w-[min(26rem,92vw)] overflow-hidden rounded-shell-control border border-shell-divider bg-shell-chrome text-shell-foreground shadow-xl">
      <Command label={studioName('commandPalette')} loop>
        <Command.Input data-dialog-autofocus aria-label={studioName('findCommand')} placeholder={studioName('findCommand')}
          className="w-full border-b border-shell-divider bg-transparent px-3 py-2" />
        <Command.List className="max-h-[55dvh] overflow-auto p-1.5 text-sm">
          <Command.Empty className="p-2 text-shell-muted">{studioName('noCommands')}</Command.Empty>
          {commands.map((command) => (
            <Command.Item key={command.id} value={command.id} keywords={[command.label]} disabled={!command.enabled}
              {...commandAttributes(command)} aria-label={undefined}
              className="flex cursor-default items-center justify-between gap-3 rounded px-2 py-1.5 aria-selected:bg-shell-hover aria-disabled:opacity-50"
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
